import datetime
import platform
import random
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from data import PFBP
import model as models
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from model.dynamic_maml import available_dymodes
from model.resnet import available_backbones
from util.file import exists_file
from util.log import CSVLogger
from util.logger import setup_logger
from util.misc import ensure_path
from util.timer import timer
from util.torchtool import load_checkpoint


class Averager():

    def __init__(self, steps):
        self.n = 0.0
        self.v = [0.0 for i in range(steps + 1)]

    def add(self, x):
        for i in range(len(self.v)):
            self.v[i] = (self.v[i] * self.n + x[i]) / (self.n + 1.0)
        self.n += 1.0

    def item(self):
        return self.v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-epoch', type=str, default='last')
    parser.add_argument('--n-way', type=int, help='n way', default=5)
    parser.add_argument('--k-spts', type=int, nargs='+', default=[1, 5, 10, 15, 20],
                        help='k shot for support set (test with multiple shots)')
    parser.add_argument('--k-qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    parser.add_argument('--imgc', type=int, help='imgc', default=3)
    parser.add_argument('--task-num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta-lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update-lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update-step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update-step_test', type=int, help='update steps for finetunning', default=10)

    # Extra args
    parser.add_argument('--backbone', type=str, default='resnet18', choices=available_backbones)
    parser.add_argument('--model', type=str, default='DynamicMAML', choices=['MAML', 'DynamicMAML'])
    parser.add_argument('--dy-mode', type=str, default='rebirth', choices=available_dymodes)
    parser.add_argument('--data-root', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='FBP5500')
    parser.add_argument('--img-dir', type=str, default='faces')
    parser.add_argument('--train-split-file', type=str, default='train_maml.txt')
    parser.add_argument('--train-episodes', type=int, default=800, help='total episodes for the train set')
    parser.add_argument('--test-split-file', type=str, default='test_maml.txt')
    parser.add_argument('--test-episodes', type=int, default=400, help='total episodes for the test set')
    parser.add_argument('--val-split-file', type=str, default='val_maml.txt')
    parser.add_argument('--val-episodes', type=int, default=100, help='total episodes for the val set')
    parser.add_argument('--work-dir', type=str, default='./save')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--seed', default=2022, type=int, help='random seed for anything')
    parser.add_argument('--pretrain-type', type=str, default='fea', choices=['imagenet', 'fea', 'none'])
    parser.add_argument('--cpu-only', action='store_true', help='run all with CPU')

    args = parser.parse_args()
    if (platform.system()).upper() == 'WINDOWS':
        args.num_workers = 0
    # Seed for anything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Make dirs
    if 'DynamicMAML' not in args.model:
        dy_mode = ''
    else:
        dy_mode = args.dy_mode
    model_name = f'{dy_mode}{args.model}_{args.backbone}'
    args.log_dir = osp.join(args.work_dir, args.dataset)
    ensure_path(args.log_dir)

    # Set data dir
    args.dataset_dir = osp.join(args.data_root, args.dataset)
    img_dir = osp.join(args.dataset_dir, args.img_dir)
    train_split_file = osp.join(args.dataset_dir, args.train_split_file)
    val_split_file = osp.join(args.dataset_dir, args.val_split_file)
    test_split_file = osp.join(args.dataset_dir, args.test_split_file)

    # Logger
    logger = setup_logger(osp.join(args.log_dir, f'{model_name}_Summary.txt'))

    print('Args <========================')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))

    train_corrs = []
    val_corrs = []
    test_corrs = []
    test_mse = []
    test_rmse = []
    print('Start Testing <========================')
    # Timer
    test_time = timer()
    test_time.tic()
    for shot in args.k_spts:
        # Model
        print('Model <========================')
        device = 'cpu' if args.cpu_only or (not torch.cuda.is_available()) else 'cuda'
        if device == 'cpu':
            print('Warning: Run with CPU!!!')
        args.k_spt = shot
        model = models.__dict__[args.model](args)
        model = model.to(device)
        model_dir = osp.join(args.log_dir, f'{model_name}_{args.n_way}W{shot}S', 'models')
        if args.load_epoch == 'best':
            print(f'Using the model with best acc')
            pth_file = osp.join(model_dir, 'best-corr.pth')
        elif args.load_epoch == 'last':
            print(f'Using the model at last epoch')
            pth_file = osp.join(model_dir, 'epoch-last.pth')
        else:
            epoch = int(args.load_epoch)
            print(f'Using the model at epoch {epoch}')
            pth_file = osp.join(model_dir, f'epoch-{epoch}.pth')
        if not exists_file(pth_file):
            print(f'pth file({pth_file}) not found, skip to next shot')
            # train_corrs.append('X/X\t\t\t')
            # val_corrs.append('X/X\t\t\t')
            test_corrs.append('X/X\t\t\t')
            test_mse.append('X/X\t\t\t')
            test_rmse.append('X/X\t\t\t')
            continue
        state_dict = load_checkpoint(pth_file)
        print(
            f'Model info: train with {state_dict["epoch"]} epochs and steps {state_dict["step"]}, cur corr: {state_dict["cur_corr"]}\n'
            f'[best: {state_dict["best_corr"]}], save time: {state_dict["save_time"]}')
        model.load_state_dict(state_dict['state_dict'])
        model = model.to(device)

        # Dataset
        # mini_train = PFBP.FBP(img_dir, mode='test', n_way=args.n_way, k_shot=shot,
        #                       k_query=args.k_qry, setname=train_split_file,
        #                       batchsz=args.train_episodes, resize=args.imgsz)
        # mini_val = PFBP.FBP(img_dir, mode='test', n_way=args.n_way, k_shot=shot,
        #                     k_query=args.k_qry, setname=val_split_file,
        #                     batchsz=args.val_episodes, resize=args.imgsz)
        mini_test = PFBP.FBP(img_dir, mode='test', n_way=args.n_way, k_shot=shot,
                             k_query=args.k_qry, setname=test_split_file,
                             batchsz=args.test_episodes, resize=args.imgsz)
        # db_train = DataLoader(mini_train, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # db_val = DataLoader(mini_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)


        def test(name, db, corrs, mses, rmses):
            acc_all = Averager(args.update_step_test)
            mse_all = Averager(args.update_step_test)
            rmse_all = Averager(args.update_step_test)
            for x_spt, y_spt, x_qry, y_qry in db:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                             x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                if dy_mode:
                    acc = model.meta_test(x_spt, y_spt, x_qry, y_qry, mode=dy_mode)
                else:
                    acc = model.meta_test(x_spt, y_spt, x_qry, y_qry)
                acc_all.add(acc['PC'])
                mse_all.add(acc['MAE'])
                rmse_all.add(acc['RMSE'])
            corr_all = acc_all.item()
            mse_all = mse_all.item()
            rmse_all = rmse_all.item()
            corr_str = ' '.join([f'{c:.4f}' for c in corr_all]) + f' ({corr_all[-1] - corr_all[0]:.4f})'
            print(f'Model: {model_name}, Split: {name}, Corrs: {corr_str}')
            corrs.append(f'{corr_all[0]:.4f}/{corr_all[-1]:.4f}')
            mses.append(f'{mse_all[0]:.4f}/{mse_all[-1]:.4f}')
            rmses.append(f'{rmse_all[0]:.4f}/{rmse_all[-1]:.4f}')


        # Do evaluation
        # test('train', db_train, train_corrs)
        # test('val', db_val, val_corrs)
        test('test', db_test, test_corrs, test_mse, test_rmse)
        print()
    test_time = datetime.timedelta(seconds=test_time.toc())
    # log
    print(
        f'Test Summary: Test Used Time: {test_time}')

    print('Summary Result<========================')
    print(f'Dataset:\t{args.dataset}')
    print(f'N way:\t{args.n_way}')
    print(f'Model:\t{args.model}({dy_mode})')
    print(f'Backbone: {args.backbone}(pretrain type:{args.pretrain_type})')
    print('Set/Shot\t' + '\t\t\t'.join([f'{i} shot' for i in args.k_spts]))
    # print('train\t\t' + '\t'.join(train_corrs))
    # print('val\t\t\t' + '\t'.join(val_corrs))
    # print('test\t\t' + '\t'.join(test_corrs))
    print('test')
    print('PC\t\t' + '\t'.join(test_corrs))
    print('MAE\t\t' + '\t'.join(test_mse))
    print('RMSE\t\t' + '\t'.join(test_rmse))
    csv = CSVLogger(osp.join(args.log_dir, f'{model_name}_Summary.csv'))
    csv.write(['Dataset', args.dataset, 'N way', args.n_way, 'Model', f'{args.model}({dy_mode})', 'Backbone', f'{args.backbone}(pretrain type:{args.pretrain_type})'])
    csv.write(['Set/Shot'] + [f'{i} shot' for i in args.k_spts])
    # csv.write(['train'] + train_corrs)
    # csv.write(['val'] + val_corrs)
    csv.write(['test'])
    csv.write(['PC'] + test_corrs)
    csv.write(['MAE'] + test_mse)
    csv.write(['RMSE'] + test_rmse)
    csv.close()