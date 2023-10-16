import argparse
import datetime
import os.path as osp
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import model as models
from data import PFBP
from model.dynamic_maml import available_dymodes
from model.resnet import available_backbones
from util.logger import setup_logger
from util.misc import ensure_path, Averager
from util.tf_logger import TFLogger
from util.timer import timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, help='epoch number', default=4)
    parser.add_argument('--n-way', type=int, help='n way', default=3)
    parser.add_argument('--k-spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k-qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    parser.add_argument('--imgc', type=int, help='imgc', default=3)
    parser.add_argument('--task-num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta-lr', type=float, help='meta-level outer learning rate', default=0.001)
    parser.add_argument('--update-lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update-step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update-step-test', type=int, help='update steps for finetunning', default=10)

    # Extra args
    parser.add_argument('--backbone', type=str, default='resnet18', choices=available_backbones)
    parser.add_argument('--model', type=str, default='DynamicMAML', choices=['MAML', 'DynamicMAML'])
    parser.add_argument('--dy-mode', type=str, default='rebirth', choices=available_dymodes)
    parser.add_argument('--data-root', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='FBP5500')
    parser.add_argument('--img-dir', type=str, default='faces')
    parser.add_argument('--train-split-file', type=str, default='train_maml.txt')
    parser.add_argument('--val-split-file', type=str, default='val_maml.txt')
    parser.add_argument('--test-split-file', type=str, default='test_maml.txt')
    parser.add_argument('--train-episodes', type=int, default=10000, help='total episodes for the train set')
    parser.add_argument('--val-episodes', type=int, default=100, help='total episodes for the val set')
    parser.add_argument('--test-episodes', type=int, default=100, help='total episodes for the test set')
    parser.add_argument('--work-dir', type=str, default='./save')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--print-freq', default=20, type=int, help='print batch log per ${print-freq} iter(s)')
    # parser.add_argument('--eval-steps', default=500, type=int, help='do evaluation per ${print-freq} iter(s)')
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
    model_name = f'{dy_mode}{args.model}_{args.backbone}_{args.n_way}W{args.k_spt}S'
    args.log_dir = osp.join(args.work_dir, args.dataset, model_name)
    args.model_dir = osp.join(args.log_dir, 'models')
    ensure_path(args.model_dir)

    # Set data dir
    args.dataset_dir = osp.join(args.data_root, args.dataset)

    # Logger
    logger = setup_logger(args.log_dir)

    print('Args <========================')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))

    # Model
    print('Model <========================')
    device = 'cpu' if args.cpu_only or (not torch.cuda.is_available()) else 'cuda'
    if device == 'cpu':
        print('Warning: Run with CPU!!!')
    model = models.__dict__[args.model](args)
    model = model.to(device)

    # batchsz here means total episode number
    print('Data <========================')
    img_dir = osp.join(args.dataset_dir, args.img_dir)
    train_split_file = osp.join(args.dataset_dir, args.train_split_file)
    val_split_file = osp.join(args.dataset_dir, args.val_split_file)
    test_split_file = osp.join(args.dataset_dir, args.test_split_file)
    mini = PFBP.FBP(img_dir, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                    k_query=args.k_qry, setname=train_split_file,
                    batchsz=args.train_episodes, resize=args.imgsz)
    mini_val = PFBP.FBP(img_dir, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, setname=val_split_file,
                        batchsz=args.val_episodes, resize=args.imgsz)
    mini_test = PFBP.FBP(img_dir, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                         k_query=args.k_qry, setname=test_split_file,
                         batchsz=args.test_episodes, resize=args.imgsz)
    print(f'Dataset info: {args.dataset}, Train size:{len(mini)}, Val size:{len(mini_val)}, Val size:{len(mini_test)}.')

    def save_model(epoch, step, cur_corr, name=None):
        model_file = osp.join(args.model_dir, f'{name}.pth' if name else f'epoch-{epoch}.pth')
        data_dict = {
            'epoch': epoch,
            'step': step,
            'cur_corr': cur_corr,
            'best_corr': best_corr,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'state_dict': model.state_dict()
        }
        torch.save(data_dict, model_file)

    # Tensorboard looger
    tf_logger = TFLogger(args.log_dir)
    # Timer
    train_timer = timer()
    epoch_timer = timer()

    best_corr = [-1.0 for i in range(args.update_step + 1)]
    global_step = 1
    # maml
    print('Start Training <========================')
    train_timer.tic()
    for epoch in range(args.max_epoch):
        epoch_timer.tic()
        print('\nStart Epoch: %d/%d' % (epoch + 1, args.max_epoch))
        # fetch meta_batchsz num of episode each time
        #print('Loading images')
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        acc_all_train = ArrAverager(args.update_step)
        loss_all_train = Averager()

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            if dy_mode:
                loss, acc = model.meta_train(x_spt, y_spt, x_qry, y_qry, mode=dy_mode)
            else:
                loss, acc = model.meta_train(x_spt, y_spt, x_qry, y_qry)
            acc_all_train.add(acc)
            loss_all_train.add(loss.item())

            if step % args.print_freq == 0:
                train_corr = acc_all_train.item()
                corr_str = ' '.join([f'{c:.4f}' for c in train_corr]) + f' ({train_corr[-1]-train_corr[0]:.4f})'
                print('Epoch: {}, Batch: {}/{}, Steps: {}, Loss: {:.4f}, Corr: {}'
                      .format(epoch + 1, step + 1, len(db), global_step, loss_all_train.item(), corr_str))
                acc_all_train = ArrAverager(args.update_step)
                loss_all_train = Averager()
            global_step += 1
        print('==>Do evaluation')
        db_val = DataLoader(mini_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        acc_all_test = ArrAverager(args.update_step_test)
        acc_all_val = ArrAverager(args.update_step_test)
        ### validation
        for x_spt, y_spt, x_qry, y_qry in db_val:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            if dy_mode:
                acc = model.meta_test(x_spt, y_spt, x_qry, y_qry, mode=dy_mode)['PC']
            else:
                acc = model.meta_test(x_spt, y_spt, x_qry, y_qry)['PC']
            acc_all_val.add(acc)
        ### testing
        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            if dy_mode:
                acc = model.meta_test(x_spt, y_spt, x_qry, y_qry, mode=dy_mode)['PC']
            else:
                acc = model.meta_test(x_spt, y_spt, x_qry, y_qry)['PC']
            acc_all_test.add(acc)
        # log data
        train_loss = loss_all_train.item()
        train_corr = acc_all_train.item()
        val_corr = acc_all_val.item()
        test_corr = acc_all_test.item()
        train_corr_str = ' '.join([f'{c:.4f}' for c in train_corr]) + f' ({train_corr[-1] - train_corr[0]:.4f})'
        val_corr_str = ' '.join([f'{c:.4f}' for c in val_corr]) + f' ({val_corr[-1] - val_corr[0]:.4f})'
        test_corr_str = ' '.join([f'{c:.4f}' for c in test_corr]) + f' ({test_corr[-1] - test_corr[0]:.4f})'
        if test_corr[-1] > best_corr[-1]:
            best_corr = acc_all_test.item()
            print(f'Saving the model at Epoch {epoch + 1} with best corr: {test_corr_str}')
            save_model(epoch + 1, global_step, test_corr, 'best-corr')
        best_corr_str = ' '.join([f'{c:.4f}' for c in best_corr]) + f' ({best_corr[-1] - best_corr[0]:.4f})'
        train_time = train_timer.toc()
        left_time = (train_time / global_step) * (len(db) * args.max_epoch - global_step)
        # print
        print(
            f'\tEpoch {epoch + 1}/{args.max_epoch}, Batch: {step + 1}/{len(db)}, Steps: {global_step}\n'
            f'\tTrain Used Time: {datetime.timedelta(seconds=train_time)} | Left Time: {datetime.timedelta(seconds=left_time)}\n'
            f'\tTrain Loss: {train_loss:.3f}\n'
            f'\tTrain Corr: {train_corr_str}\n'
            f'\tVal Corr: {val_corr_str}\n'
            f'\tTest Corr: {test_corr_str}\n'
            f'\tBest Corr: {best_corr_str}\n')
        # Log to tensorboard
        tf_logger.write_scalar_dict({
            'train/loss': train_loss,
            'train/acc': {'step0': train_corr[0], f'step{args.update_step}': train_corr[-1]},
            'val/acc': {'step0': val_corr[0], f'step{args.update_step}': val_corr[-1]},
            'test/acc': {'step0': test_corr[0], f'step{args.update_step}': test_corr[-1]},
        }, global_step)
        # Calculate time
        epoch_time = epoch_timer.toc()
        train_time = datetime.timedelta(seconds=train_timer.toc())
        left_time = datetime.timedelta(seconds=epoch_time * (args.max_epoch - epoch - 1))
        print(
            f'End Epoch: {epoch + 1}/{args.max_epoch} | Epoch Time: {epoch_time:.2f}s | Train Used Time: {train_time} | Left Time: {left_time}')
    save_model(epoch + 1, global_step, test_corr, name='epoch-last')



class ArrAverager():

    def __init__(self, steps):
        self.n = 0.0
        self.v = [0.0 for i in range(steps+1)]

    def add(self, x):
        assert len(x) == len(self.v)
        for i in range(len(x)):
            self.v[i] = (self.v[i] * self.n + x[i]) / (self.n + 1.0)
        self.n += 1.0

    def item(self):
        return self.v




if __name__ == '__main__':
    main()
