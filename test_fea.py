import argparse
import datetime
import os.path as osp
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PFBP
from model.resnet import available_backbones, backbones
from util.file import exists_file
from util.logger import setup_logger
from util.timer import timer
from util.misc import ensure_path, Averager, count_acc
from util.torchtool import load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-epoch', type=str, default='best')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-way', type=int, default=5, help='number of classes')
    parser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    parser.add_argument('--backbone', type=str, default='resnet18', choices=available_backbones)
    parser.add_argument('--dataset', type=str, default='FBP5500')
    parser.add_argument('--img-dir', type=str, default='faces')
    parser.add_argument('--test-split-file', type=str, default='test.txt')
    parser.add_argument('--work-dir', type=str, default='./save')
    parser.add_argument('--data-root', type=str, default='../datasets')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--print-freq', default=10, type=int, help='print batch log per ${print-freq} iter(s)')
    parser.add_argument('--seed', default=2022, type=int, help='random seed for anything')
    parser.add_argument('--cpu-only', action='store_true', help='run all with CPU')

    args = parser.parse_args()
    # Seed for anything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #Make dirs
    args.log_dir = osp.join(args.work_dir, args.dataset, args.backbone)
    args.model_dir = osp.join(args.log_dir, 'models')
    ensure_path(args.model_dir)

    # Set data dir
    args.dataset_dir = osp.join(args.data_root, args.dataset)

    # Logger
    logger = setup_logger(osp.join(args.log_dir, 'test.txt'))

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
    model_class = backbones[args.backbone]
    model = model_class(False, num_classes=args.train_way)
    if args.load_epoch == 'best':
        print(f'Using the model with best acc')
        pth_file = osp.join(args.model_dir, 'best-acc.pth')
    else:
        epoch = int(args.load_epoch)
        print(f'Using the model at epoch {epoch}')
        pth_file = osp.join(args.model_dir, f'epoch-{epoch}.pth')
    assert exists_file(pth_file), f'pth file({pth_file}) not found'
    state_dict = load_checkpoint(pth_file)
    print(f'Model info: train with {state_dict["epoch"]} epochs, cur acc: {state_dict["cur_acc"]:.3f}%[best: {state_dict["best_acc"]:.3f}%], save time: {state_dict["save_time"]}')
    model.load_state_dict(state_dict['state_dict'])
    model = model.to(device)

    # Data
    print('Data <========================')
    img_dir = osp.join(args.dataset_dir, args.img_dir)
    test_split_file = osp.join(args.dataset_dir, args.test_split_file)
    testset = PFBP.FBP_nouser(img_dir, mode='test', resize=args.imgsz, setname=test_split_file)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f'Dataset info: {args.dataset}, Test size:{len(testset)}.')

    # Timer
    test_timer = timer()
    # Loss function
    criterion = nn.CrossEntropyLoss()

    test_timer.tic()
    tloss_avger = Averager()
    tacc_avger = Averager()

    print('Start Testing <========================')
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, y = [_.to(device) for _ in batch]
            y = y.squeeze(1)
            output = model(x)
            loss = criterion(output, y)

            # log data
            acc = count_acc(output, y)
            tloss_avger.add(loss.item())
            tacc_avger.add(acc)
            if batch_idx % args.print_freq == 0:
                print('Batch: {}/{}, Loss: {:.4f}, Acc: {:.4f}'
                      .format(batch_idx + 1, len(test_loader), loss.item(), acc))

            tloss = tloss_avger.item()
            tacc = tacc_avger.item() * 100

    train_time = datetime.timedelta(seconds=test_timer.toc())
    # log
    print(
        f'Test Summary: Test Loss: {tloss:.3f} | Test Acc: {tacc:.3f}% | Test Used Time: {train_time}')
