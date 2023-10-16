import argparse
import datetime
import os.path as osp
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PFBP
from model.resnet import available_backbones, backbones
from util.logger import setup_logger
from util.tf_logger import TFLogger
from util.timer import timer
from util.misc import ensure_path, Averager, count_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-way', type=int, default=5, help='number of classes')
    parser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    parser.add_argument('--backbone', type=str, default='resnet18', choices=available_backbones)
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false',
                        help='use pretrain model (default: True)')
    parser.add_argument('--dataset', type=str, default='FBP5500')
    parser.add_argument('--img-dir', type=str, default='faces')
    parser.add_argument('--train-split-file', type=str, default='train.txt')
    parser.add_argument('--val-split-file', type=str, default='val.txt')
    parser.add_argument('--work-dir', type=str, default='./save')
    parser.add_argument('--data-root', type=str, default='./datasets')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--print-freq', default=30, type=int, help='print batch log per ${print-freq} iter(s)')
    parser.add_argument('--seed', default=2022, type=int, help='random seed for anything')
    parser.add_argument('--cpu-only', action='store_true', help='run all with CPU')

    args = parser.parse_args()
    # Seed for anything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Make dirs
    args.log_dir = osp.join(args.work_dir, args.dataset, args.backbone)
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
    model_class = backbones[args.backbone]
    if args.pretrain:
        print(f'Using pretrained model of {args.backbone}')
        model = model_class(True, num_classes=args.train_way)
    else:
        print('Train the model from scratch')
        model = model_class(False, num_classes=args.train_way)
    model = model.to(device)

    # Data
    print('Data <========================')
    img_dir = osp.join(args.dataset_dir, args.img_dir)
    train_split_file = osp.join(args.dataset_dir, args.train_split_file)
    val_split_file = osp.join(args.dataset_dir, args.val_split_file)
    trainset = PFBP.FBP_nouser(img_dir, mode='train', resize=args.imgsz, setname=train_split_file)
    valset = PFBP.FBP_nouser(img_dir, mode='val', resize=args.imgsz, setname=val_split_file)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f'Dataset info: {args.dataset}, Train size:{len(trainset)}, Val size:{len(valset)}.')

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(epoch, acc, name=None):
        model_file = osp.join(args.model_dir, f'{name}.pth' if name else f'epoch-{epoch}.pth')
        data_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'cur_acc': acc,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optim_dict': optimizer.state_dict(),
            'sche_dict': lr_scheduler.state_dict(),
            'state_dict': model.state_dict()
        }
        torch.save(data_dict, model_file)


    # Tensorboard looger
    tf_logger = TFLogger(args.log_dir)
    # Timer
    train_timer = timer()
    epoch_timer = timer()
    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    print('Start Training <========================')
    train_timer.tic()
    for epoch in range(args.max_epoch):
        model.train()
        tloss_avger = Averager()
        tacc_avger = Averager()

        lr = lr_scheduler.get_last_lr()[0]
        print('\nStart Epoch: %d/%d | Lr: %f' % (epoch + 1, args.max_epoch, lr))
        epoch_timer.tic()
        for batch_idx, batch in enumerate(train_loader):
            x, y = [_.to(device) for _ in batch]
            y = y.squeeze(1)
            output = model(x)
            loss = criterion(output, y)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log data
            acc = count_acc(output, y)
            tloss_avger.add(loss.item())
            tacc_avger.add(acc)
            if batch_idx % args.print_freq == 0:
                print('Epoch: {}, Batch: {}/{}, Loss: {:.4f}, Acc: {:.4f}%'
                      .format(epoch + 1, batch_idx + 1, len(train_loader), loss.item(), acc * 100))
        lr_scheduler.step()
        tloss = tloss_avger.item()
        tacc = tacc_avger.item() * 100
        # Test
        model.eval()
        vloss_avger = Averager()
        vacc_avger = Averager()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x, y = [_.to(device) for _ in batch]
                y = y.squeeze(1)
                output = model(x)
                loss = criterion(output, y)
                acc = count_acc(output, y)
                # log data
                vloss_avger.add(loss.item())
                vacc_avger.add(acc)
        vloss = vloss_avger.item()
        vacc = vacc_avger.item() * 100
        if vacc > best_acc:
            best_acc = vacc
            print(f'Saving the model at Epoch {epoch + 1} with best acc: {best_acc:.3f}%')
            save_model(epoch + 1, vacc, 'best-acc')
        # Log to tensorboard
        tf_logger.write_scalar_dict({
            'loss': {'train': tloss, 'val': vloss},
            'acc': {'train': tacc, 'val': vacc},
            'train/lr': lr,
        }, epoch)
        # Calculate time
        epoch_time = epoch_timer.toc()
        train_time = datetime.timedelta(seconds=train_timer.toc())
        left_time = datetime.timedelta(seconds=epoch_time * (args.max_epoch - epoch - 1))
        # log
        print(
            f'Epoch {epoch + 1} Summary: Train Loss: {tloss:.3f} | Train Acc: {tacc:.3f} | Test Loss: {vloss:.3f} | Test Acc: {vacc:.3f}% [Best Acc: {best_acc:.3f}%]')
        print(
            f'End Epoch: {epoch + 1}/{args.max_epoch} | Epoch Time: {epoch_time:.2f}s | Train Used Time: {train_time} | Left Time: {left_time}')
    # Save at last epoch
    save_model(epoch + 1, vacc, name='epoch-last')
