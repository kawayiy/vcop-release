"""Video clip order prediction."""
import os
import time
import random
import itertools
import argparse
import inspect

import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets.csl_daily import CSLDailyVCOPDataset
from datasets.csl_news import CSLNewsVCOPDataset
from models.vcopn import VCOPN
from models.uni_sl_r3d import UniSLR3D


def order_class_index(order):
    """Return the index of the order in its full permutation.

    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        if epoch == 1 and i == 1:
            print("tuple_orders sample:", tuple_orders[:4])
            print("targets sample:", targets[:4])
            print("input shape:", inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs)  # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch - 1) * len(train_dataloader) + i
            if writer is not None:
                writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
                writer.add_scalar('train/Accuracy', avg_acc, step)
            # writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            # writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)  # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    if writer is not None:
        writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
        writer.add_scalar('val/Accuracy', avg_acc, epoch)
    # writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    # writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--dataset', type=str, default='csl_daily', choices=['csl_daily', 'csl_news'])
    # parser.add_argument('--model', type=str, default='uni_sl_r3d', help='c3d/r3d/r21d')
    parser.add_argument('--model', type=str, default='uni_sl_r3d', help='experiment tag')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')

    parser.add_argument('--overfit-small', action='store_true', help='overfit a tiny subset for debugging')
    parser.add_argument('--small-size', type=int, default=8, help='number of samples for tiny overfit test')

    parser.add_argument('--train_split', type=str, default=None, help='path to training split file')
    parser.add_argument('--val_split', type=str, default=None, help='path to validation split file')
    parser.add_argument('--no_val', action='store_true', help='disable validation')
    parser.add_argument('--save_freq', type=int, default=20, help='save every N epochs')
    parser.add_argument('--data_root', type=str, default='/projects/u5ia/pxl416/data/CSL-Daily')
    parser.add_argument('--videos_dir', type=str, default='rgb', help='relative or absolute video directory for CSL-News')

    parser.add_argument('--disable_tb', action='store_true', help='disable tensorboard logging')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if CUDA is available')

    args = parser.parse_args()
    return args


def load_model_state(model, state_dict):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def build_sl_vcop_dataset(args, train, transforms_, split_file=None, split_name=None):
    if args.dataset == 'csl_daily':
        return CSLDailyVCOPDataset(
            args.data_root,
            args.cl,
            args.it,
            args.tl,
            train,
            transforms_,
            fixed_sampling=args.overfit_small,
            split_file=split_file,
        )

    if args.dataset == 'csl_news':
        return CSLNewsVCOPDataset(
            args.data_root,
            args.cl,
            args.it,
            args.tl,
            train,
            transforms_,
            fixed_sampling=args.overfit_small,
            split_file=split_file,
            split_name=split_name,
            videos_dir=args.videos_dir,
        )

    raise ValueError('Unsupported dataset: {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    visible_gpu_count = torch.cuda.device_count() if use_cuda else 0
    print("Visible GPU count:", visible_gpu_count)
    if args.cpu:
        print("CPU mode forced by --cpu")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    base = UniSLR3D(layer_sizes=(1, 1, 1, 1), with_classifier=False)
    vcopn = VCOPN(base_network=base, feature_size=base.output_dim, tuple_len=args.tl)

    if visible_gpu_count > 1:
        print(f"Using {visible_gpu_count} GPUs")
        vcopn = nn.DataParallel(vcopn)

    vcopn = vcopn.to(device)

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:
            # state_dict = torch.load(args.ckpt, map_location=device)
            # vcopn.load_state_dict(state_dict)
            state_dict = torch.load(args.ckpt, map_location=device)
            load_model_state(vcopn, state_dict)
            log_dir = os.path.dirname(args.ckpt)

        else:
            if args.desp:
                exp_name = '{}_{}_cl{}_it{}_tl{}_{}'.format(args.dataset, args.model, args.cl, args.it, args.tl, args.desp,
                                                            time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_{}_cl{}_it{}_tl{}_{}'.format(args.dataset, args.model, args.cl, args.it, args.tl,
                                                         time.strftime('%m%d%H%M'))
            if args.log is None:
                args.log = './debug_runs'
            log_dir = os.path.join(args.log, exp_name)
        os.makedirs(log_dir, exist_ok=True)
        # writer = SummaryWriter(log_dir)
        writer = None if args.disable_tb else SummaryWriter(log_dir)

        if args.overfit_small:
            train_transforms = transforms.Compose(
                [transforms.Resize((128, 171)), transforms.CenterCrop(112), transforms.ToTensor()])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize((128, 171)),
                transforms.RandomCrop(112),
                transforms.ToTensor()
            ])

        train_dataset = build_sl_vcop_dataset(
            args,
            True,
            train_transforms,
            split_file=args.train_split,
            split_name='train',
        )

        val_dataset = None
        if not args.no_val:
            val_dataset = build_sl_vcop_dataset(
                args,
                False,
                train_transforms,
                split_file=args.val_split,
                split_name='val',
            )
        # if val_dataset is not None:
        #     val_dataloader = DataLoader(
        #         val_dataset,
        #         batch_size=args.bs,
        #         shuffle=False,
        #         num_workers=args.workers,
        #         pin_memory=True,
        #         persistent_workers=(args.workers > 0),
        #         prefetch_factor=2 if args.workers > 0 else None
        #     )

        if args.overfit_small:
            small_indices = list(range(min(args.small_size, len(train_dataset))))
            train_dataset = Subset(train_dataset, small_indices)

            if val_dataset is not None:
                val_dataset = Subset(val_dataset, list(range(min(args.small_size, len(val_dataset)))))
                print(f'Overfit-small mode enabled: train={len(train_dataset)}, val={len(val_dataset)}')
            else:
                print(f'Overfit-small mode enabled: train={len(train_dataset)}, val=None')

        if val_dataset is None:
            print('TRAIN video number: {}, VAL: None.'.format(len(train_dataset)))
        else:
            print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

        # train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=args.bs,
        #     shuffle=True,
        #     num_workers=args.workers,
        #     pin_memory=True
        # )

        train_dataloader_kwargs = dict(
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        if args.workers > 0:
            dataloader_init_params = inspect.signature(DataLoader.__init__).parameters
            if 'persistent_workers' in dataloader_init_params:
                train_dataloader_kwargs['persistent_workers'] = True
            if 'prefetch_factor' in dataloader_init_params:
                train_dataloader_kwargs['prefetch_factor'] = 2

        train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True
            )

        # if args.ckpt:
        #     pass
        # else:
        #     for data in train_dataloader:
        #         tuple_clips, tuple_orders = data
        #         for i in range(args.tl):
        #             # writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :], i, fps=8)
        #             # writer.add_text('train/tuple_orders', str(tuple_orders[:, i].tolist()), i)
        #             pass
        #         tuple_clips = tuple_clips.to(device)
        #         break

        #     for name, param in vcopn.named_parameters():
        #         writer.add_histogram('params/{}'.format(name), param, 0)

        if (not args.ckpt) and (writer is not None):
            for name, param in vcopn.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(vcopn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None

        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            time_start = time.time()
            train(args, vcopn, criterion, optimizer, device, train_dataloader, writer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            if writer is not None:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

            val_loss = None
            if not args.no_val:
                val_loss = validate(args, vcopn, criterion, device, val_dataloader, writer, epoch)
                # scheduler.step(val_loss)

            if epoch % args.save_freq == 0:
                save_path = os.path.join(log_dir, 'model_{}.pt'.format(epoch))
                state_dict = vcopn.module.state_dict() if isinstance(vcopn, nn.DataParallel) else vcopn.state_dict()
                torch.save(state_dict, save_path)

            if (not args.no_val) and (val_loss < prev_best_val_loss):
                model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                state_dict = vcopn.module.state_dict() if isinstance(vcopn, nn.DataParallel) else vcopn.state_dict()
                torch.save(state_dict, model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path
            if writer is not None:
                writer.close()

    elif args.mode == 'test':  ########### Test #############
        # vcopn.load_state_dict(torch.load(args.ckpt))
        # state_dict = torch.load(args.ckpt, map_location=device)
        # vcopn.load_state_dict(state_dict)
        state_dict = torch.load(args.ckpt, map_location=device)
        load_model_state(vcopn, state_dict)

        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])
        # test_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        test_dataset = build_sl_vcop_dataset(
            args,
            False,
            test_transforms,
            split_name='test',
        )
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, vcopn, criterion, device, test_dataloader)
