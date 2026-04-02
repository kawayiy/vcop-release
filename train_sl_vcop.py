"""Video clip order prediction."""
import os
import time
import random
import itertools
import argparse
import inspect
import math

import numpy as np

np.float = float
np.int = int
np.bool = np.bool_

import torch
import torch.nn as nn
import torch.optim as optim
try:
    import torch.distributed as dist
except ImportError:
    dist = None
try:
    from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel
except ImportError:
    TorchDistributedDataParallel = None
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from tensorboardX import SummaryWriter

from datasets.bobsl import BOBSLVCOPDataset
from datasets.csl_daily import CSLDailyVCOPDataset
from datasets.csl_news import CSLNewsVCOPDataset
from datasets.phoenix import PhoenixVCOPDataset
from models.vcopn import VCOPN
from models.uni_sl_r3d import UniSLR3D


def order_class_index(order):
    """Return the index of the order in its full permutation.

    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


class EpochDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def is_parallel_model(model):
    if isinstance(model, nn.DataParallel):
        return True
    if TorchDistributedDataParallel is not None and isinstance(model, TorchDistributedDataParallel):
        return True
    return False


def unwrap_model(model):
    return model.module if is_parallel_model(model) else model


def is_dist_ready():
    return dist is not None and dist.is_available() and dist.is_initialized()


def get_rank(args=None):
    if args is not None and hasattr(args, 'rank'):
        return args.rank
    if is_dist_ready():
        return dist.get_rank()
    return 0


def get_world_size(args=None):
    if args is not None and hasattr(args, 'world_size'):
        return args.world_size
    if is_dist_ready():
        return dist.get_world_size()
    return 1


def is_main_process(args=None):
    return get_rank(args) == 0


def get_reduce_sum_op():
    if hasattr(dist, 'ReduceOp'):
        return dist.ReduceOp.SUM
    return dist.reduce_op.SUM


def reduce_tensor(tensor, average=False):
    if not is_dist_ready():
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=get_reduce_sum_op())
    if average:
        reduced /= float(get_world_size())
    return reduced


def reduce_scalar(value, device, average=False):
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    return reduce_tensor(tensor, average=average)


def setup_distributed(args, use_cuda):
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.distributed = args.dist or env_world_size > 1
    args.rank = 0
    args.world_size = 1
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    if not args.distributed:
        if use_cuda:
            torch.cuda.set_device(args.gpu)
        return

    if dist is None or (not dist.is_available()):
        raise RuntimeError('torch.distributed is unavailable in this PyTorch build.')
    if TorchDistributedDataParallel is None:
        raise RuntimeError('DistributedDataParallel is unavailable in this PyTorch build.')
    if not use_cuda:
        raise RuntimeError('This project only enables DDP for CUDA training.')

    args.rank = int(os.environ.get('RANK', '0'))
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if args.world_size <= 1:
        raise RuntimeError('DDP requested but WORLD_SIZE <= 1.')

    if args.local_rank < 0:
        args.local_rank = args.gpu
    if args.dist_backend is None:
        args.dist_backend = 'nccl'

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )


def cleanup_distributed():
    if is_dist_ready():
        dist.destroy_process_group()


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0.0
    running_samples = 0.0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        if epoch == 1 and i == 1 and is_main_process(args):
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
        pts = torch.argmax(outputs, dim=1)
        batch_correct = torch.sum(targets == pts).item()
        batch_size = targets.size(0)
        reduced_loss = reduce_tensor(loss.detach(), average=True)
        reduced_correct = reduce_scalar(batch_correct, device)
        reduced_batch_size = reduce_scalar(batch_size, device)
        if is_main_process(args):
            running_loss += reduced_loss.item()
            correct += reduced_correct.item()
            running_samples += reduced_batch_size.item()
        # print statistics and write summary every N batch
        if is_main_process(args) and (i % args.pf == 0):
            avg_loss = running_loss / args.pf
            avg_acc = correct / max(running_samples, 1.0)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch - 1) * len(train_dataloader) + i
            if writer is not None:
                writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
                writer.add_scalar('train/Accuracy', avg_acc, step)
            # writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            # writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0.0
            running_samples = 0.0


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0.0
    total_batches = 0.0
    total_samples = 0.0
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
        total_batches += 1
        total_samples += targets.size(0)
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    if args.distributed:
        stats = torch.tensor([total_loss, correct, total_batches, total_samples], dtype=torch.float32, device=device)
        stats = reduce_tensor(stats, average=False)
        total_loss, correct, total_batches, total_samples = [item for item in stats.tolist()]
    avg_loss = total_loss / max(total_batches, 1.0)
    avg_acc = correct / max(total_samples, 1.0)
    if writer is not None and is_main_process(args):
        writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
        writer.add_scalar('val/Accuracy', avg_acc, epoch)
    # writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    # writer.add_scalar('val/Accuracy', avg_acc, epoch)
    if is_main_process(args):
        print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0.0
    total_batches = 0.0
    total_samples = 0.0
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
        total_batches += 1
        total_samples += targets.size(0)
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    if args.distributed:
        stats = torch.tensor([total_loss, correct, total_batches, total_samples], dtype=torch.float32, device=device)
        stats = reduce_tensor(stats, average=False)
        total_loss, correct, total_batches, total_samples = [item for item in stats.tolist()]
    avg_loss = total_loss / max(total_batches, 1.0)
    avg_acc = correct / max(total_samples, 1.0)
    if is_main_process(args):
        print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--dataset', type=str, default='csl_daily', choices=['csl_daily', 'csl_news', 'bobsl', 'phoenix'])
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
    parser.add_argument('--videos_dir', type=str, default='rgb', help='relative or absolute video/frame directory for CSL-News/BOBSL/PHOENIX')
    parser.add_argument('--annotations_dir', type=str, default='manual_annotations/continuous_sign_sequences/cslr-json-v2',
                        help='relative or absolute annotation directory for BOBSL/PHOENIX')

    parser.add_argument('--disable_tb', action='store_true', help='disable tensorboard logging')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if CUDA is available')
    parser.add_argument('--dist', action='store_true', help='enable DistributedDataParallel')
    parser.add_argument('--dist_backend', type=str, default=None, help='distributed backend, defaults to nccl')
    parser.add_argument('--dist_url', type=str, default='env://', help='distributed init method')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed by distributed launcher')

    args = parser.parse_args()
    return args


def load_model_state(model, state_dict):
    unwrap_model(model).load_state_dict(state_dict)


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
            annotations_dir=args.annotations_dir,
        )

    if args.dataset == 'bobsl':
        return BOBSLVCOPDataset(
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

    if args.dataset == 'phoenix':
        features_dir = args.videos_dir
        if features_dir == 'rgb':
            features_dir = PhoenixVCOPDataset.DEFAULT_FEATURES_DIR

        annotations_dir = args.annotations_dir
        if annotations_dir == 'manual_annotations/continuous_sign_sequences/cslr-json-v2':
            annotations_dir = PhoenixVCOPDataset.DEFAULT_ANNOTATIONS_DIR

        return PhoenixVCOPDataset(
            args.data_root,
            args.cl,
            args.it,
            args.tl,
            train,
            transforms_,
            fixed_sampling=args.overfit_small,
            split_file=split_file,
            split_name=split_name,
            features_dir=features_dir,
            annotations_dir=annotations_dir,
        )

    raise ValueError('Unsupported dataset: {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    setup_distributed(args, use_cuda)
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        device_index = args.local_rank if args.distributed else args.gpu
        device = torch.device('cuda:{}'.format(device_index))
    else:
        device = torch.device('cpu')
    visible_gpu_count = torch.cuda.device_count() if use_cuda else 0
    if is_main_process(args):
        print(vars(args))
        print("Visible GPU count:", visible_gpu_count)
    if args.cpu and is_main_process(args):
        print("CPU mode forced by --cpu")

    if args.seed is not None:
        process_seed = args.seed + get_rank(args)
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(process_seed)

    ########### model ##############
    base = UniSLR3D(layer_sizes=(1, 1, 1, 1), with_classifier=False)
    vcopn = VCOPN(base_network=base, feature_size=base.output_dim, tuple_len=args.tl)
    vcopn = vcopn.to(device)
    if args.distributed:
        if is_main_process(args):
            print('Using DDP on {} GPUs.'.format(args.world_size))
        vcopn = TorchDistributedDataParallel(vcopn, device_ids=[args.local_rank], output_device=args.local_rank)
    elif visible_gpu_count > 1:
        if is_main_process(args):
            print(f"Using {visible_gpu_count} GPUs")
        vcopn = nn.DataParallel(vcopn)

    try:
        if args.mode == 'train':  ########### Train #############
            if args.ckpt:
                state_dict = torch.load(args.ckpt, map_location=device)
                load_model_state(vcopn, state_dict)
                log_dir = os.path.dirname(args.ckpt)
            else:
                if args.desp:
                    exp_name = '{}_{}_cl{}_it{}_tl{}_{}_{}'.format(
                        args.dataset, args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
                else:
                    exp_name = '{}_{}_cl{}_it{}_tl{}_{}'.format(
                        args.dataset, args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
                if args.log is None:
                    args.log = './debug_runs'
                log_dir = os.path.join(args.log, exp_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = None if (args.disable_tb or (not is_main_process(args))) else SummaryWriter(log_dir)

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

            if args.overfit_small:
                small_indices = list(range(min(args.small_size, len(train_dataset))))
                train_dataset = Subset(train_dataset, small_indices)

                if val_dataset is not None:
                    val_dataset = Subset(val_dataset, list(range(min(args.small_size, len(val_dataset)))))
                    if is_main_process(args):
                        print(f'Overfit-small mode enabled: train={len(train_dataset)}, val={len(val_dataset)}')
                elif is_main_process(args):
                    print(f'Overfit-small mode enabled: train={len(train_dataset)}, val=None')

            if val_dataset is None:
                if is_main_process(args):
                    print('TRAIN video number: {}, VAL: None.'.format(len(train_dataset)))
            elif is_main_process(args):
                print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

            train_dataloader_kwargs = dict(
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )
            train_sampler = None
            if args.distributed:
                train_sampler = EpochDistributedSampler(train_dataset, get_world_size(args), get_rank(args), shuffle=True)
                train_dataloader_kwargs['shuffle'] = False
                train_dataloader_kwargs['sampler'] = train_sampler
            if args.workers > 0:
                dataloader_init_params = inspect.signature(DataLoader.__init__).parameters
                if 'persistent_workers' in dataloader_init_params:
                    train_dataloader_kwargs['persistent_workers'] = True
                if 'prefetch_factor' in dataloader_init_params:
                    train_dataloader_kwargs['prefetch_factor'] = 2

            train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)

            val_dataloader = None
            if val_dataset is not None:
                val_sampler = None
                if args.distributed:
                    val_sampler = EpochDistributedSampler(val_dataset, get_world_size(args), get_rank(args), shuffle=False)
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.bs,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=val_sampler,
                )

            if (not args.ckpt) and (writer is not None):
                for name, param in vcopn.named_parameters():
                    writer.add_histogram('params/{}'.format(name), param, 0)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(vcopn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

            prev_best_val_loss = float('inf')
            prev_best_model_path = None

            for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                time_start = time.time()
                train(args, vcopn, criterion, optimizer, device, train_dataloader, writer, epoch)
                if is_main_process(args):
                    print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

                if writer is not None:
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

                val_loss = None
                if not args.no_val:
                    val_loss = validate(args, vcopn, criterion, device, val_dataloader, writer, epoch)
                    # scheduler.step(val_loss)

                if (epoch % args.save_freq == 0) and is_main_process(args):
                    save_path = os.path.join(log_dir, 'model_{}.pt'.format(epoch))
                    torch.save(unwrap_model(vcopn).state_dict(), save_path)

                if (not args.no_val) and (val_loss < prev_best_val_loss) and is_main_process(args):
                    model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                    torch.save(unwrap_model(vcopn).state_dict(), model_path)
                    prev_best_val_loss = val_loss
                    if prev_best_model_path:
                        os.remove(prev_best_model_path)
                    prev_best_model_path = model_path

            if writer is not None:
                writer.close()

        elif args.mode == 'test':  ########### Test #############
            state_dict = torch.load(args.ckpt, map_location=device)
            load_model_state(vcopn, state_dict)

            test_transforms = transforms.Compose([
                transforms.Resize((128, 171)),
                transforms.CenterCrop(112),
                transforms.ToTensor()
            ])
            test_dataset = build_sl_vcop_dataset(
                args,
                False,
                test_transforms,
                split_name='test',
            )
            test_sampler = None
            if args.distributed:
                test_sampler = EpochDistributedSampler(test_dataset, get_world_size(args), get_rank(args), shuffle=False)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=test_sampler,
            )
            if is_main_process(args):
                print('TEST video number: {}.'.format(len(test_dataset)))
            criterion = nn.CrossEntropyLoss()
            test(args, vcopn, criterion, device, test_dataloader)
    finally:
        cleanup_distributed()
