import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

import argparse
import datetime
import json
import numpy as np

import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

import util.lr_decay as lrd
import util.misc as misc
# from util.datasets import build_shape_surface_occupancy_dataset
from util.datasets import build_shape_surface_sdf_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mymodel

from engine_mymodel_ae import train_one_epoch
from util.loss import MeshLoss


# torch.multiprocessing.set_start_method('spawn')
def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='kl_d512_m512_l32', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--point_cloud_size', default=2048, type=int,
                        help='input size')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--warmup_steps', type=int, default=5000, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/ibex/scratch/projects/c2168/diffusion-shapes/datasets', type=str,
                        help='dataset path')
    parser.add_argument('--image_renders_path', default='/mnt/merged_nvme/lht/3D/ShapeNet_renders/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='output/feat_vae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/feat_vae/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_shape_surface_sdf_dataset('train', args=args)
    dataset_val = build_shape_surface_sdf_dataset('val', args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 主线程负责记录log
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    def collate_fn(batch):
        surface = torch.stack([item[0] for item in batch])
        mesh_vertices = [item[1].clone() for item in batch]
        mesh_faces = [item[2].clone() for item in batch]
        feat = torch.stack([item[3] for item in batch])
        return surface, mesh_vertices, mesh_faces, feat

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=1,
        # num_workers=args.num_workers,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn = collate_fn,
    )

    # 根据args.model来选择不同的create_autoencoder参数
    model = models_mymodel.__dict__[args.model](N=args.point_cloud_size)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # 有效批次大小(effective batch size)。
    # args.batch_size为每个gpu上的batch_size
    # args.accum_iter为梯度累计iter数
    # misc.get_world_size为总的gpu数量
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print(f"eff_batch_size is {eff_batch_size}")
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        print(f"Learning Rate is {args.lr}")

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # 当使用 DistributedDataParallel 包装模型时，模型被分布式化，并且会有一些额外的并行计算细节，比如参数的同步、梯度的聚合等。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # 但在一些情况下，比如你想保存模型，或者在训练之后做推断时，你可能只需要原始的模型（即 model.module），而不需要 DDP 的额外功能
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = MeshLoss()

    print("criterion = %s" % str(criterion))

    # resume model
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     # print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
    #     print("NOT IMPLEMENTED IN args.eval")
    #     exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, data_loader_val,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=None,
            args=args
        )
        # 每隔10个epoch保存模型
        # if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):

        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)
        # 每隔5个epoch进行evaluation
        # if epoch % 5 == 0 or epoch + 1 == args.epochs:
        #     test_stats = evaluate(data_loader_val, model, device, epoch)
        #     print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
        #     max_iou = max(max_iou, test_stats["iou"])
        #     print(f'Max iou: {max_iou:.2f}%')
        #
        #     # 记录log，iou和loss随epoch变化
        #     if log_writer is not None:
        #         log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
        #         log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        #
        #     #
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  **{f'test_{k}': v for k, v in test_stats.items()},
        #                  'epoch': epoch,
        #                  'n_parameters': n_parameters}

        # 训练epoch不记录test_stats
        # else:

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # 主线程负责写log文件
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)