import numpy as np
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pandas as pd
import torch
import logging

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from configs.args_base import get_args

from utils.losses import build_loss_function
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from utils.logger import build_logger
from utils.misc import setup_seed, reduce_tensor, save_checkpoint
from data import build_dataloader
from models.MIQA_base import get_torch_model, get_timm_model
from models.RA_MIQA import RegionVisionTransformer

best_srcc = best_plcc = best_klcc = 0.

def main(args):
    if args.seed is not None:

        setup_seed(args.seed)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_srcc, best_plcc, best_klcc
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    logger = build_logger(
        output_dir=args.output_dir,
        log_file='{}_train.log'.format(args.run_name),
        rank=args.rank if args.distributed else None,
        level=logging.INFO,
        console_level=logging.INFO if args.rank in [0, None] else logging.WARNING,
        file_level=logging.INFO
    )

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.arch.startswith('RA_'):
        model = RegionVisionTransformer(
            base_model_name = 'vit_small_patch16_224',
            pretrained = True,
            mmseg_config_path = 'models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
            checkpoint_path = 'models/checkpoints/sere_finetuned_vit_small_ep100.pth',
            auto_download = True,
            force_download = False
            )
    else:
        try:
            logger.info(f"Loading model form torchvision {args.arch}")
            model = get_torch_model(model_name=args.arch, pretrained=args.pretrained, num_classes=1)
        except:
            logger.info(f"Loading model form timm {args.arch}")
            model = get_timm_model(model_name=args.arch, pretrained=args.pretrained, num_classes=1)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.info('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data loading
    train_dataset, val_dataset = build_dataloader.build_dataset(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = build_loss_function(loss_name=args.loss_name)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_srcc = checkpoint['best_srcc']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_srcc = best_srcc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # evaluate on validation set
    if args.eval_only:
        validate(val_loader, model, criterion, args)
        return

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs', args.run_name))

    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        best_srcc, best_plcc, best_klcc = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, device, args, val_loader, writer, logger)

    writer.close()

    logger.info('################# Training Finished ##################')
    logger.info(f"Best SRCC: {best_srcc}, Best PLCC: {best_plcc}, Best KLCC: {best_klcc}")
    logger.info('######################################################')


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, device, args, val_loader, writer, logger):
    global best_srcc, best_plcc, best_klcc
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    srcc = AverageMeter('SRCC', ':6.4f')
    plcc = AverageMeter('PLCC', ':6.4f')
    klcc = AverageMeter('KLCC', ':6.4f')
    # mse = AverageMeter('MSE', ':6.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, srcc, plcc, klcc],
        prefix="Epoch: [{}]".format(epoch))

    validate_freq = len(train_loader) // args.validate_num

    end = time.time()
    global_step = epoch * len(train_loader)

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # images = batch['image'].cuda(args.gpu, non_blocking=True)
        # target = batch['label'].cuda(args.gpu, non_blocking=True).view(-1)

        image_cropped = batch['image_cropped'].cuda(args.gpu, non_blocking=True)
        target = batch['label'].cuda(args.gpu, non_blocking=True).view(-1)

        if 'image_resized' in batch:
            image_resized = batch['image_resized'].cuda(args.gpu, non_blocking=True)
            output = model(image_cropped, image_resized).view(-1)
        else:
            output = model(image_cropped).view(-1)

        target_len = target.size(0)
        train_loss = criterion(output, target)

        # Calculate metrics during training sessions
        srcc_train = torchmetrics.functional.spearman_corrcoef(output, target).item()
        plcc_train = torchmetrics.functional.pearson_corrcoef(output, target).item()
        klcc_train = torchmetrics.functional.kendall_rank_corrcoef(output, target).item()

        # Update Metrics
        losses.update(train_loss.item(), target_len)
        srcc.update(srcc_train, target_len)
        plcc.update(plcc_train, target_len)
        klcc.update(klcc_train, target_len)

        # Add training loss to the writer
        writer.add_scalars('Loss', {
            'train': train_loss.item()
        }, global_step + i)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # if scheduler is not None:
        scheduler.step_update(global_step + i)

        # Record the current learning rate
        if args.rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, global_step + i)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

        # Perform multiple verifications within an epoch
        if (i + 1) % validate_freq == 0:
            model.eval()

            results = validate(val_loader=val_loader, model=model, criterion=criterion, args=args, logger=logger)
            val_srcc = results['metrics']['srcc']
            val_plcc = results['metrics']['plcc']
            val_klcc = results['metrics']['klcc']
            val_loss = results['metrics']['loss']
            logger.info(f'Validation results: SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, KLCC: {val_klcc:.4f}, Loss: {val_loss:.4f}')
            if args.rank == 0:
                # Add the validation loss to the same loss chart
                writer.add_scalars('Loss', {
                    'val': val_loss
                }, global_step + i)

                # Add all performance metrics to the same Metrics chart.
                writer.add_scalars('Metrics', {
                    'SRCC': val_srcc,
                    'PLCC': val_plcc,
                    'KLCC': val_klcc
                }, global_step + i)

            is_best = val_srcc > best_srcc
            best_srcc = max(val_srcc, best_srcc)
            best_plcc = max(val_plcc, best_plcc)
            best_klcc = max(val_klcc, best_klcc)

            # Save the best model and results
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % args.ngpus_per_node == 0):
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_srcc': best_srcc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best)

                if is_best:
                    logger.info(
                        f'**BEST** Validation results: SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}, KLCC: {best_klcc:.4f}')

                    df = pd.DataFrame({
                        'image_name': results['image_names'],
                        'prediction': results['predictions'],
                        'ground_truth': results['ground_truth']
                    })
                    csv_filename = os.path.join(args.output_dir,
                                                f'{args.run_name}_best_val_results.csv')
                    df.to_csv(csv_filename, index=False)

            model.train()

    logger.info(
        f'**BEST** Validation results: SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}, KLCC: {best_klcc:.4f}')
    return best_srcc, best_plcc, best_klcc

@torch.no_grad()
def validate(val_loader, model, args, criterion, logger):
    model.eval()
    val_dataset_len = len(val_loader.dataset)
    val_loader_len = len(val_loader)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    with torch.no_grad():
        temp_pred_scores = []
        temp_gt_scores = []
        temp_img_names = []
        time_tmp = time.time()

        for i, batch in enumerate(val_loader):

            if args.gpu is not None and torch.cuda.is_available():
                device = torch.device(f'cuda:{args.gpu}')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

            images = batch['image_cropped'].to(device, non_blocking=True if device.type == 'cuda' else False)
            target = batch['label'].to(device, non_blocking=True if device.type == 'cuda' else False)

            if 'image_resized' in batch:
                image_resized = batch['image_resized'].to(device, non_blocking=True if device.type == 'cuda' else False)
                output = model(images, image_resized).view(-1)
            else:
                output = model(images).view(-1)

            # if args.gpu is not None and torch.cuda.is_available():
            #     images = batch['image'].cuda(args.gpu, non_blocking=True)
            #     target = batch['label'].cuda(args.gpu, non_blocking=True)
            # if torch.backends.mps.is_available():
            #     images = images.to('mps')
            #     target = target.to('mps')

            # output = model(images).view(-1)
            loss = criterion(output, target.view(-1))
            loss = reduce_tensor(loss)
            losses.update(loss.item(), target.size(0))

            batch_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # Save predicted values, gt values, and image names
            temp_pred_scores.append(output.view(-1))
            temp_gt_scores.append(target.view(-1))
            temp_img_names.extend(batch['image_name'])

            if i % args.print_freq == 0:
                logger.info(
                    f"Test: [{i}/{val_loader_len}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                )

    # Combine the results of all batches
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)

    # Distributed processing
    if torch.distributed.is_initialized():
        # Collect the results of all processes
        img_names_gather = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(img_names_gather, temp_img_names)
        all_img_names = []
        for names in img_names_gather:
            all_img_names.extend(names)
        all_img_names = all_img_names[:val_dataset_len]  # 截取到实际数据集大小

        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_dataset_len]

        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_dataset_len]

        if args.patch_num > 1:
            gather_preds_matrix = gather_preds.view(-1, args.patch_num)

            gather_preds = (gather_preds_matrix.mean(dim=-1)).squeeze()
            gather_grotruth = (
                (gather_grotruth.view(-1, args.patch_num)).mean(dim=-1)
            ).squeeze()

        final_preds = gather_preds.float().detach()
        final_grotruth = gather_grotruth.float().detach()
    else:
        final_preds = pred_scores.float().detach()
        final_grotruth = gt_scores.float().detach()
        all_img_names = temp_img_names

    # Calculate the correlation coefficient
    try:
        logger.info(f"len of dataset: {val_dataset_len}, final_preds shape: {final_preds.shape}, final_grotruth shape: {final_grotruth.shape}")
        # Check for the presence of NaN or inf
        if torch.isnan(final_preds).any() or torch.isinf(final_preds).any() or \
                torch.isnan(final_grotruth).any() or torch.isinf(final_grotruth).any():
            raise ValueError("Found NaN or inf values in predictions or ground truth")

        test_srcc = torchmetrics.functional.spearman_corrcoef(final_preds, final_grotruth).item()
        test_plcc = torchmetrics.functional.pearson_corrcoef(final_preds, final_grotruth).item()
        test_klcc = torchmetrics.functional.kendall_rank_corrcoef(final_preds, final_grotruth).item()

    except Exception as e:
        logger.warning(f"Error in calculating correlations: {str(e)}. Resetting cc relation to zero...")
        test_plcc = 0.0
        test_srcc = 0.0
        test_klcc = 0.0

    # Create a result dictionary containing the correspondence between image names, predicted values, and actual values.
    results = {
        'image_names': all_img_names,
        'predictions': final_preds.cpu().numpy().tolist(),
        'ground_truth': final_grotruth.cpu().numpy().tolist(),
        'metrics': {
            'srcc': test_srcc,
            'plcc': test_plcc,
            'klcc': test_klcc,
            'loss': losses.avg
        }
    }

    return results

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    args = get_args().parse_args()

    args.run_name = args.arch + '_' + args.dataset + '_' + args.metric_type

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard_logs', args.run_name), exist_ok=True)

    # save config file
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(args.__dict__.__str__())

    main(args)




