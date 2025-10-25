import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MIQA Training and Evaluation')

    # ------------------------------
    # Dataset & Path Settings
    # ------------------------------
    parser.add_argument('--dataset', default='miqa_cls', type=str, metavar='DATASET',
                        help='Dataset name: miqa_cls|miqa_det|miqa_ins (default: miqa_cls)')
    parser.add_argument('--output_dir', default='outputs', type=str, metavar='OUTPUT_DIR',
                        help='Path to output directory (default: outputs)')

    parser.add_argument('--path_miqa_cls', default='data/miqa_cls', type=str, metavar='PATH_MIQA_CLS',
                        help='Path to miqa_cls dataset (default: data/miqa_cls)')
    parser.add_argument('--path_miqa_det', default='data/miqa_det', type=str, metavar='PATH_MIQA_DET',
                        help='Path to miqa_det dataset (default: data/miqa_det)')
    parser.add_argument('--path_miqa_ins', default='data/miqa_ins', type=str, metavar='PATH_MIQAINS',
                        help='Path to miqa_ins dataset (default: data/miqa_ins)')
    # parser.add_argument('--coco_annotations', default='data/miqa_det', type=str, metavar='COCO_ANNOTATIONS',
    #                     help='Path to COCO annotations (default: data/miqa_det)')

    # ------------------------------
    # Data & Task Settings
    # ------------------------------
    parser.add_argument('--train_split_file', default='train.txt', type=str, metavar='TRAIN_SPLIT_FILE',
                        help='Path to training split file (default: train.txt)')
    parser.add_argument('--val_split_file', default='test.txt', type=str, metavar='VAL_SPLIT_FILE',
                        help='Path to validation split file (default: test.txt)')
    parser.add_argument('--metric_type', default='consistency', type=str, metavar='METRIC_TYPE',
                        help='Metric type: composite|consistency|accuracy (default: composite)')
    parser.add_argument('--return_all_metrics', action='store_true', help='Return all metrics')
    parser.add_argument('--transform_type', default='cnn_transform', type=str, metavar='TRANSFORM_TYPE',
                        choices=['cnn_transform', 'simple_transform'],
                        help='Transform type (default: cnn_transform)')
    # parser.add_argument('--num_distortion_classes', default=10, type=int, metavar='NUM_DISTORTION_CLASSES',
    #                     help='Number of distortion classes (default: 10)')
    # parser.add_argument('--num_severity_levels', default=25, type=int, metavar='NUM_SEVERITY_LEVELS',
    #                     help='Number of severity levels (default: 25)')

    # ------------------------------
    # Model Settings
    # ------------------------------
    parser.add_argument('-a', '--arch', default='resnet18', type=str, metavar='ARCH',
                        help='Model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--pretrained', action='store_true', dest='pretrained',
                        help='Use pretrained model')
    parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--loss_name', default='mse', type=str, metavar='LOSS',
                        help='Loss function (default: mse)')

    # ------------------------------
    # Training Hyperparameters
    # ------------------------------
    parser.add_argument('--num_epochs', default=10, type=int, metavar='NUM_EPOCHS',
                        help='Total number of epochs (default: 10)')
    parser.add_argument('--warmup_epochs', default=1, type=int, metavar='WARMUP_EPOCHS',
                        help='Number of warmup epochs (default: 1)')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='START_EPOCH',
                        help='Starting epoch (useful for restarts, default: 1)')
    parser.add_argument('--validate_num', default=2, type=int, metavar='VALIDATE_NUM',
                        help='Number of validations per epoch (default: 2)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='BATCH_SIZE',
                        help='Mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', dest='lr',
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='PRINT_FREQ',
                        help='Print frequency (default: 100)')


    # ------------------------------
    # Image & Data Augmentation
    # ------------------------------
    parser.add_argument('--image_size', default=(288, 288), type=int, nargs=2, metavar='IMAGE_SIZE',
                        help='Input image size (H, W), default: (288, 288)')
    parser.add_argument('--crop_size', default=(224, 224), type=int, nargs=2, metavar='CROP_SIZE',
                        help='Crop size (H, W), default: (224, 224)')
    parser.add_argument('--patch_num', default=1, type=int, metavar='PATCH_NUM',
                        help='Number of patches per image (default: 1)')
    parser.add_argument('--augmentation', action='store_true', help='Enable data augmentation')

    # ------------------------------
    # Optimizer Settings
    # ------------------------------
    parser.add_argument('--optimizer', default='AdamW', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: AdamW)')
    parser.add_argument('--eps', default=1e-8, type=float, metavar='EPS',
                        help='Epsilon for optimizer (default: 1e-8)')
    parser.add_argument('--betas', default=(0.9, 0.999), type=float, nargs=2, metavar=('BETA1', 'BETA2'),
                        help='Betas for optimizer (default: (0.9, 0.999))')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='MOMENTUM',
                        help='Momentum (for SGD, default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='WEIGHT_DECAY',
                        help='Weight decay (default: 1e-4)', dest='weight_decay')

    # ------------------------------
    # LR Scheduler Settings
    # ------------------------------
    parser.add_argument('--lr_scheduler', default='cosine', type=str, metavar='LR_SCHEDULER',
                        help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--min_lr', default=1e-8, type=float, metavar='MIN_LR',
                        help='Minimum learning rate (default: 1e-8)')
    parser.add_argument('--warmup_lr', default=5e-6, type=float, metavar='WARMUP_LR',
                        help='Warmup learning rate (default: 5e-6)')
    parser.add_argument('--decay_epochs', default=3, type=int, metavar='DECAY_EPOCHS',
                        help='Epochs between LR decay (default: 3)')
    parser.add_argument('--decay_rate', default=0.5, type=float, metavar='DECAY_RATE',
                        help='Learning rate decay rate (default: 0.5)')

    # ------------------------------
    # Evaluation Settings
    # ------------------------------
    parser.add_argument('-e', '--eval_only', dest='eval_only', action='store_true',
                        help='Evaluate model on validation set')

    # ------------------------------
    # Distributed Training Settings
    # ------------------------------
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='NUM_WORKERS',
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--world_size', default=-1, type=int, metavar='WORLD_SIZE',
                        help='Number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='RANK',
                        help='Node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str, metavar='DIST_URL',
                        help='URL for setting up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, metavar='DIST_BACKEND',
                        help='Distributed backend (default: nccl)')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training')
    parser.add_argument('--gpu', default=None, type=int, metavar='GPU',
                        help='GPU id to use')
    parser.add_argument('--seed', default=1234567, type=int, metavar='SEED',
                        help='Random seed (default: 1234567)')

    return parser

if __name__ == '__main__':
    pass
