import numpy as np
import torch
import torch.distributed as dist

from data.data_transforms import build_transform
from data.dataset_cls import MIQACLSDataset
from data.dataset_det import MIQADETDataset
from data.dataset_ins import MIQAINSDataset
from data.samplers import PatchDistributedSampler, SubsetRandomSampler


def build_dataset(config):
    """
    Build corresponding MIQA dataset based on configuration

    Args:
        config: Configuration object containing dataset type, paths, and other information

    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """

    # Build classification-oriented MIQA task dataset
    if config.dataset == "miqa_cls":
        if not config.eval_only:
            train_dataset = MIQACLSDataset(
                root=config.path_miqa_cls,  # Dataset root directory
                split_file=config.train_split_file,  # Training set split file
                patch_num=config.patch_num,  # Number of image patches, default: 1
                transforms=build_transform(is_train=True, config=config),  # Training data augmentation
                metric_type=config.metric_type,  # Evaluation metric type
                return_all_metrics=config.return_all_metrics,  # Whether to return all metrics
                is_train=True  # Mark as training mode
            )

        test_dataset = MIQACLSDataset(
            root=config.path_miqa_cls,
            split_file=config.val_split_file,  # Validation set split file
            patch_num=config.patch_num,
            transforms=build_transform(is_train=False, config=config),  # Validation data transforms (no augmentation)
            metric_type=config.metric_type,
            return_all_metrics=config.return_all_metrics,
            is_train=False  # Mark as validation mode
        )

    # Build detection-oriented MIQA task dataset
    elif config.dataset == "miqa_det":
        if not config.eval_only:
            train_dataset = MIQADETDataset(
                root=config.path_miqa_det,  # Detection dataset root directory
                split_file=config.train_split_file,
                patch_num=config.patch_num,
                transforms=build_transform(is_train=True, config=config),
                metric_type=config.metric_type,
                return_all_metrics=False,  # Detection task doesn't return all metrics
                is_train=True
            )

        test_dataset = MIQADETDataset(
            root=config.path_miqa_det,
            split_file=config.val_split_file,
            patch_num=config.patch_num,
            transforms=build_transform(is_train=False, config=config),
            metric_type=config.metric_type,
            return_all_metrics=False,
            is_train=False
        )

    # Build instance segmentation-oriented MIQA task dataset
    elif config.dataset == "miqa_ins":
        if not config.eval_only:
            train_dataset = MIQAINSDataset(
                det_root=config.path_miqa_det,  # Detection data root directory
                ins_root=config.path_label_ins,  # Instance segmentation label root directory
                split_file=config.train_split_file,
                patch_num=config.patch_num,
                transforms=build_transform(is_train=True, config=config),
                metric_type=config.metric_type,
                return_all_metrics=False,
                is_train=True
            )

        test_dataset = MIQAINSDataset(
            det_root=config.path_miqa_det,
            ins_root=config.path_miqa_ins,
            split_file=config.val_split_file,
            patch_num=config.patch_num,
            transforms=build_transform(is_train=False, config=config),
            metric_type=config.metric_type,
            return_all_metrics=False,
            is_train=False
        )

    else:
        raise NotImplementedError("We only support common IQA dataset now.")
        # Return only test_dataset if evaluating

    if config.eval_only:
        return test_dataset
    else:
        return train_dataset, test_dataset


def build_loader(config):
    """
    Build training and validation data loaders for distributed training

    Args:
        config: Configuration object containing all hyperparameters and path information

    Returns:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        data_loader_train: Training data loader
        data_loader_val: Validation data loader
    """
    # Defrost config to allow modifications
    config.defrost()

    # Build training and validation datasets
    dataset_train, dataset_val = build_dataset(config=config)

    # Freeze config to prevent accidental modifications
    config.freeze()

    # Print successful dataset building info (including local rank and global rank)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    # Get total number of processes and current process's global rank for distributed training
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Select appropriate sampler for training data based on configuration
    if config.ZIP_MODE and config.CACHE_MODE == "part":
        # If using ZIP mode with "part" cache mode, use subset random sampler
        # Each process handles specific data indices (stride sampling)
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        # Otherwise use PyTorch's distributed sampler to ensure each process gets different data subset
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    # Select validation data sampler based on configuration
    if config.test.SEQUENTIAL:
        # If sequential validation is required, use sequential sampler
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        # Otherwise use patch distributed sampler (custom sampler)
        sampler_val = PatchDistributedSampler(dataset_val)

    # Create training data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.batch_size_train,  # Training batch size
        num_workers=config.num_workers_train,
        pin_memory=config.pin_memory_train,
        drop_last=True,  # Drop the last incomplete batch
    )

    # Create validation data loader
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.batch_size_val,  # Validation batch size
        shuffle=False,
        num_workers=config.num_workers_val,
        pin_memory=config.pin_memory_val,
        drop_last=False,  # Keep the last incomplete batch
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
