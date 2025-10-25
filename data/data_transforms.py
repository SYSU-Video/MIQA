from dataclasses import dataclass
from typing import Dict, Any, List, Callable

from torchvision import transforms

def build_transform(is_train: bool, config: Any):
    """
    Factory function to build image transformation pipeline.

    Args:
        is_train (bool): Whether to build transforms for training or evaluation
        config: Configuration dictionary

    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    builder = TransformBuilder(config)
    if config.is_two_transform:
        transform_crop, transform_resize = builder.build_two_transform(is_train)
        return [transform_crop, transform_resize]
    else:
        return builder.build_transform(is_train)

@dataclass
class TransformConfig:
    """Configuration class for image transformations."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    SIMPLE_MEAN = (0.5, 0.5, 0.5)
    SIMPLE_STD = (0.5, 0.5, 0.5)


class TransformBuilder:
    """Builder class for creating image transformation pipelines."""

    def __init__(self, config: Any):
        """
        Initialize transform builder with configuration.

        Args:
            config: Configuration dictionary containing transformation parameters
        """
        self.config = config
        self.transform_config = TransformConfig()

    def _get_normalization_params(self) -> tuple:
        """Get normalization parameters based on configuration."""
        if self.config.transform_type == "cnn_transform":
            return (self.transform_config.IMAGENET_MEAN,
                    self.transform_config.IMAGENET_STD)
        return (self.transform_config.SIMPLE_MEAN,
                self.transform_config.SIMPLE_STD)

    def _build_augmentation_transforms(self) -> list:
        """Build list of augmentation transforms."""
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15)
        ]

    def _build_base_transforms(self, is_train) -> list:
        """Build list of base transforms."""
        mean, std = self._get_normalization_params()
        if is_train:
            return [
                transforms.Resize(self.config.image_size),
                transforms.RandomCrop(size=self.config.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        else:
            return [
                transforms.Resize(self.config.image_size),
                transforms.CenterCrop(size=self.config.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]

    def build_transform(self, is_train: bool = True) -> transforms.Compose:
        """
        Build transformation pipeline based on configuration.

        Args:
            is_train (bool): Whether to build transforms for training or evaluation

        Returns:
            transforms.Compose: Composed transformation pipeline
        """
        transform_list = []

        if is_train and self.config.augmentation:
            transform_list.extend(self._build_augmentation_transforms())

        transform_list.extend(self._build_base_transforms(is_train))

        return transforms.Compose(transform_list)

    def build_two_transform(self, is_train: bool = True):
        """
        Build two different transforms:
        1. One with crop (same as regular transform)
        2. One with just resize

        Returns a tuple of two transform compositions.
        """
        # mean, std = self._get_normalization_params()

        # First transform with crop (same as regular transform)
        transform_list_1 = []

        if is_train and self.config.augmentation:
            transform_list_1.extend(self._build_augmentation_transforms())

        transform_list_1.extend(self._build_base_transforms(is_train))

        # Second transform with just resize
        if is_train:
            transform_list_2 = [
                transforms.Resize(self.config.image_size),
                transforms.CenterCrop((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_config.IMAGENET_MEAN,
                                     std=self.transform_config.IMAGENET_STD)
            ]
        else:
            transform_list_2 = [
                transforms.Resize(self.config.image_size),
                transforms.CenterCrop((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_config.IMAGENET_MEAN,
                                     std=self.transform_config.IMAGENET_STD)
            ]

        return transforms.Compose(transform_list_1), transforms.Compose(transform_list_2)