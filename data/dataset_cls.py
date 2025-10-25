import os
import re

from tqdm import tqdm
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

class MIQACLSDataset(Dataset):
    '''
    Dataset for Image Classification-oriented MIQA Task.
    '''
    def __init__(
        self,
        root: str,
        split_file: str,
        patch_num: int = 1,
        transforms: Any = None,
        metric_type: str = "consistency",
        return_all_metrics: bool = False,
        is_train: bool = True
    ):
        """
        Args:
            root: Root directory containing images, labels, source (original) images, and additional info.
            split_file: (str, optional): Path to the CSV file specifying train/val split.
            patch_num: how many duplicates per distorted image to store (for patch-level augmentation).
            transforms: a sequence or list of callables. Expect at least two transforms:
                        transforms[0] -> crop transform, transforms[1] -> resize/other transform.
            metric_type: (str, optional): Metric to use: 'consistency', 'accuracy', or 'composite (weighted average of consistency and accuracy)' (default: 'consistency').
            return_all_metrics: if True, dataset will include mean/std metrics in returned dict.
            is_train (bool, optional): If True and split_file is None, use default train split CSV.
        """
        self.root = root
        self.patch_num = int(patch_num)
        self.metric_type = metric_type
        self.return_all_metrics = return_all_metrics

        # Determine split file
        if split_file is None:
            self.split_file = 'data/dataset_splitting/miqa_det_train.csv' if is_train else 'data/dataset_splitting/miqa_det_val.csv'
        else:
            self.split_file = split_file

        # minimal validation for transforms
        # if self.transforms is None or len(self.transforms) < 2:
        #     raise ValueError(
        #         "transforms must be a sequence with at least two callables: [crop_transform, resize_transform]"
        #     )

        # supported metric types
        if metric_type not in ["consistency", "accuracy", 'composite']:
            raise ValueError("metric_type must be one of 'consistency', 'accuracy', or 'composite'")

        # distortion types
        self.distortion_types = [
            'contrast', 'darkness', 'pixelate', 'jpeg_compression',
            'motion_blur', 'defocus_blur', 'glass_blur', 'fog', 'snow', 'gaussian_noise'
        ]

        # prepare mapping of levels (standard / bg_stronger / fg_stronger)
        self._initialize_distortion_levels()

        # load the split file (must contain "Image Name" and "Category" columns)
        self.split_data = pd.read_csv(split_file)

        # quick column checks
        required_split_cols = {"Image Name", "Category"}
        if not required_split_cols.issubset(set(self.split_data.columns)):
            raise KeyError(f"split_file must contain columns: {required_split_cols}")

        self.transforms_dict: Dict[str, Any] = {}
        if isinstance(transforms, (list, tuple)):
            assert len(transforms) == 2, "Expected two transforms in the list."
            self.transforms_dict["cropped"] = transforms[0]
            self.transforms_dict["resized"] = transforms[1]
        else:
            self.transforms_dict["cropped"] = transforms

        # internal storage
        self.all_image_paths: List[str] = []
        self.all_orig_image_paths: List[str] = []
        self.all_categories: List[Any] = []
        self.all_labels: List[float] = []
        self.all_metrics: List[Dict[str, float]] = []
        # self.all_distortion_areas: List[int] = []

        # process and populate lists
        self._process_all_samples()


    def _initialize_distortion_levels(self) -> None:
        """
        Initialize mapping of 25 distortion levels into three groups:
          - standard: 1..5 -> indices 0..4
          - bg_stronger: composed fg{i}_bg{j} where bg > fg  (10 combos) -> indices 5..14
          - fg_stronger: composed fg{i}_bg{j} where fg > bg  (10 combos) -> indices 15..24
        """
        # standard levels 0..4
        self.standard_levels = list(range(5))

        # produce bg_stronger strings like 'fg1_bg2', 'fg1_bg3', ... where bg > fg
        self.bg_stronger_levels = []
        for i in range(1, 5):  # fg 1..4
            for j in range(i + 1, 6):  # bg (i+1)..5
                self.bg_stronger_levels.append(f"fg{i}_bg{j}")

        # produce fg_stronger strings like 'fg2_bg1', 'fg3_bg1', ... where fg > bg
        self.fg_stronger_levels = []
        for i in range(2, 6):  # fg 2..5
            for j in range(1, i):  # bg 1..(i-1)
                self.fg_stronger_levels.append(f"fg{i}_bg{j}")

        total_levels = len(self.standard_levels) + len(self.bg_stronger_levels) + len(self.fg_stronger_levels)
        assert total_levels == 25, f"Total levels should be 25, got {total_levels}"

        # create mapping dictionaries
        self.level_mapping = {
            "standard": {str(i + 1): i for i in range(5)},  # '1'..'5' -> 0..4
            "bg_stronger": {lvl: idx + 5 for idx, lvl in enumerate(self.bg_stronger_levels)},  # 5..14
            "fg_stronger": {lvl: idx + 15 for idx, lvl in enumerate(self.fg_stronger_levels)},  # 15..24
        }

    def _get_distortion_area(self, filename: str) -> int:
        """
        Extract distortion area label from filename.

        Returns:
            int: 0=standard, 1=fg_stronger, 2=bg_stronger
        """
        # try to find pattern 'fg{num}_bg{num}'
        m = re.search(r"fg(\d+)_bg(\d+)", filename)
        if m:
            fg_val = int(m.group(1))
            bg_val = int(m.group(2))
            return 1 if fg_val > bg_val else 2 if bg_val > fg_val else 0

        # fallback: explicit markers
        if "fg_stronger" in filename:
            return 1
        if "bg_stronger" in filename:
            return 2

        # default: standard
        return 0

    def _process_all_samples(self) -> None:
        """
        Iterate over the split CSV, read per-image mmos csv files, and populate internal lists.
        Skips any distorted images that are missing on disk.
        """
        # iterate with tqdm and explicit total
        for _, row in tqdm(self.split_data.iterrows(), total=len(self.split_data), desc="Processing split"):
            image_name = row["Image Name"]
            folder_name = os.path.splitext(image_name)[0]
            mmos_file = os.path.join(self.root, 'labels', f"{folder_name}_mmos.csv")

            ori_image_path = os.path.join(self.root, 'src_images', image_name)
            if not os.path.exists(ori_image_path):
                # raise a clear error -- assert would raise AssertionError instead of FileNotFoundError
                raise FileNotFoundError(f"Original image not found: {ori_image_path}")

            if not os.path.exists(mmos_file):
                raise FileNotFoundError(f"MMOS file not found: {mmos_file}")

            mmos_data = pd.read_csv(mmos_file)

            # # check required columns in mmos CSV
            # required_mmos_cols = {
            #     "dist_images",
            #     f"weighted_{self.metric_type}" if self.metric_type in ["consistency", "accuracy"] else None,
            #     "weighted_consistency",
            #     "weighted_accuracy",
            # }
            # # remove None if present
            # required_mmos_cols = {c for c in required_mmos_cols if c is not None}
            # if not required_mmos_cols.issubset(set(mmos_data.columns)):
            #     # be permissive: if metric_type is 'composite', we need both weighted_consistency and weighted_accuracy
            #     missing = required_mmos_cols.difference(set(mmos_data.columns))
            #     raise KeyError(f"MMOS CSV {mmos_file} is missing required columns: {missing}")

            # iterate each distorted image entry
            for idx in range(len(mmos_data)):
                dist_image_name = str(mmos_data.at[idx, "dist_images"])
                image_path = os.path.join(self.root, 'images', dist_image_name)

                if not os.path.exists(image_path):
                    # skip missing distorted image files (logically they might not be generated)
                    continue

                # determine distortion area (standard / fg_stronger / bg_stronger)
                # distortion_area = self._get_distortion_area(dist_image_name)

                # compute label
                if self.metric_type == 'composite':
                    # average of the two weighted metrics
                    w_cons = float(mmos_data.at[idx, "weighted_consistency"])
                    w_acc = float(mmos_data.at[idx, "weighted_accuracy"])
                    mmos_label = 0.5 * w_cons + 0.5 * w_acc
                else:
                    label_key = f"weighted_{self.metric_type}"
                    mmos_label = float(mmos_data.at[idx, label_key])

                # optional extra metrics
                metrics_dict = {}
                if self.return_all_metrics:
                    # use .get with fallback to 0.0 to avoid KeyError (but better to validate earlier)
                    metrics_dict = {
                        "mean_consistency": float(mmos_data.at[idx, "mean_consistency"]) if "mean_consistency" in mmos_data.columns else 0.0,
                        "std_consistency": float(mmos_data.at[idx, "std_consistency"]) if "std_consistency" in mmos_data.columns else 0.0,
                        "mean_accuracy": float(mmos_data.at[idx, "mean_accuracy"]) if "mean_accuracy" in mmos_data.columns else 0.0,
                        "std_accuracy": float(mmos_data.at[idx, "std_accuracy"]) if "std_accuracy" in mmos_data.columns else 0.0,
                    }

                # append data; repeat according to patch_num
                for _ in range(self.patch_num):
                    self.all_image_paths.append(image_path)
                    self.all_orig_image_paths.append(ori_image_path)
                    self.all_categories.append(row["Category"])
                    self.all_labels.append(mmos_label)
                    # self.all_distortion_areas.append(distortion_area)
                    if self.return_all_metrics:
                        self.all_metrics.append(metrics_dict)

        print(f"split_file: {self.split_file}, Total samples: {len(self.all_image_paths)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a dict containing:
            - image_name (basename)
            - category
            - label (torch.float32)
            - distortion_region (torch.long)  # 0/1/2
            - image_cropped (transform[0](PIL.Image))
            - image_resized (transform[1](PIL.Image))
            - optionally mean/std metrics if return_all_metrics is True
        """
        image_path = self.all_image_paths[idx]
        image = self._load_image(image_path)

        result_dict: Dict[str, Any] = {
            "image_name": os.path.basename(image_path),
            "category": self.all_categories[idx],
            "label": torch.tensor(self.all_labels[idx], dtype=torch.float32),
            # "distortion_region": torch.tensor(self.all_distortion_areas[idx], dtype=torch.long),
        }

        # apply transforms
        for key, transform in self.transforms_dict.items():
            result_dict[f"image_{key}"] = transform(image)

        if self.return_all_metrics:
            # metrics were stored as dicts aligned to all_metrics
            for metric_name, metric_value in self.all_metrics[idx].items():
                result_dict[metric_name] = torch.tensor(metric_value, dtype=torch.float32)

        return result_dict

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image with PIL and convert to RGB."""
        return Image.open(image_path).convert("RGB")

    def get_split_info(self) -> Dict[str, List]:
        """
        Return summary info for the split: image paths, categories and quality scores
        (kept for debugging / external use).
        """
        return {
            "image_paths": self.all_image_paths,
            "categories": self.all_categories,
            "quality_scores": self.all_labels,
        }

if __name__ == '__main__':
    # test demo
    import torchvision

    transformers = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = MIQACLSDataset(root = '/public/datasets/miqa_cls',
                             split_file = 'dataset_splitting/miqa_cls_val.csv',
                             patch_num = 1,
                             transforms = [transformers, transformers], metric_type = 'composite', return_all_metrics = False)

    data_loader_ = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    for i, data in enumerate(data_loader_):
        print(data)
        if i == 3:
            break
