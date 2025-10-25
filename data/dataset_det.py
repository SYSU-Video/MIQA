import os
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
from typing import Dict, Any, Optional, List, Sequence
import torch
from torch.utils.data import Dataset

class MIQADETDataset(Dataset):
    """
    Dataset for Object Detection-oriented MIQA Task.

    Args:
        root (str): Root directory containing images, labels, source (original) images, and additional info.
        split_file (str, optional): Path to the CSV file specifying train/val split.
        patch_num (int, optional): Number of augmented patches per image (default: 1).
        transforms (Optional[Sequence[Callable]]): Sequence of two transforms: [crop_transform, resize_transform].
        metric_type (str, optional): Metric to use: 'consistency', 'accuracy', or 'composite (weighted average of consistency and accuracy)' (default: 'consistency').
        return_all_metrics (bool, optional): Whether to store all detailed metrics (default: False).
        is_train (bool, optional): If True and split_file is None, use default train split CSV.
    """

    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        patch_num: int = 1,
        transforms: Any = None,
        metric_type: str = 'consistency',
        return_all_metrics: bool = False,
        is_train: bool = True
    ):
        self.root = root
        self.patch_num = patch_num
        self.metric_type = metric_type
        self.return_all_metrics = return_all_metrics

        # Determine split file
        if split_file is None:
            self.split_file = 'data/dataset_splitting/miqa_det_train.csv' if is_train else 'data/dataset_splitting/miqa_det_val.csv'
        else:
            self.split_file = split_file

        # COCO annotation path
        self.coco_annotations = os.path.join(self.root, 'additional_info', 'instances_val2017.json')

        # Validate metric type
        if metric_type not in ['consistency', 'accuracy', 'composite']:
            raise ValueError("metric_type must be either 'consistency', 'accuracy', or 'composite'")

        # distortion types
        self.distortion_types = [
            'contrast', 'darkness', 'pixelate', 'jpeg_compression',
            'motion_blur', 'defocus_blur', 'glass_blur', 'fog',
            'snow', 'gaussian_noise'
        ]

        # Initialize distortion levels
        self._initialize_distortion_levels()

        # Load split CSV
        self.split_data = pd.read_csv(self.split_file)

        self.transforms_dict: Dict[str, Any] = {}
        if isinstance(transforms, (list, tuple)):
            assert len(transforms) == 2, "Expected two transforms in the list."
            self.transforms_dict["cropped"] = transforms[0]
            self.transforms_dict["resized"] = transforms[1]
        else:
            self.transforms_dict["cropped"] = transforms

        # Initialize storage lists
        self.all_image_paths: List[str] = []
        self.all_orig_image_paths: List[str] = []
        self.all_categories: List[str] = []
        self.all_labels: List[float] = []
        self.all_metrics: List[Dict[str, float]] = []
        # self.all_distortion_areas: List[int] = []

        # Process all samples
        self._process_all_samples()

    def _initialize_distortion_levels(self):
        """
        Initialize distortion level mapping for all types.
        Standard: 0-4 (5 levels)
        bg_stronger: 5-14 (10 levels)
        fg_stronger: 15-24 (10 levels)
        """
        self.standard_levels = list(range(5))
        self.bg_stronger_levels = [f'fg{i}_bg{j}' for i in range(1, 5) for j in range(i + 1, 6)]
        self.fg_stronger_levels = [f'fg{i}_bg{j}' for i in range(2, 6) for j in range(1, i)]

        total_levels = len(self.standard_levels) + len(self.bg_stronger_levels) + len(self.fg_stronger_levels)
        assert total_levels == 25, f"Total levels should be 25, got {total_levels}"

        self.level_mapping = {
            'standard': {str(i + 1): i for i in range(5)},
            'bg_stronger': {lvl: idx + 5 for idx, lvl in enumerate(self.bg_stronger_levels)},
            'fg_stronger': {lvl: idx + 15 for idx, lvl in enumerate(self.fg_stronger_levels)}
        }

    def _get_distortion_area(self, filename: str) -> int:
        """
        Extract distortion area label from filename.

        Returns:
            int: 0=standard, 1=fg_stronger, 2=bg_stronger
        """
        if 'bg_stronger' in filename:
            return 2
        elif 'fg_stronger' in filename:
            return 1
        else:
            return 0

    def _process_all_samples(self):
        """Process all images in the split and store metadata."""
        with open(self.coco_annotations, 'r') as f:
            coco_data = json.load(f)

        # Map image_id -> category
        image_id_to_category = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            cat_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == cat_id)
            if img_id not in image_id_to_category:
                image_id_to_category[img_id] = cat_name

        for _, row in tqdm(self.split_data.iterrows(), total=len(self.split_data), desc="Processing split"):
            folder_name = str(row['Image Index']).zfill(12)
            ori_image_path = os.path.join(self.root, 'src_images', folder_name + '.jpg')
            if not os.path.exists(ori_image_path):
                raise FileNotFoundError(f"Original image not found: {ori_image_path}")

            coco_image = next((img for img in coco_data['images'] if str(img['id']).zfill(12) == folder_name), None)
            category = image_id_to_category.get(coco_image['id'], 'Unknown') if coco_image else 'Unknown'

            # Load MMOS CSVs
            mmos_acc_file = os.path.join(self.root, 'labels', f"{folder_name}_accuracy.csv")
            mmos_con_file = os.path.join(self.root, 'labels', f"{folder_name}_consistency.csv")

            if self.metric_type == 'composite':
                mmos_acc_data = pd.read_csv(mmos_acc_file)
                mmos_con_data = pd.read_csv(mmos_con_file)
                mmos_data = mmos_acc_data
            elif self.metric_type == 'consistency':
                mmos_data = pd.read_csv(mmos_con_file)
            else:
                mmos_data = pd.read_csv(mmos_acc_file)

            for idx in range(len(mmos_data)):
                dist_image_name = mmos_data.iloc[idx]['filename']
                image_path = os.path.join(self.root, 'images', folder_name, dist_image_name)

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # distortion_area = self._get_distortion_area(dist_image_name)
                for _ in range(self.patch_num):
                    self.all_image_paths.append(image_path)
                    self.all_orig_image_paths.append(ori_image_path)
                    self.all_categories.append(category)
                    # self.all_distortion_areas.append(distortion_area)

                    if self.metric_type == 'composite':
                        acc_label = float(mmos_acc_data.iloc[idx]['AP_weighted_avg'])
                        con_label = float(mmos_con_data.iloc[idx]['AP_weighted_avg'])
                        self.all_labels.append(0.5 * acc_label + 0.5 * con_label)
                    else:
                        label_key = 'AP_weighted_avg'
                        self.all_labels.append(float(mmos_data.iloc[idx][label_key]))

        print(f"split_file: {self.split_file}, Total samples: {len(self.all_image_paths)}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image as RGB."""
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample: image tensor, label, and metadata."""
        image = self._load_image(self.all_image_paths[idx])
        result_dict = {
            'image_name': os.path.basename(self.all_image_paths[idx]),
            'category': self.all_categories[idx],
            'label': torch.tensor(self.all_labels[idx], dtype=torch.float32),
            # "distortion_region": torch.tensor(self.all_distortion_areas[idx], dtype=torch.long),
        }
        # Apply transforms
        for key, transform in self.transforms_dict.items():
            result_dict[f"image_{key}"] = transform(image)

        return result_dict

    def get_split_info(self) -> Dict[str, List]:
        """Return information about the dataset split."""
        return {
            'image_paths': self.all_image_paths,
            'categories': self.all_categories,
            'quality_scores': self.all_labels
        }


if __name__ == '__main__':
    from torchvision import transforms

    # Example transforms
    crop_transform = transforms.CenterCrop(224)
    resize_transform = transforms.Resize((256, 256))

    # Initialize dataset
    dataset = MIQADETDataset(
        root='/public/datasets/miqa_det',
        split_file=None,  # Use default train split
        patch_num=1,
        transforms=[crop_transform, resize_transform],
        metric_type='composite',
        return_all_metrics=False,
        is_train=True
    )

    # DataLoader example
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    for i, data in enumerate(data_loader):
        print(data.keys())
        if i == 3:
            break


