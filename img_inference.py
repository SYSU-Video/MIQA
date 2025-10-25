import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Image processing imports
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

# Import your existing model components
from models.MIQA_base import get_torch_model, get_timm_model
from models.RA_MIQA import RegionVisionTransformer
from utils.download_utils import ensure_checkpoint_with_gdown

# Model configuration - maps task types to available models and their download URLs
MODEL_CONFIGS = {
    'composite': {
                'cls': {
                    'resnet18': [f"https://drive.google.com/uc?id=1zq03_TRYbg1zYEilP66x6HXpUUQ2sV_H"], #https://drive.google.com/file/d/1zq03_TRYbg1zYEilP66x6HXpUUQ2sV_H/view?usp=sharing
                    'resnet50': [f"https://drive.google.com/uc?id=1y8cV_iOOVNIa66WaAxESqqaOLiCv-GAY"], # https://drive.google.com/file/d/1y8cV_iOOVNIa66WaAxESqqaOLiCv-GAY/view?usp=sharing
                    'efficientnet_b1': [f"https://drive.google.com/uc?id=1ERKTGO18AD2G1J-fr8zjvzoQpSbx6lAo"], # https://drive.google.com/file/d/1ERKTGO18AD2G1J-fr8zjvzoQpSbx6lAo/view?usp=sharing
                    'efficientnet_b5': [f"https://drive.google.com/uc?id=1utE5Rd8onzSlHeve0WYvgDwq4Kctl4zf"], # https://drive.google.com/file/d/1utE5Rd8onzSlHeve0WYvgDwq4Kctl4zf/view?usp=sharing
                    'vit_small_patch16_224': [f"https://drive.google.com/uc?id=11YSVK8rrjMfw3N8XAK_CqzQiL30SuOYZ"], # https://drive.google.com/file/d/11YSVK8rrjMfw3N8XAK_CqzQiL30SuOYZ/view?usp=sharing
                    'ra_miqa': [f"https://drive.google.com/uc?id=1n_NhJcnVpb8dC3B2UZ5ETl2-a96uK0Js"]# https://drive.google.com/file/d/1n_NhJcnVpb8dC3B2UZ5ETl2-a96uK0Js/view?usp=sharing
                    },
                'det': {
                    'resnet18': [f"https://drive.google.com/uc?id=1_5mP7nOc2kla6l4QaTBBs5Xlj4hSu9dE"], #https://drive.google.com/file/d/1_5mP7nOc2kla6l4QaTBBs5Xlj4hSu9dE/view?usp=sharing
                    'resnet50': [f"https://drive.google.com/uc?id=1qLiznF02he6VHEGUDkNr9p0M2-4xO3kr"],#https://drive.google.com/file/d/1qLiznF02he6VHEGUDkNr9p0M2-4xO3kr/view?usp=sharing
                    'efficientnet_b1': [f"https://drive.google.com/uc?id=1vTKaEI_AG7Vnhmrn2B9Rkfblay-GyKvu"],#https://drive.google.com/file/d/1vTKaEI_AG7Vnhmrn2B9Rkfblay-GyKvu/view?usp=sharing
                    'efficientnet_b5': [f"https://drive.google.com/uc?id=1Vx4KcZfisyrfoiZ5zHfBMJpugsFgB82p"],# https://drive.google.com/file/d/1Vx4KcZfisyrfoiZ5zHfBMJpugsFgB82p/view?usp=sharing
                    'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1-KUxxK3j0JflRp2oTKROLEVCBl5q21eF"],#https://drive.google.com/file/d/1-KUxxK3j0JflRp2oTKROLEVCBl5q21eF/view?usp=sharing
                    'ra_miqa': [f"https://drive.google.com/uc?id=1zUcrPOvvYd4rquAm1Wilnh03d8Hj1EDe"] #https://drive.google.com/file/d/1zUcrPOvvYd4rquAm1Wilnh03d8Hj1EDe/view?usp=sharing
                },
                'ins': {
                    'resnet18': [f"https://drive.google.com/uc?id=1umqAI4MiqfPK7dPiro6im_vDA_zrNfRO"],# https://drive.google.com/file/d/1umqAI4MiqfPK7dPiro6im_vDA_zrNfRO/view?usp=sharing
                    'resnet50': [f"https://drive.google.com/uc?id=1Q-zgOoUvXQb3cKtxgC8B9YtbH5YVtYyg"],#https://drive.google.com/file/d/1Q-zgOoUvXQb3cKtxgC8B9YtbH5YVtYyg/view?usp=sharing
                    'efficientnet_b1': [f"https://drive.google.com/uc?id=1aqun7dmtALkYwvhOSWzlnJByDHTPMQVn"],# https://drive.google.com/file/d/1aqun7dmtALkYwvhOSWzlnJByDHTPMQVn/view?usp=sharing
                    'efficientnet_b5': [f"https://drive.google.com/uc?id=1pi2-5Iat1qq0xP9H1vDdlcZBpN5-EUwB"],# https://drive.google.com/file/d/1pi2-5Iat1qq0xP9H1vDdlcZBpN5-EUwB/view?usp=sharing
                    'vit_small_patch16_224': [f"https://drive.google.com/uc?id=10HcI61FEISLbmXME4knZEMBzQmOR8MVs"],# https://drive.google.com/file/d/10HcI61FEISLbmXME4knZEMBzQmOR8MVs/view?usp=sharing
                    'ra_miqa': [f"https://drive.google.com/uc?id=1uvN9jEFuGK5PFQzjiuS9s7A0H9NXyOyc"]#https://drive.google.com/file/d/1uvN9jEFuGK5PFQzjiuS9s7A0H9NXyOyc/view?usp=sharing
                    }
                },
    'consistency': {
            'cls': {
                'resnet18': [f"https://drive.google.com/uc?id=19WGaOFxz9VCFAXrXsy3kz6N7Pqcl7Sb6"], # https://drive.google.com/file/d/19WGaOFxz9VCFAXrXsy3kz6N7Pqcl7Sb6/view?usp=sharing
                'resnet50': [f"https://drive.google.com/uc?id=1VUPGUNatYPTvF_q9iNJ0WUAMLmeCNdPi"], # https://drive.google.com/file/d/1VUPGUNatYPTvF_q9iNJ0WUAMLmeCNdPi/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1gao45m88gRzlY6jbcB3C0B3Y25eJpjvW"], # https://drive.google.com/file/d/1gao45m88gRzlY6jbcB3C0B3Y25eJpjvW/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1ZoRfSGJzu4NrIg7LZ03cLZ5Pwml1Di4o"],# https://drive.google.com/file/d/1ZoRfSGJzu4NrIg7LZ03cLZ5Pwml1Di4o/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1bJrNFAz4hWAP9wO680Kq36EhQ0oCl1sj"] # https://drive.google.com/file/d/1bJrNFAz4hWAP9wO680Kq36EhQ0oCl1sj/view?usp=sharing
            },
            'det': {
                'resnet50': [f"https://drive.google.com/uc?id=1HV_YiDcMGd2GNQDZiJBjq9oJQ4mmkWXs"], #https://drive.google.com/file/d/1HV_YiDcMGd2GNQDZiJBjq9oJQ4mmkWXs/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1stlveb-l4YfDW7Jd5HxqAvtkKoSpBVlO"], # https://drive.google.com/file/d/1stlveb-l4YfDW7Jd5HxqAvtkKoSpBVlO/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1yx7hMh3Bt0qEE_9oNcP5LO_SeBre7sde"], #https://drive.google.com/file/d/1yx7hMh3Bt0qEE_9oNcP5LO_SeBre7sde/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1TvyiN-DPtol0B7k2mo9bPXUoMjJ8F0Xn"] #https://drive.google.com/file/d/1TvyiN-DPtol0B7k2mo9bPXUoMjJ8F0Xn/view?usp=sharing
            },
            'ins': {
                'resnet50': [f"https://drive.google.com/uc?id=1IYpjSy2Mbr0EMw8kagPrMy3ZFd7ggNUw"], #https://drive.google.com/file/d/1IYpjSy2Mbr0EMw8kagPrMy3ZFd7ggNUw/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1mbbalTCfZGvxR9zD03BhZCoOCfKOHYhp"], #https://drive.google.com/file/d/1mbbalTCfZGvxR9zD03BhZCoOCfKOHYhp/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=10VmxqqvpWnd7uxE7mx8WcRqJQNM8dbFo"], #https://drive.google.com/file/d/10VmxqqvpWnd7uxE7mx8WcRqJQNM8dbFo/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1E9H7zerQgf2CUtLhttQBk70AsGb04hih"] # https://drive.google.com/file/d/1E9H7zerQgf2CUtLhttQBk70AsGb04hih/view?usp=sharing
                }
        },
    'accuracy': {
            'cls': {
                'resnet50': [f"https://drive.google.com/uc?id=1mXzm-EuKhLY6zRW0jeVoBAi-kfGfGU0a"], #https://drive.google.com/file/d/1mXzm-EuKhLY6zRW0jeVoBAi-kfGfGU0a/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1qz7Qwrpa6PSwtSgPczADsYf5tVOdujw3"], #https://drive.google.com/file/d/1qz7Qwrpa6PSwtSgPczADsYf5tVOdujw3/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1fkROk-dQ63PdIeqiSIyrs7suDm_sJSFH"], #https://drive.google.com/file/d/1fkROk-dQ63PdIeqiSIyrs7suDm_sJSFH/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1zVhc8Jl1TJYC7Th_4WvwpFiTwac6D6X0"] #https://drive.google.com/file/d/1zVhc8Jl1TJYC7Th_4WvwpFiTwac6D6X0/view?usp=sharing
            },
            'det': {
                'resnet50': [f"https://drive.google.com/uc?id=1e01vieTy4Fdgpqepoi1a1qpenpQLyfei"], #https://drive.google.com/file/d/1e01vieTy4Fdgpqepoi1a1qpenpQLyfei/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1rH36SwceDQ4zSr_exWCvpL_G2AOnCLT-"], #https://drive.google.com/file/d/1rH36SwceDQ4zSr_exWCvpL_G2AOnCLT-/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1K_b29iBLIx1AHCCNaNJUHYx_LT-1Rcwh"], #https://drive.google.com/file/d/1K_b29iBLIx1AHCCNaNJUHYx_LT-1Rcwh/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1gGAM7Wr-65CtN4gUdoLU0ZvN-fdFbosD"] #https://drive.google.com/file/d/1gGAM7Wr-65CtN4gUdoLU0ZvN-fdFbosD/view?usp=sharing
            },
            'ins': {
                'resnet50': [f"https://drive.google.com/uc?id=1qi9uCv_i3fAN6WVoYEHn6mI-BguFYEd-"],#https://drive.google.com/file/d/1qi9uCv_i3fAN6WVoYEHn6mI-BguFYEd-/view?usp=sharing
                'efficientnet_b5': [f"https://drive.google.com/uc?id=1DzgEkhFB182XshMBrh_MsWNHQWOYB3Ea"], #https://drive.google.com/file/d/1DzgEkhFB182XshMBrh_MsWNHQWOYB3Ea/view?usp=sharing
                'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1Ft90uII_kfMLIHsIFJ4X8D4kI_jaxWC3"], #https://drive.google.com/file/d/1Ft90uII_kfMLIHsIFJ4X8D4kI_jaxWC3/view?usp=sharing
                'ra_miqa': [f"https://drive.google.com/uc?id=1eR3ba5E-rbv6d08VBOXJ_EAUCDkVNGa9"] #https://drive.google.com/file/d/1eR3ba5E-rbv6d08VBOXJ_EAUCDkVNGa9/view?usp=sharing
                }
        }
}

# Supported image file extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', 'JPEG', '.png', '.bmp', '.tiff', '.tif'}


class MIQAInference:
    """
    Inference wrapper for MIQA models.

    This class handles model initialization, automatic weight downloading,
    image preprocessing, batch prediction, and result visualization.
    """

    def __init__(self, task: str, model_name: str = 'ra_miqa',
                 metric_type: str = 'composite', device: Optional[str] = None):
        """
        Initialize the MIQA inference system.

        Args:
            task: Task type - 'cls' (classification), 'det' (detection), or 'ins' (instance)
            model_name: Model architecture to use (default: RA_MIQA for best performance)
            metric_type: Training objective - 'composite', 'consistency', or 'accuracy'
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.task = task.lower()
        self.model_name = model_name
        self.metric_type = metric_type

        # Setup logging with clean formatting
        self.logger = self._setup_logger()

        # Determine computation device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"🚀 Initializing MIQA Inference System")
        self.logger.info(f"   Task: {self.task.upper()}")
        self.logger.info(f"   Model: {self.model_name}")
        self.logger.info(f"   Metric Type: {self.metric_type}")
        self.logger.info(f"   Device: {self.device}")

        # Validate configuration
        self._validate_config()

        # Initialize model
        self.model = self._load_model()

        # Setup image preprocessing pipeline
        self.transforms1, self.transforms2 = self._get_transforms()

        self.logger.info("✅ System ready for inference\n")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with both file and console output."""
        logger = logging.getLogger('MIQA_Inference')
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            return logger

        logger.propagate = False

        # Console handler with clean formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def _validate_config(self) -> None:
        """Validate that the requested configuration is supported."""

        if self.metric_type not in ['composite', 'consistency', 'accuracy']:
            raise ValueError(
                f"Invalid metric_type '{self.metric_type}'. "
                f"Supported: ['composite', 'consistency', 'accuracy']"
            )

        if self.task not in MODEL_CONFIGS[self.metric_type]:
            raise ValueError(
                f"Invalid task '{self.task}'. "
                f"Supported tasks: {list(MODEL_CONFIGS[self.metric_type].keys())}"
            )

        if self.model_name not in MODEL_CONFIGS[self.metric_type][self.task]:
            available = list(MODEL_CONFIGS[self.metric_type][self.task].keys())
            raise ValueError(
                f"Model '{self.model_name}' not available for task '{self.task}'. "
                f"Available models: {available}"
            )
    def _get_checkpoint_path(self) -> str:
        """Generate the path where model checkpoint should be stored."""
        base_dir = Path('models') / 'checkpoints' / f'{self.metric_type}_metric'
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = f"miqa_{self.model_name}_{self.task}_{self.metric_type}_metric.pth.tar"
        return str(base_dir / filename)

    def _download_weights(self, checkpoint_path: str) -> bool:
        """
        Download model weights if not present locally.

        Returns:
            True if weights are available (already existed or successfully downloaded)
        """
        if os.path.exists(checkpoint_path):
            self.logger.info(f"✓ Found cached model weights")
            return True

        self.logger.info(f"⏬ Downloading model weights (first time only)...")
        download_urls = MODEL_CONFIGS[self.metric_type][self.task][self.model_name]


        if ensure_checkpoint_with_gdown(checkpoint_path, download_urls):
            self.logger.info(f"✓ Successfully downloaded model weights")
            return True
        else:
            self.logger.error(f"❌ Failed to download model weights")
            return False

    def _create_model(self) -> torch.nn.Module:
        """Create the model architecture."""
        if self.model_name == 'ra_miqa':
            self.logger.info("Building Region-Aware Vision Transformer...")
            model = RegionVisionTransformer(
                base_model_name='vit_small_patch16_224',
                pretrained=False,  # We'll load our trained weights
                mmseg_config_path='models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
                checkpoint_path='models/checkpoints/sere_finetuned_vit_small_ep100.pth'
            )
        else:
            try:
                self.logger.info(f"Building {self.model_name} from PyTorch...")
                model = get_torch_model(model_name=self.model_name, pretrained=False, num_classes=1)
            except Exception:
                self.logger.info(f"Building {self.model_name} from timm library...")
                model = get_timm_model(model_name=self.model_name, pretrained=False, num_classes=1)

        return model

    def _load_model(self) -> torch.nn.Module:
        """Load model with pre-trained weights."""
        checkpoint_path = self._get_checkpoint_path()

        # Ensure weights are available
        if not self._download_weights(checkpoint_path):
            raise RuntimeError("Cannot proceed without model weights")

        # Create model architecture
        self.logger.info("🔧 Loading model...")
        model = self._create_model()

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove 'module.' prefix if present (from DataParallel training)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode

        self.logger.info("✓ Model loaded successfully")

        return model

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose | None]:
        """
        Return preprocessing transforms based on model type.
        """
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        SIMPLE_MEAN = (0.5, 0.5, 0.5)
        SIMPLE_STD = (0.5, 0.5, 0.5)

        # Default (for single-input backbones)
        transform_imagenet = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        transform_simple = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIMPLE_MEAN, std=SIMPLE_STD)
        ])

        # 1️⃣ CNNs（ResNet / EfficientNet）
        if any(k in self.model_name for k in ['resnet', 'efficientnet']):
            return transform_imagenet, None

        # 2️⃣ ViT
        elif 'vit' in self.model_name:
            return transform_simple, None

        # 3️⃣ ra_miqa
        elif 'ra_miqa' in self.model_name:
            transform_1 = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=SIMPLE_MEAN, std=SIMPLE_STD)
            ])
            transform_2 = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop((288, 288)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            return transform_1, transform_2

        # fallback
        else:
            print(f"[Warning] Unknown model type '{self.model_name}', using ImageNet normalization.")
            return transform_imagenet, None

    def _prepare_image(self, image_path: str):
        """
        Load and preprocess a single image.
        Return (img1, img2, original_img)
        """
        img = Image.open(image_path).convert('RGB')
        img1 = self.transforms1(img).unsqueeze(0)
        img2 = self.transforms2(img).unsqueeze(0) if self.transforms2 else None
        return img1, img2, img

    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict:
        """
        Prediction interface for different backbones.
        """
        img1, img2, original_img = self._prepare_image(image_path)

        img1 = img1.to(self.device)
        if img2 is not None:
            img2 = img2.to(self.device)

        if img2 is None:
            output = self.model(img1)
        else:
            output = self.model(img1, img2)

        score = output.item() if torch.is_tensor(output) else float(output)

        return {
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'quality_score': score,
            'original_image': original_img
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: List[str],
                      show_progress: bool = True) -> List[Dict]:
        """
        Run inference on multiple images with progress tracking.

        Args:
            image_paths: List of paths to image files
            show_progress: Whether to display progress bar

        Returns:
            List of prediction results for each image
        """
        results = []

        # Create progress bar if requested
        iterator = tqdm(image_paths, desc="Processing images",
                        disable=not show_progress, ncols=80)

        for img_path in iterator:
            try:
                result = self.predict_single(img_path)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to process {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'image_name': Path(img_path).name,
                    'quality_score': None,
                    'error': str(e)
                })

        return results

    def predict(self, input_path: str, show_progress: bool = True) -> List[Dict]:
        """
        Main prediction interface - handles both single images and directories.

        Args:
            input_path: Path to an image file or directory containing images
            show_progress: Whether to show progress bar for batch processing

        Returns:
            List of prediction results
        """
        input_path = Path(input_path)

        # Handle single file
        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: {input_path.suffix}. "
                    f"Supported: {SUPPORTED_EXTENSIONS}"
                )
            return [self.predict_single(str(input_path))]

        # Handle directory
        elif input_path.is_dir():
            # Find all supported images in directory
            image_paths = []
            for ext in SUPPORTED_EXTENSIONS:
                image_paths.extend(input_path.glob(f"*{ext}"))
                # image_paths.extend(input_path.glob(f"*{ext.upper()}"))

            image_paths = sorted([str(p) for p in image_paths])

            if not image_paths:
                raise ValueError(f"No supported images found in {input_path}")

            self.logger.info(f"📁 Found {len(image_paths)} images in directory")
            return self.predict_batch(image_paths, show_progress)

        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def visualize_results(self, results: List[Dict], output_dir: str = 'inference_results',
                          score_range: Tuple[float, float] = (0, 100)) -> None:
        """
        Create annotated visualizations of predictions.

        This generates images with quality scores overlaid in a color-coded box.
        Low scores appear in red, high scores in green.

        Args:
            results: List of prediction results
            output_dir: Directory to save visualizations
            score_range: Expected range of quality scores for color normalization
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"\n🎨 Creating visualizations...")

        for result in tqdm(results, desc="Generating visualizations", ncols=80):
            if result.get('quality_score') is None:
                continue  # Skip failed predictions

            img = result['original_image'].copy()
            draw = ImageDraw.Draw(img)

            # Prepare score display
            score = result['quality_score']
            score_text = f"Quality: {score:.4f}"

            # Normalize score to [0, 1] for color interpolation
            norm_score = (score - score_range[0]) / (score_range[1] - score_range[0])
            norm_score = max(0, min(1, norm_score))  # Clamp to [0, 1]

            # Color interpolation: red (low quality) -> yellow -> green (high quality)
            if norm_score < 0.5:
                # Red to yellow
                r = 255
                g = int(255 * (norm_score * 2))
                b = 0
            else:
                # Yellow to green
                r = int(255 * (2 - norm_score * 2))
                g = 255
                b = 0

            color = (r, g, b)

            # Draw colored box with score
            box_width = 250
            box_height = 50
            margin = 10

            # Position in top-left corner
            box_coords = [margin, margin, margin + box_width, margin + box_height]
            draw.rectangle(box_coords, fill=color)

            # Add text (try to load a nice font, fall back to default)
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()

            # Calculate text position to center it in the box
            bbox = draw.textbbox((0, 0), score_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = margin + (box_width - text_width) // 2
            text_y = margin + (box_height - text_height) // 2

            draw.text((text_x, text_y), score_text, fill='black', font=font)

            # Save annotated image
            output_file = output_path / f"miqa_{self.model_name}_{result['image_name']}"
            img.save(output_file)

        self.logger.info(f"✓ Visualizations saved to: {output_dir}/")

    def save_results(self, results: List[Dict], output_path: str = 'predictions.json',
                     format: str = 'json') -> None:
        """
        Save prediction results to file.

        Args:
            results: List of prediction results
            output_path: Path to save results
            format: Output format - 'json' or 'csv'
        """
        # Remove PIL image objects before saving
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k != 'original_image'}
            clean_results.append(clean_r)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'task': self.task,
                        'model': self.model_name,
                        'metric_type': self.metric_type,
                        'timestamp': datetime.now().isoformat(),
                        'num_images': len(clean_results)
                    },
                    'predictions': clean_results
                }, f, indent=2)

        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                if clean_results:
                    writer = csv.DictWriter(f, fieldnames=clean_results[0].keys())
                    writer.writeheader()
                    writer.writerows(clean_results)

        self.logger.info(f"💾 Results saved to: {output_path}")

    def print_summary(self, results: List[Dict]) -> None:
        """Print a formatted summary of prediction results."""
        valid_results = [r for r in results if r.get('quality_score') is not None]
        failed_results = [r for r in results if r.get('quality_score') is None]

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREDICTION SUMMARY")
        self.logger.info("=" * 80)

        if valid_results:
            scores = [r['quality_score'] for r in valid_results]
            self.logger.info(f"✓ Successfully processed: {len(valid_results)} images")
            self.logger.info(f"  Average quality score: {np.mean(scores):.4f}")
            self.logger.info(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            self.logger.info(f"  Standard deviation: {np.std(scores):.4f}")

            # Show top and bottom quality images
            sorted_results = sorted(valid_results, key=lambda x: x['quality_score'], reverse=True)

            self.logger.info("\n🏆 Top 3 quality images:")
            for i, r in enumerate(sorted_results[:3], 1):
                self.logger.info(f"  {i}. {r['image_name']}: {r['quality_score']:.4f}")

            if len(sorted_results) > 3:
                self.logger.info("\n⚠️  Bottom 3 quality images:")
                for i, r in enumerate(sorted_results[-3:], 1):
                    self.logger.info(f"  {i}. {r['image_name']}: {r['quality_score']:.4f}")

        if failed_results:
            self.logger.info(f"\n❌ Failed to process: {len(failed_results)} images")

        self.logger.info("=" * 80 + "\n")


def main():
    """Command-line interface for MIQA inference."""
    parser = argparse.ArgumentParser(
        description='MIQA: Machine-centric Image Quality Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Predict quality of a single image
      python img_inference.py --input image.jpg --task cls --model ra_miqa
    
      # Process all images in a directory
      python img_inference.py --input ./assets/demo_images/imagenet_demo --task det --model ra_miqa
    
      # Save results and create visualizations
      python img_inference.py --input /assets/demo_images/imagenet_demo --task ins --save-results --visualize
            """
        )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory containing images')
    parser.add_argument('--task', type=str, required=True,
                        choices=['cls', 'det', 'ins'],
                        help='Task type: cls (classification), det (detection), ins (instance)')
    parser.add_argument('--model', type=str, default='ra_miqa',
                        choices=['resnet50', 'efficientnet_b5', 'vit_small_patch16_224', 'ra_miqa'],
                        help='Model architecture (default: ra_miqa)')
    parser.add_argument('--metric-type', type=str, default='composite',
                        choices=['composite', 'consistency', 'accuracy'],
                        help='Training metric type (default: both)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run on (auto-detect if not specified)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save prediction results to file')
    parser.add_argument('--output-file', type=str, default='predictions.json',
                        help='Output file path for results (default: predictions.json)')
    parser.add_argument('--output-format', type=str, default='json',
                        choices=['json', 'csv'],
                        help='Output file format (default: json)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create annotated visualizations of predictions')
    parser.add_argument('--save-dir', type=str, default='inference_results',
                        help='Directory for save (default: inference_results)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')

    args = parser.parse_args()

    try:
        # Initialize inference system
        miqa = MIQAInference(
            task=args.task,
            model_name=args.model,
            metric_type=args.metric_type,
            device=args.device
        )

        # Run predictions
        results = miqa.predict(args.input, show_progress=not args.no_progress)

        # Print summary
        miqa.print_summary(results)

        args.save_dir = Path(args.save_dir)/ 'image' / args.task / args.metric_type
        args.output_file = f'miqa_{args.model}_'+args.output_file

        # Save results if requested
        if args.save_results:
            miqa.save_results(results, Path(args.save_dir)/args.output_file, args.output_format)

        # Create visualizations if requested
        if args.visualize:
            miqa.visualize_results(results, args.save_dir)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
