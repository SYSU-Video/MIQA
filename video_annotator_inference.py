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
import cv2  # OpenCV for video processing
import matplotlib.pyplot as plt  # Matplotlib for plotting
import io

# Image processing imports
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

# Import your existing model components
# Ensure these files (models/, utils/) are in the same directory or accessible in PYTHONPATH
from models.MIQA_base import get_torch_model, get_timm_model
from models.RA_MIQA import RegionVisionTransformer
from utils.download_utils import ensure_checkpoint_with_gdown

MODEL_CONFIGS = {
    'composite': {
        'cls': {
            'resnet18': [f"https://drive.google.com/uc?id=1zq03_TRYbg1zYEilP66x6HXpUUQ2sV_H"],
            # https://drive.google.com/file/d/1zq03_TRYbg1zYEilP66x6HXpUUQ2sV_H/view?usp=sharing
            'resnet50': [f"https://drive.google.com/uc?id=1y8cV_iOOVNIa66WaAxESqqaOLiCv-GAY"],
            # https://drive.google.com/file/d/1y8cV_iOOVNIa66WaAxESqqaOLiCv-GAY/view?usp=sharing
            'efficientnet_b1': [f"https://drive.google.com/uc?id=1ERKTGO18AD2G1J-fr8zjvzoQpSbx6lAo"],
            # https://drive.google.com/file/d/1ERKTGO18AD2G1J-fr8zjvzoQpSbx6lAo/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1utE5Rd8onzSlHeve0WYvgDwq4Kctl4zf"],
            # https://drive.google.com/file/d/1utE5Rd8onzSlHeve0WYvgDwq4Kctl4zf/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=11YSVK8rrjMfw3N8XAK_CqzQiL30SuOYZ"],
            # https://drive.google.com/file/d/11YSVK8rrjMfw3N8XAK_CqzQiL30SuOYZ/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1n_NhJcnVpb8dC3B2UZ5ETl2-a96uK0Js"]
            # https://drive.google.com/file/d/1n_NhJcnVpb8dC3B2UZ5ETl2-a96uK0Js/view?usp=sharing
        },
        'det': {
            'resnet18': [f"https://drive.google.com/uc?id=1_5mP7nOc2kla6l4QaTBBs5Xlj4hSu9dE"],
            # https://drive.google.com/file/d/1_5mP7nOc2kla6l4QaTBBs5Xlj4hSu9dE/view?usp=sharing
            'resnet50': [f"https://drive.google.com/uc?id=1qLiznF02he6VHEGUDkNr9p0M2-4xO3kr"],
            # https://drive.google.com/file/d/1qLiznF02he6VHEGUDkNr9p0M2-4xO3kr/view?usp=sharing
            'efficientnet_b1': [f"https://drive.google.com/uc?id=1vTKaEI_AG7Vnhmrn2B9Rkfblay-GyKvu"],
            # https://drive.google.com/file/d/1vTKaEI_AG7Vnhmrn2B9Rkfblay-GyKvu/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1Vx4KcZfisyrfoiZ5zHfBMJpugsFgB82p"],
            # https://drive.google.com/file/d/1Vx4KcZfisyrfoiZ5zHfBMJpugsFgB82p/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1-KUxxK3j0JflRp2oTKROLEVCBl5q21eF"],
            # https://drive.google.com/file/d/1-KUxxK3j0JflRp2oTKROLEVCBl5q21eF/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1zUcrPOvvYd4rquAm1Wilnh03d8Hj1EDe"]
            # https://drive.google.com/file/d/1zUcrPOvvYd4rquAm1Wilnh03d8Hj1EDe/view?usp=sharing
        },
        'ins': {
            'resnet18': [f"https://drive.google.com/uc?id=1umqAI4MiqfPK7dPiro6im_vDA_zrNfRO"],
            # https://drive.google.com/file/d/1umqAI4MiqfPK7dPiro6im_vDA_zrNfRO/view?usp=sharing
            'resnet50': [f"https://drive.google.com/uc?id=1Q-zgOoUvXQb3cKtxgC8B9YtbH5YVtYyg"],
            # https://drive.google.com/file/d/1Q-zgOoUvXQb3cKtxgC8B9YtbH5YVtYyg/view?usp=sharing
            'efficientnet_b1': [f"https://drive.google.com/uc?id=1aqun7dmtALkYwvhOSWzlnJByDHTPMQVn"],
            # https://drive.google.com/file/d/1aqun7dmtALkYwvhOSWzlnJByDHTPMQVn/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1pi2-5Iat1qq0xP9H1vDdlcZBpN5-EUwB"],
            # https://drive.google.com/file/d/1pi2-5Iat1qq0xP9H1vDdlcZBpN5-EUwB/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=10HcI61FEISLbmXME4knZEMBzQmOR8MVs"],
            # https://drive.google.com/file/d/10HcI61FEISLbmXME4knZEMBzQmOR8MVs/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1uvN9jEFuGK5PFQzjiuS9s7A0H9NXyOyc"]
            # https://drive.google.com/file/d/1uvN9jEFuGK5PFQzjiuS9s7A0H9NXyOyc/view?usp=sharing
        }
    },
    'consistency': {
        'cls': {
            'resnet18': [f"https://drive.google.com/uc?id=19WGaOFxz9VCFAXrXsy3kz6N7Pqcl7Sb6"],
            # https://drive.google.com/file/d/19WGaOFxz9VCFAXrXsy3kz6N7Pqcl7Sb6/view?usp=sharing
            'resnet50': [f"https://drive.google.com/uc?id=1VUPGUNatYPTvF_q9iNJ0WUAMLmeCNdPi"],
            # https://drive.google.com/file/d/1VUPGUNatYPTvF_q9iNJ0WUAMLmeCNdPi/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1gao45m88gRzlY6jbcB3C0B3Y25eJpjvW"],
            # https://drive.google.com/file/d/1gao45m88gRzlY6jbcB3C0B3Y25eJpjvW/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1ZoRfSGJzu4NrIg7LZ03cLZ5Pwml1Di4o"],
            # https://drive.google.com/file/d/1ZoRfSGJzu4NrIg7LZ03cLZ5Pwml1Di4o/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1bJrNFAz4hWAP9wO680Kq36EhQ0oCl1sj"]
            # https://drive.google.com/file/d/1bJrNFAz4hWAP9wO680Kq36EhQ0oCl1sj/view?usp=sharing
        },
        'det': {
            'resnet50': [f"https://drive.google.com/uc?id=1HV_YiDcMGd2GNQDZiJBjq9oJQ4mmkWXs"],
            # https://drive.google.com/file/d/1HV_YiDcMGd2GNQDZiJBjq9oJQ4mmkWXs/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1stlveb-l4YfDW7Jd5HxqAvtkKoSpBVlO"],
            # https://drive.google.com/file/d/1stlveb-l4YfDW7Jd5HxqAvtkKoSpBVlO/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1yx7hMh3Bt0qEE_9oNcP5LO_SeBre7sde"],
            # https://drive.google.com/file/d/1yx7hMh3Bt0qEE_9oNcP5LO_SeBre7sde/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1TvyiN-DPtol0B7k2mo9bPXUoMjJ8F0Xn"]
            # https://drive.google.com/file/d/1TvyiN-DPtol0B7k2mo9bPXUoMjJ8F0Xn/view?usp=sharing
        },
        'ins': {
            'resnet50': [f"https://drive.google.com/uc?id=1IYpjSy2Mbr0EMw8kagPrMy3ZFd7ggNUw"],
            # https://drive.google.com/file/d/1IYpjSy2Mbr0EMw8kagPrMy3ZFd7ggNUw/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1mbbalTCfZGvxR9zD03BhZCoOCfKOHYhp"],
            # https://drive.google.com/file/d/1mbbalTCfZGvxR9zD03BhZCoOCfKOHYhp/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=10VmxqqvpWnd7uxE7mx8WcRqJQNM8dbFo"],
            # https://drive.google.com/file/d/10VmxqqvpWnd7uxE7mx8WcRqJQNM8dbFo/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1E9H7zerQgf2CUtLhttQBk70AsGb04hih"]
            # https://drive.google.com/file/d/1E9H7zerQgf2CUtLhttQBk70AsGb04hih/view?usp=sharing
        }
    },
    'accuracy': {
        'cls': {
            'resnet50': [f"https://drive.google.com/uc?id=1mXzm-EuKhLY6zRW0jeVoBAi-kfGfGU0a"],
            # https://drive.google.com/file/d/1mXzm-EuKhLY6zRW0jeVoBAi-kfGfGU0a/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1qz7Qwrpa6PSwtSgPczADsYf5tVOdujw3"],
            # https://drive.google.com/file/d/1qz7Qwrpa6PSwtSgPczADsYf5tVOdujw3/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1fkROk-dQ63PdIeqiSIyrs7suDm_sJSFH"],
            # https://drive.google.com/file/d/1fkROk-dQ63PdIeqiSIyrs7suDm_sJSFH/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1zVhc8Jl1TJYC7Th_4WvwpFiTwac6D6X0"]
            # https://drive.google.com/file/d/1zVhc8Jl1TJYC7Th_4WvwpFiTwac6D6X0/view?usp=sharing
        },
        'det': {
            'resnet50': [f"https://drive.google.com/uc?id=1e01vieTy4Fdgpqepoi1a1qpenpQLyfei"],
            # https://drive.google.com/file/d/1e01vieTy4Fdgpqepoi1a1qpenpQLyfei/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1rH36SwceDQ4zSr_exWCvpL_G2AOnCLT-"],
            # https://drive.google.com/file/d/1rH36SwceDQ4zSr_exWCvpL_G2AOnCLT-/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1K_b29iBLIx1AHCCNaNJUHYx_LT-1Rcwh"],
            # https://drive.google.com/file/d/1K_b29iBLIx1AHCCNaNJUHYx_LT-1Rcwh/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1gGAM7Wr-65CtN4gUdoLU0ZvN-fdFbosD"]
            # https://drive.google.com/file/d/1gGAM7Wr-65CtN4gUdoLU0ZvN-fdFbosD/view?usp=sharing
        },
        'ins': {
            'resnet50': [f"https://drive.google.com/uc?id=1qi9uCv_i3fAN6WVoYEHn6mI-BguFYEd-"],
            # https://drive.google.com/file/d/1qi9uCv_i3fAN6WVoYEHn6mI-BguFYEd-/view?usp=sharing
            'efficientnet_b5': [f"https://drive.google.com/uc?id=1DzgEkhFB182XshMBrh_MsWNHQWOYB3Ea"],
            # https://drive.google.com/file/d/1DzgEkhFB182XshMBrh_MsWNHQWOYB3Ea/view?usp=sharing
            'vit_small_patch16_224': [f"https://drive.google.com/uc?id=1Ft90uII_kfMLIHsIFJ4X8D4kI_jaxWC3"],
            # https://drive.google.com/file/d/1Ft90uII_kfMLIHsIFJ4X8D4kI_jaxWC3/view?usp=sharing
            'ra_miqa': [f"https://drive.google.com/uc?id=1eR3ba5E-rbv6d08VBOXJ_EAUCDkVNGa9"]
            # https://drive.google.com/file/d/1eR3ba5E-rbv6d08VBOXJ_EAUCDkVNGa9/view?usp=sharing
        }
    }
}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


class MIQAInference:
    """
    MODIFIED Inference wrapper for MIQA models.
    Now includes a method to predict on PIL Image objects directly.
    """

    def __init__(self, task: str, model_name: str = 'ra_miqa',
                 metric_type: str = 'composite', device: Optional[str] = None):
        self.task = task.lower()
        self.model_name = model_name
        self.metric_type = metric_type
        self.logger = self._setup_logger()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"🚀 Initializing MIQA Inference System")
        self.logger.info(f"   Task: {self.task.upper()}")
        self.logger.info(f"   Model: {self.model_name}")
        self.logger.info(f"   Metric Type: {self.metric_type}")
        self.logger.info(f"   Device: {self.device}")

        self._validate_config()
        self.model = self._load_model()
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

    @torch.no_grad()
    def predict_image_object(self, image: Image.Image) -> float:
        """
        NEW METHOD: Run inference on a PIL Image object.
        """
        # Preprocess the image
        img1 = self.transforms1(image).unsqueeze(0).to(self.device)
        img2 = self.transforms2(image).unsqueeze(0).to(self.device) if self.transforms2 else None

        # Run inference based on model input requirements
        if img2 is None:
            output = self.model(img1)
        else:
            output = self.model(img1, img2)

        score = output.item() if torch.is_tensor(output) else float(output)
        return score


class VideoMIQAProcessor:
    """
    A wrapper to process videos using the MIQAInference engine and create
    a visualized output video with scores and plots.
    """
    # --- Visualization Constants ---
    PANEL_WIDTH = 480
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_L = 1.0
    FONT_SCALE_M = 0.8
    FONT_COLOR = (255, 255, 255)  # White
    LINE_THICKNESS = 2

    # Plotting style
    plt.style.use('dark_background')

    def __init__(self, miqa_engine: MIQAInference):
        self.miqa_engine = miqa_engine
        self.logger = miqa_engine.logger

    def _create_score_plot(self, scores: List[float], width: int, height: int) -> np.ndarray:
        """
        Creates a line chart of scores using Matplotlib and returns it as an OpenCV image.
        """
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.plot(scores, color='#4287f5', linewidth=2)
        ax.set_xlim(0, max(1, len(scores)))
        ax.set_ylim(0, 1)
        ax.set_title("Quality Score Fluctuation", fontsize=10)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=1.5)

        # Render plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Convert buffer to a PIL Image and then to an OpenCV image
        plot_img_pil = Image.open(buf)
        plot_img_np = np.array(plot_img_pil)
        plot_img_bgr = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)

        return plot_img_bgr

    def process_video(self, input_path: str, output_path: str):
        """
        Reads a video, analyzes each frame for quality, and writes an annotated output video.
        """
        self.logger.info(f"📹 Starting processing for: {Path(input_path).name}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"❌ Failed to open video: {input_path}")
            return

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # New dimensions for output video (with side panel)
        output_width = orig_width + self.PANEL_WIDTH
        output_height = orig_height

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        scores = []
        progress_bar = tqdm(range(frame_count), desc="Analyzing frames", ncols=100)

        for frame_idx in progress_bar:
            ret, frame = cap.read()
            if not ret:
                break

            # --- MIQA Inference ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            score = self.miqa_engine.predict_image_object(pil_image)
            scores.append(score)

            # --- Visualization Panel ---
            panel = np.zeros((orig_height, self.PANEL_WIDTH, 3), dtype=np.uint8)

            # 1. Task Info
            task_text = f"Task: {self.miqa_engine.task.upper()}"
            cv2.putText(panel, task_text, (20, 50), self.FONT, self.FONT_SCALE_M, self.FONT_COLOR, self.LINE_THICKNESS)

            # 2. Current Score
            score_text = f"Quality Score: {score:.3f}"
            # Color coding for score text
            norm_score = max(0, score)
            if norm_score < 0.5:
                color = (0, int(255 * (norm_score * 2)), 255)  # Red -> Yellow
            else:
                color = (0, 255, int(255 * (2 - norm_score * 2)))  # Yellow -> Green
            cv2.putText(panel, score_text, (20, 110), self.FONT, self.FONT_SCALE_L, color, self.LINE_THICKNESS + 1)

            # 3. Frame Info
            frame_text = f"Frame: {frame_idx + 1}/{frame_count}"
            cv2.putText(panel, frame_text, (20, orig_height - 30), self.FONT, self.FONT_SCALE_M, self.FONT_COLOR, 1)

            # 4. Score Plot
            if len(scores) > 1:
                plot_height = 300
                plot_width = self.PANEL_WIDTH - 40  # with margins
                plot_img = self._create_score_plot(scores, plot_width, plot_height)

                # Position the plot on the panel
                y_offset = 160
                panel[y_offset:y_offset + plot_img.shape[0], 20:20 + plot_img.shape[1]] = plot_img

            # --- Combine and Write Frame ---
            combined_frame = np.concatenate((frame, panel), axis=1)
            out.write(combined_frame)

        # Release resources
        cap.release()
        out.release()
        self.logger.info(f"✅ Finished processing. Annotated video saved to: {output_path}\n")


def main():
    """Command-line interface for Video MIQA inference."""
    parser = argparse.ArgumentParser(
        description='MIQA for Video: Machine-centric Image Quality Assessment on Video Frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Analyze a single video and save the annotated output
      python video_annotator_inference.py --input my_video.mp4 --task cls --model ra_miqa

      # Analyze all videos in a directory
      python video_annotator_inference.py --input ./video_folder/ --task det --model resnet50
            """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video file or a directory containing videos.')
    parser.add_argument('--task', type=str, required=True,
                        choices=['cls', 'det', 'ins'],
                        help='Task type: cls (classification), det (detection), ins (instance).')
    parser.add_argument('--model', type=str, default='ra_miqa',
                        choices=['resnet18', 'resnet50', 'efficientnet_b1', 'efficientnet_b5', 'vit_small_patch16_224',
                                 'ra_miqa'],
                        help='Model architecture (default: ra_miqa).')
    parser.add_argument('--metric-type', type=str, default='composite',
                        choices=['composite', 'consistency', 'accuracy'],
                        help='Training metric type (default: composite).')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run on (auto-detect if not specified).')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Directory to save the output annotated videos.')

    args = parser.parse_args()

    try:
        # Initialize the core inference engine
        miqa_engine = MIQAInference(
            task=args.task,
            model_name=args.model,
            metric_type=args.metric_type,
            device=args.device
        )

        # Initialize the video processor
        video_processor = VideoMIQAProcessor(miqa_engine)

        # Find videos to process
        input_path = Path(args.input)
        videos_to_process = []
        if input_path.is_dir():
            for ext in SUPPORTED_VIDEO_EXTENSIONS:
                videos_to_process.extend(input_path.glob(f"*{ext}"))
        elif input_path.is_file() and input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            videos_to_process.append(input_path)

        if not videos_to_process:
            raise FileNotFoundError(f"No supported video files found in '{args.input}'")


        # Create output directory
        output_dir = Path(args.output_dir) / 'video' /args.task / args.metric_type
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each video
        for video_path in videos_to_process:
            output_filename = f"{video_path.stem}_miqa_{args.model}_{args.task}.mp4"
            output_filepath = str(output_dir / output_filename)
            video_processor.process_video(str(video_path), output_filepath)

    except Exception as e:
        # Use the logger if it exists, otherwise print
        try:
            miqa_engine.logger.error(f"\n❌ An error occurred: {str(e)}")
        except:
            print(f"\n❌ An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()