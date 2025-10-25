import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict, defaultdict
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2

# Image processing imports
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# Import your existing model components
from models.MIQA_base import get_torch_model, get_timm_model
from models.RA_MIQA import RegionVisionTransformer
from utils.download_utils import ensure_checkpoint_with_gdown

# Model configuration
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

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.JPEG', '.png', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}


class VideoFrameExtractor:
    """
    Extracts and samples frames from video files intelligently.

    This class handles different sampling strategies to balance between
    thoroughness and computational efficiency.
    """

    def __init__(self, sampling_strategy: str = 'uniform',
                 target_frames: int = 30,
                 fps_sample: Optional[float] = None):
        """
        Initialize the frame extractor.

        Args:
            sampling_strategy: How to sample frames - 'uniform', 'fps', or 'keyframe'
            target_frames: Target number of frames to extract (for uniform sampling)
            fps_sample: Sample rate in frames per second (for fps sampling)
        """
        self.sampling_strategy = sampling_strategy
        self.target_frames = target_frames
        self.fps_sample = fps_sample

    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float], Dict]:
        """
        Extract frames from video based on sampling strategy.

        Returns:
            Tuple of (frames_list, timestamps_list, video_metadata)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height
        }

        # Determine which frames to sample
        frame_indices = self._get_sample_indices(total_frames, fps)

        frames = []
        timestamps = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                # Calculate timestamp in seconds
                timestamp = idx / fps if fps > 0 else idx
                timestamps.append(timestamp)

        cap.release()

        return frames, timestamps, metadata

    def _get_sample_indices(self, total_frames: int, fps: float) -> List[int]:
        """
        Determine which frame indices to sample based on strategy.
        """
        if self.sampling_strategy == 'uniform':
            # Sample frames uniformly across the video
            if total_frames <= self.target_frames:
                return list(range(total_frames))
            else:
                # Calculate step size to get approximately target_frames
                step = total_frames / self.target_frames
                indices = [int(i * step) for i in range(self.target_frames)]
                return indices

        elif self.sampling_strategy == 'fps':
            # Sample at a specific frame rate
            if self.fps_sample is None:
                raise ValueError("fps_sample must be specified for fps sampling strategy")

            frame_interval = max(1, int(fps / self.fps_sample))
            indices = list(range(0, total_frames, frame_interval))
            return indices

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")


def aggregate_scores_by_second(frame_results: List[Dict]) -> List[Dict]:
    """
    Aggregate frame-level quality scores to per-second averages.

    This function groups all frames that fall within the same second
    and computes their average quality score. This provides a smoothed
    view of quality over time, reducing noise from frame-to-frame variations.

    Args:
        frame_results: List of dictionaries with 'timestamp' and 'quality_score'

    Returns:
        List of dictionaries with per-second aggregated scores
    """
    # Group frames by their second (floor of timestamp)
    seconds_data = defaultdict(list)

    for frame in frame_results:
        second = int(frame['timestamp'])  # Floor to nearest second
        seconds_data[second].append(frame['quality_score'])

    # Calculate average for each second
    per_second_results = []
    for second in sorted(seconds_data.keys()):
        scores = seconds_data[second]
        per_second_results.append({
            'second': second,
            'timestamp': float(second),  # Use second as timestamp for plotting
            'quality_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'num_frames': len(scores),
            'std_score': np.std(scores) if len(scores) > 1 else 0.0
        })

    return per_second_results


class MIQAInference:
    """
    Inference wrapper for MIQA models supporting both images and videos.
    """

    def __init__(self, task: str, model_name: str = 'ra_miqa',
                 metric_type: str = 'composite', device: Optional[str] = None,
                 video_sampling: str = 'uniform', video_target_frames: int = 30):
        """
        Initialize the MIQA inference system.

        Args:
            task: Task type - 'cls', 'det', or 'ins'
            model_name: Model architecture to use
            metric_type: Training objective - 'composite', 'consistency', or 'accuracy'
            device: Device to run inference on
            video_sampling: Frame sampling strategy for videos
            video_target_frames: Target number of frames to extract from videos
        """
        self.task = task.lower()
        self.model_name = model_name
        self.metric_type = metric_type
        self.video_target_frames = video_target_frames

        # Setup logging
        self.logger = self._setup_logger()

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"🚀 Initializing MIQA Inference System")
        self.logger.info(f"   Task: {self.task.upper()}")
        self.logger.info(f"   Model: {self.model_name}")
        self.logger.info(f"   Device: {self.device}")

        # Validate configuration
        self._validate_config()

        # Initialize model
        self.model = self._load_model()

        # Setup image preprocessing
        self.transforms1, self.transforms2 = self._get_transforms()

        # Initialize video frame extractor
        self.frame_extractor = VideoFrameExtractor(
            sampling_strategy=video_sampling,
            target_frames=video_target_frames
        )

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
        """Load model with weights."""
        checkpoint_path = self._get_checkpoint_path()

        if not self._download_weights(checkpoint_path):
            raise RuntimeError("Cannot proceed without model weights")

        self.logger.info("🔧 Loading model...")
        model = self._create_model()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        self.logger.info("✓ Model loaded successfully")
        return model

    def _get_transforms(self) -> [transforms.Compose, transforms.Compose]:
        """
        Get image preprocessing transforms.

        These transforms normalize images to match the training distribution.
        """
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        SIMPLE_MEAN = (0.5, 0.5, 0.5)
        SIMPLE_STD = (0.5, 0.5, 0.5)

        transforms_list1 = [
            transforms.Resize(288),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIMPLE_MEAN,
                                 std=SIMPLE_STD)
        ]
        transform_list_2 = [
            transforms.Resize(288),
            transforms.CenterCrop((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)
        ]
        return transforms.Compose(transforms_list1), transforms.Compose(transform_list_2)

    def _prepare_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a video frame for model input.

        Args:
            frame: Numpy array in RGB format

        Returns:
            Tuple of (cropped_tensor, resized_tensor)
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)

        # Apply transforms
        img1 = self.transforms1(img).unsqueeze(0)
        img2 = self.transforms2(img).unsqueeze(0)

        return img1, img2

    def _prepare_image(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
        """Load and preprocess an image file."""
        img = Image.open(image_path).convert('RGB')
        img1 = self.transforms1(img).unsqueeze(0)
        img2 = self.transforms2(img).unsqueeze(0)
        return img1, img2, img

    @torch.no_grad()
    def predict_single_image(self, image_path: str) -> Dict:
        """Run inference on a single image."""
        img_cropped, img_resized, original_img = self._prepare_image(image_path)

        img_cropped = img_cropped.to(self.device)
        img_resized = img_resized.to(self.device)

        output = self.model(img_cropped, img_resized)
        score = output.item()

        return {
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'quality_score': score,
            'original_image': original_img,
            'type': 'image'
        }

    @torch.no_grad()
    def predict_video(self, video_path: str, show_progress: bool = True) -> Dict:
        """
        Run inference on a video file.

        This extracts frames, predicts quality for each, aggregates to per-second
        averages, and returns comprehensive time-series data suitable for visualization.
        """
        self.logger.info(f"🎬 Processing video: {Path(video_path).name}")

        # Extract frames
        frames, timestamps, metadata = self.frame_extractor.extract_frames(video_path)

        self.logger.info(f"   Extracted {len(frames)} frames from {metadata['duration']:.1f}s video")

        # Process each frame
        frame_results = []
        iterator = tqdm(frames, desc="Analyzing frames", disable=not show_progress, ncols=80)

        for frame, timestamp in zip(iterator, timestamps):
            img_cropped, img_resized = self._prepare_frame(frame)

            img_cropped = img_cropped.to(self.device)
            img_resized = img_resized.to(self.device)

            output = self.model(img_cropped, img_resized)
            score = output.item()

            frame_results.append({
                'timestamp': timestamp,
                'quality_score': score,
                'frame': frame  # Store for visualization if needed
            })

        # Aggregate frame results by second
        per_second_results = aggregate_scores_by_second(frame_results)

        # Calculate statistics from per-second data for better representation
        second_scores = [r['quality_score'] for r in per_second_results]

        return {
            'video_path': video_path,
            'video_name': Path(video_path).name,
            'type': 'video',
            'metadata': metadata,
            'frame_results': frame_results,
            'per_second_results': per_second_results,  # NEW: Added per-second aggregation
            'num_frames_analyzed': len(frame_results),
            'num_seconds': len(per_second_results),  # NEW: Number of unique seconds
            'average_quality': np.mean(second_scores),
            'min_quality': np.min(second_scores),
            'max_quality': np.max(second_scores),
            'std_quality': np.std(second_scores)
        }

    def predict(self, input_path: str, show_progress: bool = True) -> List[Dict]:
        """
        Main prediction interface - handles images, videos, and directories.
        """
        input_path = Path(input_path)

        # Handle single file
        if input_path.is_file():
            ext = input_path.suffix.lower()

            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                return [self.predict_single_image(str(input_path))]
            elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                return [self.predict_video(str(input_path), show_progress)]
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # Handle directory
        elif input_path.is_dir():
            results = []

            # Find all supported files
            image_paths = []
            video_paths = []

            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.extend(input_path.glob(f"*{ext}"))

            for ext in SUPPORTED_VIDEO_EXTENSIONS:
                video_paths.extend(input_path.glob(f"*{ext}"))

            image_paths = sorted([str(p) for p in image_paths])
            video_paths = sorted([str(p) for p in video_paths])

            if not image_paths and not video_paths:
                raise ValueError(f"No supported files found in {input_path}")

            self.logger.info(f"📁 Found {len(image_paths)} images and {len(video_paths)} videos")

            # Process images
            if image_paths:
                for img_path in tqdm(image_paths, desc="Processing images", ncols=80):
                    try:
                        result = self.predict_single_image(img_path)
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed: {img_path}")

            # Process videos
            if video_paths:
                for vid_path in video_paths:
                    try:
                        result = self.predict_video(vid_path, show_progress)
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed: {vid_path}")

            return results
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def visualize_video_results(self, video_result: Dict, output_dir: str = 'inference_results',
                                granularity: str = 'second') -> None:
        """
        Create time-series visualization for video quality predictions.

        This generates a line plot showing how quality varies across the video timeline.
        You can choose between frame-level and second-level granularity.

        Args:
            video_result: Dictionary containing video analysis results
            output_dir: Directory to save visualizations
            granularity: Visualization granularity - 'frame' for frame-by-frame,
                        'second' for per-second averages, or 'both' for dual plot
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if granularity == 'both':
            # Create side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

            # Frame-level plot (left)
            frame_results = video_result['frame_results']
            frame_timestamps = [r['timestamp'] for r in frame_results]
            frame_scores = [r['quality_score'] for r in frame_results]

            ax1.plot(frame_timestamps, frame_scores, linewidth=1.5, color='#2E86AB',
                     marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1,
                     alpha=0.7, label='Frame-level')
            ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
            ax1.set_title('Frame-Level Quality', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(0, 1)
            # Second-level plot (right)
            second_results = video_result['per_second_results']
            second_timestamps = [r['timestamp'] for r in second_results]
            second_scores = [r['quality_score'] for r in second_results]

            ax2.plot(second_timestamps, second_scores, linewidth=2.5, color='#A23B72',
                     marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                     label='Per-second average')

            # Add error bars showing variability within each second
            if second_results and 'std_score' in second_results[0]:
                stds = [r['std_score'] for r in second_results]
                ax2.fill_between(second_timestamps,
                                 np.array(second_scores) - np.array(stds),
                                 np.array(second_scores) + np.array(stds),
                                 alpha=0.2, color='#A23B72')

            ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
            ax2.set_title('Per-Second Averaged Quality', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, 1)
            # Add overall average line to both
            avg_score = video_result['average_quality']
            ax1.axhline(y=avg_score, color='#F18F01', linestyle='--',
                        linewidth=1.5, alpha=0.7, label=f'Overall avg: {avg_score:.2f}')
            ax2.axhline(y=avg_score, color='#F18F01', linestyle='--',
                        linewidth=1.5, alpha=0.7, label=f'Overall avg: {avg_score:.2f}')

            ax1.legend(loc='best', framealpha=0.9)
            ax2.legend(loc='best', framealpha=0.9)

            plt.suptitle(f"Video Quality Analysis: {video_result['video_name']}, {self.task}-oriented MIQA",
                         fontsize=14, fontweight='bold', y=1.02)

            suffix = 'comparison'

        else:
            # Single plot based on selected granularity
            plt.figure(figsize=(14, 6))

            if granularity == 'frame':
                frame_results = video_result['frame_results']
                timestamps = [r['timestamp'] for r in frame_results]
                scores = [r['quality_score'] for r in frame_results]
                plot_color = '#2E86AB'
                plot_label = 'Frame-level quality'
                title_suffix = '(Frame-Level)'
                suffix = 'frame'
                marker_size = 4
            else:  # second
                second_results = video_result['per_second_results']
                timestamps = [r['timestamp'] for r in second_results]
                scores = [r['quality_score'] for r in second_results]
                plot_color = '#A23B72'
                plot_label = 'Per-second average'
                title_suffix = '(Per-Second Average)'
                suffix = 'second'
                marker_size = 6

            # Main quality plot
            plt.plot(timestamps, scores, linewidth=2, color=plot_color, marker='o',
                     markersize=marker_size, markerfacecolor='white', markeredgewidth=1.5,
                     label=plot_label)

            # Add shaded region for second-level showing variability
            if granularity == 'second' and second_results and 'std_score' in second_results[0]:
                stds = [r['std_score'] for r in second_results]
                plt.fill_between(timestamps,
                             np.array(scores) - np.array(stds),
                             np.array(scores) + np.array(stds),
                             alpha=0.2, color=plot_color)

            # Add average line
            avg_score = video_result['average_quality']
            plt.axhline(y=avg_score, color='#F18F01', linestyle='--',
                        linewidth=1.5, label=f'Average: {avg_score:.2f}')

            # Styling
            plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            plt.ylabel('Quality Score', fontsize=12, fontweight='bold')
            plt.title(f"Video Quality Analysis: {video_result['video_name']} {title_suffix}",
                      fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='best', framealpha=0.9)

        # Add statistics box
        if granularity == 'both':
            stats_text = (
                f"Duration: {video_result['metadata']['duration']:.1f}s\n"
                f"Frames: {video_result['num_frames_analyzed']} | "
                f"Seconds: {video_result['num_seconds']}\n"
                f"Score Range: [{video_result['min_quality']:.2f}, {video_result['max_quality']:.2f}]\n"
                f"Std Dev: {video_result['std_quality']:.2f}"
            )
            # Add to the right subplot
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            stats_text = (
                f"Duration: {video_result['metadata']['duration']:.1f}s\n"
                f"Frames Analyzed: {video_result['num_frames_analyzed']}\n"
                f"Unique Seconds: {video_result['num_seconds']}\n"
                f"Score Range: [{video_result['min_quality']:.2f}, {video_result['max_quality']:.2f}]\n"
                f"Std Dev: {video_result['std_quality']:.2f}"
            )
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        plt.tight_layout()

        # Save figure
        output_file = output_path / f"{Path(video_result['video_name']).stem}_{self.metric_type}_quality_{suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"   Saved visualization: {output_file.name}")

    def visualize_results(self, results: List[Dict], output_dir: str = 'inference_results',
                          video_granularity: str = 'second') -> None:
        """
        Create visualizations for all results (images and videos).

        Args:
            results: List of prediction results
            output_dir: Directory to save visualizations
            video_granularity: For videos - 'frame', 'second', or 'both'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"\n🎨 Creating visualizations (granularity: {video_granularity})...")

        for result in results:
            if result['type'] == 'video':
                self.visualize_video_results(result, output_dir, video_granularity)
            elif result['type'] == 'image' and result.get('quality_score') is not None:
                # Use original image visualization logic
                img = result['original_image'].copy()
                draw = ImageDraw.Draw(img)

                score = result['quality_score']
                score_text = f"Quality: {score:.3f}"

                # Simple color coding (adjust range as needed)
                # norm_score = score / 100.0
                norm_score = max(0, score)

                if norm_score < 0.5:
                    r, g, b = 255, int(255 * norm_score * 2), 0
                else:
                    r, g, b = int(255 * (2 - norm_score * 2)), 255, 0

                color = (r, g, b)

                box_coords = [10, 10, 260, 60]
                draw.rectangle(box_coords, fill=color)

                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()

                draw.text((15, 20), score_text, fill='black', font=font)

                output_file = output_path / f"annotated_{result['image_name']}"
                img.save(output_file)

        self.logger.info(f"✓ Visualizations saved to: {output_dir}/")

    def save_results(self, results: List[Dict], output_path: str = 'predictions.json') -> None:
        """Save prediction results to JSON file."""
        # Clean results for JSON serialization
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items()
                       if k not in ['original_image', 'frame']}

            # For video results, remove frame data but keep scores
            if clean_r.get('type') == 'video' and 'frame_results' in clean_r:
                clean_r['frame_results'] = [
                    {k: v for k, v in fr.items() if k != 'frame'}
                    for fr in clean_r['frame_results']
                ]

            clean_results.append(clean_r)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'task': self.task,
                    'model': self.model_name,
                    'metric_type': self.metric_type,
                    'timestamp': datetime.now().isoformat(),
                    'total_files': len(clean_results)
                },
                'predictions': clean_results
            }, f, indent=2)

        self.logger.info(f"💾 Results saved to: {output_path}")

    def print_summary(self, results: List[Dict]) -> None:
        """Print formatted summary of prediction results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREDICTION SUMMARY")
        self.logger.info("=" * 80)

        image_results = [r for r in results if r.get('type') == 'image']
        video_results = [r for r in results if r.get('type') == 'video']

        if image_results:
            valid_images = [r for r in image_results if r.get('quality_score') is not None]
            if valid_images:
                scores = [r['quality_score'] for r in valid_images]
                self.logger.info(f"\n📸 Image Analysis ({len(valid_images)} images)")
                self.logger.info(f"   Average quality: {np.mean(scores):.2f}")
                self.logger.info(f"   Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")

        if video_results:
            self.logger.info(f"\n🎬 Video Analysis ({len(video_results)} videos)")
            for vr in video_results:
                self.logger.info(f"\n   {vr['video_name']}:")
                self.logger.info(f"      Duration: {vr['metadata']['duration']:.1f}s")
                self.logger.info(f"      Frames analyzed: {vr['num_frames_analyzed']}")
                self.logger.info(f"      Unique seconds: {vr['num_seconds']}")
                self.logger.info(f"      Average quality (per-second): {vr['average_quality']:.2f}")
                self.logger.info(f"      Quality range: [{vr['min_quality']:.2f}, {vr['max_quality']:.2f}]")
                self.logger.info(f"      Variability (std): {vr['std_quality']:.2f}")

        self.logger.info("\n" + "=" * 80 + "\n")


def main():
    """Command-line interface for MIQA inference."""
    parser = argparse.ArgumentParser(
        description='MIQA: Machine-centric Image and Video Quality Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze video with per-second visualization
  python video_analytics_inference.py --input video.mp4 --task cls --visualize --viz-granularity second

  # Analyze video with both frame and second visualizations
  python video_analytics_inference.py --input video.mp4 --task cls --visualize --viz-granularity both

  # Process directory with frame-level visualization
  python video_analytics_inference.py --input ./assets/demo_video --task det --video-frames 120 --visualize --viz-granularity second --metric-type consistency
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image/video or directory')
    parser.add_argument('--task', type=str, required=True,
                        choices=['cls', 'det', 'ins'],
                        help='Task type')
    parser.add_argument('--model', type=str, default='ra_miqa',
                        help='Model architecture: ra_miqa ONLY!')
    parser.add_argument('--metric-type', type=str, default='composite',
                        choices=['composite', 'consistency', 'accuracy'],
                        help='Training metric type')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--video-frames', type=int, default=50,
                        help='Target number of frames to sample from videos')
    parser.add_argument('--save-results', action='store_true',
                        help='Save prediction results to file')
    parser.add_argument('--output-file', type=str, default='predictions.json',
                        help='Output file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--viz-dir', type=str, default='inference_results',
                        help='Directory for visualizations')
    parser.add_argument('--viz-granularity', type=str, default='second',
                        choices=['frame', 'second', 'both'],
                        help='Visualization granularity for videos: frame-level, per-second, or both')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')

    args = parser.parse_args()

    try:
        miqa = MIQAInference(
            task=args.task,
            model_name=args.model,
            metric_type=args.metric_type,
            device=args.device,
            video_target_frames=args.video_frames
        )

        results = miqa.predict(args.input, show_progress=not args.no_progress)
        miqa.print_summary(results)

        if args.save_results:
            miqa.save_results(results, args.output_file)

        if args.visualize:
            miqa.visualize_results(results, args.viz_dir, video_granularity=args.viz_granularity)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
# import os
# import sys
# import torch
# import argparse
# import logging
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from collections import OrderedDict
# import json
# from datetime import datetime
# from tqdm import tqdm
# import numpy as np
# import cv2
# 
# # Image processing imports
# from PIL import Image, ImageDraw, ImageFont
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import matplotlib
# 
# matplotlib.use('Agg')  # Use non-interactive backend
# 
# # Import your existing model components
# from models.MIQA_base import get_torch_model, get_timm_model
# from models.RA_MIQA import RegionVisionTransformer
# from utils.download_utils import ensure_checkpoint
# 
# 
# # Model configuration
# MODEL_CONFIGS = {
#     'cls': {
#         'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
#         'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
#         'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
#         'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
#     },
#     'det': {
#         'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
#         'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
#         'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
#         'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
#     },
#     'ins': {
#         'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
#         'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
#         'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
#         'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
#     }
# }
# 
# # Supported file extensions
# SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.JPEG', '.png', '.bmp', '.tiff', '.tif'}
# SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
# 
# 
# class VideoFrameExtractor:
#     """
#     Extracts and samples frames from video files intelligently.
# 
#     This class handles different sampling strategies to balance between
#     thoroughness and computational efficiency.
#     """
# 
#     def __init__(self, sampling_strategy: str = 'uniform',
#                  target_frames: int = 30,
#                  fps_sample: Optional[float] = None):
#         """
#         Initialize the frame extractor.
# 
#         Args:
#             sampling_strategy: How to sample frames - 'uniform', 'fps', or 'keyframe'
#             target_frames: Target number of frames to extract (for uniform sampling)
#             fps_sample: Sample rate in frames per second (for fps sampling)
#         """
#         self.sampling_strategy = sampling_strategy
#         self.target_frames = target_frames
#         self.fps_sample = fps_sample
# 
#     def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float], Dict]:
#         """
#         Extract frames from video based on sampling strategy.
# 
#         Returns:
#             Tuple of (frames_list, timestamps_list, video_metadata)
#         """
#         cap = cv2.VideoCapture(video_path)
# 
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video file: {video_path}")
# 
#         # Get video properties
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         duration = total_frames / fps if fps > 0 else 0
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 
#         metadata = {
#             'total_frames': total_frames,
#             'fps': fps,
#             'duration': duration,
#             'width': width,
#             'height': height
#         }
# 
#         # Determine which frames to sample
#         frame_indices = self._get_sample_indices(total_frames, fps)
# 
#         frames = []
#         timestamps = []
# 
#         for idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
# 
#             if ret:
#                 # Convert BGR to RGB (OpenCV uses BGR)
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
#                 # Calculate timestamp in seconds
#                 timestamp = idx / fps if fps > 0 else idx
#                 timestamps.append(timestamp)
# 
#         cap.release()
# 
#         return frames, timestamps, metadata
# 
#     def _get_sample_indices(self, total_frames: int, fps: float) -> List[int]:
#         """
#         Determine which frame indices to sample based on strategy.
#         """
#         if self.sampling_strategy == 'uniform':
#             # Sample frames uniformly across the video
#             if total_frames <= self.target_frames:
#                 return list(range(total_frames))
#             else:
#                 # Calculate step size to get approximately target_frames
#                 step = total_frames / self.target_frames
#                 indices = [int(i * step) for i in range(self.target_frames)]
#                 return indices
# 
#         elif self.sampling_strategy == 'fps':
#             # Sample at a specific frame rate
#             if self.fps_sample is None:
#                 raise ValueError("fps_sample must be specified for fps sampling strategy")
# 
#             frame_interval = max(1, int(fps / self.fps_sample))
#             indices = list(range(0, total_frames, frame_interval))
#             return indices
# 
#         else:
#             raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
# 
# 
# class MIQAInference:
#     """
#     Inference wrapper for MIQA models supporting both images and videos.
#     """
# 
#     def __init__(self, task: str, model_name: str = 'ra_miqa',
#                  metric_type: str = 'composite', device: Optional[str] = None,
#                  video_sampling: str = 'uniform', video_target_frames: int = 30):
#         """
#         Initialize the MIQA inference system.
# 
#         Args:
#             task: Task type - 'cls', 'det', or 'ins'
#             model_name: Model architecture to use
#             metric_type: Training objective - 'composite', 'consistency', or 'accuracy'
#             device: Device to run inference on
#             video_sampling: Frame sampling strategy for videos
#             video_target_frames: Target number of frames to extract from videos
#         """
#         self.task = task.lower()
#         self.model_name = model_name
#         self.metric_type = metric_type
#         self.video_target_frames = video_target_frames
# 
#         # Setup logging
#         self.logger = self._setup_logger()
# 
#         # Determine device
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = torch.device(device)
# 
#         self.logger.info(f"🚀 Initializing MIQA Inference System")
#         self.logger.info(f"   Task: {self.task.upper()}")
#         self.logger.info(f"   Model: {self.model_name}")
#         self.logger.info(f"   Device: {self.device}")
# 
#         # Validate configuration
#         self._validate_config()
# 
#         # Initialize model
#         self.model = self._load_model()
# 
#         # Setup image preprocessing
#         self.transforms1, self.transforms2 = self._get_transforms()
# 
#         # Initialize video frame extractor
#         self.frame_extractor = VideoFrameExtractor(
#             sampling_strategy=video_sampling,
#             target_frames=video_target_frames
#         )
# 
#         self.logger.info("✅ System ready for inference\n")
# 
#     def _setup_logger(self) -> logging.Logger:
#         """Configure logging."""
#         logger = logging.getLogger('MIQA_Inference')
#         logger.setLevel(logging.INFO)
#         logger.handlers = []
# 
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setLevel(logging.INFO)
#         console_formatter = logging.Formatter('%(message)s')
#         console_handler.setFormatter(console_formatter)
#         logger.addHandler(console_handler)
# 
#         return logger
# 
#     def _validate_config(self) -> None:
#         """Validate configuration."""
#         if self.task not in MODEL_CONFIGS:
#             raise ValueError(f"Invalid task '{self.task}'")
#         if self.model_name not in MODEL_CONFIGS[self.task]:
#             raise ValueError(f"Model '{self.model_name}' not available for task '{self.task}'")
# 
#     def _get_checkpoint_path(self) -> str:
#         """Generate checkpoint path."""
#         base_dir = Path('models') / 'checkpoints' / f'miqa_{self.task}'
#         base_dir.mkdir(parents=True, exist_ok=True)
#         filename = f"{self.model_name}_mse_miqa_{self.task}_{self.metric_type}_best.pth.tar"
#         return str(base_dir / filename)
# 
#     def _download_weights(self, checkpoint_path: str) -> bool:
#         """Download model weights if needed."""
#         if os.path.exists(checkpoint_path):
#             self.logger.info(f"✓ Found cached model weights")
#             return True
# 
#         self.logger.info(f"⏬ Downloading model weights...")
#         download_urls = MODEL_CONFIGS[self.task][self.model_name]
# 
#         if download_urls[0] == f"https://drive.google.com/uc?id={file_id}":
#             self.logger.error(f"❌ Download URLs not configured")
#             return False
# 
#         return ensure_checkpoint(checkpoint_path, download_urls)
# 
#     def _create_model(self) -> torch.nn.Module:
#         """Create model architecture."""
#         if self.model_name == 'ra_miqa':
#             model = RegionVisionTransformer(
#                 base_model_name='vit_small_patch16_224',
#                 pretrained=False,
#                 mmseg_config_path='models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
#                 checkpoint_path='models/checkpoints/sere_finetuned_vit_small_ep100.pth'
#             )
#         else:
#             try:
#                 model = get_torch_model(model_name=self.model_name, pretrained=False, num_classes=1)
#             except Exception:
#                 model = get_timm_model(model_name=self.model_name, pretrained=False, num_classes=1)
#         return model
# 
#     def _load_model(self) -> torch.nn.Module:
#         """Load model with weights."""
#         checkpoint_path = self._get_checkpoint_path()
# 
#         if not self._download_weights(checkpoint_path):
#             raise RuntimeError("Cannot proceed without model weights")
# 
#         self.logger.info("🔧 Loading model...")
#         model = self._create_model()
# 
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
#         state_dict = checkpoint.get('state_dict', checkpoint)
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k.replace('module.', '') if k.startswith('module.') else k
#             new_state_dict[name] = v
# 
#         model.load_state_dict(new_state_dict, strict=True)
#         model = model.to(self.device)
#         model.eval()
# 
#         self.logger.info("✓ Model loaded successfully")
#         return model
# 
#     def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
#         """Get image preprocessing transforms."""
#         transforms_list1 = [
#             transforms.Resize(288),
#             transforms.CenterCrop(size=224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ]
#         transform_list_2 = [
#             transforms.Resize(288),
#             transforms.CenterCrop((288, 288)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ]
#         return transforms.Compose(transforms_list1), transforms.Compose(transform_list_2)
# 
#     def _prepare_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Preprocess a video frame for model input.
# 
#         Args:
#             frame: Numpy array in RGB format
# 
#         Returns:
#             Tuple of (cropped_tensor, resized_tensor)
#         """
#         # Convert numpy array to PIL Image
#         img = Image.fromarray(frame)
# 
#         # Apply transforms
#         img1 = self.transforms1(img).unsqueeze(0)
#         img2 = self.transforms2(img).unsqueeze(0)
# 
#         return img1, img2
# 
#     def _prepare_image(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
#         """Load and preprocess an image file."""
#         img = Image.open(image_path).convert('RGB')
#         img1 = self.transforms1(img).unsqueeze(0)
#         img2 = self.transforms2(img).unsqueeze(0)
#         return img1, img2, img
# 
#     @torch.no_grad()
#     def predict_single_image(self, image_path: str) -> Dict:
#         """Run inference on a single image."""
#         img_cropped, img_resized, original_img = self._prepare_image(image_path)
# 
#         img_cropped = img_cropped.to(self.device)
#         img_resized = img_resized.to(self.device)
# 
#         output = self.model(img_cropped, img_resized)
#         score = output.item()
# 
#         return {
#             'image_path': image_path,
#             'image_name': Path(image_path).name,
#             'quality_score': score,
#             'original_image': original_img,
#             'type': 'image'
#         }
# 
#     # @torch.no_grad()
#     # def predict_video(self, video_path: str, show_progress: bool = True) -> Dict:
#     #     """
#     #     Run inference on a video file.
#     #
#     #     This extracts frames, predicts quality for each, and returns
#     #     time-series data suitable for visualization.
#     #     """
#     #     self.logger.info(f"🎬 Processing video: {Path(video_path).name}")
#     #
#     #     # Extract frames
#     #     frames, timestamps, metadata = self.frame_extractor.extract_frames(video_path)
#     #
#     #     self.logger.info(f"   Extracted {len(frames)} frames from {metadata['duration']:.1f}s video")
#     #
#     #     # Process each frame
#     #     frame_results = []
#     #     iterator = tqdm(frames, desc="Analyzing frames", disable=not show_progress, ncols=80)
#     #
#     #     for frame, timestamp in zip(iterator, timestamps):
#     #         img_cropped, img_resized = self._prepare_frame(frame)
#     #
#     #         img_cropped = img_cropped.to(self.device)
#     #         img_resized = img_resized.to(self.device)
#     #
#     #         output = self.model(img_cropped, img_resized)
#     #         score = output.item()
#     #
#     #         frame_results.append({
#     #             'timestamp': timestamp,
#     #             'quality_score': score,
#     #             'frame': frame  # Store for visualization if needed
#     #         })
#     #
#     #     return {
#     #         'video_path': video_path,
#     #         'video_name': Path(video_path).name,
#     #         'type': 'video',
#     #         'metadata': metadata,
#     #         'frame_results': frame_results,
#     #         'num_frames_analyzed': len(frame_results),
#     #         'average_quality': np.mean([r['quality_score'] for r in frame_results]),
#     #         'min_quality': np.min([r['quality_score'] for r in frame_results]),
#     #         'max_quality': np.max([r['quality_score'] for r in frame_results]),
#     #         'std_quality': np.std([r['quality_score'] for r in frame_results])
#     #     }
#     def _aggregate_by_second(self, frame_results: List[Dict]) -> List[Dict]:
#         """
#         Aggregate frame-level predictions into per-second statistics.
# 
#         This groups all frames within each second interval and computes
#         average, min, max, and standard deviation for that second.
# 
#         Args:
#             frame_results: List of frame-level prediction dictionaries
# 
#         Returns:
#             List of per-second aggregated statistics
#         """
#         if not frame_results:
#             return []
# 
#         # Group frames by their second (floor of timestamp)
#         second_groups = {}
#         for result in frame_results:
#             second = int(result['timestamp'])  # Floor to get the second bucket
#             if second not in second_groups:
#                 second_groups[second] = []
#             second_groups[second].append(result['quality_score'])
# 
#         # Compute statistics for each second
#         second_results = []
#         for second in sorted(second_groups.keys()):
#             scores = second_groups[second]
#             second_results.append({
#                 'second': second,
#                 'timestamp': float(second),  # Use second as timestamp for consistency
#                 'average_quality': np.mean(scores),
#                 'min_quality': np.min(scores),
#                 'max_quality': np.max(scores),
#                 'std_quality': np.std(scores),
#                 'num_frames': len(scores)
#             })
# 
#         return second_results
# 
#     @torch.no_grad()
#     def predict_video(self, video_path: str, show_progress: bool = True) -> Dict:
#         """
#         Run inference on a video file.
# 
#         This extracts frames, predicts quality for each, and returns
#         time-series data suitable for visualization. Results include both
#         frame-level and per-second aggregated statistics.
#         """
#         self.logger.info(f"🎬 Processing video: {Path(video_path).name}")
# 
#         # Extract frames
#         frames, timestamps, metadata = self.frame_extractor.extract_frames(video_path)
# 
#         self.logger.info(f"   Extracted {len(frames)} frames from {metadata['duration']:.1f}s video")
# 
#         # Process each frame
#         frame_results = []
#         iterator = tqdm(frames, desc="Analyzing frames", disable=not show_progress, ncols=80)
# 
#         for frame, timestamp in zip(iterator, timestamps):
#             img_cropped, img_resized = self._prepare_frame(frame)
# 
#             img_cropped = img_cropped.to(self.device)
#             img_resized = img_resized.to(self.device)
# 
#             output = self.model(img_cropped, img_resized)
#             score = output.item()
# 
#             frame_results.append({
#                 'timestamp': timestamp,
#                 'quality_score': score,
#                 'frame': frame  # Store for visualization if needed
#             })
# 
#         # Aggregate frame results by second
#         second_results = self._aggregate_by_second(frame_results)
# 
#         return {
#             'video_path': video_path,
#             'video_name': Path(video_path).name,
#             'type': 'video',
#             'metadata': metadata,
#             'frame_results': frame_results,
#             'second_results': second_results,
#             'num_frames_analyzed': len(frame_results),
#             'num_seconds_analyzed': len(second_results),
#             'average_quality': np.mean([r['quality_score'] for r in frame_results]),
#             'min_quality': np.min([r['quality_score'] for r in frame_results]),
#             'max_quality': np.max([r['quality_score'] for r in frame_results]),
#             'std_quality': np.std([r['quality_score'] for r in frame_results])
#         }
# 
#     def predict(self, input_path: str, show_progress: bool = True) -> List[Dict]:
#         """
#         Main prediction interface - handles images, videos, and directories.
#         """
#         input_path = Path(input_path)
# 
#         # Handle single file
#         if input_path.is_file():
#             ext = input_path.suffix.lower()
# 
#             if ext in SUPPORTED_IMAGE_EXTENSIONS:
#                 return [self.predict_single_image(str(input_path))]
#             elif ext in SUPPORTED_VIDEO_EXTENSIONS:
#                 return [self.predict_video(str(input_path), show_progress)]
#             else:
#                 raise ValueError(f"Unsupported file extension: {ext}")
# 
#         # Handle directory
#         elif input_path.is_dir():
#             results = []
# 
#             # Find all supported files
#             image_paths = []
#             video_paths = []
# 
#             for ext in SUPPORTED_IMAGE_EXTENSIONS:
#                 image_paths.extend(input_path.glob(f"*{ext}"))
# 
#             for ext in SUPPORTED_VIDEO_EXTENSIONS:
#                 video_paths.extend(input_path.glob(f"*{ext}"))
# 
#             image_paths = sorted([str(p) for p in image_paths])
#             video_paths = sorted([str(p) for p in video_paths])
# 
#             if not image_paths and not video_paths:
#                 raise ValueError(f"No supported files found in {input_path}")
# 
#             self.logger.info(f"📁 Found {len(image_paths)} images and {len(video_paths)} videos")
# 
#             # Process images
#             if image_paths:
#                 for img_path in tqdm(image_paths, desc="Processing images", ncols=80):
#                     try:
#                         result = self.predict_single_image(img_path)
#                         results.append(result)
#                     except Exception as e:
#                         self.logger.warning(f"⚠️  Failed: {img_path}")
# 
#             # Process videos
#             if video_paths:
#                 for vid_path in video_paths:
#                     try:
#                         result = self.predict_video(vid_path, show_progress)
#                         results.append(result)
#                     except Exception as e:
#                         self.logger.warning(f"⚠️  Failed: {vid_path}")
# 
#             return results
#         else:
#             raise ValueError(f"Input path does not exist: {input_path}")
# 
#     def visualize_video_results(self, video_result: Dict, output_dir: str = 'inference_results') -> None:
#         """
#         Create time-series visualization for video quality predictions.
# 
#         This generates a line plot showing how quality varies across the video timeline.
#         """
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
# 
#         frame_results = video_result['frame_results']
#         timestamps = [r['timestamp'] for r in frame_results]
#         scores = [r['quality_score'] for r in frame_results]
# 
#         # Create figure with professional styling
#         plt.figure(figsize=(14, 6))
# 
#         # Plot quality score over time
#         plt.plot(timestamps, scores, linewidth=2, color='#2E86AB', marker='o',
#                  markersize=4, markerfacecolor='white', markeredgewidth=1.5)
# 
#         # Add average line
#         avg_score = video_result['average_quality']
#         plt.axhline(y=avg_score, color='#A23B72', linestyle='--',
#                     linewidth=1.5, label=f'Average: {avg_score:.2f}')
# 
#         # Styling
#         plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
#         plt.ylabel('Quality Score', fontsize=12, fontweight='bold')
#         plt.title(f"Video Quality Analysis: {video_result['video_name']}",
#                   fontsize=14, fontweight='bold', pad=20)
#         plt.grid(True, alpha=0.3, linestyle='--')
#         plt.legend(loc='best', framealpha=0.9)
# 
#         # Add statistics box
#         stats_text = (
#             f"Duration: {video_result['metadata']['duration']:.1f}s\n"
#             f"Frames Analyzed: {video_result['num_frames_analyzed']}\n"
#             f"Score Range: [{video_result['min_quality']:.2f}, {video_result['max_quality']:.2f}]\n"
#             f"Std Dev: {video_result['std_quality']:.2f}"
#         )
#         plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
#                  fontsize=9, verticalalignment='top',
#                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# 
#         plt.tight_layout()
# 
#         # Save figure
#         output_file = output_path / f"{Path(video_result['video_name']).stem}_quality_analysis.png"
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         plt.close()
# 
#         self.logger.info(f"   Saved video analysis: {output_file.name}")
# 
#     def visualize_results(self, results: List[Dict], output_dir: str = 'inference_results') -> None:
#         """
#         Create visualizations for all results (images and videos).
#         """
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
# 
#         self.logger.info(f"\n🎨 Creating visualizations...")
# 
#         for result in results:
#             if result['type'] == 'video':
#                 self.visualize_video_results(result, output_dir)
#             elif result['type'] == 'image' and result.get('quality_score') is not None:
#                 # Use original image visualization logic
#                 img = result['original_image'].copy()
#                 draw = ImageDraw.Draw(img)
# 
#                 score = result['quality_score']
#                 score_text = f"Quality: {score:.2f}"
# 
#                 # Simple color coding (adjust range as needed)
#                 norm_score = score / 100.0
#                 norm_score = max(0, min(1, norm_score))
# 
#                 if norm_score < 0.5:
#                     r, g, b = 255, int(255 * norm_score * 2), 0
#                 else:
#                     r, g, b = int(255 * (2 - norm_score * 2)), 255, 0
# 
#                 color = (r, g, b)
# 
#                 box_coords = [10, 10, 260, 60]
#                 draw.rectangle(box_coords, fill=color)
# 
#                 try:
#                     font = ImageFont.truetype("arial.ttf", 24)
#                 except:
#                     font = ImageFont.load_default()
# 
#                 draw.text((15, 20), score_text, fill='black', font=font)
# 
#                 output_file = output_path / f"annotated_{result['image_name']}"
#                 img.save(output_file)
# 
#         self.logger.info(f"✓ Visualizations saved to: {output_dir}/")
# 
#     def save_results(self, results: List[Dict], output_path: str = 'predictions.json') -> None:
#         """Save prediction results to JSON file."""
#         # Clean results for JSON serialization
#         clean_results = []
#         for r in results:
#             clean_r = {k: v for k, v in r.items()
#                        if k not in ['original_image', 'frame']}
# 
#             # For video results, remove frame data but keep scores
#             if clean_r.get('type') == 'video' and 'frame_results' in clean_r:
#                 clean_r['frame_results'] = [
#                     {k: v for k, v in fr.items() if k != 'frame'}
#                     for fr in clean_r['frame_results']
#                 ]
# 
#             clean_results.append(clean_r)
# 
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#         with open(output_path, 'w') as f:
#             json.dump({
#                 'metadata': {
#                     'task': self.task,
#                     'model': self.model_name,
#                     'metric_type': self.metric_type,
#                     'timestamp': datetime.now().isoformat(),
#                     'total_files': len(clean_results)
#                 },
#                 'predictions': clean_results
#             }, f, indent=2)
# 
#         self.logger.info(f"💾 Results saved to: {output_path}")
# 
#     def print_summary(self, results: List[Dict]) -> None:
#         """Print formatted summary of prediction results."""
#         self.logger.info("\n" + "=" * 80)
#         self.logger.info("PREDICTION SUMMARY")
#         self.logger.info("=" * 80)
# 
#         image_results = [r for r in results if r.get('type') == 'image']
#         video_results = [r for r in results if r.get('type') == 'video']
# 
#         if image_results:
#             valid_images = [r for r in image_results if r.get('quality_score') is not None]
#             if valid_images:
#                 scores = [r['quality_score'] for r in valid_images]
#                 self.logger.info(f"\n📸 Image Analysis ({len(valid_images)} images)")
#                 self.logger.info(f"   Average quality: {np.mean(scores):.2f}")
#                 self.logger.info(f"   Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
# 
#         if video_results:
#             self.logger.info(f"\n🎬 Video Analysis ({len(video_results)} videos)")
#             for vr in video_results:
#                 self.logger.info(f"\n   {vr['video_name']}:")
#                 self.logger.info(f"      Duration: {vr['metadata']['duration']:.1f}s")
#                 self.logger.info(f"      Frames analyzed: {vr['num_frames_analyzed']}")
#                 self.logger.info(f"      Average quality: {vr['average_quality']:.2f}")
#                 self.logger.info(f"      Quality range: [{vr['min_quality']:.2f}, {vr['max_quality']:.2f}]")
#                 self.logger.info(f"      Variability (std): {vr['std_quality']:.2f}")
# 
#         self.logger.info("\n" + "=" * 80 + "\n")
# 
# 
# def main():
#     """Command-line interface for MIQA inference."""
#     parser = argparse.ArgumentParser(
#         description='MIQA: Machine-centric Image and Video Quality Assessment',
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )
# 
#     parser.add_argument('--input', type=str, required=True,
#                         help='Path to input image/video or directory')
#     parser.add_argument('--task', type=str, required=True,
#                         choices=['cls', 'det', 'ins'],
#                         help='Task type')
#     parser.add_argument('--model', type=str, default='ra_miqa',
#                         help='Model architecture')
#     parser.add_argument('--metric-type', type=str, default='composite',
#                         choices=['composite', 'consistency', 'accuracy'],
#                         help='Training metric type')
#     parser.add_argument('--device', type=str, default=None,
#                         choices=['cuda', 'cpu'],
#                         help='Device to run on')
#     parser.add_argument('--video-frames', type=int, default=50,
#                         help='Target number of frames to sample from videos')
#     parser.add_argument('--save-results', action='store_true',
#                         help='Save prediction results to file')
#     parser.add_argument('--output-file', type=str, default='predictions.json',
#                         help='Output file path')
#     parser.add_argument('--visualize', action='store_true',
#                         help='Create visualizations')
#     parser.add_argument('--viz-dir', type=str, default='inference_results',
#                         help='Directory for visualizations')
#     parser.add_argument('--no-progress', action='store_true',
#                         help='Disable progress bar')
# 
#     args = parser.parse_args()
# 
#     # try:
#     miqa = MIQAInference(
#         task=args.task,
#         model_name=args.model,
#         metric_type=args.metric_type,
#         device=args.device,
#         video_target_frames=args.video_frames
#     )
# 
#     results = miqa.predict(args.input, show_progress=not args.no_progress)
#     miqa.print_summary(results)
# 
#     if args.save_results:
#         miqa.save_results(results, args.output_file)
# 
#     if args.visualize:
#         miqa.visualize_results(results, args.viz_dir)
# 
#     # except Exception as e:
#     #     print(f"\n❌ Error: {str(e)}", file=sys.stderr)
#     #     sys.exit(1)
# 
# 
# if __name__ == '__main__':
#     main()
# 
#     '''
#     python img_inference.py --input video.mp4 --task cls --visualize
# 
#     python img_inference.py --input ./media_folder/ --task det --video-frames 50 --save-results --visualize
#     '''
