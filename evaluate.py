import os
import torch
import pandas as pd
from collections import OrderedDict
import logging
import torchmetrics
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from models.MIQA_base import get_torch_model, get_timm_model
from data import build_dataloader
from train import AverageMeter
from configs.args_base import get_args
from models.RA_MIQA import RegionVisionTransformer
from utils.download_utils import ensure_checkpoint, ensure_checkpoint_with_gdown

# Model configuration - maps task types to available models and their download URLs
MODEL_DOWNLOAD_URLS = {
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


def get_checkpoint_path(model_name: str, train_dataset: str, metric_type: str = 'composite') -> str:
    """
    Generate the checkpoint path based on model configuration.

    This function creates the appropriate file path for storing or loading model checkpoints.
    The path structure organizes models by their training dataset type, and the filename
    includes the metric type used during training.

    Args:
        model_name: Name of the model architecture (e.g., 'resnet50', 'vit_small_patch16_224')
        train_dataset: Training dataset type - one of 'cls', 'det', or 'ins'
        metric_type: Training metric objective - one of 'composite', 'consistency', or 'accuracy'

    Returns:
        Full path to the checkpoint file
    """
    # Create the base directory structure for this training dataset type
    base_dir = Path('models') / 'checkpoints' / f'{metric_type}_metric'
    base_dir.mkdir(parents=True, exist_ok=True)

    filename = f"miqa_{model_name}_{train_dataset}_{metric_type}_metric.pth.tar"
    return str(base_dir / filename)


def get_available_models(train_dataset: str, metric_type: str) -> List[str]:
    """
    Get list of available models for a specific training dataset and metric type.

    This helper function is useful for validation and for providing helpful error messages
    when a user requests a model that isn't available for their chosen configuration.

    Args:
        train_dataset: Training dataset type (cls, det, ins)
        metric_type: Training metric objective (both, consistency, accuracy)

    Returns:
        List of available model names for this configuration
    """
    if metric_type in MODEL_DOWNLOAD_URLS:
        if train_dataset in MODEL_DOWNLOAD_URLS[metric_type]:
            return list(MODEL_DOWNLOAD_URLS[metric_type][train_dataset].keys())
    return []


def ensure_model_weights(model_name: str, train_dataset: str, metric_type: str,
                         logger: logging.Logger) -> Optional[str]:
    """
    Ensure model weights exist, download if necessary.

    This function implements a caching strategy: it first checks if the checkpoint already
    exists locally. If not, it attempts to download it from the configured URLs. This means
    the first run will download weights, but subsequent runs will be much faster.

    Args:
        model_name: Name of the model architecture
        train_dataset: Training dataset type (cls, det, or ins)
        metric_type: Training metric objective (both, consistency, or accuracy)
        logger: Logger instance for status messages

    Returns:
        Path to checkpoint if successful, None if weights cannot be obtained
    """
    # Generate the expected checkpoint path
    checkpoint_path = get_checkpoint_path(model_name, train_dataset, metric_type)

    # First, check if we already have this checkpoint cached locally
    if os.path.exists(checkpoint_path):
        logger.info(f"✓ Found existing checkpoint: {checkpoint_path}")
        return checkpoint_path

    # Checkpoint not found locally, so we need to download it
    logger.info(f"Checkpoint not found at {checkpoint_path}")

    # Verify this model configuration is supported
    if metric_type not in MODEL_DOWNLOAD_URLS:
        logger.error(f"✗ Metric type '{metric_type}' not recognized")
        logger.error(f"   Available metric types: {list(MODEL_DOWNLOAD_URLS.keys())}")
        return None

    if train_dataset not in MODEL_DOWNLOAD_URLS[metric_type]:
        logger.error(f"✗ Train dataset '{train_dataset}' not available for metric type '{metric_type}'")
        return None

    if model_name not in MODEL_DOWNLOAD_URLS[metric_type][train_dataset]:
        available_models = get_available_models(train_dataset, metric_type)
        logger.error(f"✗ Model '{model_name}' not available for {train_dataset}/{metric_type}")
        logger.error(f"   Available models: {available_models}")
        return None

    # Get the download URLs for this specific configuration
    download_urls = MODEL_DOWNLOAD_URLS[metric_type][train_dataset][model_name]

    # Attempt to download the checkpoint
    logger.info(f"Attempting to download checkpoint for {model_name} ({metric_type})...")

    if ensure_checkpoint_with_gdown(checkpoint_path, download_urls):
    # if ensure_checkpoint(checkpoint_path, download_urls):
        logger.info(f"✓ Successfully downloaded checkpoint")
        return checkpoint_path
    else:
        logger.error(f"✗ Failed to download checkpoint from all provided URLs")
        return None

def load_model_weights(model: torch.nn.Module, weights_path: str, args: argparse.Namespace,
                       logger: logging.Logger) -> bool:
    """
    Load model weights from checkpoint file.

    This function handles the actual loading of weights into the model, with proper error
    handling and support for different checkpoint formats (direct state dict or wrapped
    in a dictionary with metadata).

    Args:
        model: The model to load weights into
        weights_path: Path to the checkpoint file
        args: Command line arguments
        logger: Logger instance

    Returns:
        True if weights loaded successfully, False otherwise
    """
    if not os.path.isfile(weights_path):
        logger.error(f"✗ Checkpoint file not found: '{weights_path}'")
        return False

    logger.info(f"Loading checkpoint from '{weights_path}'")

    try:
        # Load checkpoint to CPU first to avoid GPU memory issues
        checkpoint = torch.load(weights_path, map_location="cpu")

        # Extract state dict - handle different checkpoint formats
        # Some checkpoints store weights directly, others wrap them in a 'state_dict' key
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove 'module.' prefix if present
        # This prefix is added when models are trained with DataParallel/DistributedDataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        # Load the processed weights into the model
        model.load_state_dict(new_state_dict)
        logger.info(f"✓ Successfully loaded checkpoint")

        # Log additional useful information from the checkpoint if available
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_srcc' in checkpoint:
            logger.info(f"  Best SRCC: {checkpoint['best_srcc']:.4f}")
        if 'metric_type' in checkpoint:
            logger.info(f"  Metric type: {checkpoint['metric_type']}")

        return True

    except Exception as e:
        logger.error(f"✗ Error loading checkpoint: {str(e)}")
        return False


def create_model(model_name: str, args: argparse.Namespace, logger: logging.Logger) -> torch.nn.Module:
    """
    Create model instance based on model name.

    This function handles the instantiation of different model architectures. It includes
    special handling for the RegionVisionTransformer (RA_MIQA) which has a different
    initialization process than standard vision models.

    Args:
        model_name: Name of the model architecture
        args: Command line arguments
        logger: Logger instance

    Returns:
        Initialized model (without loaded weights yet)
    """
    # Special handling for our custom RegionVisionTransformer architecture
    if model_name == 'ra_miqa':
        logger.info(f"Creating RA_MIQA Model")
        model = RegionVisionTransformer(
            base_model_name='vit_small_patch16_224',
            pretrained=True,
            mmseg_config_path='models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
            checkpoint_path='models/checkpoints/sere_finetuned_vit_small_ep100.pth'
        )
    else:
        # For standard architectures, try PyTorch hub first, then fall back to timm
        try:
            logger.info(f"Creating model from PyTorch: {model_name}")
            model = get_torch_model(model_name=model_name, pretrained=False, num_classes=1)
        except Exception as e:
            logger.info(f"PyTorch model not found, trying timm library: {model_name}")
            try:
                model = get_timm_model(model_name=model_name, pretrained=False, num_classes=1)
            except Exception as e:
                logger.error(f"✗ Failed to create model: {str(e)}")
                raise

    return model


@torch.no_grad()
def inference(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module,
              args: argparse.Namespace, criterion: torch.nn.Module,
              logger: logging.Logger) -> Dict:
    """
    Run inference on validation set and compute metrics.

    This function performs the actual evaluation of the model on the test dataset. It runs
    in evaluation mode with no gradient computation, processes all batches, and computes
    standard image quality assessment metrics (SRCC, PLCC, KLCC).

    Args:
        val_loader: DataLoader for validation data
        model: Model to evaluate
        args: Command line arguments
        criterion: Loss function (MSE)
        logger: Logger instance

    Returns:
        Dictionary containing predictions, ground truth, and computed metrics
    """
    # Set model to evaluation mode - this disables dropout and uses running stats for batchnorm
    model.eval()
    val_dataset_len = len(val_loader.dataset)
    val_loader_len = len(val_loader)

    # Initialize tracking variables for performance monitoring
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # Storage lists for accumulating results across all batches
    temp_pred_scores = []
    temp_gt_scores = []
    temp_img_names = []

    logger.info(f"Starting inference on {val_dataset_len} images...")

    for i, batch in enumerate(val_loader):
        # Move data to GPU if available
        image_cropped = batch['image_cropped'].cuda(args.gpu, non_blocking=True)
        image_resized = batch['image_resized'].cuda(args.gpu, non_blocking=True)
        target = batch['label'].cuda(args.gpu, non_blocking=True).view(-1)

        # Forward pass - compute predictions
        output = model(image_cropped, image_resized)
        loss = criterion(output.view(-1), target.view(-1))
        losses.update(loss.item(), target.size(0))

        # Accumulate results for later metric computation
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        temp_img_names.extend(batch['image_name'])

        # Log progress periodically
        if i % args.print_freq == 0:
            logger.info(
                f"  [{i}/{val_loader_len}] "
                f"Loss: {losses.val:.4f} (avg: {losses.avg:.4f})"
            )

    # Concatenate all batch results into single tensors
    final_preds = torch.cat(temp_pred_scores)
    final_grotruth = torch.cat(temp_gt_scores)

    # Handle patch-based predictions if the model uses multiple patches per image
    if hasattr(args, 'patch_num') and args.patch_num > 1:
        logger.info(f"Averaging predictions over {args.patch_num} patches per image")
        preds_matrix = final_preds.view(-1, args.patch_num)
        final_preds = preds_matrix.mean(dim=-1).squeeze()
        final_grotruth = final_grotruth.view(-1, args.patch_num).mean(dim=-1).squeeze()

    logger.info(
        f"Dataset size: {val_dataset_len}, "
        f"Predictions shape: {final_preds.shape}, "
        f"Ground truth shape: {final_grotruth.shape}"
    )

    # Sanity check for invalid values that would corrupt metric computation
    if torch.isnan(final_preds).any() or torch.isinf(final_preds).any():
        raise ValueError("Found NaN or inf values in predictions")
    if torch.isnan(final_grotruth).any() or torch.isinf(final_grotruth).any():
        raise ValueError("Found NaN or inf values in ground truth")

    # Compute standard image quality assessment metrics
    # SRCC: Spearman's rank correlation coefficient - measures monotonic relationship
    test_srcc = torchmetrics.functional.spearman_corrcoef(final_preds, final_grotruth).item()
    # PLCC: Pearson's linear correlation coefficient - measures linear relationship
    test_plcc = torchmetrics.functional.pearson_corrcoef(final_preds, final_grotruth).item()
    # KLCC: Kendall's rank correlation coefficient - another rank-based metric
    test_klcc = torchmetrics.functional.kendall_rank_corrcoef(final_preds, final_grotruth).item()

    # Package all results into a dictionary for return
    results = {
        'image_names': temp_img_names,
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


def save_results(results: Dict, model_name: str, train_dataset: str,
                 test_dataset: str, metric_type: str, output_dir: str,
                 logger: logging.Logger) -> None:
    """
    Save inference results to CSV file with detailed metrics.

    This function saves both detailed per-image results and prints a summary of the
    overall performance metrics. The filename includes all relevant configuration
    details for easy identification.

    Args:
        results: Results dictionary from inference
        model_name: Name of the model
        train_dataset: Training dataset type
        test_dataset: Test dataset name
        metric_type: Training metric objective
        output_dir: Base directory to save results
        logger: Logger instance
    """
    # Create the evaluations subdirectory
    eval_dir = os.path.join(output_dir, 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)

    # Prepare detailed per-image results with predictions and errors
    csv_data = []
    for img_name, pred, gt in zip(results['image_names'],
                                  results['predictions'],
                                  results['ground_truth']):
        csv_data.append({
            'image_name': img_name,
            'prediction': pred,
            'ground_truth': gt,
            'absolute_error': abs(pred - gt)
        })

    # Create descriptive filename that includes all configuration details
    csv_filename = f"{model_name}_{train_dataset}_{metric_type}_on_{test_dataset}.csv"
    csv_path = os.path.join(eval_dir, csv_filename)

    # Save detailed results to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to: {csv_path}")

    # Print formatted metrics summary to console and log
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Trained on: {train_dataset}")
    logger.info(f"Metric type: {metric_type}")
    logger.info(f"Tested on: {test_dataset}")
    logger.info("-" * 70)
    logger.info(f"SRCC (Spearman): {results['metrics']['srcc']:.4f}")
    logger.info(f"PLCC (Pearson):  {results['metrics']['plcc']:.4f}")
    logger.info(f"KLCC (Kendall):  {results['metrics']['klcc']:.4f}")
    logger.info(f"MSE Loss:        {results['metrics']['loss']:.4f}")
    logger.info("=" * 70 + "\n")

def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Main inference pipeline orchestrating all steps.

    This function coordinates the entire evaluation process: validating inputs,
    ensuring model weights are available, loading data, creating and loading the model,
    running inference, and saving results.

    Args:
        args: Command line arguments
        logger: Logger instance
    """
    # Validate required arguments
    if not args.model_name:
        raise ValueError("Please specify --model_name")
    if not args.train_dataset:
        raise ValueError("Please specify --train_dataset (cls, det, or ins)")
    if not args.test_dataset:
        raise ValueError("Please specify --test_dataset")
    if not args.metric_type:
        raise ValueError("Please specify --metric_type (both, consistency, or accuracy)")

    logger.info(f"\nStarting MIQA Inference Pipeline")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Trained on: {args.train_dataset}")
    logger.info(f"Metric type: {args.metric_type}")
    logger.info(f"Testing on: {args.test_dataset}")

    # Ensure model weights are available (download if necessary)
    checkpoint_path = ensure_model_weights(args.model_name, args.train_dataset,
                                           args.metric_type, logger)

    if checkpoint_path is None:
        logger.error("Cannot proceed without model weights")
        return

    # Build dataset and dataloader
    logger.info(f"\nLoading {args.test_dataset} dataset...")
    args.dataset = args.test_dataset  # Set dataset name for dataloader builder

    args.eval_only = True  # Indicate evaluation mode
    val_dataset = build_dataloader.build_dataset(args)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    logger.info(f"✓ Loaded {len(val_dataset)} images with {args.workers} workers")

    # Create model architecture
    logger.info(f"\nCreating model architecture...")
    args.arch = args.model_name
    model = create_model(args.model_name, args, logger)

    # Load pre-trained weights into model
    if not load_model_weights(model, checkpoint_path, args, logger):
        logger.error("Failed to load model weights")
        return

    # Move model to GPU if available
    if args.gpu is not None and torch.cuda.is_available():
        model = model.cuda(args.gpu)
        logger.info(f"✓ Model moved to GPU {args.gpu}")
    else:
        logger.warning("GPU not available, using CPU (this will be slower)")

    # Create loss function for evaluation
    criterion = torch.nn.MSELoss()

    # Run inference on the test set
    logger.info(f"\nRunning inference...")
    results = inference(val_loader, model, args, criterion, logger)

    # Save results and print summary
    save_results(results, args.model_name, args.train_dataset,
                 args.test_dataset, args.metric_type, args.output_dir, logger)


if __name__ == '__main__':
    # Parse command line arguments
    parser = get_args()
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['resnet18', 'resnet50', 'efficientnet_b1',
                                 'efficientnet_b5', 'vit_small_patch16_224', 'ra_miqa'],
                        help='Model architecture to use for evaluation')
    parser.add_argument('--train_dataset', type=str, required=True,
                        choices=['cls', 'det', 'ins'],
                        help='Dataset type the model was trained on (cls=classification, det=detection, ins=instance)')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Name of the dataset to test on')
    parser.add_argument('--metric_type', type=str, required=True,
                        choices=['composite', 'consistency', 'accuracy'],
                        help='Training metric objective used (composite=both metrics, consistency=consistency-focused, accuracy=accuracy-focused)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save results (default: outputs)')

    args = parser.parse_args()

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging to both file and console
    log_filename = f"inference_{args.model_name}_{args.train_dataset}_{args.metric_type}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('miqa_inference')

    # Run main inference pipeline with error handling
    try:
        main(args, logger)
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise