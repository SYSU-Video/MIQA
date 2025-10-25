import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple

from mmseg.models import build_segmentor
from mmengine.config import Config
from models.load_weights import load_weights_by_order
from models.download_checkpoint import ensure_checkpoint

class RegionVisionTransformer(nn.Module):
    """
    Region-aware Vision Transformer for Machine-centric Image Quality Assessment (MIQA).

    This model integrates semantic region information with vision transformer features
    for enhanced machine-centric image quality prediction. It consists of two main components:
    1. A frozen semantic segmentation encoder (SERE ViT) for region feature extraction
    2. A trainable vision transformer backbone for quality assessment

    Architecture:
        - Base Model: Vision Transformer (ViT) with patch-based image embedding
        - Mask Encoder: Pre-trained SERE ViT-S/16 (SSL+Supervised) for semantic features
        - Token Fusion: Region token + CLS token + Patch tokens
        - Quality Regressor: MLP head for final quality score prediction

    References:
        - SERE ViT: https://github.com/open-mmlab/mmsegmentation/tree/master/configs/imagenets
        - ImageNet-S Dataset: https://github.com/LUSSeg/mmsegmentation/tree/imagenets/configs/imagenets

    Pre-trained Weights:
        - Config: fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py
        - Default Checkpoint: https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth
        - Alternative: https://download.openmmlab.com/mmsegmentation/v0.5/imagenets/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919_20230208_151834-ee33230c.pth

    Args:
        base_model_name (str): Name of the base ViT model from timm library.
            Default: 'vit_small_patch16_224'
        pretrained (bool): Whether to load ImageNet pre-trained weights for base model.
            Default: True
        # img_size (int): Input image size. Default: 224
        # patch_size (int): Patch size for vision transformer. Default: 16
        mmseg_config_path (str): Path to the mmsegmentation config file for SERE model.
            Default: 'model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py'
        checkpoint_path (str): Path to the pre-trained SERE checkpoint weights.
            Default: 'checkpoints/sere_finetuned_vit_small_ep100.pth'
        semantic_feature_dim (int): Dimension of semantic features from mask encoder.
            Default: 384

    Note:
        - Requires installation of: ftfy, regex
        - The mask encoder weights are frozen during training
        - If checkpoint doesn't exist, implement auto-download functionality
    """

    # Default checkpoint URLs (primary and fallback)
    DEFAULT_CHECKPOINT_URLS = [
        'https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth',
        'https://download.openmmlab.com/mmsegmentation/v0.5/imagenets/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919_20230208_151834-ee33230c.pth'
    ]

    def __init__(
            self,
            base_model_name: str = 'vit_small_patch16_224',
            pretrained: bool = True,
            mmseg_config_path: str = 'models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
            checkpoint_path: str = 'models/checkpoints/sere_finetuned_vit_small_ep100.pth',
            mask_encoder_dim: int = 384,
            auto_download: bool = True,
            force_download: bool = False
    ):
        super().__init__()

        # ======================== Checkpoint Management ========================
        if auto_download:
            print("\n" + "=" * 60)
            print("SERE Checkpoint Management")
            print("=" * 60)

            checkpoint_available = ensure_checkpoint(
                checkpoint_path=checkpoint_path,
                download_urls=self.DEFAULT_CHECKPOINT_URLS,
                force_download=force_download
            )

            if not checkpoint_available:
                raise FileNotFoundError(
                    f"Failed to load or download checkpoint. "
                    f"Please manually download from:\n"
                    f"  1. {self.DEFAULT_CHECKPOINT_URLS[0]}\n"
                    f"  or\n"
                    f"  2. {self.DEFAULT_CHECKPOINT_URLS[1]}\n"
                    f"And save to: {checkpoint_path}"
                )

            print("=" * 60 + "\n")

        # ======================== Base Vision Transformer Initialization ========================
        # Load pre-trained ViT model from timm library
        self.base_model = timm.create_model(
            base_model_name,
            pretrained=pretrained,
            num_classes=1,  # Placeholder, will be replaced by custom regressor
        )

        # Remove the original classification head components as we'll use custom quality regressor
        del self.base_model.head, self.base_model.fc_norm, self.base_model.head_drop

        # Extract embedding dimension from the base model (e.g., 384 for ViT-Small)
        self.embed_dim = self.base_model.embed_dim

        # ======================== Semantic Mask Encoder Initialization ========================
        # Load SERE ViT-S/16 (Self-supervised + Supervised) model from mmsegmentation
        # This encoder extracts semantic region features from input masks/images
        cfg = Config.fromfile(mmseg_config_path)

        # Ensure the model configuration has the required type field
        if 'type' not in cfg.model:
            cfg.model['type'] = 'EncoderDecoder'  # Default type for mmsegmentation models

        # Disable training configuration for inference-only usage
        cfg.model.train_cfg = None

        # Build the segmentation model and extract only the backbone encoder
        self.mask_encoder = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')).backbone

        # Modify the patch embedding projection layer to enable bias
        # This is necessary for compatibility with pre-trained SERE weights
        old_conv = self.mask_encoder.patch_embed.projection
        new_conv = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=True  # Enable bias (original may have bias=False)
        )
        self.mask_encoder.patch_embed.projection = new_conv

        # Load pre-trained SERE weights with order-based matching
        self.mask_encoder = load_weights_by_order(
            self.mask_encoder,
            checkpoint_path,
            map_location='cpu'
        )

        # Freeze all parameters in the mask encoder (feature extractor only)
        for param in self.mask_encoder.parameters():
            param.requires_grad = False

        # Set mask encoder to evaluation mode permanently
        self.mask_encoder.eval()

        # ======================== Feature Projection and Fusion ========================
        # Project semantic features to match the embedding dimension of base ViT
        # This enables seamless integration of region tokens with patch tokens
        self.mask_projector = nn.Sequential(
            nn.Linear(mask_encoder_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )

        # ======================== Position Embedding Modification ========================
        # Extend position embeddings to accommodate the additional region token
        # Original: [CLS] + [Patch_1, Patch_2, ..., Patch_N]
        # Modified: [Region] + [CLS] + [Patch_1, Patch_2, ..., Patch_N]
        num_patches = self.base_model.patch_embed.num_patches
        orig_pos_embed = self.base_model.pos_embed  # Shape: [1, 1+num_patches, embed_dim]

        # Create extended position embeddings: 1 (region) + 1 (cls) + num_patches (patches)
        new_pos_embed = nn.Parameter(torch.zeros(1, 2 + num_patches, self.embed_dim))

        # Copy original CLS token position embedding to first position
        new_pos_embed.data[0, 0] = orig_pos_embed.data[0, 0]

        # Initialize region token position embedding with Gaussian noise
        nn.init.normal_(new_pos_embed.data[0, 1:2], std=0.02)

        # Copy original patch position embeddings to positions 2 onwards
        new_pos_embed.data[0, 2:] = orig_pos_embed.data[0, 1:]

        self.pos_embed = new_pos_embed

        # ======================== Quality Regression Head ========================
        # Final MLP head to predict quality score from fused [CLS + Region] tokens
        self.quality_regressor = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),  # Concatenated features
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, 1),  # Single quality score output
        )

    def extract_mask_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic region features from input mask/image using frozen SERE encoder.

        Args:
            x (torch.Tensor): Input mask or auxiliary image, shape [B, 3, H, W]

        Returns:
            torch.Tensor: Projected region token features, shape [B, C, H', W']
                         where C=embed_dim, H'=H/patch_size, W'=W/patch_size
        """
        with torch.no_grad():
            # Extract hierarchical features using the frozen backbone
            mask_features = self.mask_encoder(x)

            # Select the first feature map (typically the highest resolution)
            # Shape: [B, semantic_feature_dim, H', W'] (e.g., [B, 384, 14, 14])
            mask_features = mask_features[0].mean(dim=[2, 3])  # Global average pooling to [B, semantic_feature_dim]
        # print('-----------',mask_features.shape)
        # Project semantic features to match base model's embedding dimension
        # This enables direct concatenation with patch tokens
        region_token = self.mask_projector(mask_features)

        return region_token

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quality score prediction with region-aware features.

        Args:
            x (torch.Tensor): Primary input image, shape [B, 3, H, W]
            y (torch.Tensor): Auxiliary input (region mask or secondary image), shape [B, 3, H, W]

        Returns:
            torch.Tensor: Predicted quality scores, shape [B]

        Forward Pipeline:
            1. Extract patch embeddings from primary image
            2. Extract region features from auxiliary input
            3. Fuse tokens: [Region Token] + [CLS Token] + [Patch Tokens]
            4. Add position embeddings
            5. Process through transformer blocks
            6. Predict quality score from [CLS + Region] tokens
        """
        # ======================== Step 1: Patch Embedding ========================
        # Convert input image to patch embeddings
        # Shape: [B, N_patches, embed_dim] where N_patches = (H/patch_size) * (W/patch_size)
        x = self.base_model.patch_embed(x)

        # ======================== Step 2: Region Feature Extraction ========================
        # Extract semantic region features from auxiliary input and add sequence dimension
        # Shape: [B, C, H', W'] -> [B, 1, embed_dim] (global pooled or averaged)
        region_token = self.extract_mask_features(y).unsqueeze(1)

        # ======================== Step 3: Token Preparation ========================
        # Expand CLS token for the entire batch
        # Shape: [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.base_model.cls_token.expand(x.shape[0], -1, -1)

        # ======================== Step 4: Token Concatenation ========================
        # Concatenate all tokens: [Region] + [CLS] + [Patch_1, ..., Patch_N]
        # Final shape: [B, 2 + N_patches, embed_dim]
        x = torch.cat((region_token, cls_token, x), dim=1)

        # ======================== Step 5: Position Encoding ========================
        # Add learnable position embeddings to all tokens
        x = x + self.pos_embed
        x = self.base_model.pos_drop(x)

        # ======================== Step 6: Transformer Encoding ========================
        # Process through all transformer blocks with self-attention
        for blk in self.base_model.blocks:
            x = blk(x)

        # Apply final layer normalization
        x = self.base_model.norm(x)

        # ======================== Step 7: Token Extraction ========================
        # Extract processed region and CLS tokens for quality prediction
        region_token_output = x[:, 0]  # Region token at position 0
        cls_token_output = x[:, 1]  # CLS token at position 1

        # ======================== Step 8: Quality Score Prediction ========================
        # Concatenate region and CLS features for final regression
        # Shape: [B, embed_dim*2] -> [B, 1] -> [B]
        quality_score = self.quality_regressor(
            torch.cat([cls_token_output, region_token_output], dim=-1)
        )

        return quality_score.view(-1)


def main():
    """
    Example usage and model testing function.
    Demonstrates model initialization, parameter counting, and forward pass.
    """
    # ======================== Model Initialization ========================
    model = RegionVisionTransformer(
        base_model_name='vit_small_patch16_224',
        pretrained=True,
        mmseg_config_path='models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
        checkpoint_path='models/checkpoints/sere_finetuned_vit_small_ep100.pth'
    )

    # Display model architecture
    print(model)
    print("\n" + "=" * 80)

    # ======================== Parameter Statistics ========================
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print(f"Frozen Parameters: {frozen_params / 1e6:.2f} M")
    print("=" * 80 + "\n")

    # ======================== Forward Pass Test ========================
    # Create sample inputs (batch_size=2, channels=3, height=224, width=224)
    image = torch.randn(2, 3, 224, 224)
    image_addition = torch.randn(2, 3, 224, 224)

    # Perform forward pass
    with torch.no_grad():
        quality_score = model(image, image_addition)

    print(f"Input Shape: {image.shape}")
    print(f"Quality Score Shape: {quality_score.shape}")
    print(f"Quality Scores: {quality_score}")


def load_checkpoint_example():
    """
    Example function for loading and inspecting saved model checkpoints.
    """
    weights_path = r'checkpoints\composite_metric\miqa_ra_miqa_cls_composite_metric.pth.tar'

    # Load checkpoint from file
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Handle potential 'module.' prefix in state dict (from DataParallel/DistributedDataParallel)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Display all keys in the checkpoint
    print("Checkpoint Keys:")
    print("=" * 80)
    for key in state_dict.keys():
        print(f"  {key}")
    print("=" * 80)


if __name__ == "__main__":
    main()
