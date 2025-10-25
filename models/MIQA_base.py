import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple, Dict, Any
import logging
from typing import Optional, Union
import timm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_supported_models() -> list:
    """
    Get all supported model architectures from torchvision.models.

    Returns:
        list: List of supported model names
    """
    supported_models = sorted(
        name for name in models.__dict__
        if name.islower() and
        not name.startswith("__") and
        callable(models.__dict__[name])
    )

    logger.info("Available models:")
    for model_name in supported_models:
        logger.info(f"- {model_name}")

    return supported_models


def get_feature_dim(model: nn.Module) -> int:
    """
    Get the feature dimension of the model's last layer.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        int: Dimension of the last layer
    """
    if hasattr(model, 'fc'):
        return model.fc.in_features

    elif hasattr(model, 'head'):
        try:
            return model.head.fc.in_features
        except:
            head = model.head
            if isinstance(head, nn.Sequential):
                return head[-1].in_features
            return head.in_features

    elif hasattr(model, 'heads'):
        try:
            return model.heads.fc.in_features
        except:
            head = model.heads
            if isinstance(head, nn.Sequential):
                return head[-1].in_features
            return head.in_features

    elif hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            return classifier[-1].in_features
        return classifier.in_features
    else:
        raise AttributeError("Model architecture not supported: cannot find fc or classifier layer")


def get_torch_model(
        model_name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None
):
    """
    Create a model with optional pretrained weights and custom number of classes.

    Args:
        model_name (str): Name of the model architecture
        pretrained (bool): Whether to use pretrained weights
        num_classes (Optional[int]): Number of classes for the final layer

    Returns:
        Tuple[nn.Module, int]: (Model, feature dimension)

    Raises:
        ValueError: If model_name is not supported
        RuntimeError: If model initialization fails
    """
    try:
        # Verify model exists
        if model_name not in models.__dict__:
            raise ValueError(
                f"Model {model_name} not found in torchvision.models. "
                "Use list_supported_models() to see available models."
            )

        # Initialize model
        model_fn = models.__dict__[model_name]
        logger.info(f"{'Using pre-trained' if pretrained else 'Creating'} model '{model_name}'")

        try:
            model = model_fn(pretrained=pretrained)
        except TypeError:
            # 处理新版本torchvision的权重加载方式
            weights = 'IMAGENET1K_V1' if pretrained else None
            model = model_fn(weights=weights)

        # Get feature dimension
        feature_dim = get_feature_dim(model)

        # Modify final layer if num_classes is specified
        if num_classes is not None:
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(feature_dim, num_classes)
                logger.info(f"Modified fc layer to output {num_classes} classes")
            elif hasattr(model, 'head'):
                model.head = nn.Linear(feature_dim, num_classes)
                logger.info(f"Modified head layer to output {num_classes} classes")
            elif hasattr(model, 'heads'):
                model.heads = nn.Linear(feature_dim, num_classes)
                logger.info(f"Modified heads layer to output {num_classes} classes")
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Linear(feature_dim, num_classes)
                logger.info(f"Modified classifier layer to output {num_classes} classes")

        return model

    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

def get_timm_model(
        model_name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        freeze_backbone: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        custom_head: Optional[nn.Module] = None
) -> nn.Module:
    """
    General-purpose model loader that supports loading pretrained models
    from timm with flexible configuration.

    Args:
        model_name (str): Model name, must be supported by timm.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        num_classes (Optional[int], optional): Number of classes. If None, keep the original output. Defaults to None.
        freeze_backbone (bool, optional): Whether to freeze the backbone network. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.0.
        drop_path_rate (float, optional): Drop path rate (for deep networks). Defaults to 0.0.
        custom_head (Optional[nn.Module], optional): Custom classification head. Defaults to None.

    Returns:
        nn.Module: Configured model instance
    """
    try:
        # Try creating the model
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Freeze backbone if required
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the classification head
            if hasattr(model, 'head') and model.head is not None:
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc') and model.fc is not None:
                for param in model.fc.parameters():
                    param.requires_grad = True

        # Replace classification head with custom head if provided
        if custom_head is not None:
            if hasattr(model, 'head'):
                model.head = custom_head
            elif hasattr(model, 'fc'):
                model.fc = custom_head

        return model

    except Exception as e:
        print(f"Error occurred while loading model {model_name}: {e}")
        # You can raise an exception or return None as needed
        raise ValueError(f"Failed to load model {model_name}")


if __name__ == "__main__":

    # List available models
    # supported_models = list_supported_models()

    try:
        model_name = "efficientnet_b5"
        model = get_torch_model(
            model_name=model_name,
            pretrained=True,
            num_classes=1
        )
    except Exception as e:
        logger.error(f"Failed to run example: {str(e)}")

    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
