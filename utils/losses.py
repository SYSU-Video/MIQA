import torch.nn as nn

def build_loss_function(loss_name: str, **kwargs):
    """
    Get the specified loss function in PyTorch with its parameters.

    Args:
        loss_name (str): The name of the loss function. Options: 'mse', 'huber', 'l1'.
        **kwargs: Additional parameters for the specified loss function.
                  For example, for Huber loss, the parameter `delta` is used.

    Returns:
        loss_function (torch.nn.Module): The corresponding PyTorch loss function.
    """
    # Handle different loss functions based on `loss_name`
    if loss_name.lower() == 'mse':
        # Mean Squared Error Loss
        reduction = kwargs.get('reduction', 'mean')  # Default to 'mean' reduction
        return nn.MSELoss(reduction=reduction)

    elif loss_name.lower() == 'huber':
        # Huber Loss
        delta = kwargs.get('delta', 0.5)  # Default delta is 1.0
        reduction = kwargs.get('reduction', 'mean')  # Default to 'mean' reduction
        return nn.SmoothL1Loss(beta=delta, reduction=reduction)

    elif loss_name.lower() == 'l1':
        # L1 Loss (Mean Absolute Error)
        reduction = kwargs.get('reduction', 'mean')  # Default to 'mean' reduction
        return nn.L1Loss(reduction=reduction)

    elif loss_name.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unsupported loss function: {loss_name}. Available options are 'mse', 'huber', 'l1'.")







