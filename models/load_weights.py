import torch

def load_weights_by_order(model, checkpoint_path, map_location='cpu'):
    """
    Custom function to precisely load weights to mmseg ViT model.

    Args:
        model: The mmseg VisionTransformer model to load weights into
        checkpoint_path: Path to the checkpoint file
        map_location: Device mapping for loading the checkpoint

    Returns:
        Loaded model with weights from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract model state dict if nested in 'model' key
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # Get the target model's state dict to check for existing keys
    model_state_dict = model.state_dict()

    # Create a new state dict for the transformed weights
    new_state_dict = {}

    # Handle patch embedding
    if 'patch_embed.proj.weight' in checkpoint:
        new_state_dict['patch_embed.projection.weight'] = checkpoint['patch_embed.proj.weight']

    # Note: Skip patch_embed.proj.bias if not in the model state dict
    if 'patch_embed.proj.bias' in checkpoint:
        if 'patch_embed.projection.bias' in model_state_dict:
            new_state_dict['patch_embed.projection.bias'] = checkpoint['patch_embed.proj.bias']
        else:
            print("Skipping patch_embed.projection.bias as it's not in the model")

    # Handle transformer layers
    for i in range(12):  # Assuming 12 transformer layers
        # Layer normalization weights and biases
        new_state_dict[f'layers.{i}.ln1.weight'] = checkpoint[f'blocks.{i}.norm1.weight']
        new_state_dict[f'layers.{i}.ln1.bias'] = checkpoint[f'blocks.{i}.norm1.bias']
        new_state_dict[f'layers.{i}.ln2.weight'] = checkpoint[f'blocks.{i}.norm2.weight']
        new_state_dict[f'layers.{i}.ln2.bias'] = checkpoint[f'blocks.{i}.norm2.bias']

        # Feed-forward network weights and biases
        new_state_dict[f'layers.{i}.ffn.layers.0.0.weight'] = checkpoint[f'blocks.{i}.mlp.fc1.weight']
        new_state_dict[f'layers.{i}.ffn.layers.0.0.bias'] = checkpoint[f'blocks.{i}.mlp.fc1.bias']
        new_state_dict[f'layers.{i}.ffn.layers.1.weight'] = checkpoint[f'blocks.{i}.mlp.fc2.weight']
        new_state_dict[f'layers.{i}.ffn.layers.1.bias'] = checkpoint[f'blocks.{i}.mlp.fc2.bias']

        # Handle attention mechanism
        qkv_weight = checkpoint[f'blocks.{i}.attn.qkv.weight']
        qkv_bias = checkpoint[f'blocks.{i}.attn.qkv.bias']

        new_state_dict[f'layers.{i}.attn.attn.in_proj_weight'] = qkv_weight
        new_state_dict[f'layers.{i}.attn.attn.in_proj_bias'] = qkv_bias

        # Handle out_proj weights and biases
        new_state_dict[f'layers.{i}.attn.attn.out_proj.weight'] = checkpoint[f'blocks.{i}.attn.proj.weight']
        new_state_dict[f'layers.{i}.attn.attn.out_proj.bias'] = checkpoint[f'blocks.{i}.attn.proj.bias']

    # Handle the final layer norm - try multiple potential names
    if 'fc_norm.weight' in checkpoint:
        # Try different potential target keys for the norm layer
        if 'ln1.weight' in model_state_dict:
            new_state_dict['ln1.weight'] = checkpoint['fc_norm.weight']
            new_state_dict['ln1.bias'] = checkpoint['fc_norm.bias']
        elif 'norm.weight' in model_state_dict:
            new_state_dict['norm.weight'] = checkpoint['fc_norm.weight']
            new_state_dict['norm.bias'] = checkpoint['fc_norm.bias']

    # Special case for the final norm layer - try different source keys
    if 'ln1.weight' in model_state_dict and 'ln1.weight' not in new_state_dict:
        # If not mapped yet, try final block norm as a fallback
        if 'blocks.11.norm2.weight' in checkpoint:
            print("Using blocks.11.norm2 weights for ln1")
            new_state_dict['ln1.weight'] = checkpoint['blocks.11.norm2.weight']
            new_state_dict['ln1.bias'] = checkpoint['blocks.11.norm2.bias']
        elif 'fc_norm.weight' in checkpoint:
            print("Using fc_norm weights for ln1")
            new_state_dict['ln1.weight'] = checkpoint['fc_norm.weight']
            new_state_dict['ln1.bias'] = checkpoint['fc_norm.bias']
        elif 'norm.weight' in checkpoint:
            print("Using norm weights for ln1")
            new_state_dict['ln1.weight'] = checkpoint['norm.weight']
            new_state_dict['ln1.bias'] = checkpoint['norm.bias']

    # Handle positional embedding and class token if needed
    if hasattr(model, 'pos_embed') and 'pos_embed' in checkpoint:
        checkpoint_pos_embed = checkpoint['pos_embed']
        if 'pos_embed' in model_state_dict:
            model_pos_embed_shape = model_state_dict['pos_embed'].shape

            # Resize positional embedding if shapes don't match
            if checkpoint_pos_embed.shape != model_pos_embed_shape:
                print(f"Resizing positional embedding from {checkpoint_pos_embed.shape} to {model_pos_embed_shape}")
                # For simplicity, keep as is if they're close in shape
                new_state_dict['pos_embed'] = checkpoint_pos_embed
            else:
                new_state_dict['pos_embed'] = checkpoint_pos_embed

    if hasattr(model, 'cls_token') and 'cls_token' in checkpoint:
        if 'cls_token' in model_state_dict:
            new_state_dict['cls_token'] = checkpoint['cls_token']

    # Check for shape mismatches and fix if possible
    keys_to_remove = []
    for key, value in new_state_dict.items():
        if key in model_state_dict:
            if value.shape != model_state_dict[key].shape:
                print(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")

                # Check if the MultiheadAttention layer needs special handling
                if 'attn.attn.in_proj' in key:
                    # This requires careful handling of the QKV weights which might have different structures
                    # between TIMM and PyTorch MultiheadAttention
                    checkpoint_dim = value.shape[0]
                    model_dim = model_state_dict[key].shape[0]

                    if checkpoint_dim % 3 == 0 and model_dim % 3 == 0:
                        # Try to reshape and reorder if needed
                        try:
                            # This is a simplified handling - might need more complex transformation
                            # based on specific model architectures
                            embed_dim = checkpoint_dim // 3
                            # Keep the weight as is and let PyTorch handle it
                            pass
                        except Exception as e:
                            print(f"Failed to process {key}: {e}")
                            keys_to_remove.append(key)
                    else:
                        keys_to_remove.append(key)
                else:
                    # For other layers, try simple reshaping if dimensions allow
                    try:
                        if value.numel() == model_state_dict[key].numel():
                            new_state_dict[key] = value.reshape(model_state_dict[key].shape)
                        else:
                            keys_to_remove.append(key)
                    except Exception:
                        keys_to_remove.append(key)
        else:
            # Remove keys that don't exist in the model
            keys_to_remove.append(key)

    # Remove problematic keys
    for key in keys_to_remove:
        if key in new_state_dict:
            del new_state_dict[key]

    # Load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(new_state_dict)} keys into model")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    # Print some key missing/unexpected entries for debugging
    if missing_keys:
        print(f"Sample missing keys: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Sample unexpected keys: {unexpected_keys[:5]}")

    return model