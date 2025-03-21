import torch
import math
import numpy as np
from typing import Tuple, Dict, Optional, Union, List


def create_properly_shaped_input(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    embed_dim: int,
    use_phi: bool = False,
    phi_dim: Optional[int] = None,
    use_class_token: bool = False
) -> Dict[str, Union[torch.Tensor, Tuple[int, int]]]:
    """Create input with proper shapes for RiskFormer_ViT testing.
    
    Args:
        batch_size: Batch size
        channels: Number of input channels
        height: Height of input image
        width: Width of input image
        embed_dim: Embedding dimension
        use_phi: Whether phi network is used
        phi_dim: Phi network output dimension if use_phi is True
        use_class_token: Whether class token is used
        
    Returns:
        Dictionary containing input tensor, token tensor, and hw_shape
    """
    # Create initial input tensor
    x = torch.rand(batch_size, channels, height, width)
    
    # For testing prepare_tokens without running the actual method
    token_dim = phi_dim if use_phi else embed_dim
    seq_len = height * width
    
    # Shape that would result from reshaping (and optional phi network)
    tokens = torch.rand(batch_size, seq_len, token_dim)
    
    # Add class token if needed
    if use_class_token:
        cls_token = torch.rand(batch_size, 1, token_dim)
        tokens = torch.cat([cls_token, tokens], dim=1)
    
    return {
        'input': x,
        'tokens': tokens,
        'hw_shape': (height, width)
    }


def calculate_downscaled_shape(
    height: int, 
    width: int, 
    stride_q: Union[int, Tuple[int, int]],
    num_downscale_blocks: int = 1
) -> Tuple[int, int]:
    """Calculate shape after downscaling with given strides.
    
    Args:
        height: Original height
        width: Original width
        stride_q: Stride value or tuple for query
        num_downscale_blocks: Number of downscale blocks to apply
        
    Returns:
        Tuple of (new_height, new_width)
    """
    # Convert single int to tuple if needed
    if isinstance(stride_q, int):
        stride_q = (stride_q, stride_q)
    
    # Apply downscaling for each block
    h, w = height, width
    for _ in range(num_downscale_blocks):
        h = math.ceil(h / stride_q[0])
        w = math.ceil(w / stride_q[1])
    
    return h, w


def create_multiscale_attention_inputs(
    batch_size: int,
    height: int,
    width: int,
    dim: int,
    dim_out: int,
    num_heads: int,
    stride_q: Union[int, Tuple[int, int]] = (2, 2),
    has_cls_embed: bool = False
) -> Dict[str, Union[torch.Tensor, Tuple]]:
    """Create appropriately shaped inputs for MultiScaleAttention testing.
    
    Args:
        batch_size: Batch size
        height: Height of input feature map
        width: Width of input feature map
        dim: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        stride_q: Stride for query pooling
        has_cls_embed: Whether input includes class embedding
        
    Returns:
        Dictionary with input tensors and expected output shapes
    """
    # Sequence length calculation
    seq_len = height * width
    if has_cls_embed:
        seq_len += 1
    
    # Input tensor
    x = torch.rand(batch_size, seq_len, dim)
    
    # Calculate expected output shapes
    if isinstance(stride_q, int):
        stride_q = (stride_q, stride_q)
        
    out_h = math.ceil(height / stride_q[0])
    out_w = math.ceil(width / stride_q[1])
    out_seq_len = out_h * out_w
    if has_cls_embed:
        out_seq_len += 1
    
    # Expected output shape
    expected_output_shape = (batch_size, out_seq_len, dim_out)
    
    # Attention weights expected shape
    attn_shape = (batch_size, num_heads, out_seq_len, out_seq_len)
    
    # Output hw_shape
    output_hw_shape = (out_h, out_w)
    
    return {
        'input': x,
        'hw_shape': (height, width),
        'expected_output_shape': expected_output_shape,
        'expected_attn_shape': attn_shape,
        'output_hw_shape': output_hw_shape
    }


def create_riskformer_vit_inputs(
    batch_size: int,
    token_array_dim: int,
    channels: int,
    input_embed_dim: int,
    output_embed_dim: int,
    use_phi: bool = False,
    phi_dim: Optional[int] = None,
    use_class_token: bool = False,
    downscale_depth: int = 1,
    downscale_stride_q: int = 2,
    tasks: Optional[Dict] = None
) -> Dict[str, Union[torch.Tensor, Dict]]:
    """Create appropriately shaped inputs for RiskFormer_ViT testing.
    
    Args:
        batch_size: Batch size
        token_array_dim: Dimension of token array (height=width)
        channels: Number of input channels
        input_embed_dim: Input embedding dimension
        output_embed_dim: Output embedding dimension
        use_phi: Whether phi network is used
        phi_dim: Phi dimension (if use_phi is True)
        use_class_token: Whether class token is used
        downscale_depth: Depth of downscaling blocks
        downscale_stride_q: Stride for downscaling queries
        tasks: Task configuration
        
    Returns:
        Dictionary with input tensors and expected shapes
    """
    # Default tasks if not provided
    if tasks is None:
        tasks = {
            "binary_task": {
                "type": "binary",
                "num_classes": 1,
                "activation": "sigmoid"
            }
        }
    
    # Count total number of outputs from tasks
    total_outputs = sum(task.get('num_classes', 1) for task in tasks.values())
    
    # Create input tensor
    x = torch.rand(batch_size, channels, token_array_dim, token_array_dim)
    
    # Calculate shapes after prepare_tokens
    token_dim = phi_dim if use_phi else output_embed_dim
    seq_len = token_array_dim * token_array_dim
    if use_class_token:
        seq_len += 1
    
    # Tokens after prepare_tokens
    prepared_tokens = torch.rand(batch_size, seq_len, token_dim)
    
    # Calculate downscaled shape
    ds_h, ds_w = calculate_downscaled_shape(
        token_array_dim, 
        token_array_dim, 
        downscale_stride_q,
        downscale_depth
    )
    
    # Calculate number of tokens after downscaling
    ds_seq_len = ds_h * ds_w
    if use_class_token:
        ds_seq_len += 1
    
    # Tokens after downscaling
    downscaled_tokens = torch.rand(batch_size, ds_seq_len, output_embed_dim)
    
    # Global features after processing
    global_features = torch.rand(batch_size, output_embed_dim)
    
    # Expected output shapes
    expected_outputs = {}
    for task_name in tasks:
        num_classes = tasks[task_name].get('num_classes', 1)
        # Shape includes bag predictions plus one global prediction
        expected_outputs[task_name] = (batch_size + 1, num_classes)
    
    result = {
        'input': x,
        'prepared_tokens': prepared_tokens,
        'hw_shape': (token_array_dim, token_array_dim),
        'downscaled_tokens': downscaled_tokens,
        'downscaled_hw_shape': (ds_h, ds_w),
        'global_features': global_features,
        'expected_output_shapes': expected_outputs
    }
    
    # Include phi tensor when use_phi is True
    if use_phi:
        if phi_dim is None:
            raise ValueError("phi_dim must be provided when use_phi is True")
        result['phi'] = torch.rand(batch_size, phi_dim)
    
    return result


def trace_tensor_shapes(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, List]:
    """Trace shapes of tensors in a dictionary.
    
    Args:
        tensor_dict: Dictionary of tensors
        
    Returns:
        Dictionary with tensor names as keys and shapes as values
    """
    shape_dict = {}
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            shape_dict[name] = list(tensor.shape)
        elif isinstance(tensor, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in tensor):
            shape_dict[name] = [list(t.shape) for t in tensor]
    
    return shape_dict


def assert_expected_shapes(tensor_dict: Dict[str, torch.Tensor], 
                          expected_shapes: Dict[str, List]) -> None:
    """Assert that tensors have expected shapes.
    
    Args:
        tensor_dict: Dictionary of tensors
        expected_shapes: Dictionary with expected shapes
    """
    for name, tensor in tensor_dict.items():
        if name in expected_shapes:
            if isinstance(tensor, torch.Tensor):
                expected = expected_shapes[name]
                actual = list(tensor.shape)
                assert actual == expected, f"Shape mismatch for {name}: expected {expected}, got {actual}" 