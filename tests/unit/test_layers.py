import pytest
import torch
import math
import numpy as np
from riskformer.training.layers import (
    MultiScaleAttention,
    MultiScaleBlock,
    GlobalPoolLayer,
    Mlp,
    SinusoidalPositionalEncoding2D,
)
from unittest.mock import patch

# Fixtures for commonly used test parameters
@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def token_array_dim():
    return 16

@pytest.fixture 
def seq_length():
    return 32 * 32  # 32x32 = 1024 patches

@pytest.fixture
def embedding_dim():
    return 512

@pytest.fixture
def num_heads():
    return 4


# Test DropPath
def test_drop_path(batch_size, seq_length, embedding_dim):
    """Test the drop_path function and DropPath class."""
    from riskformer.training.layers import DropPath
    
    # Create a simple tensor
    x = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Test DropPath module with drop_prob = 0
    drop_layer_zero = DropPath(drop_prob=0.0)
    drop_layer_low = DropPath(drop_prob=0.1)
    drop_layer_high = DropPath(drop_prob=0.5)
    drop_layer_default = DropPath()

    drop_layer_zero.train()
    drop_layer_low.train()
    drop_layer_high.train()
    drop_layer_default.train()

    result_zero = drop_layer_zero(x)
    result_low = drop_layer_low(x)
    result_high = drop_layer_high(x)
    result_default = drop_layer_default(x)

    # Result should be identical to input
    assert torch.allclose(result_zero, x)
    assert torch.allclose(result_default, x)
    
    # Non-zero drop_prob should not be identical to x
    assert result_low.shape == x.shape
    assert result_high.shape == x.shape

    
    # Test DropPath module with drop_prob > 0 but in eval mode
    drop_layer_low.eval()
    drop_layer_high.eval()

    result_low = drop_layer_low(x)
    result_high = drop_layer_high(x)
    # Results should be identical to input
    assert torch.allclose(result_low, x)
    assert torch.allclose(result_high, x)
    

# Test MLP layer
def test_mlp_layer(batch_size, seq_length, embedding_dim):
    """Test the MLP layer."""
    # Create input
    x = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Create MLP with default settings
    mlp = Mlp(in_features=embedding_dim)
    
    # Forward pass
    output = mlp(x)
    
    # Check shape
    assert output.shape == x.shape
    
    # Test with different hidden dimension
    hidden_dim = embedding_dim * 2
    mlp_custom = Mlp(in_features=embedding_dim, hidden_features=hidden_dim)
    output_custom = mlp_custom(x)
    
    # Check shape remains the same
    assert output_custom.shape == x.shape
    
    # Test with dropout
    mlp_dropout = Mlp(in_features=embedding_dim, drop=0.1)
    mlp_dropout.train()  # Set to train mode for dropout
    output_dropout = mlp_dropout(x)
    
    # Check shape with dropout
    assert output_dropout.shape == x.shape


# Test GlobalMaxPoolLayer real implementation
def test_global_max_pool(batch_size, seq_length, embedding_dim):
    """Test the actual GlobalMaxPoolLayer implementation."""
    # Create input
    x = torch.rand(batch_size, seq_length, embedding_dim)
    
    # Create random mask
    mask = torch.randint(0, 2, (batch_size, seq_length, 1))

    # Create the GlobalMaxPoolLayers
    pool_max = GlobalPoolLayer(pool_method="max")
    pool_avg = GlobalPoolLayer(pool_method="avg")
    pool_combined = GlobalPoolLayer(pool_method="combined")
    pool_default = GlobalPoolLayer()
    
    # Forward pass (using 'mask' parameter as per the actual implementation)
    pooled_output_max = pool_max(x, mask=mask)
    pooled_output_avg = pool_avg(x, mask=mask)
    pooled_output_combined = pool_combined(x, mask=mask)
    pooled_output_default = pool_default(x, mask=mask)
    
    # Check output shape (should be [batch_size, embedding_dim] without class token)
    assert pooled_output_max.shape == (batch_size, embedding_dim)
    assert pooled_output_avg.shape == (batch_size, embedding_dim)
    assert pooled_output_combined.shape == (batch_size, embedding_dim)
    assert pooled_output_default.shape == (batch_size, embedding_dim)

    # Test without mask
    pooled_output_no_mask_max = pool_max(x)
    pooled_output_no_mask_avg = pool_avg(x)
    pooled_output_no_mask_combined = pool_combined(x)
    pooled_output_no_mask_default = pool_default(x)
    assert pooled_output_no_mask_max.shape == (batch_size, embedding_dim)
    assert pooled_output_no_mask_avg.shape == (batch_size, embedding_dim)
    assert pooled_output_no_mask_combined.shape == (batch_size, embedding_dim)
    assert pooled_output_no_mask_default.shape == (batch_size, embedding_dim)


# Test SinusoidalPositionalEncoding2D
def test_sinusoidal_positional_encoding_2d(batch_size, token_array_dim, embedding_dim):
    """Test the SinusoidalPositionalEncoding2D layer."""
    # Parameters
    height = width = token_array_dim
    seq_length = height * width
    
    # Create input tensor [B, H*W, C]
    x = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Create encoder
    encoder = SinusoidalPositionalEncoding2D(
        channels=embedding_dim, 
        height=height, 
        width=width,
        use_cls_token=False
    )
    encoded = encoder(x)
    
    # Check shape is preserved
    assert encoded.shape == x.shape
    
    # Also test with class token
    x_with_cls = torch.randn(batch_size, seq_length + 1, embedding_dim)

    # Create encoder with class token
    encoder_with_cls = SinusoidalPositionalEncoding2D(
        channels=embedding_dim, 
        height=height, 
        width=width,
        use_cls_token=True
    )
    
    encoded_with_cls = encoder_with_cls(x_with_cls)
    assert encoded_with_cls.shape == x_with_cls.shape

    # Test error handling
    with pytest.raises(ValueError):
        encoder_wrong_channels = SinusoidalPositionalEncoding2D(
            channels=embedding_dim * 2,
            height=height,
            width=width,
            use_cls_token=False
        )
        encoder_wrong_channels(x)

    with pytest.raises(ValueError):
        encoder_wrong_height = SinusoidalPositionalEncoding2D(
            channels=embedding_dim,
            height=height * 2,
            width=width,
            use_cls_token=False
        )
        encoder_wrong_height(x)


class TestMultiScaleAttention:
    """Comprehensive tests for the MultiScaleAttention class with different configurations."""
    
    @pytest.fixture
    def batch_size(self):
        return 2
    
    @pytest.fixture
    def height(self):
        return 16
    
    @pytest.fixture
    def width(self):
        return 16
    
    @pytest.fixture
    def dim(self):
        return 64
    
    @pytest.fixture
    def dim_out(self):
        return 128
    
    @pytest.fixture
    def num_heads(self):
        return 4
    
    def test_pool_first_no_cls_token(self, batch_size, height, width, dim, dim_out, num_heads):
        """Test with pool_first=True and has_cls_embed=False."""
        
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create MultiScaleAttention layer
        msa = MultiScaleAttention(
            dim=dim,
            dim_out=dim_out,
            input_size=(height, width),
            num_heads=num_heads,
            qkv_bias=True,
            kernel_q=(3, 3),
            kernel_kv=(3, 3),
            stride_q=(2, 2),
            stride_kv=(2, 2),
            mode="conv",
            pool_first=True,
            has_cls_embed=False
        )
        
        # Forward pass
        result = msa(x, (height, width))
        
        # Validate output shape
        if isinstance(result, tuple):
            output = result[0]
            # New shape after pooling with stride 2
            expected_h = math.ceil(height / 2)
            expected_w = math.ceil(width / 2)
            expected_tokens = expected_h * expected_w
            assert output.shape == (batch_size, expected_tokens, dim_out)
            
            # If attention weights are returned
            if len(result) > 1 and result[1] is not None:
                attn = result[1]
                assert attn.shape[0] == batch_size
                assert attn.shape[1] == num_heads
                assert attn.shape[2] == expected_tokens
                assert attn.shape[3] == expected_tokens
        else:
            assert False, "Expected a tuple return with at least output tensor"
    
    def test_pool_first_with_cls_token(self, batch_size, height, width, dim, dim_out, num_heads):
        """Test with pool_first=True and has_cls_embed=True."""
        
        # Create input with class token
        seq_len = height * width + 1  # +1 for class token
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create MultiScaleAttention layer
        msa = MultiScaleAttention(
            dim=dim,
            dim_out=dim_out,
            input_size=(height, width),
            num_heads=num_heads,
            qkv_bias=True,
            kernel_q=(3, 3),
            kernel_kv=(3, 3),
            stride_q=(2, 2),
            stride_kv=(2, 2),
            mode="conv",
            pool_first=True,
            has_cls_embed=True
        )
        
        # Forward pass
        result = msa(x, (height, width))
        
        # Validate output shape
        if isinstance(result, tuple):
            output = result[0]
            # New shape after pooling with stride 2, plus class token
            expected_h = math.ceil(height / 2)
            expected_w = math.ceil(width / 2)
            expected_tokens = expected_h * expected_w + 1  # +1 for class token
            assert output.shape == (batch_size, expected_tokens, dim_out)
        else:
            assert False, "Expected a tuple return with at least output tensor"
    
    def test_no_pool_first_with_cls_token(self, batch_size, height, width, dim, dim_out, num_heads):
        """Test with pool_first=False and has_cls_embed=True."""
        
        # Create input with class token
        seq_len = height * width + 1  # +1 for class token
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create MultiScaleAttention layer
        msa = MultiScaleAttention(
            dim=dim,
            dim_out=dim_out,
            input_size=(height, width),
            num_heads=num_heads,
            qkv_bias=True,
            kernel_q=(1, 1),  # No pooling
            kernel_kv=(1, 1),  # No pooling
            stride_q=(1, 1),  # No pooling
            stride_kv=(1, 1),  # No pooling
            mode="conv",
            pool_first=False,
            has_cls_embed=True
        )
        
        # Forward pass
        result = msa(x, (height, width))
        
        # Validate output shape
        if isinstance(result, tuple):
            output = result[0]
            # Shape should remain the same with class token
            expected_tokens = height * width + 1  # +1 for class token
            assert output.shape == (batch_size, expected_tokens, dim_out)
            
            # If attention weights are returned
            if len(result) > 1 and result[1] is not None:
                attn = result[1]
                assert attn.shape[0] == batch_size
                assert attn.shape[1] == num_heads
                assert attn.shape[2] == expected_tokens
                assert attn.shape[3] == expected_tokens
        else:
            assert False, "Expected a tuple return with at least output tensor"
    
    def test_different_pooling_modes(self, batch_size, height, width, dim, num_heads):
        """Test different pooling modes (conv, avg, max)."""
        
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Test each pooling mode
        for mode in ["conv", "avg", "max"]:
            # Create MultiScaleAttention layer
            msa = MultiScaleAttention(
                dim=dim,
                dim_out=dim,
                input_size=(height, width),
                num_heads=num_heads,
                qkv_bias=True,
                kernel_q=(3, 3),
                kernel_kv=(3, 3),
                stride_q=(2, 2),
                stride_kv=(2, 2),
                mode=mode,
                pool_first=True,
                has_cls_embed=False
            )
            
            # Forward pass
            result = msa(x, (height, width))
            
            # Validate output shape
            if isinstance(result, tuple):
                output = result[0]
                # New shape after pooling with stride 2
                expected_h = math.ceil(height / 2)
                expected_w = math.ceil(width / 2)
                expected_tokens = expected_h * expected_w
                assert output.shape == (batch_size, expected_tokens, dim)
            else:
                assert False, f"Expected a tuple return with at least output tensor for mode {mode}"
    
    def test_relative_positional_embeddings(self, batch_size, height, width, dim, num_heads):
        """Test with relative positional embeddings."""
        
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create MultiScaleAttention layer with relative positional embeddings
        msa = MultiScaleAttention(
            dim=dim,
            dim_out=dim,
            input_size=(height, width),
            num_heads=num_heads,
            qkv_bias=True,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            mode="conv",
            pool_first=False,
            has_cls_embed=False,
            rel_pos_spatial=True,
            rel_pos_zero_init=True
        )
        
        # Forward pass
        result = msa(x, (height, width))
        
        # Validate output shape
        if isinstance(result, tuple):
            output = result[0]
            assert output.shape == (batch_size, seq_len, dim)
            
            # Check if attention weights are correctly shaped
            if len(result) > 1 and result[1] is not None:
                attn = result[1]
                assert attn.shape[0] == batch_size
                assert attn.shape[1] == num_heads
                assert attn.shape[2] == seq_len
                assert attn.shape[3] == seq_len
        else:
            assert False, "Expected a tuple return with at least output tensor"
    
    def test_different_kernel_strides(self, batch_size, height, width, dim, num_heads):
        """Test different kernel and stride configurations."""
        
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Test different kernel and stride combinations
        test_configs = [
            # kernel_q, kernel_kv, stride_q, stride_kv
            ((1, 1), (1, 1), (1, 1), (1, 1)),  # No pooling
            ((3, 3), (1, 1), (2, 2), (1, 1)),  # Pool only query
            ((1, 1), (3, 3), (1, 1), (2, 2)),  # Pool only key/value
            ((5, 5), (3, 3), (4, 4), (2, 2)),  # Different pooling for q and kv
        ]
        
        for kernel_q, kernel_kv, stride_q, stride_kv in test_configs:
            # Create MultiScaleAttention layer
            msa = MultiScaleAttention(
                dim=dim,
                dim_out=dim,
                input_size=(height, width),
                num_heads=num_heads,
                qkv_bias=True,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q,
                stride_kv=stride_kv,
                mode="conv",
                pool_first=True,
                has_cls_embed=False
            )
            
            # Forward pass
            result = msa(x, (height, width))
            
            # Validate output shape
            if isinstance(result, tuple):
                output = result[0]
                # Calculate expected dimensions after pooling
                expected_h = math.ceil(height / stride_q[0])
                expected_w = math.ceil(width / stride_q[1])
                expected_tokens = expected_h * expected_w
                assert output.shape == (batch_size, expected_tokens, dim)
            else:
                assert False, f"Expected a tuple return for config {kernel_q}, {kernel_kv}, {stride_q}, {stride_kv}"
    
    def test_invalid_mode(self, batch_size, height, width, dim, num_heads):
        """Test that invalid pooling mode raises error."""
        
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Try to create MultiScaleAttention with invalid mode
        with pytest.raises(NotImplementedError):
            msa = MultiScaleAttention(
                dim=dim,
                dim_out=dim,
                input_size=(height, width),
                num_heads=num_heads,
                qkv_bias=True,
                kernel_q=(3, 3),
                kernel_kv=(3, 3),
                stride_q=(2, 2),
                stride_kv=(2, 2),
                mode="invalid_mode",  # Invalid mode
                pool_first=True,
                has_cls_embed=False
            )
            
            # This should raise NotImplementedError
            result = msa(x, (height, width))


class TestMultiScaleBlock:
    """Comprehensive tests for the MultiScaleBlock class with different configurations."""
    
    @pytest.fixture
    def batch_size(self):
        return 2
    
    @pytest.fixture
    def height(self):
        return 16
    
    @pytest.fixture
    def width(self):
        return 16
    
    @pytest.fixture
    def dim(self):
        return 64
    
    @pytest.fixture
    def dim_out(self):
        return 128
    
    @pytest.fixture
    def num_heads(self):
        return 4
    
    def test_init(self, dim, dim_out, num_heads):
        """Test initialization of MultiScaleBlock."""
        # Create with basic parameters
        block = MultiScaleBlock(
            dim=dim,
            dim_out=dim_out,
            input_size=(8, 8),
            num_heads=num_heads,
            mlp_ratio=4.0,
            has_cls_embed=False
        )
        
        # Check that critical components are initialized
        assert hasattr(block, 'attn')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'mlp')
        
        # Check output dimension has been set
        assert block.dim_out == dim_out
    
    def test_forward_without_pooling(self, batch_size, height, width, dim):
        """Test forward pass without pooling (identity mapping case)."""
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # No dimension change, no pooling
        block = MultiScaleBlock(
            dim=dim,
            dim_out=dim,  # Same dim
            input_size=(height, width),
            num_heads=8,
            kernel_q=(1, 1),  # No pooling
            kernel_kv=(1, 1),  # No pooling
            stride_q=(1, 1),  # No pooling
            stride_kv=(1, 1),  # No pooling
            has_cls_embed=False
        )
        
        # Forward pass
        output, attn_weights, hw_shape_new, attn_mask = block(x, (height, width))
        
        # Check output shape
        assert output.shape == x.shape
        assert hw_shape_new == (height, width)
        
        # Check attention weights shape
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
            # Verify attention weights are normalized (sum to 1 along axis 3)
            assert torch.allclose(attn_weights.sum(dim=-1), 
                               torch.ones(batch_size, 8, seq_len), 
                               atol=1e-5)
    
    def test_forward_with_downscaling(self, batch_size, height, width, dim, dim_out):
        """Test forward pass with downscaling."""
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Set up downscaling parameters
        stride_q = (2, 2)
        stride_kv = (2, 2)
        
        # Create block with downscaling
        block = MultiScaleBlock(
            dim=dim,
            dim_out=dim_out,  # Dimension change
            input_size=(height, width),
            num_heads=8,
            kernel_q=(3, 3),
            kernel_kv=(3, 3),
            stride_q=stride_q,
            stride_kv=stride_kv,
            has_cls_embed=False,
            rel_pos_spatial=False,  # Disable relative pos to simplify
            pool_first=True  # Use pool_first to handle dimension changes correctly
        )
        
        # Calculate expected downscaled shape
        downscaled_h = (height + stride_q[0] - 1) // stride_q[0]
        downscaled_w = (width + stride_q[1] - 1) // stride_q[1]
        downscaled_seq_len = downscaled_h * downscaled_w
        
        # Forward pass
        output, attn_weights, hw_shape_new, attn_mask = block(x, (height, width))
        
        # Verify output shape
        assert output.shape == (batch_size, downscaled_seq_len, dim_out)
        assert hw_shape_new == (downscaled_h, downscaled_w)
        
        # Validate that the block actually did downscaling by checking
        # that the output sequence length is less than input
        assert output.shape[1] < x.shape[1]
        
        # Validate that dimensions changed correctly
        assert output.shape[2] == dim_out
        
        # Check that attention weights have the right shape and characteristics
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, 8, downscaled_seq_len, downscaled_seq_len)
            # Verify attention weights are normalized (sum to 1 along axis 3)
            assert torch.allclose(attn_weights.sum(dim=-1), 
                               torch.ones(batch_size, 8, downscaled_seq_len), 
                               atol=1e-5)
    
    def test_forward_with_class_token(self, batch_size, height, width, dim):
        """Test forward pass with class token."""
        # Create input with class token
        seq_len = height * width + 1  # +1 for class token
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create block with class token but no downscaling
        block = MultiScaleBlock(
            dim=dim,
            dim_out=dim,
            input_size=(height, width),
            num_heads=8,
            kernel_q=(1, 1),  # No pooling
            kernel_kv=(1, 1),  # No pooling
            stride_q=(1, 1),  # No pooling
            stride_kv=(1, 1),  # No pooling
            has_cls_embed=True
        )
        
        # Forward pass
        output, attn_weights, hw_shape_new, attn_mask = block(x, (height, width))
        
        # Check output shape - should maintain class token
        assert output.shape == x.shape
        assert hw_shape_new == (height, width)
        
        # Check attention weights shape with class token
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
            # Verify attention weights are normalized (sum to 1 along axis 3)
            assert torch.allclose(attn_weights.sum(dim=-1), 
                               torch.ones(batch_size, 8, seq_len), 
                               atol=1e-5)
    
    def test_forward_with_dimension_multiplication(self, batch_size, height, width, dim, dim_out):
        """Test forward pass with dimension multiplication in attention."""
        # Create input
        seq_len = height * width
        x = torch.rand(batch_size, seq_len, dim)
        
        # Create block with dimension multiplication but no pooling
        block = MultiScaleBlock(
            dim=dim,
            dim_out=dim_out,
            input_size=(height, width),
            num_heads=8,
            kernel_q=(1, 1),  # No pooling
            kernel_kv=(1, 1),  # No pooling
            stride_q=(1, 1),  # No pooling
            stride_kv=(1, 1),  # No pooling
            has_cls_embed=False,
            dim_mul_in_att=True  # Enable dimension multiplication in attention
        )
        
        # Forward pass
        output, attn_weights, hw_shape_new, attn_mask = block(x, (height, width))
        
        # Check output shape - should have new dimension
        assert output.shape == (batch_size, seq_len, dim_out)
        assert hw_shape_new == (height, width)
        
        # Check attention weights shape
        if attn_weights is not None:
            assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
            # Verify attention weights are normalized (sum to 1 along axis 3)
            assert torch.allclose(attn_weights.sum(dim=-1), 
                               torch.ones(batch_size, 8, seq_len), 
                               atol=1e-5)


if __name__ == "__main__":
    pytest.main() 