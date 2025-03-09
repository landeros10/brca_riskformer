import pytest
import torch
import math
import numpy as np
from riskformer.training.model import RiskFormer_ViT
from riskformer.training.layers import SinusoidalPositionalEncoding2D

# Fixtures for commonly used test parameters
@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def input_size():
    return 16  # 16x16 patches

@pytest.fixture
def embedding_dim():
    return 128

@pytest.fixture
def basic_model_params():
    return {
        "input_embed_dim": 128,
        "output_embed_dim": 128,
        "use_phi": True,
        "drop_path_rate": 0.1,
        "drop_rate": 0.1,
        "num_classes": 2,
        "max_dim": 256,
        "depth": 2,
        "global_depth": 1,
        "encoding_method": "standard",
        "mask_num": 2,
        "mask_preglobal": False,
        "num_heads": 4,
        "use_attn_mask": True,
        "mlp_ratio": 4.0,
        "use_class_token": True,
        "global_k": -1,
        "downscale_depth": 1,
        "downscale_multiplier": 1.25,
        "noise_aug": 0.1
    }

@pytest.fixture
def create_dummy_input(batch_size, input_size, embedding_dim):
    """Create a dummy input tensor with non-zero values."""
    shape = (batch_size, input_size, input_size, embedding_dim)
    # Create tensor with small non-zero values to ensure masks work correctly
    x = torch.ones(shape) * 0.1
    # Add some larger values to simulate features
    x[:, input_size//4:input_size//2, input_size//4:input_size//2, :] = 1.0
    return x

# Test basic model initialization
def test_model_initialization(basic_model_params):
    """Test that the model initializes without errors."""
    model = RiskFormer_ViT(**basic_model_params)
    assert isinstance(model, RiskFormer_ViT)
    assert model.num_classes == basic_model_params["num_classes"]
    assert model.use_phi == basic_model_params["use_phi"]
    assert model.use_class_token == basic_model_params["use_class_token"]

# Test model with different position encoding methods
@pytest.mark.parametrize("encoding_method", ["standard", "sinusoidal", "conditional", "ppeg"])
def test_position_encoding_methods(basic_model_params, encoding_method):
    """Test that the model works with different position encoding methods."""
    params = basic_model_params.copy()
    params["encoding_method"] = encoding_method
    model = RiskFormer_ViT(**params)
    assert model.encoding_method == encoding_method

# Test basic forward pass
def test_forward_pass(basic_model_params, create_dummy_input):
    """Test the forward pass with a dummy input."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Switch to eval mode for deterministic output
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape: should be [batch_size, num_classes]
    assert output.shape == (x.shape[0], basic_model_params["num_classes"])
    
    # Verify that output values are valid probabilities (sum to 1)
    assert torch.allclose(output.sum(dim=1), torch.ones(x.shape[0]), atol=1e-6)

# Test mask generation
def test_mask_generation(basic_model_params, create_dummy_input):
    """Test that mask generation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Generate masks
    masks = model.generate_masks(x)
    
    # Check mask shape: should be [batch_size, H*W]
    expected_shape = (x.shape[0], x.shape[1] * x.shape[2])
    assert masks.shape == expected_shape
    
    # Verify mask values are boolean
    assert masks.dtype == torch.bool
    
    # Test with use_attn_mask = False
    params = basic_model_params.copy()
    params["use_attn_mask"] = False
    model = RiskFormer_ViT(**params)
    masks = model.generate_masks(x)
    assert masks is None

# Test data augmentation: flip_rotate
def test_flip_rotate_augmentation(basic_model_params, create_dummy_input):
    """Test that flip_rotate augmentation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Set a fixed seed for deterministic augmentation
    torch.manual_seed(42)
    
    # Apply augmentation in training mode
    model.train()
    augmented = model.flip_rotate(x)
    
    # Check output shape (should be unchanged)
    assert augmented.shape == x.shape
    
    # Verify that augmentation did something (tensors should be different)
    assert not torch.allclose(augmented, x)
    
    # Verify that in eval mode, no augmentation happens
    model.eval()
    no_aug = model.flip_rotate(x)
    assert torch.allclose(no_aug, x)

# Test data augmentation: random_noise
def test_random_noise_augmentation(basic_model_params, create_dummy_input):
    """Test that random_noise augmentation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Generate masks for noise application
    masks = model.generate_masks(x)
    
    # Set a fixed seed for deterministic noise
    torch.manual_seed(42)
    
    # Apply noise in training mode
    model.train()
    noisy = model.random_noise(x, masks)
    
    # Check output shape (should be unchanged)
    assert noisy.shape == x.shape
    
    # Verify that noise was applied (tensors should be different)
    assert not torch.allclose(noisy, x)
    
    # Verify that in eval mode or with noise_aug=0, no noise is added
    params = basic_model_params.copy()
    params["noise_aug"] = 0.0
    model_no_noise = RiskFormer_ViT(**params)
    model_no_noise.train()
    
    no_noise = model_no_noise.random_noise(x, masks)
    assert torch.allclose(no_noise, x)
    
    model.eval()
    no_noise_eval = model.random_noise(x, masks)
    assert torch.allclose(no_noise_eval, x)

# Test token preparation
def test_prepare_tokens(basic_model_params, create_dummy_input):
    """Test that token preparation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Test in evaluation mode
    model.eval()
    tokens, masks = model.prepare_tokens(x)
    
    # Check that tokens have the right shape
    batch_size, height, width, channels = x.shape
    expected_seq_len = height * width
    if model.use_class_token:
        expected_seq_len += 1
    assert tokens.shape == (batch_size, expected_seq_len, model.model_dim)
    
    # Test with different configuration (no class token)
    params = basic_model_params.copy()
    params["use_class_token"] = False
    model_no_cls = RiskFormer_ViT(**params)
    
    model_no_cls.eval()
    tokens_no_cls, masks_no_cls = model_no_cls.prepare_tokens(x)
    
    # Check that tokens have the right shape (no class token)
    expected_seq_len = height * width
    assert tokens_no_cls.shape == (batch_size, expected_seq_len, model_no_cls.model_dim)

# Test Sinusoidal Positional Encoding
def test_sinusoidal_positional_encoding():
    """Test that sinusoidal positional encoding works correctly."""
    # Create encoder
    channels = 128
    height = width = 16
    encoder = SinusoidalPositionalEncoding2D(channels, height, width)
    
    # Create input tensor [B, H*W, C]
    batch_size = 2
    seq_len = height * width
    x = torch.randn(batch_size, seq_len, channels)
    
    # Apply encoding
    encoded = encoder(x)
    
    # Check shape is preserved
    assert encoded.shape == x.shape
    
    # Also test with class token
    x_with_cls = torch.randn(batch_size, seq_len + 1, channels)
    encoded_with_cls = encoder(x_with_cls)
    assert encoded_with_cls.shape == x_with_cls.shape

# Test model with phi network
def test_phi_network(basic_model_params, create_dummy_input):
    """Test that the phi network works correctly."""
    # Model with phi
    model_with_phi = RiskFormer_ViT(**basic_model_params)
    assert hasattr(model_with_phi, 'phi')
    
    # Model without phi
    params = basic_model_params.copy()
    params["use_phi"] = False
    model_without_phi = RiskFormer_ViT(**params)
    assert not hasattr(model_without_phi, 'phi')
    
    # Test forward pass with phi
    x = create_dummy_input
    model_with_phi.eval()
    with torch.no_grad():
        output_with_phi = model_with_phi(x)
    
    # Should produce valid output
    assert output_with_phi.shape == (x.shape[0], basic_model_params["num_classes"])

# Test global attention and weights
def test_global_attention(basic_model_params, create_dummy_input):
    """Test that global attention and weight calculation work correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    model.eval()
    with torch.no_grad():
        # Get outputs with attention weights
        output, attns, global_weights = model(x, return_weights=True)
    
    # Check output shape
    assert output.shape == (x.shape[0], basic_model_params["num_classes"])
    
    # Check that attention maps have appropriate shapes
    assert attns is not None
    assert global_weights is not None
    
    # The exact shape depends on the internal structure, but they should be defined
    # Global weights should have a spatial dimension matching the input
    assert len(global_weights.shape) == 3  # [batch, height, width]

if __name__ == "__main__":
    pytest.main() 