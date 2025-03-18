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
        "tasks": {
            "risk": {
                "type": "binary",
                "num_classes": 1,
                "weight": 1.0,
                "activation": "sigmoid"
            }
        },
        "max_dim": 16,
        "depth": 2,
        "global_depth": 1,
        "encoding_method": "standard",
        "num_heads": 4,
        "use_attn_mask": True,
        "mlp_ratio": 2.0,
        "use_class_token": False,
        "attn_global_hidden_dim": 128,
        "phi_dim": 64,
        "downscale_depth": 1,
        "downscale_multiplier": 1.25,
        "downscale_stride_q": 2,
        "downscale_stride_k": 2,
        "noise_aug": 0.1,
        "attnpool_mode": "conv",
        "name": None,
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rotate_prob": 0.5,
        "noise_aug_prob": 0.5
    }

@pytest.fixture
def create_dummy_input(batch_size, input_size, embedding_dim):
    """Create a dummy input tensor with non-zero values."""
    shape = (batch_size, embedding_dim, input_size, input_size)
    # Create tensor with small non-zero values to ensure masks work correctly
    x = torch.ones(shape) * 0.1
    # Add some larger values to simulate features
    x[:, :, input_size//4:input_size//2, input_size//4:input_size//2] = 1.0
    return x

# Test basic model initialization
def test_model_initialization(basic_model_params):
    """Test that the model initializes without errors."""
    model = RiskFormer_ViT(**basic_model_params)
    assert isinstance(model, RiskFormer_ViT)
    assert "risk" in model.tasks
    assert model.tasks["risk"]["type"] == "binary"
    assert model.use_phi == basic_model_params["use_phi"]
    assert model.use_class_token == basic_model_params["use_class_token"]

# Test model with different position encoding methods
@pytest.mark.parametrize("encoding_method", ["standard", "sinusoidal"])
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
    
    # Check output format - should be a dictionary
    assert isinstance(output, dict)
    assert "risk" in output
    
    # Get the risk output
    risk_output = output["risk"]
    
    # Check output shape - for binary task with batch size samples
    assert risk_output.shape[0] > 0  # At least one prediction (global + instances)
    assert risk_output.shape[1] == 1  # Binary has one output node

# Test mask generation
def test_mask_generation(basic_model_params, create_dummy_input):
    """Test that mask generation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Generate masks
    masks = model.generate_masks(x)
    
    # Check mask shape and values
    assert masks.shape == (x.shape[0], x.shape[2], x.shape[3])
    assert masks.dtype == torch.bool

# Test token augmentation
def test_apply_token_augment(basic_model_params, create_dummy_input):
    """Test that token augmentation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Set a fixed seed for deterministic augmentation
    torch.manual_seed(42)
    
    # Apply augmentation in training mode
    model.train()
    augmented = model.apply_token_augment(x)
    
    # Check output shape (should be unchanged)
    assert augmented.shape == x.shape
    
    # Verify that augmentation did something (tensors should be different)
    assert not torch.allclose(augmented, x)
    
    # Verify that in eval mode, no augmentation happens
    model.eval()
    no_aug = model.apply_token_augment(x)
    assert torch.allclose(no_aug, x)

# Test data augmentation: random_noise
def test_apply_noise(basic_model_params, create_dummy_input):
    """Test that noise augmentation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Set a fixed seed for deterministic noise
    torch.manual_seed(42)
    
    # Apply noise in training mode with specified level
    noisy = model.apply_noise(x, noise_level=0.1)
    
    # Check output shape (should be unchanged)
    assert noisy.shape == x.shape
    
    # Verify that noise was applied (tensors should be different)
    assert not torch.allclose(noisy, x)
    
    # Verify that with noise_level=0, no noise is added
    no_noise = model.apply_noise(x, noise_level=0)
    assert torch.allclose(no_noise, x)

# Test token preparation
def test_prepare_tokens(basic_model_params, create_dummy_input):
    """Test that token preparation works correctly."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Test in evaluation mode
    model.eval()
    tokens, attn_mask, hw_shape = model.prepare_tokens(x)
    
    # Check that tokens have the right shape
    batch_size = x.shape[0]
    height, width = x.shape[2], x.shape[3]
    expected_seq_len = height * width
    
    assert tokens.shape[0] == batch_size
    assert tokens.shape[1] == expected_seq_len
    assert tokens.shape[2] == model.blocks_input_dim
    
    # Check attention mask shape if used
    if model.use_attn_mask:
        assert attn_mask.shape == (batch_size, expected_seq_len, 1)

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
    assert model_with_phi.phi is not None
    
    # Model without phi
    params = basic_model_params.copy()
    params["use_phi"] = False
    model_without_phi = RiskFormer_ViT(**params)
    assert model_without_phi.phi is None
    
    # Test forward pass with phi
    x = create_dummy_input
    model_with_phi.eval()
    with torch.no_grad():
        output_with_phi = model_with_phi(x)
    
    # Should produce valid output
    assert isinstance(output_with_phi, dict)
    assert "risk" in output_with_phi

# Test returning weights in forward pass
def test_return_weights(basic_model_params, create_dummy_input):
    """Test that the model can return attention weights."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    model.eval()
    with torch.no_grad():
        output = model(x, return_weights=True)
    
    # Should return a tuple of (task_outputs, attns, global_weights)
    assert isinstance(output, tuple)
    assert len(output) == 3
    
    # First element should be the task outputs dictionary
    task_outputs, attns, global_weights = output
    assert isinstance(task_outputs, dict)
    assert "risk" in task_outputs

# Test random rotation
def test_random_rotate(basic_model_params, create_dummy_input):
    """Test the random_rotate function."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Set a fixed seed for deterministic rotation
    torch.manual_seed(42)
    
    # Apply rotation with angles [1, 2, 3]
    rotated = model.random_rotate(x, angles=[1, 2, 3])
    
    # Check output shape (should be unchanged)
    assert rotated.shape == x.shape
    
    # Verify rotation does something
    assert not torch.allclose(rotated, x)

if __name__ == "__main__":
    pytest.main() 