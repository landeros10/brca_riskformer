import pytest
import torch
import math
import numpy as np
import torch.nn as nn
from riskformer.training.model import RiskFormer_ViT, RiskFormer_Head
from riskformer.training.layers import SinusoidalPositionalEncoding2D
from unittest.mock import patch, MagicMock
from tests.test_utils import create_riskformer_vit_inputs

# Fixtures for commonly used test parameters
@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def input_size():
    return 16  # 16x16 patches

@pytest.fixture
def embedding_dim():
    return 128

@pytest.fixture
def input_embed_dim():
    return 16

@pytest.fixture
def output_embed_dim():
    return 64

@pytest.fixture
def channels():
    return 3

@pytest.fixture
def basic_tasks_config():
    return {
        "binary_task": {
            "type": "binary",
            "num_classes": 1,
            "activation": "sigmoid"
        }
    }

@pytest.fixture
def multi_tasks_config():
    return {
        "binary_task": {
            "type": "binary",
            "num_classes": 1,
            "activation": "sigmoid"
        },
        "regression_task": {
            "type": "regression",
            "num_classes": 1,
            "activation": None
        },
        "multiclass_task": {
            "type": "multiclass",
            "num_classes": 3,
            "activation": "softmax"
        }
    }

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

def test_initialization_with_phi(input_embed_dim, output_embed_dim, basic_tasks_config):
    """Test initialization with phi network."""
    # Create model with phi
    phi_dim = 32
    model = RiskFormer_ViT(
        input_embed_dim=input_embed_dim,
        output_embed_dim=output_embed_dim,
        use_phi=True,
        drop_path_rate=0.0,
        drop_rate=0.0,
        tasks=basic_tasks_config,
        max_dim=16,
        depth=2,
        global_depth=1,
        encoding_method="sinusoidal",
        num_heads=2,
        use_attn_mask=False,
        mlp_ratio=4.0,
        use_class_token=False,
        attn_global_hidden_dim=32,
        phi_dim=phi_dim
    )
    
    # Check phi network
    assert hasattr(model, 'phi')
    assert model.blocks_input_dim == phi_dim

def test_initialization_with_class_token(input_embed_dim, output_embed_dim, basic_tasks_config):
    """Test initialization with class token."""
    model = RiskFormer_ViT(
        input_embed_dim=input_embed_dim,
        output_embed_dim=output_embed_dim,
        use_phi=False,
        drop_path_rate=0.0,
        drop_rate=0.0,
        tasks=basic_tasks_config,
        max_dim=16,
        depth=2,
        global_depth=1,
        encoding_method="sinusoidal",
        num_heads=2,
        use_attn_mask=False,
        mlp_ratio=4.0,
        use_class_token=True,
        attn_global_hidden_dim=32
    )
    
    assert hasattr(model, 'cls_token')
    assert model.use_class_token is True
    assert model.cls_token.shape == (1, 1, output_embed_dim)

def test_initialization_with_downscaling(input_embed_dim, output_embed_dim, basic_tasks_config):
    """Test initialization with downscaling."""
    model = RiskFormer_ViT(
        input_embed_dim=input_embed_dim,
        output_embed_dim=output_embed_dim,
        use_phi=False,
        drop_path_rate=0.0,
        drop_rate=0.0,
        tasks=basic_tasks_config,
        max_dim=16,
        depth=2,
        global_depth=1,
        encoding_method="sinusoidal",
        num_heads=2,
        use_attn_mask=False,
        mlp_ratio=4.0,
        use_class_token=False,
        attn_global_hidden_dim=32,
        downscale_depth=2,
        downscale_multiplier=1.25,
        downscale_stride_q=2,
        downscale_stride_k=2
    )
    
    assert hasattr(model, 'downscale_blocks')
    assert len(model.downscale_blocks) == 2
    assert model.blocks_output_dim > model.blocks_input_dim

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

def test_forward_multi_task(batch_size, input_size, channels, input_embed_dim, output_embed_dim, multi_tasks_config):
    """Test forward pass with multiple tasks."""
    model = RiskFormer_ViT(
        input_embed_dim=input_embed_dim,
        output_embed_dim=output_embed_dim,
        use_phi=False,
        drop_path_rate=0.0,
        drop_rate=0.0,
        tasks=multi_tasks_config,
        max_dim=input_size,
        depth=1,
        global_depth=1,
        encoding_method="sinusoidal",
        num_heads=2,
        use_attn_mask=False,
        mlp_ratio=4.0,
        use_class_token=False,
        attn_global_hidden_dim=32
    )
    
    # Generate input tensors
    inputs = create_riskformer_vit_inputs(
        batch_size=batch_size,
        input_size=input_size,
        channels=channels
    )
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Check outputs for each task
    assert "binary_task" in outputs
    assert "regression_task" in outputs
    assert "multiclass_task" in outputs
    
    # Check shapes
    assert outputs["binary_task"].shape == (batch_size, 1)
    assert outputs["regression_task"].shape == (batch_size, 1)
    assert outputs["multiclass_task"].shape == (batch_size, 3)

def test_mask_generation(basic_model_params, create_dummy_input):
    """Test attention mask generation."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Generate mask
    mask = model._generate_attention_mask(x)
    
    # Check mask shape
    expected_shape = (x.shape[0], x.shape[2] * x.shape[3], x.shape[2] * x.shape[3])
    assert mask.shape == expected_shape
    
    # Check mask values
    assert torch.all(mask >= 0)  # All values should be non-negative
    assert torch.all(mask <= 1)  # All values should be <= 1

def test_apply_token_augment(basic_model_params, create_dummy_input):
    """Test token augmentation."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Apply augmentation
    augmented = model._apply_token_augment(x)
    
    # Check shape
    assert augmented.shape == x.shape
    
    # Check that some values have changed
    assert not torch.allclose(augmented, x)

def test_apply_noise(basic_model_params, create_dummy_input):
    """Test noise augmentation."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Apply noise
    noisy = model._apply_noise(x)
    
    # Check shape
    assert noisy.shape == x.shape
    
    # Check that some values have changed
    assert not torch.allclose(noisy, x)

def test_prepare_tokens(basic_model_params, create_dummy_input):
    """Test token preparation."""
    model = RiskFormer_ViT(**basic_model_params)
    x = create_dummy_input
    
    # Prepare tokens
    tokens = model._prepare_tokens(x)
    
    # Check shape
    expected_shape = (x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    assert tokens.shape == expected_shape

def test_sinusoidal_positional_encoding(embedding_dim):
    """Test sinusoidal positional encoding."""
    # Create encoding layer
    encoding = SinusoidalPositionalEncoding2D(embedding_dim)
    
    # Create dummy input
    batch_size = 2
    height = 4
    width = 4
    x = torch.randn(batch_size, embedding_dim, height, width)
    
    # Apply encoding
    encoded = encoding(x)
    
    # Check shape
    assert encoded.shape == x.shape
    
    # Check that values have changed
    assert not torch.allclose(encoded, x)

class TestRiskFormerHead:
    """Unit tests for the RiskFormer_Head class."""
    
    @pytest.fixture
    def embed_dim(self):
        return 64
    
    @pytest.fixture
    def tasks_config(self):
        return {
            "binary_task": {
                "type": "binary",
                "num_classes": 1,
                "activation": "sigmoid"
            },
            "multiclass_task": {
                "type": "multiclass",
                "num_classes": 3,
                "activation": "softmax"
            }
        }
    
    @pytest.fixture
    def head_instance(self, tasks_config, embed_dim):
        return RiskFormer_Head(tasks_config, embed_dim)
    
    def test_initialization(self, head_instance, tasks_config, embed_dim):
        """Test head initialization."""
        assert isinstance(head_instance, RiskFormer_Head)
        assert head_instance.embed_dim == embed_dim
        assert head_instance.tasks == tasks_config
        
        # Check task heads
        for task_name, task_config in tasks_config.items():
            assert task_name in head_instance.task_heads
            assert isinstance(head_instance.task_heads[task_name], nn.Linear)
            assert head_instance.task_heads[task_name].in_features == embed_dim
            assert head_instance.task_heads[task_name].out_features == task_config["num_classes"]
    
    def test_forward(self, head_instance, tasks_config, batch_size, embed_dim):
        """Test forward pass."""
        # Create dummy input
        x = torch.randn(batch_size, embed_dim)
        
        # Forward pass
        outputs = head_instance(x)
        
        # Check outputs
        assert isinstance(outputs, dict)
        for task_name in tasks_config:
            assert task_name in outputs
            output = outputs[task_name]
            assert output.shape == (batch_size, tasks_config[task_name]["num_classes"])
    
    def test_get_task_output(self, head_instance, tasks_config, batch_size, embed_dim):
        """Test getting output for specific task."""
        # Create dummy input
        x = torch.randn(batch_size, embed_dim)
        
        # Get output for each task
        for task_name in tasks_config:
            output = head_instance.get_task_output(x, task_name)
            assert output.shape == (batch_size, tasks_config[task_name]["num_classes"])
    
    def test_head_activation(self, head_instance):
        """Test activation functions."""
        # Test binary task activation
        binary_output = torch.randn(2, 1)
        activated = head_instance._apply_activation(binary_output, "sigmoid")
        assert activated.shape == binary_output.shape
        assert torch.all(activated >= 0) and torch.all(activated <= 1)
        
        # Test multiclass task activation
        multiclass_output = torch.randn(2, 3)
        activated = head_instance._apply_activation(multiclass_output, "softmax")
        assert activated.shape == multiclass_output.shape
        assert torch.all(activated >= 0) and torch.all(activated <= 1)
        assert torch.allclose(activated.sum(dim=1), torch.ones(2))
    
    def test_error_handling(self, head_instance, batch_size, embed_dim):
        """Test error handling."""
        # Test invalid task name
        x = torch.randn(batch_size, embed_dim)
        with pytest.raises(ValueError):
            head_instance.get_task_output(x, "invalid_task")
        
        # Test invalid activation type
        with pytest.raises(ValueError):
            head_instance._apply_activation(x, "invalid_activation")

if __name__ == "__main__":
    pytest.main() 