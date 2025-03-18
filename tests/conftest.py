'''
Test configuration and fixtures for preprocessing tests.
'''
import os
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
import torch
import torch.nn as nn


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_output_dir(temp_dir):
    """Create a temporary output directory."""
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.fixture
def mock_model_dir(temp_dir):
    """Create a temporary model directory."""
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dummy model files
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        f.write(b"dummy model file")
    
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        f.write('{"model_type": "resnet18"}')
    
    return model_dir

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "model_type": "resnet18",
        "foreground_config_path": "./resources/foreground_config.json",
        "foreground_cleanup_config_path": "./resources/foreground_cleanup_config.json",
        "tiling_config_path": "./resources/tiling_config.json",
        "num_workers": 32,
        "batch_size": 256,
        "prefetch_factor": 4,
    }

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
def base_model_config():
    """Standard model configuration that can be customized per test."""
    return {
        "input_embed_dim": 128,
        "output_embed_dim": 128,
        "use_phi": True,
        "phi_dim": 64,
        "drop_path_rate": 0.1,
        "drop_rate": 0.1,
        "max_dim": 16,
        "depth": 2,
        "global_depth": 1,
        "encoding_method": "standard",
        "num_heads": 4,
        "use_attn_mask": True,
        "mlp_ratio": 2.0,
        "use_class_token": False,
        "attn_global_hidden_dim": 128,
        "downscale_depth": 1,
        "downscale_multiplier": 1.25,
        "downscale_stride_q": 2,
        "downscale_stride_k": 2,
        "noise_aug": 0.1,
        "attnpool_mode": "conv",
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rotate_prob": 0.5,
        "noise_aug_prob": 0.5,
        "name": None,
    }

@pytest.fixture
def tasks_config():
    """Standard tasks configuration."""
    return {
        "risk": {
            "type": "binary",
            "num_classes": 1,
            "weight": 1.0,
            "loss_fn": nn.BCEWithLogitsLoss(),
            "activation": "sigmoid"
        }
    }

@pytest.fixture
def multitask_config():
    """Multi-task configuration for tests that need it."""
    return {
        "binary_task": {
            "type": "binary",
            "num_classes": 1,
            "weight": 1.0,
            "loss_fn": nn.BCEWithLogitsLoss(),
            "activation": "sigmoid"
        },
        "regression_task": {
            "type": "regression",
            "num_classes": 1,
            "weight": 0.5,
            "loss_fn": nn.MSELoss(),
            "activation": "linear"
        },
        "multiclass_task": {
            "type": "multiclass",
            "num_classes": 3,
            "weight": 0.75,
            "loss_fn": nn.CrossEntropyLoss(),
            "activation": "softmax"
        }
    }

@pytest.fixture
def model_config(base_model_config, tasks_config):
    """Complete model configuration with tasks."""
    config = base_model_config.copy()
    config["tasks"] = tasks_config
    return config

@pytest.fixture
def multitask_model_config(base_model_config, multitask_config):
    """Complete model configuration with multiple tasks."""
    config = base_model_config.copy()
    config["tasks"] = multitask_config
    return config

@pytest.fixture
def optimizer_config():
    """Standard optimizer configuration for lightning tests."""
    return {
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "scheduler": "plateau",
        "patience": 5,
        "learning_rate_scaling": "linear",
        "learning_rate_warmup_epochs": 10,
    }

@pytest.fixture
def create_dummy_input(batch_size, input_size, embedding_dim):
    """Create a dummy input tensor with realistic spatial patterns."""
    shape = (batch_size, embedding_dim, input_size, input_size)
    # Create tensor with small non-zero values
    x = torch.ones(shape) * 0.01
    
    # Add some structure - create foreground regions with higher values
    for b in range(batch_size):
        # Create a random number of regions (2-5)
        num_regions = np.random.randint(2, 6)
        for _ in range(num_regions):
            # Random region parameters
            region_h = np.random.randint(4, 8)  # Region height
            region_w = np.random.randint(4, 8)  # Region width
            pos_h = np.random.randint(0, input_size - region_h)  # Region position
            pos_w = np.random.randint(0, input_size - region_w)
            
            # Set region values
            x[b, :, pos_h:pos_h+region_h, pos_w:pos_w+region_w] = 0.8
    
    return x

@pytest.fixture
def mock_batch(batch_size, embedding_dim, input_size):
    """Create a mock batch with features and labels for different tasks."""
    # Features (B, C, H, W)
    features = torch.rand(batch_size, embedding_dim, input_size, input_size)
    
    # Labels for different tasks with the expected 'labels' key
    metadata = {
        'labels': {
            'binary_task': torch.tensor([1.0, 0.0] * (batch_size//2), dtype=torch.float32).reshape(batch_size, 1),
            'regression_task': torch.tensor([42.5, 35.8] * (batch_size//2), dtype=torch.float32).reshape(batch_size, 1),
            'multiclass_task': torch.tensor([2, 1] * (batch_size//2), dtype=torch.long)
        }
    }
    
    return features, metadata