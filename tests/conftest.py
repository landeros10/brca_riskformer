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