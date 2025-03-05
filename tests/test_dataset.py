import os
import pytest
import json
import torch
import tempfile
from pathlib import Path

from riskformer.data.datasets import RiskFormerDataset, create_riskformer_dataset

@pytest.fixture
def mock_metadata_file():
    """Create a temporary metadata file for testing"""
    metadata = {
        "TCGA-GM-A2DM-01Z-00-DX1.652038F4-C370-40EB-A545-51062783C74C": {
            "odx85": "H",
            "age": 45,
            "stage": "II"
        },
        "TCGA-E9-A3QA-01Z-00-DX1.9D664AF3-9ABD-4EED-B826-4C4FBFC33F3E": {
            "odx85": "L",
            "age": 62,
            "stage": "I"
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f)
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_s3_paths():
    """Mock S3 paths for testing"""
    return [
        ("s3://test-bucket/slide1_coords.h5", "s3://test-bucket/slide1_features.h5"),
        ("s3://test-bucket/slide2_coords.h5", "s3://test-bucket/slide2_features.h5")
    ]

def test_riskformer_dataset_init(mock_s3_paths):
    """Test RiskFormerDataset initialization"""
    # Test with minimal parameters
    dataset = RiskFormerDataset(mock_s3_paths)
    assert len(dataset) == 2
    assert dataset.feature_pairs == mock_s3_paths
    assert isinstance(dataset.s3_cache.cache_dir, Path)

    # Test with custom cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = RiskFormerDataset(mock_s3_paths, cache_dir=temp_dir)
        assert str(dataset.s3_cache.cache_dir) == temp_dir

def test_create_riskformer_dataset(mock_metadata_file):
    """Test create_riskformer_dataset function"""
    # Test with required parameters
    with pytest.raises(RuntimeError, match="Failed to initialize S3 client"):
        create_riskformer_dataset(
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            metadata_file=mock_metadata_file
        )

def test_riskformer_dataset_getitem_shape(mocker, mock_s3_paths):
    """Test the shape of tensors returned by __getitem__"""
    # Mock h5py file operations
    mock_coords = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long).T
    mock_features = torch.randn(3, 256)  # Assuming 256-dim features
    
    mock_h5py = mocker.patch('h5py.File')
    mock_h5py.return_value.__enter__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.side_effect = [
        mocker.MagicMock(return_value=mock_coords.numpy()),
        mocker.MagicMock(return_value=mock_features.numpy())
    ]

    dataset = RiskFormerDataset(mock_s3_paths)
    tensor = dataset[0]
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape == (3, 256)  # (num_points, feature_dim)

def test_riskformer_dataset_invalid_paths():
    """Test RiskFormerDataset with invalid paths"""
    invalid_pairs = [
        ("invalid_path_coords.h5", "invalid_path_features.h5")
    ]
    
    with pytest.raises(Exception):
        RiskFormerDataset(invalid_pairs)

def test_riskformer_dataset_empty_pairs():
    """Test RiskFormerDataset with empty feature pairs"""
    with pytest.raises(ValueError):
        RiskFormerDataset([]) 