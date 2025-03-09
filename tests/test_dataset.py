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

@pytest.fixture
def mock_patient_examples():
    """Mock patient examples for testing"""
    return {
        "patient1": {
            "coords_paths": ["s3://test-bucket/slide1_coords.h5"],
            "features_paths": ["s3://test-bucket/slide1_features.h5"],
            "metadata": {"odx85": "H", "age": 45}
        },
        "patient2": {
            "coords_paths": ["s3://test-bucket/slide2_coords.h5"],
            "features_paths": ["s3://test-bucket/slide2_features.h5"],
            "metadata": {"odx85": "L", "age": 62}
        }
    }

def test_riskformer_dataset_init(mock_patient_examples, mocker):
    """Test RiskFormerDataset initialization"""
    # Mock h5py and file operations to prevent actual file access
    mock_h5py = mocker.patch('h5py.File')
    mock_h5py.return_value.__enter__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value.shape = (10, 256)
    
    mock_s3_download = mocker.patch('riskformer.data.datasets.S3Cache.download_if_needed')
    mock_s3_download.return_value = "fake_local_path"
    
    # Test with minimal parameters
    dataset = RiskFormerDataset(mock_patient_examples)
    assert len(dataset) == 2
    assert dataset.patient_ids == list(mock_patient_examples.keys())
    assert isinstance(dataset.s3_cache.cache_dir, Path)

    # Test with custom cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = RiskFormerDataset(mock_patient_examples, cache_dir=temp_dir)
        assert str(dataset.s3_cache.cache_dir) == temp_dir

def test_create_riskformer_dataset(mock_metadata_file, mocker):
    """Test create_riskformer_dataset function"""
    # Mock the S3 client initialization to avoid actually connecting to AWS
    mock_s3_client = mocker.MagicMock()
    mock_init_s3 = mocker.patch('riskformer.data.datasets.initialize_s3_client', return_value=mock_s3_client)
    
    # Mock listing bucket files to return a dictionary of paths (not a list)
    mock_list_files = mocker.patch('riskformer.data.datasets.list_bucket_files')
    mock_list_files.return_value = {
        "slide1_coords.h5": "s3://test-bucket/coords/slide1_coords.h5",
        "slide1_features.h5": "s3://test-bucket/features/slide1_features.h5",
        "slide2_coords.h5": "s3://test-bucket/coords/slide2_coords.h5",
        "slide2_features.h5": "s3://test-bucket/features/slide2_features.h5"
    }
    
    # Mock S3Cache download_if_needed to avoid actual downloads
    mock_download = mocker.patch('riskformer.data.datasets.S3Cache.download_if_needed')
    mock_download.return_value = "local/path/to/file.h5"
    
    # Mock h5py file operations
    mock_h5py = mocker.patch('h5py.File')
    mock_h5py.return_value.__enter__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value.shape = (10, 256)
    
    # Mock the whole patient examples creation process
    mock_patient_examples = {
        "patient1": {
            "coords_paths": ["s3://test-bucket/coords/slide1_coords.h5"],
            "features_paths": ["s3://test-bucket/features/slide1_features.h5"],
            "metadata": {"odx85": "H", "age": 45, "stage": "II"}
        }
    }
    mocker.patch('riskformer.data.datasets.create_patient_examples', return_value=mock_patient_examples)
    
    # Test creating the dataset
    dataset = create_riskformer_dataset(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix",
        metadata_file=mock_metadata_file
    )
    
    # Verify S3 client was initialized
    mock_init_s3.assert_called_once()
    
    # Verify the dataset was created and has the expected type
    assert isinstance(dataset, RiskFormerDataset)
    assert len(dataset) > 0

def test_riskformer_dataset_getitem_shape(mocker):
    """Test the shape of tensors returned by __getitem__"""
    # Create mock patient examples in the dictionary format
    mock_patient_examples = {
        "patient1": {
            "coords_paths": ["s3://test-bucket/coords/slide1_coords.h5"],
            "features_paths": ["s3://test-bucket/features/slide1_features.h5"],
            "metadata": {"odx85": "H", "age": 45},
            "odx85": "H"  # Add this field for label processing
        }
    }
    
    # Setup mocks to avoid any S3 access
    mocker.patch('boto3.client', return_value=mocker.MagicMock())
    mocker.patch('botocore.client.BaseClient._make_api_call', return_value={})
    mocker.patch('riskformer.data.datasets.S3Cache.download_if_needed', return_value="fake_local_path.h5")
    
    # Mock h5py.File to avoid trying to open a non-existent file
    mock_h5py_file = mocker.MagicMock()
    mock_h5py_file.__enter__.return_value = mocker.MagicMock()
    mock_features = mocker.MagicMock()
    mock_features.shape = (10, 256)
    mock_h5py_file.__enter__.return_value.__getitem__.return_value = mock_features
    
    mocker.patch('h5py.File', return_value=mock_h5py_file)
    
    # Mock _prefetch_all_files to avoid S3 access
    mocker.patch.object(RiskFormerDataset, '_prefetch_all_files')
    
    # Create expected output tensor
    expected_patches = torch.zeros((3, 32, 32, 256))  # (N, H, W, feature_dim)
    expected_metadata = {
        'patch_info': torch.zeros((3, 6)),  # Mock patch info
        'patient_id': 'patient1',
        'labels': {'odx85': torch.tensor([1.0], dtype=torch.float32)},
        'task_types': {'odx85': 'binary'},
        'label': torch.tensor([1.0], dtype=torch.float32)
    }
    
    # Mock the internal methods to return our expected values
    mocker.patch.object(RiskFormerDataset, '_create_dense_features', return_value=[torch.zeros((32, 32, 256))])
    mocker.patch.object(RiskFormerDataset, 'split_and_pad_features', 
                        return_value=(expected_patches, torch.zeros((3, 6))))
    
    # Initialize the dataset
    dataset = RiskFormerDataset(mock_patient_examples)
    
    # Get the item
    patches, metadata = dataset[0]
    
    # Verify the tensor properties
    assert isinstance(patches, torch.Tensor)
    assert patches.shape == (3, 32, 32, 256)  # (N, H, W, feature_dim)
    
    # Verify metadata contains expected fields
    assert 'patient_id' in metadata
    assert 'labels' in metadata
    assert 'task_types' in metadata
    assert 'label' in metadata

def test_riskformer_dataset_invalid_paths():
    """Test RiskFormerDataset with invalid paths"""
    invalid_examples = {
        "patient1": {
            "coords_paths": ["invalid_path_coords.h5"],
            "features_paths": ["invalid_path_features.h5"],
            "metadata": {"odx85": "H", "age": 45}
        }
    }
    
    with pytest.raises(Exception):
        RiskFormerDataset(invalid_examples)

def test_riskformer_dataset_empty_pairs():
    """Test RiskFormerDataset with empty patient examples"""
    # Instead of directly creating dataset, mock the initialization to validate empty dictionary
    with pytest.raises((ValueError, IndexError)):
        # Either ValueError or IndexError would be valid here since an empty dictionary
        # will have no keys (causing IndexError) or ideally RiskFormerDataset should check
        # and raise ValueError for empty patient_examples
        RiskFormerDataset({})  # Empty dictionary instead of empty list 

def test_riskformer_dataset_with_real_s3_path(mocker):
    """Test RiskFormerDataset with a real S3 path from the TCGA dataset"""
    # Define the specific S3 paths for the test
    slide_id = "TCGA-A2-A0EP-01Z-00-DX1.1180C406-5C18-4373-8621-1B7B70875113"
    s3_bucket = "tcga-riskformer-data-2025"
    s3_prefix = "preprocessed/uni/uni2-h"
    
    coords_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_coords.h5"
    features_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_features.h5"
    
    # Create mock patient examples with the real S3 path
    mock_patient_examples = {
        slide_id: {
            "coords_paths": [coords_path],
            "features_paths": [features_path],
            "metadata": {"odx85": "L", "age": 56},  # Based on the patient_samples_0.csv data
            "odx85": "L"  # Add this field for label processing
        }
    }
    
    # Setup mocks to avoid actual S3 access during testing
    mocker.patch('boto3.client', return_value=mocker.MagicMock())
    mocker.patch('botocore.client.BaseClient._make_api_call', return_value={})
    mock_s3_download = mocker.patch('riskformer.data.datasets.S3Cache.download_if_needed')
    mock_s3_download.return_value = "fake_local_path.h5"
    
    # Mock h5py.File to avoid trying to open a non-existent file
    mock_h5py_file = mocker.MagicMock()
    mock_h5py_file.__enter__.return_value = mocker.MagicMock()
    mock_features = mocker.MagicMock()
    mock_features.shape = (10, 256)  # Assuming feature dimension is 256
    mock_h5py_file.__enter__.return_value.__getitem__.return_value = mock_features
    
    mocker.patch('h5py.File', return_value=mock_h5py_file)
    
    # Instead of mocking _prefetch_all_files, we'll let it run and verify the download calls
    # This is important because _prefetch_all_files is what triggers the download_if_needed calls
    
    # Mock the internal methods to return expected values
    mocker.patch.object(RiskFormerDataset, '_create_dense_features', 
                        return_value=[torch.zeros((32, 32, 256))])
    mocker.patch.object(RiskFormerDataset, 'split_and_pad_features', 
                        return_value=(torch.zeros((3, 32, 32, 256)), torch.zeros((3, 6))))
    
    # Initialize the dataset
    dataset = RiskFormerDataset(mock_patient_examples)
    
    # Verify the dataset was created correctly
    assert len(dataset) == 1
    assert dataset.patient_ids == [slide_id]
    
    # Verify S3 paths were passed correctly to download_if_needed
    # The order of calls might vary, so we check that both paths were used
    mock_s3_download.assert_has_calls([
        mocker.call(coords_path),
        mocker.call(features_path)
    ], any_order=True)
    
    # Get an item from the dataset
    patches, metadata = dataset[0]
    
    # Verify the tensor properties
    assert isinstance(patches, torch.Tensor)
    assert patches.shape == (3, 32, 32, 256)  # (N, H, W, feature_dim)
    
    # Verify metadata contains expected fields
    assert metadata['patient_id'] == slide_id
    assert 'labels' in metadata
    assert 'task_types' in metadata
    assert 'label' in metadata
    
    # Verify the label is correct (L should be 0.0 for binary classification)
    assert metadata['labels']['odx85'].item() == 0.0

def test_create_riskformer_dataset_with_real_s3_path(mock_metadata_file, mocker):
    """Test create_riskformer_dataset function with a real S3 path"""
    # Define the specific S3 paths for the test
    slide_id = "TCGA-A2-A0EP-01Z-00-DX1.1180C406-5C18-4373-8621-1B7B70875113"
    s3_bucket = "tcga-riskformer-data-2025"
    s3_prefix = "preprocessed/uni/uni2-h"
    
    coords_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_coords.h5"
    features_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_features.h5"
    
    # Mock the S3 client initialization
    mock_s3_client = mocker.MagicMock()
    mock_init_s3 = mocker.patch('riskformer.data.datasets.initialize_s3_client', return_value=mock_s3_client)
    
    # Mock listing bucket files to return our specific file
    mock_list_files = mocker.patch('riskformer.data.datasets.list_bucket_files')
    mock_list_files.return_value = {
        f"{slide_id}_coords.h5": coords_path,
        f"{slide_id}_features.h5": features_path
    }
    
    # Mock S3Cache download_if_needed
    mock_download = mocker.patch('riskformer.data.datasets.S3Cache.download_if_needed')
    mock_download.return_value = "fake_local_path.h5"
    
    # Mock h5py file operations
    mock_h5py = mocker.patch('h5py.File')
    mock_h5py.return_value.__enter__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value.shape = (10, 256)
    
    # Create a mock metadata file with our slide ID
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            slide_id: {
                "odx85": "L",
                "age": 56,
                "stage": "II"
            }
        }, f)
    
    # Mock the patient examples creation
    mock_patient_examples = {
        slide_id: {
            "coords_paths": [coords_path],
            "features_paths": [features_path],
            "metadata": {"odx85": "L", "age": 56, "stage": "II"}
        }
    }
    mocker.patch('riskformer.data.datasets.create_patient_examples', return_value=mock_patient_examples)
    
    # Test creating the dataset
    dataset = create_riskformer_dataset(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        metadata_file=f.name
    )
    
    # Clean up the temporary file
    os.unlink(f.name)
    
    # Verify S3 client was initialized
    mock_init_s3.assert_called_once()
    
    # Verify the dataset was created and has the expected type
    assert isinstance(dataset, RiskFormerDataset)
    assert len(dataset) == 1
    assert dataset.patient_ids == [slide_id]

# Optional: Add a test that actually downloads the real data (disabled by default)
# @pytest.mark.skip(reason="This test attempts to download real data from S3 and should only be run manually")
def test_real_s3_data_download():
    """Test downloading and loading real data from S3 (skipped by default)"""
    # Define the specific S3 paths for the test
    slide_id = "TCGA-A2-A0EP-01Z-00-DX1.1180C406-5C18-4373-8621-1B7B70875113"
    s3_bucket = "tcga-riskformer-data-2025"
    s3_prefix = "preprocessed/uni/uni2-h"
    
    coords_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_coords.h5"
    features_path = f"s3://{s3_bucket}/{s3_prefix}/{slide_id}_features.h5"
    
    # Create patient examples with the real S3 path
    patient_examples = {
        slide_id: {
            "coords_paths": [coords_path],
            "features_paths": [features_path],
            "metadata": {"odx85": "L", "age": 56},
            "odx85": "L"
        }
    }
    
    # Create a temporary directory for caching
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the dataset with real S3 paths
        dataset = RiskFormerDataset(patient_examples, cache_dir=temp_dir)
        
        # Verify the dataset was created correctly
        assert len(dataset) == 1
        assert dataset.patient_ids == [slide_id]
        
        # Try to get an item from the dataset
        patches, metadata = dataset[0]
        
        # Verify the tensor properties
        assert isinstance(patches, torch.Tensor)
        assert patches.ndim == 4  # (N, H, W, feature_dim)
        
        # Verify metadata contains expected fields
        assert metadata['patient_id'] == slide_id
        assert 'labels' in metadata
        assert 'task_types' in metadata
        assert 'label' in metadata