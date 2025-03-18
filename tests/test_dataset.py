import os
import pytest
import json
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch
import boto3
import botocore

from riskformer.data.datasets import RiskFormerDataset, create_riskformer_dataset

# Set this to True to skip all AWS-dependent tests regardless of credentials
SKIP_AWS_TESTS = True

def is_aws_credentials_available():
    """Check if AWS credentials are available."""
    # Always skip if SKIP_AWS_TESTS is True
    if SKIP_AWS_TESTS:
        return False
        
    try:
        # Try to access AWS with a short timeout
        sts_client = boto3.client('sts', config=boto3.config.Config(connect_timeout=5, retries={'max_attempts': 1}))
        sts_client.get_caller_identity()
        return True
    except Exception:  # Catch all exceptions, not just specific ones
        return False

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
            "metadata": {
                "odx85": "H", 
                "age": 45,
                "ER_Status_By_IHC": "positive",
                "pr_status_by_ihc": "positive",
                "HER2Calc": "negative",
                "Necrosis": "Absent",
                "Lymphovascular Invasion (LVI)": "Absent",
                "Overall_Survival_Status": "alive"
            }
        },
        "patient2": {
            "coords_paths": ["s3://test-bucket/slide2_coords.h5"],
            "features_paths": ["s3://test-bucket/slide2_features.h5"],
            "metadata": {
                "odx85": "L", 
                "age": 62,
                "ER_Status_By_IHC": "negative",
                "pr_status_by_ihc": "negative",
                "HER2Calc": "positive",
                "Necrosis": "Present",
                "Lymphovascular Invasion (LVI)": "Present",
                "Overall_Survival_Status": "dead"
            }
        }
    }

@pytest.fixture
def mock_log_event(mocker):
    """Mock the log_event function to prevent MagicMock serialization issues"""
    return mocker.patch('riskformer.data.datasets.log_event')

def test_riskformer_dataset_init(mock_patient_examples, mocker):
    """Test RiskFormerDataset initialization"""
    # Mock h5py and file operations to prevent actual file access
    mock_h5py = mocker.patch('h5py.File')
    mock_h5py.return_value.__enter__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value = mocker.MagicMock()
    mock_h5py.return_value.__enter__.return_value.__getitem__.return_value.shape = (10, 256)
    
    # Create a separate mock for the S3Cache instance instead of patching the class
    mock_s3_instance = mocker.MagicMock()
    mock_s3_instance.get_local_path.return_value = Path("fake_local_path.h5")
    mock_s3_instance.download_if_needed.return_value = "fake_local_path.h5"
    
    # Test with minimal parameters
    dataset = RiskFormerDataset(mock_patient_examples, s3_cache=mock_s3_instance)
    assert len(dataset) == 2
    assert dataset.patient_ids == list(mock_patient_examples.keys())
    assert dataset.s3_cache == mock_s3_instance

    # Test with custom s3_cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real S3Cache instance
        from riskformer.data.datasets import S3Cache
        s3_cache = S3Cache(temp_dir)
        
        # Patch its methods to avoid actual S3 access
        real_get_local_path = s3_cache.get_local_path
        s3_cache.get_local_path = lambda s3_path: Path(temp_dir) / s3_path.split('/')[-1]
        
        # Patch the download method
        mocker.patch.object(S3Cache, 'download_if_needed', 
                            return_value=f"{temp_dir}/fake_file.h5")
        
        dataset = RiskFormerDataset(mock_patient_examples, s3_cache=s3_cache)
        assert dataset.s3_cache.cache_dir == Path(temp_dir)

@patch('riskformer.data.datasets.log_event')
@pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
def test_create_riskformer_dataset(mock_log, mock_metadata_file, mocker):
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
    
    # Create a mock S3Cache instance directly
    mock_s3_instance = mocker.MagicMock()
    mock_s3_instance.get_local_path.return_value = Path("fake/local/path.h5")
    mock_s3_instance.download_if_needed.return_value = "fake/local/path.h5"
    
    # Mock the S3Cache constructor to return our instance
    s3cache_mock = mocker.patch('riskformer.data.datasets.S3Cache', return_value=mock_s3_instance)
    
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
            "odx85": "H", 
            "age": 45, 
            "stage": "II"
        }
    }
    mocker.patch('riskformer.data.datasets.create_patient_examples', return_value=mock_patient_examples)
    
    # Also mock RiskFormerDataset constructor
    mock_dataset = mocker.MagicMock()
    mock_dataset_init = mocker.patch('riskformer.data.datasets.RiskFormerDataset', return_value=mock_dataset)
    
    # Test creating the dataset
    dataset = create_riskformer_dataset(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix",
        metadata_file=mock_metadata_file,
        cache_dir="/tmp/test_cache"
    )
    
    # Verify S3 client was initialized
    mock_init_s3.assert_called_once()
    
    # Simplify our verification - just check that the dataset was returned
    assert dataset == mock_dataset
    
    # Verify RiskFormerDataset was created
    mock_dataset_init.assert_called_once()

def test_riskformer_dataset_getitem_shape(mocker):
    """Test the shape of tensors returned by __getitem__"""
    # Create mock patient examples in the dictionary format
    mock_patient_examples = {
        "patient1": {
            "coords_paths": ["s3://test-bucket/coords/slide1_coords.h5"],
            "features_paths": ["s3://test-bucket/features/slide1_features.h5"],
            "odx85": "H", 
            "age": 45,
            "ER_Status_By_IHC": "positive",
            "pr_status_by_ihc": "positive",
            "HER2Calc": "negative"
        }
    }
    
    # Setup mocks to avoid any S3 access
    mocker.patch('boto3.client', return_value=mocker.MagicMock())
    mocker.patch('botocore.client.BaseClient._make_api_call', return_value={})
    
    # Create a mock S3Cache instance directly
    mock_s3_instance = mocker.MagicMock()
    mock_s3_instance.get_local_path.return_value = Path("fake_local_path.h5")
    mock_s3_instance.download_if_needed.return_value = "fake_local_path.h5"
    
    # Mock h5py.File to avoid trying to open a non-existent file
    mock_h5py_file = mocker.MagicMock()
    mock_h5py_file.__enter__.return_value = mocker.MagicMock()
    mock_features = mocker.MagicMock()
    mock_features.shape = (10, 256)
    mock_h5py_file.__enter__.return_value.__getitem__.return_value = mock_features
    
    mocker.patch('h5py.File', return_value=mock_h5py_file)
    
    # Mock the internal methods to return our expected values
    mocker.patch.object(RiskFormerDataset, '_create_dense_features', 
                        return_value=[torch.zeros((32, 32, 256))])
    
    # Create a more complex mock for split_and_pad_features
    patches = torch.zeros((3, 32, 32, 256))
    patches = patches.permute(0, 3, 1, 2)  # Convert to (B, C, H, W) format as the method would
    mocker.patch.object(RiskFormerDataset, 'split_and_pad_features', 
                        return_value=(patches, torch.zeros((3, 6))))
    
    # Initialize the dataset
    dataset = RiskFormerDataset(
        mock_patient_examples,
        s3_cache=mock_s3_instance,
        include_labels=["odx85", "age", "ER_Status_By_IHC", "pr_status_by_ihc", "HER2Calc"]
    )
    
    # Mock process_special_binary_fields, etc. to actually process the examples properly
    mocker.patch.object(RiskFormerDataset, 'process_special_binary_fields', 
                       side_effect=lambda patient_data, example_data: example_data['labels'].update({
                           'odx85': torch.tensor([1.0], dtype=torch.float32) # H -> 1.0
                       }))
    
    mocker.patch.object(RiskFormerDataset, 'process_binary_fields',
                       side_effect=lambda patient_data, example_data: example_data['labels'].update({
                           'er_status_by_ihc': torch.tensor([1.0], dtype=torch.float32),  # positive -> 1.0
                           'pr_status_by_ihc': torch.tensor([1.0], dtype=torch.float32),  # positive -> 1.0
                           'her2calc': torch.tensor([0.0], dtype=torch.float32)  # negative -> 0.0
                       }))
    
    mocker.patch.object(RiskFormerDataset, 'process_regression_fields',
                       side_effect=lambda patient_data, example_data: example_data['labels'].update({
                           'age': torch.tensor([45.0], dtype=torch.float32)
                       }))
    
    # Get the item
    patches, metadata = dataset[0]
    
    # Verify the tensor properties
    assert isinstance(patches, torch.Tensor)
    assert patches.ndim == 4  # Should have 4 dimensions - B, C, H, W
    assert patches.shape[0] == 3  # Should have 3 patches
    
    # Verify metadata contains expected fields
    assert 'patient_id' in metadata
    assert 'labels' in metadata
    assert 'odx85' in metadata['labels']
    assert 'age' in metadata['labels']
    assert 'er_status_by_ihc' in metadata['labels']  # Note lowercase field names
    assert 'pr_status_by_ihc' in metadata['labels']
    assert 'her2calc' in metadata['labels']

def test_riskformer_dataset_invalid_paths(mocker):
    """Test RiskFormerDataset with invalid paths"""
    invalid_examples = {
        "patient1": {
            "coords_paths": ["invalid_path_coords.h5"],
            "features_paths": ["invalid_path_features.h5"],
            "odx85": "H", 
            "age": 45
        }
    }
    
    # Create a mock S3Cache instance
    mock_s3_instance = mocker.MagicMock()
    
    # Mock S3Cache methods to raise exceptions for invalid paths
    mock_s3_instance.get_local_path.side_effect = RuntimeError("Invalid path")
    
    with pytest.raises(Exception):
        RiskFormerDataset(invalid_examples, s3_cache=mock_s3_instance)

def test_riskformer_dataset_empty_pairs(mocker):
    """Test RiskFormerDataset with empty patient examples"""
    # Create a mock S3Cache instance
    mock_s3_instance = mocker.MagicMock()
    
    # Mock logger to capture the warning message
    mock_logger = mocker.patch('riskformer.data.datasets.logger')
    
    # Create dataset with empty examples - this should not raise an exception
    # but should log a warning
    dataset = RiskFormerDataset({}, s3_cache=mock_s3_instance)
    
    # Verify that a warning was logged
    mock_logger.warning.assert_called_once_with("No patient examples provided. Creating empty Dataset.")
    
    # Verify that the dataset was created with an empty patient_ids list
    assert len(dataset.patient_ids) == 0

@pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
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
    
    # Create a mock S3Cache instance directly
    mock_s3_instance = mocker.MagicMock()
    mock_s3_instance.get_local_path.return_value = Path("fake_local_path.h5")
    mock_s3_instance.download_if_needed.return_value = "fake_local_path.h5"
    
    # Mock h5py.File to avoid trying to open a non-existent file
    mock_h5py_file = mocker.MagicMock()
    mock_h5py_file.__enter__.return_value = mocker.MagicMock()
    mock_features = mocker.MagicMock()
    mock_features.shape = (10, 256)  # Assuming feature dimension is 256
    mock_h5py_file.__enter__.return_value.__getitem__.return_value = mock_features
    
    mocker.patch('h5py.File', return_value=mock_h5py_file)
    
    # Mock the internal methods to return expected values
    mocker.patch.object(RiskFormerDataset, '_create_dense_features', 
                        return_value=[torch.zeros((32, 32, 256))])
    mocker.patch.object(RiskFormerDataset, 'split_and_pad_features', 
                        return_value=(torch.zeros((3, 32, 32, 256)), torch.zeros((3, 6))))
    
    # Initialize the dataset with the mock S3Cache
    dataset = RiskFormerDataset(mock_patient_examples, s3_cache=mock_s3_instance)
    
    # Verify the dataset was created correctly
    assert len(dataset) == 1
    assert dataset.patient_ids == [slide_id]
    
    # Get an item from the dataset
    patches, metadata = dataset[0]
    
    # Verify the tensor properties
    assert isinstance(patches, torch.Tensor)
    assert patches.shape[0] == 3  # Number of patches
    
    # Verify metadata contains expected fields
    assert metadata['patient_id'] == slide_id
    assert 'labels' in metadata
    assert 'odx85' in metadata['labels']

@patch('riskformer.data.datasets.log_event')
@pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
def test_create_riskformer_dataset_with_real_s3_path(mock_log, mock_metadata_file, mocker):
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
    
    # Create a mock S3Cache instance directly
    mock_s3_instance = mocker.MagicMock()
    mock_s3_instance.get_local_path.return_value = Path("fake/local/path.h5")
    mock_s3_instance.download_if_needed.return_value = "fake/local/path.h5"
    
    # Mock the S3Cache constructor to return our instance
    mocker.patch('riskformer.data.datasets.S3Cache', return_value=mock_s3_instance)
    
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
            "odx85": "L", 
            "age": 56, 
            "stage": "II"
        }
    }
    mocker.patch('riskformer.data.datasets.create_patient_examples', return_value=mock_patient_examples)
    
    # Mock RiskFormerDataset to verify the arguments
    mock_dataset = mocker.MagicMock()
    mock_riskformer_dataset = mocker.patch('riskformer.data.datasets.RiskFormerDataset', return_value=mock_dataset)
    
    # Test creating the dataset
    dataset = create_riskformer_dataset(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        metadata_file=f.name,
        cache_dir="/tmp/test_cache"
    )
    
    # Clean up the temporary file
    os.unlink(f.name)
    
    # Verify S3 client was initialized
    mock_init_s3.assert_called_once()
    
    # Verify the RiskFormerDataset constructor was called
    mock_riskformer_dataset.assert_called_once()
    
    # Verify the dataset was returned
    assert dataset == mock_dataset

@pytest.mark.skip(reason="This test attempts to download real data from S3 and should only be run manually")
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
    
    # Create a temporary directory for caching and an S3Cache instance
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an S3Cache instance with our temp directory
        from riskformer.data.datasets import S3Cache
        s3_cache = S3Cache(temp_dir)
        
        # Initialize the dataset with real S3 paths
        dataset = RiskFormerDataset(patient_examples, s3_cache=s3_cache)
        
        # Verify the dataset was created correctly
        assert len(dataset) == 1
        assert dataset.patient_ids == [slide_id]
        
        # Access an item to trigger data loading
        patches, metadata = dataset[0]
        
        # Verify the tensor properties (will vary based on actual data)
        assert isinstance(patches, torch.Tensor)
        assert patches.ndim > 3  # Should have at least 4 dimensions B, C, H, W
        
        # Verify metadata contains expected fields
        assert metadata['patient_id'] == slide_id
        assert 'labels' in metadata
        assert 'odx85' in metadata['labels']
        assert metadata['labels']['odx85'].item() == 0.0  # 'L' -> 0.0 for binary field