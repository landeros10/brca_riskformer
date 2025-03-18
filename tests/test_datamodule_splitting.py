import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import h5py
import numpy as np
import boto3
import botocore

from riskformer.data.datasets import RiskFormerDataModule, split_riskformer_data

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

class TestRiskFormerDataModuleSplitting:
    """
    Tests for the data splitting functionality in RiskFormerDataModule.
    These tests verify that data is split correctly into train, validation,
    and test sets according to the specified ratios.
    """
    
    @pytest.fixture
    def mock_s3_cache(self, tmp_path):
        """Mock the S3Cache class to avoid actual S3 access."""
        # Create a directory for mock cache files
        mock_cache_dir = tmp_path / "mock_cache"
        mock_cache_dir.mkdir(exist_ok=True)
        
        # Create mock H5 files for each patient in the dataset
        for i in range(40):  # 20 high risk + 20 low risk patients
            patient_id = f"patient_high_{i}" if i < 20 else f"patient_low_{i-20}"
            
            # Create coords file
            coords_file = mock_cache_dir / f"{patient_id}_coords.h5"
            with h5py.File(coords_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(10, 2))
            
            # Create features file
            features_file = mock_cache_dir / f"{patient_id}_features.h5"
            with h5py.File(features_file, 'w') as f:
                f.create_dataset('features', data=np.random.rand(10, 256))
        
        # Create feature stats JSON
        feature_stats = {
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6]
        }
        with open(mock_cache_dir / "feature_stats.json", 'w') as f:
            json.dump(feature_stats, f)
        
        # Create and configure mock
        mock_cache_instance = MagicMock()
        mock_cache_instance.get_local_path.side_effect = lambda s3_path: str(mock_cache_dir / s3_path.split('/')[-1])
        mock_cache_instance.prefetch_patient_files.return_value = feature_stats
        mock_cache_instance.cache_dir = str(mock_cache_dir)
        
        # Return the mock instance directly
        return mock_cache_instance
    
    @pytest.fixture
    def mock_metadata_file(self):
        """Create a temporary metadata file with balanced classes for testing."""
        # Create a balanced dataset with 40 samples (20 positive, 20 negative)
        metadata = {}
        
        # Create 20 "High" risk patients
        for i in range(20):
            patient_id = f"patient_high_{i}"
            metadata[patient_id] = {
                "odx85": "H",  # High risk
                "mphr": "H",   # High risk
                "age": 50 + i,
                "stage": "II" if i % 2 == 0 else "III"
            }
            
        # Create 20 "Low" risk patients
        for i in range(20):
            patient_id = f"patient_low_{i}"
            metadata[patient_id] = {
                "odx85": "L",  # Low risk
                "mphr": "L",   # Low risk
                "age": 40 + i,
                "stage": "I" if i % 2 == 0 else "II"
            }
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
        
        yield f.name
        # Clean up the temporary file
        os.unlink(f.name)
    
    @pytest.fixture
    def mock_patient_examples(self, mock_metadata_file):
        """Mock the patient examples based on our metadata."""
        with open(mock_metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Create a mock dataset with the patients from metadata
        patient_examples = {}
        for patient_id, patient_data in metadata.items():
            patient_examples[patient_id] = {
                "coords_paths": [f"s3://test-bucket/{patient_id}_coords.h5"],
                "features_paths": [f"s3://test-bucket/{patient_id}_features.h5"],
                "slide_names": [f"{patient_id}_slide"],
                "odx85": patient_data["odx85"],
                "mphr": patient_data["mphr"],
                "age": patient_data["age"],
                "stage": patient_data["stage"]
            }
        
        return patient_examples
    
    @pytest.fixture
    def mock_config_dict(self):
        """Create a mock configuration for the data module."""
        return {
            "s3_bucket": "test-bucket",
            "s3_prefix": "test-prefix",
            "max_dim": 32,
            "overlap": 0.0,
            "metadata_file": "mock_metadata.json",
            "cache_dir": "/tmp/cache",
            "profile_name": "default",
            "region_name": "us-east-1",
            "batch_size": 16,
            "num_workers": 2,
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,
            "pin_memory": True,
            "tasks": {
                "odx85": {
                    "type": "binary"
                },
                "mphr": {
                    "type": "binary"
                }
            }
        }
    
    @pytest.fixture
    def mock_config_file(self, mock_config_dict):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_config_dict, f)
        
        yield f.name
        # Clean up the temporary file
        os.unlink(f.name)
    
    def create_mock_split_datasets(self, dataset, test_split=0.2, val_split=0.25):
        """Helper function to mock the dataset splitting."""
        all_ids = list(dataset.keys())
        
        # Determine the number of patients for each split
        num_patients = len(all_ids)
        num_test = int(num_patients * test_split)
        num_val = int((num_patients - num_test) * val_split)
        num_train = num_patients - num_test - num_val
        
        # Create an even split of High/Low risk patients for the test set
        high_risk = [id for id in all_ids if "high" in id]
        low_risk = [id for id in all_ids if "low" in id]
        
        # Split the test set
        test_high = high_risk[:num_test//2]
        test_low = low_risk[:num_test//2]
        test_ids = test_high + test_low
        
        # Remaining patients
        remaining_high = high_risk[num_test//2:]
        remaining_low = low_risk[num_test//2:]
        remaining = remaining_high + remaining_low
        
        # Split the validation set
        val_ids = remaining[:num_val]
        train_ids = remaining[num_val:]
        
        # Create dataset dictionaries
        train_data = {id: dataset[id] for id in train_ids}
        val_data = {id: dataset[id] for id in val_ids}
        test_data = {id: dataset[id] for id in test_ids}
        
        return train_data, val_data, test_data
    
    @patch('riskformer.data.datasets.load_train_config')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_from_config_file(self, mock_load_config, mock_config_dict, mock_config_file):
        """Test creating a RiskFormerDataModule from a config file."""
        # Mock the load_train_config function to return our mock config
        mock_load_config.return_value = mock_config_dict
        
        # Don't actually initialize S3 client in the from_config method
        with patch('riskformer.data.datasets.initialize_s3_client') as mock_init_s3:
            # Mock the S3 client
            mock_s3_client = MagicMock()
            mock_init_s3.return_value = mock_s3_client
            
            # Skip actually setting up the patient examples to avoid S3 access
            with patch.object(RiskFormerDataModule, 'setup_patient_examples'):
                # Create the data module using from_config_file
                data_module = RiskFormerDataModule.from_config_file(mock_config_file)
                
                # Check that the data module was initialized with the correct parameters
                assert data_module.s3_bucket == mock_config_dict["s3_bucket"]
                assert data_module.s3_prefix == mock_config_dict["s3_prefix"]
                assert data_module.max_dim == mock_config_dict["max_dim"]
                assert data_module.overlap == mock_config_dict["overlap"]
                assert data_module.metadata_file == mock_config_dict["metadata_file"]
                assert data_module.cache_dir == mock_config_dict["cache_dir"]
                assert data_module.profile_name == mock_config_dict["profile_name"]
                assert data_module.region_name == mock_config_dict["region_name"]
                assert data_module.batch_size == mock_config_dict["batch_size"]
                assert data_module.num_workers == mock_config_dict["num_workers"]
                assert data_module.val_split == mock_config_dict["val_split"]
                assert data_module.test_split == mock_config_dict["test_split"]
                assert data_module.seed == mock_config_dict["seed"]
                assert data_module.pin_memory == mock_config_dict["pin_memory"]
                assert data_module.include_labels == list(mock_config_dict["tasks"].keys())
    
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_from_config_dict(self, mock_config_dict):
        """Test creating a RiskFormerDataModule from a config dictionary."""
        # Don't actually initialize S3 client in the from_config method
        with patch('riskformer.data.datasets.initialize_s3_client') as mock_init_s3:
            # Mock the S3 client
            mock_s3_client = MagicMock()
            mock_init_s3.return_value = mock_s3_client
            
            # Skip actually setting up the patient examples to avoid S3 access
            with patch.object(RiskFormerDataModule, 'setup_patient_examples'):
                # Create the data module using from_config with a dictionary
                data_module = RiskFormerDataModule.from_config(mock_config_dict)
                
                # Check that the data module was initialized with the correct parameters
                assert data_module.s3_bucket == mock_config_dict["s3_bucket"]
                assert data_module.s3_prefix == mock_config_dict["s3_prefix"]
                assert data_module.max_dim == mock_config_dict["max_dim"]
                assert data_module.overlap == mock_config_dict["overlap"]
                assert data_module.metadata_file == mock_config_dict["metadata_file"]
                assert data_module.cache_dir == mock_config_dict["cache_dir"]
                assert data_module.profile_name == mock_config_dict["profile_name"]
                assert data_module.region_name == mock_config_dict["region_name"]
                assert data_module.batch_size == mock_config_dict["batch_size"]
                assert data_module.num_workers == mock_config_dict["num_workers"]
                assert data_module.val_split == mock_config_dict["val_split"]
                assert data_module.test_split == mock_config_dict["test_split"]
                assert data_module.seed == mock_config_dict["seed"]
                assert data_module.pin_memory == mock_config_dict["pin_memory"]
                assert data_module.include_labels == list(mock_config_dict["tasks"].keys())
    
    @patch('riskformer.data.datasets.initialize_s3_client')
    @patch('riskformer.data.datasets.create_patient_examples')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_setup_patient_examples(self, mock_create_examples, mock_init_s3, mock_patient_examples, mock_s3_cache):
        """Test that patient examples are correctly set up."""
        # Mock the functions to return controlled values
        mock_create_examples.return_value = mock_patient_examples
        mock_init_s3.return_value = MagicMock()
        
        # Create the data module
        data_module = RiskFormerDataModule(
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            metadata_file="mock_metadata.json",
            cache_dir="/tmp/cache",
            include_labels=["odx85", "mphr"]
        )
        
        # Check that setup_patient_examples was called and set up the patient_examples
        assert data_module.patient_examples is not None
        assert mock_create_examples.call_count == 1
        
        # Check that the feature paths were updated to local paths
        for patient_id, patient_data in data_module.patient_examples.items():
            assert all(not path.startswith("s3://") for path in patient_data['features_paths'])
    
    @patch('riskformer.data.datasets.RiskFormerDataset')
    @patch('riskformer.data.datasets.split_riskformer_data')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_data_splitting_ratios(self, mock_split_fn, mock_dataset_class, mock_patient_examples, mock_s3_cache):
        """Test that the data is split according to the specified ratios."""
        # Create train, val, and test data splits
        test_split = 0.2
        val_split = 0.25
        
        train_data, val_data, test_data = self.create_mock_split_datasets(
            mock_patient_examples, test_split=test_split, val_split=val_split
        )
        
        # Configure the mock split function to return our splits
        mock_split_fn.return_value = (train_data, test_data)
        
        # Configure the mock datasets returned by RiskFormerDataset
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]
        
        # Mock the random_split function
        with patch('riskformer.data.datasets.random_split') as mock_random_split:
            mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)
            
            # Create the data module
            data_module = RiskFormerDataModule(
                s3_bucket="test-bucket",
                s3_prefix="test-prefix",
                metadata_file="mock_metadata.json",
                cache_dir="/tmp/cache",
                test_split=test_split,
                val_split=val_split,
                include_labels=["odx85", "mphr"]
            )
            
            # Setup the data module
            with patch.object(data_module, 'patient_examples', mock_patient_examples):
                with patch.object(data_module, 'feature_stats_path', '/tmp/cache/feature_stats.json'):
                    with patch('builtins.open', MagicMock()):
                        with patch('json.load', return_value={"mean": [0.1], "std": [0.2]}):
                            data_module.setup(stage="fit")
                            data_module.setup(stage="test")
            
            # Check that split_riskformer_data was called with the right parameters
            mock_split_fn.assert_called_once_with(
                examples=mock_patient_examples,
                label_var="odx85",
                positive_label="H",
                test_split_ratio=test_split
            )
            
            # Check that the datasets were created
            assert mock_dataset_class.call_count == 2
            assert mock_random_split.call_count == 1
    
    @patch('riskformer.data.datasets.RiskFormerDataset')
    @patch('riskformer.data.datasets.split_riskformer_data')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_dataloader_creation(self, mock_split_fn, mock_dataset_class, mock_patient_examples, mock_s3_cache):
        """Test that the correct dataloaders are created."""
        # Create train, val, and test data splits
        train_data, val_data, test_data = self.create_mock_split_datasets(mock_patient_examples)
        
        # Configure the mock split function
        mock_split_fn.return_value = (train_data, test_data)
        
        # Configure the mock datasets returned by RiskFormerDataset
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        
        # Set the length of the mock datasets
        mock_train_dataset.__len__.return_value = 10
        mock_val_dataset.__len__.return_value = 5
        mock_test_dataset.__len__.return_value = 5
        
        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]
        
        # Mock the random_split function
        with patch('riskformer.data.datasets.random_split') as mock_random_split:
            mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)
            
            # Create the data module
            data_module = RiskFormerDataModule(
                s3_bucket="test-bucket",
                s3_prefix="test-prefix",
                metadata_file="mock_metadata.json",
                cache_dir="/tmp/cache",
                batch_size=4,
                num_workers=0, # Use 0 for testing to avoid subprocess issues
                test_split=0.2,
                val_split=0.25,
                include_labels=["odx85", "mphr"]
            )
            
            # Setup the data module
            with patch.object(data_module, 'patient_examples', mock_patient_examples):
                with patch.object(data_module, 'feature_stats_path', '/tmp/cache/feature_stats.json'):
                    with patch('builtins.open', MagicMock()):
                        with patch('json.load', return_value={"mean": [0.1], "std": [0.2]}):
                            data_module.setup(stage="fit")
                            data_module.setup(stage="test")
            
            # Test the dataloaders
            data_module.train_dataset = mock_train_dataset
            data_module.val_dataset = mock_val_dataset
            data_module.test_dataset = mock_test_dataset
            
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            
            # Check that the dataloaders have the correct batch size
            assert train_loader.batch_size == 4
            assert val_loader.batch_size == 4
            assert test_loader.batch_size == 4
            
            # Check that the dataloaders have the correct number of workers
            assert train_loader.num_workers == 0
            assert val_loader.num_workers == 0
            assert test_loader.num_workers == 0
    
    @patch('riskformer.data.datasets.S3Cache.prefetch_patient_files')
    @patch('riskformer.data.datasets.load_dataset_metadata')
    @patch('riskformer.data.datasets.S3Cache')
    @patch('riskformer.data.datasets.initialize_s3_client')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_prepare_data(self, mock_init_s3, mock_s3cache_class, mock_load_metadata, mock_prefetch, mock_patient_examples, mock_s3_cache):
        """Test the prepare_data method."""
        # Setup mocks
        mock_s3cache_class.return_value = mock_s3_cache
        
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_init_s3.return_value = mock_s3_client
        
        # Mock load_dataset_metadata to return valid data
        slide_ids = set(mock_patient_examples.keys())
        mock_metadata = {id: {"odx85": "H" if "high" in id else "L"} for id in slide_ids}
        mock_load_metadata.return_value = (slide_ids, mock_metadata)
        
        # Mock create_patient_examples
        mock_create_examples = MagicMock()
        mock_create_examples.return_value = mock_patient_examples
        with patch('riskformer.data.datasets.create_patient_examples', mock_create_examples):
            # Configure the prefetch mock to return a dictionary of stats
            mock_prefetch.return_value = {
                "mean": [0.1, 0.2, 0.3],
                "std": [0.4, 0.5, 0.6]
            }
            
            # Create the data module
            data_module = RiskFormerDataModule(
                s3_bucket="test-bucket",
                s3_prefix="test-prefix",
                metadata_file="mock_metadata.json",
                cache_dir="/tmp/cache",
                include_labels=["odx85", "mphr"]
            )
            
            # Verify S3 client was initialized
            mock_init_s3.assert_called_once()
            
            # Call prepare_data
            data_module.prepare_data()
            
            # Verify prefetch_patient_files was called
            mock_prefetch.assert_called_once_with(
                patient_examples=data_module.patient_examples,
                collect_stats=True
            )
            
            # Verify the feature stats were saved to disk
            assert data_module.feature_stats_path == os.path.join("/tmp/cache", "feature_stats.json")
    
    @patch('riskformer.data.datasets.load_dataset_metadata')
    @patch('riskformer.data.datasets.S3Cache')
    @pytest.mark.skipif(not is_aws_credentials_available(), reason="AWS credentials not available")
    def test_teardown(self, mock_s3cache_class, mock_load_metadata, mock_s3_cache):
        """Test the teardown method."""
        # Setup mocks
        mock_s3cache_class.return_value = mock_s3_cache
        
        # Mock load_dataset_metadata to return valid data
        slide_ids = {"patient_1", "patient_2"}
        mock_metadata = {id: {"odx85": "H"} for id in slide_ids}
        mock_load_metadata.return_value = (slide_ids, mock_metadata)
        
        # Create the data module
        data_module = RiskFormerDataModule(
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            metadata_file="mock_metadata.json",
            cache_dir="/tmp/cache"
        )
        
        # Call teardown
        data_module.teardown(stage="fit")
        
        # No specific assertions needed as teardown simply gets rid of memory
        # Just make sure it doesn't raise any exceptions 