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
import inspect

from riskformer.data.datasets import RiskFormerDataModule, split_riskformer_data

# Directly use pytest's built-in mechanism for skipping, applied at the class level
@pytest.mark.skip(reason="S3-dependent tests currently disabled")
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
    @pytest.mark.skip(reason="AWS credentials required")
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
    
    @pytest.mark.skip(reason="AWS credentials required")
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
    @patch('riskformer.data.datasets.load_dataset_metadata')
    @patch('riskformer.data.datasets.create_patient_examples')
    @pytest.mark.skip(reason="AWS credentials required")
    def test_setup_patient_examples(self, mock_create_examples, mock_load_metadata, mock_init_s3, mock_patient_examples, mock_s3_cache):
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
    
    @patch('riskformer.data.datasets.initialize_s3_client')
    @patch('riskformer.data.datasets.load_dataset_metadata')
    @patch('riskformer.data.datasets.create_patient_examples')
    def test_setup_patient_examples_no_aws(self, mock_create_examples, mock_load_metadata, mock_init_s3):
        """Test patient examples setup without requiring AWS credentials."""
        import tempfile
        
        print("Running test_setup_patient_examples_no_aws")
        
        # Create mock metadata
        slide_ids = ["slide1", "slide2", "slide3"]
        slide_data = {
            "slide1": {"odx85": "H", "mphr": "L"},
            "slide2": {"odx85": "L", "mphr": "H"},
            "slide3": {"odx85": "H", "mphr": "H"}
        }
        mock_load_metadata.return_value = (slide_ids, slide_data)
        
        # Create mock S3 client
        mock_s3_client = MagicMock()
        mock_init_s3.return_value = mock_s3_client
        
        # Create mock patient examples
        mock_patient_examples = {
            "patient1": {
                "features_paths": ["s3://test-bucket/patient1_features.h5"],
                "odx85": "H",
                "mphr": "L"
            },
            "patient2": {
                "features_paths": ["s3://test-bucket/patient2_features.h5"],
                "odx85": "L",
                "mphr": "H"
            }
        }
        mock_create_examples.return_value = mock_patient_examples
        
        # Create a temporary directory for cache
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the data module
            data_module = RiskFormerDataModule(
                s3_bucket="test-bucket",
                s3_prefix="test-prefix",
                metadata_file="mock_metadata.json",
                cache_dir=tmp_dir,
                include_labels=["odx85", "mphr"]
            )
            
            # Replace the S3Cache with a mock version
            data_module.s3_cache = MagicMock()
            data_module.s3_cache.get_local_path.side_effect = lambda x: x.replace("s3://test-bucket/", f"{tmp_dir}/")
            
            # Call setup_patient_examples manually
            data_module.setup_patient_examples()
            
            # Verify load_dataset_metadata was called
            mock_load_metadata.assert_called_once_with("mock_metadata.json")
            
            # Verify initialize_s3_client was called
            mock_init_s3.assert_called_once()
            
            # Verify create_patient_examples was called
            mock_create_examples.assert_called_once_with(
                mock_s3_client,
                "test-bucket",
                "test-prefix",
                slide_ids,
                slide_data
            )
            
            # Verify patient_examples were properly set
            assert data_module.patient_examples is not None
            # Verify paths were converted to local paths
            for patient_id, patient_data in data_module.patient_examples.items():
                for path in patient_data['features_paths']:
                    assert not path.startswith("s3://")
                    assert path.startswith(tmp_dir)
    
    @patch('riskformer.data.datasets.RiskFormerDataset')
    @patch('riskformer.data.datasets.split_riskformer_data')
    @pytest.mark.skip(reason="AWS credentials required")
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
    @pytest.mark.skip(reason="AWS credentials required")
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
    @pytest.mark.skip(reason="AWS credentials required")
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
    @pytest.mark.skip(reason="AWS credentials required")
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

def test_datamodule_dataloaders(mocker):
    """
    Test the train, validation, and test dataloaders of RiskFormerDataModule.
    """
    # Mock dependencies for initialization
    mocker.patch.object(RiskFormerDataModule, "setup_patient_examples")
    mocker.patch.object(RiskFormerDataModule, "set_split_var")
    mock_s3_cache = mocker.MagicMock()
    mocker.patch("riskformer.data.datasets.S3Cache", return_value=mock_s3_cache)
    
    # Create a simple datamodule with test configuration
    datamodule = RiskFormerDataModule(
        s3_bucket="dummy-bucket",
        s3_prefix="dummy-prefix",
        batch_size=4,
        val_split=0.2,
        test_split=0.2,
        cache_dir="/tmp/cache"
    )
    
    # Create a mock dataset that will work with DataLoader
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            return torch.randn(5), {"labels": {}}
    
    # Set up datasets directly
    datamodule.train_dataset = MockDataset()
    datamodule.val_dataset = MockDataset()
    datamodule.test_dataset = MockDataset()
    
    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    assert train_loader is not None
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    
    # Test val dataloader
    val_loader = datamodule.val_dataloader()
    assert val_loader is not None
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    
    # Test test dataloader
    test_loader = datamodule.test_dataloader()
    assert test_loader is not None
    assert isinstance(test_loader, torch.utils.data.DataLoader)

def test_datamodule_prepare_data(mocker):
    """
    Test the prepare_data method of RiskFormerDataModule.
    """
    # Get a reference to the RiskFormerDataModule class
    dm_class = RiskFormerDataModule
    
    # Mock necessary setup methods to avoid errors
    mocker.patch.object(dm_class, "setup_patient_examples")
    mocker.patch.object(dm_class, "set_split_var")
    
    # Create a dictionary that can be JSON serialized
    feature_stats = {"mean": [1.0, 2.0, 3.0], "std": [0.1, 0.2, 0.3]}
    
    # Create mock for S3Cache with a proper return value for prefetch_patient_files
    mock_s3_cache = mocker.MagicMock()
    mock_s3_cache.prefetch_patient_files.return_value = feature_stats
    mocker.patch("riskformer.data.datasets.S3Cache", return_value=mock_s3_cache)
    
    # Mock the open function to avoid file operations
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.dump")
    
    # Create datamodule
    datamodule = dm_class(
        s3_bucket="dummy-bucket",
        s3_prefix="dummy-prefix",
        batch_size=4,
        cache_dir="/tmp/cache"
    )
    
    # Set patient_examples manually
    datamodule.patient_examples = {"patient1": {"features_paths": ["dummy_path"]}}
    
    # Call prepare_data
    datamodule.prepare_data()
    
    # Verify the s3_cache.prefetch_patient_files was called with correct parameters
    mock_s3_cache.prefetch_patient_files.assert_called_with(
        patient_examples=datamodule.patient_examples,
        collect_stats=True
    )
    
    # Verify open was called with the correct file path
    mock_open.assert_called_with(datamodule.feature_stats_path, "w")

def test_datamodule_teardown(mocker):
    """
    Test the teardown method of RiskFormerDataModule.
    """
    # Mock setup methods
    mocker.patch.object(RiskFormerDataModule, "setup_patient_examples")
    mocker.patch.object(RiskFormerDataModule, "set_split_var")
    
    # Mock S3Cache class
    mock_s3_cache = mocker.MagicMock()
    mocker.patch("riskformer.data.datasets.S3Cache", return_value=mock_s3_cache)
    
    # Create datamodule
    datamodule = RiskFormerDataModule(
        s3_bucket="dummy-bucket",
        s3_prefix="dummy-prefix",
        batch_size=4,
        cache_dir="/tmp/cache"
    )
    
    # Create mock datasets for fit and test stages
    mock_train_dataset = mocker.MagicMock()
    mock_val_dataset = mocker.MagicMock()
    mock_test_dataset = mocker.MagicMock()
    
    datamodule.train_dataset = mock_train_dataset
    datamodule.val_dataset = mock_val_dataset
    datamodule.test_dataset = mock_test_dataset
    datamodule.feature_stats = {"mean": [1.0], "std": [1.0]}
    
    # Test teardown for fit stage
    datamodule.teardown("fit")
    
    # Verify the datasets are cleared
    assert datamodule.train_dataset is None
    assert datamodule.val_dataset is None
    assert datamodule.test_dataset is mock_test_dataset  # Should not be cleared
    
    # Restore test dataset
    datamodule.test_dataset = mock_test_dataset
    
    # Test teardown for test stage
    datamodule.teardown("test")
    
    # Verify test dataset is cleared
    assert datamodule.test_dataset is None
    
    # Verify feature_stats are cleared in both cases
    assert datamodule.feature_stats is None 