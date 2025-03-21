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

from riskformer.data.datasets import RiskFormerDataModule
from tests.utils import check_aws_credentials

class TestDataModuleUnit:
    """Unit tests for RiskFormerDataModule."""
    
    @pytest.fixture
    def mock_config_dict(self):
        """Create a mock configuration dictionary for testing."""
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
                }
            }
        }
    
    @pytest.fixture
    def mock_config_file(self, mock_config_dict):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_config_dict, f)
        
        yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for testing."""
        return {
            "slide1": {
                "patient": "patient1",
                "age_at_diagnosis": 50,
                "odx85": "H",
                "mphr": "H",
                "Grade": 3.0,
                "odx_train": 1.083
            },
            "slide2": {
                "patient": "patient2",
                "age_at_diagnosis": 40,
                "odx85": "L",
                "mphr": "L",
                "Grade": 2.0,
                "odx_train": -1.491
            }
        }
    
    def test_from_config_dict(self, mock_config_dict):
        """Test creating a DataModule from a config dictionary."""
        with patch('riskformer.data.datasets.initialize_s3_client') as mock_init_s3:
            mock_s3_client = MagicMock()
            mock_init_s3.return_value = mock_s3_client
            
            with patch.object(RiskFormerDataModule, 'setup_patient_examples'):
                data_module = RiskFormerDataModule.from_config(mock_config_dict)
                
                assert data_module.s3_bucket == mock_config_dict["s3_bucket"]
                assert data_module.s3_prefix == mock_config_dict["s3_prefix"]
                assert data_module.max_dim == mock_config_dict["max_dim"]
                assert data_module.overlap == mock_config_dict["overlap"]
                assert data_module.metadata_file == mock_config_dict["metadata_file"]
                assert data_module.cache_dir == mock_config_dict["cache_dir"]
                assert data_module.batch_size == mock_config_dict["batch_size"]
                assert data_module.num_workers == mock_config_dict["num_workers"]
                assert data_module.val_split == mock_config_dict["val_split"]
                assert data_module.test_split == mock_config_dict["test_split"]
                assert data_module.seed == mock_config_dict["seed"]
                assert data_module.pin_memory == mock_config_dict["pin_memory"]
    
    def test_from_config_file(self, mock_config_file, mock_config_dict):
        """Test creating a DataModule from a config file."""
        with patch('riskformer.data.datasets.load_train_config') as mock_load_config:
            mock_load_config.return_value = mock_config_dict
            
            with patch('riskformer.data.datasets.initialize_s3_client') as mock_init_s3:
                mock_s3_client = MagicMock()
                mock_init_s3.return_value = mock_s3_client
                
                with patch.object(RiskFormerDataModule, 'setup_patient_examples'):
                    data_module = RiskFormerDataModule.from_config_file(mock_config_file)
                    
                    assert data_module.s3_bucket == mock_config_dict["s3_bucket"]
                    assert data_module.metadata_file == mock_config_dict["metadata_file"]
                    assert data_module.batch_size == mock_config_dict["batch_size"]
    
    def test_prepare_data(self, mock_config_dict):
        """Test prepare_data method."""
        with patch('riskformer.data.datasets.initialize_s3_client'):
            data_module = RiskFormerDataModule.from_config(mock_config_dict)
            
            # Mock S3Cache
            mock_s3_cache = MagicMock()
            mock_s3_cache.prefetch_patient_files.return_value = {
                "mean": [0.1, 0.2, 0.3],
                "std": [1.0, 1.0, 1.0]
            }
            data_module.s3_cache = mock_s3_cache
            
            # Mock patient examples
            data_module.patient_examples = {
                "patient1": {"features_paths": ["path1.h5"]}
            }
            
            # Mock file operations
            with patch("builtins.open", MagicMock()):
                with patch("json.dump") as mock_dump:
                    data_module.prepare_data()
                    
                    mock_s3_cache.prefetch_patient_files.assert_called_once_with(
                        patient_examples=data_module.patient_examples,
                        collect_stats=True
                    )
                    mock_dump.assert_called_once()
    
    def test_setup_patient_examples(self, mock_config_dict, mock_metadata):
        """Test setup_patient_examples method."""
        with patch('riskformer.data.datasets.initialize_s3_client'):
            data_module = RiskFormerDataModule.from_config(mock_config_dict)
            
            # Mock load_dataset_metadata
            with patch('riskformer.data.datasets.load_dataset_metadata') as mock_load_metadata:
                mock_load_metadata.return_value = (list(mock_metadata.keys()), mock_metadata)
                
                # Mock create_patient_examples
                mock_examples = {
                    "patient1": {
                        "features_paths": ["s3://bucket/path1.h5"],
                        "coords_paths": ["s3://bucket/path1_coords.h5"]
                    }
                }
                with patch('riskformer.data.datasets.create_patient_examples', return_value=mock_examples):
                    # Mock S3Cache
                    mock_s3_cache = MagicMock()
                    mock_s3_cache.get_local_path.side_effect = lambda x: str(Path(x).name)
                    data_module.s3_cache = mock_s3_cache
                    
                    data_module.setup_patient_examples()
                    
                    assert data_module.patient_examples is not None
                    assert len(data_module.patient_examples) > 0
                    # Check paths were converted to local paths
                    for patient_data in data_module.patient_examples.values():
                        assert all(not str(path).startswith("s3://") 
                                 for path in patient_data['features_paths'])
    
    def test_error_handling(self, mock_config_dict):
        """Test error handling for invalid configurations."""
        # Test missing required fields
        invalid_config = mock_config_dict.copy()
        del invalid_config["s3_bucket"]
        
        with pytest.raises(KeyError):
            RiskFormerDataModule.from_config(invalid_config)
        
        # Test invalid split ratios
        invalid_config = mock_config_dict.copy()
        invalid_config["val_split"] = 0.5
        invalid_config["test_split"] = 0.6  # Total > 1.0
        
        with pytest.raises(ValueError):
            RiskFormerDataModule.from_config(invalid_config)
    
    def test_teardown(self, mock_config_dict):
        """Test teardown method."""
        with patch('riskformer.data.datasets.initialize_s3_client'):
            data_module = RiskFormerDataModule.from_config(mock_config_dict)
            
            # Create mock datasets
            data_module.train_dataset = MagicMock()
            data_module.val_dataset = MagicMock()
            data_module.test_dataset = MagicMock()
            data_module.feature_stats = {"mean": [0.1], "std": [0.2]}
            
            # Test fit stage teardown
            data_module.teardown("fit")
            assert data_module.train_dataset is None
            assert data_module.val_dataset is None
            assert data_module.test_dataset is not None  # Should not be cleared
            
            # Test test stage teardown
            data_module.teardown("test")
            assert data_module.test_dataset is None
            assert data_module.feature_stats is None

if __name__ == "__main__":
    pytest.main() 