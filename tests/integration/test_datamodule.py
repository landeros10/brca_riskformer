import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
import h5py
import numpy as np
import shutil
from unittest.mock import patch, MagicMock
from typing import Iterable
from riskformer.data.datasets import RiskFormerDataModule, slide_to_patient_examples
from tests.utils import check_aws_credentials

class TestDataModuleIntegration:
    """Integration tests for RiskFormerDataModule."""

    @pytest.fixture
    def feature_dim(self):
        return 64
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path, feature_dim):
        """Create a mock data directory with feature files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create test slide data - this will be used for both metadata and mocked S3 files
        # Define a consistent set of slide IDs that we'll use throughout the tests
        slide_ids = [f"sample_{i}" for i in range(10)]  # 10 slides, 5 patients
        
        # Create feature files
        num_regions = 10
        
        for slide_id in slide_ids:
            # Create feature file
            feature_file = data_dir / f"{slide_id}_features.h5"
            with h5py.File(feature_file, 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_regions, feature_dim))
            
            # Create coordinate file
            coord_file = data_dir / f"{slide_id}_coords.h5"
            with h5py.File(coord_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(num_regions, 2) * 100)
        
        # Create metadata file
        # Map slides to patients (2 slides per patient)
        metadata = {}
        for i, slide_id in enumerate(slide_ids):
            patient_id = f"patient_{i//2}"  # Each patient has 2 slides
            metadata[slide_id] = {
                "patient": patient_id,
                "age_at_diagnosis": 40 + i * 5,
                "odx85": "H" if i % 4 == 0 else "L",
                "mphr": "H" if i % 3 == 0 else "L",
                "Grade": float(i % 3 + 1),
                "odx_train": 1.0 if i % 2 == 0 else -1.0,
                # Add required fields for create_patient_examples
                "Disease_Free_Months": 36.0 + i * 2.5,
                "Necrosis": "Present" if i % 2 == 0 else "Absent",
                "Pleomorph": i % 3,
                "Overall_Survival_Months": 48.0 + i * 3.0,
                "Overall_Survival_Status": "alive" if i % 4 != 0 else "dead",
                "ER_Status_By_IHC": "positive" if i % 3 != 0 else "negative",
                "pr_status_by_ihc": "positive" if i % 3 != 1 else "negative",
                "HER2Calc": "positive" if i % 5 == 0 else "negative",
                "Lymphovascular Invasion (LVI)": "Present" if i % 3 == 0 else "Absent",
                "tumor_size": float(i + 2),
                "Epithelial": float(i % 3 + 1),
                "Mitosis": f"Mitosis score {i} (score = {i % 3 + 1})"
            }
        
        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        return data_dir, metadata, slide_ids
    
    @pytest.fixture
    def config_dict(self, mock_data_dir, feature_dim):
        """Create a configuration dictionary for testing."""
        data_dir, _, _ = mock_data_dir
        return {
            "s3_bucket": "mock-bucket",  # Use a valid bucket name even though we'll mock S3
            "s3_prefix": "test-prefix",
            "max_dim": 32,
            "feature_dim": feature_dim,
            "overlap": 0.0,
            "metadata_file": str(data_dir / "metadata.json"),
            "cache_dir": str(data_dir / "cache"),
            "profile_name": "default",
            "region_name": "us-east-1",
            "batch_size": 1,
            "num_workers": 0,
            "val_split": 0.2,
            "test_split": 0.2,
            "seed": 42,
            "pin_memory": True,
            "tasks": {
                "odx85": {"type": "binary"},
                "mphr": {"type": "binary"},
                "Grade": {"type": "regression", "min": 1.0, "max": 3.0}
            }
        }
    
    @pytest.fixture
    def mock_s3_files(self, mock_data_dir):
        """Create a mock file list structure for S3."""
        _, _, slide_ids = mock_data_dir
        
        # Create a dictionary that mimics the result of list_bucket_files
        mock_files = {}
        prefix = "test-prefix/"
        
        for slide_id in slide_ids:
            # Create file entries for each slide - both coords and features
            coords_key = f"{prefix}{slide_id}_coords.h5"
            features_key = f"{prefix}{slide_id}_features.h5"
            
            # Create mock file objects
            mock_files[coords_key] = {
                "Key": coords_key,
                "Size": 1024,
                "LastModified": "2023-01-01"
            }
            
            mock_files[features_key] = {
                "Key": features_key,
                "Size": 2048,
                "LastModified": "2023-01-01"
            }
        
        return mock_files
    
    @pytest.fixture
    def mock_s3_cache(self, mock_data_dir):
        """Create a mock S3Cache that maps S3 paths to local test files."""
        data_dir, metadata, _ = mock_data_dir
        
        # Create a mock S3Cache
        mock_cache = MagicMock()
        
        # Mock get_local_path to return local test file paths
        def mock_get_local_path(s3_path):
            # Extract the file name from the S3 path
            if not s3_path.startswith("s3://"):
                return s3_path
                
            parsed_path = s3_path.replace("s3://mock-bucket/test-prefix/", "")
            # Map to our local test files
            return str(data_dir / parsed_path)
            
        mock_cache.get_local_path.side_effect = mock_get_local_path
        mock_cache.download_if_needed.side_effect = mock_get_local_path
        
        # Create feature stats
        feature_stats = {
            "mean": [0.1, 0.2, 0.3, 0.4] * 16,  # Make sure length matches feature_dim
            "std": [1.0, 0.9, 0.8, 0.7] * 16
        }
        
        # Mock prefetch_patient_files to return the original examples with stats
        def mock_prefetch(patient_examples, collect_stats=True, num_workers=4):
            # Convert S3 paths to local paths
            local_examples = {}
            for patient_id, data in patient_examples.items():
                local_examples[patient_id] = {
                    "features_paths": [mock_get_local_path(path) for path in data["features_paths"]],
                    "coords_paths": [mock_get_local_path(path) for path in data["coords_paths"]]
                }
                # Copy all other metadata
                for key, value in data.items():
                    if key not in ["features_paths", "coords_paths"]:
                        local_examples[patient_id][key] = value
                
            return local_examples, feature_stats
            
        mock_cache.prefetch_patient_files.side_effect = mock_prefetch
        
        return mock_cache

    @pytest.fixture
    def mock_load_metadata(self, mock_data_dir):
        """Create a fixture to mock load_dataset_metadata function."""
        _, metadata, _ = mock_data_dir
        
        # Create a patch for load_dataset_metadata
        # This ensures consistency between the metadata and the S3 file list
        return (set(metadata.keys()), metadata)
        
    @pytest.fixture
    def mock_create_patient_examples(self, mock_data_dir, mock_s3_files):
        """Create a mock for create_patient_examples with properly formatted values."""
        _, metadata, slide_ids = mock_data_dir
        
        slide_examples = {
            slide_id: {
                "patient_id": metadata[slide_id]["patient"],
                "coords_path": f"s3://mock-bucket/test-prefix/{slide_id}_coords.h5",
                "features_path": f"s3://mock-bucket/test-prefix/{slide_id}_features.h5"
            }
            for slide_id in slide_ids
        }
        
        # Create patient examples directly with proper field values
        patient_examples = slide_to_patient_examples(
            slide_examples=slide_examples,
            slide_data=metadata,
        )
        return patient_examples
        
    def validate_dataloader_batch(self, batch, config_dict):
        """
        Helper method to validate a batch from a dataloader.
        Consolidates common assertion checks.
        """
        
        assert isinstance(batch, Iterable)
        assert len(batch) == 2
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], dict)
        
        # Validate features
        features = batch[0]

        assert features.dim() == 5  # [B, n_regions, C, H, W]
        assert features.shape[0] <= config_dict["batch_size"]
        assert features.shape[1] >= 1
        assert features.shape[2] == config_dict["feature_dim"]
        assert features.shape[3] <= config_dict["max_dim"]
        assert features.shape[4] <= config_dict["max_dim"]
        
        # Validate labels
        labels = batch[1]["labels"]
        expected_labels = config_dict["tasks"].keys()
            
        for label_name in expected_labels:
            assert label_name.lower() in labels
            assert labels[label_name.lower()].shape[0] == features.shape[0]  # Same batch size
            
        return features, labels
    
    def setup_datamodule(self, config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples=None):
        """Helper method to set up a data module through prepare and setup stages."""
        # Make sure cache directory exists
        cache_dir = Path(config_dict["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock the S3 client initialization
        mock_s3_client = MagicMock()
        
        # Create DataModule with mocked components
        with patch('riskformer.data.datasets.initialize_s3_client', return_value=mock_s3_client):
            with patch('riskformer.data.datasets.S3Cache', return_value=mock_s3_cache):
                with patch('riskformer.data.datasets.list_bucket_files', return_value=mock_s3_files):
                    with patch('riskformer.data.datasets.load_dataset_metadata', return_value=mock_load_metadata):
                        if mock_create_patient_examples:
                            with patch('riskformer.data.datasets.create_patient_examples', return_value=mock_create_patient_examples):
                                # Create the data module
                                data_module = RiskFormerDataModule.from_config(config_dict)
                                
                                # Prepare data (which requires the cache directory to exist)
                                data_module.prepare_data()
                                
                                # Setup stages
                                data_module.setup("fit")
                                data_module.setup("test")
                                
                                return data_module
                        else:
                            # Create the data module
                            data_module = RiskFormerDataModule.from_config(config_dict)
                            
                            # Prepare data (which requires the cache directory to exist)
                            data_module.prepare_data()
                            
                            # Setup stages
                            data_module.setup("fit")
                            data_module.setup("test")
                            
                            return data_module
    
    def test_full_workflow(self, config_dict, mock_data_dir, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples):
        """Test the complete workflow of the DataModule."""
        # Create and setup DataModule
        data_module = self.setup_datamodule(config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples)
        
        # Check datasets
        assert data_module.train_dataset is not None
        assert len(data_module.train_dataset) > 0
        assert len(data_module.train_dataset) == 2 # shuold select num_pos=1 for each class for testing
        assert data_module.val_dataset is not None
        assert len(data_module.val_dataset) > 0
        assert len(data_module.val_dataset) == 1 # should select num_val=1 for each class for testing
        assert data_module.test_dataset is not None
        assert len(data_module.test_dataset) > 0
        assert len(data_module.test_dataset) == 2 # should select num_pos=1 for each class for testing
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check batch from each loader using the consolidated helper
        # TODO
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            self.validate_dataloader_batch(batch, config_dict)
        
        # Test teardown
        data_module.teardown("fit")
        assert data_module.train_dataset is None
        assert data_module.val_dataset is None
        
        # Setup for test stage
        data_module.setup("test")
        assert data_module.test_dataset is not None
        
        # Final teardown
        data_module.teardown("test")
        assert data_module.test_dataset is None
    
    def test_label_specific_processing(self, config_dict, mock_data_dir, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples):
        """
        Test specific label processing for different types of labels 
        (binary, regression, etc.)
        """
        # Create and set up data module
        data_module = self.setup_datamodule(config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples)
        
        # Get a batch from train loader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        _, labels = self.validate_dataloader_batch(batch, config_dict)
        
        # Test binary label (odx85) - should be 0.0 or 1.0 values
        odx85_values = labels["odx85"].cpu().numpy().flatten()
        assert np.all(np.logical_or(np.isclose(odx85_values, 0.0), 
                                  np.isclose(odx85_values, 1.0))), \
            f"odx85 values should be binary (0.0 or 1.0), got {odx85_values}"
        
        # Test binary label (mphr) - should be 0.0 or 1.0 values
        mphr_values = labels["mphr"].cpu().numpy().flatten()
        assert np.all(np.logical_or(np.isclose(mphr_values, 0.0), 
                                  np.isclose(mphr_values, 1.0))), \
            f"mphr values should be binary (0.0 or 1.0), got {mphr_values}"
        
        # Test regression label (Grade) - should be between 1.0 and 3.0
        grade_values = labels["Grade".lower()].cpu().numpy().flatten()
        assert np.all(grade_values >= 1.0) and np.all(grade_values <= 3.0), \
            f"Grade values should be between 1.0 and 3.0, got {grade_values}"
        
        # Test different include_labels configuration
        limited_config = config_dict.copy()
        limited_config["tasks"] = {"odx85": {"type": "binary"}}  # Only include odx85
        
        # Create limited module with mocks
        with patch('riskformer.data.datasets.initialize_s3_client'):
            with patch('riskformer.data.datasets.S3Cache', return_value=mock_s3_cache):
                with patch('riskformer.data.datasets.list_bucket_files', return_value=mock_s3_files):
                    with patch('riskformer.data.datasets.load_dataset_metadata', return_value=mock_load_metadata):
                        limited_module = RiskFormerDataModule.from_config(limited_config)
                        limited_module.prepare_data()
                        limited_module.setup("fit")
                        
                        limited_batch = next(iter(limited_module.train_dataloader()))
                        _, limited_labels = self.validate_dataloader_batch(limited_batch, limited_config)
                        
                        # Should only have odx85, not mphr or Grade
                        assert "odx85" in limited_labels
                        assert "mphr" not in limited_labels
                        assert "Grade" not in limited_labels
    
    def test_data_loading_and_caching(self, config_dict, mock_data_dir, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples):
        """Test data loading and caching functionality."""
        # Create and clean the cache directory
        cache_dir = Path(config_dict["cache_dir"])
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock the S3 client initialization
        mock_s3_client = MagicMock()
        
        # Create DataModule with mocked components
        with patch('riskformer.data.datasets.initialize_s3_client', return_value=mock_s3_client):
            with patch('riskformer.data.datasets.S3Cache', return_value=mock_s3_cache):
                with patch('riskformer.data.datasets.list_bucket_files', return_value=mock_s3_files):
                    with patch('riskformer.data.datasets.load_dataset_metadata', return_value=mock_load_metadata):
                        with patch('riskformer.data.datasets.create_patient_examples', return_value=mock_create_patient_examples):
                            # Create the data module
                            data_module = RiskFormerDataModule.from_config(config_dict)
                            
                            # Prepare data
                            data_module.prepare_data()
                            
                            # Verify S3Cache methods were called
                            mock_s3_cache.prefetch_patient_files.assert_called_once()
                            
                            # Setup and validate dataset creation
                            data_module.setup("fit")
                            
                            # Validate dataset has been created and uses the correct data
                            assert data_module.train_dataset is not None
                            assert data_module.val_dataset is not None
    
    def test_data_splits(self, config_dict, mock_data_dir, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples):
        """Test dataset splitting functionality."""
        # Create DataModule
        data_module = self.setup_datamodule(config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples)
        
        # Get total number of patients
        total_patients = len(data_module.patient_examples)
        
        # Calculate expected split sizes based on val_split and test_split
        expected_test = int(total_patients * config_dict["test_split"])
        expected_train_val = total_patients - expected_test
        expected_val = int(expected_train_val * config_dict["val_split"])
        expected_train = expected_train_val - expected_val
        
        # Account for possible rounding differences
        assert abs(len(data_module._train_data) - expected_train) <= 1
        assert abs(len(data_module._test_data) - expected_test) <= 1
        
        # Check no overlap between train and test data
        train_patients = set(data_module._train_data.keys())
        test_patients = set(data_module._test_data.keys())
        assert not train_patients & test_patients, "Train and test sets should not overlap"
    
    def test_reproducibility(self, config_dict, mock_data_dir, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples):
        """Test reproducibility of data splits."""
        # Create two DataModules with same seed
        data_module1 = self.setup_datamodule(config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples)
        data_module2 = self.setup_datamodule(config_dict, mock_s3_files, mock_s3_cache, mock_load_metadata, mock_create_patient_examples)
        
        # Compare train splits
        train_patients1 = set(data_module1._train_data.keys())
        train_patients2 = set(data_module2._train_data.keys())
        assert train_patients1 == train_patients2, "Train splits should be identical with same seed"
        
        # Compare test splits
        test_patients1 = set(data_module1._test_data.keys())
        test_patients2 = set(data_module2._test_data.keys())
        assert test_patients1 == test_patients2, "Test splits should be identical with same seed"

if __name__ == "__main__":
    pytest.main() 