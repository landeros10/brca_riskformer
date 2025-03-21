import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
import h5py
import numpy as np

from riskformer.data.datasets import RiskFormerDataModule
from tests.utils import check_aws_credentials

class TestDataModuleIntegration:
    """Integration tests for RiskFormerDataModule."""
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with feature files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create feature files
        num_samples = 5
        feature_dim = 64
        num_regions = 10
        
        for i in range(num_samples):
            # Create feature file
            feature_file = data_dir / f"sample_{i}_features.h5"
            with h5py.File(feature_file, 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_regions, feature_dim))
            
            # Create coordinate file
            coord_file = data_dir / f"sample_{i}_coords.h5"
            with h5py.File(coord_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(num_regions, 2) * 100)
        
        # Create metadata file
        metadata = {
            f"sample_{i}": {
                "patient": f"patient_{i}",
                "age_at_diagnosis": 40 + i * 5,
                "odx85": "H" if i % 2 == 0 else "L",
                "mphr": "H" if i % 3 == 0 else "L",
                "Grade": float(i % 3 + 1),
                "odx_train": 1.0 if i % 2 == 0 else -1.0
            } for i in range(num_samples)
        }
        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create feature stats file
        feature_stats = {
            "mean": np.random.randn(feature_dim).tolist(),
            "std": np.abs(np.random.randn(feature_dim)).tolist()
        }
        stats_file = data_dir / "feature_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(feature_stats, f)
        
        return data_dir
    
    @pytest.fixture
    def config_dict(self, mock_data_dir):
        """Create a configuration dictionary for testing."""
        return {
            "s3_bucket": str(mock_data_dir),  # Use local path instead of S3
            "s3_prefix": "",
            "max_dim": 32,
            "overlap": 0.0,
            "metadata_file": str(mock_data_dir / "metadata.json"),
            "cache_dir": str(mock_data_dir / "cache"),
            "batch_size": 2,
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
    
    def test_full_workflow(self, config_dict, mock_data_dir):
        """Test the complete workflow of the DataModule."""
        # Create DataModule
        data_module = RiskFormerDataModule.from_config(config_dict)
        
        # Prepare data
        data_module.prepare_data()
        
        # Setup for fit stage
        data_module.setup("fit")
        
        # Check datasets
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check batch from train loader
        batch = next(iter(train_loader))
        assert isinstance(batch, dict)
        assert "features" in batch
        assert "labels" in batch
        
        features = batch["features"]
        assert features.dim() == 4  # [B, C, H, W]
        assert features.shape[0] <= config_dict["batch_size"]
        assert features.shape[2] <= config_dict["max_dim"]
        assert features.shape[3] <= config_dict["max_dim"]
        
        labels = batch["labels"]
        assert all(task in labels for task in config_dict["tasks"])
        
        # Test teardown
        data_module.teardown("fit")
        assert data_module.train_dataset is None
        assert data_module.val_dataset is None
        
        data_module.teardown("test")
        assert data_module.test_dataset is None
    
    def test_data_splits(self, config_dict, mock_data_dir):
        """Test dataset splitting functionality."""
        data_module = RiskFormerDataModule.from_config(config_dict)
        data_module.prepare_data()
        data_module.setup("fit")
        
        # Get total number of patients
        total_patients = len(data_module.patient_examples)
        
        # Check split sizes
        expected_train = int(total_patients * (1 - config_dict["val_split"] - config_dict["test_split"]))
        expected_val = int(total_patients * config_dict["val_split"])
        expected_test = int(total_patients * config_dict["test_split"])
        
        assert len(data_module.train_dataset) >= expected_train - 1
        assert len(data_module.val_dataset) >= expected_val - 1
        assert len(data_module.test_dataset) >= expected_test - 1
        
        # Check no overlap between splits
        train_patients = set(data_module.train_dataset.patient_examples.keys())
        val_patients = set(data_module.val_dataset.patient_examples.keys())
        test_patients = set(data_module.test_dataset.patient_examples.keys())
        
        assert not train_patients & val_patients
        assert not train_patients & test_patients
        assert not val_patients & test_patients
    
    def test_reproducibility(self, config_dict, mock_data_dir):
        """Test reproducibility of data splits."""
        data_module1 = RiskFormerDataModule.from_config(config_dict)
        data_module2 = RiskFormerDataModule.from_config(config_dict)
        
        data_module1.prepare_data()
        data_module2.prepare_data()
        
        data_module1.setup("fit")
        data_module2.setup("fit")
        
        # Compare splits
        assert (set(data_module1.train_dataset.patient_examples.keys()) ==
                set(data_module2.train_dataset.patient_examples.keys()))
        assert (set(data_module1.val_dataset.patient_examples.keys()) ==
                set(data_module2.val_dataset.patient_examples.keys()))
        assert (set(data_module1.test_dataset.patient_examples.keys()) ==
                set(data_module2.test_dataset.patient_examples.keys()))

if __name__ == "__main__":
    pytest.main() 
import torch
import json
import tempfile
import os
from pathlib import Path
import h5py
import numpy as np
import shutil

from riskformer.data.datasets import RiskFormerDataModule
from tests.utils import check_aws_credentials

class TestDataModuleIntegration:
    """Integration tests for RiskFormerDataModule."""
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with feature files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create feature files
        num_samples = 5
        feature_dim = 64
        num_regions = 10
        
        for i in range(num_samples):
            # Create feature file
            feature_file = data_dir / f"sample_{i}_features.h5"
            with h5py.File(feature_file, 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_regions, feature_dim))
            
            # Create coordinate file
            coord_file = data_dir / f"sample_{i}_coords.h5"
            with h5py.File(coord_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(num_regions, 2) * 100)
        
        # Create metadata file
        metadata = {
            f"sample_{i}": {
                "patient": f"patient_{i}",
                "age_at_diagnosis": 40 + i * 5,
                "odx85": "H" if i % 2 == 0 else "L",
                "mphr": "H" if i % 3 == 0 else "L",
                "Grade": float(i % 3 + 1),
                "odx_train": 1.0 if i % 2 == 0 else -1.0
            } for i in range(num_samples)
        }
        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create feature stats file
        feature_stats = {
            "mean": np.random.randn(feature_dim).tolist(),
            "std": np.abs(np.random.randn(feature_dim)).tolist()
        }
        stats_file = data_dir / "feature_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(feature_stats, f)
        
        return data_dir
    
    @pytest.fixture
    def config_dict(self, mock_data_dir):
        """Create a configuration dictionary for testing."""
        return {
            "s3_bucket": str(mock_data_dir),  # Use local path instead of S3
            "s3_prefix": "",
            "max_dim": 32,
            "overlap": 0.0,
            "metadata_file": str(mock_data_dir / "metadata.json"),
            "cache_dir": str(mock_data_dir / "cache"),
            "profile_name": "default",
            "region_name": "us-east-1",
            "batch_size": 2,
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
    
    def test_full_workflow(self, config_dict, mock_data_dir):
        """Test the complete workflow of the DataModule."""
        # Create DataModule
        data_module = RiskFormerDataModule.from_config(config_dict)
        
        # Prepare data
        data_module.prepare_data()
        
        # Setup for fit stage
        data_module.setup("fit")
        
        # Check train dataset
        assert data_module.train_dataset is not None
        assert len(data_module.train_dataset) > 0
        
        # Check validation dataset
        assert data_module.val_dataset is not None
        assert len(data_module.val_dataset) > 0
        
        # Check test dataset
        assert data_module.test_dataset is not None
        assert len(data_module.test_dataset) > 0
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check batch from each loader
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            assert isinstance(batch, dict)
            assert "features" in batch
            assert "labels" in batch
            
            features = batch["features"]
            assert features.dim() == 4  # [B, C, H, W]
            assert features.shape[0] <= config_dict["batch_size"]
            assert features.shape[2] <= config_dict["max_dim"]
            assert features.shape[3] <= config_dict["max_dim"]
            
            labels = batch["labels"]
            assert "odx85" in labels
            assert "mphr" in labels
            assert "Grade" in labels
            
            # Check label shapes
            for label_tensor in labels.values():
                assert label_tensor.shape[0] == features.shape[0]
        
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
    
    def test_data_loading_and_caching(self, config_dict, mock_data_dir):
        """Test data loading and caching functionality."""
        # Create DataModule
        data_module = RiskFormerDataModule.from_config(config_dict)
        
        # Prepare data and check cache
        data_module.prepare_data()
        cache_dir = Path(config_dict["cache_dir"])
        assert cache_dir.exists()
        
        # Check feature stats file was created
        stats_file = cache_dir / "feature_stats.json"
        assert stats_file.exists()
        
        # Load and check feature stats
        with open(stats_file) as f:
            stats = json.load(f)
            assert "mean" in stats
            assert "std" in stats
            assert len(stats["mean"]) > 0
            assert len(stats["std"]) > 0
        
        # Setup and check cached files
        data_module.setup("fit")
        for patient_data in data_module.patient_examples.values():
            for path in patient_data["features_paths"]:
                assert Path(path).exists()
            for path in patient_data["coords_paths"]:
                assert Path(path).exists()
    
    def test_data_splits(self, config_dict, mock_data_dir):
        """Test dataset splitting functionality."""
        # Create DataModule
        data_module = RiskFormerDataModule.from_config(config_dict)
        data_module.prepare_data()
        data_module.setup("fit")
        
        # Get total number of patients
        total_patients = len(data_module.patient_examples)
        
        # Calculate expected split sizes
        expected_train = int(total_patients * (1 - config_dict["val_split"] - config_dict["test_split"]))
        expected_val = int(total_patients * config_dict["val_split"])
        expected_test = int(total_patients * config_dict["test_split"])
        
        # Check split sizes
        assert len(data_module.train_dataset) >= expected_train - 1
        assert len(data_module.val_dataset) >= expected_val - 1
        assert len(data_module.test_dataset) >= expected_test - 1
        
        # Check no overlap between splits
        train_patients = set(data_module.train_dataset.patient_examples.keys())
        val_patients = set(data_module.val_dataset.patient_examples.keys())
        test_patients = set(data_module.test_dataset.patient_examples.keys())
        
        assert not train_patients & val_patients
        assert not train_patients & test_patients
        assert not val_patients & test_patients
    
    def test_reproducibility(self, config_dict, mock_data_dir):
        """Test reproducibility of data splits."""
        # Create two DataModules with same seed
        data_module1 = RiskFormerDataModule.from_config(config_dict)
        data_module2 = RiskFormerDataModule.from_config(config_dict)
        
        # Prepare and setup both
        data_module1.prepare_data()
        data_module2.prepare_data()
        
        data_module1.setup("fit")
        data_module2.setup("fit")
        
        # Compare train splits
        train_patients1 = set(data_module1.train_dataset.patient_examples.keys())
        train_patients2 = set(data_module2.train_dataset.patient_examples.keys())
        assert train_patients1 == train_patients2
        
        # Compare validation splits
        val_patients1 = set(data_module1.val_dataset.patient_examples.keys())
        val_patients2 = set(data_module2.val_dataset.patient_examples.keys())
        assert val_patients1 == val_patients2
        
        # Compare test splits
        test_patients1 = set(data_module1.test_dataset.patient_examples.keys())
        test_patients2 = set(data_module2.test_dataset.patient_examples.keys())
        assert test_patients1 == test_patients2

if __name__ == "__main__":
    pytest.main() 
 