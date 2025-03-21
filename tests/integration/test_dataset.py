import pytest
import torch
import numpy as np
import h5py
import os
from pathlib import Path
from riskformer.data.datasets import RiskFormerDataset

class TestDatasetIntegration:
    """Integration tests for RiskFormerDataset."""
    
    @pytest.fixture
    def mock_data_files(self, tmp_path):
        """Create mock data files for testing."""
        # Create temporary directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create feature and coordinate files
        num_samples = 5
        feature_dim = 64
        num_regions = 10
        
        files = []
        for i in range(num_samples):
            # Create feature file
            feature_file = data_dir / f"sample_{i}_features.h5"
            with h5py.File(feature_file, 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_regions, feature_dim))
            files.append(str(feature_file))
            
            # Create coordinate file
            coord_file = data_dir / f"sample_{i}_coords.h5"
            with h5py.File(coord_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(num_regions, 2) * 100)
            files.append(str(coord_file))
        
        # Create feature stats file
        stats_file = data_dir / "feature_stats.json"
        feature_stats = {
            "mean": np.random.randn(feature_dim).tolist(),
            "std": np.abs(np.random.randn(feature_dim)).tolist()
        }
        with open(stats_file, 'w') as f:
            json.dump(feature_stats, f)
        
        return data_dir, files
    
    @pytest.fixture
    def patient_examples(self):
        """Create patient examples for testing."""
        return {
            "patient1": {
                "features_paths": ["path1_features.h5"],
                "coords_paths": ["path1_coords.h5"],
                "odx85": "H",
                "mphr": "L",
                "age": 45,
                "grade": 3.0,
                "mitosis_count": 2,
                "odx_train": 1.083,
                "Disease_Free_Months": 106.0195559,
                "Necrosis": "Present"
            },
            "patient2": {
                "features_paths": ["path2_features.h5"],
                "coords_paths": ["path2_coords.h5"],
                "odx85": "L",
                "mphr": "H",
                "age": 62,
                "grade": 2.0,
                "mitosis_count": 1,
                "odx_train": -1.491,
                "Disease_Free_Months": 30.17024559,
                "Necrosis": "Absent"
            }
        }
    
    def test_dataset_loading(self, mock_data_files, patient_examples):
        """Test loading a complete dataset."""
        data_dir, files = mock_data_files
        
        # Update paths in patient examples to use mock files
        for i, (patient_id, data) in enumerate(patient_examples.items()):
            data["features_paths"] = [str(data_dir / f"sample_{i}_features.h5")]
            data["coords_paths"] = [str(data_dir / f"sample_{i}_coords.h5")]
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=str(data_dir / "feature_stats.json"),
            include_labels=["odx85", "mphr", "age", "grade", "mitosis_count", "Necrosis"],
            max_dim=16
        )
        
        # Test dataset size
        assert len(dataset) == len(patient_examples)
        
        # Test getting items
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)
            assert "features" in item
            assert "labels" in item
            
            # Check features shape
            features = item["features"]
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 4  # [C, H, W, F]
            assert features.shape[1] <= dataset.max_dim
            assert features.shape[2] <= dataset.max_dim
            
            # Check labels
            labels = item["labels"]
            assert isinstance(labels, dict)
            assert all(label in labels for label in dataset.include_labels)
            assert all(isinstance(v, torch.Tensor) for v in labels.values())
    
    def test_dataset_iteration(self, mock_data_files, patient_examples):
        """Test iterating through the dataset."""
        data_dir, files = mock_data_files
        
        # Update paths in patient examples
        for i, (patient_id, data) in enumerate(patient_examples.items()):
            data["features_paths"] = [str(data_dir / f"sample_{i}_features.h5")]
            data["coords_paths"] = [str(data_dir / f"sample_{i}_coords.h5")]
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=str(data_dir / "feature_stats.json"),
            include_labels=["odx85", "mphr"],
            max_dim=16
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        # Test iteration
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert "features" in batch
            assert "labels" in batch
            
            # Check batch shapes
            features = batch["features"]
            assert features.dim() == 4
            assert features.shape[0] <= 2  # Batch size
            assert features.shape[1] <= dataset.max_dim
            assert features.shape[2] <= dataset.max_dim
            
            # Check labels
            labels = batch["labels"]
            assert isinstance(labels, dict)
            assert "odx85" in labels
            assert "mphr" in labels
            assert all(v.shape[0] == features.shape[0] for v in labels.values())
    
    def test_feature_loading_and_processing(self, mock_data_files, patient_examples):
        """Test loading and processing features from files."""
        data_dir, files = mock_data_files
        
        # Update paths in patient examples
        for i, (patient_id, data) in enumerate(patient_examples.items()):
            data["features_paths"] = [str(data_dir / f"sample_{i}_features.h5")]
            data["coords_paths"] = [str(data_dir / f"sample_{i}_coords.h5")]
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=str(data_dir / "feature_stats.json"),
            include_labels=["odx85"],
            max_dim=16
        )
        
        # Test loading features for each patient
        for patient_id, data in patient_examples.items():
            # Load features
            features = dataset.load_features(data["features_paths"][0])
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 2  # [N, F]
            
            # Load coordinates
            coords = dataset.load_coords(data["coords_paths"][0])
            assert isinstance(coords, torch.Tensor)
            assert coords.dim() == 2  # [N, 2]
            assert coords.shape[0] == features.shape[0]
            
            # Process features
            processed_features = dataset.process_features(features, coords)
            assert isinstance(processed_features, torch.Tensor)
            assert processed_features.dim() == 4  # [C, H, W, F]
            assert processed_features.shape[1] <= dataset.max_dim
            assert processed_features.shape[2] <= dataset.max_dim
    
    def test_multi_slide_patient(self, mock_data_files, patient_examples):
        """Test handling patients with multiple slides."""
        data_dir, files = mock_data_files
        
        # Modify patient1 to have multiple slides
        patient_examples["patient1"]["features_paths"] = [
            str(data_dir / "sample_0_features.h5"),
            str(data_dir / "sample_1_features.h5")
        ]
        patient_examples["patient1"]["coords_paths"] = [
            str(data_dir / "sample_0_coords.h5"),
            str(data_dir / "sample_1_coords.h5")
        ]
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=str(data_dir / "feature_stats.json"),
            include_labels=["odx85"],
            max_dim=16
        )
        
        # Test getting item for multi-slide patient
        item = dataset[0]  # patient1
        assert isinstance(item, dict)
        assert "features" in item
        assert item["features"].dim() == 4
        
        # Features should be combined from both slides
        assert item["features"].shape[1] <= dataset.max_dim
        assert item["features"].shape[2] <= dataset.max_dim

if __name__ == "__main__":
    pytest.main() 