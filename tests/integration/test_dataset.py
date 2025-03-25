import pytest
import torch
import numpy as np
import h5py
import os
from os.path import join
import json
from pathlib import Path
from riskformer.data.datasets import RiskFormerDataset, S3Cache

# Set seeds for all random processes to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TestDatasetIntegration:
    """Integration tests for RiskFormerDataset."""
    
    @pytest.fixture
    def mock_data_files(self, tmp_path):
        """Create mock data files for testing."""
        # Create temporary directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        stats_file = data_dir / "feature_stats.json"
        return data_dir, stats_file
    
    def write_temp_files(self, features_shape, features_path):
        """Generate and write feature and coordinate data for testing.
        
        Args:
            features_shape: Tuple of (H, W, feature_dim)
            features_path: Path to write the features H5 file
            
        Returns:
            features_array: The generated features array
        """
        H, W, feature_dim = features_shape
        
        # Determine how many non-zero patches to create (75% of total grid)
        total_positions = H * W
        num_nonzero = int(0.75 * total_positions)
        
        # Randomly select positions for non-zero features
        nonzero_indices = np.random.choice(total_positions, size=num_nonzero, replace=False)
        
        # Create coordinates from flat indices
        rows = nonzero_indices // W
        cols = nonzero_indices % W
        coords = np.stack([rows, cols], axis=1)
        
        # Generate random feature vectors for each non-zero position
        features = np.random.random(size=(len(coords), feature_dim)) * 2
        
        # Write features to H5 file
        with h5py.File(features_path, 'w') as f:
            f.create_dataset('features', data=features)
        
        # Write coordinates to H5 file
        with h5py.File(features_path.replace('_features.h5', '_coords.h5'), 'w') as f:
            f.create_dataset('coords', data=coords)
        
        return features

    def verify_features_tensor(self, features, expected_max_dim, expected_feature_dim):
        """Helper method to verify feature tensor properties."""
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 4  # (N, C, H, W)
        assert features.shape[1] == expected_feature_dim  # Feature dimension
        assert features.shape[2] == expected_max_dim  # Height
        assert features.shape[3] == expected_max_dim  # Width

    @pytest.fixture
    def features_shape(self):
        return (10, 10, 128)

    @pytest.fixture
    def dataset_info(self, mock_data_files, features_shape):
        """Create patient examples for testing."""
        data_dir, stats_path = mock_data_files

        patient_examples = {
            "patient1": {
                "features_paths": [join(data_dir, "path1_features.h5")],
                "coords_paths": [join(data_dir, "path1_coords.h5")],
                "odx85": "H",
                "mphr": "L",
                "age_at_diagnosis": 45,
                "Grade": 3.0,
                "Mitosis": "(score = 2) 6 to 10 per 10 HPF",
                "odx_train": 1.083,
                "Disease_Free_Months": 106.0195559,
                "Necrosis": "Present",
                "ER_Status_By_IHC": "positive",
                "pr_status_by_ihc": "positive",
                "HER2Calc": "negative"
            },
            "patient2": {
                "features_paths": [join(data_dir, "path2_features.h5")],
                "coords_paths": [join(data_dir, "path2_coords.h5")],
                "odx85": "L",
                "mphr": "H",
                "age_at_diagnosis": 62,
                "Grade": 2.0,
                "Mitosis": "(score = 1) 0 to 5 per 10 HPF",
                "odx_train": -1.491,
                "Disease_Free_Months": 30.17024559,
                "Necrosis": "Absent",
                "ER_Status_By_IHC": "negative",
                "pr_status_by_ihc": "negative",
                "HER2Calc": "positive"
            }
        }

        all_features_nonzero = []
        for example in patient_examples.values():
            for path in example["features_paths"]:
                feats_nonzero = self.write_temp_files(features_shape, path)
                all_features_nonzero.append(feats_nonzero)

        feature_stats = {
            "mean": np.concatenate(all_features_nonzero).mean(axis=0).tolist(),
            "std": np.std(np.concatenate(all_features_nonzero), axis=0).tolist(),
        }
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f)

        return patient_examples, stats_path
    
    @pytest.fixture
    def feature_stats(self, mock_data_files):
        _, stats_path = mock_data_files
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def mock_s3_cache(self, tmp_path):
        """Create a mock S3Cache for testing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        s3_cache = S3Cache(str(cache_dir))
        
        # Override get_local_path to return the file path directly without real S3 interaction
        original_get_local_path = s3_cache.get_local_path
        
        def mock_get_local_path(s3_path):
            # For testing, just assume the s3_path is already the local path
            return Path(s3_path)
        
        s3_cache.get_local_path = mock_get_local_path
        
        return s3_cache
    
    def test_dataset_loading(self, dataset_info, mock_s3_cache):
        """Test loading a complete dataset."""
        patient_examples, stats_path = dataset_info
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            include_labels=["odx85", "mphr", "age_at_diagnosis", "Grade", "Mitosis", "Necrosis"],
            max_dim=16
        )
        
        # Test dataset size
        assert len(dataset) == len(patient_examples)
        
        # Test getting items
        for i in range(len(dataset)):
            features, example_data = dataset[i]
            
            # Check features shape (B, C, H, W)
            self.verify_features_tensor(features, dataset.max_dim, dataset.feature_dim)
            
            # Check example data
            assert isinstance(example_data, dict)
            assert 'labels' in example_data
            assert 'patch_info' in example_data
            assert 'patient_id' in example_data
            
            # Check labels
            labels = example_data['labels']
            assert isinstance(labels, dict)
            for label_name in ['odx85', 'mphr', 'grade', 'necrosis']:
                assert label_name in labels
            
            # Check that each label is a tensor
            for label_name, label_value in labels.items():
                assert isinstance(label_value, torch.Tensor)

    def test_dataset_with_dataloader(self, dataset_info, mock_s3_cache):
        """Test using the dataset with PyTorch DataLoader."""
        patient_examples, stats_path = dataset_info
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            include_labels=["odx85", "mphr"],
            max_dim=32
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False
        )
        
        # Test iteration
        for batch_idx, (features, example_data) in enumerate(dataloader):
            # Check features shape
            features = features[0]

            assert features.dim() == 4  # (B, C, H, W)
            assert features.shape[0] >= 1  # Batch size
            
            # Check example data
            assert 'labels' in example_data
            assert 'patient_id' in example_data
            assert 'patch_info' in example_data
            
            # Check labels
            labels = example_data['labels']
            assert "odx85" in labels
            assert "mphr" in labels
    
    def test_feature_processing(self, dataset_info, mock_s3_cache):
        """Test processing features through the dataset."""
        patient_examples, stats_path = dataset_info
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            include_labels=["odx85"],
            max_dim=16
        )
        
        # Get first patient
        patient_id = dataset.patient_ids[0]
        patient_data = dataset.patient_examples[patient_id]
        
        # Process using _create_dense_features directly
        dense_features = dataset._create_dense_features(
            coords_paths=patient_data["coords_paths"],
            features_paths=patient_data["features_paths"]
        )
        
        # Verify dense_features
        assert isinstance(dense_features, list)
        assert len(dense_features) == len(patient_data["features_paths"])
        
        # Check each dense tensor
        for tensor in dense_features:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dim() == 3  # (H, W, D)
            assert tensor.shape[2] == dataset.feature_dim
        
        # Process into patches
        patches, patch_info = dataset.split_and_pad_features(
            features_list=dense_features,
            max_dim=dataset.max_dim,
            overlap=dataset.overlap
        )
        
        # Verify patches
        assert isinstance(patches, torch.Tensor)
        assert patches.dim() == 4  # (N, H, W, D)
        assert patches.shape[1] == dataset.max_dim
        assert patches.shape[2] == dataset.max_dim
        assert patches.shape[3] == dataset.feature_dim
        
        # Verify patch info
        assert isinstance(patch_info, torch.Tensor)
        assert patch_info.dim() == 2
        assert patch_info.shape[0] == patches.shape[0]
        assert patch_info.shape[1] == 10  # Number of info columns
    
    def test_multi_slide_patient(self, dataset_info, mock_s3_cache):
        """Test handling patients with multiple slides."""
        patient_examples, stats_path = dataset_info
        
        # Modify patient1 to have multiple slides
        original_paths = patient_examples["patient1"]["features_paths"]
        patient_examples["patient1"]["features_paths"] = original_paths * 2

        original_coords_paths = patient_examples["patient1"]["coords_paths"]
        patient_examples["patient1"]["coords_paths"] = original_coords_paths * 2
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            include_labels=["odx85"],
            max_dim=16
        )
        
        # Test getting item for multi-slide patient
        features, example_data = dataset[0]  # patient1, which has multiple slides
        
        # Verify features
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 4  # (N, C, H, W)
        
        # Test getting item for single-slide patient
        features2, example_data2 = dataset[1]  # patient2, which has a single slide
        
        # Verify features
        assert isinstance(features2, torch.Tensor)
        assert features2.dim() == 4
        
        # Multi-slide patient should generally have more patches than single-slide
        assert features.shape[0] >= features2.shape[0]
    
    def test_normalization_applied(self, dataset_info, mock_s3_cache):
        """Test that normalization is correctly applied when feature_stats are provided."""
        patient_examples, stats_path = dataset_info

        # Create dataset with normalization
        dataset_with_norm = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            include_labels=["odx85"]
        )
        assert dataset_with_norm.apply_normalization is True
        assert dataset_with_norm.feature_stats_path is not None
        
        # Create dataset without normalization
        dataset_without_norm = RiskFormerDataset(
            patient_examples=patient_examples,
            s3_cache=mock_s3_cache,
            include_labels=["odx85"]
        )
        assert dataset_without_norm.apply_normalization is False
        assert dataset_without_norm.feature_stats_path is None

        # Get features from both datasets
        features_norm, _ = dataset_with_norm[0] # [B, C, H, W]
        features_raw, _ = dataset_without_norm[0] # [B, C, H, W]

        # Normalized features should have different statistics than raw features
        assert not torch.allclose(features_norm, features_raw)
        
        # Normalized features should have approximately zero mean and unit std 
        # (approximately because patching changes the statistics)
        normalized_mean = features_norm.mean(dim=(0, 2, 3))
        
        nonzero_mask = torch.abs(features_norm).sum(dim=1, keepdim=True) != 0
        normalized_std = features_norm.masked_select(nonzero_mask).reshape(-1, features_norm.size(1)).std(dim=0)
        
        # Check that means are closer to zero and stds closer to one in normalized case
        raw_mean = features_raw.mean(dim=(0, 2, 3))

        raw_nonzero_mask = torch.abs(features_raw).sum(dim=1, keepdim=True) != 0
        raw_std = features_raw.masked_select(raw_nonzero_mask).reshape(-1, features_raw.size(1)).std(dim=0)

        mean_diff_norm = torch.abs(normalized_mean).mean()
        mean_diff_raw = torch.abs(raw_mean).mean()
        
        std_diff_norm = torch.abs(normalized_std - 1.0).mean()
        std_diff_raw = torch.abs(raw_std - 1.0).mean()
        
        assert mean_diff_norm <= mean_diff_raw
        assert std_diff_norm <= std_diff_raw
        
    def test_patch_extraction_parameters(self, dataset_info, mock_s3_cache):
        """Test how different max_dim and overlap values affect patch extraction."""
        patient_examples, stats_path = dataset_info
        
        # Test with different parameter combinations
        test_params = [
            {"max_dim": 8, "overlap": 0.0},
            {"max_dim": 16, "overlap": 0.25},
            {"max_dim": 24, "overlap": 0.5}
        ]
        
        for params in test_params:
            max_dim = params["max_dim"]
            overlap = params["overlap"]
            
            # Create dataset with these parameters
            dataset = RiskFormerDataset(
                patient_examples=patient_examples,
                feature_stats_path=stats_path,
                s3_cache=mock_s3_cache,
                include_labels=["odx85"],
                max_dim=max_dim,
                overlap=overlap
            )
            
            # Get first patient features
            features, _ = dataset[0]
            
            # Verify patches have correct dimensions
            self.verify_features_tensor(features, max_dim, dataset.feature_dim)
            
            # For non-zero overlap, compare with a dataset with no overlap
            if overlap > 0.0:
                dataset_no_overlap = RiskFormerDataset(
                    patient_examples=patient_examples,
                    feature_stats_path=stats_path,
                    s3_cache=mock_s3_cache,
                    include_labels=["odx85"],
                    max_dim=max_dim,
                    overlap=0.0
                )
                
                features_no_overlap, _ = dataset_no_overlap[0]
                
                # With higher overlap, we should generally get more patches
                # (or at least not fewer, depending on image size)
                assert features.shape[0] >= features_no_overlap.shape[0], \
                    f"Expected more or equal patches with overlap={overlap} than with overlap=0.0"
                
    def test_label_mapping(self, dataset_info, mock_s3_cache):
        """Test that labels are correctly mapped to tensors."""
        patient_examples, stats_path = dataset_info
        
        # Create dataset with all labels included
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
            # No include_labels means include all labels
        )
        
        # Get first patient's labels
        _, example_data = dataset[0]
        labels = example_data['labels']
        
        # Test binary fields
        assert labels["odx85"].item() in [0.0, 1.0]
        assert labels["mphr"].item() in [0.0, 1.0]
        assert labels["er_status_by_ihc"].item() in [0.0, 1.0]
        assert labels["pr_status_by_ihc"].item() in [0.0, 1.0]
        assert labels["her2calc"].item() in [0.0, 1.0]
        assert labels["necrosis"].item() in [0.0, 1.0]
        
        # Test regression fields
        assert isinstance(labels["odx_train"].item(), float)
        assert isinstance(labels["grade"].item(), float)
        assert isinstance(labels["disease_free_months"].item(), float)
        assert isinstance(labels["age_at_diagnosis"].item(), float)
        
        # Test mitosis score is correctly extracted from text
        assert isinstance(labels["mitosis"].item(), float)
        assert labels["mitosis"].item() in [1.0, 2.0, 3.0]  # Based on the fixture data
        
        # Get second patient's labels to verify different values
        _, example_data2 = dataset[1]
        labels2 = example_data2['labels']
        
        # Verify different values for some fields
        assert labels["odx85"].item() != labels2["odx85"].item()
        assert labels["necrosis"].item() != labels2["necrosis"].item()
        assert labels["odx_train"].item() != labels2["odx_train"].item()
        
    def test_different_feature_shapes(self, mock_data_files, mock_s3_cache):
        """Test handling of different feature shapes."""
        data_dir, stats_path = mock_data_files
        
        # Define different shapes to test
        test_shapes = [
            (5, 5, 64),     # Small square grid
            (20, 20, 128),  # Larger square grid
            (8, 30, 96)     # Non-square grid
        ]
        
        for shape_idx, shape in enumerate(test_shapes):
            # Generate features and coordinates for this shape
            features_path = join(data_dir, f"shape_{shape_idx}_features.h5")
            features = self.write_temp_files(shape, features_path)
            coords_path = features_path.replace('_features.h5', '_coords.h5')
            
            # Create feature stats based on this set of features
            feature_stats = {
                "mean": features.mean(axis=0).tolist(),
                "std": np.std(features, axis=0).tolist(),
            }
            
            shape_stats_path = join(data_dir, f"shape_{shape_idx}_stats.json")
            with open(shape_stats_path, 'w') as f:
                json.dump(feature_stats, f)
            
            # Create a test patient with these features
            patient_examples = {
                f"shape_patient_{shape_idx}": {
                    "features_paths": [features_path],
                    "coords_paths": [coords_path],
                    "odx85": "H",
                    "mphr": "L"
                }
            }
            
            # Create dataset with fixed max_dim
            max_dim = 16
            dataset = RiskFormerDataset(
                patient_examples=patient_examples,
                feature_stats_path=shape_stats_path,
                s3_cache=mock_s3_cache,
                include_labels=["odx85", "mphr"],
                max_dim=max_dim
            )
            
            # Test getting features
            features_tensor, example_data = dataset[0]
            
            # Verify feature processing handled the shape correctly
            self.verify_features_tensor(features_tensor, max_dim, shape[2])
            
            # Check that the coords were properly loaded
            assert 'patch_info' in example_data
            assert example_data['patch_info'].shape[1] == 10  # Patch info has 10 columns
    
    def test_empty_dataset(self, mock_s3_cache, mock_data_files):
        """Test behavior with an empty dataset."""
        data_dir, stats_path = mock_data_files
        
        # Create empty patient examples
        empty_patient_examples = {}
        
        # Create minimal feature stats
        feature_stats = {
            "mean": np.zeros(128).tolist(),
            "std": np.ones(128).tolist()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f)
        
        # Create empty dataset
        dataset = RiskFormerDataset(
            patient_examples=empty_patient_examples,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
        )
        
        # Verify behavior
        assert len(dataset) == 0
        assert dataset.feature_dim == 1024  # Default value when no examples
        
    def test_error_handling_invalid_paths(self, mock_s3_cache, mock_data_files, dataset_info):
        """Test error handling for invalid file paths."""
        data_dir, stats_path = mock_data_files
        patient_examples, _ = dataset_info
        
        # Clone patient data but with invalid paths
        invalid_patient = {
            "patient_invalid": {
                "features_paths": [join(data_dir, "nonexistent_features.h5")],
                "coords_paths": [join(data_dir, "nonexistent_coords.h5")],
                "odx85": "H",
                "mphr": "L"
            }
        }
        
        # Create dataset
        dataset = RiskFormerDataset(
            patient_examples=invalid_patient,
            feature_stats_path=stats_path,
            s3_cache=mock_s3_cache,
        )
        
        # Verify accessing the dataset raises appropriate error
        with pytest.raises(Exception) as excinfo:
            features, _ = dataset[0]
        
        # The error should be related to file not found
        assert "No such file" in str(excinfo.value) or "not exist" in str(excinfo.value) or "Cannot open" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main() 