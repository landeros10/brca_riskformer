import pytest
import torch
import numpy as np
import json
import tempfile
import os
from riskformer.data.datasets import RiskFormerDataset

class TestDatasetProcessingUnit:
    """Unit tests for dataset processing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "patient": "test_patient",
            "age_at_diagnosis": 65,
            "gender": "FEMALE",
            "RACE_repo": "White",
            "ethnicity": "NOT HISPANIC OR LATINO",
            "Grade": 1.0,
            "tumor_size": 35.0,
            "ER_Status_By_IHC": "positive",
            "pr_status_by_ihc": "positive",
            "HER2Calc": "negative",
            "odx_train": 0.1015,
            "odx85": "L",
            "mphr": "L",
            "Overall_Survival_Months": 86.53055857,
            "Disease_Free_Months": 86.50027929,
            "Overall_Survival_Status": "alive",
            "Neoadjuvant_Therapy": "no",
            "Adjuvant_Therapy_Administered": None,
            "2016 Histology Annotations": "Metaplastic carcinoma",
            "Lymphovascular Invasion (LVI)": "Absent",
            "Necrosis": "Absent",
            "Epithelial": 1.0,
            "Pleomorph": 2.0,
            "Mitosis": "(score = 1) 0 to 5 per 10 HPF",
            "Grade.1": 1.0,
            "Histology_Class_ICD": "Invasive carcinoma of no special type",
            "AJCC_N": "n0",
            "AJCC_T": "t2",
        "AJCC_M": "m0"
        }
    
    @pytest.fixture
    def feature_array_dim(self):
        """Create a feature array dimension for testing."""
        return 32
    
    @pytest.fixture
    def embedding_dim(self):
        """Create an embedding dimension for testing."""
        return 128

    def test_process_label_values(self, sample_data):
        """Test processing of label values."""

        # Test all labels
        dataset = RiskFormerDataset({})
        labels = dataset.process_label_values(sample_data)

        assert type(labels) == dict
        assert len(labels) > 0
        assert "patient" not in labels
        assert "gender" not in labels

        assert "odx85" in labels
        assert "mphr" in labels

        assert "ER_Status_By_IHC".lower() in labels
        assert "pr_status_by_ihc".lower() in labels
        assert "HER2Calc".lower() in labels
        assert "Necrosis".lower() in labels
        assert "Lymphovascular Invasion (LVI)".lower() in labels
        assert "Overall_Survival_Status".lower() in labels

        assert "odx_train".lower() in labels
        assert "tumor_size" in labels
        assert "Overall_Survival_Months".lower() in labels
        assert "Disease_Free_Months".lower() in labels
        assert "Epithelial".lower() in labels
        assert "Pleomorph".lower() in labels
        assert "Grade.1".lower() in labels
        assert "age_at_diagnosis" in labels

        assert "Mitosis".lower() in labels

        # Test only odx85 and mphr
        dataset = RiskFormerDataset({}, include_labels=["odx85", "mphr"])
        labels = dataset.process_label_values(sample_data)
        assert "odx85" in labels
        assert "mphr" in labels
        assert len(labels) == 2
        assert "patient" not in labels
        assert "gender" not in labels
        assert "ER_Status_By_IHC".lower() not in labels
        assert "pr_status_by_ihc".lower() not in labels
        assert "HER2Calc".lower() not in labels
        assert "Necrosis".lower() not in labels
        assert "Lymphovascular Invasion (LVI)".lower() not in labels
        assert "Overall_Survival_Status".lower() not in labels
    
    def test_create_feature_regionprops(self, feature_array_dim, embedding_dim):
        """Test creation of feature region properties."""
        # Create dummy feature data
        features = torch.randn(feature_array_dim, feature_array_dim, embedding_dim)
        
        dataset = RiskFormerDataset({}, include_labels=[])
        props = dataset._create_feature_regionprops(features)
        
        # Check that props is not empty
        assert len(props) > 0
        # Check that each prop has the expected attributes
        for prop in props:
            assert hasattr(prop, 'centroid')
            assert hasattr(prop, 'area')
            assert hasattr(prop, 'bbox')
    
    def test_normalize_features(self, feature_array_dim, embedding_dim):
        """Test feature normalization."""
        # Create dummy features
        features = torch.randn(feature_array_dim, feature_array_dim, embedding_dim)
        
        # Create feature stats
        feature_stats = {
            "mean": features.mean(dim=(0, 1)).numpy().tolist(),
            "std": features.std(dim=(0, 1)).numpy().tolist(),
        }
        tmp_dir = tempfile.mkdtemp()
        feature_stats_path = os.path.join(tmp_dir, "feature_stats.json")
        with open(feature_stats_path, "w") as f:
            json.dump(feature_stats, f)
        
        dataset = RiskFormerDataset(
            patient_examples={},
            feature_stats_path=feature_stats_path,
            include_labels=[],
        )
        normalized = dataset.normalize_features(features)
        
        # Check shape is preserved
        assert normalized.shape == features.shape
        
        # Check normalization was applied correctly
        mean = normalized.mean(dim=(0, 1))
        std = normalized.std(dim=(0, 1))
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1)
    
    def test_should_include_label(self):
        """Test label inclusion logic."""
        # Test with no include_labels restriction
        dataset = RiskFormerDataset({})
        assert dataset.should_include_label("odx85") is True
        assert dataset.should_include_label("random_field") is True
        
        # Test with specific include_labels
        dataset = RiskFormerDataset({}, include_labels=["odx85", "mphr"])
        assert dataset.should_include_label("odx85") is True
        assert dataset.should_include_label("mphr") is True
        assert dataset.should_include_label("random_field") is False
        
        # Test case-insensitivity
        dataset = RiskFormerDataset({}, include_labels=["ODX85", "MPHR"])
        assert dataset.should_include_label("odx85") is True
        assert dataset.should_include_label("mphr") is True

    def test_map_label(self, sample_data):
        """Test label mapping for different field types."""
        dataset = RiskFormerDataset({})
        
        # Test special binary fields
        odx_label = dataset.map_label("odx85", "H")
        assert odx_label.item() == 1.0
        odx_label = dataset.map_label("odx85", "L")
        assert odx_label.item() == 0.0
        
        # Test binary fields
        er_label = dataset.map_label("ER_Status_By_IHC", "positive")
        assert er_label.item() == 1.0
        necrosis_label = dataset.map_label("Necrosis", "Absent")
        assert necrosis_label.item() == 0.0
        
        # Test regression fields
        size_label = dataset.map_label("tumor_size", 35.0)
        assert size_label.item() == 35.0
        
        # Test mitosis field
        mitosis_label = dataset.map_label("Mitosis", "(score = 1) 0 to 5 per 10 HPF")
        assert mitosis_label.item() == 1.0
        
        # Test error conditions with pytest.raises
        with pytest.raises(ValueError):
            dataset.map_label("odx85", "Invalid")
        with pytest.raises(ValueError):
            dataset.map_label("Unknown_Field", "Value")
        with pytest.raises(ValueError):
            dataset.map_label("tumor_size", None)

    def test_split_and_pad_features_empty(self):
        """Test handling of empty feature lists."""
        dataset = RiskFormerDataset({}, feature_dim=128)  # Set feature_dim explicitly
        
        # Test with empty feature list
        empty_features, empty_info = dataset.split_and_pad_features([])
        
        assert empty_features.shape[0] == 0  # No patches
        assert empty_features.shape[1:] == (32, 32, dataset.feature_dim)  # Correct dimensions
        assert empty_info.shape[0] == 0  # No patch info
        assert empty_info.shape[1] == 10  # Correct number of info columns

    def test_create_single_patch(self, embedding_dim):
        """Test creation of a single patch."""
        dataset = RiskFormerDataset({}, include_labels=[], feature_dim=embedding_dim)
        
        # Create test features region
        region_features = torch.randn(20, 15, embedding_dim)
        
        # Call _create_single_patch
        patch, info = dataset._create_single_patch(
            feature_id=0,
            region_id=1,
            region_features=region_features,
            min_row=100,
            min_col=200,
            max_row=120,
            max_col=215,
            row_start=5,
            col_start=5,
            row_end=15,
            col_end=10,
            max_dim=32
        )
        
        # Check patch shape and padding
        assert patch.shape == (32, 32, embedding_dim)
        
        # Check patch info
        assert info.feature_id == 0
        assert info.region_id == 1
        assert info.region_min_row == 100
        assert info.region_min_col == 200
        assert info.patch_row_start == 5
        assert info.patch_col_start == 5

    def test_normalize_features_with_values(self):
        """Test feature normalization with specific values."""
        # Create specific feature stats
        feature_stats = {
            "mean": np.array([10.0, 20.0, 30.0]).tolist(),
            "std": np.array([2.0, 5.0, 10.0]).tolist(),
        }
        tmp_dir = tempfile.mkdtemp()
        feature_stats_path = os.path.join(tmp_dir, "feature_stats.json")
        with open(feature_stats_path, "w") as f:
            json.dump(feature_stats, f)
        
        dataset = RiskFormerDataset(
            patient_examples={},
            feature_stats_path=feature_stats_path,
            include_labels=[],
        )
        
        # Create test features with known values
        features = torch.tensor([
            [12.0, 25.0, 40.0],  # Should become [1.0, 1.0, 1.0]
            [8.0, 15.0, 20.0],   # Should become [-1.0, -1.0, -1.0]
            [10.0, 20.0, 30.0],  # Should become [0.0, 0.0, 0.0]
        ])
        
        normalized = dataset.normalize_features(features)
        
        # Check normalized values with small tolerance
        assert torch.allclose(normalized[0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)
        assert torch.allclose(normalized[1], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-5)
        assert torch.allclose(normalized[2], torch.tensor([0.0, 0.0, 0.0]), atol=1e-5)

if __name__ == "__main__":
    pytest.main() 