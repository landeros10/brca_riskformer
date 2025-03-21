import pytest
import torch
import numpy as np
from riskformer.data.datasets import RiskFormerDataset

class TestDatasetProcessing:
    """Unit tests for dataset processing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "odx85": "H",
            "mphr": "L",
            "age_at_diagnosis": 45,
            "grade": 3.0,
            "mitosis_count": 2,
            "odx_train": 1.083,
            "Disease_Free_Months": 106.0195559,
            "Necrosis": "Present"
        }
    
    def test_process_special_binary_fields(self, sample_data):
        """Test processing of special binary fields like odx85 and mphr."""
        dataset = RiskFormerDataset({}, [], [], include_labels=["odx85", "mphr"])
        
        # Test odx85 processing
        odx85_value = dataset.process_binary_field(sample_data["odx85"])
        assert odx85_value == 1.0  # "H" should be encoded as 1.0
        
        odx85_value = dataset.process_binary_field("L")
        assert odx85_value == 0.0  # "L" should be encoded as 0.0
        
        # Test mphr processing
        mphr_value = dataset.process_binary_field(sample_data["mphr"])
        assert mphr_value == 0.0  # "L" should be encoded as 0.0
        
        mphr_value = dataset.process_binary_field("H")
        assert mphr_value == 1.0  # "H" should be encoded as 1.0
    
    def test_process_mitosis_field(self, sample_data):
        """Test processing of mitosis count field."""
        dataset = RiskFormerDataset({}, [], [], include_labels=["mitosis"])
        
        # Test mitosis count processing
        mitosis_value = dataset.process_mitosis_field(sample_data["mitosis_count"])
        assert isinstance(mitosis_value, float)
        assert mitosis_value >= 0.0
        assert mitosis_value <= 1.0
        
        # Test boundary values
        assert dataset.process_mitosis_field(0) == 0.0
        assert dataset.process_mitosis_field(10) == 1.0
    
    def test_process_regression_fields(self, sample_data):
        """Test processing of regression fields like age and grade."""
        dataset = RiskFormerDataset({}, [], [], include_labels=["age", "grade"])
        
        # Test age processing
        age_value = dataset.process_regression_field(
            sample_data["age_at_diagnosis"], 
            min_val=20, 
            max_val=100
        )
        assert isinstance(age_value, float)
        assert age_value >= 0.0
        assert age_value <= 1.0
        
        # Test grade processing
        grade_value = dataset.process_regression_field(
            sample_data["grade"],
            min_val=1,
            max_val=3
        )
        assert isinstance(grade_value, float)
        assert grade_value >= 0.0
        assert grade_value <= 1.0
    
    def test_process_binary_fields(self, sample_data):
        """Test processing of binary fields."""
        dataset = RiskFormerDataset({}, [], [], include_labels=["Necrosis"])
        
        # Test Necrosis processing
        necrosis_value = dataset.process_binary_field(sample_data["Necrosis"])
        assert isinstance(necrosis_value, float)
        assert necrosis_value == 1.0  # "Present" should be encoded as 1.0
        
        necrosis_value = dataset.process_binary_field("Absent")
        assert necrosis_value == 0.0  # "Absent" should be encoded as 0.0
    
    def test_create_feature_regionprops(self):
        """Test creation of feature region properties."""
        # Create dummy feature data
        features = torch.randn(10, 64)  # 10 regions, 64 features each
        
        dataset = RiskFormerDataset({}, [], [], include_labels=[])
        props = dataset._create_feature_regionprops(features)
        
        # Check that props is not empty
        assert len(props) > 0
        # Check that each prop has the expected attributes
        for prop in props:
            assert hasattr(prop, 'centroid')
            assert hasattr(prop, 'area')
            assert hasattr(prop, 'bbox')
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create dummy features
        features = torch.randn(10, 64)  # 10 regions, 64 features each
        
        # Create feature stats
        feature_stats = {
            "mean": torch.randn(64),
            "std": torch.abs(torch.randn(64))  # Make sure std is positive
        }
        
        dataset = RiskFormerDataset({}, [], [], include_labels=[])
        normalized = dataset.normalize_features(features, feature_stats)
        
        # Check shape is preserved
        assert normalized.shape == features.shape
        
        # Check normalization was applied correctly
        mean = normalized.mean(dim=0)
        std = normalized.std(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        dataset = RiskFormerDataset({}, [], [], include_labels=[])
        
        # Test invalid binary field value
        with pytest.raises(ValueError):
            dataset.process_binary_field("Invalid")
        
        # Test invalid regression field values
        with pytest.raises(ValueError):
            dataset.process_regression_field("not_a_number", 0, 1)
        
        with pytest.raises(ValueError):
            dataset.process_regression_field(1.0, 2.0, 1.0)  # min > max

if __name__ == "__main__":
    pytest.main() 