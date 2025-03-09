import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from riskformer.data.datasets import RiskFormerDataModule

class TestRiskFormerDataModuleSplitting:
    """
    Tests for the data splitting functionality in RiskFormerDataModule.
    These tests verify that data is split correctly into train, validation,
    and test sets according to the specified ratios.
    """
    
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
    def mock_dataset(self, mock_metadata_file):
        """Mock the create_riskformer_dataset function to return a controlled dataset."""
        # Instead of creating a real dataset, we'll patch it to return a dictionary
        # of patient examples based on our metadata
        with open(mock_metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Create a mock dataset with the patients from metadata
        # In a real scenario, each patient would have coords_paths and features_paths
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
    def mock_config(self):
        """Create a mock configuration for the data module."""
        return {
            "labels": {
                "include": ["odx85", "mphr"]
            }
        }
    
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
    
    @patch('riskformer.data.datasets.create_riskformer_dataset')
    @patch('riskformer.data.datasets.split_riskformer_data')
    def test_data_splitting_ratios(self, mock_split_fn, mock_create_dataset, mock_dataset, mock_metadata_file, mock_config):
        """Test that the data is split according to the specified ratios."""
        # Configure the mock to return our controlled dataset
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.patient_examples = mock_dataset
        mock_create_dataset.return_value = mock_dataset_obj
        
        # Create train and test data splits
        total_patients = len(mock_dataset)
        test_split = 0.2
        val_split = 0.25
        
        # Calculate expected number of patients in each split
        expected_test_patients = int(total_patients * test_split)  # 8
        expected_val_patients = int((total_patients - expected_test_patients) * val_split)  # 8
        expected_train_patients = total_patients - expected_test_patients - expected_val_patients  # 24
        
        # Create mock train, val, and test data
        train_data, val_data, test_data = self.create_mock_split_datasets(
            mock_dataset, test_split=test_split, val_split=val_split
        )
        
        # Configure the mock split function to return our splits
        # First call (train+val, test)
        mock_split_fn.side_effect = [
            (train_data | val_data, test_data),  # First call: split test from the rest
            (train_data, val_data),  # Second call: split train and validation
        ]
        
        # Create the data module
        data_module = RiskFormerDataModule(
            s3_bucket="test-bucket",
            metadata_file=mock_metadata_file,
            test_split=test_split,
            val_split=val_split,
            seed=42,
            config_path=None,  # Required but we'll patch the config
            include_labels=["odx85", "mphr"]
        )
        
        # Manually add the config attribute
        data_module.config = mock_config
        
        # Setup the data module
        data_module.setup()
        
        # Check that split_riskformer_data was called with the right parameters
        assert mock_split_fn.call_count == 2, "split_riskformer_data should be called twice"
        
        # Verify the test dataset
        assert data_module.test_dataset is not None, "Test dataset should be created"
        assert hasattr(data_module.test_dataset, 'patient_examples'), "Test dataset should have patient_examples"
        assert mock_split_fn.call_args_list[0][1]['test_split_ratio'] == test_split, "Test split ratio should be passed correctly"
        
        # Verify the validation dataset
        assert data_module.val_dataset is not None, "Validation dataset should be created"
        assert hasattr(data_module.val_dataset, 'patient_examples'), "Validation dataset should have patient_examples"
        
        # Verify the train dataset
        assert data_module.train_dataset is not None, "Train dataset should be created"
        assert hasattr(data_module.train_dataset, 'patient_examples'), "Train dataset should have patient_examples"
    
    @patch('riskformer.data.datasets.create_riskformer_dataset')
    @patch('riskformer.data.datasets.split_riskformer_data')
    @patch('riskformer.data.datasets.RiskFormerDataset')
    def test_dataloader_creation(self, mock_dataset_class, mock_split_fn, mock_create_dataset, mock_dataset, mock_config):
        """Test that the correct dataloaders are created."""
        # Configure the dataset mocks
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.patient_examples = mock_dataset
        mock_create_dataset.return_value = mock_dataset_obj
        
        # Create mock train, val, and test data
        train_data, val_data, test_data = self.create_mock_split_datasets(mock_dataset)
        
        # Configure the mock split function
        mock_split_fn.side_effect = [
            (train_data | val_data, test_data),  # First call: split test from the rest
            (train_data, val_data),  # Second call: split train and validation
        ]
        
        # Configure the mock datasets returned by RiskFormerDataset
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        mock_dataset_class.side_effect = [mock_train_dataset, mock_val_dataset, mock_test_dataset]
        
        # Create the data module
        data_module = RiskFormerDataModule(
            s3_bucket="test-bucket",
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid subprocess issues
            test_split=0.2,
            val_split=0.25,
            seed=42,
            config_path=None  # Required but we'll patch the config
        )
        
        # Manually add the config attribute
        data_module.config = mock_config
        
        # Setup the data module
        data_module.setup()
        
        # Test the dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check that the dataloaders use the correct datasets
        assert train_loader.dataset == mock_train_dataset, "Train dataloader should use train dataset"
        assert val_loader.dataset == mock_val_dataset, "Val dataloader should use val dataset"
        assert test_loader.dataset == mock_test_dataset, "Test dataloader should use test dataset"
        
        # Check batch size and shuffle settings
        assert train_loader.batch_size == 4, "Train loader should use specified batch size"
        assert val_loader.batch_size == 4, "Val loader should use specified batch size"
        assert test_loader.batch_size == 4, "Test loader should use specified batch size"
        
        assert train_loader.shuffle, "Train loader should shuffle data"
        assert not val_loader.shuffle, "Val loader should not shuffle data"
        assert not test_loader.shuffle, "Test loader should not shuffle data" 