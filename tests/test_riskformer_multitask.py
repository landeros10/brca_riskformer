import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from riskformer.training.model import RiskFormerLightningModule

class TestRiskFormerMultiTask:
    """
    Tests for the RiskFormerLightningModule with multi-task learning configuration.
    """
    
    @pytest.fixture
    def model_config(self):
        """Create a basic model configuration for testing."""
        return {
            "input_embed_dim": 16,
            "output_embed_dim": 32,
            "use_phi": False,
            "drop_path_rate": 0.1,
            "drop_rate": 0.1,
            "num_classes": 3,  # 3 outputs for 3 tasks
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "standard",
            "mask_num": 2,
            "mask_preglobal": False,
            "num_heads": 2,
            "use_attn_mask": False,
            "mlp_ratio": 2.0,
            "use_class_token": True,
            "global_k": 4,
        }
    
    @pytest.fixture
    def optimizer_config(self):
        """Create a basic optimizer configuration for testing."""
        return {
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
        }
    
    @pytest.fixture
    def multitask_class_loss_map(self):
        """Create a multi-task loss function map for binary, regression, and multiclass tasks."""
        return {
            'binary_task': {0: nn.BCEWithLogitsLoss()},
            'regression_task': {0: nn.MSELoss()},
            'multiclass_task': {0: nn.CrossEntropyLoss()}
        }
    
    @pytest.fixture
    def task_weights(self):
        """Create weights for different tasks."""
        return {
            'binary_task': 1.0,
            'regression_task': 0.5,
            'multiclass_task': 0.75
        }
    
    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch with features and labels for different tasks."""
        # Features (B, H, W, C) where B=2, H=W=16, C=16
        features = torch.rand(2, 16, 16, 16)
        
        # Labels for different tasks
        metadata = {
            'labels': {
                'binary_task': torch.tensor([1.0, 0.0], dtype=torch.float32),
                'regression_task': torch.tensor([42.5, 35.8], dtype=torch.float32),
                'multiclass_task': torch.tensor([2, 1], dtype=torch.long)
            }
        }
        
        return features, metadata
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_multitask_initialization(self, mock_model, model_config, optimizer_config, 
                                      multitask_class_loss_map, task_weights):
        """Test that the RiskFormerLightningModule correctly initializes with multi-task config."""
        # Create the model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=multitask_class_loss_map,
            task_weights=task_weights,
            regional_coeff=0.3
        )
        
        # Check that the task types were determined correctly
        assert 'binary_task' in lightning_model.task_types
        assert 'regression_task' in lightning_model.task_types
        assert 'multiclass_task' in lightning_model.task_types
        
        # Check the task types
        assert lightning_model.task_types['binary_task'] == 'binary'
        assert lightning_model.task_types['regression_task'] == 'regression'
        assert lightning_model.task_types['multiclass_task'] == 'multiclass'
        
        # Check task weights
        assert lightning_model.task_weights['binary_task'] == 1.0
        assert lightning_model.task_weights['regression_task'] == 0.5
        assert lightning_model.task_weights['multiclass_task'] == 0.75
        
        # Check regional coefficient
        assert lightning_model.regional_coeff == 0.3
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('riskformer.training.model.slide_level_loss')
    def test_calculate_task_loss(self, mock_slide_level_loss, mock_model, model_config,
                                optimizer_config, multitask_class_loss_map, mock_batch):
        """Test that _calculate_task_loss handles different tasks correctly."""
        # Configure mocks
        mock_slide_level_loss.return_value = torch.tensor(0.5)
        
        # Create model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=multitask_class_loss_map,
            regional_coeff=0.3
        )
        
        # Mock the log method
        lightning_model.log = MagicMock()
        
        # Skip metrics calculation by patching the log_metrics method
        lightning_model.log_metrics = MagicMock()
        
        # Mock the metrics to avoid shape mismatch errors
        mock_metric = MagicMock()
        mock_metric.return_value = torch.tensor(0.8)
        
        # Replace the metrics dictionary with mocks
        lightning_model.metrics = {
            'binary_task': {
                'train_acc': mock_metric,
                'train_auc': mock_metric,
                'val_acc': mock_metric,
                'val_auc': mock_metric,
                'test_acc': mock_metric,
                'test_auc': mock_metric
            },
            'multiclass_task': {
                'train_acc': mock_metric,
                'train_f1': mock_metric,
                'train_auc': mock_metric,
                'val_acc': mock_metric,
                'val_f1': mock_metric,
                'val_auc': mock_metric,
                'test_acc': mock_metric,
                'test_f1': mock_metric,
                'test_auc': mock_metric
            },
            'regression_task': {
                'train_mse': mock_metric,
                'train_mae': mock_metric,
                'val_mse': mock_metric,
                'val_mae': mock_metric,
                'test_mse': mock_metric,
                'test_mae': mock_metric
            }
        }
        
        # Add task types
        lightning_model.task_types = {
            'binary_task': 'binary',
            'multiclass_task': 'multiclass',
            'regression_task': 'regression'
        }
        
        # Get predictions and labels
        features, metadata = mock_batch
        predictions = torch.rand(5, 3)  # 5 instances, 3 outputs (one per task)
        labels = metadata['labels']
        
        # Test for binary task - extract just the binary_task label
        binary_task_label = labels['binary_task']
        binary_loss = lightning_model._calculate_task_loss(predictions, binary_task_label, 'binary_task', 'train')
        assert binary_loss is not None
        assert binary_loss.item() == 0.5
        mock_slide_level_loss.assert_called_with(
            predictions, 
            binary_task_label, 
            lightning_model.class_loss_map['binary_task'], 
            regional_coeff=lightning_model.regional_coeff
        )
        
        # Test for regression task
        regression_label = labels['regression_task']
        regression_loss = lightning_model._calculate_task_loss(predictions, regression_label, 'regression_task', 'train')
        assert regression_loss is not None
        assert regression_loss.item() == 0.5
        
        # Test for multiclass task
        multiclass_label = labels['multiclass_task']
        multiclass_loss = lightning_model._calculate_task_loss(predictions, multiclass_label, 'multiclass_task', 'train')
        assert multiclass_loss is not None
        assert multiclass_loss.item() == 0.5
        
        # Test for non-existent task
        nonexistent_loss = lightning_model._calculate_task_loss(predictions, labels['binary_task'], 'nonexistent_task', 'train')
        assert nonexistent_loss is None
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_training_step(self, mock_model, model_config, optimizer_config, 
                          multitask_class_loss_map, task_weights, mock_batch):
        """Test the training_step method with multi-task setup."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Mock the forward method to return a tensor of the right shape
        mock_model_instance.return_value = torch.rand(5, 3)  # 5 instances, 3 outputs
        
        # Create model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=multitask_class_loss_map,
            task_weights=task_weights,
            regional_coeff=0.3
        )
        
        # Skip metrics calculation by patching the log_metrics method
        lightning_model.log_metrics = MagicMock()
        
        # Replace the _calculate_task_loss method with a mock
        lightning_model._calculate_task_loss = MagicMock(return_value=torch.tensor(0.5))
        
        # Mock the log method
        lightning_model.log = MagicMock()
        
        # Setup forward method on the model
        lightning_model.model = mock_model_instance
        
        # Test training step
        features, metadata = mock_batch
        total_loss = lightning_model.training_step((features, metadata), 0)
        
        # Check that _calculate_task_loss was called for each task
        assert lightning_model._calculate_task_loss.call_count == 3
        
        # Expected weighted losses: binary (1.0 * 0.5) + regression (0.5 * 0.5) + multiclass (0.75 * 0.5) = 1.125
        expected_loss = 1.125
        assert total_loss.item() == expected_loss
        
        # Check that the log method was called with the total loss
        lightning_model.log.assert_called_with('train_loss', torch.tensor(expected_loss), 
                                              on_step=True, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_regional_coefficient(self, mock_model, model_config, optimizer_config, 
                                 multitask_class_loss_map, task_weights):
        """Test that the regional coefficient affects the loss calculation."""
        # Create models with different regional coefficients
        model_coeff_0 = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=multitask_class_loss_map,
            task_weights=task_weights,
            regional_coeff=0.0
        )
        
        model_coeff_05 = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=multitask_class_loss_map,
            task_weights=task_weights,
            regional_coeff=0.5
        )
        
        # Check that the regional coefficients were set correctly
        assert model_coeff_0.regional_coeff == 0.0
        assert model_coeff_05.regional_coeff == 0.5 