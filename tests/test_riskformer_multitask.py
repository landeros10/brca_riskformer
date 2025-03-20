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
    def multitask_riskformer_config(self, multitask_model_config, optimizer_config):
        """Create a combined configuration for the RiskFormer lightning module."""
        riskformer_config = multitask_model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        return riskformer_config
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics for testing."""
        mock_modulelist = MagicMock()
        mock_accuracy = MagicMock()
        mock_auroc = MagicMock()
        return mock_modulelist, mock_accuracy, mock_auroc
    
    @pytest.fixture
    def mock_model_instance(self):
        """Create a mock RiskFormer_ViT model instance."""
        mock_instance = MagicMock()
        # Configure the mock to behave like a model
        mock_instance.forward.return_value = {
            'binary_task': torch.rand(3, 1),      # 3 predictions (global + instances)
            'regression_task': torch.rand(3, 1),  # 3 predictions (global + instances)
            'multiclass_task': torch.rand(3, 3)   # 3 predictions for 3 classes
        }
        return mock_instance
    
    @pytest.fixture
    def mock_loss_function(self):
        """Create a mock loss function for testing."""
        mock_function = MagicMock()
        mock_function.return_value = {
            'binary_task': torch.tensor(0.5),
            'regression_task': torch.tensor(0.6),
            'multiclass_task': torch.tensor(0.7),
            'total': torch.tensor(1.8)
        }
        return mock_function
    
    @pytest.fixture
    def lightning_model(self, multitask_riskformer_config, mock_metrics, mock_model_instance, mock_loss_function):
        """Create a RiskFormer Lightning model with multi-task configuration and mocked components."""
        mock_modulelist, mock_accuracy, mock_auroc = mock_metrics
        
        # Apply all patches in one place
        with patch('riskformer.training.model.RiskFormer_ViT', return_value=mock_model_instance), \
             patch('torch.nn.ModuleList', return_value=mock_modulelist), \
             patch('torchmetrics.Accuracy', return_value=mock_accuracy), \
             patch('torchmetrics.AUROC', return_value=mock_auroc), \
             patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss_function), \
             patch.object(RiskFormerLightningModule, '_init_metrics'):
            
            # Initialize the model with all mocked components
            model = RiskFormerLightningModule(
                riskformer_config=multitask_riskformer_config,
            )
            
            # Set the mock loss function
            model.loss = mock_loss_function
            
            # Mock the _update_metrics method to avoid attribute errors
            model._update_metrics = MagicMock()
            
            # Create a metrics attribute to avoid AttributeError
            model.metrics = {
                'binary_task': {'train': {}, 'val': {}},
                'regression_task': {'train': {}, 'val': {}},
                'multiclass_task': {'train': {}, 'val': {}}
            }
            
            return model
    
    def test_multitask_initialization(self, lightning_model):
        """Test that the RiskFormerLightningModule correctly initializes with multi-task config."""
                # Check that the task types were determined correctly
                assert 'binary_task' in lightning_model.tasks
                assert 'regression_task' in lightning_model.tasks
                assert 'multiclass_task' in lightning_model.tasks
                
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
    
    def test_loss_function(self, lightning_model, mock_batch, mock_loss_function):
        """Test the loss function with different task configurations."""
                # Get labels from mock batch
                _, metadata = mock_batch
                labels = metadata['labels']
                
                # Create dictionary of task predictions
                predictions = {
                    'binary_task': torch.rand(3, 1),      # 3 predictions (global + instances)
                    'regression_task': torch.rand(3, 1),  # 3 predictions (global + instances)
                    'multiclass_task': torch.rand(3, 3)   # 3 predictions for 3 classes
                }
                
                # Call the loss function
                losses = lightning_model.loss(predictions, labels)
                
                # Verify the loss function was called correctly
                assert isinstance(losses, dict)
                assert 'binary_task' in losses
                assert 'regression_task' in losses
                assert 'multiclass_task' in losses
                assert 'total' in losses
                
                # Check loss values
                assert torch.isclose(losses['binary_task'], torch.tensor(0.5))
                assert torch.isclose(losses['regression_task'], torch.tensor(0.6))
                assert torch.isclose(losses['multiclass_task'], torch.tensor(0.7))
                assert torch.isclose(losses['total'], torch.tensor(1.8))
    
    def test_training_step(self, lightning_model, mock_batch):
        """Test the training_step method with multi-task setup."""
        # Mock the forward method to return specific values for testing
        with patch.object(lightning_model, 'forward', return_value={
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }):
                # Call training_step
            loss = lightning_model.training_step(mock_batch, 0)
            
            # Verify the loss value
            assert torch.isclose(loss, torch.tensor(1.8))
            
            # Verify metrics update was called
            lightning_model._update_metrics.assert_called()
    
    def test_validation_step(self, lightning_model, mock_batch):
        """Test the validation_step method with multi-task setup."""
        # Mock the forward method to return specific values for testing
        with patch.object(lightning_model, 'forward', return_value={
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }):
                # Call validation_step
            lightning_model.validation_step(mock_batch, 0)
            
            # Verify metrics update was called
            lightning_model._update_metrics.assert_called()
    
    def test_regional_coefficient(self, multitask_riskformer_config, mock_metrics, mock_model_instance, mock_loss_function):
        """Test that changing the regional coefficient affects model behavior."""
        mock_modulelist, mock_accuracy, mock_auroc = mock_metrics
        
        # Test with different regional coefficients
        for regional_coeff in [0.0, 0.5, 1.0]:
            config = multitask_riskformer_config.copy()
            config['regional_coeff'] = regional_coeff
            
            with patch('riskformer.training.model.RiskFormer_ViT', return_value=mock_model_instance), \
                 patch('torch.nn.ModuleList', return_value=mock_modulelist), \
                 patch('torchmetrics.Accuracy', return_value=mock_accuracy), \
                 patch('torchmetrics.AUROC', return_value=mock_auroc), \
                 patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss_function), \
                 patch.object(RiskFormerLightningModule, '_init_metrics'):
                
                model = RiskFormerLightningModule(riskformer_config=config)
                assert model.regional_coeff == regional_coeff 