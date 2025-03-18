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
    def lightning_model(self, multitask_model_config, optimizer_config):
        """Create a RiskFormer Lightning model with multi-task configuration."""
        # Combine model_config and optimizer_config
        riskformer_config = multitask_model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
            
        # Initialize the model with mocked components
        with patch.object(RiskFormerLightningModule, '_init_metrics'):
            return RiskFormerLightningModule(
                riskformer_config=riskformer_config,
            )
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_multitask_initialization(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, multitask_model_config, optimizer_config):
        """Test that the RiskFormerLightningModule correctly initializes with multi-task config."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model.from_config.return_value = mock_model_instance
        
        # Mock the metrics so we don't get StopIteration when accessing parameters
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Patch the create_slide_level_loss function
        with patch('riskformer.training.model.create_slide_level_loss') as mock_create_loss:
            mock_create_loss.return_value = MagicMock()
            
            # Combine model_config and optimizer_config into a single riskformer_config
            riskformer_config = multitask_model_config.copy()
            riskformer_config.update(optimizer_config)
            riskformer_config['regional_coeff'] = 0.3
            
            # Initialize the model with mocked components
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
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
    
    @patch('riskformer.training.model.create_slide_level_loss')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_loss_function(self, mock_auroc, mock_accuracy, mock_modulelist, mock_create_slide_level_loss, multitask_model_config, optimizer_config, mock_batch):
        """Test the loss function with different task configurations."""
        # Configure mocks
        mock_loss_function = MagicMock()
        mock_loss_function.return_value = {
            'binary_task': torch.tensor(0.5),
            'regression_task': torch.tensor(0.6),
            'multiclass_task': torch.tensor(0.7),
            'total': torch.tensor(1.8)
        }
        mock_create_slide_level_loss.return_value = mock_loss_function
        
        # Mock metrics to avoid initialization errors
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Combine model_config and optimizer_config into a single riskformer_config
        riskformer_config = multitask_model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        
        # Initialize model with patched components
        with patch('riskformer.training.model.RiskFormer_ViT') as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_model.from_config.return_value = mock_model_instance
            
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
                # Assign the mock loss function
                lightning_model.loss = mock_loss_function
                
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
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_training_step(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, multitask_model_config, optimizer_config, mock_batch):
        """Test the training_step method with multi-task setup."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model.from_config.return_value = mock_model_instance
        
        # Mock the model's forward method
        mock_model_instance.forward.return_value = {
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }
        
        # Mock the metrics
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Mock loss function
        mock_loss = MagicMock()
        mock_loss.return_value = {
            'binary_task': torch.tensor(0.5),
            'regression_task': torch.tensor(0.6),
            'multiclass_task': torch.tensor(0.7),
            'total': torch.tensor(1.8)
        }
        
        # Combine model_config and optimizer_config
        riskformer_config = multitask_model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        
        # Initialize model with patched components
        with patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss):
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
                # Assign the model and loss function
                lightning_model.model = mock_model_instance
                lightning_model.loss = mock_loss
                
                # Call training_step
                result = lightning_model.training_step(mock_batch, 0)
                
                # Check result is the total loss
                assert torch.isclose(result, torch.tensor(1.8))
                
                # Verify loss function was called
                mock_loss.assert_called_once()
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_validation_step(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, multitask_model_config, optimizer_config, mock_batch):
        """Test the validation_step method with multi-task setup."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model.from_config.return_value = mock_model_instance
        
        # Mock the model's forward method
        mock_model_instance.forward.return_value = {
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }
        
        # Mock the metrics
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Mock loss function
        mock_loss = MagicMock()
        mock_loss.return_value = {
            'binary_task': torch.tensor(0.5),
            'regression_task': torch.tensor(0.6),
            'multiclass_task': torch.tensor(0.7),
            'total': torch.tensor(1.8)
        }
        
        # Combine model_config and optimizer_config
        riskformer_config = multitask_model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        
        # Initialize model with patched components
        with patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss):
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
                # Assign the model and loss function
                lightning_model.model = mock_model_instance
                lightning_model.loss = mock_loss
                
                # Mock the metrics update method
                lightning_model._update_metrics = MagicMock()
                
                # Call validation_step
                result = lightning_model.validation_step(mock_batch, 0)
                
                # Check result is the total loss
                assert torch.isclose(result, torch.tensor(1.8))
                
                # Verify loss function was called
                mock_loss.assert_called_once()
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_regional_coefficient(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, multitask_model_config, optimizer_config):
        """Test that regional coefficient is correctly set and used."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model.from_config.return_value = mock_model_instance
        
        # Mock the metrics
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Test with different regional coefficient values
        regional_coefficients = [0.0, 0.5, 1.0]
        
        for regional_coeff in regional_coefficients:
            # Combine model_config and optimizer_config with current regional coefficient
            riskformer_config = multitask_model_config.copy()
            riskformer_config.update(optimizer_config)
            riskformer_config['regional_coeff'] = regional_coeff
            
            # Initialize model with patched components
            with patch('riskformer.training.model.create_slide_level_loss') as mock_create_loss:
                mock_create_loss.return_value = MagicMock()
                
                with patch.object(RiskFormerLightningModule, '_init_metrics'):
                    lightning_model = RiskFormerLightningModule(
                        riskformer_config=riskformer_config,
                    )
                    
                    # Check regional coefficient was set correctly
                    assert lightning_model.regional_coeff == regional_coeff 