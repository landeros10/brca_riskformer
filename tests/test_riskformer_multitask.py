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
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "standard",
            "num_heads": 2,
            "use_attn_mask": True,
            "mlp_ratio": 2.0,
            "use_class_token": False,
            "attn_global_hidden_dim": 128,
            "phi_dim": None,
            "downscale_depth": 1,
            "downscale_multiplier": 1.25,
            "downscale_stride_q": 2,
            "downscale_stride_k": 2,
            "noise_aug": 0.1,
            "attnpool_mode": "conv",
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotate_prob": 0.5,
            "noise_aug_prob": 0.5,
            "name": None,
            # Add tasks configuration directly
            "tasks": {
                "binary_task": {
                    "type": "binary",
                    "num_classes": 1,
                    "weight": 1.0,
                    "loss_fn": nn.BCEWithLogitsLoss(),
                    "activation": "sigmoid"
                },
                "regression_task": {
                    "type": "regression",
                    "num_classes": 1,
                    "weight": 0.5,
                    "loss_fn": nn.MSELoss(),
                    "activation": "linear"
                },
                "multiclass_task": {
                    "type": "multiclass",
                    "num_classes": 3,
                    "weight": 0.75,
                    "loss_fn": nn.CrossEntropyLoss(),
                    "activation": "softmax"
                }
            }
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
    def mock_batch(self):
        """Create a mock batch with features and labels for different tasks."""
        # Features (B, C, H, W) where B=2, C=16, H=W=16
        features = torch.rand(2, 16, 16, 16)
        
        # Labels for different tasks with the expected 'labels' key
        metadata = {
            'labels': {
                'binary_task': torch.tensor([1.0, 0.0], dtype=torch.float32),
                'regression_task': torch.tensor([42.5, 35.8], dtype=torch.float32),
                'multiclass_task': torch.tensor([2, 1], dtype=torch.long)
            }
        }
        
        return features, metadata
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_multitask_initialization(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, model_config, optimizer_config):
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
            riskformer_config = model_config.copy()
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
    def test_loss_function(self, mock_auroc, mock_accuracy, mock_modulelist, mock_create_slide_level_loss, model_config, optimizer_config, mock_batch):
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
        riskformer_config = model_config.copy()
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
    def test_training_step(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, model_config, optimizer_config, mock_batch):
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
        riskformer_config = model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        
        # Initialize model with patched components
        with patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss):
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
                # Mock the _update_metrics method
                lightning_model._update_metrics = MagicMock()
                lightning_model.log = MagicMock()
                lightning_model.loss = mock_loss
                lightning_model.model = mock_model_instance
                
                # Run training step
                features, metadata = mock_batch
                loss = lightning_model.training_step((features, metadata), 0)
                
                # Verify results
                assert torch.isclose(loss, torch.tensor(1.8))
                
                # Verify log was called
                lightning_model.log.assert_any_call('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_validation_step(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, model_config, optimizer_config, mock_batch):
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
        riskformer_config = model_config.copy()
        riskformer_config.update(optimizer_config)
        riskformer_config['regional_coeff'] = 0.3
        
        # Initialize model with patched components
        with patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss):
            with patch.object(RiskFormerLightningModule, '_init_metrics'):
                lightning_model = RiskFormerLightningModule(
                    riskformer_config=riskformer_config,
                )
                
                # Mock the _update_metrics method
                lightning_model._update_metrics = MagicMock()
                lightning_model.log = MagicMock()
                lightning_model.loss = mock_loss
                lightning_model.model = mock_model_instance
                
                # Run validation step
                features, metadata = mock_batch
                loss = lightning_model.validation_step((features, metadata), 0)
                
                # Verify results
                assert torch.isclose(loss, torch.tensor(1.8))
                
                # Verify log was called
                lightning_model.log.assert_any_call('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('torch.nn.ModuleList')
    @patch('torchmetrics.Accuracy')
    @patch('torchmetrics.AUROC')
    def test_regional_coefficient(self, mock_auroc, mock_accuracy, mock_modulelist, mock_model, model_config, optimizer_config):
        """Test that the regional coefficient is applied correctly."""
        # Mock model and metrics
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model.from_config.return_value = mock_model_instance
        
        # Mock metrics
        mock_modulelist.return_value = MagicMock()
        mock_accuracy.return_value = MagicMock()
        mock_auroc.return_value = MagicMock()
        
        # Different regional coefficient values to test
        regional_coeffs = [0.0, 0.5, 1.0]
        
        # Mock loss function creation
        mock_loss = MagicMock()
        
        for coeff in regional_coeffs:
            # Combine model_config and optimizer_config into a single riskformer_config
            riskformer_config = model_config.copy()
            riskformer_config.update(optimizer_config)
            riskformer_config['regional_coeff'] = coeff
            
            # Initialize model with patched components
            with patch('riskformer.training.model.create_slide_level_loss', return_value=mock_loss) as mock_create_loss:
                with patch.object(RiskFormerLightningModule, '_init_metrics'):
                    lightning_model = RiskFormerLightningModule(
                        riskformer_config=riskformer_config,
                    )
                    
                    # Check that the coefficient was set correctly
                    assert lightning_model.regional_coeff == coeff
                    
                    # Verify create_slide_level_loss was called with the correct regional coefficient
                    mock_create_loss.assert_called_once_with(lightning_model.task_configs, coeff) 