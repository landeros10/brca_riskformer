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
    def test_multitask_initialization(self, mock_model, model_config, optimizer_config):
        """Test that the RiskFormerLightningModule correctly initializes with multi-task config."""
        # Create the model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
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
        
        # Check that loss functions were set correctly
        assert isinstance(lightning_model.class_loss_map['binary_task'][0], nn.BCEWithLogitsLoss)
        assert isinstance(lightning_model.class_loss_map['regression_task'][0], nn.MSELoss)
        assert isinstance(lightning_model.class_loss_map['multiclass_task'][0], nn.CrossEntropyLoss)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    @patch('riskformer.training.model.slide_level_loss')
    def test_calculate_task_loss(self, mock_slide_level_loss, mock_model, model_config,
                                optimizer_config, mock_batch):
        """Test that _calculate_task_loss handles different tasks correctly."""
        # Configure mocks
        mock_slide_level_loss.return_value = torch.tensor(0.5)
        
        # Create model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            regional_coeff=0.3
        )
        
        # Mock the log method
        lightning_model.log = MagicMock()
        
        # Mock the metrics methods rather than replacing the objects
        for task in lightning_model.metrics:
            for metric_name in lightning_model.metrics[task]:
                # For each metric, patch its update and compute methods
                metric = lightning_model.metrics[task][metric_name]
                metric.update = MagicMock()
                metric.compute = MagicMock(return_value=torch.tensor(0.8))
        
        # Get labels from mock batch
        _, metadata = mock_batch
        labels = metadata['labels']
        
        # Create dictionary of task predictions
        predictions = {
            'binary_task': torch.rand(3, 1),      # 3 predictions (global + instances)
            'regression_task': torch.rand(3, 1),  # 3 predictions (global + instances)
            'multiclass_task': torch.rand(3, 3)   # 3 predictions for 3 classes
        }
        
        # Test for binary task
        binary_task_labels = {'binary_task': labels['binary_task']}
        binary_loss = lightning_model._calculate_task_loss(predictions, binary_task_labels, 'binary_task', 'train')
        assert binary_loss is not None
        assert binary_loss.item() == 0.5
        mock_slide_level_loss.assert_called_with(
            predictions['binary_task'], 
            labels['binary_task'], 
            lightning_model.class_loss_map['binary_task'], 
            regional_coeff=lightning_model.regional_coeff
        )
        
        # Test for regression task
        regression_task_labels = {'regression_task': labels['regression_task']}
        regression_loss = lightning_model._calculate_task_loss(predictions, regression_task_labels, 'regression_task', 'train')
        assert regression_loss is not None
        assert regression_loss.item() == 0.5
        
        # Test for multiclass task
        multiclass_task_labels = {'multiclass_task': labels['multiclass_task']}
        multiclass_loss = lightning_model._calculate_task_loss(predictions, multiclass_task_labels, 'multiclass_task', 'train')
        assert multiclass_loss is not None
        assert multiclass_loss.item() == 0.5
        
        # Test for non-existent task
        nonexistent_loss = lightning_model._calculate_task_loss(predictions, binary_task_labels, 'nonexistent_task', 'train')
        assert nonexistent_loss is None
        
        # Test with tuple return format (task_outputs, attns, global_weights)
        predictions_tuple = (predictions, torch.rand(2, 2, 4), torch.rand(2, 1))
        tuple_loss = lightning_model._calculate_task_loss(predictions_tuple, binary_task_labels, 'binary_task', 'train')
        assert tuple_loss is not None
        assert tuple_loss.item() == 0.5
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_training_step(self, mock_model, model_config, optimizer_config, mock_batch):
        """Test the training_step method with multi-task setup."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            regional_coeff=0.3
        )
        
        # Replace the _calculate_task_loss method with a mock
        lightning_model._calculate_task_loss = MagicMock(return_value=torch.tensor(0.5))
        
        # Mock the log method
        lightning_model.log = MagicMock()
        
        # Setup forward method on the model
        lightning_model.model = mock_model_instance
        
        # Create dictionary of task predictions for model output
        predictions = {
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }
        mock_model_instance.forward.return_value = predictions
        
        # Test training step
        features, metadata = mock_batch
        loss = lightning_model.training_step((features, metadata), 0)
        
        # Verify loss calculation
        assert loss is not None
        # With 3 tasks, each with weight and loss of 0.5: (1.0*0.5 + 0.5*0.5 + 0.75*0.5)
        # But in the mock we always return 0.5, so it's 0.5 * (1.0 + 0.5 + 0.75) = 1.125
        expected_loss = 0.5 * (1.0 + 0.5 + 0.75)
        assert loss.item() == expected_loss
        
        # Check that we logged the total loss
        lightning_model.log.assert_any_call('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_validation_step(self, mock_model, model_config, optimizer_config, mock_batch):
        """Test the validation_step method with multi-task setup."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create model
        lightning_model = RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            regional_coeff=0.3
        )
        
        # Replace the _calculate_task_loss method with a mock
        lightning_model._calculate_task_loss = MagicMock(return_value=torch.tensor(0.5))
        
        # Mock the log method
        lightning_model.log = MagicMock()
        
        # Setup forward method on the model
        lightning_model.model = mock_model_instance
        
        # Create dictionary of task predictions for model output
        predictions = {
            'binary_task': torch.rand(3, 1),
            'regression_task': torch.rand(3, 1),
            'multiclass_task': torch.rand(3, 3)
        }
        mock_model_instance.forward.return_value = predictions
        
        # Test validation step
        features, metadata = mock_batch
        loss = lightning_model.validation_step((features, metadata), 0)
        
        # Verify loss calculation
        assert loss is not None
        # With 3 tasks, each with weight and loss of 0.5: (1.0*0.5 + 0.5*0.5 + 0.75*0.5)
        expected_loss = 0.5 * (1.0 + 0.5 + 0.75)
        assert loss.item() == expected_loss
        
        # Check that we logged the total loss
        lightning_model.log.assert_any_call('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.RiskFormer_ViT')
    def test_regional_coefficient(self, mock_model, model_config, optimizer_config):
        """Test that the regional coefficient is applied correctly."""
        # Different regional coefficient values to test
        regional_coeffs = [0.0, 0.5, 1.0]
        
        for coeff in regional_coeffs:
            # Create model with this coefficient
            lightning_model = RiskFormerLightningModule(
                model_config=model_config,
                optimizer_config=optimizer_config,
                regional_coeff=coeff
            )
            
            # Check that the coefficient was set correctly
            assert lightning_model.regional_coeff == coeff 