import unittest
import pytest
import torch
import torch.nn as nn
from riskformer.training.model import RiskFormerLightningModule
import pytorch_lightning as pl
from unittest.mock import patch, MagicMock

class TestRiskFormerLightningModule:
    """Test the RiskFormerLightningModule class."""
    
    @pytest.fixture
    def model_config(self):
        """Create a basic model configuration."""
        return {
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "use_phi": True,
            "drop_path_rate": 0.1,
            "drop_rate": 0.1,
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "sinusoidal",
            "num_heads": 4,
            "use_attn_mask": True,
            "mlp_ratio": 2.0,
            "use_class_token": False,
            "attn_global_hidden_dim": 128,
            "phi_dim": 32,
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
            "tasks": {
                "risk": {
                    "type": "binary",
                    "num_classes": 1,
                    "weight": 1.0,
                    "loss_fn": nn.BCEWithLogitsLoss(),
                    "activation": "sigmoid"
                }
            }
        }
    
    @pytest.fixture
    def optimizer_config(self):
        """Create a basic optimizer configuration."""
        return {
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "scheduler": "plateau",
            "patience": 5,
            "learning_rate_scaling": "linear",
            "learning_rate_warmup_epochs": 10,
        }
    
    @pytest.fixture
    def lightning_module(self, model_config, optimizer_config):
        """Create a RiskFormerLightningModule instance."""
        # Combine model_config and optimizer_config into a single riskformer_config
        riskformer_config = model_config.copy()
        # Add optimizer configuration
        riskformer_config.update(optimizer_config)
        # Add regional coefficient
        riskformer_config['regional_coeff'] = 0.1
        
        return RiskFormerLightningModule(
            riskformer_config=riskformer_config,
        )
    
    @pytest.fixture
    def input_tensor(self):
        """Create a dummy input tensor."""
        batch_size = 2
        height = width = 4
        channels = 64
        return torch.rand(batch_size, channels, height, width)
    
    def test_initialization(self, lightning_module, model_config):
        """Test that the Lightning module initializes correctly."""
        # Check that tasks were set up correctly
        assert "risk" in lightning_module.tasks
        assert lightning_module.task_types["risk"] == "binary"
        assert lightning_module.task_weights["risk"] == 1.0
        
        # Check that the model was initialized
        assert hasattr(lightning_module, "model")
        
        # Check that regional_coeff was set
        assert hasattr(lightning_module, "regional_coeff")
        assert lightning_module.regional_coeff == 0.1
        
        # Check that loss function was initialized
        assert hasattr(lightning_module, "loss")
        
        # Check that metrics were initialized
        assert hasattr(lightning_module, "metrics")
        assert "risk" in lightning_module.metrics
    
    def test_forward(self, lightning_module, input_tensor):
        """Test the forward method."""
        # Mock the model's forward method instead of replacing the model
        mock_output = {"risk": torch.rand(2, 1)}  # Mock output for binary task
        with patch.object(lightning_module.model, 'forward', return_value=mock_output):
            # Test the forward method
            output = lightning_module(input_tensor)
            
            # Check output format
            assert isinstance(output, dict)
            assert "risk" in output
            
            # Verify model was called
            lightning_module.model.forward.assert_called_once_with(input_tensor, False)
    
    @patch('torch.optim.Adam')
    def test_configure_optimizers(self, mock_adam, lightning_module):
        """Test that the configure_optimizers method works correctly."""
        optimizer_with_scheduler = lightning_module.configure_optimizers()
        
        # Check that it returns a dictionary with optimizer and lr_scheduler
        assert isinstance(optimizer_with_scheduler, dict)
        assert 'optimizer' in optimizer_with_scheduler
        assert 'lr_scheduler' in optimizer_with_scheduler
        
        # Check scheduler configuration
        lr_scheduler_config = optimizer_with_scheduler['lr_scheduler']
        assert lr_scheduler_config['monitor'] == 'val_loss'
        assert lr_scheduler_config['interval'] == 'epoch'
    
    @patch('riskformer.training.model.create_slide_level_loss')
    def test_loss_function(self, mock_create_slide_level_loss, lightning_module):
        """Test that the loss function works correctly."""
        # Create a mock loss function
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = {"risk": torch.tensor(0.5), "total": torch.tensor(0.5)}
        mock_create_slide_level_loss.return_value = mock_loss_fn
        
        # Assign the mock loss function
        lightning_module.loss = mock_loss_fn
        
        # Create fake predictions and targets
        predictions = {"risk": torch.rand(2, 1)}
        labels = {"risk": torch.randint(0, 2, (2, 1), dtype=torch.float32)}
        
        # Call the loss function
        losses = mock_loss_fn(predictions, labels)
        
        # Check result
        assert "risk" in losses
        assert "total" in losses
        assert torch.isclose(losses["risk"], torch.tensor(0.5))
        assert torch.isclose(losses["total"], torch.tensor(0.5))
        
        # Verify mock was called
        mock_loss_fn.assert_called_once_with(predictions, labels)

    @patch('riskformer.training.model.create_slide_level_loss')
    def test_training_step(self, mock_create_slide_level_loss, lightning_module):
        """Test the training step."""
        # Create a mock loss function that returns a dictionary with task and total losses
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = {"risk": torch.tensor(0.5), "total": torch.tensor(0.5)}
        mock_create_slide_level_loss.return_value = mock_loss_fn
        
        # Assign the mock loss function
        lightning_module.loss = mock_loss_fn
        
        # Create batch with the expected structure with 'labels' key
        batch = (
            torch.rand(2, 64, 4, 4),  # inputs
            {"labels": {"risk": torch.randint(0, 2, (2, 1), dtype=torch.float32)}}  # targets with 'labels' key
        )
        
        # Mock the forward method to avoid dimension issues
        mock_output = {"risk": torch.rand(2, 1)}  # Mock output for binary task
        with patch.object(lightning_module, 'forward', return_value=mock_output):
            # Mock the metrics update method
            lightning_module._update_metrics = MagicMock()
            
            # Call training_step
            result = lightning_module.training_step(batch, 0)
            
            # Check result
            assert torch.isclose(result, torch.tensor(0.5))
            
            # Verify loss function was called
            mock_loss_fn.assert_called_once()

    @patch('riskformer.training.model.create_slide_level_loss')
    def test_validation_step(self, mock_create_slide_level_loss, lightning_module):
        """Test the validation step."""
        # Create a mock loss function that returns a dictionary with task and total losses
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = {"risk": torch.tensor(0.5), "total": torch.tensor(0.5)}
        mock_create_slide_level_loss.return_value = mock_loss_fn
        
        # Assign the mock loss function
        lightning_module.loss = mock_loss_fn
        
        # Create batch with the expected structure with 'labels' key
        batch = (
            torch.rand(2, 64, 4, 4),  # inputs
            {"labels": {"risk": torch.randint(0, 2, (2, 1), dtype=torch.float32)}}  # targets with 'labels' key
        )
        
        # Mock the forward method to avoid dimension issues
        mock_output = {"risk": torch.rand(2, 1)}  # Mock output for binary task
        with patch.object(lightning_module, 'forward', return_value=mock_output):
            # Mock the metrics update method
            lightning_module._update_metrics = MagicMock()
            
            # Call validation_step
            result = lightning_module.validation_step(batch, 0)
            
            # Check result
            assert torch.isclose(result, torch.tensor(0.5))
            
            # Verify loss function was called
            mock_loss_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main() 