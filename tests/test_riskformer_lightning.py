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
        return RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            regional_coeff=0.1,
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
        
        # Check that loss functions were set up
        assert "risk" in lightning_module.class_loss_map
        assert isinstance(lightning_module.class_loss_map["risk"][0], nn.BCEWithLogitsLoss)
        
        # Check that metrics were initialized
        assert "risk" in lightning_module.metrics
        assert "train_acc" in lightning_module.metrics["risk"]
        assert "train_auc" in lightning_module.metrics["risk"]
    
    def test_forward(self, lightning_module, input_tensor):
        """Test the forward method."""
        # Test in eval mode
        lightning_module.eval()
        output = lightning_module(input_tensor)
        
        # Check output format
        assert isinstance(output, dict)
        assert "risk" in output
        
        # Get the risk output
        risk_output = output["risk"]
        
        # Check output shape for risk task
        assert risk_output.shape[0] > 0  # Should include at least global prediction
        assert risk_output.shape[1] == 1  # Binary task with 1 output node
    
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
    
    @patch('riskformer.training.model.slide_level_loss')
    def test_calculate_task_loss(self, mock_slide_level_loss, lightning_module):
        """Test that _calculate_task_loss correctly handles task outputs."""
        # Setup mock
        mock_slide_level_loss.return_value = torch.tensor(0.5)
        
        # Create a prediction dict like RiskFormer_ViT returns
        predictions = {"risk": torch.rand(3, 1)}  # 3 predictions (global + 2 instances)
        
        # Create labels
        labels = {"risk": torch.tensor([1.0])}
        
        # Calculate loss for training
        lightning_module.log = MagicMock()  # Mock the log method
        loss = lightning_module._calculate_task_loss(predictions, labels, "risk", "train")
        
        # Check loss is calculated correctly
        assert loss is not None
        assert loss.item() == 0.5
        
        # Check behavior when given a tuple
        # (task_outputs, attns, global_weights) as would be returned with return_weights=True
        predictions_tuple = (predictions, torch.rand(2, 2, 4), torch.rand(2, 1))
        loss_with_tuple = lightning_module._calculate_task_loss(predictions_tuple, labels, "risk", "train")
        assert loss_with_tuple is not None
        assert loss_with_tuple.item() == 0.5
    
    @patch('riskformer.training.model.slide_level_loss')
    def test_training_step(self, mock_slide_level_loss, lightning_module):
        """Test the training_step method."""
        # Setup mock
        mock_slide_level_loss.return_value = torch.tensor(0.5)
        
        # Create a batch
        batch = (
            torch.rand(2, 64, 4, 4),  # Input features
            {"labels": {"risk": torch.tensor([1.0, 0.0])}}  # Metadata with labels
        )
        
        # Mock the model's forward method
        lightning_module.model = MagicMock()
        lightning_module.model.return_value = {"risk": torch.rand(3, 1)}
        
        # Mock the log method
        lightning_module.log = MagicMock()
        
        # Call training_step
        loss = lightning_module.training_step(batch, 0)
        
        # Check loss is calculated correctly
        assert loss is not None
        assert loss.item() == 0.5
        
        # Check that log was called with total loss
        lightning_module.log.assert_any_call('train_loss', loss, 
                                             on_step=True, on_epoch=True, prog_bar=True)
    
    @patch('riskformer.training.model.slide_level_loss')
    def test_validation_step(self, mock_slide_level_loss, lightning_module):
        """Test the validation_step method."""
        # Setup mock
        mock_slide_level_loss.return_value = torch.tensor(0.5)
        
        # Create a batch
        batch = (
            torch.rand(2, 64, 4, 4),  # Input features
            {"labels": {"risk": torch.tensor([1.0, 0.0])}}  # Metadata with labels
        )
        
        # Mock the model's forward method
        lightning_module.model = MagicMock()
        lightning_module.model.return_value = {"risk": torch.rand(3, 1)}
        
        # Mock the log method
        lightning_module.log = MagicMock()
        
        # Call validation_step
        loss = lightning_module.validation_step(batch, 0)
        
        # Check loss is calculated correctly
        assert loss is not None
        assert loss.item() == 0.5
        
        # Check that log was called with total loss
        lightning_module.log.assert_any_call('val_loss', loss, 
                                             on_step=False, on_epoch=True, prog_bar=True)


if __name__ == "__main__":
    unittest.main() 