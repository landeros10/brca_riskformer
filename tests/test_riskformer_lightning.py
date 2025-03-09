import unittest
import pytest
import torch
import torch.nn as nn
from riskformer.training.model import RiskFormerLightningModule
import pytorch_lightning as pl
from unittest.mock import patch

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
            "num_classes": 2,
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "sinusoidal",
            "mask_num": 0,
            "mask_preglobal": False,
            "num_heads": 4,
            "use_attn_mask": True,
            "mlp_ratio": 4.0,
            "use_class_token": True,
            "global_k": 8,
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
    def class_loss_map(self):
        """Create a basic class loss map."""
        return {
            "risk": {
                0: nn.BCELoss(),
                1: nn.BCELoss(),
            }
        }
    
    @pytest.fixture
    def lightning_module(self, model_config, optimizer_config, class_loss_map):
        """Create a RiskFormerLightningModule instance."""
        return RiskFormerLightningModule(
            model_config=model_config,
            optimizer_config=optimizer_config,
            class_loss_map=class_loss_map,
            task_weights={"risk": 1.0},
            regional_coeff=0.1,
        )
    
    @pytest.fixture
    def input_tensor(self):
        """Create a dummy input tensor."""
        batch_size = 2
        height = width = 4
        channels = 64
        return torch.rand(batch_size, height * width, channels)
    
    def test_forward(self, lightning_module, input_tensor):
        """Test the forward method."""
        # Test in eval mode
        lightning_module.eval()
        output = lightning_module.model(input_tensor)
        
        # Check output shape
        assert output.shape == (input_tensor.shape[0], lightning_module.model_config["num_classes"])
        
        # Check that output sums to 1 (softmax)
        assert torch.allclose(output.sum(dim=1), torch.ones(input_tensor.shape[0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main() 