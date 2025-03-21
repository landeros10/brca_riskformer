import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from riskformer.training.model import RiskFormerLightningModule

class TestRiskFormerLightningModule:
    """Unit tests for RiskFormerLightningModule."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "tasks": {
                "binary_task": {
                    "type": "binary",
                    "num_classes": 1,
                    "weight": 1.0,
                    "activation": "sigmoid"
                }
            },
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "task_loss_weights": {
                "binary_task": 1.0
            }
        }
    
    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch for testing."""
        batch_size = 2
        features = torch.randn(batch_size, 64, 16, 16)
        labels = {
            "binary_task": torch.randint(0, 2, (batch_size, 1)).float()
        }
        return {"features": features, "labels": labels}
    
    def test_initialization(self, basic_config):
        """Test model initialization."""
        model = RiskFormerLightningModule(basic_config)
        assert isinstance(model, RiskFormerLightningModule)
        assert model.learning_rate == basic_config["learning_rate"]
        assert model.weight_decay == basic_config["weight_decay"]
        assert hasattr(model, 'model')
        assert hasattr(model, 'loss_fn')
    
    def test_configure_optimizers(self, basic_config):
        """Test optimizer configuration."""
        model = RiskFormerLightningModule(basic_config)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.defaults["lr"] == basic_config["learning_rate"]
        assert optimizer.defaults["weight_decay"] == basic_config["weight_decay"]
    
    def test_loss_computation(self, basic_config, mock_batch):
        """Test loss computation with mocked model outputs."""
        model = RiskFormerLightningModule(basic_config)
        
        # Mock the model's forward pass
        batch_size = mock_batch["features"].shape[0]
        mock_outputs = {
            "binary_task": torch.rand(batch_size + 1, 1)  # +1 for global prediction
        }
        
        with patch.object(model.model, 'forward', return_value=mock_outputs):
            # Test training step
            loss = model.training_step(mock_batch, 0)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
    
    def test_step_logging(self, basic_config, mock_batch):
        """Test that metrics are properly logged in each step."""
        model = RiskFormerLightningModule(basic_config)
        
        # Mock the model's forward pass
        batch_size = mock_batch["features"].shape[0]
        mock_outputs = {
            "binary_task": torch.sigmoid(torch.rand(batch_size + 1, 1))
        }
        
        with patch.object(model.model, 'forward', return_value=mock_outputs):
            # Mock the log method
            with patch.object(model, 'log') as mock_log:
                # Test training step
                model.training_step(mock_batch, 0)
                mock_log.assert_called()
                
                # Test validation step
                model.validation_step(mock_batch, 0)
                mock_log.assert_called()
                
                # Test test step
                model.test_step(mock_batch, 0)
                mock_log.assert_called()
    
    def test_forward(self, basic_config, mock_batch):
        """Test forward pass with mocked internal model."""
        model = RiskFormerLightningModule(basic_config)
        
        # Mock the internal model's forward pass
        batch_size = mock_batch["features"].shape[0]
        mock_outputs = {
            "binary_task": torch.rand(batch_size + 1, 1)
        }
        
        with patch.object(model.model, 'forward', return_value=mock_outputs):
            outputs = model(mock_batch["features"])
            assert isinstance(outputs, dict)
            assert "binary_task" in outputs
            assert outputs["binary_task"].shape == (batch_size + 1, 1)
    
    def test_error_handling(self, basic_config):
        """Test error handling for invalid configurations."""
        # Test invalid task weight
        invalid_config = basic_config.copy()
        invalid_config["task_loss_weights"] = {
            "nonexistent_task": 1.0
        }
        
        with pytest.raises(KeyError):
            RiskFormerLightningModule(invalid_config)
        
        # Test missing required config
        incomplete_config = basic_config.copy()
        del incomplete_config["learning_rate"]
        
        with pytest.raises(KeyError):
            RiskFormerLightningModule(incomplete_config)

if __name__ == "__main__":
    pytest.main() 