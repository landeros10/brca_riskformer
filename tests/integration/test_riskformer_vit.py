import pytest
import torch
import numpy as np
from riskformer.training.model import RiskFormer_ViT
from tests.test_utils import create_riskformer_vit_inputs

# Integration Tests
class TestRiskFormerIntegration:
    """Integration tests for the RiskFormer ViT model."""
    
    @pytest.fixture
    def model_config(self):
        """Create a model configuration for testing."""
        return {
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "use_phi": True,
            "drop_path_rate": 0.0,
            "drop_rate": 0.0,
            "tasks": {
                "binary_task": {
                    "type": "binary",
                    "num_classes": 1,
                    "activation": "sigmoid"
                },
                "multiclass_task": {
                    "type": "multiclass",
                    "num_classes": 3,
                    "activation": "softmax"
                }
            },
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "sinusoidal",
            "num_heads": 4,
            "use_attn_mask": True,
            "mlp_ratio": 4.0,
            "use_class_token": False,
            "attn_global_hidden_dim": 32,
            "phi_dim": 32,
            "downscale_depth": 1,
            "downscale_multiplier": 1.25,
            "downscale_stride_q": 2,
            "downscale_stride_k": 2,
            "noise_aug": 0.1,
            "attnpool_mode": "conv",
            "name": None,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotate_prob": 0.5,
            "noise_aug_prob": 0.5
        }
    
    @pytest.fixture
    def input_tensor(self):
        """Create input tensor for testing."""
        batch_size = 2
        channels = 3
        input_size = 16
        return create_riskformer_vit_inputs(
            batch_size=batch_size,
            input_size=input_size,
            channels=channels
        )
    
    def test_model_training_mode(self, model_config, input_tensor):
        """Test model behavior in training mode."""
        model = RiskFormer_ViT(**model_config)
        model.train()
        
        # Forward pass
        outputs = model(input_tensor)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert "binary_task" in outputs
        assert "multiclass_task" in outputs
        
        # Check shapes
        assert outputs["binary_task"].shape == (input_tensor.shape[0], 1)
        assert outputs["multiclass_task"].shape == (input_tensor.shape[0], 3)
        
        # Check values
        assert torch.all(outputs["binary_task"] >= 0) and torch.all(outputs["binary_task"] <= 1)
        assert torch.all(outputs["multiclass_task"] >= 0) and torch.all(outputs["multiclass_task"] <= 1)
        assert torch.allclose(outputs["multiclass_task"].sum(dim=1), torch.ones(input_tensor.shape[0]))
    
    def test_model_eval_mode(self, model_config, input_tensor):
        """Test model behavior in evaluation mode."""
        model = RiskFormer_ViT(**model_config)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert "binary_task" in outputs
        assert "multiclass_task" in outputs
        
        # Check shapes
        assert outputs["binary_task"].shape == (input_tensor.shape[0], 1)
        assert outputs["multiclass_task"].shape == (input_tensor.shape[0], 3)
        
        # Check values
        assert torch.all(outputs["binary_task"] >= 0) and torch.all(outputs["binary_task"] <= 1)
        assert torch.all(outputs["multiclass_task"] >= 0) and torch.all(outputs["multiclass_task"] <= 1)
        assert torch.allclose(outputs["multiclass_task"].sum(dim=1), torch.ones(input_tensor.shape[0]))
    
    def test_position_encoding_variations(self, model_config, input_tensor):
        """Test model with different position encoding methods."""
        encoding_methods = ["standard", "sinusoidal", "none"]
        
        for method in encoding_methods:
            config = model_config.copy()
            config["encoding_method"] = method
            model = RiskFormer_ViT(**config)
            model.eval()
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Check outputs
            assert isinstance(outputs, dict)
            assert "binary_task" in outputs
            assert "multiclass_task" in outputs
    
    def test_attention_masks(self, model_config, input_tensor):
        """Test model with and without attention masks."""
        config = model_config.copy()
        
        # Test with attention masks
        config["use_attn_mask"] = True
        model_with_mask = RiskFormer_ViT(**config)
        model_with_mask.eval()
        
        # Test without attention masks
        config["use_attn_mask"] = False
        model_without_mask = RiskFormer_ViT(**config)
        model_without_mask.eval()
        
        with torch.no_grad():
            outputs_with_mask = model_with_mask(input_tensor)
            outputs_without_mask = model_without_mask(input_tensor)
        
        # Check outputs
        assert isinstance(outputs_with_mask, dict)
        assert isinstance(outputs_without_mask, dict)
        assert "binary_task" in outputs_with_mask
        assert "binary_task" in outputs_without_mask
    
    def test_phi_variations(self, model_config, input_tensor):
        """Test model with and without phi network."""
        config = model_config.copy()
        
        # Test with phi network
        config["use_phi"] = True
        model_with_phi = RiskFormer_ViT(**config)
        model_with_phi.eval()
        
        # Test without phi network
        config["use_phi"] = False
        model_without_phi = RiskFormer_ViT(**config)
        model_without_phi.eval()
        
        with torch.no_grad():
            outputs_with_phi = model_with_phi(input_tensor)
            outputs_without_phi = model_without_phi(input_tensor)
        
        # Check outputs
        assert isinstance(outputs_with_phi, dict)
        assert isinstance(outputs_without_phi, dict)
        assert "binary_task" in outputs_with_phi
        assert "binary_task" in outputs_without_phi
    
    def test_class_token_variations(self, model_config, input_tensor):
        """Test model with and without class token."""
        config = model_config.copy()
        
        # Test with class token
        config["use_class_token"] = True
        model_with_token = RiskFormer_ViT(**config)
        model_with_token.eval()
        
        # Test without class token
        config["use_class_token"] = False
        model_without_token = RiskFormer_ViT(**config)
        model_without_token.eval()
        
        with torch.no_grad():
            outputs_with_token = model_with_token(input_tensor)
            outputs_without_token = model_without_token(input_tensor)
        
        # Check outputs
        assert isinstance(outputs_with_token, dict)
        assert isinstance(outputs_without_token, dict)
        assert "binary_task" in outputs_with_token
        assert "binary_task" in outputs_without_token
    
    def test_multiple_tasks(self, model_config, input_tensor):
        """Test model with multiple tasks."""
        # Add more tasks to the configuration
        config = model_config.copy()
        config["tasks"].update({
            "regression_task": {
                "type": "regression",
                "num_classes": 1,
                "activation": None
            }
        })
        
        model = RiskFormer_ViT(**config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert "binary_task" in outputs
        assert "multiclass_task" in outputs
        assert "regression_task" in outputs
        
        # Check shapes
        assert outputs["binary_task"].shape == (input_tensor.shape[0], 1)
        assert outputs["multiclass_task"].shape == (input_tensor.shape[0], 3)
        assert outputs["regression_task"].shape == (input_tensor.shape[0], 1)

if __name__ == "__main__":
    pytest.main() 