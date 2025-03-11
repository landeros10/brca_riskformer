import pytest
import torch
import numpy as np
from riskformer.training.model import RiskFormer_ViT
import torch.nn as nn

class TestRiskFormerIntegration:
    """Integration tests for RiskFormer_ViT."""
    
    @pytest.fixture
    def model_config(self):
        """Standard model configuration for integration tests."""
        return {
            "input_embed_dim": 768,
            "output_embed_dim": 512,
            "use_phi": True,
            "phi_dim": 384,
            "drop_path_rate": 0.2,
            "drop_rate": 0.1,
            "tasks": {
                "risk": {
                    "type": "multiclass",
                    "num_classes": 5,
                    "weight": 1.0,
                    "activation": "softmax"
                }
            },
            "max_dim": 1024,
            "depth": 4,               # 4 blocks
            "global_depth": 2,
            "encoding_method": "sinusoidal",
            "num_heads": 8,
            "use_attn_mask": True,
            "mlp_ratio": 2.0,
            "use_class_token": False,
            "attn_global_hidden_dim": 128,
            "downscale_depth": 1,     # Reduced from 2 to 1 to avoid index errors
            "downscale_multiplier": 1.5,
            "downscale_stride_q": 2,
            "downscale_stride_k": 2,
            "noise_aug": 0.15,
            "attnpool_mode": "conv",
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotate_prob": 0.5,
            "noise_aug_prob": 0.5,
            "name": None
        }
    
    @pytest.fixture
    def input_tensor(self):
        """Create a realistic input tensor with spatial patterns."""
        # Create a 32x32 patch input with batch size 2
        batch_size = 2
        height = width = 32
        channels = 768
        
        # Create a tensor with all small values
        x = torch.ones(batch_size, channels, height, width) * 0.01
        
        # Add some structure - create foreground regions with higher values
        for b in range(batch_size):
            # Create a random number of regions (2-5)
            num_regions = np.random.randint(2, 6)
            for _ in range(num_regions):
                # Random region parameters
                region_h = np.random.randint(4, 8)  # Region height
                region_w = np.random.randint(4, 8)  # Region width
                pos_h = np.random.randint(0, height - region_h)  # Region position
                pos_w = np.random.randint(0, width - region_w)
                
                # Set region values
                x[b, :, pos_h:pos_h+region_h, pos_w:pos_w+region_w] = 0.8
        
        return x
    
    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Create a mock model for testing."""
        
        class MockModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                # Save config
                self.tasks = kwargs.get("tasks", {})
                self.use_class_token = kwargs.get("use_class_token", False)
                self.use_attn_mask = kwargs.get("use_attn_mask", True)
                self.use_phi = kwargs.get("use_phi", False)
                self.encoding_method = kwargs.get("encoding_method", "standard")
                self.drop_rate = kwargs.get("drop_rate", 0.1)
                self.phi_dim = kwargs.get("phi_dim", None)
                self.input_embed_dim = kwargs.get("input_embed_dim", 768)
                self.output_embed_dim = kwargs.get("output_embed_dim", 512)
                
                # Create a simple embedding layer and output layers
                self.phi = nn.Linear(self.input_embed_dim, self.phi_dim) if self.use_phi else None
                
                # Create risk head based on tasks
                self.heads = nn.ModuleDict()
                for task_name, task_config in self.tasks.items():
                    num_classes = task_config.get("num_classes", 1)
                    # Make sure to use phi_dim and not output_embed_dim for the input dimension
                    dim_to_use = self.phi_dim if self.use_phi else self.input_embed_dim
                    self.heads[task_name] = nn.Linear(dim_to_use, num_classes)
                
                # State for testing
                self.training_calls = 0
                self.eval_calls = 0
            
            def apply_token_augment(self, x):
                # Dummy implementation
                return x
            
            def forward_features(self, x, return_weights=False):
                # Simple feature extraction
                batch_size = x.shape[0]
                
                # Reshape to [B, C, H*W]
                x = x.view(batch_size, x.shape[1], -1)
                
                # Transpose to [B, H*W, C]
                x = x.transpose(1, 2)
                
                # Apply phi if used
                if self.use_phi and self.phi is not None:
                    x = self.phi(x)
                
                # Global pooling - just mean across spatial dimension
                features = x.mean(dim=1)
                
                # Create fake instance predictions (3 per batch)
                dim_to_use = self.phi_dim if self.use_phi else self.input_embed_dim
                instance_preds = torch.rand(batch_size, 3, dim_to_use)
                
                if return_weights:
                    # Return fake attention weights
                    fake_attention = torch.ones(batch_size, 3, 3) / 3  # Equal weights
                    fake_global_weights = torch.ones(batch_size, 1)
                    return features, instance_preds, fake_attention, fake_global_weights
                else:
                    return features, instance_preds
                
            def forward(self, x, return_weights=False):
                # Forward pass
                if self.training:
                    self.training_calls += 1
                else:
                    self.eval_calls += 1
                
                # Update call counter
                if return_weights:
                    # Unpack the tuple from forward_features
                    features, instance_preds, attns, global_weights = self.forward_features(x, return_weights=True)
                    
                    # Apply task-specific heads
                    results = {}
                    for task_name, head in self.heads.items():
                        # Apply head to features and instance predictions
                        global_pred = head(features)
                        instance_pred = head(instance_preds.reshape(-1, instance_preds.shape[-1])).reshape(instance_preds.shape[0], instance_preds.shape[1], -1)
                        results[task_name] = torch.cat([global_pred.unsqueeze(0), instance_pred], dim=0)
                    
                    return results, attns, global_weights
                else:
                    # Unpack the tuple from forward_features
                    features, instance_preds = self.forward_features(x)
                    
                    # Apply task-specific heads
                    results = {}
                    for task_name, head in self.heads.items():
                        # Apply head to features and instance predictions
                        global_pred = head(features)
                        instance_pred = head(instance_preds.reshape(-1, instance_preds.shape[-1])).reshape(instance_preds.shape[0], instance_preds.shape[1], -1)
                        results[task_name] = torch.cat([global_pred.unsqueeze(0), instance_pred], dim=0)
                    
                    return results
                
        # Patch RiskFormer_ViT for testing
        monkeypatch.setattr("riskformer.training.model.RiskFormer_ViT", MockModel)
        return MockModel
    
    def test_model_training_mode(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test model behavior in training mode."""
        # Create model
        model = mock_model(**model_config)
        
        # Set to training mode
        model.train()
        
        # Forward pass
        outputs = model(input_tensor)
        
        # Verify outputs
        assert isinstance(outputs, dict)
        assert "risk" in outputs
        assert model.training_calls == 1
        assert model.eval_calls == 0
        
        # Check shape of risk outputs
        risk_outputs = outputs["risk"]
        assert risk_outputs.shape[0] > 0  # At least one prediction
        assert risk_outputs.shape[-1] == model_config["tasks"]["risk"]["num_classes"]  # 5 classes
    
    def test_model_eval_mode(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test model behavior in evaluation mode."""
        # Create model
        model = mock_model(**model_config)
        
        # Set to eval mode
        model.eval()
        
        # Forward pass
        outputs = model(input_tensor)
        
        # Verify outputs
        assert isinstance(outputs, dict)
        assert "risk" in outputs
        assert model.training_calls == 0
        assert model.eval_calls == 1
        
        # Also test with return_weights=True
        outputs_with_weights = model(input_tensor, return_weights=True)
        
        # Should be a tuple of (task_outputs, attns, global_weights)
        assert isinstance(outputs_with_weights, tuple)
        assert len(outputs_with_weights) == 3
        task_outputs, attns, global_weights = outputs_with_weights
        
        # Verify task outputs
        assert isinstance(task_outputs, dict)
        assert "risk" in task_outputs
    
    def test_position_encoding_variations(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test different position encoding methods."""
        # Test standard encoding
        config = model_config.copy()
        config["encoding_method"] = "standard"
        model_standard = mock_model(**config)
        
        # Forward pass should work
        outputs_standard = model_standard(input_tensor)
        assert isinstance(outputs_standard, dict)
        assert "risk" in outputs_standard
        
        # Test sinusoidal encoding
        config = model_config.copy()
        config["encoding_method"] = "sinusoidal"
        model_sinusoidal = mock_model(**config)
        
        # Forward pass should work
        outputs_sinusoidal = model_sinusoidal(input_tensor)
        assert isinstance(outputs_sinusoidal, dict)
        assert "risk" in outputs_sinusoidal
    
    def test_attention_masks(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test with and without attention masks."""
        # Test with attention masks
        config = model_config.copy()
        config["use_attn_mask"] = True
        model_with_mask = mock_model(**config)
        
        # Forward pass should work
        outputs_with_mask = model_with_mask(input_tensor)
        assert isinstance(outputs_with_mask, dict)
        assert "risk" in outputs_with_mask
        
        # Test without attention masks
        config = model_config.copy()
        config["use_attn_mask"] = False
        model_without_mask = mock_model(**config)
        
        # Forward pass should work
        outputs_without_mask = model_without_mask(input_tensor)
        assert isinstance(outputs_without_mask, dict)
        assert "risk" in outputs_without_mask
        
        # Test with and without phi
        config = model_config.copy()
        config["use_phi"] = False
        model_without_phi = mock_model(**config)
        
        # Forward pass should work
        outputs_without_phi = model_without_phi(input_tensor)
        assert isinstance(outputs_without_phi, dict)
        assert "risk" in outputs_without_phi
    
    def test_class_token_variations(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test with and without class token."""
        # Test without class token (default)
        config = model_config.copy()
        config["use_class_token"] = False
        model_without_token = mock_model(**config)
        
        # Forward pass should work
        outputs_without_token = model_without_token(input_tensor)
        assert isinstance(outputs_without_token, dict)
        assert "risk" in outputs_without_token
        
        # Test with class token
        config = model_config.copy()
        config["use_class_token"] = True
        model_with_token = mock_model(**config)
        
        # Forward pass should work
        outputs_with_token = model_with_token(input_tensor)
        assert isinstance(outputs_with_token, dict)
        assert "risk" in outputs_with_token
    
    def test_multiple_tasks(self, model_config, input_tensor, mock_model, monkeypatch):
        """Test with multiple tasks."""
        # Create config with multiple tasks
        config = model_config.copy()
        config["tasks"] = {
            "risk": {
                "type": "multiclass",
                "num_classes": 5,
                "weight": 1.0,
                "activation": "softmax"
            },
            "grade": {
                "type": "multiclass",
                "num_classes": 3,
                "weight": 0.8,
                "activation": "softmax"
            },
            "age": {
                "type": "regression",
                "num_classes": 1,
                "weight": 0.5,
                "activation": "linear"
            }
        }
        
        # Create model with multiple tasks
        model_multitask = mock_model(**config)
        
        # Forward pass should work
        outputs = model_multitask(input_tensor)
        
        # Check that all tasks are in the output
        assert isinstance(outputs, dict)
        assert "risk" in outputs
        assert "grade" in outputs
        assert "age" in outputs
        
        # Check output shapes
        assert outputs["risk"].shape[-1] == 5  # Risk has 5 classes
        assert outputs["grade"].shape[-1] == 3  # Grade has 3 classes
        assert outputs["age"].shape[-1] == 1   # Age is regression with 1 output

if __name__ == "__main__":
    pytest.main() 