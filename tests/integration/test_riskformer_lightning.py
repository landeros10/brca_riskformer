import pytest
import torch
import pytorch_lightning as pl
from riskformer.training.model import RiskFormerLightningModule

class TestRiskFormerLightningIntegration:
    """Integration tests for RiskFormerLightningModule."""
    
    @pytest.fixture
    def model_config(self):
        """Full model configuration for integration testing."""
        return {
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "use_phi": True,
            "phi_dim": 32,
            "drop_path_rate": 0.1,
            "drop_rate": 0.1,
            "tasks": {
                "binary_task": {
                    "type": "binary",
                    "num_classes": 1,
                    "weight": 1.0,
                    "activation": "sigmoid"
                },
                "regression_task": {
                    "type": "regression",
                    "num_classes": 1,
                    "weight": 0.5,
                    "activation": None
                }
            },
            "max_dim": 16,
            "depth": 2,
            "global_depth": 1,
            "encoding_method": "sinusoidal",
            "num_heads": 4,
            "use_attn_mask": True,
            "mlp_ratio": 2.0,
            "use_class_token": False,
            "attn_global_hidden_dim": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "task_loss_weights": {
                "binary_task": 1.0,
                "regression_task": 0.5
            }
        }
    
    @pytest.fixture
    def real_batch(self):
        """Create a real batch with appropriate dimensions."""
        batch_size = 2
        features = torch.randn(batch_size, 64, 16, 16)
        labels = {
            "binary_task": torch.randint(0, 2, (batch_size, 1)).float(),
            "regression_task": torch.randn(batch_size, 1)
        }
        return {"features": features, "labels": labels}
    
    def test_full_training_loop(self, model_config, real_batch):
        """Test a complete training loop with real data."""
        # Create model
        model = RiskFormerLightningModule(model_config)
        
        # Create trainer with minimal epochs
        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu"
        )
        
        # Create a small dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.features = batch["features"]
                self.labels = batch["labels"]
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return {
                    "features": self.features[idx],
                    "labels": {k: v[idx] for k, v in self.labels.items()}
                }
        
        dataset = SimpleDataset(real_batch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        
        # Train for one epoch
        trainer.fit(model, dataloader)
        
        # Test predictions
        model.eval()
        with torch.no_grad():
            outputs = model(real_batch["features"])
            
            # Check outputs
            assert isinstance(outputs, dict)
            assert "binary_task" in outputs
            assert "regression_task" in outputs
            
            # Check shapes
            batch_size = real_batch["features"].shape[0]
            assert outputs["binary_task"].shape[0] == batch_size + 1  # +1 for global prediction
            assert outputs["regression_task"].shape[0] == batch_size + 1
    
    def test_multi_task_training(self, model_config, real_batch):
        """Test training with multiple tasks."""
        model = RiskFormerLightningModule(model_config)
        
        # Test training step
        loss = model.training_step(real_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Test validation step
        val_result = model.validation_step(real_batch, 0)
        assert isinstance(val_result, dict)
        assert "val_loss" in val_result
        assert not torch.isnan(val_result["val_loss"])
        
        # Test test step
        test_result = model.test_step(real_batch, 0)
        assert isinstance(test_result, dict)
        assert "test_loss" in test_result
        assert not torch.isnan(test_result["test_loss"])
    
    def test_model_checkpointing(self, model_config, real_batch, tmp_path):
        """Test model saving and loading."""
        # Create and train model
        model = RiskFormerLightningModule(model_config)
        
        # Create trainer with checkpointing
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = pl.Trainer(
            max_epochs=1,
            default_root_dir=str(checkpoint_dir),
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu"
        )
        
        # Create minimal dataset
        dataset = torch.utils.data.TensorDataset(
            real_batch["features"],
            real_batch["labels"]["binary_task"]
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        # Train and save
        trainer.fit(model, dataloader)
        
        # Load from checkpoint
        checkpoint_path = list(checkpoint_dir.glob("**/*.ckpt"))[0]
        loaded_model = RiskFormerLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=model_config
        )
        
        # Verify loaded model works
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(real_batch["features"])
            assert isinstance(outputs, dict)
            assert "binary_task" in outputs
            assert "regression_task" in outputs
    
    def test_learning_rate_scheduling(self, model_config, real_batch):
        """Test learning rate scheduling during training."""
        # Modify config to include lr scheduler
        config_with_scheduler = model_config.copy()
        config_with_scheduler["lr_scheduler"] = {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.1
        }
        
        model = RiskFormerLightningModule(config_with_scheduler)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu"
        )
        
        # Create minimal dataset
        dataset = torch.utils.data.TensorDataset(
            real_batch["features"],
            real_batch["labels"]["binary_task"]
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        # Train for two epochs to see lr change
        trainer.fit(model, dataloader)
        
        # Verify learning rate was updated
        optimizer = model.optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr < model_config["learning_rate"]

if __name__ == "__main__":
    pytest.main() 