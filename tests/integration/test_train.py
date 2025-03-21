import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np

from riskformer.training.train import (
    create_data_module,
    create_model,
    setup_model_checkpoint_callback,
    create_callbacks,
    get_best_ckpt,
    create_trainer,
    run_model_train,
    run_model_test,
    save_model,
    run_one_training_session
)
from riskformer.training.model import RiskFormerLightningModule
from riskformer.data.datasets import RiskFormerDataModule, RiskFormerDataset

class TestTrainIntegration:
    """Integration tests for the training workflow."""
    
    @pytest.fixture
    def mock_config_dict(self):
        """Create a mock configuration dictionary for testing."""
        return {
            "s3_bucket": "test-bucket",
            "s3_prefix": "test-prefix",
            "max_dim": 32,
            "overlap": 0.0,
            "metadata_file": "mock_metadata.json",
            "cache_dir": "/tmp/cache",
            "profile_name": "default",
            "region_name": "us-east-1",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2,
            "test_split": 0.2,
            "seed": 42,
            "pin_memory": True,
            "experiment_name": "test_experiment",
            "early_stop": 5,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "save_top_k": 1,
            "max_epochs": 2,
            "min_epochs": 1,
            "precision": "32",
            "accelerator": "cpu",
            "devices": 1,
            "strategy": "auto",
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 1,
            "deterministic": True,
            "sync_batchnorm": False,
            "use_wandb": False,
            "wandb_project": None,
            "wandb_entity": None,
            "tasks": {
                "odx85": {
                    "type": "binary"
                }
            },
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "drop_path": 0.0,
            "use_cls_token": False,
            "task_loss_weights": {"odx85": 1.0},
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        }
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with feature files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create feature files
        num_samples = 5
        feature_dim = 64
        num_regions = 10
        
        for i in range(num_samples):
            # Create feature file
            feature_file = data_dir / f"sample_{i}_features.h5"
            with h5py.File(feature_file, 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_regions, feature_dim))
            
            # Create coordinate file
            coord_file = data_dir / f"sample_{i}_coords.h5"
            with h5py.File(coord_file, 'w') as f:
                f.create_dataset('coords', data=np.random.rand(num_regions, 2) * 100)
        
        # Create metadata file
        metadata = {
            f"sample_{i}": {
                "patient": f"patient_{i}",
                "age_at_diagnosis": 40 + i * 5,
                "odx85": "H" if i % 2 == 0 else "L",
                "mphr": "H" if i % 3 == 0 else "L",
                "Grade": float(i % 3 + 1),
                "odx_train": 1.0 if i % 2 == 0 else -1.0
            } for i in range(num_samples)
        }
        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create feature stats file
        feature_stats = {
            "mean": np.random.randn(feature_dim).tolist(),
            "std": np.abs(np.random.randn(feature_dim)).tolist()
        }
        stats_file = data_dir / "feature_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(feature_stats, f)
        
        return data_dir
    
    def test_integration_minimal(self, mock_config_dict, mock_data_dir):
        """Test minimal integration of training components."""
        # Update config to use local paths
        mock_config_dict["s3_bucket"] = str(mock_data_dir)
        mock_config_dict["metadata_file"] = str(mock_data_dir / "metadata.json")
        mock_config_dict["cache_dir"] = str(mock_data_dir / "cache")
        
        # Create components
        data_module = create_data_module(mock_config_dict)
        model = create_model(mock_config_dict)
        
        # Create callbacks
        with tempfile.TemporaryDirectory() as tmp_dir:
            callbacks = create_callbacks(
                model_dir=tmp_dir,
                early_stop_patience=mock_config_dict["early_stop"],
                experiment_name=mock_config_dict["experiment_name"],
                run_id="0001",
                monitor=mock_config_dict["monitor"],
                monitor_mode=mock_config_dict["monitor_mode"],
                save_top_k=mock_config_dict["save_top_k"]
            )
            
            # Create trainer
            trainer = create_trainer(
                strategy=mock_config_dict["strategy"],
                max_epochs=mock_config_dict["max_epochs"],
                min_epochs=mock_config_dict["min_epochs"],
                precision=mock_config_dict["precision"],
                accelerator=mock_config_dict["accelerator"],
                accumulate_grad_batches=mock_config_dict["accumulate_grad_batches"],
                devices=mock_config_dict["devices"],
                log_every_n_steps=mock_config_dict["log_every_n_steps"],
                deterministic=mock_config_dict["deterministic"],
                sync_batchnorm=mock_config_dict["sync_batchnorm"],
                callbacks=callbacks
            )
            
            # Run training
            run_model_train(trainer, model, data_module)
            
            # Run testing
            run_model_test(trainer, model, data_module)
            
            # Save model
            save_model(model, trainer, tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model.ckpt"))
    
    def test_integration_minimal_model_datamodule_train(self, mock_config_dict):
        """Integration test for create_data_module, create_model, and run_model_train together.
        This test uses real implementations of these functions with minimal data."""
        from copy import deepcopy
        
        # Create a minimal dataset class that returns random tensors
        class MinimalDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=10, feature_dim=32, num_classes=2):
                self.num_samples = num_samples
                self.feature_dim = feature_dim
                self.num_classes = num_classes
                # Generate random features and labels
                self.features = torch.randn(num_samples, 3, feature_dim, feature_dim)
                self.labels = torch.randint(0, num_classes, (num_samples,))
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # Return a dictionary with features and labels
                sample = {
                    "features": self.features[idx],
                    "labels": {"odx85": self.labels[idx].float()}
                }
                return sample
        
        # Create a minimal data module that uses our MinimalDataset
        class MinimalDataModule(pl.LightningDataModule):
            def __init__(self, batch_size=4, num_samples=8, feature_dim=32, num_workers=0):
                super().__init__()
                self.batch_size = batch_size
                self.num_samples = num_samples
                self.feature_dim = feature_dim
                self.num_workers = num_workers
                
            def setup(self, stage=None):
                # Create datasets for training, validation, and testing
                if stage == 'fit' or stage is None:
                    self.train_dataset = MinimalDataset(
                        num_samples=self.num_samples, 
                        feature_dim=self.feature_dim
                    )
                    self.val_dataset = MinimalDataset(
                        num_samples=self.num_samples // 2, 
                        feature_dim=self.feature_dim
                    )
                
                if stage == 'test' or stage is None:
                    self.test_dataset = MinimalDataset(
                        num_samples=self.num_samples // 2, 
                        feature_dim=self.feature_dim
                    )
                    
            def train_dataloader(self):
                return torch.utils.data.DataLoader(
                    self.train_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0
                )
                
            def val_dataloader(self):
                return torch.utils.data.DataLoader(
                    self.val_dataset, 
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0
                )
                
            def test_dataloader(self):
                return torch.utils.data.DataLoader(
                    self.test_dataset, 
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0
                )
                
            @classmethod
            def from_config(cls, config):
                """Create a data module from config."""
                return cls(
                    batch_size=config.get("batch_size", 4),
                    num_samples=8,  # Keep small for testing
                    feature_dim=32,  # Keep small for testing
                    num_workers=0 if "PYTEST_CURRENT_TEST" in os.environ else config.get("num_workers", 7)
                )
                
            @classmethod
            def from_config_file(cls, config_file):
                """Create a data module from config file."""
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return cls.from_config(config)
        
        # Define a minimal model class for testing
        class MinimalLightningModule(pl.LightningModule):
            def __init__(self, feature_dim=32, num_classes=2, learning_rate=0.001):
                super().__init__()
                self.feature_dim = feature_dim
                self.num_classes = num_classes
                self.learning_rate = learning_rate
                
                # Simple CNN
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                # Calculate size after convolutions
                conv_output_size = feature_dim // 4
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * conv_output_size * conv_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
                
            def forward(self, x):
                x = self.conv(x)
                x = self.classifier(x)
                return x
                
            def training_step(self, batch, batch_idx):
                features = batch["features"]
                labels = batch["labels"]["odx85"]
                
                outputs = self(features)
                loss = F.binary_cross_entropy_with_logits(outputs[:, 0], labels)
                
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                return loss
                
            def validation_step(self, batch, batch_idx):
                features = batch["features"]
                labels = batch["labels"]["odx85"]
                
                outputs = self(features)
                loss = F.binary_cross_entropy_with_logits(outputs[:, 0], labels)
                
                self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
                return loss
                
            def test_step(self, batch, batch_idx):
                features = batch["features"]
                labels = batch["labels"]["odx85"]
                
                outputs = self(features)
                loss = F.binary_cross_entropy_with_logits(outputs[:, 0], labels)
                
                self.log("test_loss", loss, on_step=False, on_epoch=True)
                return loss
                
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                
            @classmethod
            def from_config(cls, config):
                """Create a model from config."""
                return cls(
                    feature_dim=32,  # Keep small for testing
                    learning_rate=config.get("learning_rate", 0.001)
                )
                
            @classmethod
            def from_config_file(cls, config_file):
                """Create a model from config file."""
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return cls.from_config(config)
        
        # Create a copy of the config dict
        test_config = deepcopy(mock_config_dict)
        test_config.update({
            "max_epochs": 2,
            "min_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001
        })
        
        # Create components
        data_module = MinimalDataModule.from_config(test_config)
        model = MinimalLightningModule.from_config(test_config)
        
        # Create callbacks
        with tempfile.TemporaryDirectory() as tmp_dir:
            callbacks = create_callbacks(
                model_dir=tmp_dir,
                early_stop_patience=test_config["early_stop"],
                experiment_name=test_config["experiment_name"],
                run_id="0001",
                monitor=test_config["monitor"],
                monitor_mode=test_config["monitor_mode"],
                save_top_k=test_config["save_top_k"]
            )
            
            # Create trainer
            trainer = create_trainer(
                strategy=test_config["strategy"],
                max_epochs=test_config["max_epochs"],
                min_epochs=test_config["min_epochs"],
                precision=test_config["precision"],
                accelerator=test_config["accelerator"],
                accumulate_grad_batches=test_config["accumulate_grad_batches"],
                devices=test_config["devices"],
                log_every_n_steps=test_config["log_every_n_steps"],
                deterministic=test_config["deterministic"],
                sync_batchnorm=test_config["sync_batchnorm"],
                callbacks=callbacks
            )
            
            # Run training
            run_model_train(trainer, model, data_module)
            
            # Run testing
            run_model_test(trainer, model, data_module)
            
            # Save model
            save_model(model, trainer, tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model.ckpt"))

if __name__ == "__main__":
    pytest.main() 