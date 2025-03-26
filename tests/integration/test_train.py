import pytest
import os
import tempfile
import json
import torch
import h5py
import numpy as np

from riskformer.training.train import (
    create_data_module,
    create_model,
    run_one_training_session
)

class TestTrainIntegration:
    """Integration tests for the training workflow."""
    
    @pytest.fixture
    def mock_config_dict(self):
        """Create a mock configuration dictionary for testing."""
        return {
            # Data parameters
            "s3_bucket": "test-bucket",
            "s3_prefix": "test-prefix",
            "max_dim": 16,  # Reduced for faster testing
            "overlap": 0.0,
            "metadata_file": "mock_metadata.json",
            "cache_dir": "/tmp/cache",
            
            # DataLoader parameters
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,
            "pin_memory": True,
            
            # Training parameters
            "experiment_name": "test_experiment",
            "early_stop": 3,
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
            
            # Logging parameters
            "use_wandb": False,
            "wandb_project": "test_project",
            "wandb_entity": None,
            
            # Model parameters
            "tasks": {
                "odx_train": {
                    "type": "regression",
                    "num_classes": 1,
                    "loss_fn": "MSELoss",
                    "weight": 1.0,
                    "metrics": ["mse", "mae"],
                    "activation": "tanh"
                }
            },
            "input_embed_dim": 64,
            "output_embed_dim": 32,
            "depth": 2,
            "global_depth": 1,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "drop_rate": 0.1,
            "use_phi": True,
            "use_class_token": True,
            "attn_global_hidden_dim": 64,
            
            # Optimizer parameters
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "plateau",
            "regional_coeff": 0.0,
            
            # Additional required parameters
            "model_dir": "./models",  # Required for checkpointing
            "log_dir": "./logs",      # Required for logging
            "debug": False,           # Required for logging setup
            "profile_name": "default", # Required for S3 access
            "region_name": "us-east-1" # Required for S3 access
        }
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create a mock data directory with feature files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create feature files
        num_samples = 5
        feature_dim = 16  # Reduced for faster testing
        num_regions = 5
        
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
                "odx_train": float(np.random.randn())  # Random regression target
            } for i in range(num_samples)
        }
        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        return data_dir

    def test_minimal_training_run(self, mock_config_dict, mock_data_dir):
        """Test a minimal training run with the actual implementation."""
        # Update config to use local paths
        mock_config_dict["s3_bucket"] = str(mock_data_dir)
        mock_config_dict["metadata_file"] = str(mock_data_dir / "metadata.json")
        mock_config_dict["cache_dir"] = str(mock_data_dir / "cache")
        
        # Create temporary directories for outputs
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "models")
            log_dir = os.path.join(tmp_dir, "logs")
            results_dir = os.path.join(tmp_dir, "results")
            
            # Update config with temporary directories
            mock_config_dict["model_dir"] = model_dir
            mock_config_dict["log_dir"] = log_dir
            
            # Run training session
            results = run_one_training_session(
                config=mock_config_dict,
                model_dir=model_dir,
                log_dir=log_dir,
                results_file_path=os.path.join(results_dir, "results.json"),
                run_id="test_run",
                validate_config=True
            )
            
            # Basic validation of outputs
            assert results is not None, "Training results are None"
            assert isinstance(results, dict), "Results should be a dictionary"
            assert "config" in results, "Results missing config"
            assert "run_id" in results, "Results missing run_id"
            assert "timestamp" in results, "Results missing timestamp"
            assert "best_checkpoint_path" in results, "Results missing best checkpoint path"
            
            # Validate checkpoint
            best_ckpt = results["best_checkpoint_path"]
            assert os.path.exists(best_ckpt), "Best checkpoint file not found"
            
            # Validate metrics
            assert any(key.startswith("test_") for key in results.keys()), "No test metrics found"
            
            # Validate results file was created
            results_file = os.path.join(results_dir, "results.json")
            assert os.path.exists(results_file), "Results file not created"
            
            # Validate results file content
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
                assert saved_results["run_id"] == "test_run", "Results file has incorrect run_id"
                assert "config" in saved_results, "Results file missing config"
                assert "timestamp" in saved_results, "Results file missing timestamp"

    def test_data_loading(self, mock_config_dict, mock_data_dir):
        """Test data loading and preprocessing."""
        # Update config to use local paths
        mock_config_dict["s3_bucket"] = str(mock_data_dir)
        mock_config_dict["metadata_file"] = str(mock_data_dir / "metadata.json")
        mock_config_dict["cache_dir"] = str(mock_data_dir / "cache")
        
        # Create and setup data module
        data_module = create_data_module(mock_config_dict)
        data_module.setup()
        
        # Basic validation
        assert len(data_module.train_dataset) > 0, "Training dataset is empty"
        assert len(data_module.val_dataset) > 0, "Validation dataset is empty"
        assert len(data_module.test_dataset) > 0, "Test dataset is empty"
        
        # Test batch loading
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        # Validate batch structure
        assert "features" in batch, "Batch missing features"
        assert "labels" in batch, "Batch missing labels"
        assert "odx_train" in batch["labels"], "Batch missing odx_train labels"
        
        # Validate tensor shapes
        assert batch["features"].shape[0] == mock_config_dict["batch_size"], "Incorrect batch size"
        assert batch["features"].shape[1] == 3, "Incorrect number of channels"
        assert batch["features"].shape[2] == mock_config_dict["max_dim"], "Incorrect feature dimension"
        assert batch["features"].shape[3] == mock_config_dict["max_dim"], "Incorrect feature dimension"
        
        # Test data types
        assert batch["features"].dtype == torch.float32, "Features should be float32"
        assert batch["labels"]["odx_train"].dtype == torch.float32, "Labels should be float32"

    def test_model_forward_pass(self, mock_config_dict):
        """Test model forward pass and loss computation."""
        # Create model
        model = create_model(mock_config_dict)
        
        # Create dummy input
        batch_size = mock_config_dict["batch_size"]
        max_dim = mock_config_dict["max_dim"]
        dummy_input = torch.randn(batch_size, 3, max_dim, max_dim)
        
        # Test forward pass
        output = model(dummy_input)
        assert output is not None, "Model forward pass failed"
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        
        # Test loss computation
        dummy_labels = torch.randn(batch_size, 1)  # For regression task
        loss = model.training_step({"features": dummy_input, "labels": {"odx_train": dummy_labels}}, 0)
        assert loss is not None, "Loss computation failed"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is infinite"
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        
        # Test optimizer configuration
        optimizer = model.configure_optimizers()
        assert optimizer is not None, "Optimizer configuration failed"
        assert isinstance(optimizer, torch.optim.Optimizer), "Invalid optimizer type"

if __name__ == "__main__":
    pytest.main() 