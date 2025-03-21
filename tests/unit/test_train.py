import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
from riskformer.utils.training_utils import create_slide_level_loss

class TestTrainFunctionsUnit:
    """Unit tests for the functions in the train.py module."""
    
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
            "batch_size": 16,
            "num_workers": 2,
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,
            "pin_memory": True,
            "experiment_name": "test_experiment",
            "early_stop": 5,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "save_top_k": 1,
            "max_epochs": 10,
            "min_epochs": 2,
            "precision": "32",
            "accelerator": "cpu",
            "devices": 1,
            "strategy": "auto",
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 10,
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
            "output_embed_dim": 128,
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
    def mock_config_file(self, mock_config_dict):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_config_dict, f)
        
        yield f.name
        os.unlink(f.name)
    
    @patch('riskformer.data.datasets.RiskFormerDataModule.from_config')
    def test_data_module_from_dict(self, mock_from_config, mock_config_dict):
        """Test creating a data module from a config dictionary."""
        mock_from_config.return_value = MagicMock(spec=RiskFormerDataModule)
        data_module = create_data_module(mock_config_dict)
        mock_from_config.assert_called_once_with(mock_config_dict)
        assert data_module is not None
        assert isinstance(data_module, MagicMock)
    
    @patch('riskformer.data.datasets.RiskFormerDataModule.from_config_file')
    def test_data_module_from_file(self, mock_from_config_file, mock_config_file):
        """Test creating a data module from a config file."""
        mock_from_config_file.return_value = MagicMock(spec=RiskFormerDataModule)
        data_module = create_data_module(mock_config_file)
        mock_from_config_file.assert_called_once_with(mock_config_file)
        assert data_module is not None
        assert isinstance(data_module, MagicMock)
    
    @patch('riskformer.training.model.RiskFormerLightningModule.from_config')
    def test_model_from_dict(self, mock_from_config, mock_config_dict):
        """Test creating a model from a config dictionary."""
        mock_from_config.return_value = MagicMock(spec=RiskFormerLightningModule)
        model = create_model(mock_config_dict)
        mock_from_config.assert_called_once_with(mock_config_dict)
        assert model is not None
        assert isinstance(model, MagicMock)
    
    @patch('riskformer.training.model.RiskFormerLightningModule.from_config_file')
    def test_model_from_file(self, mock_from_config_file, mock_config_file):
        """Test creating a model from a config file."""
        mock_from_config_file.return_value = MagicMock(spec=RiskFormerLightningModule)
        model = create_model(mock_config_file)
        mock_from_config_file.assert_called_once_with(mock_config_file)
        assert model is not None
        assert isinstance(model, MagicMock)
    
    def test_checkpoint_callback(self, mock_config_dict):
        """Test setting up a model checkpoint callback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            callback = setup_model_checkpoint_callback(
                model_dir=tmp_dir,
                experiment_name=mock_config_dict["experiment_name"],
                run_id="0001",
                monitor=mock_config_dict["monitor"],
                monitor_mode=mock_config_dict["monitor_mode"],
                save_top_k=mock_config_dict["save_top_k"]
            )
            assert callback is not None
            assert isinstance(callback, ModelCheckpoint)
            assert callback.monitor == mock_config_dict["monitor"]
            assert callback.mode == mock_config_dict["monitor_mode"]
            assert callback.save_top_k == mock_config_dict["save_top_k"]
    
    def test_callbacks(self, mock_config_dict):
        """Test creating training callbacks."""
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
            assert callbacks is not None
            assert len(callbacks) == 3
            assert isinstance(callbacks[0], ModelCheckpoint)
            assert isinstance(callbacks[1], EarlyStopping)
            assert isinstance(callbacks[2], LearningRateMonitor)
    
    def test_best_checkpoint_from_trainer(self):
        """Test getting the best checkpoint path from a trainer."""
        mock_trainer = MagicMock()
        mock_trainer.checkpoint_callback.best_model_path = "/path/to/best_model.ckpt"
        ckpt_path = get_best_ckpt(trainer=mock_trainer)
        assert ckpt_path == "/path/to/best_model.ckpt"
    
    def test_best_checkpoint_from_callbacks(self):
        """Test getting the best checkpoint path from callbacks."""
        mock_callback = MagicMock(spec=ModelCheckpoint)
        mock_callback.best_model_path = "/path/to/best_model.ckpt"
        ckpt_path = get_best_ckpt(callbacks=[mock_callback])
        assert ckpt_path == "/path/to/best_model.ckpt"
    
    def test_trainer(self, mock_config_dict):
        """Test creating a PyTorch Lightning trainer."""
        callbacks = [MagicMock(spec=ModelCheckpoint)]
        mock_logger = MagicMock()
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
            callbacks=callbacks,
            pl_logger=mock_logger,
        )
        assert trainer is not None
        assert isinstance(trainer, pl.Trainer)
    
    def test_model_train(self):
        """Test model training function."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_model = MagicMock(spec=RiskFormerLightningModule)
        mock_data_module = MagicMock(spec=RiskFormerDataModule)
        
        run_model_train(mock_trainer, mock_model, mock_data_module)
        
        mock_trainer.fit.assert_called_once_with(mock_model, mock_data_module)
    
    def test_model_test(self):
        """Test model testing function."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_model = MagicMock(spec=RiskFormerLightningModule)
        mock_data_module = MagicMock(spec=RiskFormerDataModule)
        
        run_model_test(mock_trainer, mock_model, mock_data_module)
        
        mock_trainer.test.assert_called_once_with(mock_model, mock_data_module)
    
    def test_save_model(self):
        """Test model saving function."""
        mock_model = MagicMock(spec=RiskFormerLightningModule)
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_trainer.checkpoint_callback.best_model_path = "/path/to/best_model.ckpt"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_model(mock_model, mock_trainer, tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model.ckpt"))
    
    def test_data_module_error(self):
        """Test error handling in data module creation."""
        with pytest.raises(ValueError):
            create_data_module({})  # Empty config
    
    def test_model_error(self):
        """Test error handling in model creation."""
        with pytest.raises(ValueError):
            create_model({})  # Empty config
    
    def test_trainer_error(self):
        """Test error handling in trainer creation."""
        with pytest.raises(ValueError):
            create_trainer(devices=0)  # Invalid number of devices

class TestLossFunctionsUnit:
    """Unit tests for loss functions."""
    
    @pytest.fixture
    def create_slide_level_loss_fn(self):
        """Create a slide-level loss function."""
        return create_slide_level_loss
    
    @pytest.fixture
    def binary_pred_single_instance(self):
        """Create binary prediction for single instance."""
        return {
            "predictions": torch.tensor([[0.8]]),
            "labels": torch.tensor([[1.0]]),
            "task": "odx85",
            "task_type": "binary"
        }
    
    @pytest.fixture
    def binary_pred_multi_instance(self):
        """Create binary prediction for multiple instances."""
        return {
            "predictions": torch.tensor([[0.8, 0.2, 0.9]]),
            "labels": torch.tensor([[1.0, 0.0, 1.0]]),
            "task": "odx85",
            "task_type": "binary"
        }
    
    @pytest.fixture
    def multiclass_pred_single_instance(self):
        """Create multiclass prediction for single instance."""
        return {
            "predictions": torch.tensor([[0.1, 0.7, 0.2]]),
            "labels": torch.tensor([[1]]),
            "task": "grade",
            "task_type": "multiclass"
        }
    
    @pytest.fixture
    def multitask_pred_single_instance(self):
        """Create multitask prediction for single instance."""
        return {
            "predictions": {
                "odx85": torch.tensor([[0.8]]),
                "grade": torch.tensor([[0.1, 0.7, 0.2]])
            },
            "labels": {
                "odx85": torch.tensor([[1.0]]),
                "grade": torch.tensor([[1]])
            },
            "task_types": {
                "odx85": "binary",
                "grade": "multiclass"
            }
        }
    
    def test_binary_classification_single_instance(self, create_slide_level_loss_fn, binary_pred_single_instance):
        """Test binary classification loss for single instance."""
        loss_fn = create_slide_level_loss_fn(**binary_pred_single_instance)
        loss = loss_fn()
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_binary_classification_multi_instance(self, create_slide_level_loss_fn, binary_pred_multi_instance):
        """Test binary classification loss for multiple instances."""
        loss_fn = create_slide_level_loss_fn(**binary_pred_multi_instance)
        loss = loss_fn()
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_multiclass_classification(self, multiclass_pred_single_instance):
        """Test multiclass classification loss."""
        loss_fn = create_slide_level_loss_fn(**multiclass_pred_single_instance)
        loss = loss_fn()
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_multitask_learning(self, multitask_pred_single_instance):
        """Test multitask learning loss."""
        loss_fn = create_slide_level_loss_fn(**multitask_pred_single_instance)
        loss = loss_fn()
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_regional_coefficient_effect(self, binary_pred_multi_instance):
        """Test effect of regional coefficient on loss."""
        # Test with different regional coefficients
        for reg_coef in [0.0, 0.5, 1.0]:
            binary_pred_multi_instance["regional_coefficient"] = reg_coef
            loss_fn = create_slide_level_loss_fn(**binary_pred_multi_instance)
            loss = loss_fn()
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0

if __name__ == "__main__":
    pytest.main() 