#!/usr/bin/env python
'''
Created June 2023
author: landeros10
Updated with PyTorch Lightning support

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital

Massachusetts Institute of Technology

Core Training Module - Contains reusable training logic
'''
from __future__ import division, print_function

import os
import logging
from typing import Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from riskformer.training.model import RiskFormerLightningModule
from riskformer.data.datasets import RiskFormerDataModule
from riskformer.utils.logger_config import log_event

logger = logging.getLogger(__name__)

SIZE = 256

def create_data_module(
        config: Union[str, dict]
) -> RiskFormerDataModule:
    """Create data module from config
    
    Args:
        config (Union[str, dict]): Path to the configuration file or dictionary
        
    Returns:
        RiskFormerDataModule: Data module for training
    """
    try:
        if isinstance(config, str):
            data_module = RiskFormerDataModule.from_config_file(config)
        elif isinstance(config, dict):
            data_module = RiskFormerDataModule.from_config(config)
            
        log_event("info", "data_module_created", "Data module created successfully")
        return data_module
    except Exception as e:
        log_event("error", "data_module_creation_failed", "Failed to create data module", error_message=str(e))
        raise

def create_model(
        config: Union[str, dict]
) -> RiskFormerLightningModule:
    """Create model from config
    
    Args:
        config (Union[str, dict]): Path to the configuration file or dictionary
        
    Returns:
        RiskFormerLightningModule: Lightning module for training
    """
    try:
        if isinstance(config, str):
            model = RiskFormerLightningModule.from_config_file(config)
        elif isinstance(config, dict):
            model = RiskFormerLightningModule.from_config(config)
        
        log_event("info", "model_created", "Model created successfully", model_type=type(model).__name__)
        return model
    except Exception as e:
        log_event("error", "model_creation_failed", "Failed to create model", error_message=str(e))
        raise

def create_callbacks(model_dir=None, early_stop_patience=25):
    """Create training callbacks
    
    Args:
        model_dir (str, optional): Directory to save model checkpoints. Defaults to './models'.
        early_stop_patience (int, optional): Patience for early stopping. Defaults to 25.
        
    Returns:
        list: List of PyTorch Lightning callbacks
    """
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="riskformer-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
            dirpath=os.path.join(model_dir or './models', 'checkpoints'),
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    log_event("info", "callbacks_created", "Training callbacks created")
    return callbacks

def create_trainer(config, callbacks, logger):
    """Create PyTorch Lightning trainer
    
    Args:
        config (dict): Training configuration
        callbacks (list): List of PyTorch Lightning callbacks
        logger: PyTorch Lightning logger
        
    Returns:
        pl.Trainer: PyTorch Lightning trainer
    """
    try:
        strategy = config.get('strategy', 'auto')

        trainer = pl.Trainer(
            max_epochs=config.get('max_epochs', 100),
            min_epochs=config.get('min_epochs', 10),
            callbacks=callbacks,
            logger=logger,
            precision=config.get('precision', '32'),
            accelerator=config.get('accelerator', 'auto'),
            accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
            devices=config.get('devices', 'auto'),
            strategy=strategy,
            log_every_n_steps=config.get('log_every_nsteps', 10),
            deterministic=config.get('deterministic', None),
            sync_batchnorm=config.get('sync_batchnorm', False),
        )
        
        # Log trainer configuration details with special attention to distributed setup
        log_params = {
            'max_epochs': config.get('max_epochs', 100),
            'devices': config.get('devices', 1),
            'accelerator': config.get('accelerator', 'auto'),
            'strategy': strategy if isinstance(strategy, str) else str(strategy),
            'precision': config.get('precision', '32')
        }
        
        log_event("info", "trainer_created", "PyTorch Lightning Trainer created", **log_params)
        return trainer
    except Exception as e:
        log_event("error", "trainer_creation_failed", "Failed to create trainer", error_message=str(e))
        raise

def train_model(trainer, model, data_module):
    """Train the model
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        model (RiskFormerLightningModule): Lightning module for training
        data_module (RiskFormerDataModule): Data module for training
        
    Returns:
        trainer: Trained PyTorch Lightning trainer
    """
    try:
        log_event("info", "training_started", "Model training started")
        trainer.fit(model, data_module)
        log_event("info", "training_completed", "Model training completed successfully")
        return trainer
    except Exception as e:
        log_event("error", "training_failed", "Model training failed", error_message=str(e))
        raise

def test_model(trainer, model, data_module):
    """Test the model
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        model (RiskFormerLightningModule): Lightning module for training
        data_module (RiskFormerDataModule): Data module for training
        
    Returns:
        list: Test results
    """
    try:
        log_event("info", "testing_started", "Model testing started")
        test_results = trainer.test(model, data_module)
        
        # Ensure the test results are in a standard format
        if test_results and isinstance(test_results, list):
            metrics_str = {k: f"{v:.4f}" if isinstance(v, float) else v 
                          for k, v in test_results[0].items()}
            log_event("info", "testing_completed", "Model testing completed successfully", 
                     **metrics_str)
        else:
            log_event("warning", "testing_results_unexpected", 
                     "Unexpected test results format", 
                     result_type=type(test_results).__name__)
            
        return test_results
    except Exception as e:
        log_event("error", "testing_failed", "Model testing failed", error_message=str(e))
        raise

def save_model(trainer, model_dir, filename='final_model.ckpt'):
    """Save the trained model
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        model_dir (str): Directory to save model
        filename (str, optional): Filename for the saved model. Defaults to 'final_model.ckpt'.
        
    Returns:
        str: Path to the saved model
    """
    try:
        final_model_path = os.path.join(model_dir or './models', filename)
        trainer.save_checkpoint(final_model_path)
        log_event("info", "model_saved", "Final model saved successfully", 
                 model_path=final_model_path)
        return final_model_path
    except Exception as e:
        log_event("error", "model_save_failed", "Failed to save final model", error_message=str(e))
        raise
