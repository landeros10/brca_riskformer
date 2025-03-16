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
import json
from datetime import datetime
from typing import Union
import pytorch_lightning as pl # type: ignore
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor # type: ignore

from riskformer.training.model import RiskFormerLightningModule
from riskformer.data.datasets import RiskFormerDataModule
from riskformer.utils.logger_config import setup_training_run_logger

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
            
        return data_module
    except Exception as e:
        logger.error(f"Failed to create data module: {str(e)}")
        logger.debug("Exception details", exc_info=True)
        raise RuntimeError(f"Failed to create PyTorch Lightning DataModule: {str(e)}") from e


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
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        logger.debug("Exception details", exc_info=True)
        raise RuntimeError(f"Failed to create PyTorch Lightning LightningModule: {str(e)}") from e


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
        
        return trainer
    except Exception as e:
        logger.error(f"Failed to create trainer: {str(e)}")
        logger.debug(f"Trainer configuration that caused the error: {config}", exc_info=True)
        raise RuntimeError(f"Failed to create PyTorch Lightning trainer: {str(e)}") from e


def train_model(
        trainer: pl.Trainer, 
        model: RiskFormerLightningModule, 
        data_module: RiskFormerDataModule,
        ckpt_path: str = None,
) -> pl.Trainer:
    """Train the model
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        model (RiskFormerLightningModule): Lightning module for training
        data_module (RiskFormerDataModule): Data module for training
        ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
    Returns:
        trainer: Trained PyTorch Lightning trainer
    """
    try:
        # TODO: incorporate ckpt_path
        trainer.fit(model, data_module)
        return trainer
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        logger.debug("Training error details", exc_info=True)

        # Check if this is a CUDA out of memory error, which needs special handling
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error - try reducing batch size or model size")
            raise RuntimeError("GPU memory exceeded during training. Try reducing batch size or model complexity.") from e
        raise RuntimeError(f"Model training failed: {str(e)}") from e


def test_model(trainer, model, data_module):
    """Test the model
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        model (RiskFormerLightningModule): Lightning module for training
        data_module (RiskFormerDataModule): Data module for training
        
    Returns:
        dict: Test results
    """
    try:
        test_results = trainer.test(model, data_module)
        if test_results and isinstance(test_results, list):
            return test_results[0]
        else:
            logger.error("No test results returned from trainer.test()")
            raise RuntimeError("Testing failed to produce any results. Check your test dataset and model configuration.")
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        logger.debug("Testing error details", exc_info=True)

        # Check if this is a CUDA out of memory error, which needs special handling
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error during testing")
            raise RuntimeError("GPU memory exceeded during testing. Try reducing batch size.") from e
        raise RuntimeError(f"Model testing failed: {str(e)}") from e


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
        logger.info(f"Model saved successfully to {final_model_path}")
        return final_model_path
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        logger.debug("Save model error details", exc_info=True)
        # Check for common file system errors
        if "No space left on device" in str(e) or "Disk quota exceeded" in str(e):
            raise RuntimeError(f"Insufficient disk space to save model to {model_dir}") from e
        elif "Permission denied" in str(e):
            raise RuntimeError(f"Permission denied when saving model to {model_dir}") from e
        raise RuntimeError(f"Failed to save model to {model_dir}: {str(e)}") from e


def get_trainer_callbacks(
        model_dir: str, 
        early_stop_patience: int, 
        experiment_name: str, 
        run_id: str, 
        monitor: str, 
        monitor_mode: str, 
        save_top_k: int
):
    callbacks = create_callbacks(model_dir, early_stop_patience)
    checkpoint_callback, checkpoint_dir = setup_model_checkpoint_callback(
        model_dir=model_dir, 
        experiment_name=experiment_name, 
        run_id=run_id, 
        monitor=monitor, 
        monitor_mode=monitor_mode, 
        save_top_k=save_top_k
    )
    # Add to callbacks list - replace any existing ModelCheckpoint
    for i, callback in enumerate(callbacks):
        if isinstance(callback, ModelCheckpoint):
            callbacks[i] = checkpoint_callback
            break
    else:
        callbacks.append(checkpoint_callback)
    return callbacks


def run_one_training_session(
        config: dict,
        save_results: bool = True,
        model_dir: str = "./models",
        log_dir: str = "./logs",
        run_id: str = "0000",    
):
    """Main training function."""
    # Create data module using core functionality
    data_module = create_data_module(config)
    
    # Create model using core functionality
    model = create_model(config)
    
    # Create callbacks using core functionality
    callbacks = get_trainer_callbacks(
        model_dir=model_dir, 
        early_stop_patience=config['early_stop'], 
        experiment_name=config['experiment_name'], 
        run_id=run_id, 
        monitor=config['monitor'], 
        monitor_mode=config['monitor_mode'], 
        save_top_k=config['save_top_k']
    )

    # Create logger with the imported function
    tb_logger = setup_training_run_logger(
        use_wandb=config['use_wandb'],
        log_dir=log_dir,
        experiment_name=config['experiment_name'],
        wandb_project=config['wandb_project'],
        wandb_entity=config['wandb_entity']
    )
    
    # Create trainer using core functionality
    trainer = create_trainer(config, callbacks, tb_logger)
    
    # Get checkpoint path if resuming
    ckpt_path = config.get('resume_from_checkpoint', None)
    
    ### Training Run ###
    trainer = train_model(
        trainer=trainer, 
        model=model, 
        data_module=data_module, 
        ckpt_path=ckpt_path
    )
    
    ### Testing ###
    test_results = test_model(trainer, model, data_module)
        
    # Save test results with complete config
    if test_results and save_results:
        best_checkpoint_path = None
        for callback in callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                if hasattr(callback, 'best_model_path') and callback.best_model_path:
                    best_checkpoint_path = callback.best_model_path
                    break
        test_results['config'] = config
        test_results['run_id'] = run_id
        test_results["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        test_results["best_checkpoint_path"] = best_checkpoint_path

        results_dir = os.path.join(model_dir, "results")
        results_file = save_test_results(
            test_results=test_results,
            results_dir=results_dir
        )
        return results_file
    else:
        return test_results


def setup_model_checkpoint_callback(
        model_dir: str, 
        experiment_name: str, 
        run_id: str, 
        monitor: str, 
        monitor_mode: str, 
        save_top_k: int
) -> ModelCheckpoint:

    checkpoint_dir = os.path.join(model_dir, f"{experiment_name}_{run_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    metric_name = monitor.replace('val_', '').replace('test_', '')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'epoch={{epoch:02d}}-{metric_name}={{' + monitor + ':.4f}}',
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    
    logger.info(f"Model checkpoints will be saved to {checkpoint_dir}")
    logger.debug(f"Checkpoint naming format: epoch=XX-{metric_name}=Y.YYYY.ckpt")

    return checkpoint_callback, checkpoint_dir


def save_test_results(
        test_results: dict,
        results_dir: str,
) -> str | None:
    """Save test results to JSON for hyperparameter optimization
    
    Args:
        test_results (dict): Results from model testing
        results_dir (str): Directory to save results
    
    Returns:
        str: Path to the saved results file, or None if not saved
    """    
    try:
        # Extract key identifiers for the filename
        config = test_results.get('config', {})
        experiment_name = config.get('experiment_name', 'unknown_experiment')
        run_id = test_results.get('run_id', 'unknown_run')
        ts = test_results.get('timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        # Look for metrics with "test_" prefix first
        metric = None
        metric_val = None
        
        # First try to find metrics that start with "test_"
        for key, value in test_results.items():
            if (key.startswith('test_') and 
                isinstance(value, (int, float)) and 
                key != 'test_epoch'):
                metric = key
                metric_val = value
                break
        
        # If no test_ metrics found, fall back to first numeric metric
        if metric is None:
            for key, value in test_results.items():
                if (isinstance(value, (int, float)) and 
                    key != 'epoch' and 
                    not key.startswith('config') and 
                    not key in ['run_id', 'timestamp']):
                    metric = key
                    metric_val = value
                    break

        # Set filename
        filename = f"{experiment_name}_{run_id}_{ts}"
        if metric and metric_val is not None:
            if isinstance(metric_val, float):
                metric_str = f"{metric}={metric_val:.4f}"
            else:
                metric_str = f"{metric}={metric_val}"
            filename += f"_{metric_str}"
        filename = filename.replace(" ", "_").replace("/", "_")

        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"{filename}.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=4)
        return results_file
    except Exception as e:
        logger.error(f"Failed to save test results: {str(e)}")
        logger.debug("Error details", exc_info=True)
        return None