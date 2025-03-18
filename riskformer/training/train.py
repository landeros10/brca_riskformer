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
    """Create data module from config.

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
    
    logger.debug(f"Checkpoint naming format: epoch=XX-{metric_name}=Y.YYYY.ckpt")
    return checkpoint_callback


def create_callbacks(
    model_dir: str, 
    early_stop_patience: int, 
    experiment_name: str, 
    run_id: str, 
    monitor: str, 
    monitor_mode: str, 
    save_top_k: int,
):
    """Create training callbacks
    
    Args:
        model_dir (str, optional): Directory to save model checkpoints. Defaults to './models'.
        early_stop_patience (int, optional): Patience for early stopping. Defaults to 25.
        
    Returns:
        list: List of PyTorch Lightning callbacks
    """
    callbacks = [
        setup_model_checkpoint_callback(
            model_dir=model_dir, 
            experiment_name=experiment_name, 
            run_id=run_id, 
            monitor=monitor, 
            monitor_mode=monitor_mode, 
            save_top_k=save_top_k
        ),
        EarlyStopping(
            monitor=monitor,
            patience=early_stop_patience,
            mode=monitor_mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return callbacks


def get_best_ckpt(
        trainer: pl.Trainer = None,
        callbacks: list = None,
) -> str:
    """Get the best checkpoint path from the trainer or callbacks
    
    Args:
        trainer (pl.Trainer, optional): PyTorch Lightning trainer. Defaults to None.
        callbacks (list, optional): List of PyTorch Lightning callbacks. Defaults to None.

    Returns:
        str: Best checkpoint path
    """
    best_ckpt_path = None
    if trainer and hasattr(trainer, 'checkpoint_callback'):
        if hasattr(trainer.checkpoint_callback, 'best_model_path'):
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
        else:
            logger.warning("No best checkpoint path found in trainer")
            return None
    
    elif callbacks:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                if hasattr(callback, 'best_model_path'):
                    best_ckpt_path = callback.best_model_path
                    break
    
    else:
        logger.warning("No best checkpoint path found")
        return None

    return best_ckpt_path


def create_trainer(
        strategy: str = "auto",
        max_epochs: int = 100,
        min_epochs: int = 10,
        precision: str = "32",
        accelerator: str = "auto",
        accumulate_grad_batches: int = 1,
        devices: str = "auto",
        log_every_n_steps: int = 10,
        deterministic: bool = None,
        sync_batchnorm: bool = False,
        callbacks: list = None,
        pl_logger: pl.loggers.Logger = None,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer
    
    Args:
        callbacks (list): List of PyTorch Lightning callbacks
        pl_logger: PyTorch Lightning logger
        
    Returns:
        pl.Trainer: PyTorch Lightning trainer
    """
    try:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            callbacks=callbacks,
            logger=pl_logger,
            precision=precision,
            accelerator=accelerator,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=devices,
            strategy=strategy,
            log_every_n_steps=log_every_n_steps,
            deterministic=deterministic,
            sync_batchnorm=sync_batchnorm,
        )
        
        # Log trainer configuration details with special attention to distributed setup
        log_params = {
            'max_epochs': max_epochs,
            'devices': devices,
            'accelerator': accelerator,
            'strategy': strategy if isinstance(strategy, str) else str(strategy),
            'precision': precision
        }
        logger.debug(f"Trainer configuration: {log_params}")
        return trainer
    except Exception as e:
        logger.error(f"Failed to create trainer: {str(e)}")
        logger.debug(f"Trainer configuration that caused the error: {trainer.state_dict()}", exc_info=True)
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
        trainer.fit(
            model=model, 
            datamodule=data_module, 
            ckpt_path=ckpt_path
        )
        
        return trainer
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        logger.debug("Training error details", exc_info=True)

        # Check if this is a CUDA out of memory error, which needs special handling
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error - try reducing batch size or model size")
            raise RuntimeError("GPU memory exceeded during training. Try reducing batch size or model complexity.") from e
        raise RuntimeError(f"Model training failed: {str(e)}") from e


def test_model(
        trainer: pl.Trainer, 
        data_module: RiskFormerDataModule,
        model: RiskFormerLightningModule = None,
        ckpt_path: str = None,
) -> dict:
    """Test the model using the best checkpoint from the callback
    established during trainer creation.
    
    If no model or ckpt_path is provided, the best checkpoint from the
    trainer will be used. Must be called after training.
    
    Args:
        trainer (pl.Trainer): PyTorch Lightning trainer
        data_module (RiskFormerDataModule): Data module for training
        model (RiskFormerLightningModule, optional): Lightning module for testing. Defaults to None.
        ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
    Returns:
        dict: Test results
    """
    try:
        test_results = trainer.test(
            datamodule=data_module, 
            model=model,
            ckpt_path=ckpt_path,
        )
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


def run_one_training_session(
        config: dict,
        results_file_path: str = None,
        model_dir: str = "./models",
        log_dir: str = "./logs",
        run_id: str = "0000",
        validate_config: bool = False,    
):
    """Main training function.
    
    This function expects a configuration dictionary that has been validated
    
    Args:
        config (dict): Configuration dictionary containing model and training parameters.
                      Must have been validated through riskformer.utils.training_utils.
        results_file_path (str, optional): Path where to save test results. If None, results won't be saved.
                                         Defaults to None.
        model_dir (str, optional): Directory to save model checkpoints. Defaults to "./models".
        log_dir (str, optional): Directory to save logs. Defaults to "./logs".
        run_id (str, optional): Unique identifier for this run. Defaults to "0000".
        validate_config (bool, optional): Whether to validate the config using 
                                        _validate_training_config(). Set to True 
                                        if the config hasn't been validated yet. 
                                        Defaults to False.
    
    Returns:
        Union[str, dict]: Path to results file if save_results is True, otherwise test results dictionary.
        
    Raises:
        ValueError: If validate_config is True and config fails validation.
    """
    # Validate config if requested
    if validate_config:
        from riskformer.utils.training_utils import _validate_training_config
        _validate_training_config(config)
    
    # Create data module using core functionality
    data_module = create_data_module(config)
    
    # Create model using core functionality
    model = create_model(config)
    
    # Create callbacks using core functionality
    callbacks = create_callbacks(
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
    trainer = create_trainer(
        strategy=config['strategy'],
        max_epochs=config['max_epochs'],
        min_epochs=config['min_epochs'],
        precision=config['precision'],
        accelerator=config['accelerator'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        devices=config['devices'],
        log_every_n_steps=config['log_every_n_steps'],
        deterministic=config['deterministic'],
        sync_batchnorm=config['sync_batchnorm'],
        callbacks=callbacks,
        pl_logger=tb_logger,
    )
    
    # Get checkpoint path if resuming
    ckpt_path = config.get('resume_from_checkpoint', None)
    
    ### Training Run ###
    trainer = train_model(
        trainer=trainer, 
        model=model, 
        data_module=data_module, 
        ckpt_path=ckpt_path,
    )
    
    ### Testing ###
    test_results = test_model(
        trainer=trainer, 
        data_module=data_module,
    )
        
    # Save test results with complete config
    if test_results:
        # Fallback if results_file_path is not provided
        if not results_file_path:
            results_file_path = os.path.join(
                model_dir,
                "results",
                f"{config['experiment_name']}_{run_id}.json"
            )

        test_results['config'] = config
        test_results['run_id'] = run_id
        test_results["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        test_results["best_checkpoint_path"] = get_best_ckpt(trainer)

        save_test_results(
            test_results=test_results,
            results_file_path=results_file_path
        )


def save_test_results(
        test_results: dict,
        results_file_path: str,
) -> str | None:
    """Save test results to JSON for hyperparameter optimization
    
    Args:
        test_results (dict): Results from model testing
        results_file_path (str): File path to save results
    
    Returns:
        str: Path to the saved results file, or None if not saved
    """     
    try:
        # Create directory if it doesn't exist
        results_dir = os.path.dirname(results_file_path)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to the specified path
        with open(results_file_path, "w") as f:
            json.dump(test_results, f, indent=4)
            
        return results_file_path
    
    except Exception as e:
        logger.error(f"Failed to save test results: {str(e)}")
        logger.debug("Error details", exc_info=True)
        return None
