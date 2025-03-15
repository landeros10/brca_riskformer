'''
train.py
Author: landeros10
Created: 2023-06-01
Modified to work as entrypoint and support hyperparameter optimization
'''
import torch
import argparse
import logging
import os
import json
import shutil
import uuid
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from riskformer.training.train import (
    create_data_module,
    create_model,
    create_callbacks,
    create_trainer,
    train_model,
    test_model,
)
from riskformer.utils.training_utils import clear_gpu_memory
from riskformer.utils.config_utils import load_train_config
from riskformer.utils.logger_config import logger_setup, log_event, setup_training_run_logger

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RiskFormer Training Entrypoint")

    # Config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    
    # Setup Files
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata file (overrides config)")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save model checkpoints (overrides config)")
    parser.add_argument("--log_dir", type=str, default="./logs/lightning_logs",
                        help="Directory for logs (overrides config)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment (overrides config)")
    
    # Logging and environment
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config)")
    parser.add_argument("--debug", action="store_true",
                        help="Set to run in debug mode")

    # RiskFormerDataModule Config
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--max_dim", type=int, default=32,
                        help="Maximum dimension for input token array")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap between patches")


    # RiskFormerLightningModule Config
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Optimizer to use: adam, sgd, etc. (overrides config)")    
    
    # Trainer COnfig
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of epochs (overrides config)")
    parser.add_argument("--precision", type=str, default=None,
                        help="Precision for training: 32 (default), 16 (mixed precision), bf16 (bfloat16), or 64 (double precision). Note that 16/bf16 requires GPU.")
    parser.add_argument("--accelerator", type=str, default=None,
                        help="Accelerator type: cpu, gpu, tpu, etc. (overrides config)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None,
                        help="Number of batches to accumulate gradients over (overrides config)")
    parser.add_argument("--devices", type=int, default=None,
                        help="Number of devices to use (overrides config)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Distributed training strategy: ddp, deepspeed, etc. (overrides config)")
    parser.add_argument("--sync_batchnorm", action="store_true", 
                        help="Use synchronized batch normalization in distributed training")

    
    # Hyperparameter optimization integration
    parser.add_argument("--save_results", action="store_true",
                        help="Save test results to a JSON file for hyperparameter optimization")
    parser.add_argument("--results_dir", type=str, default="./",
                        help="Directory to save results JSON file")
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")

    return parser.parse_args()


def log_gpu_info():
    """Log information about available GPUs"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        log_event("info", "gpu_setup", "GPU information", 
                 device_count=device_count,
                 cuda_version=torch.version.cuda)
        
        for i in range(device_count):
            log_event("info", "gpu_device", "GPU device information",
                     device_index=i,
                     device_name=torch.cuda.get_device_name(i),
                     memory_allocated=f"{torch.cuda.memory_allocated(i)/1024**3:.2f} GB",
                     memory_reserved=f"{torch.cuda.memory_reserved(i)/1024**3:.2f} GB")
    else:
        log_event("warning", "gpu_setup", "No GPUs available, using CPU")


def check_disk_space(path, required_gb=10):
    """Check if there's enough disk space available
    
    Args:
        path (str): Path to check disk space for
        required_gb (float): Required space in GB
        
    Returns:
        bool: True if there's enough space, False otherwise
    """
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)  # Convert to GB
        
        if free_gb < required_gb:
            log_event("warning", "low_disk_space", "Low disk space detected", 
                    path=path, free_space_gb=f"{free_gb:.2f}", required_gb=required_gb)
            return False
        
        log_event("info", "disk_space_check", "Sufficient disk space available", 
                path=path, free_space_gb=f"{free_gb:.2f}")
        return True
    except Exception as e:
        log_event("error", "disk_space_check_failed", "Failed to check disk space", 
                error_message=str(e))
        return True  # Assume there's enough space if check fails


def cleanup_resources():
    """Clean up resources to prevent memory leaks and ensure proper shutdown"""
    # Clear GPU memory
    clear_gpu_memory()
    
    # Close any open file handles or connections
    # Note: Most handles should be closed by their respective objects' __del__ methods
    log_event("debug", "resources_cleaned", "Resources cleaned up")


def load_training_run_config(
        config_path: str,
        args: argparse.Namespace
) -> dict:
    """Load the training configuration from a YAML file
    
    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Training configuration
    """
    try:
        config = load_train_config(config_path)
        if not config:
            log_event("error", "config_load_failed", "Failed to load configuration", config_path=config_path)
            raise ValueError(f"Failed to load config from {config_path}")
    
    except Exception as e:
        log_event("error", "config_load_error", "Unexpected error loading configuration", config_path=config_path, error_message=str(e))
        raise
    
    config = update_overrrides_config(config, args)
    log_event("info", "config_details", "Configuration loaded with details", config_path=config_path)
    log_event("debug", "config_dump", "Full configuration dump", config=config)

    return config


def update_overrrides_config(
        config: dict,
        args: argparse.Namespace
) -> dict:
    """Update the config with command line arguments"""
    # Override with command-line arguments if provided
    if args.metadata_file:
        config['metadata_file'] = args.metadata_file
    if args.log_dir:
        config['log_dir'] = args.log_dir
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.use_wandb:
        config['use_wandb'] = True
    if args.seed is not None:
        config['seed'] = args.seed
    if args.debug:
        config['debug'] = True
    if args.devices:
        config['devices'] = args.devices
    if args.accelerator:
        config['accelerator'] = args.accelerator
    if args.accumulate_grad_batches:
        config['accumulate_grad_batches'] = args.accumulate_grad_batches
    if args.optimizer:
        config['optimizer'] = args.optimizer
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.strategy:
        config['strategy'] = args.strategy
    if args.precision:
        if args.precision in ['16', '16-mixed'] and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            log_event("warning", "precision_fallback", "16-bit precision requires GPU, falling back to 32-bit", 
                        requested_precision=args.precision)
            config['precision'] = '32'
        else:
            config['precision'] = args.precision
    if args.sync_batchnorm:
        config['sync_batchnorm'] = True
    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    return config


def check_local_disks(
        model_dir: str, 
        log_dir: str
) -> bool:
    # Check disk space for logs and model artifacts
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Check for sufficient disk space (requires at least 5GB)
    if not check_disk_space(model_dir, required_gb=5):
        log_event("error", "insufficient_disk_space", "Not enough disk space for model artifacts", 
                    path=model_dir)
        return False
        
    if not check_disk_space(log_dir, required_gb=2):
        log_event("warning", "low_disk_space_logs", "Low disk space for logging", 
                    path=log_dir)
    return True


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

    # Create the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{' + monitor + ':.4f}',
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )

    return checkpoint_callback, checkpoint_dir


def setup_callbacks(
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

    log_event("info", "checkpoint_callback_created", "Model checkpoint callback created", 
                checkpoint_dir=checkpoint_dir, save_top_k=save_top_k, monitor=monitor)
    
    log_event("debug", "callbacks_created", "Training callbacks created", 
                early_stop_patience=early_stop_patience, model_dir=model_dir)
    return callbacks
    

def save_results_to_json(
        test_results: list,
        config: dict,
        results_dir: str = "./",
        best_checkpoint_path: str = None
) -> str | None:
    """Save test results to a JSON file for hyperparameter optimization
    
    Args:
        test_results (list): Results from testing
        config (dict): Configuration used for training
        results_dir (str): Directory to save results
        best_checkpoint_path (str): Path to the best checkpoint
    """
    if not test_results:
        log_event("warning", "save_results_to_json", "No test results to save")
        return None
    
    # Create a normalized dictionary from the test results
    results_dict = {}
    if isinstance(test_results, list) and test_results:
        results_dict = test_results[0]  # Get the first element if it's a list
    elif isinstance(test_results, dict):
        results_dict = test_results
    
    # Add the entire config to the results
    results_dict["config"] = config
    
    # Add checkpoint path information
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        results_dict["best_checkpoint_path"] = best_checkpoint_path
    
    # Add timestamp and experiment name
    results_dict["timestamp"] = datetime.now().isoformat()
    experiment_name = config.get("experiment_name", "riskformer")
    
    # Save to file
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{experiment_name}.json")
    
    try:
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=4)
        log_event("info", "results_saved", "Test results saved to JSON file", file_path=results_file)
        return results_file
    except Exception as e:
        log_event("error", "results_save_failed", "Failed to save test results", error_message=str(e))
        return None


def save_test_results(
        test_results: dict,
        save_results: bool,
        results_dir: str,
        callbacks: list,
        config: dict
) -> str | None:
    """Save test results to JSON for hyperparameter optimization
    
    Args:
        test_results (dict): Results from model testing
        save_results (bool): Whether to save results
        results_dir (str): Directory to save results
        callbacks (list): List of callbacks that may contain ModelCheckpoint
        config (dict): Complete configuration used for training
    
    Returns:
        str: Path to the saved results file, or None if not saved
    """
    if not save_results:
        return None
    
    # Get the best checkpoint path from the ModelCheckpoint callback
    best_checkpoint_path = None
    for callback in callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                best_checkpoint_path = callback.best_model_path
                break
    
    # Save the complete config with the results
    results_file = save_results_to_json(test_results, config, results_dir, best_checkpoint_path)
    
    if results_file:
        log_event("info", "results_saved", "Test results saved to JSON", file_path=results_file)
    
    return results_file


def main():
    """Main training function."""
    try:
        ### Setup ###
        args = parse_args()
        
        # Set up logger
        logger_setup(log_group="riskformer-train", debug=args.debug, log_dir=args.log_dir)
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Load configuration from file
        config_path = args.config
        config = load_training_run_config(config_path, args)

        # Set seed for reproducibility
        seed = config.get('seed', 42)
        pl.seed_everything(seed)
        run_id = str(uuid.uuid4())[:8]
        log_event("debug", "seed_set", "Random seed set for reproducibility", seed=seed)
        
        # Set up local dirs
        model_dir = config['model_dir']
        log_dir = config['log_dir']
        if not check_local_disks(model_dir, log_dir):
            log_event("error", "disk_space_check_failed", "Disk space check failed, aborting training")
            return 1

        # Create data module using core functionality
        data_module = create_data_module(config)
        log_event("info", "data_module_created", "Data module created successfully")
        
        # Create model using core functionality
        model = create_model(config)
        log_event("info", "model_created", "Model architecture created successfully")
        
        # Create callbacks using core functionality
        callbacks = setup_callbacks(
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
        log_event("info", "trainer_created", "PyTorch Lightning trainer created successfully")
        
        # Get checkpoint path if resuming
        ckpt_path = config.get('resume_from_checkpoint', None)
        if ckpt_path:
            log_event("info", "resuming_training", "Resuming training from checkpoint", checkpoint_path=ckpt_path)
        
        ### Training Run ###
        log_event("info", "training_start", "Starting RiskFormer Training Pipeline")
        log_gpu_info()
        clear_gpu_memory()
            
        trainer = train_model(trainer, model, data_module, ckpt_path)
        clear_gpu_memory()
        log_event("info", "training_completed", "Model training completed successfully")
        
        ### Testing ###
        test_results = test_model(trainer, model, data_module)
        log_event("info", "testing_completed", "Model evaluation completed", metrics=test_results)
        
        # Add run_id to config
        config['run_id'] = run_id
        
        # Save test results with complete config
        save_test_results(
            test_results=test_results,
            save_results=args.save_results,
            results_dir=args.results_dir,
            callbacks=callbacks,
            config=config
        )
        
        log_event("info", "training_pipeline_complete", "Training pipeline completed successfully")
        return 0  # Success exit code
        
    except Exception as e:
        log_event("error", "training_pipeline_failed", "Training pipeline failed with uncaught exception", 
                 error_type=type(e).__name__, error_message=str(e))
        log_event("debug", "exception_traceback", "Full exception traceback", exc_info=True)
        return 1  # Error exit code
    
    finally:
        # Ensure resources are cleaned up regardless of success or failure
        cleanup_resources()
        log_event("info", "pipeline_shutdown", "Training pipeline shutdown complete")


if __name__ == "__main__":
    exit(main())
