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

from riskformer.training.train import run_one_training_session
from riskformer.utils.training_utils import clear_gpu_memory, load_train_config
from riskformer.utils.logger_config import logger_setup, log_event

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
    parser.add_argument("--results_file_path", type=str, default=None,
                        help="Path where to save test results. If not provided, results won't be saved.")
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")

    return parser.parse_args()


def log_gpu_info():
    """Log information about available GPUs"""
    try:
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
    except Exception as e:
        log_event("error", "gpu_info_error", "Error while checking GPU information", 
                error_type=type(e).__name__, error_message=str(e))
        logger.debug("GPU info error details", exc_info=True)


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
         

def main():
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

    log_gpu_info()
    clear_gpu_memory()
    
    try:
        results = run_one_training_session(
            config=config,
            results_file_path=args.results_file_path,
            model_dir=model_dir,
            log_dir=log_dir,
            run_id=run_id,
        )
        log_event("info", "training_pipeline_complete", "Training pipeline completed successfully")

    except Exception as e:
        log_event("error", "training_pipeline_failed", "Training pipeline failed with uncaught exception", 
                 error_type=type(e).__name__, error_message=str(e))
        log_event("debug", "exception_traceback", "Full exception traceback", exc_info=True)
        return 1
    finally:
        clear_gpu_memory()
        log_event("debug", "resources_cleaned", "Resources cleaned up")
        log_event("info", "pipeline_shutdown", "Training pipeline shutdown complete")
        return 0

if __name__ == "__main__":
    exit(main())
