'''
train.py
Author: landeros10
Created: 2023-06-01
Modified to work as entrypoint and support hyperparameter optimization
'''
import torch
import argparse
import logging
import pytorch_lightning as pl
import os
import json

from riskformer.training.train import (
    create_data_module,
    create_model,
    create_callbacks,
    create_trainer,
    train_model,
    test_model,
    save_model
)
from riskformer.utils.training_utils import load_train_config, validate_config
from riskformer.utils.logger_config import logger_setup, log_event, create_tensorboard_logger, create_wandb_logger

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RiskFormer Training Entrypoint")

    # Config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing preprocessed data")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata file (overrides config)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory to save model checkpoints (overrides config)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for logs (overrides config)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment (overrides config)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    
    # Logging and environment
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config)")
    parser.add_argument("--debug", action="store_true",
                        help="Set to run in debug mode")
    
    # Hardware configuration
    parser.add_argument("--devices", type=int, default=None,
                        help="Number of devices to use (overrides config)")
    parser.add_argument("--accelerator", type=str, default=None,
                        help="Accelerator type: cpu, gpu, tpu, etc. (overrides config)")
    
    # Multi-GPU specific options
    parser.add_argument("--strategy", type=str, default=None,
                        help="Distributed training strategy: ddp, deepspeed, etc. (overrides config)")
    parser.add_argument("--precision", type=str, default=None,
                        help="Precision for training: 32, 16, bf16, etc. (overrides config)")
    parser.add_argument("--sync_batchnorm", action="store_true", 
                        help="Use synchronized batch normalization in distributed training")
    parser.add_argument("--find_unused_parameters", action="store_true",
                        help="Find unused parameters in DDP (helps with certain model architectures)")
    
    # Hyperparameter optimization integration
    parser.add_argument("--save_results", action="store_true",
                        help="Save test results to a JSON file for hyperparameter optimization")
    parser.add_argument("--results_dir", type=str, default="./",
                        help="Directory to save results JSON file")
    
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


def save_results_to_json(test_results, config, results_dir="./"):
    """Save test results to a JSON file for hyperparameter optimization
    
    Args:
        test_results (list): Results from testing
        config (dict): Configuration used for training
        results_dir (str): Directory to save results
    """
    if not test_results:
        log_event("warning", "results_save_skipped", "No test results to save")
        return None
    
    # Create a normalized dictionary from the test results
    results_dict = {}
    if isinstance(test_results, list) and test_results:
        results_dict = test_results[0]  # Get the first element if it's a list
    elif isinstance(test_results, dict):
        results_dict = test_results
    
    # Add key configuration parameters to the results
    results_dict["learning_rate"] = config.get("learning_rate")
    results_dict["batch_size"] = config.get("batch_size")
    results_dict["model_type"] = config.get("model_type")
    results_dict["max_epochs"] = config.get("max_epochs")
    
    # Add timestamp and experiment name
    from datetime import datetime
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


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up logger
        logger_setup(
            log_group="riskformer-train",
            debug=args.debug,
        )
        
        log_event("info", "args_parsed", "Command line arguments parsed", config_path=args.config)
        
        log_event("info", "training_start", "Starting RiskFormer Training Pipeline")
        
        # Set logger debug level if in debug mode
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Log GPU information
        log_gpu_info()
        
        # Load configuration from file
        config_path = args.config
        try:
            validate_config(config_path)
            log_event("info", "config_validated", "Configuration file validated", config_path=config_path)
        except ValueError as e:
            log_event("error", "invalid_config", "Invalid configuration file", config_path=config_path, error_message=str(e))
            raise ValueError(f"Invalid config file: {config_path}") from e
        
        try:
            config = load_train_config(config_path)
            if not config:
                log_event("error", "config_load_failed", "Failed to load configuration", config_path=config_path)
                raise ValueError(f"Failed to load config from {config_path}")
            log_event("info", "config_loaded", "Configuration loaded successfully", config_path=config_path)
        except Exception as e:
            log_event("error", "config_load_error", "Error loading configuration", config_path=config_path, error_message=str(e))
            raise
        
        # Override with command-line arguments if provided
        if args.data_dir:
            config['data_dir'] = args.data_dir
        if args.metadata_file:
            config['metadata_file'] = args.metadata_file
        if args.model_dir:
            config['model_dir'] = args.model_dir
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
        
        # Add multi-GPU specific options
        if args.strategy:
            config['strategy'] = args.strategy
        if args.precision:
            config['precision'] = args.precision
        if args.sync_batchnorm:
            config['sync_batchnorm'] = True
        if args.find_unused_parameters:
            config['find_unused_parameters'] = True
        
        # Log the loaded configuration
        log_event("info", "config_details", "Configuration loaded with details", config_path=config_path)
        log_event("debug", "config_dump", "Full configuration dump", config=config)
        
        # Set seed for reproducibility
        seed = config.get('seed', 42)
        pl.seed_everything(seed)
        log_event("info", "seed_set", "Random seed set for reproducibility", seed=seed)
        
        # Create data module using core functionality
        data_module = create_data_module(config_path, config)
        log_event("info", "data_module_created", "Data module created successfully")
        
        # Create model using core functionality
        model = create_model(config_path, config)
        log_event("info", "model_created", "Model architecture created successfully")
        
        # Apply sync_batchnorm if requested (for multi-GPU training)
        if config.get('sync_batchnorm', False) and torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            log_event("info", "sync_batchnorm", "Converted model to use synchronized batch normalization")
        
        # Create callbacks using core functionality
        model_dir = config.get('model_dir', './models')
        early_stop_patience = config.get('early_stop', 25)
        callbacks = create_callbacks(model_dir, early_stop_patience)
        log_event("debug", "callbacks_created", "Training callbacks created", 
                 early_stop_patience=early_stop_patience, model_dir=model_dir)
        
        # Create logger
        if config.get('use_wandb', False):
            # Try to create WandB logger
            tb_logger = create_wandb_logger(config)
            # Fall back to TensorBoard if WandB fails
            if tb_logger is None:
                log_dir = config.get('log_dir', 'lightning_logs')
                experiment_name = config.get('experiment_name', 'riskformer')
                tb_logger = create_tensorboard_logger(log_dir, experiment_name)
                log_event("warning", "wandb_fallback", "Failed to create WandB logger, falling back to TensorBoard", 
                         log_dir=log_dir, experiment_name=experiment_name)
            else:
                log_event("info", "wandb_logger_created", "WandB logger created successfully")
        else:
            # Create TensorBoard logger using core functionality
            log_dir = config.get('log_dir', 'lightning_logs')
            experiment_name = config.get('experiment_name', 'riskformer')
            tb_logger = create_tensorboard_logger(log_dir, experiment_name)
            log_event("info", "tensorboard_logger_created", "TensorBoard logger created", 
                     log_dir=log_dir, experiment_name=experiment_name)
        
        # Create trainer using core functionality
        trainer = create_trainer(config, callbacks, tb_logger)
        log_event("info", "trainer_created", "PyTorch Lightning trainer created successfully")
        
        # Train model using core functionality
        log_event("info", "training_started", "Model training started")
        trainer = train_model(trainer, model, data_module)
        log_event("info", "training_completed", "Model training completed successfully")
        
        # Test model using core functionality
        log_event("info", "testing_started", "Model evaluation started")
        test_results = test_model(trainer, model, data_module)
        log_event("info", "testing_completed", "Model evaluation completed", metrics=test_results)
        
        # Save test results to JSON for hyperparameter optimization
        if args.save_results or 'save_results' in config and config['save_results']:
            results_dir = args.results_dir or config.get('results_dir', './')
            save_results_to_json(test_results, config, results_dir)
        
        # Save the final model using core functionality
        try:
            final_model_path = save_model(trainer, model_dir, 'final_model.ckpt')
            log_event("info", "model_saved", "Final model saved successfully", 
                     model_path=final_model_path)
        except Exception as e:
            log_event("error", "model_save_failed", "Failed to save final model", error_message=str(e))
            
        log_event("info", "training_pipeline_complete", "Training pipeline completed successfully")
        return 0  # Success exit code
        
    except Exception as e:
        log_event("error", "training_pipeline_failed", "Training pipeline failed with uncaught exception", 
                 error_type=type(e).__name__, error_message=str(e))
        log_event("debug", "exception_traceback", "Full exception traceback", exc_info=True)
        return 1  # Error exit code


if __name__ == "__main__":
    exit(main())
