'''
train.py
Author: landeros10
Created: 2023-06-01
Modified to work as entrypoint
'''
import os
import argparse
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from riskformer.training.model import RiskFormerLightningModule
from riskformer.utils.training_utils import load_train_config, validate_config, set_seed
from riskformer.data.datasets import RiskFormerDataModule
from riskformer.utils.logger_config import logger_setup, log_event

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
    
    return parser.parse_args()


def main():
    """Main training function."""
    try:
        # Set up logger
        logger_setup(
            log_group="riskformer-train",
            debug=False,
        )
        
        log_event("info", "training_start", "Starting Training Pipeline")
        logger.info("Starting RiskFormer Training Pipeline...")
        logger.info("=" * 50)
        
        # Parse arguments
        args = parse_args()
        log_event("info", "args_parsed", "Command line arguments parsed", config_path=args.config)
        
        # Set logger debug level if in debug mode
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
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
        
        # Log the loaded configuration
        logger.info(f"Loaded configuration from {config_path}")
        logger.debug(f"Configuration: {config}")
        
        # Set seed for reproducibility
        seed = config.get('seed', 42)
        pl.seed_everything(seed)
        log_event("info", "seed_set", "Random seed set for reproducibility", seed=seed)
        
        # Create data module
        try:
            data_module = RiskFormerDataModule.from_config_file(config_path)
            log_event("info", "data_module_created", "Data module created successfully")
        except Exception as e:
            log_event("error", "data_module_creation_failed", "Failed to create data module", error_message=str(e))
            raise
        
        # Create model from config
        try:
            model = RiskFormerLightningModule.from_config_file(config_path)
            log_event("info", "model_created", "Model created successfully", model_type=type(model).__name__)
        except Exception as e:
            log_event("error", "model_creation_failed", "Failed to create model", error_message=str(e))
            raise
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                filename="riskformer-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
                dirpath=os.path.join(config.get('model_dir', './models'), 'checkpoints'),
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=config.get('early_stop', 25),
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        log_event("info", "callbacks_created", "Training callbacks created")
        
        # Create logger
        if config.get('use_wandb', False):
            try:
                tb_logger = WandbLogger(
                    project=config.get('wandb_project', 'riskformer'),
                    name=config.get('experiment_name', 'riskformer'),
                    entity=config.get('wandb_entity'),
                    log_model=True,
                )
                log_event("info", "logger_created", "WandB logger created", 
                          project=config.get('wandb_project', 'riskformer'), 
                          experiment=config.get('experiment_name', 'riskformer'))
            except Exception as e:
                log_event("error", "wandb_logger_failed", "Failed to create WandB logger, falling back to TensorBoard", 
                         error_message=str(e))
                # Fall back to TensorBoard if WandB fails
                tb_logger = TensorBoardLogger(
                    save_dir=config.get('log_dir', 'lightning_logs'),
                    name=config.get('experiment_name', 'riskformer'),
                )
        else:
            tb_logger = TensorBoardLogger(
                save_dir=config.get('log_dir', 'lightning_logs'),
                name=config.get('experiment_name', 'riskformer'),
            )
            log_event("info", "logger_created", "TensorBoard logger created", 
                     log_dir=config.get('log_dir', 'lightning_logs'), 
                     experiment=config.get('experiment_name', 'riskformer'))
        
        # Create trainer
        try:
            trainer = pl.Trainer(
                max_epochs=config.get('max_epochs', 100),
                min_epochs=config.get('min_epochs', 10),
                callbacks=callbacks,
                logger=tb_logger,
                precision=config.get('precision', '32'),
                accelerator=config.get('accelerator', 'auto'),
                devices=config.get('devices', 1),
                strategy=config.get('strategy', 'auto'),
                log_every_n_steps=10,
                deterministic=config.get('deterministic', True),
            )
            log_event("info", "trainer_created", "PyTorch Lightning Trainer created",
                     max_epochs=config.get('max_epochs', 100),
                     devices=config.get('devices', 1),
                     accelerator=config.get('accelerator', 'auto'))
        except Exception as e:
            log_event("error", "trainer_creation_failed", "Failed to create trainer", error_message=str(e))
            raise
        
        # Train model
        try:
            log_event("info", "training_started", "Model training started")
            trainer.fit(model, data_module)
            log_event("info", "training_completed", "Model training completed successfully")
        except Exception as e:
            log_event("error", "training_failed", "Model training failed", error_message=str(e))
            raise
        
        # Test model
        try:
            log_event("info", "testing_started", "Model testing started")
            test_results = trainer.test(model, data_module)
            log_event("info", "testing_completed", "Model testing completed successfully", 
                     test_results=str(test_results))
        except Exception as e:
            log_event("error", "testing_failed", "Model testing failed", error_message=str(e))
            raise
            
        # Save the final model
        try:
            final_model_path = os.path.join(config.get('model_dir', './models'), 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            log_event("info", "model_saved", "Final model saved successfully", 
                     model_path=final_model_path)
        except Exception as e:
            log_event("error", "model_save_failed", "Failed to save final model", error_message=str(e))
            
        log_event("info", "training_pipeline_complete", "Training pipeline completed successfully")
        
    except Exception as e:
        log_event("error", "training_pipeline_failed", "Training pipeline failed with uncaught exception", 
                 error_type=type(e).__name__, error_message=str(e))
        logger.exception("Training pipeline failed with uncaught exception")
        raise


if __name__ == "__main__":
    main()
