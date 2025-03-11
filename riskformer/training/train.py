#!/usr/bin/env python
'''
Created June 2023
author: landeros10
Updated with PyTorch Lightning support

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital

Massachusetts Institute of Technology

Main Training Pipeline
'''
from __future__ import division, print_function

import os
import argparse
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from os.path import join

from riskformer.training.model import RiskFormerLightningModule
from riskformer.data.datasets import RiskFormerDataModule
from riskformer.utils.logger_config import logger_setup
from riskformer.utils.config_utils import load_train_config

logger_setup()
logger = logging.getLogger(__name__)

SIZE = 256

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RiskFormer Training Configurations")

    # Config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata file (overrides config)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for logs (overrides config)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment (overrides config)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config)")
    parser.add_argument("--debug", action="store_true",
                        help="Set to run in debug mode")
    return parser.parse_args()


def main():
    """Main training function."""
    logger.info("Starting Training Pipeline...")
    logger.info("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration from file
    config = load_train_config(args.config)
    if not config:
        raise ValueError(f"Failed to load config from {args.config}")
    
    # Override with command-line arguments if provided
    if args.metadata_file:
        config['metadata_file'] = args.metadata_file
    if args.log_dir:
        config['log_dir'] = args.log_dir
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.use_wandb:
        config['use_wandb'] = True
    if args.seed is not None:
        config['seed'] = args.seed
    if args.debug:
        config['debug'] = True
    
    # Log the loaded configuration
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    pl.seed_everything(seed)
    
    # Create data module
    data_module = RiskFormerDataModule(
        s3_bucket=config['s3_bucket'],
        s3_prefix=config.get('s3_prefix', ''),
        max_dim=config.get('max_dim', 32),
        overlap=config.get('overlap', 0.0),
        metadata_file=config.get('metadata_file'),
        cache_dir=config.get('cache_dir'),
        profile_name=config.get('profile_name'),
        region_name=config.get('region_name'),
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        val_split=config.get('val_split', 0.2),
        test_split=config.get('test_split', 0.1),
        seed=seed,
        config_path=args.config,  # Pass the config path to the data module
    )
    
    # Create model from config
    model = RiskFormerLightningModule.from_config_file(args.config)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="riskformer-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.get('early_stop', 25),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Create logger
    if config.get('use_wandb', False):
        tb_logger = WandbLogger(
            project=config.get('wandb_project', 'riskformer'),
            name=config.get('experiment_name', 'riskformer'),
            entity=config.get('wandb_entity'),
            log_model=True,
        )
    else:
        tb_logger = TensorBoardLogger(
            save_dir=config.get('log_dir', 'lightning_logs'),
            name=config.get('experiment_name', 'riskformer'),
        )
    
    # Create trainer
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
        deterministic=True,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
