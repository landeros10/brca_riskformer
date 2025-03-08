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
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from riskformer.training.model import RiskFormerLightningModule
from riskformer.data.datasets import RiskFormerDataModule
from riskformer.utils.logger_config import logger_setup
from riskformer.utils import log_training_params

logger_setup()
logger = logging.getLogger(__name__)

SIZE = 256

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RiskFormer Training Configurations")

    # Data parameters
    parser.add_argument("--s3_bucket", type=str, required=True,
                        help="S3 bucket containing the data")
    parser.add_argument("--s3_prefix", type=str, default="",
                        help="Prefix for S3 objects")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache S3 files")
    parser.add_argument("--profile_name", type=str, default=None,
                        help="AWS profile name")
    parser.add_argument("--region_name", type=str, default=None,
                        help="AWS region name")
    
    # Dataset parameters
    parser.add_argument("--max_dim", type=int, default=32,
                        help="Maximum dimension for patches")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap between patches")
    parser.add_argument("--sample_size", type=int, default=-1,
                        help="Size of the training dataset. Set to -1 to use full dataset")
    
    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloaders")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model parameters
    parser.add_argument("--input_embed_dim", type=int, default=1024,
                        help="Input embedding dimension")
    parser.add_argument("--output_embed_dim", type=int, default=512,
                        help="Output embedding dimension")
    parser.add_argument("--use_phi", type=bool, default=True,
                        help="Whether to use phi embedding")
    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                        help="Drop path rate")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of output classes")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of the local transformer blocks")
    parser.add_argument("--global_depth", type=int, default=2,
                        help="Depth of the global transformer blocks")
    parser.add_argument("--encoding_method", type=str, default="sinusoidal",
                        help="Position encoding method")
    parser.add_argument("--mask_num", type=int, default=0,
                        help="Number of masks to use")
    parser.add_argument("--mask_preglobal", type=bool, default=False,
                        help="Whether to mask before global attention")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--use_attn_mask", type=bool, default=False,
                        help="Whether to use attention mask")
    parser.add_argument("--mlp_ratio", type=float, default=4.0,
                        help="MLP ratio")
    parser.add_argument("--use_class_token", type=bool, default=True,
                        help="Whether to use class token")
    parser.add_argument("--global_k", type=int, default=64,
                        help="Number of global tokens")
    
    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd", "momentum", "nadam", "lars", "lamb"],
                        help="Optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "onecycle"],
                        help="Learning rate scheduler")
    parser.add_argument("--learning_rate_scaling", type=str, default="linear",
                        choices=["linear", "sqrt"],
                        help="How to scale the learning rate as a function of batch size.")
    parser.add_argument("--learning_rate_warmup_epochs", type=int, default=25,
                        help="Number of epochs for warmup.")
    parser.add_argument("--regional_coeff", type=float, default=0.0,
                        help="Regional coefficient for loss")
    parser.add_argument("--patch_coeff", type=float, default=1.0,
                        help="Weight to apply to patch-level risk predictions.")
    parser.add_argument("--l2_coeff", type=float, default=1e-6,
                        help="L2 regularization coefficient.")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--min_epochs", type=int, default=10,
                        help="Minimum number of epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["32", "16", "bf16"],
                        help="Precision for training")
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="Accelerator to use")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices to use")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy for distributed training")
    parser.add_argument("--train_steps", type=int, default=100,
                        help="Total number of training steps.")
    parser.add_argument("--eval_every", type=int, default=8,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--early_stop", type=int, default=25,
                        help="Number of epochs to wait before early stopping.")
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="lightning_logs",
                        help="Directory for logs")
    parser.add_argument("--experiment_name", type=str, default="riskformer",
                        help="Name of the experiment")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="riskformer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Set to run in debug mode.")

    args = parser.parse_args()
    
    # If config file is provided, load it and update args
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    config_params = vars(args)
    log_training_params(logger, config_params)
    
    return args


def main():
    """Main training function."""
    logger.info("Starting Training Pipeline...")
    logger.info("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = RiskFormerDataModule(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        max_dim=args.max_dim,
        overlap=args.overlap,
        metadata_file=args.metadata_file,
        cache_dir=args.cache_dir,
        profile_name=args.profile_name,
        region_name=args.region_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    
    # Create model config
    model_config = {
        "input_embed_dim": args.input_embed_dim,
        "output_embed_dim": args.output_embed_dim,
        "use_phi": args.use_phi,
        "drop_path_rate": args.drop_path_rate,
        "drop_rate": args.drop_rate,
        "num_classes": args.num_classes,
        "max_dim": args.max_dim,
        "depth": args.depth,
        "global_depth": args.global_depth,
        "encoding_method": args.encoding_method,
        "mask_num": args.mask_num,
        "mask_preglobal": args.mask_preglobal,
        "num_heads": args.num_heads,
        "use_attn_mask": args.use_attn_mask,
        "mlp_ratio": args.mlp_ratio,
        "use_class_token": args.use_class_token,
        "global_k": args.global_k,
    }
    
    # Create optimizer config
    optimizer_config = {
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "patience": args.patience,
        "learning_rate_scaling": args.learning_rate_scaling,
        "learning_rate_warmup_epochs": args.learning_rate_warmup_epochs,
    }
    
    # Create loss functions
    class_loss_map = {}
    
    # Binary classification tasks
    if args.num_classes == 1:
        # ODX85 - Binary classification (High/Low risk)
        class_loss_map['odx85'] = {0: nn.BCEWithLogitsLoss()}
        
        # MPHR - Binary classification (High/Low risk)
        class_loss_map['mphr'] = {0: nn.BCEWithLogitsLoss()}
        
        # Necrosis - Binary classification (Present/Absent)
        class_loss_map['necrosis'] = {0: nn.BCEWithLogitsLoss()}
        
        # Pleomorphism - Binary classification
        class_loss_map['pleo'] = {0: nn.BCEWithLogitsLoss()}
    else:
        # Multi-class classification
        class_loss_map['odx85'] = {i: nn.CrossEntropyLoss() for i in range(args.num_classes)}
    
    # Regression tasks
    class_loss_map['odx_train'] = {0: nn.MSELoss()}
    class_loss_map['dfm'] = {0: nn.MSELoss()}
    
    # Task weights (optional)
    task_weights = {
        'odx85': 1.0,      # Primary task
        'mphr': 0.5,       # Secondary task
        'necrosis': 0.3,   # Tertiary task
        'pleo': 0.3,       # Tertiary task
        'odx_train': 0.5,  # Secondary task
        'dfm': 0.3,        # Tertiary task
    }
    
    # Create model
    model = RiskFormerLightningModule(
        model_config=model_config,
        optimizer_config=optimizer_config,
        class_loss_map=class_loss_map,
        task_weights=task_weights,
        regional_coeff=args.regional_coeff,
    )
    
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
            patience=args.early_stop,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Create logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            entity=args.wandb_entity,
            log_model=True,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.experiment_name,
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy if args.strategy else "auto",
        log_every_n_steps=10,
        deterministic=True,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
