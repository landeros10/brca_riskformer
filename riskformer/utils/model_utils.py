"""
Utility functions for creating and handling RiskFormer models.
"""

import os
import yaml
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from riskformer.training.model import RiskFormer_ViT, RiskFormerLightningModule
from riskformer.utils.training_utils import load_training_config

logger = logging.getLogger(__name__)

def create_model_from_config(config_path: str) -> RiskFormer_ViT:
    """
    Create a RiskFormer_ViT model from a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        An initialized RiskFormer_ViT model.
    """
    logger.info(f"Creating RiskFormer_ViT model from config file: {config_path}")
    return RiskFormer_ViT.from_config_file(config_path)

def setup_loss_functions(config: Dict[str, Any]) -> Dict[str, Dict[int, nn.Module]]:
    """
    Set up loss functions based on task types in the config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping task names to loss functions.
    """
    class_loss_map = {}
    
    # Get label configuration
    if 'labels' not in config:
        raise ValueError("Config must contain 'labels' section with task types")
    
    labels_config = config['labels']
    task_types = labels_config.get('task_types', {})
    
    for task, task_type in task_types.items():
        if task_type == 'binary':
            class_loss_map[task] = {0: nn.BCEWithLogitsLoss()}
        elif task_type == 'regression':
            class_loss_map[task] = {0: nn.MSELoss()}
        elif task_type == 'multiclass':
            # Assuming num_classes for multiclass tasks is specified in the config
            num_classes = config.get('num_classes', {}).get(task, 2)
            class_loss_map[task] = {0: nn.CrossEntropyLoss()}
        else:
            raise ValueError(f"Unknown task type: {task_type} for task {task}")
    
    return class_loss_map


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_usage()
