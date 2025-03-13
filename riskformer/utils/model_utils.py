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

def create_lightning_module_from_config(
        config_path: str,
        task_weights: Optional[Dict[str, float]] = None
) -> RiskFormerLightningModule:
    """
    Create a RiskFormerLightningModule from a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        task_weights: Optional dictionary mapping task names to task weights.
        
    Returns:
        An initialized RiskFormerLightningModule.
    """
    logger.info(f"Creating RiskFormerLightningModule from config file: {config_path}")
    
    # Load config
    config = load_training_config(config_path)
    
    # Prepare task configurations if not already in config
    if 'tasks' not in config:
        raise ValueError("Config must contain 'tasks' section with task types")
    
    # If task weights are provided, update the weights in the tasks config
    if task_weights:
        for task, weight in task_weights.items():
            if task in config['tasks']:
                config['tasks'][task]['weight'] = weight
    
    # Get regional coefficient
    regional_coeff = config.get('regional_coeff', 0.0)
    
    # Create Lightning module
    return RiskFormerLightningModule.from_config(
        config=config,
        regional_coeff=regional_coeff
    )

def example_usage():
    """
    Example of how to use the config-based model creation.
    This function is for demonstration purposes.
    """
    # Path to config file
    config_path = "configs/training/riskformer_config.yaml"
    
    # Create base model directly
    model = create_model_from_config(config_path)
    print(f"Created RiskFormer_ViT model with input dimension: {model.input_embed_dim}")
    
    # Create Lightning module with default task weights
    lightning_module = create_lightning_module_from_config(config_path)
    print(f"Created RiskFormerLightningModule with optimizer: {lightning_module.optimizer_config['optimizer']}")
    
    # Create Lightning module with custom task weights
    task_weights = {"odx_train": 1.0, "odx85": 0.5, "mphr": 0.5}
    lightning_module_weighted = create_lightning_module_from_config(
        config_path, 
        task_weights=task_weights
    )
    print(f"Created weighted RiskFormerLightningModule with tasks: {list(task_weights.keys())}")
    
    return model, lightning_module, lightning_module_weighted

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_usage()
