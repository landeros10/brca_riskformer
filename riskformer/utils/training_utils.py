'''
Created Feb 2022
author: landeros10

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital
'''
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
import torch
import numpy as np
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

@dataclass
class PatchInfo:
    """
    Data class to store information about a patch for reconstruction.
    
    Attributes:
        feature_id: ID of the feature this patch belongs to
        region_id: ID of the region within the feature
        region_min_row: Minimum row coordinate of the region in the feature
        region_min_col: Minimum column coordinate of the region in the feature
        region_max_row: Maximum row coordinate of the region in the feature
        region_max_col: Maximum column coordinate of the region in the feature
        patch_row_start: Starting row of the patch within the region
        patch_col_start: Starting column of the patch within the region
        patch_row_end: Ending row of the patch within the region
        patch_col_end: Ending column of the patch within the region
    """
    feature_id: int
    region_id: int
    region_min_row: int
    region_min_col: int
    region_max_row: int
    region_max_col: int
    patch_row_start: int
    patch_col_start: int
    patch_row_end: int
    patch_col_end: int
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'PatchInfo':
        """Convert a tensor row to a PatchInfo instance."""
        if len(tensor) >= 10:
            return cls(
                feature_id=int(tensor[0].item()),
                region_id=int(tensor[1].item()),
                region_min_row=int(tensor[2].item()),
                region_min_col=int(tensor[3].item()),
                region_max_row=int(tensor[4].item()),
                region_max_col=int(tensor[5].item()),
                patch_row_start=int(tensor[6].item()),
                patch_col_start=int(tensor[7].item()),
                patch_row_end=int(tensor[8].item()),
                patch_col_end=int(tensor[9].item())
            )
        raise ValueError(f"Expected tensor with at least 10 elements, got {len(tensor)}")
    
    @classmethod
    def from_tensor_batch(cls, tensor_batch: torch.Tensor) -> List['PatchInfo']:
        """Convert a batch of tensor rows to a list of PatchInfo instances."""
        return [cls.from_tensor(row) for row in tensor_batch]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert a PatchInfo instance to a tensor."""
        return torch.tensor([
            self.feature_id, self.region_id,
            self.region_min_row, self.region_min_col, self.region_max_row, self.region_max_col,
            self.patch_row_start, self.patch_col_start, self.patch_row_end, self.patch_col_end
        ], dtype=torch.int32)
    
    @property
    def patch_height(self) -> int:
        """Get the height of the patch."""
        return self.patch_row_end - self.patch_row_start
    
    @property
    def patch_width(self) -> int:
        """Get the width of the patch."""
        return self.patch_col_end - self.patch_col_start
    
    @property
    def region_row_start(self) -> int:
        """Get the starting row of the patch in the feature space."""
        return self.region_min_row + self.patch_row_start
    
    @property
    def region_col_start(self) -> int:
        """Get the starting column of the patch in the feature space."""
        return self.region_min_col + self.patch_col_start
    
    @property
    def region_row_end(self) -> int:
        """Get the ending row of the patch in the feature space."""
        return self.region_row_start + self.patch_height
    
    @property
    def region_col_end(self) -> int:
        """Get the ending column of the patch in the feature space."""
        return self.region_col_start + self.patch_width


def set_seed(seed):
    """
    Set all relevant seeds for reproducibility in Python, NumPy, and PyTorch.
        
    Args:
        seed (int): seed to set
    """
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rearrange_xl_patches(xl_patches, patch_info):
    """
    Rearrange the patches into their original orders and fill in with zeros.
    
    Args:
        xl_patches (torch.Tensor): The patches to rearrange. Shape (N, D, H, W)
        patch_info (torch.Tensor): The information about the patches.
        
    Returns:
        torch.Tensor: The rearranged patches.
        tuple: The row and column starts for each feature.
    """
    # R-arrange tensor to (N, H, W, D)
    xl_patches = xl_patches.permute(0, 2, 3, 1)
    feature_dim = xl_patches.shape[-1]

    # Convert tensor to PatchInfo objects for easier handling
    patch_infos = PatchInfo.from_tensor_batch(patch_info)
    
    # Get number of unique features
    if not patch_infos:
        return []
    
    feature_ids = set(info.feature_id for info in patch_infos)
    
    reconstructed_features = []
    patch_id = 0
    row_starts = []
    col_starts = []
    
    for feature_id in sorted(feature_ids):
        # Get all patches for this feature
        feature_patches = [p for p in patch_infos if p.feature_id == feature_id]
        
        # Get all region IDs for this feature
        region_ids = set(p.region_id for p in feature_patches)
        
        # Get feature dimensions
        feature_max_row = max(p.region_max_row for p in feature_patches)
        feature_max_col = max(p.region_max_col for p in feature_patches)
        
        # Create empty feature tensor
        single_feature = torch.zeros((feature_max_row, feature_max_col, feature_dim))
        
        # Fill in patches
        for patch_info in feature_patches:
            
            # Get the patch
            patch = xl_patches[patch_id]
            
            # Place the patch in the feature
            single_feature[
                patch_info.region_row_start:patch_info.region_row_end,
                patch_info.region_col_start:patch_info.region_col_end,
                :
            ] = patch[:patch_info.patch_height, :patch_info.patch_width, :]
            
            patch_id += 1
            row_starts.append((feature_id, patch_info.region_row_start))
            col_starts.append((feature_id, patch_info.region_col_start))
            
        reconstructed_features.append(single_feature)
    
    return reconstructed_features, (set(row_starts), set(col_starts))


def get_loss_function(loss_fn_name, task_type=None):
    """
    Convert a loss function string name to an actual PyTorch loss function.
    
    Args:
        loss_fn_name: String name of the loss function or existing loss function object
        task_type: Optional task type ('binary', 'multiclass', 'regression') for default loss
        
    Returns:
        An instantiated PyTorch loss function
    """
    # If already a loss function (not a string), return it directly
    if not isinstance(loss_fn_name, str):
        return loss_fn_name
        
    # Map string names to actual loss functions
    if loss_fn_name == "MSELoss":
        return nn.MSELoss()
    elif loss_fn_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_fn_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_fn_name == "L1Loss":
        return nn.L1Loss()
    elif loss_fn_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif loss_fn_name == "BCELoss":
        return nn.BCELoss()
    else:
        # If not recognized and task_type provided, use default for task type
        if task_type:
            return _get_default_loss(task_type)
        else:
            raise ValueError(f"Unrecognized loss function: {loss_fn_name}")


def create_slide_level_loss(task_configs: Dict[str, Dict], regional_coeff: float = 0.0):
    """
    Create a slide-level loss function based on task configurations.
    
    Args:
        task_configs: Dictionary mapping task names to their configurations
        regional_coeff: Coefficient for weighting local vs global loss
        
    Returns:
        A function that calculates losses for each task based on the provided configurations
    """
    # Extract essential information from task configurations
    task_types = {task: cfg['type'] for task, cfg in task_configs.items()}
    task_weights = {task: cfg.get('weight', 1.0) for task, cfg in task_configs.items()}
    
    # Set up loss functions for each task
    loss_fns = {}
    for task, cfg in task_configs.items():
        if 'loss_fn' in cfg:
            loss_fns[task] = get_loss_function(cfg['loss_fn'], task_types[task])
        else:
            loss_fns[task] = _get_default_loss(cfg['type'])
    
    def loss_fn(predictions, labels):
        """
        Calculate loss for predictions and labels across all tasks.
        
        Args:
            predictions: Dictionary mapping task names to prediction tensors
                Each tensor has shape [N, C] where N is number of predictions 
                (1 global + M instance predictions) and C is number of classes/outputs
                OR a tuple containing (task_outputs, attns, global_weights) where task_outputs is the dict
            labels: Dictionary mapping task names to label tensors
                
        Returns:
            Dictionary mapping task names to their individual losses, with 'total' key for weighted sum
        """
        # Handle predictions which may include attention weights (legacy format)
        if isinstance(predictions, tuple):
            # Extract just the predictions from the tuple (task_outputs, attns, global_weights)
            predictions = predictions[0]
            
        # If predictions is not a dictionary, create one using task names (legacy format)
        if not isinstance(predictions, dict):
            # Create dictionary mapping first task to predictions
            if task_configs:
                first_task = next(iter(task_configs.keys()))
                predictions = {first_task: predictions}
            else:
                return {"total": torch.tensor(0.0, device=predictions.device if hasattr(predictions, 'device') else 'cpu')}
                
        # If labels is not a dictionary, create one using task names (legacy format)
        if not isinstance(labels, dict):
            # Create dictionary mapping tasks to the same labels
            labels = {task: labels for task in predictions.keys()}
            
        task_losses = {}
        total_loss = 0.0
        
        # Process each task
        for task_name, pred_tensor in predictions.items():
            # Skip tasks without corresponding labels
            if task_name not in labels:
                continue
                
            # Skip tasks not defined in task_configs or types
            if task_name not in task_types:
                continue
                
            # Skip tasks without loss functions
            if task_name not in loss_fns:
                continue
            
            # Get label and ensure it has proper shape and type
            label = labels[task_name]
            
            # Skip if either prediction or label is None
            if pred_tensor is None or label is None:
                continue
                
            # Get device for creating tensors
            device = pred_tensor.device if hasattr(pred_tensor, 'device') else 'cpu'
            
            if isinstance(label, torch.Tensor):
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                
                # Convert to long type for multiclass tasks
                if task_types[task_name] == 'multiclass' and label.dtype != torch.long:
                    label = label.long()
            
            # Ensure prediction tensor has proper shape
            if len(pred_tensor.shape) == 1:
                pred_tensor = pred_tensor.unsqueeze(0)
            
            # Get loss function for this task
            loss_fn = loss_fns[task_name]
            
            # Get task weight
            task_weight = task_weights.get(task_name, 1.0)
            
            # Calculate global loss (from first prediction)
            global_pred = pred_tensor[0].unsqueeze(0)  # Add batch dimension
            global_loss = loss_fn(global_pred, label)
            
            # Apply regional coefficient to global loss
            global_loss = global_loss * (1 - regional_coeff) * task_weight
            
            # Initialize total task loss with global loss
            task_loss = global_loss
            
            # Calculate local loss if regional coefficient > 0 and we have instance predictions
            if regional_coeff > 0 and pred_tensor.shape[0] > 1:
                # Get instance predictions (all except the first)
                instance_preds = pred_tensor[1:]
                
                # Skip local loss if no instance predictions
                if instance_preds.shape[0] == 0:
                    task_losses[task_name] = task_loss
                    total_loss = total_loss + task_loss
                    continue
                
                # Calculate number of top instances to use (10% of total)
                total_instances = instance_preds.shape[0]
                k = max(1, total_instances // 10)
                
                # Select top-k instances based on confidence score
                # For binary tasks, use the positive class score (index 1 or last column)
                # For multiclass, use the predicted class or first column
                # For regression, use the prediction value directly
                if task_types[task_name] == 'binary' and instance_preds.shape[1] > 1:
                    # Binary classification with multiple outputs - use positive class score
                    confidence_scores = instance_preds[:, -1]
                else:
                    # Use first column for all other cases
                    confidence_scores = instance_preds[:, 0]
                
                # Get top-k indices and corresponding predictions
                top_k_values, top_k_indices = torch.topk(
                    confidence_scores, k=min(k, total_instances)
                )
                top_k_preds = instance_preds[top_k_indices]
                
                # Expand label to match top-k predictions
                if isinstance(label, torch.Tensor):
                    # Handle different label shapes
                    if label.dim() > 0:
                        expanded_dims = [top_k_indices.shape[0]] + [1] * (label.dim() - 1)
                        expanded_label = label.expand(*expanded_dims, *label.shape[1:])
                    else:
                        expanded_label = label.expand(top_k_indices.shape[0])
                else:
                    expanded_label = torch.tensor([label] * top_k_indices.shape[0], device=device)
                
                # Calculate local loss
                local_loss = loss_fn(top_k_preds, expanded_label)
                
                # If loss returned has multiple values (one per sample), reduce to mean
                if isinstance(local_loss, torch.Tensor) and local_loss.dim() > 0:
                    local_loss = torch.mean(local_loss)
                
                # Apply regional coefficient to local loss
                local_loss = local_loss * regional_coeff * task_weight
                
                # Add local loss to task loss
                task_loss = task_loss + local_loss
            
            # Store task loss
            task_losses[task_name] = task_loss
            
            # Add to total loss
            total_loss = total_loss + task_loss
        
        # If no losses calculated, return zero tensor
        if not task_losses:
            device = next(iter(predictions.values())).device if predictions else 'cpu'
            return {"total": torch.tensor(0.0, device=device)}
            
        # Add total loss to the dictionary
        task_losses['total'] = total_loss
        
        return task_losses
    
    return loss_fn


def _get_default_loss(task_type):
    """Get default loss function based on task type"""
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        return nn.CrossEntropyLoss()
    else:  # regression
        return nn.MSELoss()


def convert_to_soft_label(score, beta=1.50):
    cutoff = 0.7169
    min_score = -2.009
    max_score = 2.744
    if score <= cutoff:
        soft_label = (score - min_score) / (cutoff - min_score)
        return 0.50 * soft_label ** beta
    else:
        soft_label = (score - cutoff) / (max_score - cutoff)
        return 1 - 0.50 * (1 - soft_label) ** beta
    return soft_label


def split_riskformer_data(
    examples: Dict[str, Dict],
    label_var: str = "odx85",
    positive_label: str = "H",
    test_split_ratio: float = 0.2,
    seed: int = 42
):
    """
    Split data into train and test sets. Balances test set to have
    equal number of positive and negative samples based on the data variable provided.
    
    Args:
        examples (dict): Dictionary of SVS file paths and corresponding dictionary of data.
        label_var (str): The key in the data dictionary that contains the label.
        positive_label (str): The value that indicates a positive sample.
        test_split_ratio (float): Ratio of data to use for testing.
    
    Returns:
        tuple: Two dictionaries, one for training data and one for testing data.
    """
    patient_ids = np.array(list(examples.keys()))
    labels = np.array([
        examples[patient_id][label_var]
        for patient_id in patient_ids
    ])


    num_pos = int(len(patient_ids) * (test_split_ratio) / 2)
    if num_pos == 0:
        logger.error("Test split ratio too low, not enough samples.")
        raise ValueError("Test split ratio too low, not enough samples.")

    pos_samples = patient_ids[labels == positive_label]
    neg_samples = patient_ids[labels != positive_label]
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        logger.error("No positive or negative samples found.")
        raise ValueError("No positive or negative samples found.")

    logger.debug(f"Dataset contains {len(patient_ids)} samples, {len(pos_samples)} positive and {len(neg_samples)} negative samples.")
    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)

    test_data = {
        **{patient_id: examples[patient_id] for patient_id in pos_samples[:num_pos]},
        **{patient_id: examples[patient_id] for patient_id in neg_samples[:num_pos]}
    }
    logger.debug(f"Created Test Dataset with {len(test_data)} samples, {num_pos} positive and {num_pos} negative samples.")
    train_data = {
        **{patient_id: examples[patient_id] for patient_id in pos_samples[num_pos:]},
        **{patient_id: examples[patient_id] for patient_id in neg_samples[num_pos:]}
    }
    logger.debug(f"Created Train Dataset with {len(train_data)} samples, {len(pos_samples) - num_pos} positive and {len(neg_samples) - num_pos} negative samples.")
    return train_data, test_data


