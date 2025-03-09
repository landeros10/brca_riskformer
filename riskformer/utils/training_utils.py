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
from typing import List, Tuple, Optional
import torch.nn as nn

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rearrange_xl_patches(xl_patches, patch_info):
    """
    Rearrange the patches into their original orders and fill in with zeros.
    
    Args:
        xl_patches (torch.Tensor): The patches to rearrange.
        patch_info (torch.Tensor): The information about the patches.
        
    Returns:
        torch.Tensor: The rearranged patches.
        tuple: The row and column starts for each feature.
    """
    max_dim = xl_patches.shape[1]
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


def slide_level_loss(
        predictions, 
        labels,
        class_loss_map,
        regional_coeff=0.0,
        task_types=None
        ):
    """
    Calculate the slide-level loss for a batch of predictions and labels. Supports multi-task
    learning with different task types (binary, regression, multiclass).
    
    Args:
        predictions (torch.Tensor or dict): The predictions from the model, shape (batch_size, num_classes)
                                           or a dictionary of task to prediction tensors
        labels (torch.Tensor or dict): The labels for the batch or a dictionary of task to label tensors
        class_loss_map (dict): A dictionary mapping task name or class index to loss function
        regional_coeff (float): Coefficient for weighting local vs global loss
        task_types (dict, optional): Dictionary mapping task names to their types ('binary', 'regression', 'multiclass')
        
    Returns:
        torch.Tensor: The slide-level loss.
    """
    # Check if predictions is a dictionary (multi-task)
    is_dict_predictions = isinstance(predictions, dict)
    
    # If not a dictionary, check tensor shape
    if not is_dict_predictions:
        # Ensure predictions has the right shape
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
    
        # Get the device of the predictions tensor
        device = predictions.device
    
        # Global Loss
        global_pred = predictions[0]  # (num_classes,)
    else:
        # Get the device from the first prediction tensor in the dictionary
        device = next(iter(predictions.values())).device
        
        # For dictionary output, we'll calculate loss per task
        global_loss = torch.tensor(0.0, device=device)
        
        # Process each task in the predictions dictionary
        for task_name, pred_tensor in predictions.items():
            if task_name in labels:
                # Get the label for this task
                task_label = labels[task_name]
                
                # Get task type if available
                task_type = task_types.get(task_name, 'binary') if task_types else 'binary'
                
                # Get loss function for this task
                loss_fn = class_loss_map.get(task_name, next(iter(class_loss_map.values())))
                
                # Ensure loss_fn is callable
                loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                
                # Apply appropriate transformations based on task_type
                if task_type == 'multiclass':
                    # Ensure label is the right dtype for multiclass
                    if task_label.dtype != torch.long:
                        task_label = task_label.long()
                    
                    # Ensure prediction is right shape for CrossEntropyLoss [batch_size, num_classes]
                    if len(pred_tensor.shape) == 1:
                        pred_tensor = pred_tensor.unsqueeze(0)
                
                # Calculate task loss
                task_loss = loss_fn_callable(pred_tensor, task_label)
                
                # If the loss function returns one value per sample (reduction='none'),
                # we need to reduce it to a scalar before adding to global_loss
                if task_loss.dim() > 0:
                    task_loss = task_loss.mean()
                    
                global_loss += task_loss
        
        # If using dictionary format, we're done - no need to process legacy behavior
        if regional_coeff == 0:
            return global_loss
        
        # If regional coefficient is provided, we need to use the legacy behavior
        # We'll extract task info from the first task to use for regional loss
        first_task = next(iter(predictions.keys()))
        predictions = predictions[first_task]
        labels = labels[first_task]
        
        # Ensure we have the right shape
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
        
        # Get global_pred for regional loss calculation
        global_pred = predictions[0]
    
    # Initialize global loss (only if not already calculated)
    if not is_dict_predictions:
        global_loss = torch.tensor(0.0, device=device)
    
        # Handle multi-task learning based on task types
        if task_types is not None:
            # Multi-task learning with specified task types
            for task_name, loss_fn in class_loss_map.items():
                if task_name in task_types:
                    task_type = task_types[task_name]
                    task_idx = list(task_types.keys()).index(task_name)
                    
                    # Get the prediction for this task
                    if len(global_pred.shape) > 0 and task_idx < global_pred.shape[0]:
                        task_pred = global_pred[task_idx].unsqueeze(0)
                    else:
                        # If predictions aren't aligned with task indices, use the task_name as index
                        # This assumes predictions are ordered according to task names in task_types
                        task_pred = global_pred
                    
                    # Get the label for this task
                    if task_name in labels:
                        task_label = labels[task_name]
                    elif isinstance(labels, torch.Tensor) and len(labels.shape) > 0 and task_idx < labels.shape[0]:
                        task_label = labels[task_idx].unsqueeze(0)
                    else:
                        # Default case - use the first available label
                        task_label = next(iter(labels.values())) if isinstance(labels, dict) else labels
                    
                    # Apply appropriate processing based on task type
                    if task_type == 'binary':
                        # Binary classification
                        if not isinstance(task_label, torch.Tensor):
                            task_label = torch.tensor([float(task_label)], device=device)
                        
                        # Get the loss function, which is stored in a dictionary with key 0
                        loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                        task_loss = loss_fn_callable(task_pred, task_label)
                    
                    elif task_type == 'regression':
                        # Regression tasks
                        if not isinstance(task_label, torch.Tensor):
                            task_label = torch.tensor([float(task_label)], device=device)
                        
                        # Get the loss function, which is stored in a dictionary with key 0
                        loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                        task_loss = loss_fn_callable(task_pred, task_label)
                    
                    elif task_type == 'multiclass':
                        # Multi-class classification
                        if not isinstance(task_label, torch.Tensor):
                            task_label = torch.tensor([int(task_label)], device=device, dtype=torch.long)
                        elif task_label.dtype != torch.long:
                            task_label = task_label.long()
                        
                        # CrossEntropyLoss expects raw logits, not probabilities
                        # The shape of predictions should be [batch_size, num_classes]
                        # The shape of targets should be [batch_size] (not one-hot encoded)
                        if len(task_pred.shape) == 1:
                            # If we have a single output, add batch dimension
                            task_pred = task_pred.unsqueeze(0)
                        
                        # Get the loss function, which is stored in a dictionary with key 0
                        loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                        task_loss = loss_fn_callable(task_pred, task_label)
                    
                    else:
                        # Unknown task type, default behavior
                        loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                        task_loss = loss_fn_callable(task_pred, task_label)
                    
                    # If the loss function returns one value per sample (reduction='none'),
                    # we need to reduce it to a scalar before adding to global_loss
                    if task_loss.dim() > 0:
                        task_loss = task_loss.mean()
                        
                    # Add to global loss
                    global_loss += task_loss
        else:
            # Legacy behavior - calculate loss using class indices
            is_binary = predictions.shape[1] == 1 or (len(class_loss_map) == 1 and 0 in class_loss_map)
            
            # Calculate all class losses at once if using the same loss function
            if len(set(class_loss_map.values())) == 1:
                # If all classes use the same loss function
                loss_fn = next(iter(class_loss_map.values()))
                # If the loss function is a dictionary, get the actual callable
                if isinstance(loss_fn, dict):
                    loss_fn = loss_fn[0]
                
                # For binary classification with a single label value
                if is_binary and isinstance(labels, torch.Tensor) and len(labels.shape) == 1 and labels.shape[0] == 1:
                    # Binary classification with a single label
                    if global_pred.shape[0] > 1:
                        # If prediction has multiple outputs but we're treating it as binary
                        # Use the first output
                        global_loss = loss_fn(global_pred[0].unsqueeze(0), labels)
                    else:
                        global_loss = loss_fn(global_pred, labels)
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    # Special handling for CrossEntropyLoss to ensure labels are long type
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor([int(labels)], device=device, dtype=torch.long)
                    elif labels.dtype != torch.long:
                        labels = labels.long()
                    
                    # Ensure prediction has shape [batch_size, num_classes]
                    if len(global_pred.shape) == 1:
                        # For multiclass, reshape to [1, num_classes]
                        global_pred = global_pred.unsqueeze(0)
                    
                    global_loss = loss_fn(global_pred, labels)
                else:
                    # Multi-class or properly shaped binary
                    global_loss = loss_fn(global_pred, labels)
            else:
                # If different classes use different loss functions
                for class_idx, loss_fn in class_loss_map.items():
                    # If the loss function is a dictionary, get the actual callable
                    if isinstance(loss_fn, dict):
                        loss_fn = loss_fn[0]
                    class_pred = global_pred[class_idx].unsqueeze(0)
                    
                    # Handle different label shapes
                    if len(labels.shape) == 1 and labels.shape[0] == 1 and len(class_loss_map) > 1:
                        # Single label value but multiple classes - use the same label for all
                        class_label = labels.expand(global_pred.shape[0])
                    elif len(labels.shape) >= 1 and class_idx < labels.shape[0]:
                        # Multiple labels, one per class
                        class_label = labels[class_idx].unsqueeze(0)
                    else:
                        # Default case - use the first label
                        class_label = labels[0].unsqueeze(0)
                        
                    # Special handling for CrossEntropyLoss
                    if isinstance(loss_fn, nn.CrossEntropyLoss):
                        if class_label.dtype != torch.long:
                            class_label = class_label.long()
                    
                    # Calculate task loss
                    task_loss = loss_fn(class_pred, class_label)
                    
                    # If the loss function returns one value per sample (reduction='none'),
                    # we need to reduce it to a scalar before adding to global_loss
                    if task_loss.dim() > 0:
                        task_loss = task_loss.mean()
                        
                    global_loss += task_loss
    
    global_loss = global_loss * (1 - regional_coeff)

    # Skip local loss calculation if regional_coeff is 0
    if regional_coeff == 0:
        return global_loss

    # Top-K Instance Local Loss
    total_instances = predictions.shape[0] - 1
    if total_instances == 0:
        return global_loss
        
    k = max(1, total_instances // 10)

    # Choose top k based on first class
    instance_preds = predictions[1:]  # All instance predictions
    
    # Handle different prediction shapes
    if instance_preds.shape[1] > 0:
        top_k_values, top_k_indices = torch.topk(instance_preds[:, 0], k=min(k, instance_preds.shape[0]))
        
        # More efficient gathering of top-k predictions
        top_k_preds = instance_preds[top_k_indices]
        
        # Calculate instance loss for each in top k
        local_loss = torch.tensor(0.0, device=device)
        
        # Handle multi-task learning with task types
        if task_types is not None:
            for task_name, loss_fn in class_loss_map.items():
                if task_name in task_types:
                    task_type = task_types[task_name]
                    task_idx = list(task_types.keys()).index(task_name)
                    
                    # Get predictions for this task
                    if task_idx < top_k_preds.shape[1]:
                        task_preds = top_k_preds[:, task_idx]
                    else:
                        # Default to first prediction
                        task_preds = top_k_preds[:, 0]
                    
                    # Get labels for this task
                    if task_name in labels:
                        task_label = labels[task_name]
                    elif isinstance(labels, torch.Tensor) and len(labels.shape) > 0 and task_idx < labels.shape[0]:
                        task_label = labels[task_idx]
                    else:
                        # Default case - use the first available label
                        task_label = next(iter(labels.values())) if isinstance(labels, dict) else labels
                    
                    # Expand labels to match predictions
                    if isinstance(task_label, torch.Tensor) and len(task_label.shape) == 1 and task_label.shape[0] == 1:
                        expanded_label = task_label.expand(top_k_indices.shape[0])
                    else:
                        expanded_label = task_label.expand(top_k_indices.shape[0]) if isinstance(task_label, torch.Tensor) else torch.tensor([task_label] * top_k_indices.shape[0], device=device)
                    
                    # Get the loss function from the dictionary
                    loss_fn_callable = loss_fn[0] if isinstance(loss_fn, dict) else loss_fn
                    
                    # Calculate loss based on task type
                    if task_type == 'multiclass' and expanded_label.dtype != torch.long:
                        expanded_label = expanded_label.long()  # Ensure labels are long type for multiclass
                        
                    task_loss = loss_fn_callable(task_preds, expanded_label)
                    
                    # If the loss function returns one value per sample (reduction='none'),
                    # we need to reduce it to a scalar before adding to local_loss
                    if task_loss.dim() > 0:
                        task_loss = task_loss.mean()
                        
                    local_loss += task_loss
        else:
            # Legacy behavior
            if len(set(class_loss_map.values())) == 1:
                # If all classes use the same loss function
                loss_fn = next(iter(class_loss_map.values()))
                # If the loss function is a dictionary, get the actual callable
                if isinstance(loss_fn, dict):
                    loss_fn = loss_fn[0]
                
                # For binary classification with a single label value
                is_binary = predictions.shape[1] == 1 or (len(class_loss_map) == 1 and 0 in class_loss_map)
                if is_binary and len(labels.shape) == 1 and labels.shape[0] == 1:
                    # Expand single label to match top_k_preds shape for binary classification
                    expanded_labels = labels.expand(top_k_indices.shape[0])
                    if top_k_preds.shape[1] > 1:
                        # If prediction has multiple outputs but we're treating it as binary
                        local_loss = loss_fn(top_k_preds[:, 0], expanded_labels)
                    else:
                        local_loss = loss_fn(top_k_preds.squeeze(1), expanded_labels)
                else:
                    # Expand labels to match top_k_preds shape for multiclass
                    expanded_labels = labels.unsqueeze(0).expand(top_k_indices.shape[0], -1)
                    if isinstance(loss_fn, nn.CrossEntropyLoss) and expanded_labels.dtype != torch.long:
                        expanded_labels = expanded_labels.long()  # Convert to long for CrossEntropyLoss
                    local_loss = loss_fn(top_k_preds, expanded_labels)
            else:
                for class_idx, loss_fn in class_loss_map.items():
                    class_pred = top_k_preds[:, class_idx]
                    
                    # If the loss function is a dictionary, get the actual callable
                    if isinstance(loss_fn, dict):
                        loss_fn = loss_fn[0]
                    
                    # Handle different label shapes
                    if len(labels.shape) == 1 and labels.shape[0] == 1 and len(class_loss_map) > 1:
                        # Single label value but multiple classes - use the same label for all
                        class_label = labels.expand(top_k_indices.shape[0])
                    elif len(labels.shape) >= 1 and class_idx < labels.shape[0]:
                        # Multiple labels, one per class
                        class_label = labels[class_idx].expand(top_k_indices.shape[0])
                    else:
                        # Default case - use the first label
                        class_label = labels[0].expand(top_k_indices.shape[0])
                    
                    # Check if we need to convert labels to long for CrossEntropyLoss
                    if isinstance(loss_fn, nn.CrossEntropyLoss) and class_label.dtype != torch.long:
                        class_label = class_label.long()
                        
                    # Calculate task loss
                    task_loss = loss_fn(class_pred, class_label)
                    
                    # If the loss function returns one value per sample (reduction='none'),
                    # we need to reduce it to a scalar before adding to local_loss
                    if task_loss.dim() > 0:
                        task_loss = task_loss.mean()
                        
                    local_loss += task_loss
        
        local_loss = local_loss / k * regional_coeff
        
        return global_loss + local_loss
    else:
        # No instance predictions
        return global_loss


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


def split_riskformer_data(svs_paths_data_dict, label_var="odx85", positive_label="H", test_split_ratio=0.2):
    """
    Split data into train and test sets. Balances test set to have
    equal number of positive and negative samples based on the data variable provided.
    
    Args:
        svs_paths_data_dict (dict): Dictionary of SVS file paths and corresponding dictionary of data.
        label_var (str): The key in the data dictionary that contains the label.
        positive_label (str): The value that indicates a positive sample.
        test_split_ratio (float): Ratio of data to use for testing.
    
    Returns:
        tuple: Two dictionaries, one for training data and one for testing data.
    """
    svs_paths = np.array(list(svs_paths_data_dict.keys()))
    labels = np.array([svs_paths_data_dict[svs_path][label_var] for svs_path in svs_paths])
    num_pos = int(len(svs_paths) * (test_split_ratio) / 2)
    if num_pos == 0:
        logger.error("Test split ratio too low, not enough samples.")
        raise ValueError("Test split ratio too low, not enough samples.")

    pos_samples = svs_paths[labels == positive_label]
    neg_samples = svs_paths[labels != positive_label]
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        logger.error("No positive or negative samples found.")
        raise ValueError("No positive or negative samples found.")

    logger.debug(f"Dataset contains {len(svs_paths)} samples, {len(pos_samples)} positive and {len(neg_samples)} negative samples.")
    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)

    test_data = {
        **{svs_path: svs_paths_data_dict[svs_path] for svs_path in pos_samples[:num_pos]},
        **{svs_path: svs_paths_data_dict[svs_path] for svs_path in neg_samples[:num_pos]}
    }
    logger.debug(f"Created Test Dataset with {len(test_data)} samples, {num_pos} positive and {num_pos} negative samples.")
    train_data = {
        **{svs_path: svs_paths_data_dict[svs_path] for svs_path in pos_samples[num_pos:]},
        **{svs_path: svs_paths_data_dict[svs_path] for svs_path in neg_samples[num_pos:]}
    }
    logger.debug(f"Created Train Dataset with {len(train_data)} samples, {len(pos_samples) - num_pos} positive and {len(neg_samples) - num_pos} negative samples.")
    return train_data, test_data


