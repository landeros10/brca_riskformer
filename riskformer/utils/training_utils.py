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
        ):
    """
    Calculate the slide-level loss for a batch of predictions and labels. Assumes multilabel 
    classification with per-class loss function given by class_loss_map.
    
    Args:
        predictions (torch.Tensor): The predictions from the model, shape (batch_size, num_classes)
        labels (torch.Tensor): The labels for the batch, shape (num_classes)
        class_loss_map (dict): A dictionary mapping class index to loss function
        regional_coeff (float): Coefficient for weighting local vs global loss
        
    Returns:
        torch.Tensor: The slide-level loss.
    """
    # Global Loss - vectorized approach
    global_pred = predictions[0]  # (num_classes,)
    
    # Calculate all class losses at once if using the same loss function
    if len(set(class_loss_map.values())) == 1:
        # If all classes use the same loss function
        loss_fn = next(iter(class_loss_map.values()))
        global_loss = loss_fn(global_pred, labels)
    else:
        # If different classes use different loss functions
        global_loss = torch.tensor(0.0, device=predictions.device)
        for class_idx, loss_fn in class_loss_map.items():
            class_pred = global_pred[class_idx].unsqueeze(0)
            class_label = labels[class_idx].unsqueeze(0)
            global_loss += loss_fn(class_pred, class_label)
    
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
    top_k_values, top_k_indices = torch.topk(instance_preds[:, 0], k=k)
    
    # More efficient gathering of top-k predictions
    top_k_preds = instance_preds[top_k_indices]
    
    # Calculate instance loss for each in top k
    if len(set(class_loss_map.values())) == 1:
        # If all classes use the same loss function
        loss_fn = next(iter(class_loss_map.values()))
        # Expand labels to match top_k_preds shape
        expanded_labels = labels.unsqueeze(0).expand(k, -1)
        local_loss = loss_fn(top_k_preds, expanded_labels)
    else:
        local_loss = torch.tensor(0.0, device=predictions.device)
        for class_idx, loss_fn in class_loss_map.items():
            class_pred = top_k_preds[:, class_idx]
            class_label = labels[class_idx].expand(k)
            local_loss += loss_fn(class_pred, class_label)
    
    local_loss = local_loss / k * regional_coeff
    
    return global_loss + local_loss
    

    