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
            
        reconstructed_features.append(single_feature)
    
    return reconstructed_features
        
