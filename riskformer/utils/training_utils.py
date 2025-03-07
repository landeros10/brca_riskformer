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

logger = logging.getLogger(__name__)

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

    n_features = int(patch_info[:, 0].max()) + 1
    reconstructed_features = []
    patch_id = 0
    for feature_id in range(n_features):
        feature_info = patch_info[patch_info[:, 0] == feature_id, :]
        n_regions = int(feature_info[:, 1].max()) + 1

        feature_size_rows = max(feature_info[:, 4])
        feature_size_cols = max(feature_info[:, 5])
        single_feature = torch.zeros((feature_size_rows, feature_size_cols, feature_dim))

        for region_id in range(n_regions):
            single_region_info = feature_info[feature_info[:, 1] == region_id, :]
            region_bbox = single_region_info[0, 2:6]
            min_row, min_col, max_row, max_col = region_bbox
            
            for patch_bbox in single_region_info[:, 6:]:
                patch_row_start, patch_col_start, patch_row_end, patch_col_end = patch_bbox
                
                # Calculate actual patch dimensions (might be smaller than max_dim)
                patch_height = patch_row_end - patch_row_start
                patch_width = patch_col_end - patch_col_start
                
                # Calculate region coordinates
                region_row_start = patch_row_start + min_row
                region_col_start = patch_col_start + min_col
                region_row_end = region_row_start + patch_height
                region_col_end = region_col_start + patch_width
                
                # Use the correct slice of the patch (up to patch_height and patch_width)
                single_feature[region_row_start:region_row_end, region_col_start:region_col_end, :] = xl_patches[patch_id, :patch_height, :patch_width, :]
                patch_id += 1
        reconstructed_features.append(single_feature)
    return reconstructed_features
        
