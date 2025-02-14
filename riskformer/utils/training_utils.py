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
