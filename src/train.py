'''
Created June 2023
author: landeros10

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital

Massachusetts Institute of Technology

Main Training Pipeline
'''
from __future__ import division, print_function

# import os
# import datetime
# import argparse

# from os.path import join
# from model import RS_Predictor_ViT
# from data_prep import (HOME_DIR, RESOURCE_DIR, FOREGROUND_DIR, SLIDES_PRS,
#                         SLIDES_PRS_DATA)
# from util import build_dataset
# from datetime import datetime

import argparse
import logging

from src.logger_config import logger_setup
from src.util import log_training_params

logger_setup()
logger = logging.getLogger(__name__)

SIZE = 256
HOME_DIR = './'

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Riskformer Taining Configurations")

    # Model Parameters
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of head output classes")
    parser.add_argument("--num_embed", type=int, default=1024,
                        help="Number of whole slide patches to evaluate at once.")
    
    # Training Parameters
    parser.add_argument("--sample_size", type=int, default=-1,
                            help="Size of the training dataset. Set to -1 to use full dataset")
    parser.add_argument("--train_steps", type=int, default=100,
                            help="Total number of training steps.")
    parser.add_argument("--eval_every", type=int, default=8,
                            help="Number of training steps between evaluations.")
    parser.add_argument("--early_stop", type=int, default=25,
                            help="Number of epochs to wait before early stopping.")
    
    # Optimization Parameters
    parser.add_argument("--optimizer", type=str, default="adam", 
                            choices=["momentum", "adam", "nadam", "lars", "lamb"],
                            help="Optimizer to use for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                            help="Initial learning rate.")
    parser.add_argument("--learning_rate_scaling", type=str, default="linear",
                            choices=["linear", "sqrt"],
                            help="How to scale the learning rate as a function of batch size.")
    parser.add_argument("--learning_rate_warmup_epochs", type=int, default=25,
                            help="Number of epochs for warmup.")
    parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size for training.")
    parser.add_argument("--patch_coeff", type=float, default=1.0,
                            help="Weight to apply to patch-level risk predictions.")
    
    # Data Augmentation Parameters
    parser.add_argument("--l2_coeff", type=float, default=1e-6,
                            help="L2 regularization coefficient.")
    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                            help="StochasticDepth dropout rate.")
    
    # Set up debugmode
    parser.add_argument("--debug", action="store_true",
                            help="Set to run in debug mode.")

    config_params = vars(parser.parse_args())
    log_training_params(logger, config_params)
    
    return config_params


def main():
    logger.info("Starting Training Pipeline...")
    logger.info("=" * 50)

    # Load Configurations
    config = parse_args()

    # Load Data

    # Load Model

if __name__ == "__main__":
    main()
