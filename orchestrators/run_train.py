'''
run_sagemaker_preprocess.py

Run a preprocessing job on SageMaker.
Author: landeros10
Created: 2025-02-05
'''
import os
import logging
import argparse
import numpy as np

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from riskformer.utils.logger_config import logger_setup
from riskformer.utils.data_utils import initialize_s3_client, load_slide_paths
logger = logging.getLogger(__name__)


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


def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")

if __name__ == "__main__":
    main()