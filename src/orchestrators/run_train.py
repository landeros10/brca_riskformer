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

from src.logger_config import logger_setup
from src.data.data_utils import initialize_s3_client, load_slide_paths
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
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--input_dir", type=str, default="raw", help="Path to input data")
    parser.add_argument("--output_dir", type=str, default="processed", help="Path to output data")
    parser.add_argument("--svs_paths_file", type=str, default="/data/resources/riskformer_slides.json", help="Path to slides list")

    parser.add_argument("--test_split_ratio", type=float, default=0.2, help="Train/test split ratio")
    parser.add_argument("--label_var", type=str, default="odx85", help="Variable name to use as label")
    parser.add_argument("--positive_label", type=str, default="H", help="Positive label value")
    parser.add_argument("--model_type", type=str, default="standard", help="Model type")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Set up AWS
    os.environ["AWS_REGION"] = args.region
    s3_client, session = initialize_s3_client(
        args.profile,
        region_name=args.region,
        return_session=True)
    logger.debug(f"Using AWS profile: {args.profile}")
    logger.debug(f"Using AWS region: {args.region}")
    logger.debug("Initialized S3 client.")

    sagemaker_session = sagemaker.Session(boto_session=session)
    logger.debug(f"Using SageMaker session: {sagemaker_session}")
    try:
        role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
        logger.debug(f"Using IAM role from Sagemaker: {role}")
    except Exception as e:
        logger.warning(f"Failed to get IAM role from Sagemaker: {e}")
        role = None
    
    processor = ScriptProcessor(
        role=role,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-cpu-py38",        command=["python3"],
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
    )
    logger.debug(f"Loaded SageMaker processor with config: {vars(processor)}")
    
    # TODO
    # create train test split json file
    # Set up processor

    logger.debug(f"Processing job completed and saved to s3://{args.bucket}/{args.output_dir}/")

if __name__ == "__main__":
    main()