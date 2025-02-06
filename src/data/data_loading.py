'''
data_loading.py
'''
import os
import logging
import argparse
import time
import numpy as np

from src.logger_config import logger_setup
from src.data.data_utils import (load_slide_paths, list_bucket_files, wipe_bucket_dir, initialize_s3_client,
                                 upload_large_files_to_bucket)
from src.utils import set_seed, collect_patients_svs_files
logger = logging.getLogger(os.path.basename(__file__))


def process_svs_foregrounds(svs_files):
    pass


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


def clear_riskformer_bucket(s3_client, bucket_name):
    """
    Clear all files in the S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    files = list_bucket_files(s3_client, bucket_name)
    if files is None:
        logger.warning(f"Skipping bucket cleanup: Failed to list files in s3://{bucket_name}/")
        return False
    elif len(files) > 0:
        logger.info(f"Found {len(files)} files in s3://{bucket_name}/")
        success = wipe_bucket_dir(s3_client, bucket_name)
        if not success:
            logger.error(f"Cannot proceed. Files not deleted from s3://{bucket_name}/")
            return False
    return True


def upload_svs_files_to_bucket(
        s3_client,
        bucket_name, 
        files_list,
        prefix="raw", 
        ext="",
        reupload=False,
    ):
    """
    Upload SVS files to S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        files_list (list): List of file paths to upload.
        prefix (str): S3 key prefix.
        ext (str): File extension to filter files.
        reupload (bool): Reupload files even if they exist.
    """
    upload_large_files_to_bucket(s3_client, bucket_name, files_list, prefix=prefix, ext=ext, reupload=reupload)

def prepare_riskformer_data(
        s3_client,
        bucket_name,
        svs_paths_file,
        test_split_ratio=0.2,
        split_params={},
        reupload=False):
    """
    Prepare S3 bucket for training.
    
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): name of the S3 bucket.
        svs_paths_file (str): JSON file containing SVS file paths.
        split_ratio (float): Proportion of files to use for training (default: 0.8).
    """
    # Split train/test
    try:
        svs_paths_data_dict = load_slide_paths(svs_paths_file)
        logger.info(f"Loaded {len(svs_paths_data_dict)} SVS files.")
    except Exception as e:
        logger.error(f"Failed to load SVS files: {e}")
        return False
    try:
        train_data, test_data = split_riskformer_data(
            svs_paths_data_dict,
            test_split_ratio=test_split_ratio,
            **split_params
        )
        logger.info(f"Split data into {len(train_data)} training samples and {len(test_data)} testing samples.")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        return False


    # Upload files to S3 to train/test directories
    for dir_name, data in [("train", train_data), ("test", test_data)]:
        if not data:
            logger.debug(f"Skipping upload: No files in {dir_name}.")
            continue
        try:
            s3_client.put_object(Bucket=bucket_name, Key=f"{dir_name}/")
            logger.info(f"Created s3://{bucket_name}/{dir_name}/")
            upload_svs_files_to_bucket(s3_client, bucket_name, list(data.keys()), prefix=dir_name, reupload=reupload)
            logger.info(f"Uploaded {len(data)} files to s3://{bucket_name}/{dir_name}/")
        except Exception as e:
            logger.error(f"Failed to create s3://{bucket_name}/{dir_name}/: {e}")
            return False
    return True
    

def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--wipe_bucket", action="store_true", help="Wipe bucket before uploading data")
    parser.add_argument("--reupload", action="store_true", help="Reupload files even if they exist")
    parser.add_argument("--svs_paths_file", type=str, default="/data/resources/riskformer_slides.json", help="Path to slides list")
    parser.add_argument("--test_split_ratio", type=float, default=0.2, help="Train/test split ratio")
    parser.add_argument("--label_var", type=str, default="odx85", help="Variable name to use as label")
    parser.add_argument("--positive_label", type=str, default="H", help="Positive label value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    logger_setup(debug=args.debug)

    set_seed(args.seed) 

    bucket_name = "tcga-riskformer-data-2025"
    s3_client = initialize_s3_client(args.profile)
    if s3_client is None:
        logger.error("Failed to initialize S3 client.")
        return
    
    # Clear bucket
    if args.wipe_bucket:
        success = clear_riskformer_bucket(s3_client, bucket_name)
        if not success:
            logger.error("Failed to clear bucket.")
            return

    # Upload SVS files to S3 bucket
    success = prepare_riskformer_data(
        s3_client,
        bucket_name,
        svs_paths_file=args.svs_paths_file,
        test_split_ratio=args.test_split_ratio,
        split_params={"label_var": args.label_var, "positive_label": args.positive_label},
        reupload=args.reupload,
    )
    if not success:
        logger.error("Failed to upload training and test data.")
        return


if __name__ == "__main__":
    main()