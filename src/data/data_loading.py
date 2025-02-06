import os
import io
import logging
import argparse
import boto3
import pandas as pd
import numpy as np

from src.logger_config import logger_setup
from src.data.data_utils import load_slide_paths
from src.utils import set_seed, collect_patients_svs_files

logger_setup()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def wipe_bucket_dir(s3_client, bucket_name, bucket_prefix=""):
    """
    Deletes all files under a specific prefix in an S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): Prefix (directory) to delete.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    files_deleted = 0
    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        for page in pages:
            if "Contents" in page:
                try:
                    objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
                    files_deleted += len(objects)
                    logger.debug(f"Deleted {len(objects)} files")
                except Exception as e:
                    logger.error(f"Failed to delete files in page {page}: {e}")
                    return False
        logger.debug(f"Deleted {files_deleted} files under s3://{bucket_name}/{bucket_prefix}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete files under s3://{bucket_name}/{bucket_prefix}: {e}")
        return False


def list_bucket_files(s3_client, bucket_name, bucket_prefix=""):
    """
    Lists all files under a specific prefix in an S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): Prefix (directory) to list.

    Returns:
        list: List of file paths in S3.
    """
    
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        file_list = [obj["Key"] for page in pages if "Contents" in page for obj in page["Contents"]]
        if not file_list:
            logger.debug(f"No files found in s3://{bucket_name}/{bucket_prefix}")
        return file_list

    except Exception as e:
        logger.error(f"Failed to list files in s3://{bucket_name}/{bucket_prefix}: {e}")
        return None


def process_svs_foregrounds(svs_files):
    pass


def upload_svs_files_to_bucket(s3_client, bucket_name, file_list, prefix="raw", ext=".svs"):
    """
    Prepare S3 bucket for training.
    
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): name of the S3 bucket.
        preifx (str): prefix to upload files to.
        file_list (list): List of local file paths to upload.
    """
    for file_path in file_list:
        if os.path.exists(file_path) and os.path.isfile(file_path) and file_path.endswith(ext):
            file_name = os.path.basename(file_path)
            try:
                s3_client.upload_file(file_path, bucket_name, f"{prefix}/{file_name}")
                logger.debug(f"Uploaded: {file_name} to s3://{bucket_name}/{prefix}/")
            except Exception as e:
                logger.error(f"Failed to upload {file_name}: {e}")
        else:
            logger.warning(f"Skipping: {file_path} (File not found or invalid)")


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


def prepare_riskformer_data(s3_client, bucket_name, svs_paths_file, test_split_ratio=0.2):
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
        train_data, test_data = split_riskformer_data(svs_paths_data_dict, test_split_ratio=test_split_ratio)
        logger.info(f"Split data into {len(train_data)} training samples and {len(test_data)} testing samples.")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        return False


    # Upload files to S3 to train/test directories
    for dir_name, data in [("train", train_data), ("test", test_data)]:
        if not data:
            logger.warning(f"Skipping upload: No files in {dir_name}.")
            continue
        try:
            s3_client.put_object(Bucket=bucket_name, Key=f"{dir_name}/")
            logger.info(f"Created s3://{bucket_name}/{dir_name}/")
            upload_svs_files_to_bucket(s3_client, bucket_name, list(data.keys()), prefix=dir_name)
            logger.info(f"Uploaded {len(data)} files to s3://{bucket_name}/{dir_name}/")
        except Exception as e:
            logger.error(f"Failed to create s3://{bucket_name}/{dir_name}/: {e}")
            return False
    return True
    

def initialize_s3_client(profile_name):
    """
    Initialize boto3 session and S3 client.
    
    Args:
        profile_name (str): AWS profile name.
    
    Returns:
        boto3.client: S3 boto3 client.
    """
    try:
        session = boto3.Session(profile_name=profile_name)
        logger.debug("Created boto3 session")
    except Exception as e:
        logger.error(f"Failed to create boto3 session: {e}")
        return
    
    try:
        s3_client = session.client("s3")
        logger.debug("Created S3 client")
        logger.debug(f"Available buckets: {s3_client.list_buckets().get('Buckets')}")
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return
    return s3_client


def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--wipe-bucket", action="store_true", help="Wipe bucket before uploading data")
    parser.add_argument("--svs_paths_file", type=str, default="/data/resources/riskformer_slides.json", help="Path to slides list")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

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
    success = prepare_riskformer_data(s3_client, bucket_name, args.svs_paths_file)
    if not success:
        logger.error("Failed to upload training and test data.")
        return


if __name__ == "__main__":
    main()