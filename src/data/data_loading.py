'''
data_loading.py
'''
import os
import logging
import argparse
import time
import numpy as np

from src.logger_config import logger_setup
from src.data.data_utils import (load_slide_paths,
                                 initialize_s3_client,
                                 upload_large_files_to_bucket, wipe_bucket)
from src.utils import set_seed
logger = logging.getLogger(__name__)


def process_svs_foregrounds(svs_files):
    pass


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
        destination_dir="raw",
        reupload=False):
    """
    Prepare S3 bucket for training.
    
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): name of the S3 bucket.
        svs_paths_file (str): JSON file containing SVS file paths.
        split_ratio (float): Proportion of files to use for training (default: 0.8).
    """
    # Load SVS files
    try:
        svs_paths_data_dict = load_slide_paths(svs_paths_file)
        logger.info(f"Loaded {len(svs_paths_data_dict)} SVS files.")
    except Exception as e:
        logger.error(f"Failed to load SVS files: {e}")
        return False

    # Upload files to S3 to train/test directories
    if not svs_paths_data_dict:
        logger.debug(f"Skipping upload: No files retrieved from {svs_paths_file}.")
        return False
    try:
        s3_client.put_object(Bucket=bucket_name, Key=f"{destination_dir}/")
        logger.info(f"Created s3://{bucket_name}/{destination_dir}/")
        upload_svs_files_to_bucket(s3_client, bucket_name, list(svs_paths_data_dict.keys()), prefix=destination_dir, reupload=reupload)
        logger.info(f"Uploaded {len(svs_paths_data_dict)} files to s3://{bucket_name}/{destination_dir}/")
    except Exception as e:
        logger.error(f"Failed to create s3://{bucket_name}/{destination_dir}/: {e}")
        return False
    return True
    

def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--wipe_bucket", action="store_true", help="Wipe bucket before uploading data")
    parser.add_argument("--reupload", action="store_true", help="Reupload files even if they exist")
    parser.add_argument("--svs_paths_file", type=str, default="/resources/riskformer_slides.json", help="Path to slides list")
    parser.add_argument("--destination_dir", type=str, default="raw", help="Destination directory in S3 bucket")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    set_seed(args.seed) 

    s3_client = initialize_s3_client(args.profile)
    if s3_client is None:
        logger.error("Failed to initialize S3 client.")
        return
    
    # Clear bucket
    if args.wipe_bucket:
        success = wipe_bucket(s3_client, args.bucket)
        if not success:
            logger.error("Failed to clear bucket.")
            return

    # Upload SVS files to S3 bucket
    success = prepare_riskformer_data(
        s3_client,
        args.bucket,
        svs_paths_file=args.svs_paths_file,
        reupload=args.reupload,
    )
    if not success:
        logger.error("Failed to upload training and test data.")
        return


if __name__ == "__main__":
    main()