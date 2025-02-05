import os
import logging
import argparse
import boto3

from src.logger_config import logger_setup
from src.data.data_preprocess import SLIDES_PRS
from src.utils import set_seed

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

    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        files_deleted = 0
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    try:
                        s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
                        files_deleted += 1
                        logger.debug(f"Deleted: {obj['Key']}")
                    except Exception as e:
                        logger.error(f"Failed to delete {obj['Key']}: {e}")

        if files_deleted == 0:
            logger.info(f"No files deleted under s3://{bucket_name}/{bucket_prefix}")
        else:
            logger.info(f"Deleted {files_deleted} files under s3://{bucket_name}/{bucket_prefix}")

    except Exception as e:
        logger.error(f"Failed to delete files under s3://{bucket_name}/{bucket_prefix}: {e}")
        raise e


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
            logger.info(f"No files found in s3://{bucket_name}/{bucket_prefix}")
        return file_list

    except Exception as e:
        logger.error(f"Failed to list files in s3://{bucket_name}/{bucket_prefix}: {e}")
        raise e


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
                logger.info(f"Uploaded: {file_name} to s3://{bucket_name}/{prefix}/")
            except Exception as e:
                logger.error(f"Failed to upload {file_name}: {e}")
        else:
            logger.warning(f"Skipping: {file_path} (File not found or invalid)")


def split_riskformer_data(svs_files, split_ratio=0.8):
    """
    Split data into train and test sets. Balances test set to have
    equal number of positive and negative samples.
    
    Args:
        svs_files (list): List of local file paths to split.
        trasplit_ratioin_ratio (float): Ratio of data to use for training.
    Returns:
        tuple: Train and test file lists.
    """
    pass


def prepare_riskformer_data(s3_client, bucket_name, svs_files, split_ratio=0.8):
    """
    Prepare S3 bucket for training.
    
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): name of the S3 bucket.
        save_dir (str): directory to save files to.
        svs_files (list): List of local file paths to upload.
    """
    
    # Wipe existing files in the bucket if any
    files = list_bucket_files(s3_client, bucket_name)
    if files:
        logger.info(f"Found {len(files)} files in s3://{bucket_name}/")
        logger.info(f"First 5 files: {files[:5]}")
        wipe_bucket_dir(s3_client, bucket_name)

    # Create train/test directories
    for dir_name in ["train", "test"]:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=f"{dir_name}/")
            logger.info(f"Created s3://{bucket_name}/{dir_name}/")
        except Exception as e:
            logger.error(f"Failed to create s3://{bucket_name}/{dir_name}/: {e}")

    # Split train/test data
    train_files, test_files = split_riskformer_data(svs_files, split_ratio=0.8)
    # TODO
    # finish implementing the split_riskformer_data function



def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    set_seed(args.seed)    

    bucket_name = "tcga-riskformer-data-2025"
    try:
        session = boto3.Session(profile_name=args.profile)
        logger.debug("Created boto3 session")
    except Exception as e:
        logger.error(f"Failed to create boto3 session: {e}")
        
    try:
        s3_client = session.client("s3")
        logger.debug("Created S3 client")
        logger.debug(f"Available buckets: {s3_client.list_buckets().get('Buckets')}")
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return

    svs_paths = [f.replace("./resources", "/data/resources") for f in SLIDES_PRS]
    prepare_riskformer_data(s3_client, bucket_name, svs_paths)


if __name__ == "__main__":
    main()