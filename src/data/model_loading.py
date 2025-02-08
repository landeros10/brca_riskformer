'''
data_loading.py
'''
import os
import json
import logging
import argparse
import time
import numpy as np

from src.logger_config import logger_setup
from src.data.data_utils import (initialize_s3_client,
                                 upload_large_files_to_bucket, wipe_bucket)
from src.utils import set_seed
logger = logging.getLogger(__name__)

def upload_models(s3_client, models_info, bucket_name):
    for model_type, model_info in models_info.items():
        model_path = model_info.get("model_path")
        model_arch = model_info.get("arch")
        if not model_path or not model_arch:
            logger.error(f"Model info for {model_type} is missing model_path or arch. Skipping upload.")
            continue
            
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            logger.error(f"Provided model path is invalid. Skipping upload.")
            continue

        prefix = f"{model_type}/{model_arch}"
        upload_large_files_to_bucket(s3_client, bucket_name, [model_path], file_names=["model.pth"], prefix=prefix)

        for config_name, config in model_info.items():
            if config_name in ["model_path", "arch"]:
                continue

            if not isinstance(config, dict):
                logger.warning(f"Skipping {config_name} as it is not a dictionary.")
                continue

            file_name = f"{config_name}.json"
            temp_file_path = os.path.join("/tmp", file_name)
            with open(temp_file_path, "w") as f:
                json.dump(config, f)
            try:
                s3_client.upload_file(temp_file_path, bucket_name, f"{prefix}/{file_name}")
                logger.info(f"Uploaded {file_name} to {bucket_name}/{prefix}/{file_name}")
            except Exception as e:
                logger.error(f"Failed s3 client upload {file_name} to {bucket_name}/{prefix}/{file_name}: {e}")


def load_models_info(models_json_file):
    """
    Load model configurations from JSON file
    
    Args:
        models_json_file (str): Path to the JSON file containing model information.

    Returns:
        dict: Dictionary mapping model names to their configurations.
    """
    try:
        with open(models_json_file, "r") as f:
            models_info = json.load(f)
        return models_info
    except Exception as e:
        logger.error(f"Failed to load models info from {models_json_file}: {e}")
        return {}

def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--wipe_bucket", action="store_true", help="Wipe bucket before uploading data")
    parser.add_argument("--reupload", action="store_true", help="Reupload files even if they exist")

    parser.add_argument("--models_file", type=str, default="/data/resources/preprocessing_models.json", help="Path to models list")

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

    try:
        s3_client.head_bucket(Bucket=args.bucket)
        logger.info(f"Bucket {args.bucket} exists.")
    except s3_client.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logger.info(f"Bucket {args.bucket} does not exist. Creating bucket.")
            s3_client.create_bucket(Bucket=args.bucket)
        else:
            logger.error(f"Unexpected Error accessing s3 bucket {args.bucket}: {e}")
            return

    # Clear bucket
    if args.wipe_bucket:
        success = wipe_bucket(s3_client, args.bucket)
        if not success:
            logger.error("Failed to clear bucket.")
            return

    models_info = load_models_info(args.models_file)
    if not models_info:
        logger.error("No models info found.")
        return
    
    upload_models(s3_client, models_info, args.bucket)


if __name__ == "__main__":
    main()