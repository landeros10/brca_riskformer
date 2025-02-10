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
from src.utils import set_seed, initialize_s3_client, upload_large_files_to_bucket, wipe_bucket
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # Moves up to project root
default_uni_model_dir = os.path.join(project_root, "models", "uni2-h")
defautl_models_info_file = os.path.join(project_root, "src", "config", "preprocessing_models.json")


def download_uni_model(uni_file_dir, filename="pytorch_model.bin"):
    from huggingface_hub import login, hf_hub_download
    try:
        login()
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        return
    
    os.makedirs(uni_file_dir, exist_ok=True)
    logging.debug(f"Model download dir was created or exists: {uni_file_dir}")

    try:
        hf_hub_download("MahmoodLab/UNI2-h", filename=filename, local_dir=uni_file_dir, force_download=False)
    except Exception as e:
        logger.error(f"Failed to download UNI2-h model: {e}")
        return
    
    logging.info("Downloaded UNI2-h model from Hugging Face Hub.")


def upload_models(s3_client, models_info, bucket_name):
    for model_type, model_info in models_info.items():
        model_path = model_info.get("model_path", None)
        model_arch = model_info.get("arch", model_type)
        prefix = f"{model_type}/{model_arch}"

        # Make the prefix folder in bucket_name if doesn't exist
        dummy_key = f"{prefix}/.keep"
        s3_client.put_object(Bucket=bucket_name, Key=dummy_key, Body="")
        logger.info(f"Created prefix {model_type}/{model_arch}/ in bucket {bucket_name}")


        if not model_path:
            logger.warning(f"Model info for {model_type} does not include model path. Will be downloaded with timm library.")
            logger.info(f"Only uploading config files for {model_type}")
        
        else:
            abs_model_path = os.path.join(project_root, model_path)
            if not os.path.exists(abs_model_path) or not os.path.isfile(abs_model_path):
                logger.warning(f"Provided model path is invalid. Skipping file upload.")
                logger.info(f"Only uploading config files for {abs_model_path}")
            else:
                logger.info(f"Uploading {abs_model_path} to {bucket_name}/{prefix}/{os.path.basename(model_path)}")
                upload_large_files_to_bucket(s3_client, bucket_name, [abs_model_path], file_names=[os.path.basename(model_path)], prefix=prefix)

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-preprocessing-models", help="S3 bucket name")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--wipe_bucket", action="store_true", help="Wipe bucket before uploading data")
    
    parser.add_argument("--upload_models", action="store_true", help="Upload models to S3 bucket")
    parser.add_argument("--models_info_file", type=str, default=defautl_models_info_file, help="Path to models list")
    
    parser.add_argument("--download_uni_model", action="store_true", help="Download uni2-h model from hugginface")
    parser.add_argument("--uni_file_dir", type=str, default=default_uni_model_dir, help="Path to save uni2-h model from hugginface")
    args = parser.parse_args()
    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    set_seed(args.seed)

    # Initialize S3 client
    logger.info("Initializing S3 client...")
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
        logger.info(f"Wiping bucket {args.bucket}...")
        success = wipe_bucket(s3_client, args.bucket)
        if not success:
            logger.error("Failed to clear bucket.")
            return
        
    #Download uni model
    if args.download_uni_model:
        logger.info("Downloading uni2-h model from Hugging Face Hub...")
        download_uni_model(args.uni_file_dir)

    # Upload models to S3 bucket
    if args.upload_models:
        logger.info("Uploading models to S3 bucket...")
        models_info = load_models_info(args.models_info_file)
        if not models_info:
            logger.error("No models info found.")
            return
    
        upload_models(s3_client, models_info, args.bucket)


if __name__ == "__main__":
    main()