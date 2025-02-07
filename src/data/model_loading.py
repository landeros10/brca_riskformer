'''
data_loading.py
'''
import os
import logging
import argparse
import time
import numpy as np

from src.logger_config import logger_setup
from src.data.data_utils import (initialize_s3_client,
                                 upload_large_files_to_bucket, wipe_bucket)
from src.utils import set_seed
logger = logging.getLogger(__name__)

def upload_models(s3_client, bucket_name, model_key):
    model_pths = ["test.pth", "test2.pth"]
    model_keys = ["uni", "dinov2"]
    for model_pth, model_key in zip(model_pths, model_keys):
        upload_large_files_to_bucket(s3_client, bucket_name, [model_pth], prefix=model_key)
        pass
    pass


def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--wipe_bucket", action="store_true", help="Wipe bucket before uploading data")
    parser.add_argument("--reupload", action="store_true", help="Reupload files even if they exist")

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


if __name__ == "__main__":
    main()