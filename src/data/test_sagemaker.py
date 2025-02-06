"""
test_sagemaker.py
"""
# import sagemaker
# from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
# from sagemaker.s3 import S3Uploader
import boto3
import os
import argparse
import logging

from src.logger_config import logger_setup
from src.data.data_utils import initialize_s3_client, list_bucket_files
logger = logging.getLogger(__name__)

def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--input_dir", type=str, default="testing/raw", help="Path to input data")
    parser.add_argument("--output_dir", type=str, default="testing/processed", help="Path to output data")
    parser.add_argument("--filename", type=str, default="input.txt", help="Name of file to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    s3_client, session = initialize_s3_client(args.profile, return_session=True)
    logger.debug("Initialized S3 client.")
    logger.debug("Creating test input data...")
    s3_client.put_object(Bucket=args.bucket, Key=f"{args.input_dir}/{args.filename}", Body="Test input data")
    logger.debug(f"Created s3://{args.bucket}/{args.input_dir}/{args.filename}")
    testing_files = list_bucket_files(s3_client, args.bucket, prefix=args.input_dir)
    logger.debug(f"Found {len(testing_files)} files in s3://{args.bucket}/{args.input_dir}/")
    logger.debug(f"First file: {testing_files[0]}")
    return

    sagemaker_session = sagemaker.Session(boto_session=session)
    logger.debug(f"Using sagemaker session: {sagemaker_session}")

    role = sagemaker.get_execution_role()
    logger.debug(f"Using role: {role}")

if __name__ == "__main__":
    main()