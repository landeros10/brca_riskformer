"""
test_sagemaker.py
"""
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.s3 import S3Uploader
import boto3
import os
import argparse
import logging

from src.logger_config import logger_setup
from src.data.data_utils import initialize_s3_client 

logger_setup()
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--input_dir", type=str, default="testing_raw", help="Path to input data")
    parser.add_argument("--output_dir", type=str, default="testing_processed", help="Path to output data")
    parser.add_argument("--filename", type=str, default="input.txt", help="Name of file to process")
    args = parser.parse_args()

    s3_client, session = initialize_s3_client(args.profile, return_session=True)
    s3_client.put_object(Bucket=args.bucket, Key=f"{args.input_dir}/{args.filename}", Body="Test input data")
    logger.info(f"Created s3://{args.bucket}/{args.input_dir}/{args.filename}")

    sagemaker_session = sagemaker.Session(boto_session=session)
    logger.debug(f"Using sagemaker session: {sagemaker_session}")

    role = sagemaker.get_execution_role()
    logger.debug(f"Using role: {role}")

if __name__ == "__main__":
    main()