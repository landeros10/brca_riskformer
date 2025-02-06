import sagemaker
from sagemaker.processing import ScriptProcessor
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
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--input-path", type=str, default="raw", help="Path to input data")
    parser.add_argument("--output-path", type=str, default="processed", help="Path to output data")
    args = parser.parse_args()

    s3_client = initialize_s3_client()



if __name__ == "__main__":
    main()