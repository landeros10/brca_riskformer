import logging
import argparse

from src.logger_config import logger_setup
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--input_dir", type=str, default="testing/raw", help="Path to input data")
    parser.add_argument("--output_dir", type=str, default="testing/processed", help="Path to output data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--sagemaker_image", type=str, default="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-cpu-py38", help="Sagemaker image")
    parser.add_argument("--sagemaker_instance", type=str, default="ml.m5.xlarge", help="Sagemaker instance type")
    parser.add_argument("--sagemaker_instance_count", type=int, default=1, help="Sagemaker instance count")

    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # TODO - make sure preprocessing functions are prepped to work with s3 buckets

    # TODO - load feature extraction model

    # TODO - go through slides and convert patches to features using all_coords
    pass


if __name__ == "__main__":
    main()
