'''
run_sagemaker_preprocess.py

Run a preprocessing job on SageMaker.
Author: landeros10
Created: 2025-02-05
'''
import os
import logging
import argparse
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from src.logger_config import logger_setup
from src.aws_utils import initialize_s3_client
from src.utils import load_slide_paths
logger = logging.getLogger(__name__)


def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025", help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--input_dir", type=str, default="raw", help="Path to input data")
    parser.add_argument("--output_dir", type=str, default="processed", help="Path to output data")
    parser.add_argument("--svs_paths_file", type=str, default="/data/resources/riskformer_slides.json", help="Path to slides list")
    parser.add_argument("--tiling_config", type=str, default="./tiling_config.yaml", help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, default="./foreground_config.yaml", help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, default="./foreground_cleanup.yaml", help="Foreground cleanup YAML file")

    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")
    parser.add_argument("--models_file", type=str, default="/config/preprocessing_models.json", help="Path to models list")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Set up AWS
    os.environ["AWS_REGION"] = args.region
    s3_client, session = initialize_s3_client(
        args.profile,
        region_name=args.region,
        return_session=True)
    logger.debug(f"Using AWS profile: {args.profile}")
    logger.debug(f"Using AWS region: {args.region}")
    logger.debug("Initialized S3 client.")

    sagemaker_session = sagemaker.Session(boto_session=session)
    logger.debug(f"Using SageMaker session: {sagemaker_session}")
    try:
        role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
        logger.debug(f"Using IAM role from Sagemaker: {role}")
    except Exception as e:
        logger.warning(f"Failed to get IAM role from Sagemaker: {e}")
        role = None
    
    processor = ScriptProcessor(
        role=role,
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-cpu-py38",        command=["python3"],
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
    )
    logger.debug(f"Loaded SageMaker processor with config: {vars(processor)}")

    svs_files = load_slide_paths(args.svs_paths_file)
    for svs_file in svs_files:
        processor.run(
            code="sagemaker_scripts/preprocess.py",
            inputs=[
                ProcessingInput(
                    source=f"s3://{args.bucket}/{args.input_dir}/{svs_file}",
                    destination="/opt/ml/processing/input/",
                ),
                ProcessingInput(source=args.tiling_config, destination=f"/opt/ml/processing/input/"),
                ProcessingInput(source=args.foreground_config, destination="/opt/ml/processing/input/"),
                ProcessingInput(source=args.foreground_cleanup_config, destination="/opt/ml/processing/input/"),
                ProcessingInput(
                    source = f"s3://{args.bucket}/{args.model_dir}/",
                    destination = "/opt/ml/processing/model/"
                )
            ],
            arguments=[
                "--input_file", f"/opt/ml/processing/input/{svs_file}",
                "--output_dir", "/opt/ml/processing/output",
                "--tiling_config", f"/opt/ml/processing/input/{os.path.basename(args.tiling_config)}",
                "--foreground_config", f"/opt/ml/processing/input/{os.path.basename(args.foreground_config)}",
                "--foreground_cleanup_config", f"/opt/ml/processing/input/{os.path.basename(args.foreground_cleanup_config)}",
                "--model_dir", "/opt/ml/processing/model/",
                "--model_type", model_type,
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{args.bucket}/{args.output_dir}/",
                )
            ],
        )
    logger.debug(f"Processing job completed and saved to s3://{args.bucket}/{args.output_dir}/")

if __name__ == "__main__":
    main()