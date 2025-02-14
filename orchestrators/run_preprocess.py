#!/usr/bin/env python3
"""
run_preprocess.py

Orchestrates a preprocessing job over new files in S3.

Author: landeros10
Created: 2025-12-05
"""
import os
import json
import logging
import argparse
import subprocess

from riskformer.utils.logger_config import logger_setup
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, upload_large_files_to_bucket

logger = logging.getLogger(__name__)


def load_dataset_files(s3_client, args, project_root):
    raw_files = list_bucket_files(s3_client, args.bucket, args.input_dir)
    logger.info(f"Found {len(raw_files)} files in {args.input_dir}...")
    processed_files = list_bucket_files(s3_client, args.bucket, args.output_dir)
    logger.info(f"Found {len(processed_files)} files in {args.output_dir}...")

    logger.info("Loading riskformer dataset metadata...")
    metadata_file = os.path.join(project_root, args.metadata_file)
    riskformer_dataset = json.load(open(metadata_file, "r"))
    logger.debug(f"First 5 keys in riskformer dataset: {list(riskformer_dataset.keys())[:5]}")

    to_process = [file.split("/")[1] for file in raw_files if file.endswith(".svs")]
    logger.debug(f"Now filtered to {len(to_process)} .svs files")

    logger.debug(f"file to keys map: {to_process[0]}: {to_process[0].split('.svs')[0]}")
    to_process = [file for file in to_process if file.split(".svs")[0] in riskformer_dataset.keys()]
    logger.debug(f"Now filtered to {len(to_process)} files in riskformer dataset")

    to_process = [file for file in to_process if file not in processed_files]

    return to_process


def download_s3_model_files(s3_client, args, model_dir):
    logger.debug(f"Downloading model files from s3://{args.model_bucket}/{args.model_key} to {model_dir}")

    model_files = list_bucket_files(s3_client, args.model_bucket, args.model_key)
    logger.debug(f"Found {len(model_files)} model files in {args.model_key}...")
    for file in model_files:
        logger.debug(f"Found file: {file}")
        file_name = os.path.basename(file)
        local_file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(local_file_path):
            logger.info(f"Downloading s3://{args.model_bucket}/{args.model_key}/{file_name} to {local_file_path}")
            s3_client.download_file(args.model_bucket, file, local_file_path)
        else:
            logger.info(f"File {local_file_path} already exists, skipping download.")
    return model_files


def upload_preprocessing_results(s3_client, args, local_out_dir):
    """
    Uploads the preprocessing results to S3.
    """
    logger.debug(f"Uploading results from {local_out_dir} to s3://{args.bucket}/{args.output_dir}")
    for filename in os.listdir(local_out_dir):
        local_file_path = os.path.join(local_out_dir, filename)
        if os.path.isfile(local_file_path):
            try:
                upload_large_files_to_bucket(
                    s3_client,
                    bucket_name=args.bucket,
                    files_list=[local_file_path],
                    prefix=args.output_dir,
                    reupload=False,
                )
            except Exception as e:
                logger.error(f"Error uploading {local_file_path} to S3: {e}")
                continue
        else:
            logger.warning(f"{local_file_path} is not a file, skipping upload.")
            


def arg_parse():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Data loading / Preprocessing Orchestrator")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess",
                        help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025",
                        help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--input_dir", type=str, default="raw", help="Path (prefix) to input data in S3")
    parser.add_argument("--output_dir", type=str, default="preprocessed", help="Path (prefix) to output data in S3")

    parser.add_argument("--metadata_file", type=str, default="resources/riskformer_slide_samples.json",)

    parser.add_argument("--tiling_config", type=str, default="configs/tiling_config.yaml",
                        help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, default="configs/foreground_config.yaml",
                        help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, default="configs/foreground_cleanup.yaml",
                        help="Foreground cleanup YAML file")

    parser.add_argument("--model_bucket", type=str, default="tcga-riskformer-preprocessing-models",)
    parser.add_argument("--model_key", type=str, default="uni/uni2-h", help="local dir for model artifact and config files")
    parser.add_argument("--model_type", type=str, default="uni", help="Model type")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for preprocessing")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    logger.info("Arguments parsed successfully.")

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    return args


def main():
    args = arg_parse()
    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logger.info(f"Project root: {project_root}")

    os.environ["AWS_REGION"] = args.region
    try:
        s3_client, session = initialize_s3_client(
            args.profile,
            region_name=args.region,
            return_session=True
        )
    except Exception as e:
        logger.error(f"Couldn't initialize S3 client: {e}")
        return
    logger.debug(f"Using AWS profile: {args.profile}, region: {args.region}")

    # Load dataset files
    to_process = load_dataset_files(s3_client, args, project_root)
    logger.info(f"Need to process {len(to_process)} new .svs files")

    # Download model files
    tmp_dir = os.path.join(project_root, "tmp")
    model_dir = os.path.join(tmp_dir, args.model_key)
    local_input_dir = os.path.join(tmp_dir, args.input_dir)
    local_out_dir = os.path.join(tmp_dir, args.output_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(f"{tmp_dir}/{args.input_dir}", exist_ok=True)
    os.makedirs(f"{tmp_dir}/{args.output_dir}", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_files = download_s3_model_files(s3_client, args, model_dir)
    logger.info(f"Downloaded {len(model_files)} model files to {model_dir}")

    # Set up Arguments
    foreground_config = os.path.join(project_root, args.foreground_config)
    foreground_cleanup_config = os.path.join(project_root, args.foreground_cleanup_config)
    tiling_config = os.path.join(project_root, args.tiling_config)

    # save dir
    local_out_dir = os.path.join(tmp_dir, args.output_dir)

    logger.info(f"Starting preprocessing for {len(to_process)} files...")
    logger.info("Using the following parameters:")
    logger.info(f"foreground_config: {foreground_config}")
    logger.info(f"foreground_cleanup_config: {foreground_cleanup_config}")
    logger.info(f"tiling_config: {tiling_config}")
    logger.info(f"model_dir: {model_dir}")
    logger.info(f"model_type: {args.model_type}")
    logger.info(f"num_workers: {args.num_workers}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"prefetch_factor: {args.prefetch_factor}")
    logger.info(f"output_dir: {local_out_dir}")
    
    for raw_key in to_process:
        raw_s3_path = f"s3://{args.bucket}/{args.input_dir}/{raw_key}"
        out_s3_dir = f"s3://{args.bucket}/{args.output_dir}"

        # TODO - download svs file to matching tmp/args.input_dir
        local_file_path = os.path.join(local_input_dir, raw_key)
        logger.info(f"Downloading {raw_s3_path} to {local_file_path}")
        s3_client.download_file(args.bucket, f"{args.input_dir}/{raw_key}", local_file_path)
        

        cmd = [
            "python", "-m", "entrypoints.preprocess",
            "--input_filename", local_file_path,
            "--output_dir", local_out_dir,
            "--foreground_config", foreground_config,
            "--foreground_cleanup_config", foreground_cleanup_config,
            "--tiling_config", tiling_config,
            "--model_dir", model_dir,
            "--model_type", args.model_type,
            "--num_workers", str(args.num_workers),
            "--batch_size", str(args.batch_size),
            "--prefetch_factor", str(args.prefetch_factor),
        ]
        if args.debug:
            cmd.append("--debug")

        logger.info("Running command...")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Preprocessing completed for {raw_s3_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during preprocessing: {e}")
            continue

        try:
            upload_preprocessing_results(s3_client, args, local_out_dir)
            logger.info(f"Uploaded preprocessing results for {out_s3_dir}")
        except Exception as e:
            logger.error(f"Error uploading preprocessing results: {e}")
            continue

        # remove tmp dir recursively
        try:
            logger.info(f"Removing tmp dir {tmp_dir}")
            subprocess.run(["rm", "-rf", tmp_dir], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing tmp dir: {e}")
            continue

        break
        

    logger.info("All done!")

if __name__ == "__main__":
    main()
