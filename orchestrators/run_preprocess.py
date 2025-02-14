#!/usr/bin/env python3
"""
run_preprocess.py

Orchestrates a preprocessing job over new files in S3.

Author: landeros10
Created: 2025-12-05
"""
import os
import logging
import argparse
import boto3
import subprocess

from src.logger_config import logger_setup
from src.aws_utils import initialize_s3_client

logger = logging.getLogger(__name__)

def list_s3_keys(s3_client, bucket_name, prefix=""):
    """
    Return a list of all object keys under `prefix` in the given bucket.
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def main():
    # -----------------------------
    # Parse Command Line Args
    # -----------------------------
    parser = argparse.ArgumentParser(description="Data loading / Preprocessing Orchestrator")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess",
                        help="AWS profile name")
    parser.add_argument("--bucket", type=str, default="tcga-riskformer-data-2025",
                        help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--input_dir", type=str, default="raw", help="Path (prefix) to input data in S3")
    parser.add_argument("--output_dir", type=str, default="preprocessed", help="Path (prefix) to output data in S3")

    parser.add_argument("--tiling_config", type=str, default="./tiling_config.yaml",
                        help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, default="./foreground_config.yaml",
                        help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, default="./foreground_cleanup.yaml",
                        help="Foreground cleanup YAML file")

    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")
    parser.add_argument("--models_file", type=str, default="/config/preprocessing_models.json",
                        help="Path to models list (if needed)")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # -----------------------------
    # Logger Setup
    # -----------------------------
    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # -----------------------------
    # Initialize S3
    # -----------------------------
    os.environ["AWS_REGION"] = args.region
    s3_client, session = initialize_s3_client(
        args.profile,
        region_name=args.region,
        return_session=True
    )
    logger.debug(f"Using AWS profile: {args.profile}, region: {args.region}")
    logger.debug("Initialized S3 client.")

    bucket = args.bucket
    raw_prefix = args.input_dir.rstrip("/") + "/"
    preproc_prefix = args.output_dir.rstrip("/") + "/"

    # -----------------------------
    # 1) List raw files
    # -----------------------------
    raw_keys = list_s3_keys(s3_client, bucket, raw_prefix)
    # Filter if you only want certain extensions:
    raw_keys = [k for k in raw_keys if k.lower().endswith(".svs")]
    logger.info(f"Found {len(raw_keys)} potential raw .svs files in s3://{bucket}/{raw_prefix}")

    # -----------------------------
    # 2) List existing preprocessed
    # -----------------------------
    preproc_keys = list_s3_keys(s3_client, bucket, preproc_prefix)
    preproc_basenames = {os.path.basename(k) for k in preproc_keys}
    logger.info(f"Found {len(preproc_keys)} items under s3://{bucket}/{preproc_prefix}")

    # -----------------------------
    # 3) Determine which raw files are NOT preprocessed
    #    (by comparing base filenames)
    # -----------------------------
    to_process = []
    for rk in raw_keys:
        base_name = os.path.basename(rk)  # e.g. "SAMPLE.svs"
        if base_name not in preproc_basenames:
            to_process.append(rk)

    logger.info(f"Need to process {len(to_process)} new files...")

    # -----------------------------
    # 4) For each file, run src.preprocess
    #    passing S3 input_filename & output_dir if your script supports it.
    #    If not, you'll need to cp down/up locally.
    # -----------------------------
    for raw_key in to_process:
        raw_s3_path = f"s3://{bucket}/{raw_key}"
        out_s3_path = f"s3://{bucket}/{preproc_prefix}"

        # Build the command
        cmd = [
            "python", "-m", "src.preprocess",
            "--input_filename", raw_s3_path,
            "--output_dir", out_s3_path,
            "--model_type", args.model_type,
            "--foreground_config", args.foreground_config,
            "--foreground_cleanup_config", args.foreground_cleanup_config,
            "--tiling_config", args.tiling_config,
        ]

        # If your preprocess script needs a local model_dir, read from models_file,
        # or if you want to pass model_dir directly, adapt here.
        # Example: let's assume you pass --model_dir=the directory of your model
        # Or skip if your script doesn't need it:

        # For demonstration, let's parse or guess a directory from "models_file"
        # (You might have a separate approach, adapt as needed)
        # ...
        # cmd += ["--model_dir", "/workspace/models/resnet50"]  # Example only
        # OR if you store your model_type => model_dir in preprocessing_models.json, parse that here.

        if args.debug:
            cmd.append("--debug")

        logger.info(f"\n=== Processing {raw_s3_path} ===\nCMD: {' '.join(cmd)}\n")
        try:
            subprocess.check_call(cmd)
            logger.info(f"Finished: {raw_s3_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while processing {raw_s3_path}: {e}")
            # Decide if you want to continue or abort on error. For now, let's continue.
            continue

    logger.info("All done!")

if __name__ == "__main__":
    main()
