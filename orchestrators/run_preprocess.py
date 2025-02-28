#!/usr/bin/env python3
"""
run_preprocess.py

Orchestrates a preprocessing job over new files in S3.

Author: landeros10
Created: 2025-12-05
"""
import os
import shutil
import json
import time
import logging
import argparse
import torch

from entrypoints.preprocess import preprocess_one_slide
from riskformer.utils.logger_config import logger_setup
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, upload_large_files_to_bucket

logger = logging.getLogger(__name__)


def load_dataset_files(s3_client, args, project_root):
    raw_files = list_bucket_files(s3_client, args.bucket, args.input_dir)
    logger.debug(f"Found {len(raw_files)} files in s3://{args.bucket}/{args.input_dir}...")

    processed_prefix = f"{args.output_dir}/{args.model_key}"
    processed_files = list_bucket_files(s3_client, args.bucket, processed_prefix)
    logger.debug(f"Found {len(processed_files)//4} file sets in {args.output_dir}...")
    processed_ids = set([name.split("_")[0] for name in processed_files.keys()])
    complete_sets = [
        name.split("/")[-1] for name in processed_ids if len([f for f in processed_files.keys() if f.startswith(name)]) == 4
    ]

    logger.info("Loading riskformer dataset metadata...")
    metadata_file = os.path.join(project_root, args.metadata_file)
    riskformer_dataset = json.load(open(metadata_file, "r"))
    logger.info("Metadata structure for item 0:")
    test_datapoint = list(riskformer_dataset.values())[0]
    for key, value in test_datapoint.items():
        logger.info(f"\t{key}:\t{value}")

    to_process = [file.split("/")[1] for file in raw_files if file.endswith(".svs")]
    to_process = [file for file in to_process if file.split(".svs")[0] in riskformer_dataset.keys()]
    logger.debug(f"Filtered to {len(to_process)} SVS files in Riskformer dataset")

    to_process = [file for file in to_process if file.split(".svs")[0] not in complete_sets]
    logger.info(f"{len(to_process)} files not pre-processed in Riskformer dataset")

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
    local_files = []
    for filename in os.listdir(local_out_dir):
        filepath = os.path.join(local_out_dir, filename)
        if os.path.isfile(filepath):
            local_files.append(filepath)
        else:
            # Not a file; skip it
            logger.warning(f"Skipping {filepath}, not a file.")

    try:
        upload_large_files_to_bucket(
            s3_client,
            bucket_name=args.bucket,
            files_list=local_files,
            prefix=f"{args.output_dir}/{args.model_key}",
            reupload=False,
        )
    except Exception as e:
        logger.error(f"Error uploading files from {local_out_dir} to S3: {e}")
    

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

    parser.add_argument("--foreground_config", type=str, default="configs/foreground_config.yaml", help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, default="configs/foreground_cleanup.yaml", help="Foreground cleanup YAML file")
    parser.add_argument("--tiling_config", type=str, default="configs/tiling_config.yaml", help="Tiling parameters YAML file")

    parser.add_argument("--model_type", type=str, default="uni", help="Model type")
    parser.add_argument("--model_bucket", type=str, default="tcga-riskformer-preprocessing-models",)
    parser.add_argument("--model_key", type=str, default="uni/uni2-h", help="local dir for model artifact and config files")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for preprocessing")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")

    parser.add_argument("--stop_on_fail", action="store_true", help="Stop on first slide failure")
    parser.add_argument("--use_cloudwatch", action="store_true", help="Use CloudWatch for logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    logger.info("Arguments parsed successfully.")

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    return args


def main():
    args = arg_parse()
    logger_setup(
        log_group="run_preprocess",
        debug=args.debug,
        use_cloudwatch=args.use_cloudwatch,
        profile_name=args.profile,
        region_name=args.region,
    )
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    
    logger.info(f"Available CPUs: {os.cpu_count()}")  # Should be 32
    logger.info(f"PyTorch Threads: {torch.get_num_threads()}")  # Might be 1

    torch.set_num_threads(os.cpu_count())  # Force PyTorch to use all CPUs
    torch.set_num_interop_threads(os.cpu_count())  # Helps inter-op parallelism
    
    logger.info(f"Updated PyTorch Threads: {torch.get_num_threads()}")
    logger.info("=" * 50)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logger.info(f"Project root: {project_root}")

    os.environ["AWS_REGION"] = args.region
    try:
        s3_client, _ = initialize_s3_client(
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
    
    overall_time = time.time()
    for i, raw_key in enumerate(to_process):
        logger.info(f"({(time.time() - overall_time) / 60:.2f} minutes) Processing {i+1}/{len(to_process)}: {raw_key}...")
        bucket_prefix=f"{args.output_dir}/{args.model_key}"
        existing_files = list_bucket_files(s3_client, args.bucket, bucket_prefix)

        start_time = time.time()
        raw_s3_path = f"s3://{args.bucket}/{args.input_dir}/{raw_key}"
        out_s3_dir = f"s3://{args.bucket}/{args.output_dir}"

        local_file_path = os.path.join(local_input_dir, raw_key)
        logger.info(f"Downloading {raw_s3_path} to {local_file_path}")
        try:
            s3_client.download_file(args.bucket, f"{args.input_dir}/{raw_key}", local_file_path)
        except Exception as e:
            logger.error(f"Error downloading {raw_s3_path}: {e}")
            continue

        try:
            preprocess_one_slide(
                input_filename=local_file_path,
                output_dir=local_out_dir,
                model_dir=model_dir,
                model_type=args.model_type,
                foreground_config_path=foreground_config,
                foreground_cleanup_config_path=foreground_cleanup_config,
                tiling_config_path=tiling_config,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                prefetch_factor=args.prefetch_factor,
            )
            logger.info(f"Finished pre-processing {raw_key}")
            upload_preprocessing_results(s3_client, args, local_out_dir)
            logger.info(f"Successfully uploaded preprocessing results to {out_s3_dir}")
        except Exception as e:
            logger.error(f"Error preprocessing slide {raw_key}: {e}")
            if args.stop_on_fail:
                raise e
            else:
                continue

        logger.info(f"Removing tmp dirs: {local_input_dir}, {local_out_dir}")
        shutil.rmtree(local_input_dir)
        shutil.rmtree(local_out_dir)
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_out_dir, exist_ok=True)
        logger.info(f"Time taken for {raw_key}: {(time.time() - start_time) / 60:.2f} minutes")
        logger.info("=" * 50)
        logger.info("=" * 50)
    logger.info(f"All done! Total time: {(time.time() - overall_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
