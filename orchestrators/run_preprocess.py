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
from riskformer.utils.logger_config import logger_setup, log_event
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, upload_large_files_to_bucket

logger = logging.getLogger(__name__)


def load_dataset_files(s3_client, args, project_root):
    log_event("debug", "load_dataset_files", "started",
              s3_bucket=args.bucket, s3_prefix=args.input_dir, metadata_file=args.metadata_file)

    raw_files = list_bucket_files(s3_client, args.bucket, args.input_dir)
    log_event("debug", "list_s3_bucket_files", "success",
              s3_bucket=args.bucket, s3_prefix=args.input_dir, file_count=len(raw_files))

    processed_prefix = f"{args.output_dir}/{args.model_key}"
    processed_files = list_bucket_files(s3_client, args.bucket, processed_prefix)
    processed_ids = set([name.split("_")[0] for name in processed_files.keys()])
    complete_sets = [
        name.split("/")[-1] for name in processed_ids if len([f for f in processed_files.keys() if f.startswith(name)]) == 4
    ]
    log_event("debug", "list_s3_bucket_processed_files", "success",
              s3_bucket=args.bucket, s3_prefix=processed_prefix, file_sets_count=len(complete_sets))

    metadata_file = os.path.join(project_root, args.metadata_file)
    riskformer_dataset = json.load(open(metadata_file, "r"))
    log_event("debug", "load_riskformer_metadata", "success",
              metadata_file=args.metadata_file)
    
    test_datapoint = list(riskformer_dataset.values())[0]
    log_event("debug", "riskformer_metadata_structure", "info",
              message="Metadata structure for item 0", **test_datapoint)
    
    to_process = [file.split("/")[1] for file in raw_files if file.endswith(".svs")]
    to_process = [file for file in to_process if file.split(".svs")[0] in riskformer_dataset.keys()]
    to_process = [file for file in to_process if file.split(".svs")[0] not in complete_sets]
    log_event("info", "generate_riskformer_dataset", "success",
              filtered_count=len(to_process))
    return to_process


def download_s3_model_files(s3_client, args, model_dir):
    log_event("debug", "download_s3_model_files", "started",
              s3_bucket=args.model_bucket, s3_prefix=args.model_key, local_dir=model_dir)
    
    model_files = list_bucket_files(s3_client, args.model_bucket, args.model_key)
    log_event("debug", "list_s3_model_files", "success",
              s3_bucket=args.model_bucket, s3_prefix=args.model_key, file_count=len(model_files))
    for file in model_files:
        file_name = os.path.basename(file)
        local_file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(local_file_path):
            s3_client.download_file(args.model_bucket, file, local_file_path)
            log_event("info", "download_s3_model_file", "success",
                      s3_bucket=args.model_bucket, s3_prefix=args.model_key,
                      s3_key=file_name, local_path=local_file_path)
        else:
            log_event("info", "download_s3_model_file", "skipped",
                      s3_bucket=args.model_bucket, s3_prefix=args.model_key,
                      s3_key=file_name, local_path=local_file_path)
            
    log_event("debug", "download_s3_model_files", "success",
              s3_bucket=args.model_bucket, s3_prefix=args.model_key, local_dir=model_dir)
    return model_files


def upload_preprocessing_results(s3_client, args, local_out_dir):
    """
    Uploads the preprocessing results to S3.
    """
    log_event("debug", "upload_preprocessing_results", "started",
              local_dir=local_out_dir, s3_bucket=args.bucket, s3_prefix=f"{args.output_dir}/{args.model_key}")
    
    local_files = []
    for filename in os.listdir(local_out_dir):
        filepath = os.path.join(local_out_dir, filename)
        if os.path.isfile(filepath):
            local_files.append(filepath)
    log_event("debug", "list_local_preprocessing_files", "success",
              local_dir=local_out_dir, file_count=len(local_files))

    try:
        upload_large_files_to_bucket(
            s3_client,
            bucket_name=args.bucket,
            files_list=local_files,
            prefix=f"{args.output_dir}/{args.model_key}",
            reupload=False,
        )
        log_event("info", "upload_preprocessing_results", "success",
                  local_dir=local_out_dir, s3_bucket=args.bucket,
                  s3_prefix=f"{args.output_dir}/{args.model_key}", file_count=len(local_files))
    except Exception as e:
        raise e
    log_event("debug", "upload_preprocessing_results", "success",
              local_dir=local_out_dir, s3_bucket=args.bucket, s3_prefix=f"{args.output_dir}/{args.model_key}")
    return
    

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

    log_event("info", "arg_parse", "success", **vars(args))
    return args


def main():
    log_event("info", "run_preprocess", "started")
    args = arg_parse()
    logger_setup(
        log_group="riskformer_preprocessing_ec2",
        debug=args.debug,
        use_cloudwatch=args.use_cloudwatch,
        profile_name=args.profile,
        region_name=args.region,
    )
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    

    torch.set_num_threads(os.cpu_count())  # Force PyTorch to use all CPUs
    torch.set_num_interop_threads(os.cpu_count())
    log_event("info", "torch_threads", "success",
              available_cpus=os.cpu_count(),
              torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
              gpu_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tmp_dir = os.path.join(project_root, "tmp")
    model_dir = os.path.join(tmp_dir, args.model_key)
    local_input_dir = os.path.join(tmp_dir, args.input_dir)
    local_out_dir = os.path.join(tmp_dir, args.output_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(f"{tmp_dir}/{args.input_dir}", exist_ok=True)
    os.makedirs(f"{tmp_dir}/{args.output_dir}", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_event("info", "project_info", "info",
              project_root=project_root, tmp_dir=tmp_dir, model_dir=model_dir,
              local_input_dir=local_input_dir, local_out_dir=local_out_dir)
    
    os.environ["AWS_REGION"] = args.region
    try:
        s3_client, _ = initialize_s3_client(
            args.profile,
            region_name=args.region,
            return_session=True
        )
    except Exception as e:
        log_event("error", "initialize_s3_client", "error",
                  profile=args.profile, region=args.region, error=str(e))
        raise e
    log_event("info", "initialize_s3_client", "success",
              profile=args.profile, region=args.region)

    # Load dataset and model files
    to_process = load_dataset_files(s3_client, args, project_root)
    model_files = download_s3_model_files(s3_client, args, model_dir)
    log_event("info", "load_dataset_and_model_files", "success",
              dataset_files_count=len(to_process), model_files_count=len(model_files))

    # Set up Arguments
    foreground_config = os.path.join(project_root, args.foreground_config)
    foreground_cleanup_config = os.path.join(project_root, args.foreground_cleanup_config)
    tiling_config = os.path.join(project_root, args.tiling_config)

    log_event("info", "preprocessing_parameters", "info",
              foreground_config=foreground_config, foreground_cleanup_config=foreground_cleanup_config,
              tiling_config=tiling_config, model_dir=model_dir, model_type=args.model_type,
              num_workers=args.num_workers, batch_size=args.batch_size, prefetch_factor=args.prefetch_factor,
              output_dir=local_out_dir)
    
    for i, raw_key in enumerate(to_process):
        percent_done = ((i + 1) / len(to_process)) * 100
        log_event("info", "preprocess_slide_orchestrator", "started",
                  file_index=i, file_count=len(to_process), percent_done=percent_done,
                  file_name=raw_key)
    
        raw_s3_path = f"s3://{args.bucket}/{args.input_dir}/{raw_key}"        
        local_file_path = os.path.join(local_input_dir, raw_key)
        try:
            s3_client.download_file(args.bucket, f"{args.input_dir}/{raw_key}", local_file_path)
        except Exception as e:
            log_event("error", "download_svs_file", "error",
                      s3_path=raw_s3_path, local_path=local_file_path, error=str(e), file_name=raw_key)
            if args.stop_on_fail:
                raise e
            else:
                continue
        log_event("info", "download_svs_file", "success",
                  s3_path=raw_s3_path, local_path=local_file_path, file_name=raw_key)

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
            log_event("debug", "preprocess_one_slide", "success",
                      file_name=raw_key)
        except Exception as e:
            log_event("error", "preprocess_one_slide", "error",
                      error=str(e), file_name=raw_key)
            if args.stop_on_fail:
                raise e
            else:
                continue

        try:
            upload_preprocessing_results(s3_client, args, local_out_dir)
        except Exception as e:
            log_event("error", "upload_preprocessing_results", "error",
                      error=str(e), local_dir=local_out_dir)
            if args.stop_on_fail:
                raise e
            else:
                continue

        shutil.rmtree(local_input_dir)
        shutil.rmtree(local_out_dir)
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_out_dir, exist_ok=True)
        log_event("debug", "remove_tmp_dirs", "success",
                  local_input_dir=local_input_dir, local_out_dir=local_out_dir)

        log_event("info", "preprocess_slide_orchestrator", "success",
                  file_index=i, file_count=len(to_process), percent_done=percent_done,
                  file_name=raw_key)
        
    log_event("info", "run_preprocess", "success")


if __name__ == "__main__":
    main()
