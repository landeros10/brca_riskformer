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
import subprocess
import yaml
from pathlib import Path

from entrypoints.preprocess import preprocess_one_slide
from riskformer.utils.logger_config import logger_setup, log_event
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, upload_large_files_to_bucket
from riskformer.utils.config_utils import load_preprocessing_config

logger = logging.getLogger(__name__)


def load_dataset_files(s3_client, config, project_root):
    """Load and filter dataset files from S3."""
    log_event("debug", "load_dataset_files", "started",
              s3_bucket=config['s3']['data_bucket'], 
              s3_prefix=config['s3']['input_dir'], 
              metadata_file=config['config_files']['metadata'])

    raw_files = list_bucket_files(s3_client, config['s3']['data_bucket'], config['s3']['input_dir'])
    log_event("debug", "list_s3_bucket_files", "success",
              s3_bucket=config['s3']['data_bucket'], 
              s3_prefix=config['s3']['input_dir'], 
              file_count=len(raw_files))

    processed_prefix = f"{config['s3']['output_dir']}/{config['model']['key']}"
    processed_files = list_bucket_files(s3_client, config['s3']['data_bucket'], processed_prefix)
    processed_ids = set([name.split("_")[0] for name in processed_files.keys()])
    complete_sets = [
        name.split("/")[-1] for name in processed_ids if len([f for f in processed_files.keys() if f.startswith(name)]) == 4
    ]
    log_event("debug", "list_s3_bucket_processed_files", "success",
              s3_bucket=config['s3']['data_bucket'], 
              s3_prefix=processed_prefix, 
              file_sets_count=len(complete_sets))

    metadata_file = os.path.join(project_root, config['config_files']['metadata'])
    riskformer_dataset = json.load(open(metadata_file, "r"))
    log_event("debug", "load_riskformer_metadata", "success",
              metadata_file=config['config_files']['metadata'])
    
    test_datapoint = list(riskformer_dataset.values())[0]
    log_event("debug", "riskformer_metadata_structure", "info",
              message="Metadata structure for item 0", **test_datapoint)
    
    to_process = [file.split("/")[1] for file in raw_files if file.endswith(".svs")]
    to_process = [file for file in to_process if file.split(".svs")[0] in riskformer_dataset.keys()]
    to_process = [file for file in to_process if file.split(".svs")[0] not in complete_sets]
    log_event("info", "generate_riskformer_dataset", "success",
              filtered_count=len(to_process))
    return to_process

def download_s3_model_files(s3_client, config, model_dir):
    """Download model files from S3."""
    log_event("debug", "download_s3_model_files", "started",
              s3_bucket=config['s3']['model_bucket'], 
              s3_prefix=config['model']['key'], 
              local_dir=model_dir)
    
    model_files = list_bucket_files(s3_client, config['s3']['model_bucket'], config['model']['key'])
    log_event("debug", "list_s3_model_files", "success",
              s3_bucket=config['s3']['model_bucket'], 
              s3_prefix=config['model']['key'], 
              file_count=len(model_files))
    
    for file in model_files:
        file_name = os.path.basename(file)
        local_file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(local_file_path):
            s3_client.download_file(config['s3']['model_bucket'], file, local_file_path)
            log_event("info", "download_s3_model_file", "success",
                      s3_bucket=config['s3']['model_bucket'], 
                      s3_prefix=config['model']['key'],
                      s3_key=file_name, 
                      local_path=local_file_path)
        else:
            log_event("info", "download_s3_model_file", "skipped",
                      s3_bucket=config['s3']['model_bucket'], 
                      s3_prefix=config['model']['key'],
                      s3_key=file_name, 
                      local_path=local_file_path)
            
    log_event("debug", "download_s3_model_files", "success",
              s3_bucket=config['s3']['model_bucket'], 
              s3_prefix=config['model']['key'], 
              local_dir=model_dir)
    return model_files

def upload_preprocessing_results(s3_client, config, local_out_dir):
    """Upload preprocessing results to S3."""
    log_event("debug", "upload_preprocessing_results", "started",
              local_dir=local_out_dir, 
              s3_bucket=config['s3']['data_bucket'], 
              s3_prefix=f"{config['s3']['output_dir']}/{config['model']['key']}")
    
    local_files = []
    for filename in os.listdir(local_out_dir):
        filepath = os.path.join(local_out_dir, filename)
        if os.path.isfile(filepath):
            local_files.append(filepath)
    log_event("debug", "list_local_preprocessing_files", "success",
              local_dir=local_out_dir, 
              file_count=len(local_files))

    try:
        upload_large_files_to_bucket(
            s3_client,
            bucket_name=config['s3']['data_bucket'],
            files_list=local_files,
            prefix=f"{config['s3']['output_dir']}/{config['model']['key']}",
            reupload=False,
        )
        log_event("info", "upload_preprocessing_results", "success",
                  local_dir=local_out_dir, 
                  s3_bucket=config['s3']['data_bucket'],
                  s3_prefix=f"{config['s3']['output_dir']}/{config['model']['key']}", 
                  file_count=len(local_files))
    except Exception as e:
        raise e
    log_event("debug", "upload_preprocessing_results", "success",
              local_dir=local_out_dir, 
              s3_bucket=config['s3']['data_bucket'], 
              s3_prefix=f"{config['s3']['output_dir']}/{config['model']['key']}")
    return

def arg_parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data loading / Preprocessing Orchestrator")
    parser.add_argument("--config", type=str, default="configs/preprocessing/ec2_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Override debug mode from config")
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    config = load_preprocessing_config(args.config)
    
    # Override config with command line arguments if provided
    if args.debug:
        config['processing']['debug'] = True
    
    log_event("info", "run_preprocess", "started")
    logger_setup(
        log_group="riskformer_preprocessing_ec2",
        debug=config['processing']['debug'],
        use_cloudwatch=config['processing']['use_cloudwatch'],
        profile_name=config['aws']['profile'],
        region_name=config['aws']['region'],
    )
    
    logger.setLevel(logging.DEBUG if config['processing']['debug'] else logging.INFO)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    
    torch.set_num_threads(os.cpu_count())  # Force PyTorch to use all CPUs
    torch.set_num_interop_threads(os.cpu_count())
    log_event("info", "torch_threads", "success",
              available_cpus=os.cpu_count(),
              torch_threads=torch.get_num_threads(), 
              torch_interop_threads=torch.get_num_interop_threads(),
              gpu_available=torch.cuda.is_available(), 
              gpu_count=torch.cuda.device_count())
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tmp_dir = os.path.join(project_root, "tmp")
    model_dir = os.path.join(tmp_dir, config['model']['key'])
    local_input_dir = os.path.join(tmp_dir, config['s3']['input_dir'])
    local_out_dir = os.path.join(tmp_dir, config['s3']['output_dir'])
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(f"{tmp_dir}/{config['s3']['input_dir']}", exist_ok=True)
    os.makedirs(f"{tmp_dir}/{config['s3']['output_dir']}", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_event("info", "project_info", "info",
              project_root=project_root, 
              tmp_dir=tmp_dir, 
              model_dir=model_dir,
              local_input_dir=local_input_dir, 
              local_out_dir=local_out_dir)
    
    os.environ["AWS_REGION"] = config['aws']['region']
    try:
        s3_client, _ = initialize_s3_client(
            config['aws']['profile'],
            region_name=config['aws']['region'],
            return_session=True
        )
    except Exception as e:
        log_event("error", "initialize_s3_client", "error",
                  profile=config['aws']['profile'], 
                  region=config['aws']['region'], 
                  error=str(e))
        raise e
    log_event("info", "initialize_s3_client", "success",
              profile=config['aws']['profile'], 
              region=config['aws']['region'])

    # Load dataset and model files
    to_process = load_dataset_files(s3_client, config, project_root)
    model_files = download_s3_model_files(s3_client, config, model_dir)
    log_event("info", "load_dataset_and_model_files", "success",
              dataset_files_count=len(to_process), 
              model_files_count=len(model_files))

    # Set up Arguments
    foreground_config = os.path.join(project_root, config['config_files']['foreground'])
    foreground_cleanup_config = os.path.join(project_root, config['config_files']['foreground_cleanup'])
    tiling_config = os.path.join(project_root, config['config_files']['tiling'])

    log_event("info", "preprocessing_parameters", "info",
              foreground_config=foreground_config, 
              foreground_cleanup_config=foreground_cleanup_config,
              tiling_config=tiling_config, 
              model_dir=model_dir, 
              model_type=config['model']['type'],
              num_workers=config['processing']['num_workers'], 
              batch_size=config['processing']['batch_size'], 
              prefetch_factor=config['processing']['prefetch_factor'],
              output_dir=local_out_dir)
    
    for i, raw_key in enumerate(to_process):
        percent_done = ((i + 1) / len(to_process)) * 100
        log_event("info", "preprocess_slide_orchestrator", "started",
                  file_index=i, 
                  file_count=len(to_process), 
                  percent_done=percent_done,
                  file_name=raw_key)
    
        raw_s3_path = f"s3://{config['s3']['data_bucket']}/{config['s3']['input_dir']}/{raw_key}"        
        local_file_path = os.path.join(local_input_dir, raw_key)
        try:
            s3_client.download_file(config['s3']['data_bucket'], f"{config['s3']['input_dir']}/{raw_key}", local_file_path)
        except Exception as e:
            log_event("error", "download_svs_file", "error",
                      s3_path=raw_s3_path, 
                      local_path=local_file_path, 
                      error=str(e), 
                      file_name=raw_key)
            if config['processing']['stop_on_fail']:
                raise e
            else:
                continue
        log_event("info", "download_svs_file", "success",
                  s3_path=raw_s3_path, 
                  local_path=local_file_path, 
                  file_name=raw_key)

        try:
            preprocess_one_slide(
                input_filename=local_file_path,
                output_dir=local_out_dir,
                model_dir=model_dir,
                model_type=config['model']['type'],
                foreground_config_path=foreground_config,
                foreground_cleanup_config_path=foreground_cleanup_config,
                tiling_config_path=tiling_config,
                num_workers=config['processing']['num_workers'],
                batch_size=config['processing']['batch_size'],
                prefetch_factor=config['processing']['prefetch_factor'],
            )
            log_event("debug", "preprocess_one_slide", "success",
                      file_name=raw_key)
        except Exception as e:
            log_event("error", "preprocess_one_slide", "error",
                      error=str(e), 
                      file_name=raw_key)
            if config['processing']['stop_on_fail']:
                raise e
            else:
                continue

        try:
            upload_preprocessing_results(s3_client, config, local_out_dir)
        except Exception as e:
            log_event("error", "upload_preprocessing_results", "error",
                      error=str(e), 
                      local_dir=local_out_dir)
            if config['processing']['stop_on_fail']:
                raise e
            else:
                continue

        shutil.rmtree(local_input_dir)
        shutil.rmtree(local_out_dir)
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_out_dir, exist_ok=True)
        log_event("debug", "remove_tmp_dirs", "success",
                  local_input_dir=local_input_dir, 
                  local_out_dir=local_out_dir)

        log_event("info", "preprocess_slide_orchestrator", "success",
                  file_index=i, 
                  file_count=len(to_process), 
                  percent_done=percent_done,
                  file_name=raw_key)
        
    log_event("info", "run_preprocess", "success")

if __name__ == "__main__":
    main()
