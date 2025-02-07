import logging
import argparse
import yaml
import os

import torch
from transformers import AutoModel

from src.logger_config import logger_setup
from src.data.data_preprocess import (get_svs_samplepoints, SingleSlideDataset,
                                      TilingConfigSchema, ForegroundConfigSchema, ForegroundCleanupConfigSchema,
                                      load_model)
from src.data.data_utils import initialize_s3_client
logger = logging.getLogger(__name__)

def log_config(config, tag):
    """ Logs the configuration parameters.
    
    Args:
        config (dict): configuration parameters.
        tag (str): tag for the configuration.
    """
    logger.info(f"{tag} configuration:" + "=" * 20)
    for key, value in config.items():
        logger.info(f"{key}: {value}")


def load_yaml_config(config_path, schema):
    """Load a YAML config file and validate it against a schema."""
    if not config_path or not os.path.isfile(config_path):
        logger.warning(f"Config file {config_path} not found or not provided. Using defaults.")
        return schema().dict()

    try:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            if not isinstance(yaml_config, dict):
                logger.warning(f"Invalid YAML format in {config_path}. Using defaults.")
                return schema().dict()
    except Exception as e:
        logger.warning(f"Failed to load YAML config {config_path}. Error: {e}. Using defaults.")
        return schema().dict()
    try:
        return schema(**yaml_config).dict()
    except Exception as e:
        logger.warning(f"Invalid values in {config_path}. Using defaults. Error: {e}")
        return schema().dict()
    

def load_preprocessing_configs(args):
    tiling_config = load_yaml_config(args.tiling_config, TilingConfigSchema)
    foreground_config = load_yaml_config(args.foreground_config, ForegroundConfigSchema)
    foreground_cleanup_config = load_yaml_config(args.foreground_cleanup_config, ForegroundCleanupConfigSchema)
    
    log_config(tiling_config, "Tiling Parameters")
    log_config(foreground_config, "Foreground Detection Parameters")
    log_config(foreground_cleanup_config, "Foreground Cleanup Parameters")

    return {
        "tiling_config": tiling_config,
        "foreground_config": foreground_config,
        "foreground_cleanup_config": foreground_cleanup_config,
    }


def download_model_from_s3(bucket_name, model_key, local_model_path="opt/ml/model"):
    """Download a model from S3 to a local path."""
    s3_client = initialize_s3_client()
    local_model_path = os.path.join(local_model_path, os.path.basename(model_key))

    logger.info(f"Downloading model from S3: s3://{bucket_name}/{model_key} to {local_model_path}")
    s3_client.download_file(bucket_name, model_key, local_model_path)
    return local_model_path


def load_feature_extractor(model_type, local_model_path):
    """Load a feature extractor from a local model path."""
    model = load_model(model_type, local_model_path)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for SVS slides in SageMaker")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output", help="Output directory")
    
    parser.add_argument("--tiling_config", type=str, required=False, help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, required=False, help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, required=False, help="Foreground cleanup YAML file")

    parser.add_argument("--model_bucket", type=str, required=True, help="S3 bucket for model artifacts")
    parser.add_argument("--model_key", type=str, required=True, help="S3 key for model artifacts")
    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.debug(f"Input filename: {args.input_filename}")

    # keys: "tiling_config", "foreground_config", "foreground_cleanup_config"
    preprocessing_params = load_preprocessing_configs(args)

    # Collect sample points for svs file
    sample_coords, sample_size, slide_obj, metadata, _ = get_svs_samplepoints(
        args.input_filename,
        preprocessing_params["tiling_config"],
        preprocessing_params["foreground_config"],
        preprocessing_params["foreground_cleanup_config"],
        return_heatmap=False,
    )

    # Log cuda availability and set proper device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, transform = load_model(args.model_type, device)

    # Create dataset object
    transform = None
    dataset = SingleSlideDataset(
        slide_obj=slide_obj,
        slide_metadata=metadata,
        sample_coords=sample_coords,
        sample_size=sample_size,
        output_size=preprocessing_params["tiling_config"]["size"],
        transform=transform
    )

    # TODO - go through slides and convert patches to features using all_coords
    pass


if __name__ == "__main__":
    main()
