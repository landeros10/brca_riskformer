import logging
import argparse
import yaml
import os

import torch
from transformers import AutoModel

from src.logger_config import logger_setup
from src.data.data_preprocess import (get_svs_samplepoints, SingleSlideDataset,
                                      TilingConfigSchema, ForegroundConfigSchema, ForegroundCleanupConfigSchema,
                                      load_model_from_path)
from src.data.data_utils import initialize_s3_client
logger = logging.getLogger(__name__)

MODEL_EXTS = [".pth", ".bin", ".pt"]
CONFIG_EXTS = [".json", ".yaml", ".yml"]

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


def find_model_files(model_dir):
    """
    Find model and config files in the given directory.
    Args:
        model_dir (str): The directory to search for model and config files.
        
    Returns:
        tuple: A tuple containing the model path and a dictionary of config files."""
    model_path = None
    config_files = {}

    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        logger.error(f"Model directory '{model_dir}' does not exist or is not a directory.")
        return model_path, config_files

    # Scan local directory for model and config files
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            if file.endswith(tuple(MODEL_EXTS)) and model_path is None:
                model_path = file_path
            elif file.endswith(tuple(MODEL_EXTS)):
                logger.warning(f"Found multiple model files. Using: {model_path}, ignoring: {file_path}")
            elif file.endswith(tuple(CONFIG_EXTS)):
                config_name = os.path.splitext(file)[0]
                config_files[config_name] = file_path
            else:
                logger.warning(f"Found unexpected file type: {file}")

    # Ensure a model file is found
    if model_path is None:
        logger.error(f"No valid model file found in {model_dir}")

    if not config_files:
        logger.warning("No model config files found! Using default model config.")

    return model_path, config_files


def load_feature_extractor(model_dir, model_type):
    """
    Load the feature extractor model from a local directory.

    Args:
        model_dir (str): The local directory containing the model and config files.
        model_type (str): The model type.

    Returns:
        model: The loaded model.
    """
    model = None
    transform = None
    model_path, config_files = find_model_files(model_dir)
    if model_path is None:
        logger.error("No model file found in the specified directory.")

    # Load model
    try:
        model, transform = load_model_from_path(model_type, model_path, config_files)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
    return model, transform


def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for SVS slides in SageMaker")
    parser.add_argument("--profile", type=str, default="651340551631_AWSPowerUserAccess", help="AWS profile name")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output", help="Output directory")
    
    parser.add_argument("--tiling_config", type=str, required=False, help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, required=False, help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, required=False, help="Foreground cleanup YAML file")

    parser.add_argument("--model_dir", type=str, required=True, help="local dir for model artifact and config files")
    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")
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


    # Initialize s3 client
    s3_client = initialize_s3_client()
    if s3_client is None:
        logger.error("Failed to initialize S3 client.")
        return

    # Load feature extraction model
    model, transform = load_feature_extractor(args.model_dir, args.model_type)
    if model is None:
        logger.error("Failed to load feature extraction model.")
        return

    # Create dataset object
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
