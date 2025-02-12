'''
preprocess.py
Author: landeros10
Created: 2025-02-05
'''
import logging
import argparse
import yaml
import os

import torch
from PIL import Image

from src.logger_config import logger_setup
from src.data.data_preprocess import (get_svs_samplepoints, load_encoder, extract_features,
                                      TilingConfigSchema, ForegroundConfigSchema, ForegroundCleanupConfigSchema,
                                      )
from src.data.datasets import SingleSlideDataset
from src.aws_utils import initialize_s3_client
logger = logging.getLogger(__name__)

MODEL_EXTS = [".pth", ".bin", ".pt"]
CONFIG_EXTS = [".json", ".yaml", ".yml"]

def log_config(config, tag):
    """ Logs the configuration parameters.
    
    Args:
        config (dict): configuration parameters.
        tag (str): tag for the configuration.
    """
    if not isinstance(config, dict):
        try:
            config = config.dict()
        except Exception as e:
            logger.warning(f"Failed to load config dict. Error: {e}")
            raise e

    logger.info(f"{tag} configuration:" + "=" * 20)
    for key, value in config.items():
        logger.info(f"{key}: {value}")


def load_yaml_config(config_path, schema):
    """Load a YAML config file and validate it against a schema."""
    if not config_path or not os.path.isfile(config_path):
        logger.warning(f"Config file {config_path} not found or not provided. Using defaults.")
        return schema()

    try:
        logger.info(f"Loading YAML config from {config_path}")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            logger.info("Successfully loaded YAML config.")
            if not isinstance(yaml_config, dict):
                logger.warning(f"Invalid YAML format in {config_path}. Using defaults.")
                return schema()
    except Exception as e:
        logger.warning(f"Failed to load YAML config {config_path}. Error: {e}. Using defaults.")
        return schema()
    try:
        return schema(**yaml_config)
    except Exception as e:
        logger.warning(f"Invalid values in {config_path}. Using defaults. Error: {e}")
        return schema()
    

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


def load_encoder_wrapper(model_dir, model_type):
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
        logger.warning("No model file found in the specified directory.")

    # Load model
    try:
        model, transform = load_encoder(model_type, model_path, config_files)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
    return model, transform


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for SVS slides in SageMaker")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    parser.add_argument("--tiling_config", type=str, required=False, help="Tiling parameters YAML file")
    parser.add_argument("--foreground_config", type=str, required=False, help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, required=False, help="Foreground cleanup YAML file")

    parser.add_argument("--model_dir", type=str, required=True, help="local dir for model artifact and config files")
    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()
    logger.info("Arguments parsed successfully.")

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    return args


def main():
    args = parse_args()
    
    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.info(f"Input filename: {args.input_filename}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 50)

    # keys: "tiling_config", "foreground_config", "foreground_cleanup_config"
    preprocessing_params = load_preprocessing_configs(args)

    # Collect sample points for svs file
    logger.info("[Collecting sample points for svs file]")
    try:
        sample_coords, slide_obj, slide_metadata, sampling_size, heatmap, thumb = get_svs_samplepoints(
            args.input_filename,
            foreground_config=preprocessing_params["foreground_config"],
            foreground_cleanup_config=preprocessing_params["foreground_cleanup_config"],
            tiling_config=preprocessing_params["tiling_config"],
        )
    except Exception as e:
        logger.error(f"Failed to get sample points for {args.input_filename}. Error: {e}")
        return

    if len(sample_coords) == 0:
        logger.error(f"No valid sample points found for {args.input_filename}")
        return

    if thumb is not None and heatmap is not None:
        logger.info(f"[Saving thumbnail and heatmap to output directory {args.output_dir}]")
        thumb_file = os.path.join(args.output_dir, f"{os.path.basename(args.input_filename)}_thumbnail.png")
        heatmap_file = os.path.join(args.output_dir, f"{os.path.basename(args.input_filename)}_heatmap.png")
        logger.info(f"Saving thumbnail to {thumb_file}")
        logger.info(f"Saving heatmap to {heatmap_file}")

        # thumb is already a PIL RGB image and can be saved using pillow methods
        thumb.save(thumb_file)

        #heatmap is a np.ndarray float with arbitrary range >=0 and should be converted to uint8 [0,255] for visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255.0
        heatmap = heatmap.astype("uint8")
        Image.fromarray(heatmap).save(heatmap_file)


    # Load feature extraction model
    logger.info("[Loading feature extraction model...]")
    model, transform = load_encoder_wrapper(args.model_dir, args.model_type)
    if model is None:
        logger.error("Failed to load feature extraction model.")
        return
    logger.info(f"Model successfully loaded: {args.model_type} from {args.model_dir}")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.eval().to(device)

    try:
        slide_dataset = SingleSlideDataset(
            slide_obj=slide_obj,
            slide_metadata=slide_metadata,
            sample_coords=sample_coords,
            sample_size=sampling_size,
            output_size=preprocessing_params["tiling_config"].tile_size,
            transform=transform
        )
    except Exception as e:
        logger.error(f"Failed to create single-slide dataset. Error: {e}")
        return
    logger.info(f"Dataset created with {len(slide_dataset)} samples.")
    logger.info("=" * 50)

    # TODO
    # Feature Extraction
    try:
        slide_features = extract_features(
            slide_dataset=slide_dataset,
            model=model,
            device=device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"Failed to extract features. Error: {e}")
        return        
    logger.info("Feature extraction completed.")
    


if __name__ == "__main__":
    main()
