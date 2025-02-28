'''
preprocess.py
Author: landeros10
Created: 2025-02-05
'''
import logging
import argparse
import os

import torch
from PIL import Image
import numpy as np
import h5py

from riskformer.utils.logger_config import logger_setup
from riskformer.data import data_preprocess as preprocessor
from riskformer.data.datasets import SingleSlideDataset
logger = logging.getLogger(__name__)


def save_sparse_feature_array(
    sample_coords,
    sampling_size,
    tile_overlap,
    slide_features,
    output_dir,
    basename,
):
    logger.debug("Saving sparse feature vectors...")
    coo_coords = preprocessor.get_COO_coords(
        coords=sample_coords,
        sampling_size=sampling_size,
        tile_overlap=tile_overlap,
    )
    if len(coo_coords) != slide_features.shape[0]:
        logger.error(f"Number of feature vectors does not match number of sample points.")
        return
    logger.debug(f"Generated coordinates for sparse representation.")

    try:
        preprocessor.save_features_h5(
            output_path=os.path.join(output_dir, basename),
            coo_coords=coo_coords,
            slide_features=slide_features,
            chunk_size=min(5000, max(1000, slide_features.shape[0] // 4)),
        )
    except Exception as e:
        logger.error(f"Failed to save feature vectors to zarr file. Error: {e}")
        return
    logger.debug("Feature vectors saved successfully.")    


def save_image_output(
        image,
        output_dir,
        basename,
        tag="thumbnail",
        normalize=False
    ):
    if image is not None:
        image_file = os.path.join(output_dir, f"{basename}_{tag}.png")
        logger.debug(f"[Saving {tag} to {image_file}")

        if isinstance(image, Image.Image):
            image = np.array(image)

        if normalize:
            image = image.astype("float")
            image = (image - image.min()) / (1e-6 + (image.max() - image.min())) * 255.0
        
        image = image.astype("uint8")
        Image.fromarray(image).save(image_file)
    else:
        logger.warning(f"Provided NoneType image for {tag}. Skipping save.")


def preprocess_one_slide(
        input_filename,
        output_dir,
        model_dir,
        model_type,
        foreground_config_path,
        foreground_cleanup_config_path,
        tiling_config_path,
        num_workers,
        batch_size,
        prefetch_factor,
):
    # keys: "tiling_config", "foreground_config", "foreground_cleanup_config"
    preprocessing_params = preprocessor.load_preprocessing_configs(
        foreground_config_path=foreground_config_path,
        foreground_cleanup_config_path=foreground_cleanup_config_path,
        tiling_config_path=tiling_config_path,
    )
    foreground_config = preprocessing_params["foreground_config"]
    foreground_cleanup_config = preprocessing_params["foreground_cleanup_config"]
    tiling_config = preprocessing_params["tiling_config"]

    ### Collect sample points for svs file ###
    logger.info("Collecting sample points for svs file...")
    try:
        sample_coords, slide_obj, slide_metadata, sampling_size, heatmap, thumb = preprocessor.get_svs_samplepoints(
            input_filename,
            foreground_config=foreground_config,
            foreground_cleanup_config=foreground_cleanup_config,
            tiling_config=tiling_config,
        )
        if len(sample_coords) == 0:
            logger.error(f"No valid sample points found for {input_filename}")
            return
    except Exception as e:
        logger.error(f"Failed to get sample points for {input_filename}. Error: {e}")
        raise e

    ### Load feature extraction model ###
    logger.info("Loading feature extraction model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    model, transform = preprocessor.load_encoder(
            model_dir=model_dir,
            model_type=model_type,
            device=device,
        )
    if model is None:
        logger.error("Failed to load feature extraction model.")
        raise ValueError("Model loading failed.")
    logger.info(f"Model successfully loaded!")

    model = model.eval().to(device)
    logger.info(f"Creating single-slide dataset with {len(sample_coords)} samples...")
    try:
        slide_dataset = SingleSlideDataset(
            slide_obj=slide_obj,
            slide_metadata=slide_metadata,
            sample_coords=sample_coords,
            sample_size=sampling_size,
            transform=transform
        )
    except Exception as e:
        logger.error(f"Failed to create single-slide dataset. Error: {e}")
        raise e
    logger.info(f"Dataset created!")

    ### Feature Extraction ###
    logger.info("Extracting features from sampled images...")
    try:
        slide_features = preprocessor.extract_features(
            slide_dataset=slide_dataset,
            model=model,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
        )
    except Exception as e:
        logger.error(f"Failed to extract features. Error: {e}")
        raise e        
    logger.info(f"Successfully extracted tile features (dim={slide_features.shape[1]})")

    ### Save Output ###
    logger.info("Saving processed data to local output dir...")
    slide_id = os.path.basename(input_filename).split(".svs")[0]
    try:
        save_image_output(
            thumb,
            output_dir=output_dir,
            basename=slide_id,
            tag="thumbnail")
        save_image_output(
            heatmap,
            output_dir=output_dir,
            basename=slide_id,
            tag="heatmap",
            normalize=True)
        save_sparse_feature_array(
            sample_coords=sample_coords,
            sampling_size=sampling_size,
            tile_overlap=tiling_config.tile_overlap,
            slide_features=slide_features,
            output_dir=output_dir,
            basename=slide_id,
        )
    except Exception as e:
        logger.error(f"Failed to save output data. Error: {e}")
        raise e
    logger.info("Output data saved successfully to local output dir.")
    slide_dataset.close_slide()


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for SVS slides in SageMaker")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    parser.add_argument("--foreground_config", type=str, required=False, help="Foreground detection YAML file")
    parser.add_argument("--foreground_cleanup_config", type=str, required=False, help="Foreground cleanup YAML file")
    parser.add_argument("--tiling_config", type=str, required=False, help="Tiling parameters YAML file")

    parser.add_argument("--model_dir", type=str, required=True, help="local dir for model artifact and config files")
    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")

    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")
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

    preprocess_one_slide(
        args.input_filename,
        args.output_dir,
        args.model_dir,
        args.model_type,
        args.foreground_config,
        args.foreground_cleanup_config,
        args.tiling_config,
        args.num_workers,
        args.batch_size,
        args.prefetch_factor,
    )



if __name__ == "__main__":
    main()
