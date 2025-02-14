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

from riskformer.utils.logger_config import logger_setup
from riskformer.data import data_preprocess as preprocessor
from riskformer.data.datasets import SingleSlideDataset
logger = logging.getLogger(__name__)


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

    # keys: "tiling_config", "foreground_config", "foreground_cleanup_config"
    preprocessing_params = preprocessor.load_preprocessing_configs(args)

    # Collect sample points for svs file
    logger.info("[Collecting sample points for svs file]")
    try:
        sample_coords, slide_obj, slide_metadata, sampling_size, heatmap, thumb = preprocessor.get_svs_samplepoints(
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
    model, transform = preprocessor.load_encoder(args.model_dir, args.model_type)
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

    # Feature Extraction
    logger.info("Extracting features from sampled images...")
    try:
        slide_features = preprocessor.extract_features(
            slide_dataset=slide_dataset,
            model=model,
            device=device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch_factor,
        )
    except Exception as e:
        logger.error(f"Failed to extract features. Error: {e}")
        return        
    logger.info(f"Successfully extracted tile features for foreground samples. Feature shape: {slide_features.shape}")

    # save features as zarr file in args.output_dir
    logger.info(f"Saving feature vectors to {args.output_dir}")
    coo_coords = preprocessor.get_COO_coords(
        coords=sample_coords,
        sampling_size=sampling_size,
        tile_overlap=preprocessing_params["tiling_config"].tile_overlap,
    )
    if len(coo_coords) != slide_features.shape[0]:
        logger.error(f"Number of feature vectors does not match number of sample points.")
        return
    logger.debug(f"Generated coordinates for sparse representation.")

    logger.debug("Saving feature vectors to zarr file...")
    try:
        preprocessor.save_features_zarr(
            output_path=os.path.join(args.output_dir, f"{os.path.basename(args.input_filename)}.zarr"),
            coo_coords=coo_coords,
            slide_features=slide_features,
            chunk_size=min(5000, max(1000, slide_features.shape[0] // 4)),
        )
    except Exception as e:
        logger.error(f"Failed to save feature vectors to zarr file. Error: {e}")
        return
    logger.info("Feature vectors saved successfully.")    


if __name__ == "__main__":
    main()
