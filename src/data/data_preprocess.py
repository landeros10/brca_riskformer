'''
data_preprocess.py

SVS preprocessing functions
Author: landeros10
Created: 2025-02-05
'''
import logging
from typing import Type

import os
import time
import json
import yaml
import zarr
import numcodecs
from multiprocessing import Pool, cpu_count
from pydantic import BaseModel, Field

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
import timm # type: ignore
from timm.data import resolve_data_config # type: ignore
from timm.data.transforms_factory import create_transform # type: ignore
# from src.randstainna import RandStainNA

from src.logger_config import logger_setup
from src.data.data_utils import (open_svs, get_slide_samplepoints,)

logger = logging.getLogger(__name__)

# yaml_file = '/home/ubuntu/notebooks/cpc_hist/src/CRC_LAB_randomTrue_n0.yaml'
# stain_augmentor = RandStainNA(yaml_file, std_hyper=-0.0)
# stain_normalizer = RandStainNA(yaml_file, std_hyper=-1.0)


class ForegroundConfigSchema(BaseModel):
    thumb_size: int = Field(default=500, description="Size of the thumbnail to use for foreground.")
    sampling_fraction: float = Field(default=0.10, description="Fraction of the image to use for foreground sampling.")
    bandwidth: int = Field(default=2, description="Bandwidth of the Gaussian kernel to use for foreground estimation.")
    min_tissue_prob: float = Field(default=0.05, description="Minimum tissue probability to consider a pixel as foreground.")
    min_foreground_ratio: float = Field(default=0.15, description="Minimum foreground/background ratio in slide.")


class ForegroundCleanupConfigSchema(BaseModel):
    opening_kernel: int = Field(default=3, description="Size of the structuring element to use for opening foreground mask.")
    square_fill_kernel: int = Field(default=15, description="Size of the structuring element to use for closing foreground mask.")
    square_dilation_kernel: int = Field(default=1, description="Size of the structuring element to use for final dilation of foreground mask.")
    edge_margin: int = Field(default=50, description="Size of the border to consider for aberrant regions.")
    max_width_ratio: float = Field(default=0.9, description="Percentage of the mask width covered to consider as aberrant.")
    max_height_ratio: float = Field(default=0.75, description="Percentage of the mask height covered to consider as aberrant.")


# This was designed for images taken at (1/4) resolution of 20X
class TilingConfigSchema(BaseModel):
    tile_size: int = Field(default=256, description="Size of the tile to extract from the slide.")
    tile_overlap: float = Field(default=0.1, description="Overlap between tiles.")
    patch_foreground_ratio: float = Field(default=0.5, description="Minimum fraction of foreground pixels in a tile.")
    reference_mag: float = Field(default=20.0, description="Reference magnification level for tiling.")


class DefaultUni2hConfig(BaseModel):
    model_name: str = Field(default="vit_giant_patch14_224", description="UNI2-h model name")
    img_size: int = Field(default=224, description="Image size")
    patch_size: int = Field(default=14, description="ViT patch size")
    depth: int = Field(default=24, description="ViT depth")
    num_heads: int = Field(default=24, description="ViT number of heads")
    init_values: float = Field(default=1e-5, description="ViT init values")
    embed_dim: int = Field(default=1536, description="ViT embedding dimension")
    mlp_ratio: float = Field(default=2.66667*2, description="ViT MLP ratio")
    num_classes: int = Field(default=0, description="Number of classes")
    no_embed_class: bool = Field(default=True, description="No embed class")
    mlp_layer: Type[torch.nn.Module] = Field(default=timm.layers.SwiGLUPacked, description="MLP layer")
    act_layer: Type[torch.nn.Module] = Field(default=torch.nn.SiLU, description="Activation layer")
    reg_tokens: int = Field(default=8, description="Number of reg tokens")
    dynamic_img_size: bool = Field(default=True, description="Dynamic image size")


DEFAULT_FOREGROUND_CONFIG = ForegroundConfigSchema().dict()
DEFAULT_FOREGROUND_CLEANUP_CONFIG = ForegroundCleanupConfigSchema().dict()
DEFAULT_TILING_CONFIG = TilingConfigSchema().dict()
DEFAULT_UNI_CONFIG = DefaultUni2hConfig().dict()


def get_svs_samplepoints(
    svs_file, 
    foreground_config, 
    foreground_cleanup_config, 
    tiling_config, 
):
    """
    Extracts sampling points from the SVS file based on the foreground mask.
    
    Args:
        svs_file (str): path to SVS file.
        foreground_config (ForegroundConfigSchema): histopathology foreground detection parameters.
        foreground_cleanup_config (ForegroundCleanupConfigSchema): foreground cleanup parameters.
        tiling_config (TilingConfigSchema): whole-slide sampling parameters.

    Returns:
        sampling_coords (np.ndarray): array of coordinates. Shape (N, 2).
        sampling_size (int): size of the sampling points.
        slide_obj (openslide.OpenSlide): OpenSlide object of the SVS file.
        metadata (dict): metadata of the SVS file.
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    if not os.path.isfile(svs_file):
        logger.error(f"SVS file not found: {svs_file}")
        raise FileNotFoundError(f"SVS file not found: {svs_file}")

    slide_obj, slide_metadata = open_svs(svs_file, default_mag=DEFAULT_TILING_CONFIG["reference_mag"])
    logger.debug(f"Successfully opened slide: {svs_file}")
    logger.debug("Collecting sampling points...")
    try:
        sampling_coords, heatmap, thumb = get_slide_samplepoints(
            slide_obj,
            slide_metadata=slide_metadata,
            thumb_size=foreground_config.thumb_size,
            sampling_fraction=foreground_config.sampling_fraction,
            min_tissue_prob=foreground_config.min_tissue_prob,
            bandwidth=foreground_config.bandwidth,
            min_foreground_ratio=foreground_config.min_foreground_ratio,
            opening_kernel=foreground_cleanup_config.opening_kernel,
            square_fill_kernel=foreground_cleanup_config.square_fill_kernel,
            square_dilation_kernel=foreground_cleanup_config.square_dilation_kernel,
            edge_margin=foreground_cleanup_config.edge_margin,
            max_width_ratio=foreground_cleanup_config.max_width_ratio,
            max_height_ratio=foreground_cleanup_config.max_height_ratio,
            tile_size=tiling_config.tile_size,
            tile_overlap=tiling_config.tile_overlap,
            patch_foreground_ratio=tiling_config.patch_foreground_ratio,
            reference_mag=tiling_config.reference_mag,
            return_heatmap=True,
            return_thumb=True,
        )
        logger.info(f"Successfully extracted sampling points from slide: {svs_file}")

        slide_mag = slide_metadata["mag"]    
        sampling_size = np.around(float(tiling_config.tile_size) * (slide_mag / tiling_config.reference_mag))
        sampling_size = int(sampling_size)
        logger.debug(f"Sampling size from high-resolution whole slide image: {sampling_size}")

    except Exception as e:
        logger.error(f"Failed to process slide: {svs_file}\n{e}")
        raise e

    return sampling_coords, slide_obj, slide_metadata, sampling_size, heatmap, thumb


def load_dino_encoder(model_path):
    # model_path = join(RESOURCE_DIR, "ViT", 'vit256_small_dino.pth')        
    return None
    # model256 = get_vit256(pretrained_weights=model_path, device=device256)
    # return model256.to(device)


def load_uni_encoder(model_path, config_files):
    """ Load the UNI2-h model with predefined timm_kwargs.
    Default timm_kwargs are loaded from DefaultUni2hConfig. Default transform:
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    Args:
        model_path (str): path to the model file.
        config_files (dict): dictionary of config files.
    
    Returns:
        model (torch.nn.Module): loaded model.
        transform: image transform.
    """
    model_name = "hf-hub:MahmoodLab/UNI2-h"
    model = None
    transform = None
    timm_kwargs = {}
    for config_name, config_path in config_files.items():
        if "timm" in config_name:
            try:
                if config_path.endswith(".json"):
                    with open(config_path, "r") as f:
                        timm_kwargs = json.load(f)
                elif config_path.endswith(".yaml"):
                    with open(config_path, "r") as f:
                        timm_kwargs = yaml.safe_load(f)
                else:
                    logger.warning(f"Unexpected config file type : {config_path}. Using defaults.")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}. Error: {e}. Using defaults.")
    
    if not timm_kwargs:
        logger.debug("No valid config files given. Using default timm_kwargs.")
        timm_kwargs = DefaultUni2hConfig().dict().copy()
    else:
        default_config =  DefaultUni2hConfig().dict()
        default_config.update(timm_kwargs)
        timm_kwargs = default_config.copy()
    
    try:
        logger.debug("Loading model with timm_kwargs:")
        for key, value in timm_kwargs.items():
            logger.debug(f"{key}: {value}")
        model = timm.create_model(**timm_kwargs)        
    except Exception as e:
        logger.warning(f"Failed to create UNI2-h model with given kwargs: {e}")
        raise e
    
    try:
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False),
            strict=True
            )
        logger.info(f"Loaded UNI2-h model weights from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load UNI2-h model weights from {model_path}: {e}")
        raise e
    
    try:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    except Exception as e:
        logger.error(f"Failed to create transform: {e}")
        raise e

    return model, transform


def load_resnet_encoder(model_path, config_files, resnet_type):
    """ Load a pre-trained ResNet model with parameters defined in config_files.
    
    Args:
        model_path (str): path to the model file.
        config_files (dict): dictionary of config files.
        resnet_type (str): type of the ResNet model.
    
    Returns:
        model (torch.nn.Module): loaded model.
        transform (callable): image transform.
    """
    model = None
    transform = None
    try:
        model = timm.create_model(f'resnet{resnet_type}', pretrained=True)
        if model_path and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False), strict=False)
            logger.info(f"Loaded ResNet{resnet_type} model weights from {model_path}")
        else:
            logger.info("Using pre-trained ResNet model weights.")
        transform = create_transform(**resolve_data_config({}, model=model))
            
    except Exception as e:
        logger.error(f"Failed to load ResNet{resnet_type} model: {e}")
        raise e
    return model, transform


def load_encoder(model_type, model_path, config_files):
    """
    Load the encoder model based on the given model type.
    
    Args:
        model_type (str): The type of the model.
        model_path (str): The path to the model file.
        config_files (dict): The config files.
        
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = None
    transform = None
    model_type = model_type.lower().strip()

    if model_type == "uni":
        if model_path is None:
            logger.error("Cannot load UNI2-h model without pre-downloaded model.")
            raise ValueError("Model path is None, must be given for UNI2-h feature extractor.")
        model, transform = load_uni_encoder(model_path, config_files)

    elif model_type.startswith("resnet"):
        resnet_version = model_type.replace("resnet", "").strip()
        if resnet_version not in ["50", "101"]:
            logger.warning("Unsupported ResNet version. Defaulting to ResNet50.")
            resnet_version = "50"
        
        model, transform = load_resnet_encoder(model_path, config_files, resnet_version)
    
    # TODO - Load SimCLR tensorflow model somehow
    elif model_type == "simclr":
        logger.warning("SimCLR model not supported yet.")
        raise NotImplementedError("SimCLR model not supported yet.")

    else:
        logger.warning("Model type not supported. Using ResNet50 as default.")
        model, transform = load_resnet_encoder(model_path, config_files, "50")

    return model, transform


def extract_features(slide_dataset, model, device, num_workers=1, batch_size=256, prefetch_factor=2):
    """
    Extract features from the slide dataset using the given model.
    
    Args:
        slide_dataset (Dataset): The slide dataset.
        model (torch.nn.Module): The model to use for feature extraction.
        device (torch.device): The device to use for feature extraction.
        num_workers (int, optional): The number of workers to use for data loading. Defaults to 1.
        batch_size (int, optional): The batch size to use for data loading. Defaults to 256.
    
    Returns:
        features_array (np.ndarray): The extracted features of shape (len(slide_dataset), feature_dim).
    """
    dataloader = DataLoader(
        slide_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    logger.debug(f"DataLoader initialized with {num_workers} workers and batch size {batch_size}")

    torch.backends.mkldnn.enabled = True
    model.eval()
    if device.type == "cpu":
        try:
            model = torch.jit.script(model)
            logger.info("JIT optimization enabled.")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}. Running model without JIT.")

    logger.debug('Extracting features...')
    n_batches = len(slide_dataset) // batch_size + 1
    count = 0
    start_time = time.time()

    feature_dim = model(torch.randn(1, *slide_dataset[0].shape).to(device)).shape[1]
    start_idx = 0
    features = np.empty((len(slide_dataset), feature_dim), dtype=np.float32)
    with torch.inference_mode():
        for count, batch_images in enumerate(dataloader):
            batch_images = batch_images.to(device, dtype=torch.float32)
            batch_features = model(batch_images).cpu().numpy().astype(np.float32)

            end_idx = start_idx + batch_images.shape[0]
            features[start_idx:end_idx, :] = batch_features
            start_idx = end_idx
            
            if count % 10 == 0:
                current_time = time.time()
                logger.debug(f"({(current_time - start_time)/60:.2f}m) -Processed batch {count}/{n_batches} ({(count/n_batches)*100:.2f}%)")
    return features


def get_COO_coords(coords, sampling_size, tile_overlap):
    """
    Convert sampling coordinates into COO-style indices.

    Args:
        coords (np.ndarray): Array of original coordinates from high-res WSI.
        sampling_size (int): Size of each tile in pixels.
        tile_overlap (float): Fractional tile overlap (e.g., 0.1 for 10% overlap).

    Returns:
        np.ndarray: Adjusted coordinate indices for unique feature mapping.
    """
    # Normalize coordinates so the lowest point is (0,0)
    coords = coords.astype(float) - np.min(coords, axis=0)
    step_size = np.around(sampling_size * (1 - tile_overlap)).astype(int)
    scaled_coords = coords // step_size
    return scaled_coords



def save_features_zarr(output_path, coo_coords, slide_features, chunk_size=10000, compressor=None):
    """
    Saves COO-style coordinates and feature vectors to a Zarr store.

    Args:
        output_path (str): Path to save Zarr file (can be local or S3).
        coo_coords (np.ndarray): (N, 2) array of (row, col) coordinates.
        slide_features (np.ndarray): (N, D) feature vectors.
        chunk_size (int): Chunk size for Zarr storage (default: 10000).
        compressor (str): Compression algorithm (default: 'blosc').
    """
    if compressor is None:
        compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    root = zarr.open(output_path, mode='w')
    logger.debug(f"Opened Zarr store at {output_path}")
    try:
        root.create_dataset(
                "coords",
                data=coo_coords,
                dtype="int32",
                chunks=(chunk_size, 2),
                compressor=compressor
            )
    except Exception as e:
        logger.error(f"Failed to create dataset 'coords' in Zarr store: {e}")
        raise e
    try:
        root.create_dataset(
            "features",
            data=slide_features,
            dtype="float32",
            chunks=(chunk_size, slide_features.shape[1]), 
            compressor=compressor
        )
    except Exception as e:
        logger.error(f"Failed to create dataset 'features' in Zarr store: {e}")
        raise e
    