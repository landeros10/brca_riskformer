'''
data_preprocess.py

SVS preprocessing functions
Author: landeros10
Created: 2025-02-05
'''
import logging
import argparse

import time
import os
import json
import yaml
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pydantic import BaseModel, Field

import numpy as np
from PIL import Image

import torch
from torchvision import transforms # type: ignore
from torch.utils.data import Dataset, DataLoader
import timm # type: ignore
from timm.data import resolve_data_config # type: ignore
from timm.data.transforms_factory import create_transform # type: ignore
from src.randstainna import RandStainNA

from src.logger_config import logger_setup
from src.data.data_utils import (open_svs, get_slide_samplepoints, get_crop_size,
                                 extract_slide_image)

logger = logging.getLogger(__name__)

yaml_file = '/home/ubuntu/notebooks/cpc_hist/src/CRC_LAB_randomTrue_n0.yaml'
stain_augmentor = RandStainNA(yaml_file, std_hyper=-0.0)
stain_normalizer = RandStainNA(yaml_file, std_hyper=-1.0)


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
    tile_overlap: float = Field(default=0.75, description="Overlap between tiles.")
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
    mlp_layer: str = Field(default="timm.layers.SwiGLUPacked", description="MLP layer")
    act_layer: str = Field(default="torch.nn.SiLU", description="Activation layer")
    reg_tokens: int = Field(default=8, description="Number of reg tokens")
    dynamic_img_size: bool = Field(default=True, description="Dynamic image size")


DEFAULT_FOREGROUND_CONFIG = ForegroundConfigSchema().dict()
DEFAULT_FOREGROUND_CLEANUP_CONFIG = ForegroundCleanupConfigSchema().dict()
DEFAULT_TILING_CONFIG = TilingConfigSchema().dict()
DEFAULT_UNI_CONFIG = DefaultUni2hConfig().dict()


class SingleSlideDataset(Dataset):
    """
    PyTorch dataset for a single slide at specified sample points.
    
    Args:
        slide_obj (openslide.OpenSlide): OpenSlide object of the slide.
        slide_metadata (dict): metadata of the slide.
        sample_coords (np.ndarray): array of sample coordinates. Shape (N, 2).
        sample_size (int): size of square patch to sample.
        output_size (int): size of the output images.
        transform (callable, optional): transform to apply to the images. Must return a tensor.
        
    Example:
        dataset = SingleSlideDataset(slide_obj, slide_metadata, sample_coords, sample_size, output_size)
        first_image = dataset[0]
    """
    def __init__(
            self,
            slide_obj,
            slide_metadata: dict,
            sample_coords: np.ndarray,
            sample_size: int,
            output_size: int,
            transform=None,
        ):
        self.slide_obj = slide_obj
        self.slide_metadata = slide_metadata
        self.sample_coords = sample_coords
        self.sample_size = sample_size
        self.output_size = output_size
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sample_coords)

    def __getitem__(self, idx):
        x, y = self.sample_coords[idx]
        image = self.sample_slide(x, y)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image) # (C, H, W), scaled to [0, 1]

        return image

    def sample_slide(self, x, y):
        """
        Samples a slide at the given coordinates.
        
        Args:
            x (int): x coordinate of the sample.
            y (int): y coordinate of the sample.
        
        Returns:
            image (PIL.Image): sampled image.
        """
        image = extract_slide_image(self.slide_obj, x, y, self.sample_size)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.resize((self.output_size, self.output_size), Image.BILINEAR)
        return image


def get_svs_samplepoints(svs_file, tiling_config, foreground_config, foreground_cleanup_config, return_heatmap=False):
    """
    Extracts sampling points from the SVS file based on the foreground mask.
    
    Args:
        svs_file (str): path to SVS file.
        tiling_config (dict): dictionary of tiling parameters.
        foreground_config (dict): dictionary of foreground parameters.
        foreground_cleanup_config (dict): dictionary of foreground cleanup parameters.
        return_heatmap (bool): whether to return heatmap of the sampling points.
    
    Returns:
        sampling_coords (np.ndarray): array of coordinates. Shape (N, 2).
        sampling_size (int): size of the sampling points.
        slide_obj (openslide.OpenSlide): OpenSlide object of the SVS file.
        metadata (dict): metadata of the SVS file.
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    if not os.path.isfile(svs_file):
        logger.error(f"SVS file not found: {svs_file}")
        return np.empty((0, 2), dtype=int), None
    
    slide_obj, metadata = open_svs(svs_file, default_mag=DEFAULT_TILING_CONFIG["reference_mag"])

    slide_mag = metadata.get("mag", DEFAULT_TILING_CONFIG["reference_mag"])
    tile_size = tiling_config.get("size", DEFAULT_TILING_CONFIG["size"])
    reference_mag = tiling_config.get("reference_mag", DEFAULT_TILING_CONFIG["reference_mag"])
    sampling_size = get_crop_size(slide_mag, reference_mag, tile_size)


    logger.debug(f"Processing slide:\n{svs_file}")
    try:
        sampling_coords, heatmap = get_slide_samplepoints(
            slide_obj,
            slide_metadata=metadata,
            thumb_size=foreground_config.get("thumb_size", DEFAULT_FOREGROUND_CONFIG["thumb_size"]),
            fraction=foreground_config.get("sampling_fraction", DEFAULT_FOREGROUND_CONFIG["sampling_fraction"]),
            min_tissue_prob=foreground_config.get("min_tissue_prob", DEFAULT_FOREGROUND_CONFIG["min_tissue_prob"]),
            bandwidth=foreground_config.get("bandwidth", DEFAULT_FOREGROUND_CONFIG["bandwidth"]),
            min_foreground_ratio=foreground_config.get("min_foreground_ratio", DEFAULT_FOREGROUND_CONFIG["min_foreground_ratio"]),
            opening_kernel=foreground_cleanup_config.get("opening_kernel", DEFAULT_FOREGROUND_CLEANUP_CONFIG["opening_kernel"]),
            edge_margin=foreground_cleanup_config.get("edge_margin", DEFAULT_FOREGROUND_CLEANUP_CONFIG["edge_margin"]),
            max_width_ratio=foreground_cleanup_config.get("max_width_ratio", DEFAULT_FOREGROUND_CLEANUP_CONFIG["max_width_ratio"]),
            max_height_ratio=foreground_cleanup_config.get("max_height_ratio", DEFAULT_FOREGROUND_CLEANUP_CONFIG["max_height_ratio"]),
            square_fill_kernel=foreground_cleanup_config.get("square_fill_kernel", DEFAULT_FOREGROUND_CLEANUP_CONFIG["square_fill_kernel"]),
            square_dilation_kernel=foreground_cleanup_config.get("square_dilation_kernel", DEFAULT_FOREGROUND_CLEANUP_CONFIG["square_dilation_kernel"]),
            tile_size=tiling_config.get("tile_size", DEFAULT_TILING_CONFIG["tile_size"]),
            reference_mag=tiling_config.get("reference_mag", DEFAULT_TILING_CONFIG["reference_mag"]),
            tile_overlap=tiling_config.get("tile_overlap", DEFAULT_TILING_CONFIG["tile_overlap"]),
            patch_foreground_ratio=tiling_config.get("patch_foreground_ratio", DEFAULT_TILING_CONFIG["patch_foreground_ratio"]),
            return_heatmap=False,
        )
    except Exception as e:
        logger.error(f"Failed to process slide: {svs_file}\n{e}")
        return (np.empty((0, 2), dtype=int), None)

    return sampling_coords, sampling_size, slide_obj, metadata, heatmap


def get_all_samplepoints(
        svs_files,
        tiling_config,
        foreground_config,
        foreground_cleanup_config,
        return_heatmap=False,
        parallel=False,
    ):
    """
    Get sampling points for all SVS files in the list svs_files.

    Args:
        svs_files (list): list of SVS file paths.
        tiling_config (dict): dictionary of tiling parameters.
        foreground_config (dict): dictionary of foreground parameters.
        foreground_cleanup_config (dict): dictionary of foreground cleanup parameters.
        return_heatmap (bool): whether to return heatmap of the sampling points.
    
    Returns:
        all_coords (dict): dictionary of sampling points for each SVS file.
        all_heatmaps (dict): dictionary of heatmaps for the sampling points. Empty dict if not requested.
    """

    start_time = time.time()
    num_workers = min(cpu_count(), len(svs_files))
    logger.info(f"Processing {len(svs_files)} slides using {min(cpu_count(), len(svs_files))} workers.")

    all_coords = {}
    all_heatmaps = {}

    if parallel:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.starmap(get_svs_samplepoints, [(f, tiling_config, foreground_config, foreground_cleanup_config, return_heatmap) for f in svs_files]),
                total=len(svs_files),
            ))
    else:
        results = []
        for f in tqdm(svs_files):
            results.append(get_svs_samplepoints(f, tiling_config, foreground_config, foreground_cleanup_config, return_heatmap)) 

    for svs_file, (coords, heatmap) in zip(svs_files, results):
        all_coords[svs_file] = coords
        if return_heatmap:
            all_heatmaps[svs_file] = heatmap
    
    logger.info(f"Finished processing all slides in {time.time() - start_time:.2f}s")
    return all_coords, all_heatmaps


def load_dino_model(model_path):
    # model_path = join(RESOURCE_DIR, "ViT", 'vit256_small_dino.pth')        
    return None
    # model256 = get_vit256(pretrained_weights=model_path, device=device256)
    # return model256.to(device)


def load_uni_model(model_path, config_files):
    """ Load the UNI2-h model with predefined timm_kwargs.
    Default timm_kwargs are loaded from DefaultUni2hConfig. Default transform:
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    Args:
        model_path (str): path to the model file.
        config_files (dict): dictionary of config files.
    
    Returns:
        model: loaded model.
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
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu"),
            strict=True
            )
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
    except Exception as e:
        logger.warning(f"Failed to load UNI2-h mode with given kwargs: {e}")
        raise e

    if model is not None:
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
            )

    return model, transform


def load_resnet_model(model_path, config_files, resnet_type):
    """ Load a pre-trained ResNet model with parameters defined in config_files.
    
    Args:
        model_path (str): path to the model file."""
    model = None
    transform = None
    try:
        model = timm.create_model(f'resnet{resnet_type}', pretrained=True)
        if model_path and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
            logger.info(f"Loaded ResNet{resnet_type} model weights from {model_path}")
        else:
            logger.info("Using pre-trained ResNet model weights.")
        transform = create_transform(**resolve_data_config({}, model=model))
            
    except Exception as e:
        logger.error(f"Failed to load ResNet{resnet_type} model: {e}")
        raise e
    return model, transform


def load_model_from_path(model_type, model_path, config_files):
    model = None
    transform = None
    model_type = model_type.lower().strip()

    if model_type == "uni":
        if model_path is None:
            logger.error("Cannot load UNI2-h model without pre-downloaded model.")
            raise ValueError("Model path is None, must be given for UNI2-h feature extractor.")
        model, transform = load_uni_model(model_path, config_files)

    elif model_type.startswith("resnet"):
        resnet_version = model_type.replace("resnet", "").strip()
        if resnet_version not in ["50", "101"]:
            logger.warning("Unsupported ResNet version. Defaulting to ResNet50.")
            resnet_version = "50"
        
        model, transform = load_resnet_model(model_path, config_files, resnet_version)

    else:
        logger.warning("Model type not supported. Using ResNet50 as default.")
        model, transform = load_resnet_model(model_path, config_files, "50")

    return model, transform

def extract_features(test_file, coords, model, image_size, crop_size, bs=256):
    # TODO review this function
    use_cpu = next(model.parameters()).device.type == "cpu"

    # TODO - replace lambda func
    # dataset = SingleSlideDataset(test_file, coords, lambda x: x, image_size, crop_size)
    # dataloader = DataLoader(dataset, batch_size=bs, num_workers=1)

    # features = []
    # for batch_images in dataloader:
    #     if not use_cpu:
    #         batch_images = batch_images.cuda()
    #     batch_features = model(batch_images)
    #     batch_features = batch_features.detach().cpu().numpy()
    #     features.append(batch_features)

    # features_array = np.concatenate(features, axis=0)
    # dataset.close_slide()
    # return features_array