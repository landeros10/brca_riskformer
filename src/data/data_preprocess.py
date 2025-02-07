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
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torchvision import transforms # type: ignore
from torch.utils.data import Dataset, DataLoader

from src.logger_config import logger_setup
from src.data.data_utils import (open_svs, get_slide_samplepoints, get_crop_size,
                                 sample_slide)

logger = logging.getLogger(__name__)


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
        image = sample_slide(self.slide_obj, x, y, self.sample_size)
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
    
    slide_obj, metadata = open_svs(svs_file)
    sampling_size = get_crop_size(metadata, tiling_config)
    logger.debug(f"Processing slide:\n{svs_file}")
    try:
        sampling_coords, heatmap = get_slide_samplepoints(
            slide_obj, metadata,
            tiling_config,
            foreground_config,
            foreground_cleanup_config,
            return_heatmap=return_heatmap,
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