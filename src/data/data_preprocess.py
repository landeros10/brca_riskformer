'''
data_preprocess.py

SVS preprocessing functions
Author: landeros10
Created: 2025-02-05
'''
import logging
import argparse
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from src.logger_config import logger_setup
from src.data.data_utils import (open_svs, get_slide_samplepoints, get_crop_size)
from src.utils import read_slide_

logger = logging.getLogger(__name__)


class SingleSlideDataset(Dataset):
    def __init__(self, slide_path, coords, transform, image_size, crop_size):
        self.slide_path = slide_path
        self.coords = coords
        self.transform = transform
        self.image_size = image_size
        self.crop_size = crop_size
        self.slide_data = open_svs(slide_path)
        self.slide_obj = self.slide_data[0]
        self.slide_metadata = self.slide_data[1]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        image = read_slide_(self.slide_path, x, y, self.crop_size, self.image_size)
        image = image.resize((self.image_size, self.image_size))
        transformed_image = self.transform(image)
        return transformed_image

    def close_slide(self):
        self.slide_obj.close()


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
        coords (np.ndarray): array of coordinates. Shape (N, 2).
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    if not os.path.isfile(svs_file):
        logger.error(f"SVS file not found: {svs_file}")
        return np.empty((0, 2), dtype=int), None
    
    slideObj, metadata = open_svs(svs_file)
    crop_size = get_crop_size(metadata, tiling_config)
    logger.debug(f"Processing slide:\n{svs_file}")
    try:
        coords, heatmap = get_slide_samplepoints(
            slideObj, metadata,
            tiling_config,
            foreground_config,
            foreground_cleanup_config,
            return_heatmap=return_heatmap,
        )
    except Exception as e:
        logger.error(f"Failed to process slide: {svs_file}\n{e}")
        return (np.empty((0, 2), dtype=int), None)

    return coords, crop_size, heatmap


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