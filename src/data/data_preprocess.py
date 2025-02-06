import logging
import argparse
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from openslide import OpenSlide

from torch.utils.data import Dataset, DataLoader

from src.logger_config import logger_setup
from src.data.data_utils import (get_bbox, bbox_to_coords, filter_coords_mask, open_svs,
                      get_slide_foreground, mask_clean_up_and_resize, coords_to_heatmap,
                      SlideDataset)
from src.utils import read_slide_

logger = logging.getLogger(__name__)

REFERENCE_MAG = 20.0
DEFAULT_TILING_SIZE = 256
DEFAULT_FOREGROUND_SIZE = 500
DEFAULT_TISSUE_PROB = 0.15
TILING_PARAMS = {
    256: {
        "size": 256,
        "overlap": 0,
        "p_foreground": 0.125
    },
    4096: {
        "size": 4096,
        "overlap": (4 / 16),
        "p_foreground": 0.25
    }
}

# TODO improve slide dataloading
class SlideDataset(Dataset):
    def __init__(self, slide_path, coords, transform, image_size, crop_size):
        self.slide_path = slide_path
        self.coords = coords
        self.transform = transform
        self.image_size = image_size
        self.crop_size = crop_size
        self.slideObj = OpenSlide(slide_path)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        image = read_slide_(self.slide_path, x, y, self.crop_size, self.image_size)
        image = image.resize((self.image_size, self.image_size))
        transformed_image = self.transform(image)
        return transformed_image

    def close_slide(self):
        self.slideObj.close()


def get_mask_samplepoints(foreground_mask, slide_metadata, tiling_params, reference_mag=REFERENCE_MAG):
    tile_overlap = tiling_params.get("overlap", 0.75)
    p_foreground = tiling_params.get("p_foreground", 0.5)

    slide_mag = slide_metadata["mag"]
    tile_size = tiling_params.get("size", 256)
    crop_size = np.around(float(tile_size) * (slide_mag / reference_mag))
    crop_size = int(crop_size)

    logger.debug("Generating sampling points")
    logger.debug(f"slide mag: {slide_mag}, reference mag: {reference_mag}")
    logger.debug(f"Tile size: {crop_size}, overlap: {tile_overlap}, p_foreground: {p_foreground}")

    true_dim = slide_metadata["full_dims"][0]
    logger.debug(f"True slide dimensions: {slide_metadata['full_dims']}")
    fg_scale = true_dim / float(foreground_mask.shape[0])
    logger.debug(f"Foreground scale: {fg_scale}")

    # Get slide bounding box
    fg_bbox = get_bbox(foreground_mask)
    slide_bbox = np.around(fg_bbox.astype(float) * fg_scale).astype(int)
    logger.debug(f"Foreground bbox: {fg_bbox}, Slide bbox: {slide_bbox}")

    # Generate sampling coordinates
    coords = bbox_to_coords(slide_bbox, crop_size, overlap=tile_overlap)
    logger.debug(f"Unfiltered coords shape: {coords.shape}")
    logger.debug("Filtering sampling coords based on foreground mask")
    start_time = time.time()
    coords = filter_coords_mask(coords, foreground_mask, fg_scale, crop_size,
                                p_foreground=p_foreground)
    logger.debug(f"Filtered coords: {len(coords)} in time: {time.time() - start_time:.1f}s")

    if len(coords) > 0:
        logger.debug(f"Generated {len(coords)} sampling coords")
        coords = coords[:, [1, 0]]  # Swap row and cols to match OpenSlide
    else:
        logger.warning(f"No sampling coords could be generated for slide: {slide_metadata['file']}")
        coords = np.empty((0, 2), dtype=int)

    return coords, crop_size


def get_slide_samplepoints(slideObj, metadata, tiling_params, thumb_size=DEFAULT_FOREGROUND_SIZE, min_tissue_prob=DEFAULT_TISSUE_PROB, return_heatmap=False):
    """
    Extracts sampling points from the slide based on the foreground mask.
    
    Args:
        slideObj (OpenSlide): OpenSlide object.
        metadata (dict): dictionary of slide metadata.
        tiling_params (dict): dictionary of tiling parameters.
        thumb_size (int): size of the thumbnail to use for foreground mask generation.
        min_tissue_prob (float): minimum tissue probability for foreground mask generation.
        return_heatmap (bool): whether to return heatmap of the sampling points.
    
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    start_time = time.time()
    foreground_mask = get_slide_foreground(slideObj, size=thumb_size, min_tissue_prob=min_tissue_prob)
    clean_mask = mask_clean_up_and_resize(foreground_mask)
    logger.debug(f"Generated foreground mask in {time.time() - start_time:.1f}s")

    coords, crop_size = get_mask_samplepoints(
        clean_mask,
        metadata,
        tiling_params,
    )

    heatmap = None
    if return_heatmap:
        true_dim = metadata["full_dims"][0]
        fg_scale = true_dim / float(clean_mask.shape[0])
        heatmap = coords_to_heatmap(coords, fg_scale, crop_size, clean_mask.shape)
    logger.debug(f"Finished processing slide\n{metadata['file']}\nin {time.time() - start_time:.2f}s")
    return coords, heatmap


def get_svs_samplepoints(svs_file, tiling_params, thumb_size=DEFAULT_FOREGROUND_SIZE, min_tissue_prob=DEFAULT_TISSUE_PROB, return_heatmap=False):
    """
    Extracts sampling points from the SVS file based on the foreground mask.
    
    Args:
        svs_file (str): path to SVS file.
        tiling_params (dict): dictionary of tiling parameters.
        thumb_size (int): size of the thumbnail to use for foreground mask generation.
        min_tissue_prob (float): minimum tissue probability for foreground mask generation.
        return_heatmap (bool): whether to return heatmap of the sampling points.
    
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    if not os.path.isfile(svs_file):
        logger.error(f"SVS file not found: {svs_file}")
        return np.empty((0, 2), dtype=int), None
    
    slideObj, metadata = open_svs(svs_file)
    logger.debug(f"Processing slide:\n{svs_file}")
    try:
        coords, heatmap = get_slide_samplepoints(
            slideObj, metadata,
            tiling_params,
            thumb_size=thumb_size,
            min_tissue_prob=min_tissue_prob,
            return_heatmap=return_heatmap,
        )
    except Exception as e:
        logger.error(f"Failed to process slide: {svs_file}\n{e}")
        return (np.empty((0, 2), dtype=int), None)

    return coords, heatmap


def get_all_samplepoints(
        svs_files,
        tiling_params,
        thumb_size=DEFAULT_FOREGROUND_SIZE,
        min_tissue_prob=DEFAULT_TISSUE_PROB,
        return_heatmap=False,
        parallel=False,
    ):
    """
    Get sampling points for all SVS files in the list svs_files.

    Args:
        svs_files (list): list of SVS file paths.
        tiling_params (dict): dictionary of tiling parameters.
        thumb_size (int): size of the thumbnail to use for foreground mask generation.
        min_tissue_prob (float): minimum tissue probability for foreground mask generation.
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
                pool.starmap(get_svs_samplepoints, [(f, tiling_params, thumb_size, min_tissue_prob, return_heatmap) for f in svs_files]),
                total=len(svs_files),
            ))
    else:
        results = []
        for f in tqdm(svs_files):
            results.append(get_svs_samplepoints(f, tiling_params, thumb_size, min_tissue_prob, return_heatmap)) 

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
    dataset = SlideDataset(test_file, coords, lambda x: x, image_size, crop_size)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=1)

    features = []
    for batch_images in dataloader:
        if not use_cpu:
            batch_images = batch_images.cuda()
        batch_features = model(batch_images)
        batch_features = batch_features.detach().cpu().numpy()
        features.append(batch_features)

    features_array = np.concatenate(features, axis=0)
    dataset.close_slide()
    return features_array


def main():
    parser = argparse.ArgumentParser(description="Data loading script")
    parser.add_argument("--reference_mag", type=float, default=20.0, help="Reference Obj magnification that all slides should be scaled to")
    parser.add_argument("--tiling_size", type=int, default=256, help="Tile size")
    parser.add_argument("--titling_params", type=str, default="/data/resources/tiling_params.json", help="Tiling parameters")
    parser.add_argument("--thumb_size", type=int, default=500, help="Thumbnail size to use for foreground calc")
    parser.add_argument("--foreground_params", type=str, default="/data/resources/foreground_params.json", help="HistomicsTK foreground detection parameters")
    parser.add_argument("--model_type", type=str, default="resnet50", help="Model type")

    args = parser.parse_args()

    logger_setup(debug=args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # TODO - make sure preprocessing functions are prepped to work with s3 buckets

    # TODO - load feature extraction model

    # TODO - go through slides and convert patches to features using all_coords
    pass


if __name__ == "__main__":
    main()
