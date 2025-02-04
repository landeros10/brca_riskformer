import os
import logging
from os.path import abspath, join
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.transform import resize, rescale
from skimage.filters import threshold_triangle

from histomicstk.segmentation import simple_mask
from openslide import OpenSlide

from src.logger_config import logger_setup
from src.util import collect_patients_svs_files

logger_setup()
logger = logging.getLogger(__name__)

FOREGROUND_PARAMS = {
    "fraction": 0.05,
    "min_tissue_prob": 0.01,
    "fill": 25,
    "dil": 0,
}

FOREGROUND_PARAMS_SUPERVISED = {
    "fraction": 0.25,
    "bandwidth": 1,
    "min_tissue_prob": 0.05
}

CLEANUP_PARAMS = {
    "open": 15,
    "fill": 20,
    "dil": 5
}

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

MAG = 20.0

HOME_DIR = './'
RESOURCE_DIR = abspath(join(HOME_DIR, "resources"))
FOREGROUND_DIR = join(RESOURCE_DIR, "foregrounds")

SVS_FILES = np.load(join(RESOURCE_DIR, "svs_files.npy"))
PATIENT_FILE = join(RESOURCE_DIR, "n0samples.csv")
PATIENT_FILE_2 = join(RESOURCE_DIR, "n1samples.csv")

SLIDES_PRS = collect_patients_svs_files(PATIENT_FILE, SVS_FILES)
SLIDES_PRS_DATA = pd.read_csv(PATIENT_FILE)

SLIDES_PRS_2 = collect_patients_svs_files(PATIENT_FILE_2, SVS_FILES)
SLIDES_PRS_DATA_2 = pd.read_csv(PATIENT_FILE_2)

THRESH_PARAMS = {"fraction": 0.5, "bandwidth": 1, "min_tissue_prob": 0.05}
CLEANUP_PARAMS = {"open": 5, "fill": 25, "dil": 0, "border": 50}


def open_svs(svs_file):
    """
    Open SVS file and extract metadata.
    
    Args:
        svs_file (str): path to SVS file.
    
    Returns:
        slideObj (OpenSlide): OpenSlide object.
        metadata (dict): dictionary of metadata.
    """
    slideObj = OpenSlide(svs_file)
    mag = float(slideObj.properties["aperio.AppMag"].replace(",", "."))
    full_dims = np.array(slideObj.level_dimensions[0]).astype(float)[::-1] # (rows, cols)

    metadata = {
        "file": svs_file,
        "mag": mag,
        "full_dims": full_dims,
    }
    return slideObj, metadata


def process_svs_foregrounds(svs_files):
    pass


def eval_slide_samples(svs_file, tiling_params, resource_dir, foreground_mask, desired_mag):
    slideObj, metadata = open_svs(svs_file)

    true_dim = metadata["full_dims"][0] # get height of slide
    fg_scale = true_dim / float(foreground_mask.shape[0])
    slide_mag = metadata["mag"]

    coords, crop_size, heatmap = get_slide_samplepoints(
        slideObj, metadata, tiling_params, desired_mag, foreground_mask, return_heatmap=True)
    slideObj.close()
    return slideObj, heatmap, coords, fg_scale, crop_size


def make_foreground_mask(image, fraction=0.25, bandwidth=1,
                         min_tissue_prob=0.05, pThresh=0.25,
                         fill=15, dil=3):
    """ Take RGB image (H x W x C) and produce foreground mask using publicly
    available histomicsTK library.

    Returns: (H x W) boolean mask indicative of tissue"""
    if isinstance(image, Image.Image):
        shape = image.size[::-1]
    elif isinstance(image, np.ndarray):
        shape = image.shape[:2]

    mask = None
    pfill = 0.0
    count = 0

    while mask is None or pfill < pThresh:
        count += 1
        if count > 3:
            mask = color.rgb2gray(image)
            mask = mask <= threshold_triangle(mask)
            break
        try:
            mask = simple_mask(image,
                               bandwidth=bandwidth, fraction=fraction,
                               min_tissue_prob=min_tissue_prob)
            pfill = mask.sum() / (mask.size)
        except:
            pThresh = max(pThresh - 0.1, 0.0)
            fraction = min(fraction + 0.2, 0.95)
    return mask


def get_slide_samplepoints(slideObj, metadata, tiling_params, desired_mag, foreground_mask,
                           return_heatmap=False):
    tile_size = tiling_params.get("size", 256)
    tile_overlap = tiling_params.get("overlap", 0.75)
    p_foreground = tiling_params.get("p_foreground", 0.5)

    slide_mag = metadata["mag"]
    query_size = np.around(float(tile_size) * (slide_mag / desired_mag))
    query_size = int(query_size)

    true_dim = metadata["full_dims"][0]
    fg_scale = true_dim / float(foreground_mask.shape[0])

    # Get slide bounding box
    fg_bbox = get_bbox(foreground_mask)
    slide_bbox = np.around(fg_bbox.astype(float) * fg_scale).astype(int)
    
    # Generate sampling coordinates
    coords = bbox_to_coords(slide_bbox, tile_size, overlap=tile_overlap)
    coords = filter_coords_mask(coords, foreground_mask, fg_scale, tile_size,
                                p_foreground=p_foreground)


    if len(coords) > 0:
        coords = coords[:, [1, 0]]  # Swap row and cols to match OpenSlide
    else:
        logger.warning(f"No sampling coords could be generated for slide: {metadata['file']}")

    return coords, query_size


def filter_coords_mask(coords, foreground, fg_scale, size, p_foreground=0.1):
    if len(coords) == 0:
        return coords
    
    to_keep = []
    N = len(coords)
    coords = np.array(coords).astype(float)

    # To evaluate % foreground cover, we need to take a smaller window in
    # the downsampled foreground mask
    size_fg = int(np.around(float(size) / fg_scale))
    # If the scaled window is too small, upsample the foreground by 2
    if size_fg < 8:
        foreground = rescale(foreground, 4)
        fg_scale = fg_scale / 4
        size_fg = int(np.around(float(size) / fg_scale))

    ri = np.clip(
        np.floor(coords[:, 0] / fg_scale).astype(int),
        0,
        foreground.shape[0] - size_fg)
    ci = np.clip(
        np.floor(coords[:, 1] / fg_scale).astype(int),
        0,
        foreground.shape[1] - size_fg)

    row_indices = ri[:, None, None] + np.arange(size_fg)[:, None]
    col_indices = ci[:, None, None] + np.arange(size_fg)[None, :]

    windows = foreground[row_indices, col_indices]
    frac_values = windows.sum(axis=(1, 2), keepdims=False) / (size_fg * size_fg)
    to_keep = frac_values >= p_foreground
    return coords[to_keep, :]


def get_bbox(mask):
    """
    Get bounding box of the foreground in the mask
    
    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    
    Returns:
        bbox (list): array of bounding box coordinates. [min_row, max_row, min_col, max_col].
         Returns empty list if no foreground detected.
    """
    if mask.sum() == 0:
        logger.warning("No foreground detected in mask")
        return []

    bbox = [np.nonzero(mask.max(axis=i))[0][[0, -1]] for i in range(1, -1, -1)]
    return np.concatenate(bbox)


def bbox_to_coords(bbox, size, overlap=0.0):
    """
    Generate coordinates from bounding box
    
    Args:
        bbox (list): list of bounding box coordinates. [min_row, max_row, min_col, max_col].
        size (int): size of the window.
        overlap (float): overlap of the windows.
    
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
            Returns empty array if bounding box is empty.
    """
    if len(bbox) == 0:
        return np.array([])
    
    delta = np.around(size * (1 - overlap)).astype(int)
    
    min_row, max_row, min_col, max_col = bbox
    row_points = np.arange(min_row, max_row, delta)
    col_points = np.arange(min_col, max_col, delta)
    return np.array(np.meshgrid(row_points, col_points)).T.reshape(-1, 2)




# def pad_and_split_features(labeled_mask, features):
#     mask_shape = labeled_mask.shape
#     padded_features = np.zeros((*mask_shape, features.shape[1]))

#     binary_mask = (labeled_mask > 0).astype(int)
#     pos_coords = np.where(binary_mask == 1)
#     pos_coords = np.column_stack(pos_coords)
#     for index, coord in enumerate(pos_coords):
#         padded_features[coord[0], coord[1]] = features[index]
#     # return [padded_features]

#     # List to store split feature regions
#     split_feature_regions = []

#     # For each label in the labeled mask, extract the respective feature region
#     for region in regionprops(labeled_mask):

#         # Get bounding box of the region
#         min_row, min_col, max_row, max_col = region.bbox

#         # Extract the corresponding feature region
#         feature_region = np.zeros((max_row-min_row, max_col-min_col, features.shape[1]))

#         # Copy the features only in the positions of the region label
#         for row in range(min_row, max_row):
#             for col in range(min_col, max_col):
#                 if labeled_mask[row, col] == region.label:
#                     feature_region[row-min_row, col-min_col] = padded_features[row, col]

#         split_feature_regions.append(feature_region)
#     return split_feature_regions

# def get_svs_mag(svs_file):
#     slideObj = OpenSlide(svs_file)
#     mag = float(slideObj.properties["aperio.AppMag"].replace(",", "."))
#     slideObj.close()
#     return mag

# def process_coordinates(coords, svs_file, mag, ps, crop_size):
#     """Format and process coordinates."""
#     slide_info = ["file", "mag", "patch_size", "crop_size"]

#     df = pd.DataFrame(coords.copy(), columns=["x", "y"])
#     df[slide_info] = [svs_file, mag, ps, crop_size]

#     df = df.astype({
#         'x': np.uint32,
#         'y': np.uint32,
#         'mag': np.uint8,
#         'patch_size': np.uint16,
#         'crop_size': np.uint16
#     }).reset_index(drop=True)

#     return df
