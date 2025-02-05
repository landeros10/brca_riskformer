'''
author: landeros10
created: 2/25/2025
'''
import logging
import time

import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import resize, rescale
from skimage.filters import threshold_triangle
from skimage.morphology import opening, closing, dilation, square
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from openslide import OpenSlide
from torch.utils.data import Dataset

from histomicstk.segmentation import simple_mask

from src.logger_config import logger_setup
from src.utils import read_slide_

logger_setup()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FOREGROUND_THRESH_PARAMS = {
    "fraction": 0.10,
    "bandwidth": 2,
    "min_tissue_prob": 0.05,
    "min_foreground_ratio": 0.15
}

FOREGROUND_CLEANUP_PARAMS = {
    "open": 3,
    "fill": 15,
    "dil": 1,
    "border": 50
}


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


def filter_coords_mask(coords, foreground, fg_scale, size, p_foreground=0.1):
    if len(coords) == 0:
        logger.warning("No coordinates to filter")
        return coords
    
    to_keep = []
    N = len(coords)
    coords = coords.astype(float)

    # To evaluate % foreground cover, we need to take a smaller window in
    # the downsampled foreground mask
    size_fg = int(np.around(float(size) / fg_scale))
    # If the scaled window is too small, upsample the foreground by 2
    while size_fg < 8:
        logger.debug("Upsampling foreground mask by 4")
        foreground = rescale(foreground, 4)
        fg_scale = fg_scale / 4
        size_fg = int(np.around(float(size) / fg_scale))
    logger.debug(f"Tile size to evaluate in foreground: {size_fg}")
    logger.debug(f"foreground shape: {foreground.shape}")

    ri = np.clip(
        np.floor(coords[:, 0] / fg_scale).astype(int),
        0,
        foreground.shape[0] - size_fg)
    ci = np.clip(
        np.floor(coords[:, 1] / fg_scale).astype(int),
        0,
        foreground.shape[1] - size_fg)
    logger.debug(f"Foreground indices: {ri}, {ci}")

    row_indices = ri[:, None, None] + np.arange(size_fg)[:, None]
    col_indices = ci[:, None, None] + np.arange(size_fg)[None, :]

    windows = foreground[row_indices, col_indices]
    frac_values = windows.sum(axis=(1, 2), keepdims=False) / (size_fg * size_fg)
    to_keep = frac_values >= p_foreground
    return coords[to_keep, :]


def coords_to_heatmap(coords, f, size, shape):
    coords = np.around(coords.astype(float) / f).astype(int)
    heatmap = np.zeros(shape).astype(float)

    sc_size = int(np.around(float(size) / f))
    for (ci, ri) in coords:
        window = heatmap[ri:(ri + sc_size), ci:(ci + sc_size)]
        heatmap[ri:(ri + sc_size), ci:(ci + sc_size)] = window + 1.0
    return heatmap


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


def get_svs_thumb(svs_file, size=None):
    """
    Get low-level thumbnail of the SVS file.
    
    Args:
        svs_file (str): path to SVS file.
    
    Returns:
        image (PIL.Image): thumbnail image.
    """
    slideObj, _ = open_svs(svs_file)
    return get_slide_thumb(slideObj, size=size)


def get_slide_thumb(slideObj, size=None):
    """
    Get low-level thumbnail of the slide.
    
    Args:
        slideObj (OpenSlide): OpenSlide object.
    
    Returns:
        image (PIL.Image): thumbnail image.
    """
    
    levels = slideObj.level_count
    thumbsize = max(slideObj.level_dimensions[-1]) if size is None else size
    logger.debug(f"Loading thumbnail of max size: {thumbsize}")

    image = slideObj.get_thumbnail((thumbsize, thumbsize))
    logger.debug(f"Loaded thumbnail of size: {image.size}")
    return image


def get_hist_foreground(image,
    fraction: float = FOREGROUND_THRESH_PARAMS["fraction"],
    bandwidth: int = FOREGROUND_THRESH_PARAMS["bandwidth"],
    min_tissue_prob: float = FOREGROUND_THRESH_PARAMS["min_tissue_prob"],
    min_foreground_ratio: float = FOREGROUND_THRESH_PARAMS["min_foreground_ratio"]
    ):
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

    logger.debug("Starting mask generation...")
    try:
        logger.debug(f"Using HistomicsTK with min_tissue_prob: {min_tissue_prob}, fraction: {fraction}, bandwidth: {bandwidth}")
        start_time = time.time()
        mask = simple_mask(image, min_tissue_prob=min_tissue_prob, fraction=fraction, bandwidth=bandwidth)
        logger.debug(f"Generated foreground mask with histomicsTK in {time.time() - start_time:.1f}s")
        pfill = mask.sum() / (mask.size)
        if pfill < min_foreground_ratio:
            raise ValueError("pfill is below the threshold")
    except Exception as e:
        logger.debug(f"Failed to generate mask with histomicsTK: {e}")
        logger.debug("Using triangle threshold")
        mask = color.rgb2gray(image)
        mask = mask <= threshold_triangle(mask)
        pfill = mask.sum() / (mask.size)
    
    logger.debug(f"Foreground mask fill: {pfill:.2f}")
    return mask


def remove_large_horizontal_regions(mask, width_percentage=0.9, height_percentage=0.25):
    """
    Remove regions that are wider than a given percentage of the mask width
    and shorter than a given percentage of the mask height.

    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        width_percentage (float): percentage of mask width to consider as wide.
        height_percentage (float): percentage of mask height to consider as short.
    
    Returns:
        mask (np.ndarray): binary cleaned mask. Shape (H, W).
    """
    labeled = label(mask)
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        region_width = maxc - minc
        region_height = maxr - minr

        if region_width > width_percentage * mask.shape[1] and region_height < height_percentage * mask.shape[0]:
            labeled[labeled == region.label] = 0

    return labeled > 0


def mask_clean_up_and_resize(
        mask: np.ndarray,
        open: float = FOREGROUND_CLEANUP_PARAMS["open"],
        fill: float = FOREGROUND_CLEANUP_PARAMS["fill"],
        dil: float = FOREGROUND_CLEANUP_PARAMS["dil"],
        border: float = FOREGROUND_CLEANUP_PARAMS["border"],
        shape: tuple = None
):
    """
    Cleans up binary mask by removing aberrant regions, filling holes.
    Optionally, dilates the mask and resizes it to a given shape.

    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        open (int): size of opening kernel.
        fill (int): size of filling kernel.
        dil (int): size of dilation kernel.
        shape (tuple): shape to resize the mask to.
        border (int): size of border to remove.
    
    Returns:
        mask (np.ndarray): cleaned mask. Shape (H, W).
    """
    start_time = time.time()
    # This was designed for images taken at (1/4) resolution of 20X
    mask = opening(mask, square(open))
    logger.debug(f"Foreground mask fill after opening: {mask.sum() / mask.size:.2f}")

    if mask.astype(int).sum() == 0:
        return mask
    mask = binary_fill_holes(mask)
    logger.debug(f"Foreground mask fill after filling holes: {mask.sum() / mask.size:.2f}")

    mask[border, :] = mask[-border, :] = 0
    logger.debug(f"Foreground mask fill after removing border: {mask.sum() / mask.size:.2f}")
    mask = remove_large_horizontal_regions(mask)
    logger.debug(f"Foreground mask fill after removing large regions: {mask.sum() / mask.size:.2f}")


    if fill > 0:
        mask = closing(mask, square(fill))
        mask = binary_fill_holes(mask)
    if dil > 0:
        mask = dilation(mask, square(dil))
    mask = opening(mask, square(open * 4))

    logger.debug(f"Foreground mask fill after closing and dilation: {mask.sum() / mask.size:.2f}")

    # Check if any border of the mask is fully set to 1
    height, width = mask.shape

    # Top border
    if mask[0, :].sum() >= 0.9 * width:
        mask[:border, :] = 0
    # Bottom border
    if mask[-1, :].sum() >= 0.9 * width:
        mask[-border:, :] = 0
    # Left border
    if mask[:, 0].sum() >= 0.9 * height:
        mask[:, :border] = 0
    # Right border
    if mask[:, -1].sum() >= 0.9 * height:
        mask[:, -border:] = 0

    if shape is not None:
        mask = resize(mask, shape)
    mask = np.ceil(mask).astype(bool)
    logger.debug(f"Foreground mask fill after resizing: {mask.sum() / mask.size:.2f}")
    logger.debug(f"Cleaned mask in {time.time() - start_time:.1f}s")
    return mask


def get_slide_foreground(slideObj, size=None, **kwargs):
    """ 
    Get foreground mask for openslide object.

    Args:
        slideObj (OpenSlide): OpenSlide object.
        thumbsize (int): size of thumbnail to use for mask generation.
    Returns:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    """
    foreground = get_slide_thumb(slideObj, size=size)
    return get_hist_foreground(foreground, **kwargs)


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