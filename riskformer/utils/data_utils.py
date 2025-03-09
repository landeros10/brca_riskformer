'''
data_utils.py
author: landeros10
created: 2/25/2025
'''
import logging
import time
import os

import numpy as np
from PIL import Image
import openslide
from openslide import OpenSlide
from skimage import color
from skimage.transform import resize, rescale
from skimage.filters import threshold_triangle
from skimage.morphology import opening, closing, dilation, square
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from histomicstk.segmentation import simple_mask # type: ignore

logger = logging.getLogger(__name__)


def open_svs(svs_file, default_mag=20.0):
    """
    Open SVS file and extract metadata.
    
    Args:
        svs_file (str): path to SVS file.
    
    Returns:
        slideObj (OpenSlide): OpenSlide object.
        metadata (dict): dictionary of metadata.

    Raises:
        openslide.OpenSlideError: if the SVS file cannot be opened.
        FileNotFoundError: if the SVS file is not found.
    """
    try:
        if not os.path.isfile(svs_file):
            raise FileNotFoundError(f"SVS file not found: {svs_file}")

        slideObj = OpenSlide(svs_file)
        try:
            mag = float(slideObj.properties["aperio.AppMag"].replace(",", "."))
        except Exception as e:
            try:
                mag = float(slideObj.properties.get("openslide.objective-power", default_mag))
            except Exception as e:
                logger.warning(f"Could not parse magnification from SVS file {svs_file}, using default mag {default_mag}: {e}")
                mag = default_mag

        try:
            full_dims = np.array(slideObj.level_dimensions[0]).astype(float)[::-1] # (rows, cols)
        except Exception as e:
            logger.warning(f"Failed to extract slide dimensions: {e}")
            full_dims = (0, 0)

        metadata = {
            "file": svs_file,
            "mag": mag,
            "full_dims": full_dims,
        }
        return slideObj, metadata

    except openslide.OpenSlideError as e:
        logger.error(f"Failed to open SVS file {svs_file}: {e}")
        raise e
    except FileNotFoundError as e:
        logger.error(f"File {svs_file} not found: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e


def sample_slide_image(slide_obj, x, y, sample_size):
    """
    Samples a slide at the given coordinates and returns a PIL image.
    
    Args:
        slide_obj (openslide.OpenSlide): slide object
        x (int): x coordinate of the sample
        y (int): y coordinate of the sample
        sample_size (int): size of the sample

    Returns:
        image (PIL.Image): sampled image
    
    Raises:
        ValueError: if slide_obj is not a valid SVS file path or OpenSlide object
        TypeError: if slide_obj is not an OpenSlide object or SVS file path
    """
    try:
        if isinstance(slide_obj, str):
            slide_obj, _ = open_svs(slide_obj)
        if not isinstance(slide_obj, OpenSlide):
            raise TypeError("slide_obj must be an OpenSlide object or valid SVS file path")
        
        x, y, sample_size = map(int, (x, y, sample_size))
        width, height = slide_obj.level_dimensions[0]
        if sample_size < 16 or sample_size > min(width, height):
            logger.warning(f"Adjusting sample_size from {sample_size} to fit within valid range (16, {min(width, height)})")
        sample_size = max(16, min(sample_size, width, height))

        if not (0 <= x < width and 0 <= y < height):
            logger.warning(f"Coordinates ({x}, {y}) are out of bounds for slide size ({width}, {height}). Adjusting.")
            x, y = np.clip([x, y], 0, [width - sample_size, height - sample_size])
        image = slide_obj.read_region((x, y), 0, (sample_size, sample_size)).convert('RGB')

        if image.size != (sample_size, sample_size):
            logger.warning(f"Sampled image size {image.size}, expected ({sample_size}, {sample_size})")
            padded_image = Image.new('RGB', (sample_size, sample_size), (255, 255, 255))
            paste_x = (sample_size - image.size[0]) // 2
            paste_y = (sample_size - image.size[1]) // 2
            padded_image.paste(image, (paste_x, paste_y))  # Center pad
            image = padded_image
        return image
    
    except (TypeError, ValueError) as e:
        logger.error(f"Error sampling slide: {e}")
        raise
    except openslide.OpenSlideError as e:
        logger.error(f"OpenSlide error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


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


def bbox_to_coords(bbox, size, overlap):
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


def filter_coords_foreground_coverage(coords, foreground, fg_scale, size, p_foreground):
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


def map_coords_to_heatmap(coords, f, size, shape):
    coords = np.around(coords.astype(float) / f).astype(int)
    heatmap = np.zeros(shape).astype(float)

    sc_size = int(np.around(float(size) / f))
    for (ci, ri) in coords:
        window = heatmap[ri:(ri + sc_size), ci:(ci + sc_size)]
        heatmap[ri:(ri + sc_size), ci:(ci + sc_size)] = window + 1.0
    return heatmap


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


def calculate_sampling_size(slide_mag, tile_size, reference_mag):
    """
    Get crop size for the slide based on desired tilsize in the reference magnification.
    
    Args:
        slide_mag (float): magnification of the slide.
        tile_size (int): size of the tile in pixels.
        reference_mag (float): reference magnification.
    
    Returns:
        crop_size (int): size of the crop in pixels.
    """
    crop_size = np.around(float(tile_size) * (slide_mag / reference_mag))
    return int(crop_size)


def get_hist_foreground(image, sampling_fraction, min_tissue_prob, bandwidth, min_foreground_ratio):
    """ Take RGB image (H x W x C) and produce foreground mask using publicly
    available histomicsTK library.

    Args:
        image (np.ndarray): RGB image. Shape (H, W, C).
        sampling_fraction (float): fraction of the image to sample.
        min_tissue_prob (float): minimum tissue probability.
        bandwidth (float): bandwidth for the Gaussian kernel to estimate foreground boundary.
        min_foreground_ratio (float): minimum foreground ratio in the whole slide image.
    Returns:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    """

    if isinstance(image, Image.Image):
        shape = image.size[::-1]
    elif isinstance(image, np.ndarray):
        shape = image.shape[:2]

    mask = None
    pfill = 0.0
    count = 0

    logger.debug("Starting mask generation...")
    try:
        logger.debug(f"Using HistomicsTK with min_tissue_prob: {min_tissue_prob}, fraction: {sampling_fraction}, bandwidth: {bandwidth}")
        start_time = time.time()
        mask = simple_mask(image, min_tissue_prob=min_tissue_prob, fraction=sampling_fraction, bandwidth=bandwidth)
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


def remove_large_horizontal_regions(mask, max_width_ratio, max_height_ratio):
    """
    Remove regions that are wider than a given percentage of the mask width
    and shorter than a given percentage of the mask height.

    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        max_width_ratio (float): percentage of mask width to consider as too wide.
        max_height_ratio (float): percentage of mask height to consider too tall.
    
    Returns:
        mask (np.ndarray): binary cleaned mask. Shape (H, W).
    """
    labeled = label(mask)
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        region_width = maxc - minc
        region_height = maxr - minr

        if region_width > max_width_ratio * mask.shape[1] and region_height < (1 - max_height_ratio) * mask.shape[0]:
            labeled[labeled == region.label] = 0
        elif region_height > max_height_ratio * mask.shape[0] and region_width < (1 - max_width_ratio) * mask.shape[1]:
            labeled[labeled == region.label] = 0

    return labeled > 0


def mask_clean_up_and_resize(
        mask: np.ndarray,
        opening_kernel: int,
        square_fill_kernel: int,
        square_dilation_kernel: int,
        edge_margin: int,
        max_width_ratio: float,
        max_height_ratio: float,
        reshape_size: tuple = None,

):
    """
    Cleans up binary mask by removing aberrant regions, filling holes.
    Optionally, dilates the mask and resizes it to a given shape.

    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        opening_kernel (int): size of the kernel to use for opening the mask.
        edge_margin (int): margin to remove from the edges of the mask.
        max_width_ratio (float): percentage of mask width to consider as too wide.
        max_height_ratio (float): percentage of mask height to consider too tall.
        square_fill_kernel (int): size of the kernel to use for closing the mask.
        square_dilation_kernel (int): size of the kernel to use for dilation.
        reshape_size (tuple): shape to resize the mask to. If None, no resizing is done.

    Returns:
        mask (np.ndarray): cleaned mask. Shape (H, W).
    """
    start_time = time.time()
    mask = opening(mask, square(opening_kernel))
    logger.debug(f"Foreground mask fill after opening: {mask.sum() / mask.size:.2f}")

    if mask.astype(int).sum() == 0:
        return mask
    mask = binary_fill_holes(mask)
    logger.debug(f"Foreground mask fill after filling holes: {mask.sum() / mask.size:.2f}")

    mask[edge_margin, :] = mask[-edge_margin, :] = 0
    logger.debug(f"Foreground mask fill after removing border: {mask.sum() / mask.size:.2f}")
    mask = remove_large_horizontal_regions(mask, max_width_ratio, max_height_ratio)
    logger.debug(f"Foreground mask fill after removing large regions: {mask.sum() / mask.size:.2f}")


    if square_fill_kernel > 0:
        mask = closing(mask, square(square_fill_kernel))
        mask = binary_fill_holes(mask)
    if square_dilation_kernel > 0:
        mask = dilation(mask, square(square_dilation_kernel))
    mask = opening(mask, square(opening_kernel * 4))

    logger.debug(f"Foreground mask fill after closing and dilation: {mask.sum() / mask.size:.2f}")

    # Check if any border of the mask is fully set to 1
    height, width = mask.shape

    # Top border
    if mask[0, :].sum() >= 0.9 * width:
        mask[:edge_margin, :] = 0
    # Bottom border
    if mask[-1, :].sum() >= 0.9 * width:
        mask[-edge_margin:, :] = 0
    # Left border
    if mask[:, 0].sum() >= 0.9 * height:
        mask[:, :edge_margin] = 0
    # Right border
    if mask[:, -1].sum() >= 0.9 * height:
        mask[:, -edge_margin:] = 0

    if reshape_size is not None:
        mask = resize(mask, reshape_size)
    mask = np.ceil(mask).astype(bool)
    logger.debug(f"Foreground mask fill after resizing: {mask.sum() / mask.size:.2f}")
    logger.debug(f"Cleaned mask in {time.time() - start_time:.1f}s")
    return mask


def get_slide_foreground(
        slideObj: OpenSlide, 
        thumb_size: int, 
        sampling_fraction: float, 
        min_tissue_prob: float, 
        bandwidth: int,
        min_foreground_ratio: float,
        return_thumb: bool = False,
    ):
    """ 
    Get foreground mask for openslide object.

    Args:
        slideObj (OpenSlide): OpenSlide object.
        thumb_size (int): size of the whole slide image thumbnail to use for calculating foreground mask.
        sampling_fraction (float): fraction of the image to sample.
        min_tissue_prob (float): minimum tissue probability.
        bandwidth (float): bandwidth for the Gaussian kernel to estimate foreground boundary.
        min_foreground_ratio (float): minimum foreground ratio in the whole slide image.
        return_thumb (bool): whether to return thumbnail of the slide.
    
    Returns:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    """

    thumb = get_slide_thumb(slideObj, size=thumb_size)
    foreground = get_hist_foreground(thumb, sampling_fraction, min_tissue_prob, bandwidth, min_foreground_ratio)
    if return_thumb:
        return foreground, thumb
    return foreground


def get_mask_samplepoints(
        foreground_mask: np.ndarray,
        whole_slide_height: int,
        sampling_size: int,
        tile_overlap: float,
        patch_foreground_ratio: float,
):
    """
    Get sampling points from the foreground mask with given tile size, overlap ratio, and minimum foreground ratio.
    
    Args:
        foreground_mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        whole_slide_height (int): height of the whole slide image.
        sampling_size (int): size of the sampling window at highest resolution.
        tile_overlap (float): overlap ratio between windows.
        patch_foreground_ratio (float): minimum foreground ratio in the sampling window.
    
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
    """
    logger.debug(f"True slide dimensions: {whole_slide_height}")
    logger.debug(f"Tile size: {sampling_size}, overlap: {tile_overlap}, patch_foreground_ratio: {patch_foreground_ratio}")

    fg_scale = whole_slide_height / float(foreground_mask.shape[0])
    logger.debug(f"Foreground scale: {fg_scale}")

    # Get slide bounding box
    fg_bbox = get_bbox(foreground_mask)
    
    # Handle case when fg_bbox is an empty list (no foreground detected)
    if len(fg_bbox) == 0:
        logger.warning("No valid bounding box found. Returning empty coordinates array.")
        return np.array([], dtype=np.int32).reshape(0, 2)
        
    slide_bbox = np.around(fg_bbox.astype(float) * fg_scale).astype(int)
    logger.debug(f"Foreground bbox: {fg_bbox}, Slide bbox: {slide_bbox}")

    # Generate sampling coordinates
    coords = bbox_to_coords(slide_bbox, sampling_size, overlap=tile_overlap)
    logger.debug(f"Unfiltered coords shape: {coords.shape}")
    logger.debug("Filtering sampling coords based on foreground mask")
    start_time = time.time()
    coords = filter_coords_foreground_coverage(coords, foreground_mask, fg_scale, sampling_size,
                                p_foreground=patch_foreground_ratio)
    logger.debug(f"Filtered coords: {len(coords)} in time: {time.time() - start_time:.1f}s")

    if len(coords) > 0:
        logger.debug(f"Generated {len(coords)} sampling coords")
        coords = coords[:, [1, 0]] # (x, y) -> (col, row)

    return coords


def get_slide_samplepoints(
        slideObj: OpenSlide,
        slide_metadata: dict,
        thumb_size: int,
        sampling_fraction: float,
        min_tissue_prob: float,
        bandwidth: float,
        min_foreground_ratio: float,
        opening_kernel: int,
        square_fill_kernel: int,
        square_dilation_kernel: int,
        edge_margin: int,
        max_width_ratio: float,
        max_height_ratio: float,
        tile_size: int,
        tile_overlap: float,
        patch_foreground_ratio: float,
        reference_mag: float,
        return_heatmap: bool = False,
        return_thumb: bool = False,
):
    """
    Extracts sampling points from the slide based on the foreground mask.
    
    Args:
        slideObj (OpenSlide): OpenSlide object.
        slide_metadata (dict): dictionary of slide metadata.
        thumb_size (int): size of the whole slide image thumbnail to use for calculating foreground mask.
        sampling_fraction (float): fraction of the whole slide image to sample for foreground detection.
        min_tissue_prob (float): minimum tissue probability for foreground detection.
        bandwidth (float): bandwidth for the Gaussian kernel to estimate foreground boundary.
        min_foreground_ratio (float): minimum foreground ratio in the whole slide image.
        opening_kernel (int): size of the kernel to use for opening the foreground mask.
        square_fill_kernel (int): size of the kernel to use for closing the foreground mask.
        square_dilation_kernel (int): size of the kernel to use for dilation.
        edge_margin (int): margin to remove from the edges of the mask.
        max_width_ratio (float): percentage of mask width to consider as too wide for portions that touch mask edge.
        max_height_ratio (float): percentage of mask height to consider too tall for portions that touch mask edge.
        tile_size (int): size of the sampling window at highest resolution.
        tile_overlap (float): overlap ratio between windows.
        patch_foreground_ratio (float): minimum foreground ratio in the sampling window.
        reference_mag (float): reference magnification for the sampling window (sampling window scaled to match this mag).
        return_heatmap (bool): whether to return heatmap of the sampling points.
        return_thumb (bool): whether to return thumbnail of the slide.
            
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    heatmap = None
    thumb = None
    start_time = time.time()
    foreground_mask, thumb = get_slide_foreground(
        slideObj=slideObj,
        thumb_size=thumb_size,
        sampling_fraction=sampling_fraction,
        min_tissue_prob=min_tissue_prob,
        bandwidth=bandwidth,
        min_foreground_ratio=min_foreground_ratio,
        return_thumb=return_thumb,
    )
    logger.info(f"Generated foreground mask in {time.time() - start_time:.1f}s")

    clean_mask = mask_clean_up_and_resize(
        mask=foreground_mask,
        opening_kernel=opening_kernel,
        square_fill_kernel=square_fill_kernel,
        square_dilation_kernel=square_dilation_kernel,
        edge_margin=edge_margin,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio,
    )
    logger.debug(f"Generated foreground mask in {time.time() - start_time:.1f}s")

    slide_mag = slide_metadata["mag"]    
    sampling_size = np.around(float(tile_size) * (slide_mag / reference_mag))
    sampling_size = int(sampling_size)

    logger.debug("Collecting valid sampling points from foreground mask...")
    coords = get_mask_samplepoints(
        clean_mask,
        whole_slide_height=slide_metadata["full_dims"][0],
        sampling_size=sampling_size,
        tile_overlap=tile_overlap,
        patch_foreground_ratio=patch_foreground_ratio,
    )
    if len(coords) == 0:
        logger.warning(f"No valid sampling points found in foreground mask for slide: {slide_metadata['file']}")
        return np.empty((0, 2), dtype=int), heatmap, thumb

    if return_heatmap:
        true_dim = slide_metadata["full_dims"][0]
        fg_scale = true_dim / float(clean_mask.shape[0])
        heatmap = map_coords_to_heatmap(coords, fg_scale, sampling_size, clean_mask.shape)
    return coords, heatmap, thumb
