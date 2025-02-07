'''
data_utils.py
author: landeros10
created: 2/25/2025
'''
import logging
import time
import json
import os
from pydantic import BaseModel, Field

import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import resize, rescale
from skimage.filters import threshold_triangle
from skimage.morphology import opening, closing, dilation, square
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from openslide import OpenSlide
from histomicstk.segmentation import simple_mask

import boto3
import botocore

logger = logging.getLogger(__name__)

class ForegroundConfigSchema(BaseModel):
    thumb_size: int = Field(default=500, description="Size of the thumbnail to use for foreground.")
    fraction: float = Field(default=0.10, description="Fraction of the image to use for foreground sampling.")
    bandwidth: int = Field(default=2, description="Bandwidth of the Gaussian kernel to use for foreground estimation.")
    min_tissue_prob: float = Field(default=0.05, description="Minimum tissue probability to consider a pixel as foreground.")
    min_foreground_ratio: float = Field(default=0.15, description="Minimum foreground/background ratio in slide.")


class ForegroundCleanupConfigSchema(BaseModel):
    open: int = Field(default=3, description="Size of the structuring element to use for opening foreground mask.")
    fill: int = Field(default=15, description="Size of the structuring element to use for closing foreground mask.")
    dil: int = Field(default=1, description="Size of the structuring element to use for final dilation of foreground mask.")
    border: int = Field(default=50, description="Size of the border to consider for aberrant regions.")
    width_percentage: float = Field(default=0.9, description="Percentage of the mask width covered to consider as aberrant.")
    height_percentage: float = Field(default=0.25, description="Percentage of the mask height covered to consider as aberrant.")


# This was designed for images taken at (1/4) resolution of 20X
class TilingConfigSchema(BaseModel):
    size: int = Field(default=256, description="Size of the tile to extract from the slide.")
    overlap: float = Field(default=0.75, description="Overlap between tiles.")
    p_foreground: float = Field(default=0.5, description="Minimum fraction of foreground pixels in a tile.")
    reference_mag: float = Field(default=20.0, description="Reference magnification level for tiling.")


DEFAULT_FOREGROUND_CONFIG = ForegroundConfigSchema().dict()
DEFAULT_FOREGROUND_CLEANUP_CONFIG = ForegroundCleanupConfigSchema().dict()
DEFAULT_TILING_CONFIG = TilingConfigSchema().dict()


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


def filter_coords_mask(coords, foreground, fg_scale, size, p_foreground):
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


def get_hist_foreground(image, foreground_config):
    """ Take RGB image (H x W x C) and produce foreground mask using publicly
    available histomicsTK library.

    Args:
        image (np.ndarray): RGB image. Shape (H, W, C).
        foreground_config (dict): dictionary of foreground detection parameters.
    Returns:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    """
    fraction = foreground_config.get("fraction", DEFAULT_FOREGROUND_CONFIG["fraction"])
    bandwidth = foreground_config.get("bandwidth", DEFAULT_FOREGROUND_CONFIG["bandwidth"])
    min_tissue_prob = foreground_config.get("min_tissue_prob", DEFAULT_FOREGROUND_CONFIG["min_tissue_prob"])
    min_foreground_ratio = foreground_config.get("min_foreground_ratio", DEFAULT_FOREGROUND_CONFIG["min_foreground_ratio"])

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


def remove_large_horizontal_regions(mask, width_percentage, height_percentage):
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
        foreground_cleanup_config,
        shape: tuple = None
):
    """
    Cleans up binary mask by removing aberrant regions, filling holes.
    Optionally, dilates the mask and resizes it to a given shape.

    Args:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
        foreground_cleanup_config (dict): dictionary of foreground cleanup parameters.
        shape (tuple): shape to resize the mask to.
        border (int): size of border to remove.
    
    Returns:
        mask (np.ndarray): cleaned mask. Shape (H, W).
    """
    open = foreground_cleanup_config.get("open", DEFAULT_FOREGROUND_CLEANUP_CONFIG["open"])
    fill = foreground_cleanup_config.get("fill", DEFAULT_FOREGROUND_CLEANUP_CONFIG["fill"])
    dil = foreground_cleanup_config.get("dil", DEFAULT_FOREGROUND_CLEANUP_CONFIG["dil"])
    border = foreground_cleanup_config.get("border", DEFAULT_FOREGROUND_CLEANUP_CONFIG["border"])

    start_time = time.time()
    mask = opening(mask, square(open))
    logger.debug(f"Foreground mask fill after opening: {mask.sum() / mask.size:.2f}")

    if mask.astype(int).sum() == 0:
        return mask
    mask = binary_fill_holes(mask)
    logger.debug(f"Foreground mask fill after filling holes: {mask.sum() / mask.size:.2f}")

    mask[border, :] = mask[-border, :] = 0
    logger.debug(f"Foreground mask fill after removing border: {mask.sum() / mask.size:.2f}")
    width_percentage = foreground_cleanup_config.get("width_percentage", DEFAULT_FOREGROUND_CLEANUP_CONFIG["width_percentage"])
    height_percentage = foreground_cleanup_config.get("height_percentage", DEFAULT_FOREGROUND_CLEANUP_CONFIG["height_percentage"])
    mask = remove_large_horizontal_regions(mask, width_percentage, height_percentage)
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


def get_slide_foreground(slideObj, foreground_config):
    """ 
    Get foreground mask for openslide object.

    Args:
        slideObj (OpenSlide): OpenSlide object.
        foreground_config (dict): dictionary of foreground detection parameters.
    Returns:
        mask (np.ndarray): binary mask of the foreground. Shape (H, W).
    """
    thumb_size = foreground_config.get("thumb_size", DEFAULT_FOREGROUND_CONFIG["thumb_size"])
    thumb = get_slide_thumb(slideObj, size=thumb_size)
    return get_hist_foreground(thumb, foreground_config)


def get_mask_samplepoints(foreground_mask, slide_metadata, tiling_config):
    tile_overlap = tiling_config.get("overlap", DEFAULT_TILING_CONFIG["overlap"])
    p_foreground = tiling_config.get("p_foreground", DEFAULT_TILING_CONFIG["p_foreground"])

    slide_mag = slide_metadata["mag"]
    tile_size = tiling_config.get("size", DEFAULT_TILING_CONFIG["size"])
    reference_mag = tiling_config.get("reference_mag", DEFAULT_TILING_CONFIG["reference_mag"])
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
        coords = coords[:, [1, 0]]
    else:
        logger.warning(f"No sampling coords could be generated for slide: {slide_metadata['file']}")
        coords = np.empty((0, 2), dtype=int)

    return coords, crop_size


def get_slide_samplepoints(slideObj, metadata, tiling_config, foreground_config, foreground_cleanup_config, return_heatmap=False):
    """
    Extracts sampling points from the slide based on the foreground mask.
    
    Args:
        slideObj (OpenSlide): OpenSlide object.
        metadata (dict): dictionary of slide metadata.
        tiling_config (dict): dictionary of tiling parameters.
        foreground_config (dict): dictionary of foreground detection parameters.
        foreground_cleanup_config (dict): dictionary of foreground cleanup parameters.
        return_heatmap (bool): whether to return heatmap of the sampling points.
    
    Returns:
        coords (np.ndarray): array of coordinates. Shape (N, 2).
        heatmap (np.ndarray): heatmap of the sampling points. Shape (H, W). None if not requested.
    """
    # thumb_size, min_tissue_prob, reference_mag,
    start_time = time.time()
    foreground_mask = get_slide_foreground(slideObj, foreground_config)
    clean_mask = mask_clean_up_and_resize(foreground_mask, foreground_cleanup_config)
    logger.debug(f"Generated foreground mask in {time.time() - start_time:.1f}s")

    coords, crop_size = get_mask_samplepoints(
        clean_mask,
        metadata,
        tiling_config,
    )

    heatmap = None
    if return_heatmap:
        true_dim = metadata["full_dims"][0]
        fg_scale = true_dim / float(clean_mask.shape[0])
        heatmap = coords_to_heatmap(coords, fg_scale, crop_size, clean_mask.shape)
    logger.debug(f"Finished processing slide\n{metadata['file']}\nin {time.time() - start_time:.2f}s")
    return coords, heatmap


def load_slide_paths(slides_list_file):
    """
    Load slide paths from a list of slides.

    Args:
        slides_list_file (str): json file containing the list of slides.

    Returns:
        slides_dict (dict): dictionary where 
        the keys are slide paths (str) and the values are metadata dictionaries
    """
    if slides_list_file or os.path.isfile(slides_list_file):
        with open(slides_list_file, "r") as f:
            slides_dict = json.load(f)

        if not slides_dict:
            logger.error(f"No slides found in {slides_list_file}")
            raise ValueError(f"No slides found in {slides_list_file}")
        return slides_dict
    else:
        logger.error(f"Slide list file not found: {slides_list_file}")
        raise FileNotFoundError(f"Slide list file not found: {slides_list_file}")


def initialize_s3_client(
        profile_name,
        region_name=None,
        return_session=False):
    """
    Initialize boto3 session and S3 client.
    
    Args:
        profile_name (str): AWS profile name.
        return_session (bool): Return boto3 session if True.
    Returns:
        boto3.client: S3 boto3 client.
    """
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        logger.debug("Created boto3 session")
    except Exception as e:
        logger.error(f"Failed to create boto3 session: {e}")
        return
    
    try:
        boto_config = botocore.config.Config(max_pool_connections=50)
        s3_client = session.client("s3", config=boto_config, use_ssl=False)
        logger.debug("Created S3 client")
        logger.debug(f"Available buckets: {s3_client.list_buckets().get('Buckets')}")
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return
    if return_session:
        return s3_client, session
    return s3_client


def wipe_bucket_dir(s3_client, bucket_name, bucket_prefix=""):
    """
    Deletes all files under a specific prefix in an S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): Prefix (directory) to delete.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    files_deleted = 0
    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        for page in pages:
            if "Contents" in page:
                try:
                    objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
                    files_deleted += len(objects)
                    logger.debug(f"Deleted {len(objects)} files")
                except Exception as e:
                    logger.error(f"Failed to delete files in page {page}: {e}")
                    return False
        logger.debug(f"Deleted {files_deleted} files under s3://{bucket_name}/{bucket_prefix}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete files under s3://{bucket_name}/{bucket_prefix}: {e}")
        return False


def list_bucket_files(s3_client, bucket_name, bucket_prefix=""):
    """
    Get a list of all files in an S3 bucket under a given prefix.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): S3 prefix (folder) to list objects from.

    Returns:
        dict: {file_name: file_size_in_bytes} for all files in S3.
    """            
    existing_files = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page.get("Contents", []):
                    existing_files[obj["Key"]] = obj["Size"]
        if not existing_files:
            logger.debug(f"No files found in s3://{bucket_name}/{bucket_prefix}")
    except Exception as e:
        logger.error(f"Failed to list files in s3://{bucket_name}/{bucket_prefix}: {e}")
    return existing_files


def upload_large_files_to_bucket(
        s3_client,
        bucket_name, 
        files_list,
        prefix="raw", 
        ext="",
        reupload=False,
        threshold=20 * 1024 * 1024,
        chunk_size=20 * 1024 * 1024,
        max_concurrency=5):
    """
    Upload large files to S3 bucket using multipart upload.
        
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        files_list (list): List of file paths to upload.
        prefix (str): S3 key prefix.
        ext (str): File extension to filter files.
        reupload (bool): Reupload files even if they exist.
        threshold (int): Multipart upload threshold in bytes.
        chunk_size (int): Multipart upload chunk size in bytes.
        max_concurrency (int): Maximum number of concurrent uploads.
    """

    config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=threshold,
        multipart_chunksize=chunk_size,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    logger.debug(f"Using multipart_threshold: {(threshold)/(1024*1024):.2f} MB, chunk size: {(chunk_size)/(1024 * 1024):.2f} MB, max concurrency: {max_concurrency}")
    
    existing_files = list_bucket_files(s3_client, bucket_name, prefix)
    start_time = time.time()
    count = 0
    total_files = len(files_list)
    for file_path in files_list:
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)

        if file_exists and (not ext or file_path.endswith(ext)):
            file_name = os.path.basename(file_path)
            s3_key = f"{prefix}/{file_name}"
            local_size = os.path.getsize(file_path)

            if not reupload and s3_key in existing_files and existing_files[s3_key] == local_size:                
                count += 1
                total_time_str = time.strftime("%M:%S", time.gmtime((time.time() - start_time)))
                logger.debug(f"({total_time_str}) ({count}/{total_files}) Skipping: {file_name}")
                continue
            
            try:
                count += 1
                s3_client.upload_file(file_path, bucket_name, f"{prefix}/{file_name}", Config=config)
                total_time_str = time.strftime("%M:%S", time.gmtime((time.time() - start_time)))
                logger.debug(f"({total_time_str}) ({count}/{total_files}) Uploaded: {file_name} to s3://{bucket_name}/{prefix}/")
            except Exception as e:
                logger.error(f"Failed to upload {file_name}: {e}")
        else:
            logger.warning(f"Skipping: {file_path} (File not found or invalid)")
