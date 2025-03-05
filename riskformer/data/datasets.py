import numpy as np
from PIL import Image
import zarr
import openslide
import h5py
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import boto3
import botocore
from urllib.parse import urlparse
import logging
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from riskformer.utils.data_utils import sample_slide_image
from riskformer.utils.randstainna import RandStainNA
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, is_s3_path
from riskformer.utils.logger_config import log_event

logger = logging.getLogger(__name__)

# yaml_file = '/home/ubuntu/notebooks/cpc_hist/src/CRC_LAB_randomTrue_n0.yaml'
# stain_normalizer = RandStainNA(yaml_file, std_hyper=-1.0)


class SingleSlideDataset(Dataset):
    """
    PyTorch dataset for a single slide. Image patches are sampled at the specified coordinates and transformed
    using the provided transform function.
    
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
            slide_obj: openslide.OpenSlide,
            slide_metadata: dict,
            sample_coords: np.ndarray,
            sample_size: int,
            transform=None,
        ):
        self.slide_obj = slide_obj
        self.slide_metadata = slide_metadata
        self.sample_coords = sample_coords
        self.sample_size = sample_size
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.sample_coords)

    def __getitem__(self, idx):
        x, y = self.sample_coords[idx]
        # Returns PIL RGB image
        image = sample_slide_image(self.slide_obj, x, y, self.sample_size)
        image = self.transform(image)
        return image
    
    def close_slide(self):
        """Ensure OpenSlide file is closed properly to prevent memory leaks."""
        self.slide_obj.close()


class ZarrFeatureDataset(Dataset):
    def __init__(self, zarr_path):
        """
        Initialize dataset from a Zarr store.

        Args:
            zarr_path (str): Path to Zarr file (local or S3).
        """
        self.root = zarr.open(zarr_path, mode='r')
        self.coords = self.root["coords"]
        self.features = self.root["features"]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        """
        Fetch a single sample.
        """
        coord = self.coords[idx]  # (2,) coordinate
        feature = self.features[idx]  # (D,) feature vector
        return torch.tensor(coord, dtype=torch.int32), torch.tensor(feature, dtype=torch.float32)


class S3Cache:
    """
    Handles caching of S3 files locally.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize S3 cache handler.
        
        Args:
            cache_dir: Directory to store cached files. If None, uses system temp directory.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "riskformer_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.s3_client = boto3.client('s3')
        self._download_lock = threading.Lock()
        
    def get_local_path(self, s3_path: str) -> Path:
        """Get local cache path for S3 file."""
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return self.cache_dir / bucket / key
    
    def download_if_needed(self, s3_path: str) -> str:
        """
        Download file from S3 if not in cache.
        
        Args:
            s3_path: S3 path in format 's3://bucket/key'
            
        Returns:
            str: Path to local cached file
        """
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        local_path = self.get_local_path(s3_path)
        
        # Use lock to prevent multiple downloads of same file
        with self._download_lock:
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.s3_client.download_file(bucket, key, str(local_path))
                except botocore.exceptions.ClientError as e:
                    raise RuntimeError(f"Failed to download {s3_path}: {e}")
                
        return str(local_path)


class RiskFormerDataset(Dataset):
    """
    PyTorch dataset for RiskFormer model training. Takes a list of (coordinates, features) pairs
    and constructs sparse tensors for each sample. Supports both local and S3 paths.
    
    Args:
        feature_pairs: List of tuples (coords_path, features_path) pointing to H5 files.
                      Paths can be local or S3 URLs (s3://bucket/key)
        cache_dir: Directory to cache S3 files. If None, uses system temp directory.
        
    Example:
        pairs = [
            ("s3://bucket/slide1_coords.h5", "s3://bucket/slide1_features.h5"),
            ("s3://bucket/slide2_coords.h5", "s3://bucket/slide2_features.h5"),
        ]
        dataset = RiskFormerDataset(pairs)
        sparse_tensor = dataset[0]  # Get sparse tensor for first slide
    """
    def __init__(
        self,
        feature_pairs: List[Tuple[str, str]],
        cache_dir: Optional[str] = None,
    ):
        self.feature_pairs = feature_pairs
        self.s3_cache = S3Cache(cache_dir)
        
        # Prefetch first file to get feature dimension
        _, features_path = self._get_local_paths(feature_pairs[0])
        with h5py.File(features_path, 'r') as f:
            self.feature_dim = f['features'].shape[1]
            
        # Start background prefetch of other files
        self._prefetch_executor = ThreadPoolExecutor(max_workers=4)
        self._prefetch_all_files()
    
    def _get_local_paths(self, pair: Tuple[str, str]) -> Tuple[str, str]:
        """Get local paths for a pair of files, downloading from S3 if needed."""
        coords_path, features_path = pair
        
        if is_s3_path(coords_path):
            coords_path = self.s3_cache.download_if_needed(coords_path)
        if is_s3_path(features_path):
            features_path = self.s3_cache.download_if_needed(features_path)
            
        return coords_path, features_path
    
    def _prefetch_all_files(self):
        """Start background download of all S3 files."""
        for pair in self.feature_pairs[1:]:  # Skip first pair as it's already downloaded
            self._prefetch_executor.submit(self._get_local_paths, pair)

    def __len__(self):
        return len(self.feature_pairs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get sparse tensor representation of a slide's features.
        
        Args:
            idx: Index of the slide to fetch
            
        Returns:
            torch.Tensor: Sparse COO tensor of shape (N, D) in COO format
        """
        coords_path, features_path = self._get_local_paths(self.feature_pairs[idx])
        
        with h5py.File(coords_path, 'r') as f:
            coords = torch.tensor(f['coords'][:].T, dtype=torch.long)  # Shape: (2, N)
            
        with h5py.File(features_path, 'r') as f:
            values = torch.tensor(f['features'][:], dtype=torch.float32)  # Shape: (N, D)
            
        return torch.sparse_coo_tensor(
            indices=coords,
            values=values,
        )
    
    def __del__(self):
        """Cleanup background threads."""
        if hasattr(self, '_prefetch_executor'):
            self._prefetch_executor.shutdown(wait=False)


def create_riskformer_dataset(
    s3_bucket: str,
    s3_prefix: str = "",
    metadata_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
) -> RiskFormerDataset:
    """
    Create a RiskFormerDataset by discovering preprocessed H5 files in S3.
    
    Args:
        s3_bucket: S3 bucket containing the H5 files
        s3_prefix: Prefix (folder) in the S3 bucket
        metadata_file: JSON file containing the list of slides to use for training
        cache_dir: Directory to cache downloaded files
        profile_name: AWS profile name (optional)
        region_name: AWS region name (optional)
        
    Returns:
        RiskFormerDataset: Dataset containing sparse tensors for each slide
        
    Example:
        dataset = create_riskformer_dataset(
            s3_bucket="tcga-riskformer-data-2025",
            s3_prefix="preprocessed/uni/uni2-h",
            metadata_file="resources/riskformer_slide_samples.json"
        )
    """
    log_event("debug", "create_riskformer_dataset", "started",
              s3_bucket=s3_bucket, 
              s3_prefix=s3_prefix,
              metadata_file=metadata_file)
    
    # Initialize S3 client
    s3_client = initialize_s3_client(profile_name, region_name)
    if s3_client is None:
        raise RuntimeError("Failed to initialize S3 client")
    
    # Load metadata file if provided
    target_slides = None
    if metadata_file and os.path.isfile(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                slides_dict = json.load(f)
            target_slides = set(slides_dict.keys())
            log_event("info", "load_metadata", "success",
                     total_slides=len(target_slides))
        except Exception as e:
            log_event("error", "load_metadata", "error",
                     error=str(e))
            raise RuntimeError(f"Failed to load metadata file: {e}")
    
    # List all files in the prefix
    all_files = list_bucket_files(s3_client, s3_bucket, s3_prefix)
    log_event("debug", "list_s3_bucket_files", "success",
              s3_bucket=s3_bucket, 
              s3_prefix=s3_prefix, 
              file_count=len(all_files))
    
    # Find all unique basenames that have both coords and features files
    coords_files = {k for k in all_files.keys() if k.endswith("_coords.h5")}
    features_files = {k for k in all_files.keys() if k.endswith("_features.h5")}
    
    # Extract basenames and ensure both files exist
    basenames = set()
    for coords_file in coords_files:
        basename = coords_file[:-len("_coords.h5")]
        features_file = f"{basename}_features.h5"
        if features_file in features_files:
            # If metadata file provided, only include slides from metadata
            if target_slides is not None:
                slide_id = os.path.basename(basename)
                if slide_id in target_slides:
                    basenames.add(basename)
            else:
                basenames.add(basename)
    
    # Log statistics about found files
    if target_slides is not None:
        log_event("info", "find_complete_slide_sets", "success",
                  total_files=len(all_files),
                  total_slides_in_metadata=len(target_slides),
                  preprocessed_slides_found=len(basenames),
                  missing_slides=len(target_slides - {os.path.basename(b) for b in basenames}))
    else:
        log_event("info", "find_complete_slide_sets", "success",
                  total_files=len(all_files),
                  coords_files=len(coords_files),
                  features_files=len(features_files),
                  complete_sets=len(basenames))
    
    if not basenames:
        raise RuntimeError(f"No complete feature sets found in s3://{s3_bucket}/{s3_prefix}")
    
    # Create feature pairs
    feature_pairs = []
    for basename in sorted(basenames):  # Sort for consistent ordering
        coords_path = f"s3://{s3_bucket}/{basename}_coords.h5"
        features_path = f"s3://{s3_bucket}/{basename}_features.h5"
        feature_pairs.append((coords_path, features_path))
        
    # Create dataset
    try:
        dataset = RiskFormerDataset(feature_pairs, cache_dir)
        log_event("info", "create_riskformer_dataset", "success",
                  slide_count=len(dataset),
                  feature_dim=dataset.feature_dim,
                  cache_dir=str(dataset.s3_cache.cache_dir))
        return dataset
    except Exception as e:
        log_event("error", "create_riskformer_dataset", "error",
                  error=str(e))
        raise e


