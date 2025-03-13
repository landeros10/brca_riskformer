import numpy as np
import random
from PIL import Image
from skimage.measure import label, regionprops
import zarr
import openslide
import h5py
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import boto3
import botocore
from urllib.parse import urlparse
import logging
import json

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

from riskformer.utils.data_utils import sample_slide_image
from riskformer.utils.training_utils import PatchInfo, split_riskformer_data, load_training_config
from riskformer.utils.aws_utils import initialize_s3_client, list_bucket_files, is_s3_path
from riskformer.utils.logger_config import log_event

logger = logging.getLogger(__name__)


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
    
    def prefetch_patient_files(
            self,
            patient_examples: Dict[str, Dict[str, Any]],
            collect_stats: bool = True,
            max_workers: int = 4,
    ) -> None:
        """
        Prefetch all files for a collection of patients in parallel.
        
        Args:
            patient_examples: Dictionary of patient examples containing file paths
            collect_stats: Whether to collect statistics about the downloaded files
            max_workers: Maximum number of parallel download workers
        """
        if collect_stats:
            feature_dim = None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for patient_id in patient_examples:
                coords_paths = patient_examples[patient_id]["coords_paths"]
                features_paths = patient_examples[patient_id]["features_paths"]
                for coords_path, features_path in zip(coords_paths, features_paths):
                    futures.append(executor.submit(self.download_if_needed, coords_path))
                    futures.append(executor.submit(self.download_if_needed, features_path))

                    if collect_stats:
                        local_features_path = self.get_local_path(features_path)
                        with h5py.File(local_features_path, 'r') as f:
                            features = f['features']
                            if feature_dim is None:
                                feature_dim = features.shape[1]
                                rolling_sum = torch.zeros(feature_dim)
                                rolling_sum_of_squares = torch.zeros(feature_dim)
                                rolling_count = 0

                            example_sum = features.sum(axis=0) # (D,)
                            example_sum_of_squares = (features ** 2).sum(axis=0) # (D,)

                            rolling_sum += example_sum
                            rolling_sum_of_squares += example_sum_of_squares
                            rolling_count += features.shape[0]

            # Wait for all downloads to complete
            for future in futures:
                future.result()

            if collect_stats:
                feature_stats = {
                    'mean': rolling_sum / rolling_count,
                    'std': np.sqrt(rolling_sum_of_squares / rolling_count - (rolling_sum / rolling_count) ** 2),
                }
                return feature_stats
        
        log_event("info", "prefetch_patient_files", "complete", 
                 patient_count=len(patient_examples),
                 file_count=sum(len(patient_examples[p]["coords_paths"]) * 2 for p in patient_examples))


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
    # Class-level constants for field definitions
    SPECIAL_BINARY_FIELDS = {
        'odx85': {'H': 1.0, 'L': 0.0},
        'mphr': {'H': 1.0, 'L': 0.0}
    }
    
    BINARY_FIELDS = {
        'ER_Status_By_IHC': {'positive': 1.0, 'negative': 0.0},
        'pr_status_by_ihc': {'positive': 1.0, 'negative': 0.0},
        'HER2Calc': {'positive': 1.0, 'negative': 0.0},
        'Necrosis': {'Present': 1.0, 'Absent': 0.0},
        'Lymphovascular Invasion (LVI)': {'Present': 1.0, 'Absent': 0.0},
        'Overall_Survival_Status': {'dead': 1.0, 'alive': 0.0}
    }
    
    REGRESSION_FIELDS = [
        'odx_train', 'Grade', 'tumor_size', 'Overall_Survival_Months', 
        'Disease_Free_Months', 'Epithelial', 'Pleomorph', 'Grade.1', 'age_at_diagnosis'
    ]
    
    def __init__(
        self,
        patient_examples: Dict[str, Dict[str, Any]],
        max_dim: int = 32,
        overlap: float = 0.1,
        feature_stats: Optional[Dict[str, float]] = None,
        s3_cache: Optional[S3Cache] = None,
        include_labels: Optional[List[str]] = None,
    ):
        """
        Initialize the RiskFormer dataset.
        
        Args:
            patient_examples: Dictionary of patient examples
            max_dim: Maximum dimension of patches
            overlap: Overlap between patches
            feature_stats: Dictionary of mean and std of the features, channel-wise.
            s3_cache: S3Cache object
            include_labels: List of label names to include
        """
        self.patient_examples = patient_examples
        self.patient_ids = list(patient_examples.keys())
        self.s3_cache = s3_cache

        self.max_dim = max_dim
        self.overlap = overlap

        self.normalize_features = False
        if feature_stats is not None:
            self.normalize_features = True
            self.feature_stats = feature_stats
        
        # Set up labels to include
        self.include_labels = include_labels

        # Determine feature dimension
        if len(self.patient_ids) > 0:
            test_features_s3_path = patient_examples[self.patient_ids[0]]["features_paths"][0]
            test_features_local_path = self.s3_cache.get_local_path(test_features_s3_path)
            with h5py.File(str(test_features_local_path), 'r') as f:
                self.feature_dim = f['features'].shape[1]
        else:
            # Default feature dimension if no data available
            self.feature_dim = 1024
            logger.warning("No patient examples provided. Creating empty Dataset.")
        
        # Precompute lowercase include labels for faster comparison
        self._lowercase_include_labels = None
        if self.include_labels is not None:
            self._lowercase_include_labels = [l.lower() for l in self.include_labels]
    
    def __len__(self):
        return len(self.patient_ids)
    
    def _create_dense_features(
            self,
            coords_paths: List[str],
            features_paths: List[str],
    ) -> List[torch.Tensor]:
        """
        Create a dense features tensor from a list of coordinates and features paths.
        
        Args:
            coords_paths: List of paths to H5 files containing coordinates
            features_paths: List of paths to H5 files containing features
            
        Returns:
            dense_features: List of dense tensors, each of shape (H, W, D)
        """
        assert len(coords_paths) == len(features_paths), "Number of coordinates and features paths must match"
        
        # Get sparse tensors
        all_sparse_tensors = self._create_sparse_features(coords_paths, features_paths)
        
        # Convert to dense tensors
        dense_features = [sparse.to_dense() for sparse in all_sparse_tensors]
        
        return dense_features
    
    def _create_sparse_features(
            self,
            coords_paths: List[str],
            features_paths: List[str],
    ) -> torch.Tensor:
        """
        Create a sparse features tensor from a list of coordinates and features paths.
        
        Args:
            coords_paths: List of paths to H5 files containing coordinates
            features_paths: List of paths to H5 files containing features
            
        Returns:
            sparse_features: Sparse tensor of shape (N, H, W, D)
        """
        assert len(coords_paths) == len(features_paths), "Number of coordinates and features paths must match"
        all_sparse_tensors = []
        for coords_path, features_path in zip(coords_paths, features_paths):
            # Get local paths (files should already be downloaded by prepare_data)
            local_coords_path = str(self.s3_cache.get_local_path(coords_path))
            local_features_path = str(self.s3_cache.get_local_path(features_path))
            
            with h5py.File(local_coords_path, 'r') as f:
                coords = torch.tensor(f['coords'][:].T, dtype=torch.long)  # Shape: (2, N)
                
            with h5py.File(local_features_path, 'r') as f:
                feats = torch.tensor(f['features'][:], dtype=torch.float32)  # Shape: (N, D)
                
            # Get dimensions for sparse tensor
            H, W = coords.max(dim=1).values + 1
            sparse_size = (H, W, self.feature_dim)
            sparse_tensor = torch.sparse_coo_tensor(
                coords,  # indices
                feats,   # values 
                size=sparse_size
            )
            all_sparse_tensors.append(sparse_tensor)

        return all_sparse_tensors
    
    def _create_feature_regionprops(
            self,
            features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a feature mask from a dense feature tensor.
        """
        binary_mask = (features != 0).to(torch.int).max(dim=-1).values
        labeled_mask = label(binary_mask.cpu().numpy())
        return regionprops(labeled_mask)
    
    def _process_region(
            self, 
            feature_id: int, 
            region_id: int, 
            region: object, 
            features: torch.Tensor,
            max_dim: int, 
            overlap: float
    ) -> Tuple[List[torch.Tensor], List['PatchInfo']]:
        """
        Process a single region and extract patches with their metadata.
        
        Args:
            feature_id: ID of the feature being processed
            region_id: ID of the region within the feature
            region: Region properties object
            features: Feature tensor
            max_dim: Maximum dimension for patches
            overlap: Fraction of overlap between patches
            
        Returns:
            Tuple containing:
                - List of patch tensors
                - List of PatchInfo objects
        """
        
        region_patches = []
        region_info = []
        
        # Extract region bounding box
        min_row, min_col, max_row, max_col = region.bbox
        height, width = max_row - min_row, max_col - min_col
        
        # Extract the region
        region_features = features[min_row:max_row, min_col:max_col]
        
        if height <= max_dim and width <= max_dim:
            # Small region case - just pad it
            patch, info = self._create_single_patch(
                feature_id, region_id, region_features, 
                min_row, min_col, max_row, max_col, 
                0, 0, height, width, max_dim
            )
            region_patches.append(patch)
            region_info.append(info)
        else:
            # Large region case - split into overlapping patches
            patches_info = self._split_large_region(
                feature_id, region_id, region_features,
                min_row, min_col, max_row, max_col,
                height, width, max_dim, overlap
            )
            for patch, info in patches_info:
                region_patches.append(patch)
                region_info.append(info)
        
        return region_patches, region_info
    
    def _create_single_patch(
            self,
            feature_id: int,
            region_id: int,
            region_features: torch.Tensor,
            min_row: int,
            min_col: int,
            max_row: int,
            max_col: int,
            row_start: int,
            col_start: int,
            row_end: int,
            col_end: int,
            max_dim: int
    ) -> Tuple[torch.Tensor, 'PatchInfo']:
        """
        Create a single patch and its metadata.
        
        Args:
            feature_id: ID of the feature
            region_id: ID of the region
            region_features: Features for this region
            min_row, min_col, max_row, max_col: Region bounding box
            row_start, col_start, row_end, col_end: Patch coordinates within region
            max_dim: Maximum dimension for patches
            
        Returns:
            Tuple containing:
                - Padded patch tensor
                - PatchInfo object
        """
        
        # Extract patch
        patch = region_features[row_start:row_end, col_start:col_end]
        
        # Pad patch if needed
        if patch.shape[0] < max_dim or patch.shape[1] < max_dim:
            patch = F.pad(
                patch,
                (0, 0,  # feature dimension - no padding
                 0, max_dim - patch.shape[1],  # width padding
                 0, max_dim - patch.shape[0])  # height padding
            )
        
        # Verify patch has the correct shape
        assert patch.shape[0] == max_dim, f"Expected height {max_dim}, got {patch.shape[0]}"
        assert patch.shape[1] == max_dim, f"Expected width {max_dim}, got {patch.shape[1]}"
        assert patch.shape[2] == self.feature_dim, f"Expected feature dim {self.feature_dim}, got {patch.shape[2]}"
        
        # Create patch info
        patch_info = PatchInfo(
            feature_id=feature_id,
            region_id=region_id,
            region_min_row=min_row,
            region_min_col=min_col,
            region_max_row=max_row,
            region_max_col=max_col,
            patch_row_start=row_start,
            patch_col_start=col_start,
            patch_row_end=row_end,
            patch_col_end=col_end
        )
        
        return patch, patch_info
    
    def _split_large_region(
            self,
            feature_id: int,
            region_id: int,
            region_features: torch.Tensor,
            min_row: int,
            min_col: int,
            max_row: int,
            max_col: int,
            height: int,
            width: int,
            max_dim: int,
            overlap: float
    ) -> List[Tuple[torch.Tensor, 'PatchInfo']]:
        """
        Split a large region into overlapping patches.
        
        Args:
            feature_id: ID of the feature
            region_id: ID of the region
            region_features: Features for this region
            min_row, min_col, max_row, max_col: Region bounding box
            height, width: Height and width of the region
            max_dim: Maximum dimension for patches
            overlap: Fraction of overlap between patches
            
        Returns:
            List of tuples containing (patch, patch_info)
        """
        result = []
        
        # Calculate step sizes with overlap
        row_step = max(1, int(max_dim * (1 - overlap)))
        col_step = max(1, int(max_dim * (1 - overlap)))
        
        # Loop bounds to ensure we cover the entire region
        row_end_limit = max(0, height - max_dim)
        col_end_limit = max(0, width - max_dim)
        
        # Generate splits with overlap
        for row_start in range(0, row_end_limit + row_step, row_step):
            for col_start in range(0, col_end_limit + col_step, col_step):
                # Ensure we don't exceed boundaries
                row_end = min(row_start + max_dim, height)
                col_end = min(col_start + max_dim, width)
                
                # Adjust start positions for last patches to ensure they are max_dim sized
                if row_end == height and height > max_dim:
                    row_start = height - max_dim
                if col_end == width and width > max_dim:
                    col_start = width - max_dim
                
                # Create patch and its info
                patch, info = self._create_single_patch(
                    feature_id, region_id, region_features,
                    min_row, min_col, max_row, max_col,
                    row_start, col_start, row_end, col_end, max_dim
                )
                
                result.append((patch, info))
        
        return result

    def split_and_pad_features(
            self,
            features_list: List[torch.Tensor],
            max_dim: int = 32,
            overlap: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split features into patches and pad them to the same size.
        
        Args:
            features_list: List of dense feature tensors, each of shape (H, W, D)
            max_dim: Maximum dimension for patches
            overlap: Fraction of overlap between patches (default: 0.1)
            
        Returns:
            Tuple containing:
                - patches: Tensor of shape (N, max_dim, max_dim, D) where N is total number of patches
                - patch_info: Tensor containing patch information for reconstruction
        """
        
        # Ensure we have at least one feature tensor
        if not features_list or len(features_list) == 0:
            empty_patches = torch.zeros((0, max_dim, max_dim, self.feature_dim))
            empty_info = torch.zeros((0, 10), dtype=torch.int32)  # Changed from 8 to 10 columns
            return empty_patches, empty_info
        
        all_patches = []
        all_patch_info = []
        
        for feature_id, features in enumerate(features_list):
            logger.debug(f'feature set {feature_id}, shape {features.shape}')
            
            # Create regionprops
            rprops = self._create_feature_regionprops(features)
            logger.debug(f"Number of regions: {len(rprops)}")
            
            # Process each region
            for region_id, region in enumerate(rprops):
                region_patches, region_info = self._process_region(
                    feature_id, region_id, region, features, max_dim, overlap
                )
                all_patches.extend(region_patches)
                all_patch_info.extend(region_info)
        
        if not all_patches:
            empty_patches = torch.zeros((0, max_dim, max_dim, self.feature_dim))
            empty_info = torch.zeros((0, 10), dtype=torch.int32)  # Changed from 8 to 10 columns
            return empty_patches, empty_info
        
        # Convert patch info to tensor
        patch_info_tensor = torch.stack([info.to_tensor() for info in all_patch_info], dim=0)
        
        return torch.stack(all_patches, dim=0), patch_info_tensor
    
    def should_include_label(self, label_name: str) -> bool:
        """
        Check if a label should be included based on the include_labels list.
        
        Args:
            label_name: Name of the label to check
            
        Returns:
            True if the label should be included, False otherwise
        """
        # If no include_labels list is provided, include all labels
        if self.include_labels is None:
            return True
        # Otherwise, only include labels in the list
        return label_name.lower() in self._lowercase_include_labels
        
    def process_special_binary_fields(
            self,
            patient_data: Dict[str, Any],
            example_data: Dict[str, Any]
    ) -> None:
        """
        Process special binary classification fields.
        
        Args:
            patient_data: Dictionary of patient data
            example_data: Dictionary to store processed example data
        """
        for field, mapping in self.SPECIAL_BINARY_FIELDS.items():
            if self.should_include_label(field):
                if field in patient_data and patient_data[field] is not None:
                    if patient_data[field] in mapping:
                        example_data['labels'][field.lower()] = torch.tensor([mapping[patient_data[field]]], dtype=torch.float32)
    
    def process_binary_fields(self, patient_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Process standard binary categorical fields.
        
        Args:
            patient_data: Dictionary of patient data
            metadata: Dictionary to store processed metadata
        """
        for field, mapping in self.BINARY_FIELDS.items():
            if field in patient_data and patient_data[field] is not None and self.should_include_label(field):
                if patient_data[field] in mapping:
                    field_lower = field.lower()
                    metadata['labels'][field_lower] = torch.tensor([mapping[patient_data[field]]], dtype=torch.float32)
    
    def process_regression_fields(self, patient_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Process regression fields.
        
        Args:
            patient_data: Dictionary of patient data
            metadata: Dictionary to store processed metadata
        """
        for field in self.REGRESSION_FIELDS:
            if field in patient_data and patient_data[field] is not None and self.should_include_label(field):
                try:
                    value = float(patient_data[field])
                    field_lower = field.lower()
                    metadata['labels'][field_lower] = torch.tensor([value], dtype=torch.float32)
                except (ValueError, TypeError):
                    # Skip if the value cannot be converted to float
                    pass
    
    def process_mitosis_field(self, patient_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Process the special Mitosis field which has ordered categories.
        
        Args:
            patient_data: Dictionary of patient data
            metadata: Dictionary to store processed metadata
        """
        if 'Mitosis' in patient_data and patient_data['Mitosis'] is not None and self.should_include_label('Mitosis'):
            mitosis_text = patient_data['Mitosis']
            score = None
            
            if '(score = 1)' in mitosis_text:
                score = 1.0
            elif '(score = 2)' in mitosis_text:
                score = 2.0
            elif '(score = 3)' in mitosis_text:
                score = 3.0
            else:
                # Try to extract numeric score if present
                import re
                match = re.search(r'\(score = (\d+)\)', mitosis_text)
                score = float(match.group(1)) if match else None
                
            if score is not None:
                metadata['labels']['mitosis'] = torch.tensor([score], dtype=torch.float32)
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to have zero mean and unit variance.
        """
        return (features - self.feature_stats['mean']) / self.feature_stats['std']
    
    def __getitem__(
            self,
            idx: int,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a patient's data by index.
        
        Args:
            idx: Index of patient in dataset
            
        Returns:
            patches: Batch of channels-first tensors of shape (B, C, H, W)
            metadata: Dictionary containing patient metadata including labels for multiple tasks
                The 'labels' field contains a nested dictionary with:
                - Task name as key
                - Tensor value representing label for that task
        """
        patient_id = self.patient_ids[idx]
        patient_data = self.patient_examples[patient_id]

        coords_paths = patient_data["coords_paths"]
        features_paths = patient_data["features_paths"]
        
        # Create dense features from sparse representations
        dense_features = self._create_dense_features(
            coords_paths=coords_paths,
            features_paths=features_paths,
        )
        
        # Process features into patches
        example_features, patch_info = self.split_and_pad_features(
            features_list=dense_features,
            max_dim=self.max_dim,
            overlap=self.overlap,
        )

        # TODO: dataset-wide normalization
        
        # Create metadata dictionary with patch info and patient ID
        example_data = {
            'patch_info': patch_info,
            'patient_id': patient_id,
            'labels': {},  # Dictionary to store multiple labels for different tasks
        }
        
        # Process all field types using the helper methods
        self.process_special_binary_fields(patient_data, example_data)
        self.process_binary_fields(patient_data, example_data)
        self.process_regression_fields(patient_data, example_data)
        self.process_mitosis_field(patient_data, example_data)
                
        # Return patches in channels-first format
        example_features = example_features.permute(0, 3, 1, 2)
        if self.normalize_features:
            example_features = self.normalize_features(example_features)
        return example_features, example_data
    

class RiskFormerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for RiskFormer datasets.
    
    This module handles the loading, splitting, and preparation of data for training,
    validation, and testing with PyTorch Lightning.
    """
    # TODO: add feature selector learnable parameters
    # state_dict() and load_state_dict() should save and load these parameters
    
    @classmethod
    def from_config(
            cls,
            config: Union[Dict[str, Any], str]
    ) -> 'RiskFormerDataModule':
        """
        Create a RiskFormerDataModule from a config.
        
        Args:
            config: Either a dictionary containing configuration parameters or a path to a JSON file containing the configuration.
        
        Returns:
            A new RiskFormerDataModule instance configured with the parameters from the config.
        """
        if isinstance(config, str):
            config = load_training_config(config)

        include_labels = list(config['tasks'].keys())

        # Create the data module with parameters from config
        return cls(
            s3_bucket=config['s3_bucket'],
            s3_prefix=config.get('s3_prefix', ''),
            max_dim=config.get('max_dim', 32),
            overlap=config.get('overlap', 0.0),
            metadata_file=config.get('metadata_file'),
            cache_dir=config.get('cache_dir'),
            profile_name=config.get('profile_name'),
            region_name=config.get('region_name'),
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.2),
            test_split=config.get('test_split', 0.1),
            seed=config.get('seed', 42),
            pin_memory=config.get('pin_memory', True),
            include_labels=include_labels,
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> 'RiskFormerDataModule':
        """
        Create a RiskFormerDataModule from a configuration file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            A new RiskFormerDataModule instance configured with the parameters from the config file.
        """
        return cls.from_config(config_path)

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "",
        max_dim: int = 32,
        overlap: float = 0.0,
        metadata_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
        include_labels: Optional[List[str]] = None,
    ):
        """
        Initialize the RiskFormer DataModule.
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: Prefix for S3 objects
            max_dim: Maximum dimension for patches
            overlap: Overlap between patches
            metadata_file: Path to metadata file
            cache_dir: Directory to cache S3 files
            profile_name: AWS profile name
            region_name: AWS region name
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            pin_memory: Whether to pin memory for dataloaders
            include_labels: List of label names to include in the dataset (optional)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Dataset configuration parameters
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.metadata_file = metadata_file
        self.cache_dir = cache_dir
        self.profile_name = profile_name
        self.region_name = region_name
        self.val_split = val_split
        self.test_split = test_split
        self.include_labels = include_labels
        
        # Dataset tunable parameters
        self.max_dim = max_dim
        self.overlap = overlap
        self.batch_size = batch_size

        # DataLoader configuration parameters
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
                
        # Dataset objects
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Set random seeds for reproducibility
        self.set_seed()

        # Set dataset splitting variable
        self.set_split_var()

        # Create S3 cache handler
        self.s3_cache = S3Cache(self.cache_dir)

        # Setup patient examples
        self.setup_patient_examples()

        # Will feature stats to disk upon downloading files from s3
        self.feature_stats_path = os.path.join(self.cache_dir, "feature_stats.json")
        self.feature_stats = None

    def setup_patient_examples(self):
        """
        Gather available examples and update paths to local paths
        """
        slide_ids, slide_data = load_dataset_metadata(self.metadata_file)
        s3_client = initialize_s3_client(self.profile_name, self.region_name)
        if s3_client is None:
            raise RuntimeError("Failed to initialize S3 client")
        
        # Gather available examples and update paths to local paths
        self.patient_examples = create_patient_examples(
            s3_client,
            self.s3_bucket,
            self.s3_prefix,
            slide_ids,
            slide_data
        )
        for patient_id, patient_data in self.patient_examples.items():
            updated_paths = [
                self.s3_cache.get_local_path(path)
                for path in patient_data['features_paths']
            ]
            self.patient_examples[patient_id]['features_paths'] = updated_paths

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_split_var(self):
        """
        Set the split variable based on the config and include_labels.
        """
        # Determine which label to use for stratification
        label_var = "odx85"
        if self.include_labels:
            label_var = self.include_labels[0]
        self.split_var = label_var

    def prepare_data(self):
        """
        Download and prepare data. This method is called only once and on 1 GPU.
        
        This is where we can download data, preprocess it, etc.
        """
        # Prefetch all files
        feature_stats = self.s3_cache.prefetch_patient_files(
            patient_examples=self.patient_examples, 
            collect_stats=True,
        )

        # Save feature stats (mean and std) to disk using cache_dir
        with open(self.feature_stats_path, "w") as f:
            json.dump(feature_stats, f)

        # Log event
        log_event("info", "prepare_data", "complete", 
                 patient_count=len(self.patient_examples),
                 s3_bucket=self.s3_bucket,
                 s3_prefix=self.s3_prefix)
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        
        This method is called on every GPU.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Load feature stats from disk
        if self.feature_stats is None:
            with open(self.feature_stats_path, "r") as f:
                self.feature_stats = json.load(f)
                            
        # Split dataset for training and validation
        train_data, test_data = split_riskformer_data(
            examples=self.patient_examples,
            label_var=self.split_var,
            positive_label="H",
            test_split_ratio=self.test_split
        )
        
        if stage == "fit":
            # Create training and validation datasets
            self.train_dataset = RiskFormerDataset(
                patient_examples=train_data,
                max_dim=self.max_dim,
                overlap=self.overlap,
                s3_cache=self.s3_cache,
                include_labels=self.include_labels,
            )
            
            if self.val_split > 0:
                # random split train data into train and validation
                num_val = int(len(train_data) * self.val_split)
                num_train = len(train_data) - num_val
                self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset, 
                    [num_train, num_val]
                )
        elif stage == "test":
            self.test_dataset = RiskFormerDataset(
                patient_examples=test_data,
                max_dim=self.max_dim,
                overlap=self.overlap,
                s3_cache=self.s3_cache,
                include_labels=self.include_labels,
            )
    
    def teardown(self, stage: Optional[str] = None):
        """
        Tear down datasets for training, validation, and testing.
        
        This method is called on every GPU.
        """
        if stage == "fit":
            self.train_dataset = None
            self.val_dataset = None
        elif stage == "test":
            self.test_dataset = None
        self.feature_stats = None

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


def load_dataset_metadata(metadata_file: str) -> Tuple[Set[str], Dict[str, dict]]:
    """
    Load metadata file and return a set of slide IDs and a dictionary of slide metadata.

    Args:
        metadata_file: Path to metadata file

    Returns:
        target_slides: Set of slide IDs to include in dataset
        slide_data: Dictionary of slide metadata
    """
    slide_ids = None
    if metadata_file and os.path.isfile(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                slide_data = json.load(f)
            slide_ids = set(slide_data.keys())
            log_event("info", "load_metadata", "success",
                     total_slides=len(slide_ids))
        except Exception as e:
            log_event("error", "load_metadata", "error",
                     error=str(e))
            raise RuntimeError(f"Failed to load metadata file: {e}")
    return slide_ids, slide_data


def create_slide_examples(
    s3_client: boto3.client,
    s3_bucket: str,
    s3_prefix: str,
    slide_ids: Optional[Set[str]] = None,
    slide_data: Optional[Dict[str, dict]] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Create a list of slide examples from the S3 bucket.
    
    Args:
        s3_client (boto3.client): S3 client
        s3_bucket (str): S3 bucket
        s3_prefix (str): S3 prefix
        slide_ids (Optional[Set[str]]): Set of slide IDs to include in dataset
        slide_data (Optional[Dict[str, dict]]): Dictionary of slide metadata
    
    Returns:
        examples (List[Tuple[str, str, str, str]]): List of slide examples
    """
    # List all files in the prefix
    all_files = list_bucket_files(s3_client, s3_bucket, s3_prefix)
    log_event("debug", "list_s3_bucket_files", "success",
              s3_bucket=s3_bucket, 
              s3_prefix=s3_prefix, 
              file_count=len(all_files))
    
    # Find all unique basenames that have both coords and features files
    coords_files = {k for k in all_files.keys() if k.endswith("_coords.h5")}
    features_files = {k for k in all_files.keys() if k.endswith("_features.h5")}
    
    # Extract s3://bucket/ basenames and slide exists in metadata
    s3_slide_ids = set()
    for coords_file in coords_files:
        s3_slide_id = coords_file[:-len("_coords.h5")]
        features_file = f"{s3_slide_id}_features.h5"
        if features_file in features_files:
            if slide_ids is not None:
                slide_id = os.path.basename(s3_slide_id)
                if slide_id in slide_ids:
                    s3_slide_ids.add(s3_slide_id)
            else:
                s3_slide_ids.add(s3_slide_id)
    
    # Log statistics about found files
    if slide_ids is not None:
        log_event("info", "find_complete_slide_sets", "success",
                  total_files=len(all_files),
                  total_slides_in_metadata=len(slide_ids),
                  preprocessed_slides_found=len(s3_slide_ids),
                  missing_slides=len(slide_ids - {os.path.basename(b) for b in s3_slide_ids}))
    else:
        log_event("info", "find_complete_slide_sets", "success",
                  total_files=len(all_files),
                  coords_files=len(coords_files),
                  features_files=len(features_files),
                  complete_sets=len(s3_slide_ids))
    
    if not s3_slide_ids:
        raise RuntimeError(f"No complete feature sets found in s3://{s3_bucket}/{s3_prefix}")
    
    # Create feature pairs
    slide_examples = {}
    for s3_slide_id in sorted(s3_slide_ids):  # Sort for consistent ordering
        slide_id = os.path.basename(s3_slide_id)
        patient_id = slide_data[slide_id]["patient"]
        coords_path = f"s3://{s3_bucket}/{s3_slide_id}_coords.h5"
        features_path = f"s3://{s3_bucket}/{s3_slide_id}_features.h5"

        slide_examples[slide_id] = {"patient_id": patient_id, "coords_path": coords_path, "features_path": features_path}
    return slide_examples


def create_patient_examples(
    s3_client: boto3.client,
    s3_bucket: str,
    s3_prefix: str,
    slide_ids: Optional[Set[str]] = None,
    slide_data: Optional[Dict[str, dict]] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Create a list of patient examples from the S3 bucket.
    """
    
    slide_examples = create_slide_examples(s3_client, s3_bucket, s3_prefix, slide_ids, slide_data)
    all_patients_slides = {}
    for slide_id, data in slide_examples.items():
        patient_id = data["patient_id"]
        if patient_id not in all_patients_slides:
            all_patients_slides[patient_id] = []
        all_patients_slides[patient_id].append(slide_id)

    patient_examples = {}
    for patient_id in all_patients_slides:
        patient_slides = all_patients_slides[patient_id]
        patient_coords_paths = [slide_examples[slide_id]["coords_path"] for slide_id in patient_slides]
        patient_features_paths = [slide_examples[slide_id]["features_path"] for slide_id in patient_slides]

        patient_examples[patient_id] = {}
        patient_examples[patient_id]["coords_paths"] = patient_coords_paths
        patient_examples[patient_id]["features_paths"] = patient_features_paths


        slides_odx_train = [float(slide_data[slide_id]["odx_train"]) for slide_id in patient_slides]
        slides_odx85 = [int(slide_data[slide_id]["odx85"] == "H") for slide_id in patient_slides]
        slides_mphr = [int(slide_data[slide_id]["mphr"] == "H") for slide_id in patient_slides]
        slides_dfm = [float(slide_data[slide_id]["Disease_Free_Months"]) for slide_id in patient_slides]
        slides_necrosis = [int(slide_data[slide_id]["Necrosis"] == "Present") for slide_id in patient_slides]
        slides_pleo = [int(slide_data[slide_id]["Pleomorph"]) for slide_id in patient_slides]

        patient_odx_train = max(slides_odx_train)
        patient_odx85 = max(slides_odx85)
        patient_mphr = max(slides_mphr)
        patient_dfm = min(slides_dfm)
        patient_necrosis = max(slides_necrosis)
        patient_pleo = max(slides_pleo)

        patient_examples[patient_id]["odx_train"] = patient_odx_train
        patient_examples[patient_id]["odx85"] = patient_odx85
        patient_examples[patient_id]["mphr"] = patient_mphr
        patient_examples[patient_id]["dfm"] = patient_dfm
        patient_examples[patient_id]["necrosis"] = patient_necrosis
        patient_examples[patient_id]["pleo"] = patient_pleo

    return patient_examples


def create_riskformer_dataset(
    s3_bucket: str,
    s3_prefix: str = "",
    max_dim: int = 32,
    overlap: float = 0.0,
    metadata_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    config_path: Optional[str] = None,
    include_labels: Optional[List[str]] = None,
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
        config_path: Path to config file (optional, will override include_labels)
        include_labels: List of label names to include (optional)
        
    Returns:
        RiskFormerDataset: Dataset containing sparse tensors for each slide
        
    Example:
        dataset = create_riskformer_dataset(
            s3_bucket="tcga-riskformer-data-2025",
            s3_prefix="preprocessed/uni/uni2-h",
            metadata_file="resources/riskformer_slide_samples.json",
            config_path="configs/training/riskformer_config.yaml"
        )
    """
    log_event("debug", "create_riskformer_dataset", "started",
              s3_bucket=s3_bucket, 
              s3_prefix=s3_prefix,
              metadata_file=metadata_file,
              config_path=config_path)
    
    # Initialize S3 client
    s3_client = initialize_s3_client(profile_name, region_name)
    if s3_client is None:
        raise RuntimeError("Failed to initialize S3 client")
    
    # Load metadata file if provided
    slide_ids, slide_data = load_dataset_metadata(metadata_file)
    patient_examples = create_patient_examples(s3_client, s3_bucket, s3_prefix, slide_ids, slide_data)
        
    # Create dataset
    try:
        dataset = RiskFormerDataset(
            patient_examples=patient_examples,
            max_dim=max_dim,
            overlap=overlap,
            cache_dir=cache_dir,
            config_path=config_path,
            include_labels=include_labels,
        )
        log_event("info", "create_riskformer_dataset", "success",
                  slide_count=len(dataset),
                  feature_dim=dataset.feature_dim,
                  cache_dir=str(dataset.s3_cache.cache_dir),
                  include_labels=dataset.include_labels)
        return dataset
    except Exception as e:
        log_event("error", "create_riskformer_dataset", "error",
                  error=str(e))
        raise e


