import os
import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader, random_split

from riskformer.data.datasets import RiskFormerDataset, create_riskformer_dataset


class RiskFormerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for RiskFormer datasets.
    
    This module handles the loading, splitting, and preparation of data for training,
    validation, and testing with PyTorch Lightning.
    """
    
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
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Dataset parameters
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.max_dim = max_dim
        self.overlap = overlap
        self.metadata_file = metadata_file
        self.cache_dir = cache_dir
        self.profile_name = profile_name
        self.region_name = region_name
        
        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.pin_memory = pin_memory
        
        # Datasets
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Download and prepare data. This method is called only once and on 1 GPU.
        
        This is where we can download data, preprocess it, etc.
        """
        # Nothing to do here as the dataset will handle downloading when created
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        
        This method is called on every GPU.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Create the dataset if it doesn't exist
        if self.dataset is None:
            self.dataset = create_riskformer_dataset(
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                max_dim=self.max_dim,
                overlap=self.overlap,
                metadata_file=self.metadata_file,
                cache_dir=self.cache_dir,
                profile_name=self.profile_name,
                region_name=self.region_name
            )
        
        # Split the dataset
        if stage == 'fit' or stage is None:
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * self.val_split)
            test_size = int(dataset_size * self.test_split)
            train_size = dataset_size - val_size - test_size
            
            # Use random_split with generator for reproducibility
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, 
                [train_size, val_size, test_size],
                generator=generator
            )
    
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