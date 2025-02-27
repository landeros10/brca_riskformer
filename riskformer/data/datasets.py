import numpy as np
from PIL import Image
import zarr
import openslide

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from riskformer.utils.data_utils import sample_slide_image
from riskformer.utils.randstainna import RandStainNA

yaml_file = '/home/ubuntu/notebooks/cpc_hist/src/CRC_LAB_randomTrue_n0.yaml'
stain_normalizer = RandStainNA(yaml_file, std_hyper=-1.0)


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


