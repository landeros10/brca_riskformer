import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.data.data_utils import sample_slide_image


class SingleSlideDataset(Dataset):
    """
    PyTorch dataset for a single slide at specified sample points.
    
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
            slide_obj,
            slide_metadata: dict,
            sample_coords: np.ndarray,
            sample_size: int,
            output_size: int,
            transform=None,
        ):
        self.slide_obj = slide_obj
        self.slide_metadata = slide_metadata
        self.sample_coords = sample_coords
        self.sample_size = sample_size
        self.output_size = output_size
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sample_coords)

    def __getitem__(self, idx):
        x, y = self.sample_coords[idx]
        image = self.sample_slide(x, y)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image) # (C, H, W), scaled to [0, 1]

        return image

    def sample_slide(self, x, y):
        """
        Samples a slide at the given coordinates.
        
        Args:
            x (int): x coordinate of the sample.
            y (int): y coordinate of the sample.
        
        Returns:
            image (PIL.Image): sampled image.
        """
        image = sample_slide_image(self.slide_obj, x, y, self.sample_size)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.resize((self.output_size, self.output_size), Image.BILINEAR)
        return image


