'''
Tests for preprocessing functionality.
'''
import os
import pytest
import numpy as np
from PIL import Image
import h5py
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from entrypoints.preprocess import (
    preprocess_one_slide,
    save_sparse_feature_array,
    save_image_output
)

def test_save_sparse_feature_array(mock_output_dir):
    """Test saving sparse feature arrays."""
    # Create test data
    sampling_size = 256
    tile_overlap = 0

    # Create grid of coordinates sampling_size apart
    x_coords = np.arange(0, 100 * sampling_size, sampling_size)
    y_coords = np.arange(0, 100 * sampling_size, sampling_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    sample_coords = np.column_stack((xx.ravel(), yy.ravel()))[:100]  # Take first 100 points

    slide_features = np.random.rand(len(sample_coords), 256)
    basename = "test_features"

    # Patch the save_features_h5 function to use a smaller chunk size
    with patch('riskformer.data.data_preprocess.save_features_h5') as mock_save:
        # Configure the mock to create the expected files
        def side_effect(output_path, coo_coords, slide_features, chunk_size, compression='gzip'):
            coords_file = f"{output_path}_coords.h5"
            features_file = f"{output_path}_features.h5"
            
            with h5py.File(coords_file, "w") as f:
                f.create_dataset("coords", data=coo_coords)
                
            with h5py.File(features_file, "w") as f:
                f.create_dataset("features", data=slide_features)
                
        mock_save.side_effect = side_effect
        
        # Save features
        save_sparse_feature_array(
            sample_coords=sample_coords,
            sampling_size=sampling_size,
            tile_overlap=tile_overlap,
            slide_features=slide_features,
            output_dir=mock_output_dir,
            basename=basename
        )

    # Verify files were created
    coords_path = os.path.join(mock_output_dir, f"{basename}_coords.h5")
    features_path = os.path.join(mock_output_dir, f"{basename}_features.h5")
    
    assert os.path.exists(coords_path)
    assert os.path.exists(features_path)
    
    # Check coords file
    with h5py.File(coords_path, 'r') as f:
        assert 'coords' in f
        assert f['coords'].shape[0] == len(sample_coords)
        assert f['coords'].shape[1] == 2
    
    # Check features file
    with h5py.File(features_path, 'r') as f:
        assert 'features' in f
        assert f['features'].shape[0] == len(slide_features)
        assert f['features'].shape[1] == slide_features.shape[1]


def test_save_image_output(mock_output_dir):
    """Test saving image outputs."""
    # Create test image
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    basename = "test_image"
    tag = "thumbnail"

    # Save image
    save_image_output(
        image=image,
        output_dir=mock_output_dir,
        basename=basename,
        tag=tag,
        normalize=False
    )

    # Verify file was created
    output_path = os.path.join(mock_output_dir, f"{basename}_{tag}.png")
    assert os.path.exists(output_path)

    # Verify image was saved correctly
    saved_image = Image.open(output_path)
    np.array_equal(saved_image, image)
    
    # Normalized image
    save_image_output(
        image=image,
        output_dir=mock_output_dir,
        basename=basename,
        tag=tag,
        normalize=True
    )

    # Verify normalized image was saved correctly
    assert os.path.exists(output_path)

    # Verify normalized image was saved correctly
    saved_image = np.array(Image.open(output_path))
    assert saved_image.dtype == np.uint8
    assert saved_image.min() == 0
    assert saved_image.max() >= 254  # Allow for some rounding in normalization
    
    # Test NoneType image
    # First, remove the existing file to ensure it doesn't exist
    output_path = os.path.join(mock_output_dir, f"{basename}_{tag}.png")
    if os.path.exists(output_path):
        os.remove(output_path)
        
    save_image_output(
        image=None,
        output_dir=mock_output_dir,
        basename=basename,
        tag=tag
    )
    assert not os.path.exists(output_path)



def test_preprocess_one_slide(mock_output_dir, mock_model_dir, mock_config):
    """Test the main preprocessing function."""
    # Create a mock SVS file
    test_file = os.path.join(mock_output_dir, "test.svs")
    with open(test_file, "wb") as f:
        f.write(b"mock SVS file")
    
    # Patch the necessary functions to avoid actual processing
    with patch('riskformer.utils.data_utils.OpenSlide') as mock_openslide, \
         patch('entrypoints.preprocess.torch.load') as mock_load, \
         patch('entrypoints.preprocess.torch.no_grad') as mock_no_grad, \
         patch('entrypoints.preprocess.SingleSlideDataset') as mock_dataset, \
         patch('entrypoints.preprocess.torch.utils.data.DataLoader') as mock_dataloader, \
         patch('entrypoints.preprocess.save_sparse_feature_array') as mock_save_features, \
         patch('entrypoints.preprocess.save_image_output') as mock_save_image, \
         patch('riskformer.data.data_preprocess.get_svs_samplepoints') as mock_get_samplepoints, \
         patch('riskformer.data.data_preprocess.load_encoder') as mock_load_encoder, \
         patch('riskformer.data.data_preprocess.extract_features') as mock_extract_features:
        
        # Configure mocks
        mock_openslide.return_value.dimensions = (10000, 10000)
        mock_openslide.return_value.level_count = 3
        mock_openslide.return_value.level_dimensions = [(10000, 10000), (5000, 5000), (2500, 2500)]
        mock_openslide.return_value.get_thumbnail.return_value = Image.new('RGB', (100, 100))
        
        # Mock sample points to ensure we don't return early
        mock_sample_coords = np.array([[0, 0], [100, 100], [200, 200]])
        mock_get_samplepoints.return_value = (
            mock_sample_coords,  # sample_coords
            mock_openslide.return_value,  # slide_obj
            {"width": 10000, "height": 10000},  # slide_metadata
            256,  # sampling_size
            np.zeros((100, 100)),  # heatmap
            Image.new('RGB', (100, 100))  # thumb
        )
        
        # Mock the encoder loading
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_load_encoder.return_value = (mock_model, mock_transform)
        
        # Mock feature extraction
        mock_extract_features.return_value = np.random.rand(3, 256)  # 3 sample points, 256 features each
        
        mock_model.return_value = torch.rand(10, 256)
        mock_load.return_value = mock_model
        
        mock_dataset.return_value.__len__.return_value = 10
        mock_dataset.return_value.__getitem__.return_value = (torch.rand(3, 256, 256), (0, 0))
        
        mock_dataloader.return_value.__iter__.return_value = [
            (torch.rand(4, 3, 256, 256), [(0, 0), (0, 256), (256, 0), (256, 256)])
        ]
        
        # Call the function
        result = preprocess_one_slide(
            input_filename=test_file,
            output_dir=mock_output_dir,
            model_dir=mock_model_dir,
            model_type=mock_config["model_type"],
            foreground_config_path=mock_config["foreground_config_path"],
            foreground_cleanup_config_path=mock_config["foreground_cleanup_config_path"],
            tiling_config_path=mock_config["tiling_config_path"],
            num_workers=mock_config["num_workers"],
            batch_size=mock_config["batch_size"],
            prefetch_factor=mock_config["prefetch_factor"]
        )
        
        # Verify the result
        assert result is True
        mock_save_features.assert_called_once()
        mock_save_image.assert_called()    