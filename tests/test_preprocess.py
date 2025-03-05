'''
Tests for preprocessing functionality.
'''
import os
import pytest
import numpy as np
from PIL import Image
import h5py
from pathlib import Path

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

    # Save features
    save_sparse_feature_array(
        sample_coords=sample_coords,
        sampling_size=sampling_size,
        tile_overlap=tile_overlap,
        slide_features=slide_features,
        output_dir=mock_output_dir,
        basename=basename
    )

    # Verify file was created
    for file_tag, arr in zip(["coords", "features"], [sample_coords, slide_features]):
        output_path = os.path.join(mock_output_dir, f"{basename}_{file_tag}.h5")
        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as f:
            assert file_tag in f
            assert f[file_tag].shape == arr.shape


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
    assert saved_image.max() == 255
    
    # Test NoneType image
    save_image_output(
        image=None,
        output_dir=mock_output_dir,
        basename=basename,
        tag=tag
    )
    assert not os.path.exists(output_path)



def test_preprocess_one_slide(mock_output_dir, mock_model_dir, mock_config):
    """Test the main preprocessing function."""
    test_file = "./resources/test.svs"
    assert os.path.exists(test_file), f"Test file does not exist: {test_file}"
    assert test_file.endswith(".svs"), f"Test file must have .svs extension: {test_file}"
    
    # Process the slide
    preprocess_one_slide(
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

    # Verify output
    basename = os.path.basename(test_file).split(".svs")[0]
    thumb_path = os.path.join(mock_output_dir, f"{basename}_thumbnail.png")
    heatmap_path = os.path.join(mock_output_dir, f"{basename}_heatmap.png")
    coords_path = os.path.join(mock_output_dir, f"{basename}_coords.h5")
    features_path = os.path.join(mock_output_dir, f"{basename}_features.h5")

    assert os.path.exists(thumb_path)
    assert os.path.exists(heatmap_path)
    assert os.path.exists(coords_path)
    assert os.path.exists(features_path)

    # Files not empty
    assert os.path.getsize(thumb_path) > 0, f"Thumbnail file is empty: {thumb_path}"
    assert os.path.getsize(heatmap_path) > 0, f"Heatmap file is empty: {heatmap_path}"
    assert os.path.getsize(coords_path) > 0, f"Coords file is empty: {coords_path}"
    assert os.path.getsize(features_path) > 0, f"Features file is empty: {features_path}"

    # Test output formats
    with h5py.File(coords_path, 'r') as f:
        assert 'coords' in f
        assert f['coords'].dtype == np.float32
        assert len(f['coords'].shape) == 2
        assert f['coords'].shape[1] == 2

    with h5py.File(features_path, 'r') as f:
        assert 'features' in f
        assert f['features'].dtype == np.float32
        assert len(f['features'].shape) == 2

    thumb_img = Image.open(thumb_path)
    heatmap_img = Image.open(heatmap_path)
    assert thumb_img.mode in ["RGB", "RGBA"]
    assert heatmap_img.mode in ["RGB", "RGBA"]

    assert thumb_img.size[0] > 0 and thumb_img.size[1] > 0
    assert heatmap_img.size[0] > 0 and heatmap_img.size[1] > 0

    with h5py.File(coords_path, 'r') as f_coords, h5py.File(features_path, 'r') as f_features:
        n_features = f_features['features'].shape[0]
        assert f_coords['coords'].shape[0] == n_features
        assert f_features['features'].shape[0] == n_features    