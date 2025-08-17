"""Simple tests for InferX utilities"""

import pytest
import numpy as np
import tempfile
import cv2
from pathlib import Path

from inferx.utils import ImageProcessor, FileUtils, preprocess_for_inference


def test_resize_image():
    """Test resizing image"""
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    
    # Test square resize
    resized = ImageProcessor.resize_image(test_image, 224)
    assert resized.shape == (224, 224, 3)
    
    # Test rectangular resize
    resized = ImageProcessor.resize_image(test_image, (320, 240))
    assert resized.shape == (240, 320, 3)


def test_file_utils():
    """Test file utilities"""
    # Test image file detection
    assert FileUtils.is_image_file("test.jpg")
    assert FileUtils.is_image_file("test.png")
    assert not FileUtils.is_image_file("test.txt")


def test_preprocess_for_inference():
    """Test complete preprocessing pipeline"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_path = temp_path / "test_image.jpg"
        
        # Create and save a test image
        test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        
        # Test preprocessing
        processed = preprocess_for_inference(
            image_path=image_path,
            target_size=224,
            normalize=True,
            color_format="RGB"
        )
        
        # Check output shape
        assert processed.shape == (1, 3, 224, 224)  # (B, C, H, W)
        assert processed.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])