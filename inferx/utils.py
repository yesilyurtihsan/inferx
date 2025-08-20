"""Image processing and utility functions for InferX"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, List, Any
import logging

from .exceptions import (
    InputNotFoundError,
    InputInvalidFormatError,
    SecurityError,
    PathTraversalError,
    ErrorCode
)


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image preprocessing utilities for model inference"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in BGR format (OpenCV default)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        
        # Security check: prevent path traversal
        try:
            resolved_path = image_path.resolve()
            # Basic path traversal check
            if ".." in str(image_path):
                raise PathTraversalError(str(image_path))
        except Exception as e:
            if isinstance(e, PathTraversalError):
                raise
            raise SecurityError(
                message=f"Security error while processing path: {image_path}",
                error_code=ErrorCode.PATH_TRAVERSAL_DETECTED,
                original_error=e,
                context={"path": str(image_path)}
            )
        
        if not image_path.exists():
            raise InputNotFoundError(str(image_path))
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise InputInvalidFormatError(
                input_path=str(image_path),
                expected_formats=["jpg", "jpeg", "png", "bmp", "tiff"],
                context={"file_extension": image_path.suffix}
            )
        
        logger.debug(f"Loaded image {image_path} with shape {image.shape}")
        return image
    
    @staticmethod
    def resize_image(
        image: np.ndarray, 
        target_size: Union[int, Tuple[int, int]], 
        maintain_aspect: bool = True,
        fill_color: Tuple[int, int, int] = (114, 114, 114)
    ) -> np.ndarray:
        """Resize image to target size
        
        Args:
            image: Input image array
            target_size: Target size as (width, height) or single int for square
            maintain_aspect: Whether to maintain aspect ratio with letterboxing
            fill_color: Fill color for letterboxing (BGR format)
            
        Returns:
            Resized image array
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        if not maintain_aspect:
            # Simple resize without aspect ratio preservation
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Resized image from {w}x{h} to {target_w}x{target_h}")
            return resized
        
        # Resize with aspect ratio preservation (letterboxing)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image
        letterboxed = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        letterboxed[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        logger.debug(f"Letterboxed image from {w}x{h} to {target_w}x{target_h} with scale {scale:.3f}")
        return letterboxed
    
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: float = 1.0 / 255.0
    ) -> np.ndarray:
        """Normalize image with mean and std
        
        Args:
            image: Input image array
            mean: Mean values for normalization (RGB order)
            std: Standard deviation values for normalization (RGB order)
            scale: Scale factor to apply before normalization
            
        Returns:
            Normalized image array as float32
        """
        # Convert to float and scale
        normalized = image.astype(np.float32) * scale
        
        # Apply mean and std normalization
        mean_array = np.array(mean, dtype=np.float32)
        std_array = np.array(std, dtype=np.float32)
        
        # OpenCV uses BGR, so we need to reverse the order
        mean_bgr = mean_array[::-1]
        std_bgr = std_array[::-1]
        
        normalized = (normalized - mean_bgr) / std_bgr
        
        logger.debug(f"Normalized image with mean={mean_bgr} and std={std_bgr}")
        return normalized
    
    @staticmethod
    def convert_color_format(image: np.ndarray, target_format: str = "RGB") -> np.ndarray:
        """Convert image color format
        
        Args:
            image: Input image array
            target_format: Target color format ("RGB", "BGR", "GRAY")
            
        Returns:
            Converted image array
            
        Raises:
            ValueError: If target format is not supported
        """
        if len(image.shape) == 2:
            # Already grayscale
            current_format = "GRAY"
        elif image.shape[2] == 3:
            # Assume BGR (OpenCV default)
            current_format = "BGR"
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        target_format = target_format.upper()
        
        if current_format == target_format:
            return image
        
        if current_format == "BGR" and target_format == "RGB":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif current_format == "RGB" and target_format == "BGR":
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif current_format == "BGR" and target_format == "GRAY":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif current_format == "RGB" and target_format == "GRAY":
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif current_format == "GRAY" and target_format == "BGR":
            converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif current_format == "GRAY" and target_format == "RGB":
            converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Conversion from {current_format} to {target_format} not supported")
        
        logger.debug(f"Converted image from {current_format} to {target_format}")
        return converted
    
    @staticmethod
    def prepare_model_input(
        image: np.ndarray,
        transpose_to_chw: bool = True,
        add_batch_dim: bool = True
    ) -> np.ndarray:
        """Prepare image for model input (transpose and add batch dimension)
        
        Args:
            image: Input image array (H, W, C)
            transpose_to_chw: Whether to transpose from HWC to CHW format
            add_batch_dim: Whether to add batch dimension
            
        Returns:
            Model-ready image array
        """
        processed = image.copy()
        
        if transpose_to_chw and len(processed.shape) == 3:
            # Transpose from HWC to CHW
            processed = np.transpose(processed, (2, 0, 1))
            logger.debug("Transposed image from HWC to CHW")
        
        if add_batch_dim:
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
            logger.debug(f"Added batch dimension, final shape: {processed.shape}")
        
        return processed


class FileUtils:
    """File and path utilities"""
    
    @staticmethod
    def is_image_file(file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is supported image format
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        file_path = Path(file_path)
        return file_path.suffix.lower() in image_extensions
    
    @staticmethod
    def get_image_files(directory: Union[str, Path]) -> List[Path]:
        """Get all image files from directory
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        
        image_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and FileUtils.is_image_file(file_path):
                image_files.append(file_path)
        
        image_files.sort()
        logger.debug(f"Found {len(image_files)} image files in {directory}")
        return image_files
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Create directory if it doesn't exist
        
        Args:
            directory: Directory path to create
            
        Returns:
            Path object of the directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def preprocess_for_inference(
    image_path: Union[str, Path],
    target_size: Union[int, Tuple[int, int]],
    normalize: bool = True,
    color_format: str = "RGB",
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Complete preprocessing pipeline for model inference
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        normalize: Whether to apply normalization
        color_format: Target color format
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Preprocessed image ready for model inference
    """
    # Load image
    image = ImageProcessor.load_image(image_path)
    
    # Resize image
    image = ImageProcessor.resize_image(image, target_size, maintain_aspect=True)
    
    # Convert color format
    image = ImageProcessor.convert_color_format(image, color_format)
    
    # Normalize if requested
    if normalize:
        image = ImageProcessor.normalize_image(image, mean, std)
    
    # Prepare for model input
    image = ImageProcessor.prepare_model_input(image)
    
    logger.info(f"Preprocessed image {image_path} to shape {image.shape}")
    return image