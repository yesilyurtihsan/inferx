"""Tests for YOLO inferencer"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from inferx.inferencers.yolo import YOLOInferencer


def test_yolo_inferencer_initialization():
    """Test YOLO inferencer initialization"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
    
    try:
        # Mock the ONNX Runtime session
        with patch('inferx.inferencers.yolo.ort.InferenceSession') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.get_inputs.return_value = [Mock(name='input', shape=[1, 3, 640, 640])]
            mock_session_instance.get_outputs.return_value = [Mock(name='output')]
            mock_session_instance.get_providers.return_value = ['CPUExecutionProvider']
            mock_session.return_value = mock_session_instance
            
            inferencer = YOLOInferencer(model_path)
            
            assert inferencer.model_path == model_path
            assert "class_names" in inferencer.config
            assert len(inferencer.config["class_names"]) == 80  # COCO classes
            assert inferencer.config["input_size"] == 640
    finally:
        if model_path.exists():
            model_path.unlink()


def test_letterbox_function():
    """Test letterbox resize function"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
    
    try:
        # Mock the ONNX Runtime session
        with patch('inferx.inferencers.yolo.ort.InferenceSession') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.get_inputs.return_value = [Mock(name='input', shape=[1, 3, 640, 640])]
            mock_session_instance.get_outputs.return_value = [Mock(name='output')]
            mock_session_instance.get_providers.return_value = ['CPUExecutionProvider']
            mock_session.return_value = mock_session_instance
            
            inferencer = YOLOInferencer(model_path)
            
            # Create a test image (100x150)
            test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            
            # Test letterbox function
            letterboxed, ratio, pad = inferencer._letterbox(test_image, (640, 640))
            
            assert letterboxed.shape == (640, 640, 3)
            assert isinstance(ratio, tuple)
            assert isinstance(pad, tuple)
    finally:
        if model_path.exists():
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])