"""Pytest configuration and shared fixtures for InferX tests"""

import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_onnx_model_path(temp_dir):
    """Provide a mock ONNX model file path"""
    model_path = temp_dir / "test_model.onnx"
    model_path.touch()  # Create empty file
    return model_path


@pytest.fixture
def mock_openvino_model_path(temp_dir):
    """Provide a mock OpenVINO model file path"""
    xml_path = temp_dir / "test_model.xml"
    bin_path = temp_dir / "test_model.bin"
    
    # Create empty files
    xml_path.touch()
    bin_path.touch()
    
    return xml_path


@pytest.fixture
def mock_yolo_onnx_path(temp_dir):
    """Provide a mock YOLO ONNX model file path"""
    model_path = temp_dir / "yolov8n.onnx"
    model_path.touch()
    return model_path


@pytest.fixture
def mock_yolo_openvino_path(temp_dir):
    """Provide a mock YOLO OpenVINO model file path"""
    xml_path = temp_dir / "yolov8n.xml"
    bin_path = temp_dir / "yolov8n.bin"
    
    xml_path.touch()
    bin_path.touch()
    
    return xml_path


@pytest.fixture
def mock_image_path(temp_dir):
    """Provide a mock image file path"""
    image_path = temp_dir / "test_image.jpg"
    
    # Create a real test image
    test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), test_image)
    
    return image_path


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary"""
    return {
        "device": "cpu",
        "runtime": "onnx",
        "preprocessing": {
            "target_size": [224, 224],
            "normalize": True,
            "color_format": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "session_options": {
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "inter_op_num_threads": 4,
            "intra_op_num_threads": 4
        }
    }


@pytest.fixture
def test_image_array():
    """Provide a test image as numpy array"""
    return np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def batch_image_paths(temp_dir):
    """Provide multiple mock image files for batch testing"""
    image_paths = []
    
    for i in range(3):
        image_path = temp_dir / f"test_image_{i}.jpg"
        test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        image_paths.append(image_path)
    
    return image_paths


@pytest.fixture
def mock_inference_result():
    """Provide a mock inference result"""
    return {
        "raw_outputs": [[0.1, 0.2, 0.7]],
        "output_shapes": [[1, 3]],
        "num_outputs": 1,
        "model_type": "onnx_generic",
        "output_0_shape": [1, 3],
        "output_0_dtype": "float32",
        "output_0_min": 0.1,
        "output_0_max": 0.7,
        "output_0_mean": 0.33
    }


@pytest.fixture
def mock_yaml_config(temp_dir):
    """Provide a mock YAML configuration file"""
    config_content = """
device: gpu
preprocessing:
  target_size: [640, 640]
  normalize: true
  color_format: RGB
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

session_options:
  graph_optimization_level: ORT_ENABLE_ALL
  inter_op_num_threads: 8

output:
  format: json
"""
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


@pytest.fixture
def mock_inferx_config(temp_dir):
    """Provide a mock InferX configuration file"""
    config_content = """
# Test InferX Configuration
model_detection:
  yolo_keywords:
    - "yolo"
    - "yolov8"
    - "test_yolo"

device_mapping:
  auto: "CPU"
  gpu: "GPU"
  cpu: "CPU"

model_defaults:
  yolo:
    input_size: 640
    confidence_threshold: 0.25
    nms_threshold: 0.45
    class_names:
      - "person"
      - "car"
      - "test_class"

performance_presets:
  test_preset:
    openvino:
      performance_hint: "LATENCY"
      num_streams: 1
    onnx:
      providers: ["CPUExecutionProvider"]
"""
    
    config_path = temp_dir / "inferx_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


@pytest.fixture
def mock_openvino_config():
    """Provide a mock OpenVINO configuration dictionary"""
    return {
        "device": "CPU",
        "runtime": "openvino",
        "performance_hint": "LATENCY",
        "num_streams": 1,
        "num_threads": 4,
        "preprocessing": {
            "target_size": [640, 640],
            "normalize": True,
            "color_format": "RGB"
        }
    }


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring external dependencies"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "cli: Tests for CLI functionality"
    )
    config.addinivalue_line(
        "markers", "openvino: Tests for OpenVINO functionality"
    )
    config.addinivalue_line(
        "markers", "config: Tests for configuration management"
    )


# Custom assertions
def assert_valid_inference_result(result):
    """Assert that a result has the expected inference result structure"""
    required_keys = ["raw_outputs", "output_shapes", "num_outputs", "model_type"]
    
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    assert isinstance(result["raw_outputs"], list)
    assert isinstance(result["output_shapes"], list)
    assert isinstance(result["num_outputs"], int)
    assert isinstance(result["model_type"], str)
    assert result["num_outputs"] == len(result["raw_outputs"])
    assert result["num_outputs"] == len(result["output_shapes"])


def assert_valid_model_info(info):
    """Assert that model info has the expected structure"""
    required_keys = ["model_path", "model_size", "status"]
    
    for key in required_keys:
        assert key in info, f"Missing required key: {key}"
    
    assert isinstance(info["model_path"], str)
    assert isinstance(info["status"], str)
    assert info["status"] in ["loaded", "not_loaded"]


# Add custom assertions to pytest namespace
pytest.assert_valid_inference_result = assert_valid_inference_result
pytest.assert_valid_model_info = assert_valid_model_info