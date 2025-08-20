"""Tests for configuration management system"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from inferx.config import InferXConfig, get_config, validate_config, create_user_config_template


@pytest.mark.config
@pytest.mark.unit
class TestInferXConfig:
    """Test InferXConfig class functionality"""
    
    def test_config_initialization(self):
        """Test basic config initialization"""
        config = InferXConfig()
        assert config is not None
        
        # Should have default values
        assert "model_detection" in config._config
        assert "device_mapping" in config._config
        assert "model_defaults" in config._config
    
    def test_config_get_method(self):
        """Test config get method with dot notation"""
        config = InferXConfig()
        
        # Test getting existing values
        yolo_keywords = config.get("model_detection.yolo_keywords", [])
        assert isinstance(yolo_keywords, list)
        
        # Test getting non-existing values with default
        non_existing = config.get("non.existing.key", "default")
        assert non_existing == "default"
    
    def test_config_set_method(self):
        """Test config set method with dot notation"""
        config = InferXConfig()
        
        # Set a new value
        config.set("test.setting", "test_value")
        assert config.get("test.setting") == "test_value"
        
        # Set nested value
        config.set("test.nested.value", 42)
        assert config.get("test.nested.value") == 42
    
    def test_model_type_detection(self, temp_dir):
        """Test model type detection based on filename"""
        config = InferXConfig()
        
        # Test YOLO ONNX detection
        yolo_onnx_path = temp_dir / "yolov8n.onnx"
        model_type = config.detect_model_type(yolo_onnx_path)
        assert model_type == "yolo_onnx"
        
        # Test YOLO OpenVINO detection
        yolo_openvino_path = temp_dir / "yolov8n.xml"
        model_type = config.detect_model_type(yolo_openvino_path)
        assert model_type == "yolo_openvino"
        
        # Test generic ONNX
        generic_onnx_path = temp_dir / "resnet.onnx"
        model_type = config.detect_model_type(generic_onnx_path)
        assert model_type == "onnx"
        
        # Test generic OpenVINO
        generic_openvino_path = temp_dir / "model.xml"
        model_type = config.detect_model_type(generic_openvino_path)
        assert model_type == "openvino"
    
    def test_get_model_defaults(self):
        """Test getting model-specific defaults"""
        config = InferXConfig()
        
        yolo_defaults = config.get_model_defaults("yolo")
        assert isinstance(yolo_defaults, dict)
        assert "input_size" in yolo_defaults
        assert "confidence_threshold" in yolo_defaults
        
        # Test non-existing model type
        empty_defaults = config.get_model_defaults("non_existing")
        assert empty_defaults == {}
    
    def test_device_name_mapping(self):
        """Test device name mapping"""
        config = InferXConfig()
        
        # Test known mappings
        assert config.get_device_name("auto") == "AUTO"
        assert config.get_device_name("cpu") == "CPU"
        assert config.get_device_name("gpu") == "GPU"
        assert config.get_device_name("myriad") == "MYRIAD"
        
        # Test unknown mapping (should return uppercase)
        assert config.get_device_name("unknown") == "UNKNOWN"
    
    @patch("builtins.open", mock_open(read_data="model_detection:\n  yolo_keywords:\n    - custom_yolo"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_config_loading_from_file(self):
        """Test loading configuration from file"""
        with patch("inferx.config.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "model_detection": {
                    "yolo_keywords": ["custom_yolo"]
                }
            }
            
            config = InferXConfig("test_config.yaml")
            
            # Should have merged with defaults
            yolo_keywords = config.get("model_detection.yolo_keywords", [])
            assert "custom_yolo" in yolo_keywords


@pytest.mark.config
@pytest.mark.unit
class TestConfigUtilities:
    """Test configuration utility functions"""
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration"""
        valid_config = {
            "model_detection": {
                "yolo_keywords": ["yolo", "yolov8"]
            },
            "device_mapping": {
                "auto": "AUTO",
                "cpu": "CPU"
            },
            "model_defaults": {
                "yolo": {
                    "confidence_threshold": 0.5
                }
            }
        }
        
        warnings = validate_config(valid_config)
        assert len(warnings) == 0
    
    def test_validate_config_invalid(self):
        """Test validation of invalid configuration"""
        invalid_config = {
            "model_detection": {
                "yolo_keywords": "not_a_list"  # Should be list
            },
            "device_mapping": {
                "auto": 123  # Should be string
            },
            "model_defaults": {
                "yolo": {
                    "confidence_threshold": 1.5  # Should be between 0 and 1
                }
            }
        }
        
        warnings = validate_config(invalid_config)
        assert len(warnings) > 0
        
        # Check for specific warning types
        warning_text = " ".join(warnings)
        assert "should be a list" in warning_text
        assert "should be a string" in warning_text
        assert "between 0 and 1" in warning_text
    
    def test_create_user_config_template(self, temp_dir):
        """Test creating user config template"""
        template_path = temp_dir / "user_config.yaml"
        
        create_user_config_template(template_path)
        
        # Should create the file
        assert template_path.exists()
        
        # Should contain expected content
        content = template_path.read_text()
        assert "InferX User Configuration" in content
        assert "model_detection:" in content
        assert "device_mapping:" in content


@pytest.mark.config
@pytest.mark.integration
class TestConfigIntegration:
    """Test configuration system integration"""
    
    def test_get_global_config(self):
        """Test getting global config instance"""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        assert config1 is config2
    
    def test_config_hierarchy(self, temp_dir, monkeypatch):
        """Test configuration hierarchy loading"""
        # Create a project local config
        project_config = temp_dir / "inferx_config.yaml"
        project_config.write_text("""
model_defaults:
  yolo:
    confidence_threshold: 0.3
""")
        
        # Change working directory to temp_dir
        monkeypatch.chdir(temp_dir)
        
        # Create new config instance (should load project config)
        config = InferXConfig()
        
        # Should have project-specific value
        threshold = config.get("model_defaults.yolo.confidence_threshold")
        assert threshold == 0.3
    
    @patch("pathlib.Path.home")
    def test_user_global_config_path(self, mock_home, temp_dir):
        """Test user global config path generation"""
        mock_home.return_value = temp_dir
        
        config = InferXConfig()
        expected_path = temp_dir / ".inferx" / "config.yaml"
        assert config._get_user_global_config_path() == expected_path