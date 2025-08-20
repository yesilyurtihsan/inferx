"""Enhanced tests for runtime engine with OpenVINO and config support"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from inferx.runtime import InferenceEngine


@pytest.mark.integration
class TestEnhancedInferenceEngine:
    """Test enhanced inference engine with OpenVINO and config support"""
    
    @patch('inferx.runtime.get_config')
    def test_initialization_with_config_path(self, mock_get_config, mock_onnx_model_path):
        """Test initialization with custom config path"""
        mock_config = Mock()
        mock_config.detect_model_type.return_value = "yolo_onnx"
        mock_config.get_model_defaults.return_value = {"input_size": 640}
        mock_config.get_preprocessing_defaults.return_value = {}
        mock_config.get_device_name.return_value = "CPU"
        mock_get_config.return_value = mock_config
        
        with patch('inferx.runtime.YOLOInferencer') as mock_inferencer:
            engine = InferenceEngine(
                model_path=mock_onnx_model_path,
                config_path="custom_config.yaml"
            )
            
            # Should call get_config with custom path
            mock_get_config.assert_called_with("custom_config.yaml")
            
            # Should use detected model type
            mock_config.detect_model_type.assert_called_once()
            assert engine.model_type == "yolo_onnx"
    
    def test_model_type_detection_yolo_onnx(self, mock_yolo_onnx_path):
        """Test YOLO ONNX model type detection"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.YOLOInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "yolo_onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_yolo_onnx_path)
            
            assert engine.model_type == "yolo_onnx"
    
    def test_model_type_detection_yolo_openvino(self, mock_yolo_openvino_path):
        """Test YOLO OpenVINO model type detection"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.YOLOOpenVINOInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "yolo_openvino"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_yolo_openvino_path)
            
            assert engine.model_type == "yolo_openvino"
    
    def test_model_type_detection_generic_openvino(self, mock_openvino_model_path):
        """Test generic OpenVINO model type detection"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.OpenVINOInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "openvino"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_openvino_model_path)
            
            assert engine.model_type == "openvino"
    
    def test_runtime_detection_auto(self, mock_yolo_onnx_path):
        """Test automatic runtime detection"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.YOLOInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "yolo_onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_yolo_onnx_path, runtime="auto")
            
            # Should detect ONNX runtime for .onnx files
            assert engine._detect_runtime() == "onnx"
    
    def test_runtime_detection_openvino_xml(self, mock_openvino_model_path):
        """Test OpenVINO runtime detection for .xml files"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.OpenVINOInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "openvino"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_openvino_model_path, runtime="auto")
            
            # Should detect OpenVINO runtime for .xml files
            assert engine._detect_runtime() == "openvino"
    
    def test_config_building_with_defaults(self, mock_onnx_model_path):
        """Test configuration building with model defaults"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.ONNXInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {
                "input_size": 224,
                "normalize": True
            }
            mock_config.get_preprocessing_defaults.return_value = {
                "color_format": "RGB"
            }
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_onnx_model_path)
            
            # Should merge model defaults with preprocessing defaults
            assert engine.config["input_size"] == 224
            assert engine.config["normalize"] is True
            assert engine.config["preprocessing"]["color_format"] == "RGB"
            assert engine.config["device"] == "CPU"
    
    def test_config_merging_with_user_config(self, mock_onnx_model_path):
        """Test configuration merging with user-provided config"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.ONNXInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {"input_size": 224}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            user_config = {
                "input_size": 640,  # Override default
                "custom_setting": "value"
            }
            
            engine = InferenceEngine(mock_onnx_model_path, config=user_config)
            
            # Should override default with user config
            assert engine.config["input_size"] == 640
            assert engine.config["custom_setting"] == "value"
    
    def test_inferencer_creation_yolo_onnx(self, mock_yolo_onnx_path):
        """Test creating YOLO ONNX inferencer"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.YOLOInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "yolo_onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_yolo_onnx_path)
            
            # Should create YOLOInferencer
            mock_inferencer_class.assert_called_once_with(
                mock_yolo_onnx_path, engine.config
            )
    
    def test_inferencer_creation_yolo_openvino(self, mock_yolo_openvino_path):
        """Test creating YOLO OpenVINO inferencer"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.YOLOOpenVINOInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "yolo_openvino"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_yolo_openvino_path)
            
            # Should create YOLOOpenVINOInferencer
            mock_inferencer_class.assert_called_once_with(
                mock_yolo_openvino_path, engine.config
            )
    
    def test_inferencer_creation_generic_openvino(self, mock_openvino_model_path):
        """Test creating generic OpenVINO inferencer"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.OpenVINOInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "openvino"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_openvino_model_path)
            
            # Should create OpenVINOInferencer
            mock_inferencer_class.assert_called_once_with(
                mock_openvino_model_path, engine.config
            )
    
    def test_cross_runtime_inference(self, mock_onnx_model_path):
        """Test running ONNX model on OpenVINO runtime"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.OpenVINOInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            # Force OpenVINO runtime for ONNX model
            engine = InferenceEngine(mock_onnx_model_path, runtime="openvino")
            
            # Should create OpenVINO inferencer even for .onnx file
            mock_inferencer_class.assert_called_once()
    
    def test_device_mapping(self, mock_onnx_model_path):
        """Test device name mapping"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.ONNXInferencer'):
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "GPU"  # Maps "gpu" to "GPU"
            mock_get_config.return_value = mock_config
            
            engine = InferenceEngine(mock_onnx_model_path, device="gpu")
            
            # Should map device name through config
            mock_config.get_device_name.assert_called_with("gpu")
            assert engine.config["device"] == "GPU"
    
    def test_predict_delegation(self, mock_onnx_model_path, mock_image_path):
        """Test prediction delegation to inferencer"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.ONNXInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            mock_inferencer = Mock()
            mock_inferencer.predict.return_value = {"result": "test"}
            mock_inferencer_class.return_value = mock_inferencer
            
            engine = InferenceEngine(mock_onnx_model_path)
            result = engine.predict(mock_image_path)
            
            # Should delegate to inferencer
            mock_inferencer.predict.assert_called_once_with(mock_image_path)
            assert result == {"result": "test"}
    
    def test_predict_batch_delegation(self, mock_onnx_model_path, batch_image_paths):
        """Test batch prediction delegation to inferencer"""
        with patch('inferx.runtime.get_config') as mock_get_config, \
             patch('inferx.runtime.ONNXInferencer') as mock_inferencer_class:
            
            mock_config = Mock()
            mock_config.detect_model_type.return_value = "onnx"
            mock_config.get_model_defaults.return_value = {}
            mock_config.get_preprocessing_defaults.return_value = {}
            mock_config.get_device_name.return_value = "CPU"
            mock_get_config.return_value = mock_config
            
            mock_inferencer = Mock()
            mock_inferencer.predict_batch.return_value = [{"result": "test"}] * 3
            mock_inferencer_class.return_value = mock_inferencer
            
            engine = InferenceEngine(mock_onnx_model_path)
            result = engine.predict_batch(batch_image_paths)
            
            # Should delegate to inferencer
            mock_inferencer.predict_batch.assert_called_once_with(batch_image_paths)
            assert len(result) == 3