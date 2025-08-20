"""Tests for OpenVINO inferencer classes"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from inferx.inferencers.openvino_inferencer import OpenVINOInferencer
from inferx.inferencers.yolo_openvino import YOLOOpenVINOInferencer


@pytest.mark.openvino
@pytest.mark.unit
class TestOpenVINOInferencer:
    """Test OpenVINO inferencer basic functionality"""
    
    @patch('openvino.Core')
    def test_initialization(self, mock_ov_core, mock_openvino_model_path, mock_openvino_config):
        """Test basic OpenVINO inferencer initialization"""
        # Mock OpenVINO core and model
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_core.compile_model.return_value = mock_compiled_model
        
        # Mock input/output info
        mock_input = Mock()
        mock_input.shape = [1, 3, 224, 224]
        mock_input.any_name = "input"
        mock_input.index = 0
        
        mock_output = Mock()
        mock_output.shape = [1, 1000]
        mock_output.any_name = "output"
        mock_output.index = 0
        
        mock_compiled_model.inputs = [mock_input]
        mock_compiled_model.outputs = [mock_output]
        
        # Create inferencer
        inferencer = OpenVINOInferencer(mock_openvino_model_path, mock_openvino_config)
        
        # Verify initialization
        assert inferencer.model_path == mock_openvino_model_path
        assert inferencer.core == mock_core
        assert inferencer.compiled_model == mock_compiled_model
        assert inferencer.input_shape == [1, 3, 224, 224]
        assert inferencer.output_shape == [1, 1000]
    
    @patch('openvino.Core')
    def test_device_config_setup(self, mock_ov_core, mock_openvino_model_path):
        """Test device configuration setup"""
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        mock_core.read_model.return_value = Mock()
        
        mock_compiled_model = Mock()
        mock_compiled_model.inputs = [Mock(shape=[1, 3, 224, 224], any_name="input", index=0)]
        mock_compiled_model.outputs = [Mock(shape=[1, 1000], any_name="output", index=0)]
        mock_core.compile_model.return_value = mock_compiled_model
        
        config = {
            "device": "CPU",
            "performance_hint": "THROUGHPUT",
            "num_threads": 4,
            "num_streams": 2
        }
        
        inferencer = OpenVINOInferencer(mock_openvino_model_path, config)
        
        # Verify compile_model was called with correct config
        mock_core.compile_model.assert_called_once()
        call_args = mock_core.compile_model.call_args
        
        assert call_args[1]["device_name"] == "CPU"
        assert "PERFORMANCE_HINT" in call_args[1]["config"]
    
    def test_preprocessing_image_path(self, mock_openvino_model_path, mock_image_path):
        """Test preprocessing with image path input"""
        with patch('openvino.Core'), \
             patch.object(OpenVINOInferencer, '_load_model'):
            
            inferencer = OpenVINOInferencer(mock_openvino_model_path)
            
            with patch('inferx.utils.preprocess_for_inference') as mock_preprocess:
                mock_preprocess.return_value = np.random.rand(1, 3, 224, 224)
                
                result = inferencer.preprocess(mock_image_path)
                
                # Should call preprocess_for_inference
                mock_preprocess.assert_called_once()
                assert isinstance(result, np.ndarray)
    
    def test_preprocessing_numpy_array(self, mock_openvino_model_path):
        """Test preprocessing with numpy array input"""
        with patch('openvino.Core'), \
             patch.object(OpenVINOInferencer, '_load_model'):
            
            inferencer = OpenVINOInferencer(mock_openvino_model_path)
            
            # Test with 3D array (should add batch dimension)
            input_array = np.random.rand(3, 224, 224).astype(np.float64)
            result = inferencer.preprocess(input_array)
            
            assert result.shape == (1, 3, 224, 224)
            assert result.dtype == np.float32
    
    @patch('openvino.Core')
    def test_inference_execution(self, mock_ov_core, mock_openvino_model_path):
        """Test inference execution"""
        # Setup mocks
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_core.compile_model.return_value = mock_compiled_model
        
        # Mock input/output
        mock_input = Mock(shape=[1, 3, 224, 224], any_name="input", index=0)
        mock_output = Mock(shape=[1, 1000], any_name="output", index=0)
        mock_compiled_model.inputs = [mock_input]
        mock_compiled_model.outputs = [mock_output]
        
        # Mock inference request
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        
        # Mock output tensor
        mock_output_tensor = Mock()
        mock_output_tensor.data = np.random.rand(1, 1000)
        mock_infer_request.get_output_tensor.return_value = mock_output_tensor
        
        inferencer = OpenVINOInferencer(mock_openvino_model_path)
        
        # Run inference
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        result = inferencer._run_inference(input_data)
        
        # Verify inference was called
        mock_infer_request.set_input_tensor.assert_called_once()
        mock_infer_request.infer.assert_called_once()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
    
    def test_postprocessing(self, mock_openvino_model_path):
        """Test basic postprocessing"""
        with patch('openvino.Core'), \
             patch.object(OpenVINOInferencer, '_load_model'):
            
            inferencer = OpenVINOInferencer(mock_openvino_model_path)
            inferencer.output_layers = {"output": Mock()}
            
            # Mock model outputs
            mock_outputs = [np.random.rand(1, 1000)]
            
            result = inferencer.postprocess(mock_outputs)
            
            # Should return structured result
            assert "raw_outputs" in result
            assert "output_shapes" in result
            assert "num_outputs" in result
            assert "model_type" in result
            assert result["model_type"] == "openvino_generic"
    
    @patch('openvino.Core')
    def test_get_model_info(self, mock_ov_core, mock_openvino_model_path):
        """Test getting model information"""
        # Setup basic mocks
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        mock_core.read_model.return_value = Mock()
        
        mock_compiled_model = Mock()
        mock_compiled_model.inputs = [Mock(shape=[1, 3, 224, 224], any_name="input", index=0)]
        mock_compiled_model.outputs = [Mock(shape=[1, 1000], any_name="output", index=0)]
        mock_core.compile_model.return_value = mock_compiled_model
        
        inferencer = OpenVINOInferencer(mock_openvino_model_path)
        
        info = inferencer.get_model_info()
        
        # Should contain OpenVINO-specific info
        assert "runtime" in info
        assert info["runtime"] == "openvino"
        assert "device" in info
        assert "input_shape" in info
        assert "output_shape" in info


@pytest.mark.openvino
@pytest.mark.unit
class TestYOLOOpenVINOInferencer:
    """Test YOLO-specific OpenVINO inferencer"""
    
    @patch('openvino.Core')
    def test_yolo_initialization(self, mock_ov_core, mock_yolo_openvino_path):
        """Test YOLO OpenVINO inferencer initialization"""
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        mock_core.read_model.return_value = Mock()
        
        mock_compiled_model = Mock()
        mock_compiled_model.inputs = [Mock(shape=[1, 3, 640, 640], any_name="input", index=0)]
        mock_compiled_model.outputs = [Mock(shape=[1, 25200, 85], any_name="output", index=0)]
        mock_core.compile_model.return_value = mock_compiled_model
        
        inferencer = YOLOOpenVINOInferencer(mock_yolo_openvino_path)
        
        # Should have YOLO-specific config
        assert inferencer.config["input_size"] == 640
        assert "confidence_threshold" in inferencer.config
        assert "nms_threshold" in inferencer.config
        assert "class_names" in inferencer.config
        assert len(inferencer.config["class_names"]) == 80  # COCO classes
    
    def test_yolo_letterbox_preprocessing(self, mock_yolo_openvino_path, test_image_array):
        """Test YOLO letterbox preprocessing"""
        with patch('openvino.Core'), \
             patch.object(YOLOOpenVINOInferencer, '_load_model'):
            
            inferencer = YOLOOpenVINOInferencer(mock_yolo_openvino_path)
            inferencer.input_shape = [1, 3, 640, 640]
            
            result, ratio, pad = inferencer.preprocess(test_image_array)
            
            # Should return correct shapes and types
            assert result.shape == (1, 3, 640, 640)
            assert result.dtype == np.float32
            assert isinstance(ratio, tuple)
            assert isinstance(pad, tuple)
            assert len(ratio) == 2
            assert len(pad) == 2
    
    @patch('cv2.dnn.NMSBoxes')
    def test_yolo_postprocessing(self, mock_nms, mock_yolo_openvino_path):
        """Test YOLO postprocessing with NMS"""
        with patch('openvino.Core'), \
             patch.object(YOLOOpenVINOInferencer, '_load_model'):
            
            inferencer = YOLOOpenVINOInferencer(mock_yolo_openvino_path)
            inferencer.img_height = 100
            inferencer.img_width = 150
            
            # Mock NMS to return indices
            mock_nms.return_value = [0, 1]
            
            # Create mock model output (YOLO format)
            mock_output = np.random.rand(1, 85, 25200)  # [batch, features, detections]
            mock_output[0, 4, :] = 0.8  # Set confidence scores
            mock_output[0, 5:, :] = 0.1  # Set class scores
            mock_output[0, 85-1, 0] = 0.9  # Set one high class score for first detection
            
            result = inferencer.postprocess([mock_output], (1.0, 1.0), (0, 0))
            
            # Should return YOLO detection format
            assert "detections" in result
            assert "num_detections" in result
            assert "model_type" in result
            assert result["model_type"] == "yolo_openvino"
            
            # Should have processed detections
            assert isinstance(result["detections"], list)
            assert result["num_detections"] == len(result["detections"])
    
    def test_yolo_device_optimization_config(self, mock_yolo_openvino_path):
        """Test YOLO-specific device optimization configuration"""
        with patch('openvino.Core'), \
             patch.object(YOLOOpenVINOInferencer, '_load_model'):
            
            config = {"device": "CPU", "num_threads": 8}
            inferencer = YOLOOpenVINOInferencer(mock_yolo_openvino_path, config)
            
            device_config = inferencer._setup_device_config()
            
            # Should have YOLO-specific optimizations
            assert "CPU_BIND_THREAD" in device_config
            assert device_config["CPU_BIND_THREAD"] == "YES"
            assert "CPU_THROUGHPUT_STREAMS" in device_config
    
    def test_yolo_get_model_info(self, mock_yolo_openvino_path):
        """Test getting YOLO model information"""
        with patch('openvino.Core'), \
             patch.object(YOLOOpenVINOInferencer, '_load_model'):
            
            inferencer = YOLOOpenVINOInferencer(mock_yolo_openvino_path)
            inferencer.compiled_model = Mock()
            
            info = inferencer.get_model_info()
            
            # Should contain YOLO-specific info
            assert info["model_type"] == "yolo_openvino"
            assert "input_size" in info
            assert "confidence_threshold" in info
            assert "nms_threshold" in info
            assert "num_classes" in info


@pytest.mark.openvino
@pytest.mark.integration
class TestOpenVINOIntegration:
    """Integration tests for OpenVINO components"""
    
    def test_openvino_availability(self):
        """Test if OpenVINO is available for testing"""
        try:
            import openvino
            assert True  # OpenVINO is available
        except ImportError:
            pytest.skip("OpenVINO not available for testing")
    
    @patch('openvino.Core')
    def test_error_handling_missing_bin_file(self, mock_ov_core, temp_dir):
        """Test error handling when .bin file is missing"""
        # Create only .xml file (missing .bin)
        xml_path = temp_dir / "model.xml"
        xml_path.touch()
        
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        mock_core.read_model.side_effect = Exception("Cannot read model")
        
        with pytest.raises(RuntimeError, match="Failed to load OpenVINO model"):
            OpenVINOInferencer(xml_path)
    
    @patch('openvino.Core')
    def test_error_handling_invalid_model(self, mock_ov_core, mock_openvino_model_path):
        """Test error handling with invalid model file"""
        mock_core = Mock()
        mock_ov_core.return_value = mock_core
        mock_core.read_model.side_effect = Exception("Invalid model file")
        
        with pytest.raises(RuntimeError, match="Failed to load OpenVINO model"):
            OpenVINOInferencer(mock_openvino_model_path)