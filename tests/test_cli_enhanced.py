"""Enhanced tests for CLI with OpenVINO and config support"""

import pytest
import json
from unittest.mock import Mock, patch
from click.testing import CliRunner

from inferx.cli import cli


@pytest.mark.cli
@pytest.mark.integration
class TestEnhancedCLI:
    """Test enhanced CLI functionality"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    @patch('inferx.cli.InferenceEngine')
    def test_run_command_openvino_model(self, mock_engine_class, mock_yolo_openvino_path, mock_image_path):
        """Test run command with OpenVINO model"""
        mock_engine = Mock()
        mock_engine.predict.return_value = {
            "detections": [],
            "num_detections": 0,
            "model_type": "yolo_openvino"
        }
        mock_engine_class.return_value = mock_engine
        
        result = self.runner.invoke(cli, [
            'run',
            str(mock_yolo_openvino_path),
            str(mock_image_path),
            '--device', 'cpu',
            '--runtime', 'openvino'
        ])
        
        assert result.exit_code == 0
        assert "yolo_openvino" in result.output or "Inference Summary" in result.output
        
        # Should create engine with correct parameters
        mock_engine_class.assert_called_once()
        call_args = mock_engine_class.call_args
        assert call_args[1]['device'] == 'cpu'
        assert call_args[1]['runtime'] == 'openvino'
    
    @patch('inferx.cli.InferenceEngine')
    def test_run_command_with_openvino_devices(self, mock_engine_class, mock_openvino_model_path, mock_image_path):
        """Test run command with OpenVINO-specific devices"""
        mock_engine = Mock()
        mock_engine.predict.return_value = {
            "raw_outputs": [],
            "model_type": "openvino_generic"
        }
        mock_engine_class.return_value = mock_engine
        
        # Test with MYRIAD device
        result = self.runner.invoke(cli, [
            'run',
            str(mock_openvino_model_path),
            str(mock_image_path),
            '--device', 'myriad'
        ])
        
        assert result.exit_code == 0
        
        # Verify device parameter was passed
        call_args = mock_engine_class.call_args
        assert call_args[1]['device'] == 'myriad'
    
    @patch('inferx.cli.InferenceEngine')
    def test_run_command_cross_runtime(self, mock_engine_class, mock_onnx_model_path, mock_image_path):
        """Test running ONNX model with OpenVINO runtime"""
        mock_engine = Mock()
        mock_engine.predict.return_value = {
            "raw_outputs": [],
            "model_type": "onnx_generic"
        }
        mock_engine_class.return_value = mock_engine
        
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path),
            '--runtime', 'openvino'
        ])
        
        assert result.exit_code == 0
        
        # Should pass OpenVINO runtime for ONNX model
        call_args = mock_engine_class.call_args
        assert call_args[1]['runtime'] == 'openvino'
    
    def test_config_init_command(self):
        """Test config init command"""
        with patch('inferx.cli.init_user_config') as mock_init:
            result = self.runner.invoke(cli, ['config', '--init'])
            
            assert result.exit_code == 0
            mock_init.assert_called_once()
    
    def test_config_template_command(self, tmp_path):
        """Test config template creation command"""
        template_path = tmp_path / "test_template.yaml"
        
        with patch('inferx.cli.create_user_config_template') as mock_create:
            result = self.runner.invoke(cli, [
                'config', 
                '--template', 
                str(template_path)
            ])
            
            assert result.exit_code == 0
            mock_create.assert_called_once_with(str(template_path))
            assert "Config template created" in result.output
    
    @patch('inferx.cli.get_config')
    @patch('inferx.cli.validate_config')
    def test_config_validate_command(self, mock_validate, mock_get_config):
        """Test config validation command"""
        mock_config = Mock()
        mock_config.to_dict.return_value = {"test": "config"}
        mock_get_config.return_value = mock_config
        
        # Test with no warnings
        mock_validate.return_value = []
        
        result = self.runner.invoke(cli, ['config', '--validate'])
        
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output
        mock_validate.assert_called_once()
    
    @patch('inferx.cli.get_config')
    @patch('inferx.cli.validate_config')
    def test_config_validate_with_warnings(self, mock_validate, mock_get_config):
        """Test config validation with warnings"""
        mock_config = Mock()
        mock_config.to_dict.return_value = {"test": "config"}
        mock_get_config.return_value = mock_config
        
        # Test with warnings
        mock_validate.return_value = ["Warning 1", "Warning 2"]
        
        result = self.runner.invoke(cli, ['config', '--validate'])
        
        assert result.exit_code == 0
        assert "Configuration warnings" in result.output
        assert "Warning 1" in result.output
        assert "Warning 2" in result.output
    
    @patch('inferx.cli.get_config')
    def test_config_show_command(self, mock_get_config):
        """Test config show command"""
        mock_config = Mock()
        test_config = {"test": "config", "nested": {"value": 123}}
        mock_config.to_dict.return_value = test_config
        mock_get_config.return_value = mock_config
        
        result = self.runner.invoke(cli, ['config', '--show'])
        
        assert result.exit_code == 0
        assert "Current configuration" in result.output
        
        # Should contain JSON representation of config
        assert '"test": "config"' in result.output or "test" in result.output
    
    @patch('inferx.cli.init_user_config')
    def test_init_global_command(self, mock_init):
        """Test init command with global flag"""
        result = self.runner.invoke(cli, ['init', '--global'])
        
        assert result.exit_code == 0
        mock_init.assert_called_once()
    
    def test_init_template_command(self):
        """Test init command with template (not yet implemented)"""
        result = self.runner.invoke(cli, ['init', '--template', 'yolo'])
        
        assert result.exit_code == 0
        assert "Project initialization" in result.output or "coming soon" in result.output
    
    @patch('inferx.cli.InferenceEngine')
    def test_run_with_config_file(self, mock_engine_class, mock_onnx_model_path, mock_image_path, mock_yaml_config):
        """Test run command with config file"""
        mock_engine = Mock()
        mock_engine.predict.return_value = {"model_type": "onnx_generic"}
        mock_engine_class.return_value = mock_engine
        
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path),
            '--config', str(mock_yaml_config)
        ])
        
        assert result.exit_code == 0
        
        # Should pass config_path to InferenceEngine
        call_args = mock_engine_class.call_args
        assert 'config_path' in call_args[1] or call_args[0][2] is not None  # config parameter
    
    @patch('inferx.cli.InferenceEngine')
    def test_run_batch_processing(self, mock_engine_class, mock_openvino_model_path, batch_image_paths):
        """Test batch processing with OpenVINO model"""
        mock_engine = Mock()
        mock_engine.predict.return_value = {
            "model_type": "openvino_generic",
            "inference_time": 0.05
        }
        mock_engine_class.return_value = mock_engine
        
        # Create temporary directory with images
        batch_dir = batch_image_paths[0].parent
        
        result = self.runner.invoke(cli, [
            'run',
            str(mock_openvino_model_path),
            str(batch_dir),
            '--runtime', 'openvino'
        ])
        
        assert result.exit_code == 0
        assert "Batch processing completed" in result.output
        
        # Should call predict multiple times for batch
        assert mock_engine.predict.call_count >= 3
    
    @patch('inferx.cli.InferenceEngine')
    def test_error_handling_missing_model(self, mock_engine_class):
        """Test error handling with missing model file"""
        result = self.runner.invoke(cli, [
            'run',
            'non_existent_model.xml',
            'image.jpg'
        ])
        
        # Should fail with file not found error
        assert result.exit_code != 0
    
    @patch('inferx.cli.InferenceEngine')
    def test_error_handling_inference_failure(self, mock_engine_class, mock_onnx_model_path, mock_image_path):
        """Test error handling when inference fails"""
        mock_engine = Mock()
        mock_engine.predict.side_effect = Exception("Inference failed")
        mock_engine_class.return_value = mock_engine
        
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path)
        ])
        
        assert result.exit_code == 0  # CLI handles error gracefully
        assert "Inference failed" in result.output
    
    def test_verbose_output(self):
        """Test verbose output with global flag"""
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        
        assert result.exit_code == 0
        # Verbose flag should not cause issues with help command
    
    def test_version_command(self):
        """Test version command"""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        # Should show version information


@pytest.mark.cli 
@pytest.mark.unit
class TestCLIArgumentValidation:
    """Test CLI argument validation"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_run_device_choices(self, mock_onnx_model_path, mock_image_path):
        """Test device choice validation"""
        # Valid device
        with patch('inferx.cli.InferenceEngine'):
            result = self.runner.invoke(cli, [
                'run',
                str(mock_onnx_model_path),
                str(mock_image_path),
                '--device', 'myriad'
            ])
            assert result.exit_code == 0
        
        # Invalid device should fail
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path),
            '--device', 'invalid_device'
        ])
        assert result.exit_code != 0
    
    def test_run_runtime_choices(self, mock_onnx_model_path, mock_image_path):
        """Test runtime choice validation"""
        # Valid runtime
        with patch('inferx.cli.InferenceEngine'):
            result = self.runner.invoke(cli, [
                'run',
                str(mock_onnx_model_path),
                str(mock_image_path),
                '--runtime', 'openvino'
            ])
            assert result.exit_code == 0
        
        # Invalid runtime should fail
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path),
            '--runtime', 'invalid_runtime'
        ])
        assert result.exit_code != 0
    
    def test_output_format_choices(self, mock_onnx_model_path, mock_image_path):
        """Test output format choice validation"""
        # Valid format
        with patch('inferx.cli.InferenceEngine'):
            result = self.runner.invoke(cli, [
                'run',
                str(mock_onnx_model_path),
                str(mock_image_path),
                '--format', 'yaml'
            ])
            assert result.exit_code == 0
        
        # Invalid format should fail
        result = self.runner.invoke(cli, [
            'run',
            str(mock_onnx_model_path),
            str(mock_image_path),
            '--format', 'invalid_format'
        ])
        assert result.exit_code != 0