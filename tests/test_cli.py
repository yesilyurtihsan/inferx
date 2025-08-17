"""Simple tests for InferX CLI interface"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from inferx.cli import cli, _load_config, _save_results


def test_load_config_valid_yaml():
    """Test loading valid YAML config"""
    config_data = """
device: gpu
preprocessing:
  target_size: [640, 640]
  normalize: true
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_data)
        config_path = Path(f.name)
    
    try:
        config = _load_config(config_path)
        assert config['device'] == 'gpu'
        assert config['preprocessing']['target_size'] == [640, 640]
        assert config['preprocessing']['normalize'] is True
    finally:
        if config_path.exists():
            config_path.unlink()


def test_cli_version():
    """Test CLI version command"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert "InferX" in result.output


@patch('inferx.cli.InferenceEngine')
@patch('inferx.cli.FileUtils')
def test_run_command_single_image(mock_file_utils, mock_engine_class):
    """Test run command with single image"""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image_path = Path(f.name)
    
    try:
        # Mock FileUtils
        mock_file_utils.is_image_file.return_value = True
        
        # Mock InferenceEngine
        mock_engine = Mock()
        mock_engine.predict.return_value = {
            "predictions": [0.1, 0.9],
            "model_type": "onnx_generic",
            "num_outputs": 1
        }
        mock_engine_class.return_value = mock_engine
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run',
            str(model_path),
            str(image_path),
            '--device', 'cpu'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting inference..." in result.output
        assert "✅ Model loaded" in result.output
        assert "✅ Inference completed" in result.output
    finally:
        if model_path.exists():
            model_path.unlink()
        if image_path.exists():
            image_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])