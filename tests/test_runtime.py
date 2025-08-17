"""Simple tests for InferX runtime and engine"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from inferx.runtime import InferenceEngine
from inferx.inferencers.base import BaseInferencer


class MockInferencer(BaseInferencer):
    """Simple mock inferencer for testing"""
    
    def __init__(self, model_path, config=None):
        # Don't call super().__init__ to avoid actual model loading
        self.model_path = Path(model_path)
        self.config = config or {}
        self.session = Mock()
    
    def _load_model(self):
        """Mock model loading"""
        pass
    
    def preprocess(self, input_data):
        """Mock preprocessing"""
        return np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    def _run_inference(self, preprocessed_data):
        """Mock inference"""
        return [np.random.rand(1, 1000).astype(np.float32)]
    
    def postprocess(self, model_outputs):
        """Mock postprocessing"""
        return {
            "raw_outputs": [output.tolist() for output in model_outputs],
            "output_shapes": [output.shape for output in model_outputs],
            "num_outputs": len(model_outputs),
            "model_type": "mock"
        }


def test_model_type_detection_onnx():
    """Test model type detection for ONNX files"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = Path(f.name)
    
    try:
        with patch('inferx.runtime.ONNXInferencer') as mock_inferencer:
            mock_inferencer.return_value = MockInferencer(onnx_path)
            
            engine = InferenceEngine(onnx_path)
            assert engine.model_type == "onnx"
    finally:
        if onnx_path.exists():
            onnx_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])