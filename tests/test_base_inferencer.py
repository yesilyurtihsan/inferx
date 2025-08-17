"""Simple tests for base inferencer functionality"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from inferx.inferencers.base import BaseInferencer


class ConcreteInferencer(BaseInferencer):
    """Simple implementation of BaseInferencer for testing"""
    
    def __init__(self, model_path, config=None):
        # Override init to avoid calling _load_model
        self.model_path = Path(model_path)
        self.config = config or {}
        self.session = None
        # Don't call super().__init__() to avoid _load_model()
    
    def _load_model(self):
        """Mock model loading"""
        self.session = Mock()
        self.session.loaded = True
    
    def preprocess(self, input_data):
        """Mock preprocessing"""
        if isinstance(input_data, str):
            # Simulate loading and preprocessing an image
            return np.random.rand(1, 3, 224, 224).astype(np.float32)
        elif isinstance(input_data, np.ndarray):
            return input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _run_inference(self, preprocessed_data):
        """Mock inference"""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Simulate model inference
        batch_size = preprocessed_data.shape[0]
        return [np.random.rand(batch_size, 1000).astype(np.float32)]
    
    def postprocess(self, model_outputs):
        """Mock postprocessing"""
        output = model_outputs[0]
        return {
            "predictions": output.tolist(),
            "shape": output.shape,
            "max_prediction": float(np.max(output)),
            "model_type": "concrete_test"
        }


def test_inferencer_initialization():
    """Test basic inferencer initialization"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
    
    try:
        inferencer = ConcreteInferencer(model_path)
        
        assert inferencer.model_path == model_path
        assert inferencer.config == {}
        assert inferencer.session is None  # Not loaded yet
    finally:
        if model_path.exists():
            model_path.unlink()


def test_predict_pipeline():
    """Test complete prediction pipeline"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
    
    try:
        inferencer = ConcreteInferencer(model_path)
        inferencer._load_model()
        
        result = inferencer.predict("test_image.jpg")
        
        # Check result structure
        assert "predictions" in result
        assert "model_type" in result
        assert result["model_type"] == "concrete_test"
    finally:
        if model_path.exists():
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])