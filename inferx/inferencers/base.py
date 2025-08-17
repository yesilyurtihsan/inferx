"""Base inferencer class for all model types"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseInferencer(ABC):
    """Base class for all model-specific inferencers
    
    This abstract class defines the interface that all inferencers must implement.
    Subclasses should implement model loading, preprocessing, and postprocessing
    specific to their model type.
    """
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """Initialize the inferencer
        
        Args:
            model_path: Path to the model file (.onnx, .xml, etc.)
            config: Optional configuration dictionary
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        self.session = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model into memory
        
        This method should initialize self.session with the loaded model.
        Implementation depends on the runtime (ONNX, OpenVINO, etc.)
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input data for model inference
        
        Args:
            input_data: Raw input data (image path, image array, etc.)
            
        Returns:
            Preprocessed data ready for model inference
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Postprocess model outputs into human-readable format
        
        Args:
            model_outputs: Raw outputs from model inference
            
        Returns:
            Dictionary containing processed results
        """
        pass
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run complete inference pipeline
        
        Args:
            input_data: Raw input data
            
        Returns:
            Dictionary containing inference results
            
        Raises:
            RuntimeError: If model is not loaded properly
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Preprocess input
        preprocessed_data = self.preprocess(input_data)
        
        # Run inference
        model_outputs = self._run_inference(preprocessed_data)
        
        # Postprocess outputs
        results = self.postprocess(model_outputs)
        
        return results
    
    def predict_batch(self, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """Run inference on a batch of inputs
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of inference results
        """
        results = []
        for input_data in input_batch:
            result = self.predict(input_data)
            results.append(result)
        return results
    
    @abstractmethod
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run model inference on preprocessed data
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model
        
        Returns:
            Dictionary containing model metadata
        """
        if self.session is None:
            return {"status": "not_loaded"}
        
        return {
            "model_path": str(self.model_path),
            "model_size": f"{self.model_path.stat().st_size / 1024 / 1024:.1f} MB",
            "status": "loaded",
            "config": self.config
        }