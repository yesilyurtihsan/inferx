"""Base inferencer class for all model types"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

from ..exceptions import (
    ModelError, 
    InferenceError,
    create_model_error,
    create_inference_error,
    ErrorCode
)


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
            
        Raises:
            ModelError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        self.session = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        try:
            self._load_model()
        except Exception as e:
            # Convert generic exceptions to specific model errors
            if not isinstance(e, ModelError):
                raise create_model_error(str(self.model_path), e, self._get_runtime_name())
            raise
    
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
            raise ModelError(
                message="Model not loaded properly",
                error_code=ErrorCode.MODEL_NOT_LOADED,
                suggestions=[
                    "Ensure model file exists and is valid",
                    "Check model file permissions",
                    "Verify runtime compatibility",
                    "Try reloading the model"
                ],
                recovery_actions=[
                    "Recreate the inferencer instance",
                    "Use a different model file",
                    "Check system resources"
                ],
                context={"model_path": str(self.model_path)}
            )
        
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
            
        Raises:
            InferenceError: If batch processing fails
        """
        if not input_batch:
            raise InferenceError(
                message="Empty input batch provided",
                error_code=ErrorCode.BATCH_SIZE_MISMATCH,
                suggestions=["Provide at least one input item"],
                context={"batch_size": 0}
            )
        
        results = []
        failed_items = []
        
        for i, input_data in enumerate(input_batch):
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to process batch item {i}: {e}")
                failed_items.append({"index": i, "error": str(e)})
                
                # Add error recovery: continue with other items but log failures
                results.append({
                    "error": True,
                    "error_message": str(e),
                    "index": i
                })
        
        # Log summary of batch processing
        if failed_items:
            self.logger.warning(
                f"Batch processing completed with {len(failed_items)} failures out of {len(input_batch)} items"
            )
        
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
    
    def _get_runtime_name(self) -> str:
        """Get runtime name for error reporting
        
        Returns:
            Runtime name (e.g., 'onnx', 'openvino')
        """
        class_name = self.__class__.__name__.lower()
        if "onnx" in class_name:
            return "onnx"
        elif "openvino" in class_name:
            return "openvino"
        else:
            return "unknown"