"""Main runtime interface for InferX"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .inferencers.base import BaseInferencer
from .inferencers.onnx_inferencer import ONNXInferencer
from .inferencers.yolo import YOLOInferencer


class InferenceEngine:
    """Main inference engine for InferX
    
    This class provides a high-level interface for running inference
    with different model types and runtimes.
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        runtime: str = "auto"
    ):
        """Initialize the inference engine
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (yolo, anomalib, etc.). Auto-detected if None
            config: Configuration dictionary
            device: Device to run on (auto, cpu, gpu)
            runtime: Runtime to use (auto, onnx, openvino)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type or self._detect_model_type()
        self.config = config or {}
        self.device = device
        self.runtime = runtime
        
        # Update config with runtime settings
        self.config.update({
            "device": device,
            "runtime": runtime
        })
        
        self.inferencer = self._create_inferencer()
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type from file path or metadata"""
        file_extension = self.model_path.suffix.lower()
        
        if file_extension == '.onnx':
            # Try to detect if it's a YOLO model based on filename
            model_name = self.model_path.stem.lower()
            if any(keyword in model_name for keyword in ['yolo', 'yolov', 'yolov8']):
                return "yolo"
            return "onnx"
        elif file_extension == '.xml':
            # OpenVINO format
            return "openvino"
        else:
            # Default to ONNX for unknown types
            return "onnx"
    
    def _create_inferencer(self) -> BaseInferencer:
        """Create appropriate inferencer based on model type"""
        if self.model_type == "yolo":
            return YOLOInferencer(self.model_path, self.config)
        elif self.model_type == "onnx" or self.runtime == "onnx":
            return ONNXInferencer(self.model_path, self.config)
        else:
            # Fallback to ONNX for unsupported types
            return ONNXInferencer(self.model_path, self.config)
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run inference on single input
        
        Args:
            input_data: Input data (image path, array, etc.)
            
        Returns:
            Inference results
        """
        return self.inferencer.predict(input_data)
    
    def predict_batch(self, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """Run inference on batch of inputs
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of inference results
        """
        return self.inferencer.predict_batch(input_batch)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata
        
        Returns:
            Model information dictionary
        """
        info = self.inferencer.get_model_info()
        info.update({
            "model_type": self.model_type,
            "device": self.device,
            "runtime": self.runtime
        })
        return info