"""ONNX Runtime inferencer implementation"""

import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .base import BaseInferencer
from ..utils import ImageProcessor, preprocess_for_inference


logger = logging.getLogger(__name__)


class ONNXInferencer(BaseInferencer):
    """Generic ONNX Runtime inferencer for any ONNX model
    
    This class provides basic ONNX model loading and inference capabilities.
    It can be used directly for simple models or extended for model-specific logic.
    """
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """Initialize ONNX inferencer
        
        Args:
            model_path: Path to ONNX model file
            config: Optional configuration dictionary
        """
        self.providers = None
        self.input_names = None
        self.output_names = None
        self.input_shapes = None
        self.output_shapes = None
        
        # Default configuration
        self.default_config = {
            "device": "auto",  # auto, cpu, gpu
            "providers": None,  # Custom providers list
            "provider_options": {},  # Provider-specific options
            "session_options": {
                "graph_optimization_level": "ORT_ENABLE_ALL",
                "inter_op_num_threads": 0,  # 0 = use default
                "intra_op_num_threads": 0   # 0 = use default
            },
            "preprocessing": {
                "target_size": [224, 224],  # Default input size
                "normalize": True,
                "color_format": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        
        # Merge with user config
        if config:
            self._merge_config(self.default_config, config)
        
        super().__init__(model_path, self.default_config)
    
    def _merge_config(self, default: Dict, user: Dict) -> None:
        """Recursively merge user config with default config"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _load_model(self) -> None:
        """Load ONNX model using ONNX Runtime"""
        try:
            # Setup execution providers
            self.providers = self._setup_providers()
            
            # Setup session options
            sess_options = self._setup_session_options()
            
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.providers
            )
            
            # Get model metadata
            self._extract_model_metadata()
            
            logger.info(f"Loaded ONNX model: {self.model_path}")
            logger.info(f"Providers: {self.session.get_providers()}")
            logger.info(f"Input shapes: {self.input_shapes}")
            logger.info(f"Output shapes: {self.output_shapes}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.model_path}: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _setup_providers(self) -> List[str]:
        """Setup execution providers based on device configuration"""
        if self.config.get("providers"):
            return self.config["providers"]
        
        device = self.config.get("device", "auto").lower()
        
        if device == "gpu":
            # Try GPU providers first
            available_providers = ort.get_available_providers()
            gpu_providers = []
            
            if "CUDAExecutionProvider" in available_providers:
                gpu_providers.append("CUDAExecutionProvider")
            if "ROCMExecutionProvider" in available_providers:
                gpu_providers.append("ROCMExecutionProvider")
            if "OpenVINOExecutionProvider" in available_providers:
                gpu_providers.append("OpenVINOExecutionProvider")
            
            if gpu_providers:
                gpu_providers.append("CPUExecutionProvider")  # Fallback
                return gpu_providers
            else:
                logger.warning("GPU requested but no GPU providers available, falling back to CPU")
                return ["CPUExecutionProvider"]
        
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        
        else:  # auto
            # Use best available provider
            available_providers = ort.get_available_providers()
            
            # Priority order: CUDA > OpenVINO > CPU
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "OpenVINOExecutionProvider" in available_providers:
                return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            else:
                return ["CPUExecutionProvider"]
    
    def _setup_session_options(self) -> ort.SessionOptions:
        """Setup ONNX Runtime session options"""
        sess_options = ort.SessionOptions()
        
        config = self.config.get("session_options", {})
        
        # Graph optimization level
        opt_level = config.get("graph_optimization_level", "ORT_ENABLE_ALL")
        if opt_level == "ORT_DISABLE_ALL":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif opt_level == "ORT_ENABLE_BASIC":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif opt_level == "ORT_ENABLE_EXTENDED":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:  # ORT_ENABLE_ALL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Thread settings
        if config.get("inter_op_num_threads"):
            sess_options.inter_op_num_threads = config["inter_op_num_threads"]
        if config.get("intra_op_num_threads"):
            sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        
        # Enable profiling if requested
        if config.get("enable_profiling", False):
            sess_options.enable_profiling = True
        
        return sess_options
    
    def _extract_model_metadata(self) -> None:
        """Extract input/output metadata from loaded model"""
        # Input metadata
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.input_shapes = {input.name: input.shape for input in self.session.get_inputs()}
        
        # Output metadata
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = {output.name: output.shape for output in self.session.get_outputs()}
        
        logger.debug(f"Model inputs: {self.input_names}")
        logger.debug(f"Model outputs: {self.output_names}")
    
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input data for ONNX model
        
        Args:
            input_data: Input data (image path, image array, etc.)
            
        Returns:
            Preprocessed data ready for model inference
        """
        if isinstance(input_data, (str, Path)):
            # Input is image path
            preprocessing_config = self.config.get("preprocessing", {})
            
            return preprocess_for_inference(
                image_path=input_data,
                target_size=preprocessing_config.get("target_size", [224, 224]),
                normalize=preprocessing_config.get("normalize", True),
                color_format=preprocessing_config.get("color_format", "RGB"),
                mean=tuple(preprocessing_config.get("mean", [0.485, 0.456, 0.406])),
                std=tuple(preprocessing_config.get("std", [0.229, 0.224, 0.225]))
            )
        
        elif isinstance(input_data, np.ndarray):
            # Input is already a numpy array
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Ensure batch dimension exists
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            
            return input_data
        
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run ONNX model inference
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            # Prepare input dictionary
            input_dict = {}
            
            if len(self.input_names) == 1:
                # Single input model
                input_dict[self.input_names[0]] = preprocessed_data
            else:
                # Multiple input model - this is a basic implementation
                # More sophisticated logic needed for complex models
                input_dict[self.input_names[0]] = preprocessed_data
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            
            logger.debug(f"Inference completed, output shapes: {[out.shape for out in outputs]}")
            return outputs
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"ONNX inference failed: {e}")
    
    def postprocess(self, model_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Basic postprocessing for ONNX model outputs
        
        Args:
            model_outputs: Raw outputs from model inference
            
        Returns:
            Dictionary containing processed results
        """
        # Basic postprocessing - just return raw outputs with metadata
        results = {
            "raw_outputs": [output.tolist() for output in model_outputs],
            "output_shapes": [output.shape for output in model_outputs],
            "num_outputs": len(model_outputs),
            "model_type": "onnx_generic"
        }
        
        # Add some basic analysis
        for i, output in enumerate(model_outputs):
            output_name = self.output_names[i] if i < len(self.output_names) else f"output_{i}"
            results[f"{output_name}_shape"] = output.shape
            results[f"{output_name}_dtype"] = str(output.dtype)
            
            # Basic statistics
            if output.size > 0:
                results[f"{output_name}_min"] = float(np.min(output))
                results[f"{output_name}_max"] = float(np.max(output))
                results[f"{output_name}_mean"] = float(np.mean(output))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the ONNX model
        
        Returns:
            Dictionary containing model metadata
        """
        base_info = super().get_model_info()
        
        if self.session is None:
            return base_info
        
        # Add ONNX-specific information
        onnx_info = {
            "runtime": "onnx",
            "providers": self.session.get_providers(),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
        }
        
        # Model metadata from ONNX
        try:
            model_meta = self.session.get_modelmeta()
            onnx_info.update({
                "model_version": model_meta.version,
                "model_producer": model_meta.producer_name,
                "model_domain": model_meta.domain,
                "model_description": model_meta.description,
            })
        except Exception as e:
            logger.debug(f"Could not extract model metadata: {e}")
        
        base_info.update(onnx_info)
        return base_info