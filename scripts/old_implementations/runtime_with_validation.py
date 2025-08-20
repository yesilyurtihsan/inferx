"""
Enhanced Runtime with Pydantic Configuration Validation

This demonstrates how to integrate Pydantic validation into runtime.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .inferencers.base import BaseInferencer
from .inferencers.onnx_inferencer import ONNXInferencer
from .inferencers.openvino_inferencer import OpenVINOInferencer
from .inferencers.yolo import YOLOInferencer
from .inferencers.yolo_openvino import YOLOOpenVINOInferencer

# Import enhanced config with validation
from .config_validator import get_enhanced_config
from .exceptions import ConfigurationError, ModelError, ErrorCode
from .recovery import with_model_loading_retry, with_inference_retry

logger = logging.getLogger(__name__)


class EnhancedInferenceEngine:
    """
    Enhanced Inference Engine with Pydantic Configuration Validation
    
    Drop-in replacement for InferenceEngine with type-safe configuration
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        runtime: str = "auto",
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize enhanced inference engine with validation
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (yolo, anomalib, etc.). Auto-detected if None
            config: Configuration dictionary
            device: Device to run on (auto, cpu, gpu)
            runtime: Runtime to use (auto, onnx, openvino)
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If configuration validation fails
            ModelError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.device = device
        self.runtime = runtime
        
        try:
            # Load and validate global configuration
            self.global_config = get_enhanced_config(config_path)
            logger.debug("Global configuration loaded and validated")
            
            # Show validation status
            if self.global_config.typed:
                logger.info("✅ Configuration validation enabled with Pydantic")
            else:
                logger.warning("⚠️  Configuration validation disabled (Pydantic not available)")
            
        except ConfigurationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(
                message="Failed to load global configuration",
                suggestions=[
                    "Check configuration file syntax",
                    "Verify configuration file exists",
                    "Install pydantic for enhanced validation"
                ],
                original_error=e
            )
        
        # Detect model type using validated configuration
        try:
            self.model_type = model_type or self.global_config.detect_model_type(
                self.model_path, runtime
            )
            logger.debug(f"Model type detected: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Model type detection failed: {e}")
            raise ModelError(
                message=f"Failed to detect model type for {self.model_path}",
                error_code=ErrorCode.MODEL_INCOMPATIBLE,
                suggestions=[
                    "Specify model_type explicitly",
                    "Check model file extension",
                    "Ensure model file exists"
                ],
                original_error=e,
                context={"model_path": str(self.model_path)}
            )
        
        # Build final configuration with validation
        try:
            self.config = self._build_validated_config(config)
            logger.debug("Final configuration built and validated")
            
        except ConfigurationError as e:
            logger.error(f"Configuration building failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in config building: {e}")
            raise ConfigurationError(
                message="Failed to build final configuration",
                original_error=e
            )
        
        # Create inferencer with validated config
        self.inferencer = self._create_inferencer()
    
    def _build_validated_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build final configuration with enhanced validation
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Final validated and merged configuration
            
        Raises:
            ConfigurationError: If configuration validation fails
        """
        try:
            # Get model-specific defaults with validation
            model_defaults = self.global_config.get_model_defaults(self.model_type)
            config = model_defaults.copy()
            logger.debug(f"Model defaults loaded for {self.model_type}: {len(model_defaults)} settings")
            
            # Add preprocessing defaults based on runtime
            detected_runtime = self._detect_runtime()
            preprocessing_defaults = self.global_config.get_preprocessing_defaults(detected_runtime)
            if preprocessing_defaults:
                config.setdefault("preprocessing", {}).update(preprocessing_defaults)
                logger.debug(f"Preprocessing defaults added for {detected_runtime}")
            
            # Add runtime settings with device validation
            mapped_device = self.global_config.get_device_name(self.device)
            config.update({
                "device": mapped_device,
                "runtime": detected_runtime
            })
            logger.debug(f"Device '{self.device}' mapped to '{mapped_device}'")
            
            # Merge with user configuration
            if user_config:
                self._merge_and_validate_user_config(config, user_config)
                logger.debug("User configuration merged and validated")
            
            # Perform final validation if Pydantic is available
            self._validate_final_config(config)
            
            return config
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                message="Failed to build configuration",
                suggestions=[
                    "Check model type is supported",
                    "Verify device name is valid",
                    "Check user configuration format"
                ],
                original_error=e,
                context={
                    "model_type": self.model_type,
                    "device": self.device,
                    "runtime": detected_runtime
                }
            )
    
    def _merge_and_validate_user_config(self, config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """Merge user configuration with validation"""
        try:
            self._merge_config(config, user_config)
        except Exception as e:
            raise ConfigurationError(
                message="Failed to merge user configuration",
                suggestions=[
                    "Check user configuration format",
                    "Ensure all values are valid types",
                    "Verify required fields are present"
                ],
                original_error=e
            )
    
    def _validate_final_config(self, config: Dict[str, Any]) -> None:
        """Perform final configuration validation"""
        if not self.global_config.typed:
            return  # Skip validation if Pydantic not available
        
        # Validate specific configuration rules
        validation_warnings = []
        
        # YOLO-specific validations
        if self.model_type.startswith("yolo"):
            input_size = config.get("input_size", 640)
            if input_size % 32 != 0:
                validation_warnings.append(f"YOLO input size {input_size} is not divisible by 32")
            
            confidence = config.get("confidence_threshold", 0.25)
            if not (0.0 <= confidence <= 1.0):
                raise ConfigurationError(
                    message=f"Invalid confidence threshold: {confidence}",
                    suggestions=["Set confidence threshold between 0.0 and 1.0"]
                )
        
        # OpenVINO-specific validations
        if self.model_type.endswith("openvino"):
            if not str(self.model_path).endswith(".xml"):
                validation_warnings.append("OpenVINO models should have .xml extension")
        
        # Log warnings
        for warning in validation_warnings:
            logger.warning(f"Configuration validation warning: {warning}")
    
    def _detect_runtime(self) -> str:
        """Detect runtime from model file extension with validation"""
        if self.runtime != "auto":
            return self.runtime
        
        model_ext = self.model_path.suffix.lower()
        
        # Get supported formats from validated config
        supported_formats = self.global_config.get("supported_formats", {})
        
        for runtime, extensions in supported_formats.items():
            if model_ext in extensions:
                logger.debug(f"Runtime '{runtime}' detected from extension '{model_ext}'")
                return runtime
        
        # Default fallback
        logger.warning(f"No runtime detected for extension '{model_ext}', using ONNX")
        return "onnx"
    
    def _merge_config(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    @with_model_loading_retry()
    def _create_inferencer(self) -> BaseInferencer:
        """Create appropriate inferencer with enhanced error handling"""
        try:
            logger.info(f"Creating {self.model_type} inferencer with validated configuration")
            
            if self.model_type == "yolo" or self.model_type == "yolo_onnx":
                return YOLOInferencer(self.model_path, self.config)
            elif self.model_type == "yolo_openvino":
                return YOLOOpenVINOInferencer(self.model_path, self.config)
            elif self.model_type == "openvino":
                return OpenVINOInferencer(self.model_path, self.config)
            elif self.model_type == "onnx" or self._detect_runtime() == "onnx":
                return ONNXInferencer(self.model_path, self.config)
            elif self._detect_runtime() == "openvino":
                return OpenVINOInferencer(self.model_path, self.config)
            else:
                # Fallback to ONNX for unsupported types
                logger.warning(f"Unsupported model type '{self.model_type}', falling back to ONNX")
                return ONNXInferencer(self.model_path, self.config)
                
        except Exception as e:
            # Enhanced error information
            available_types = ["yolo", "yolo_openvino", "onnx", "openvino"]
            raise ModelError(
                message=f"Failed to create inferencer for model type '{self.model_type}'",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
                suggestions=[
                    "Check if model file exists and is readable",
                    "Verify model format is supported",
                    f"Supported model types: {', '.join(available_types)}",
                    "Try different runtime (onnx/openvino)",
                    "Check system requirements and dependencies"
                ],
                recovery_actions=[
                    "Use auto model type detection",
                    "Convert model to different format",
                    "Check available runtimes",
                    "Verify model file integrity"
                ],
                original_error=e,
                context={
                    "model_path": str(self.model_path),
                    "model_type": self.model_type,
                    "runtime": self.runtime,
                    "device": self.device,
                    "config_keys": list(self.config.keys()) if self.config else []
                }
            )
    
    @with_inference_retry()
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run inference with enhanced error handling and validation"""
        try:
            logger.debug(f"Running inference with {self.model_type} model")
            result = self.inferencer.predict(input_data)
            
            # Add metadata if configured
            if self.global_config.typed:
                output_config = self.global_config.typed.output if hasattr(self.global_config.typed, 'output') else None
                if output_config and getattr(output_config, 'include_metadata', {}).get('model_info', False):
                    result['_metadata'] = {
                        'model_type': self.model_type,
                        'device': self.config.get('device'),
                        'runtime': self.config.get('runtime')
                    }
            
            return result
            
        except Exception as e:
            if not isinstance(e, (ModelError, ConfigurationError)):
                # Wrap unexpected errors
                raise ModelError(
                    message=f"Inference failed for {self.model_type} model",
                    error_code=ErrorCode.INFERENCE_FAILED,
                    suggestions=[
                        "Check input data format",
                        "Verify model is compatible with input",
                        "Try different device or runtime"
                    ],
                    original_error=e,
                    context={
                        "model_type": self.model_type,
                        "input_type": type(input_data).__name__
                    }
                )
            raise
    
    def predict_batch(self, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """Run batch inference with enhanced error handling"""
        if not input_batch:
            raise ConfigurationError(
                message="Empty input batch provided",
                suggestions=["Provide at least one input item"]
            )
        
        logger.debug(f"Running batch inference on {len(input_batch)} items")
        return self.inferencer.predict_batch(input_batch)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information with validation status"""
        info = self.inferencer.get_model_info()
        
        # Add validation information
        info.update({
            "validation_enabled": self.global_config.typed is not None,
            "model_type_detected": self.model_type,
            "config_validation_warnings": self.global_config.validate() if hasattr(self.global_config, 'validate') else []
        })
        
        return info
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return detailed report"""
        validation_report = {
            "validation_enabled": self.global_config.typed is not None,
            "config_valid": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            if hasattr(self.global_config, 'validate'):
                warnings = self.global_config.validate()
                validation_report["warnings"] = warnings
                
            # Additional custom validations
            if self.model_type.startswith("yolo"):
                input_size = self.config.get("input_size", 640)
                if input_size % 32 != 0:
                    validation_report["warnings"].append(f"YOLO input size {input_size} not divisible by 32")
            
        except Exception as e:
            validation_report["config_valid"] = False
            validation_report["errors"].append(str(e))
        
        return validation_report


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_inference_engine(*args, **kwargs) -> EnhancedInferenceEngine:
    """
    Factory function for creating enhanced inference engine
    
    Drop-in replacement for InferenceEngine with validation
    """
    return EnhancedInferenceEngine(*args, **kwargs)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_enhanced_runtime():
    """Demonstrate enhanced runtime with validation"""
    print("Enhanced InferX Runtime with Pydantic Validation")
    print("=" * 60)
    
    try:
        # Example 1: Valid configuration
        print("\n1. Creating engine with valid configuration:")
        engine = EnhancedInferenceEngine(
            model_path="yolov8_demo.onnx",  # Fake path for demo
            model_type="yolo",
            device="auto"
        )
        print("✅ Engine created successfully with validation")
        
        # Example 2: Configuration validation report
        print("\n2. Configuration validation report:")
        validation_report = engine.validate_configuration()
        print(f"   Validation enabled: {validation_report['validation_enabled']}")
        print(f"   Configuration valid: {validation_report['config_valid']}")
        if validation_report['warnings']:
            print(f"   Warnings: {len(validation_report['warnings'])}")
        
        # Example 3: Model information with validation status
        print("\n3. Model information:")
        model_info = engine.get_model_info()
        print(f"   Model type: {model_info.get('model_type_detected')}")
        print(f"   Validation enabled: {model_info.get('validation_enabled')}")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        if e.suggestions:
            print("   Suggestions:")
            for suggestion in e.suggestions:
                print(f"     • {suggestion}")
    
    except ModelError as e:
        print(f"❌ Model error: {e}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n🎯 Enhanced Runtime Benefits:")
    print("   • Type-safe configuration with Pydantic validation")
    print("   • Enhanced error messages with actionable suggestions")
    print("   • Automatic configuration validation during initialization")
    print("   • Backward compatibility with existing runtime API")
    print("   • Detailed validation reports and model information")


if __name__ == "__main__":
    demonstrate_enhanced_runtime()