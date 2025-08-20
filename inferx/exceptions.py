"""
InferX Custom Exception Classes

This module defines custom exception classes with error codes, actionable suggestions,
and structured error information for better error handling and debugging.
"""

import logging
from typing import List, Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Error codes for different types of InferX errors"""
    
    # Model Loading Errors (1000-1099)
    MODEL_NOT_FOUND = "INFERX_1001"
    MODEL_INVALID_FORMAT = "INFERX_1002"
    MODEL_LOAD_FAILED = "INFERX_1003"
    MODEL_INCOMPATIBLE = "INFERX_1004"
    MODEL_CORRUPTED = "INFERX_1005"
    MODEL_NOT_LOADED = "INFERX_1006"
    
    # Runtime Errors (1100-1199)
    RUNTIME_NOT_AVAILABLE = "INFERX_1101"
    RUNTIME_INIT_FAILED = "INFERX_1102"
    DEVICE_NOT_AVAILABLE = "INFERX_1103"
    PROVIDER_NOT_FOUND = "INFERX_1104"
    MEMORY_ERROR = "INFERX_1105"
    
    # Input/Output Errors (1200-1299)
    INPUT_NOT_FOUND = "INFERX_1201"
    INPUT_INVALID_FORMAT = "INFERX_1202"
    INPUT_INVALID_SIZE = "INFERX_1203"
    INPUT_INVALID_TYPE = "INFERX_1204"
    OUTPUT_WRITE_FAILED = "INFERX_1205"
    
    # Configuration Errors (1300-1399)
    CONFIG_NOT_FOUND = "INFERX_1301"
    CONFIG_INVALID_FORMAT = "INFERX_1302"
    CONFIG_VALIDATION_FAILED = "INFERX_1303"
    CONFIG_MISSING_REQUIRED = "INFERX_1304"
    
    # Inference Errors (1400-1499)
    INFERENCE_FAILED = "INFERX_1401"
    PREPROCESSING_FAILED = "INFERX_1402"
    POSTPROCESSING_FAILED = "INFERX_1403"
    BATCH_SIZE_MISMATCH = "INFERX_1404"
    
    # Template Generation Errors (1500-1599)
    TEMPLATE_NOT_FOUND = "INFERX_1501"
    TEMPLATE_GENERATION_FAILED = "INFERX_1502"
    PROJECT_ALREADY_EXISTS = "INFERX_1503"
    TEMPLATE_INVALID = "INFERX_1504"
    
    # Network/IO Errors (1600-1699)
    NETWORK_TIMEOUT = "INFERX_1601"
    FILE_PERMISSION_DENIED = "INFERX_1602"
    DISK_SPACE_FULL = "INFERX_1603"
    PATH_TRAVERSAL_DETECTED = "INFERX_1604"


class InferXError(Exception):
    """Base exception class for all InferX errors
    
    Provides structured error information with error codes, suggestions,
    and context for better debugging and error recovery.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        recovery_actions: Optional[List[str]] = None
    ):
        """Initialize InferX error
        
        Args:
            message: Human-readable error message
            error_code: Structured error code
            suggestions: List of actionable suggestions for user
            context: Additional context information
            original_error: Original exception that caused this error
            recovery_actions: Suggested recovery actions
        """
        super().__init__(message)
        self.error_code = error_code
        self.suggestions = suggestions or []
        self.context = context or {}
        self.original_error = original_error
        self.recovery_actions = recovery_actions or []
        
        # Log the error with structured information
        self._log_error()
    
    def _log_error(self):
        """Log error with structured information"""
        logger = logging.getLogger(__name__)
        
        log_data = {
            "error_code": self.error_code.value,
            "message": str(self),
            "suggestions": self.suggestions,
            "context": self.context,
            "recovery_actions": self.recovery_actions
        }
        
        if self.original_error:
            log_data["original_error"] = str(self.original_error)
            log_data["original_error_type"] = type(self.original_error).__name__
        
        logger.error(f"InferX Error: {self.error_code.value}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization"""
        return {
            "error_code": self.error_code.value,
            "message": str(self),
            "suggestions": self.suggestions,
            "context": self.context,
            "recovery_actions": self.recovery_actions,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        """Enhanced string representation with suggestions"""
        base_msg = super().__str__()
        
        if self.suggestions:
            suggestions_str = "\n".join(f"  • {suggestion}" for suggestion in self.suggestions)
            base_msg += f"\n\nSuggestions:\n{suggestions_str}"
        
        if self.recovery_actions:
            recovery_str = "\n".join(f"  • {action}" for action in self.recovery_actions)
            base_msg += f"\n\nRecovery Actions:\n{recovery_str}"
        
        base_msg += f"\n\nError Code: {self.error_code.value}"
        
        return base_msg


class ModelError(InferXError):
    """Exception raised for model-related errors"""
    pass


class ModelNotFoundError(ModelError):
    """Exception raised when model file is not found"""
    
    def __init__(self, model_path: str, **kwargs):
        suggestions = [
            f"Check if the file exists: {model_path}",
            "Verify the file path is correct",
            "Ensure you have read permissions for the file",
            "Try using an absolute path instead of relative path"
        ]
        
        recovery_actions = [
            "Use a different model file",
            "Download the model file again",
            "Check your current working directory"
        ]
        
        super().__init__(
            message=f"Model file not found: {model_path}",
            error_code=ErrorCode.MODEL_NOT_FOUND,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"model_path": model_path},
            **kwargs
        )


class ModelLoadError(ModelError):
    """Exception raised when model loading fails"""
    
    def __init__(self, model_path: str, runtime: str = "unknown", **kwargs):
        suggestions = [
            f"Verify the model format is compatible with {runtime} runtime",
            "Check if the model file is corrupted",
            "Ensure you have sufficient memory available",
            "Try using a different runtime (ONNX, OpenVINO)",
            "Verify model file permissions"
        ]
        
        recovery_actions = [
            "Try loading with CPU device instead of GPU",
            "Reduce batch size or input resolution",
            "Free up system memory and try again",
            "Convert model to a different format"
        ]
        
        super().__init__(
            message=f"Failed to load model: {model_path} (runtime: {runtime})",
            error_code=ErrorCode.MODEL_LOAD_FAILED,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"model_path": model_path, "runtime": runtime},
            **kwargs
        )


class ModelIncompatibleError(ModelError):
    """Exception raised when model is incompatible with runtime"""
    
    def __init__(self, model_path: str, runtime: str, expected_format: str, **kwargs):
        suggestions = [
            f"Convert model to {expected_format} format for {runtime} runtime",
            f"Use a different runtime that supports this model format",
            "Check model export parameters",
            "Verify model architecture compatibility"
        ]
        
        recovery_actions = [
            "Export model again with correct format",
            "Use auto runtime detection",
            "Try different model format"
        ]
        
        super().__init__(
            message=f"Model {model_path} is incompatible with {runtime} runtime. Expected format: {expected_format}",
            error_code=ErrorCode.MODEL_INCOMPATIBLE,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={
                "model_path": model_path,
                "runtime": runtime,
                "expected_format": expected_format
            },
            **kwargs
        )


class RuntimeError(InferXError):
    """Exception raised for runtime-related errors"""
    pass


class DeviceNotAvailableError(RuntimeError):
    """Exception raised when requested device is not available"""
    
    def __init__(self, device: str, available_devices: List[str] = None, **kwargs):
        available_str = ", ".join(available_devices) if available_devices else "unknown"
        
        suggestions = [
            f"Use one of the available devices: {available_str}",
            "Use 'auto' device for automatic selection",
            "Install appropriate drivers for the requested device",
            "Check device availability with device management tools"
        ]
        
        recovery_actions = [
            "Fall back to CPU device",
            "Use auto device selection",
            "Check system requirements"
        ]
        
        super().__init__(
            message=f"Device '{device}' is not available. Available devices: {available_str}",
            error_code=ErrorCode.DEVICE_NOT_AVAILABLE,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"requested_device": device, "available_devices": available_devices},
            **kwargs
        )


class InferenceError(InferXError):
    """Exception raised for inference-related errors"""
    pass


class InferenceFailedError(InferenceError):
    """Exception raised when inference fails"""
    
    def __init__(self, model_type: str = "unknown", **kwargs):
        suggestions = [
            "Check input data format and size",
            "Verify model is loaded correctly",
            "Ensure sufficient memory is available",
            "Try reducing batch size",
            "Check input preprocessing"
        ]
        
        recovery_actions = [
            "Reload the model",
            "Use smaller input size",
            "Free up system memory",
            "Try different device (CPU/GPU)"
        ]
        
        super().__init__(
            message=f"Inference failed for {model_type} model",
            error_code=ErrorCode.INFERENCE_FAILED,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"model_type": model_type},
            **kwargs
        )


class InputError(InferXError):
    """Exception raised for input-related errors"""
    pass


class InputNotFoundError(InputError):
    """Exception raised when input file is not found"""
    
    def __init__(self, input_path: str, **kwargs):
        suggestions = [
            f"Check if the file exists: {input_path}",
            "Verify the file path is correct",
            "Ensure you have read permissions for the file",
            "Use absolute path instead of relative path"
        ]
        
        recovery_actions = [
            "Use a different input file",
            "Check current working directory",
            "Verify file permissions"
        ]
        
        super().__init__(
            message=f"Input file not found: {input_path}",
            error_code=ErrorCode.INPUT_NOT_FOUND,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"input_path": input_path},
            **kwargs
        )


class InputInvalidFormatError(InputError):
    """Exception raised when input format is invalid"""
    
    def __init__(self, input_path: str, expected_formats: List[str] = None, **kwargs):
        expected_str = ", ".join(expected_formats) if expected_formats else "supported format"
        
        suggestions = [
            f"Convert input to one of the supported formats: {expected_str}",
            "Check file extension and content",
            "Use a different input file",
            "Verify file is not corrupted"
        ]
        
        recovery_actions = [
            "Convert file to supported format",
            "Use different input file",
            "Check file integrity"
        ]
        
        super().__init__(
            message=f"Invalid input format: {input_path}. Expected: {expected_str}",
            error_code=ErrorCode.INPUT_INVALID_FORMAT,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"input_path": input_path, "expected_formats": expected_formats},
            **kwargs
        )


class ConfigurationError(InferXError):
    """Exception raised for configuration-related errors"""
    pass


class ConfigNotFoundError(ConfigurationError):
    """Exception raised when configuration file is not found"""
    
    def __init__(self, config_path: str, **kwargs):
        suggestions = [
            f"Create configuration file at: {config_path}",
            "Use default configuration (remove --config option)",
            "Check file path spelling",
            "Use 'inferx config --template config.yaml' to create template"
        ]
        
        recovery_actions = [
            "Create default configuration",
            "Use built-in defaults",
            "Copy from example configuration"
        ]
        
        super().__init__(
            message=f"Configuration file not found: {config_path}",
            error_code=ErrorCode.CONFIG_NOT_FOUND,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"config_path": config_path},
            **kwargs
        )


class TemplateError(InferXError):
    """Exception raised for template-related errors"""
    pass


class TemplateNotFoundError(TemplateError):
    """Exception raised when template is not found"""
    
    def __init__(self, template_name: str, available_templates: List[str] = None, **kwargs):
        available_str = ", ".join(available_templates) if available_templates else "unknown"
        
        suggestions = [
            f"Use one of the available templates: {available_str}",
            "Check template name spelling",
            "Run 'inferx template --list' to see available templates",
            "Create custom template"
        ]
        
        recovery_actions = [
            "Use different template",
            "Create custom template",
            "Check available templates"
        ]
        
        super().__init__(
            message=f"Template '{template_name}' not found. Available templates: {available_str}",
            error_code=ErrorCode.TEMPLATE_NOT_FOUND,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"template_name": template_name, "available_templates": available_templates},
            **kwargs
        )


class SecurityError(InferXError):
    """Exception raised for security-related errors"""
    pass


class PathTraversalError(SecurityError):
    """Exception raised when path traversal is detected"""
    
    def __init__(self, attempted_path: str, **kwargs):
        suggestions = [
            "Use relative paths within the project directory",
            "Avoid using '..' in file paths",
            "Use absolute paths that don't traverse outside allowed directories",
            "Contact administrator if this path should be allowed"
        ]
        
        recovery_actions = [
            "Use different file path",
            "Place file in allowed directory",
            "Contact system administrator"
        ]
        
        super().__init__(
            message=f"Path traversal detected: {attempted_path}",
            error_code=ErrorCode.PATH_TRAVERSAL_DETECTED,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            context={"attempted_path": attempted_path},
            **kwargs
        )


# Convenience functions for common error scenarios
def create_model_error(model_path: str, error: Exception, runtime: str = "unknown") -> ModelError:
    """Create appropriate model error based on the original exception"""
    if isinstance(error, FileNotFoundError):
        return ModelNotFoundError(model_path, original_error=error)
    elif "format" in str(error).lower() or "extension" in str(error).lower():
        return ModelIncompatibleError(model_path, runtime, "compatible format", original_error=error)
    else:
        return ModelLoadError(model_path, runtime, original_error=error)


def create_input_error(input_path: str, error: Exception) -> InputError:
    """Create appropriate input error based on the original exception"""
    if isinstance(error, FileNotFoundError):
        return InputNotFoundError(input_path, original_error=error)
    elif "format" in str(error).lower():
        return InputInvalidFormatError(input_path, original_error=error)
    else:
        return InputError(
            message=f"Input processing failed: {input_path}",
            error_code=ErrorCode.INPUT_INVALID_TYPE,
            suggestions=["Check input file format", "Verify file is not corrupted"],
            original_error=error,
            context={"input_path": input_path}
        )


def create_inference_error(model_type: str, error: Exception) -> InferenceError:
    """Create appropriate inference error based on the original exception"""
    if "memory" in str(error).lower():
        return InferenceError(
            message=f"Inference failed due to memory error: {model_type}",
            error_code=ErrorCode.MEMORY_ERROR,
            suggestions=[
                "Reduce batch size",
                "Use smaller input resolution",
                "Free up system memory",
                "Try CPU device instead of GPU"
            ],
            recovery_actions=[
                "Restart application",
                "Close other applications",
                "Use smaller model"
            ],
            original_error=error,
            context={"model_type": model_type}
        )
    else:
        return InferenceFailedError(model_type, original_error=error)