"""InferX - Lightweight ML Inference Runtime"""

__version__ = "1.0.0"
__author__ = "InferX Team"
__email__ = "team@inferx.dev"

from .runtime import InferenceEngine
from .exceptions import (
    InferXError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    InferenceFailedError,
    InputError,
    InputNotFoundError,
    ConfigurationError,
    ErrorCode
)

__all__ = [
    "InferenceEngine",
    "InferXError",
    "ModelError", 
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "InferenceFailedError",
    "InputError",
    "InputNotFoundError", 
    "ConfigurationError",
    "ErrorCode"
]