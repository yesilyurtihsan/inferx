"""
Error Recovery Mechanisms for InferX

This module provides automatic error recovery mechanisms including retry logic,
fallback strategies, and graceful degradation capabilities.
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, List, Any, Optional, Dict, Union
from pathlib import Path

from .exceptions import (
    InferXError,
    ModelError,
    RuntimeError,
    DeviceNotAvailableError,
    ErrorCode
)


logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


class FallbackStrategy:
    """Base class for fallback strategies"""
    
    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the given error"""
        raise NotImplementedError
    
    def execute_fallback(self, original_func: Callable, *args, **kwargs) -> Any:
        """Execute the fallback strategy"""
        raise NotImplementedError


class DeviceFallbackStrategy(FallbackStrategy):
    """Fallback from GPU to CPU when device is not available"""
    
    def __init__(self, fallback_device: str = "cpu"):
        self.fallback_device = fallback_device
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, DeviceNotAvailableError)
    
    def execute_fallback(self, original_func: Callable, *args, **kwargs) -> Any:
        """Try again with CPU device"""
        logger.warning("GPU device not available, falling back to CPU")
        
        # Modify kwargs to use CPU device
        if 'config' in kwargs and kwargs['config']:
            kwargs['config'] = kwargs['config'].copy()
            kwargs['config']['device'] = self.fallback_device
        elif len(args) > 1:
            # If config is passed as positional argument
            if isinstance(args[1], dict):
                new_config = args[1].copy()
                new_config['device'] = self.fallback_device
                args = list(args)
                args[1] = new_config
                args = tuple(args)
        
        return original_func(*args, **kwargs)


class RuntimeFallbackStrategy(FallbackStrategy):
    """Fallback between different runtimes (ONNX <-> OpenVINO)"""
    
    def __init__(self, fallback_runtime: str = "onnx"):
        self.fallback_runtime = fallback_runtime
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, (ModelError, RuntimeError))
    
    def execute_fallback(self, original_func: Callable, *args, **kwargs) -> Any:
        """Try again with different runtime"""
        logger.warning(f"Runtime failed, falling back to {self.fallback_runtime}")
        
        # Modify kwargs to use fallback runtime
        if 'config' in kwargs and kwargs['config']:
            kwargs['config'] = kwargs['config'].copy()
            kwargs['config']['runtime'] = self.fallback_runtime
        
        return original_func(*args, **kwargs)


class ModelFormatFallbackStrategy(FallbackStrategy):
    """Fallback to alternative model format if available"""
    
    def __init__(self, format_mappings: Dict[str, List[str]] = None):
        self.format_mappings = format_mappings or {
            '.xml': ['.onnx'],  # If OpenVINO model fails, try ONNX
            '.onnx': ['.xml']   # If ONNX model fails, try OpenVINO
        }
    
    def can_handle(self, error: Exception) -> bool:
        return isinstance(error, ModelError)
    
    def execute_fallback(self, original_func: Callable, *args, **kwargs) -> Any:
        """Try with alternative model format"""
        # Extract model path from arguments
        model_path = None
        if args:
            model_path = Path(args[0])
        elif 'model_path' in kwargs:
            model_path = Path(kwargs['model_path'])
        
        if not model_path:
            raise Exception("Could not determine model path for fallback")
        
        # Look for alternative formats
        current_ext = model_path.suffix.lower()
        if current_ext in self.format_mappings:
            for alt_ext in self.format_mappings[current_ext]:
                alt_path = model_path.with_suffix(alt_ext)
                if alt_path.exists():
                    logger.warning(f"Model format failed, trying alternative: {alt_path}")
                    
                    # Update arguments with new model path
                    if args:
                        new_args = list(args)
                        new_args[0] = str(alt_path)
                        args = tuple(new_args)
                    else:
                        kwargs['model_path'] = str(alt_path)
                    
                    return original_func(*args, **kwargs)
        
        # No alternative found, re-raise original error
        raise


def with_retry(
    retry_config: Optional[RetryConfig] = None,
    retry_on: Optional[List[Type[Exception]]] = None,
    fallback_strategies: Optional[List[FallbackStrategy]] = None
):
    """Decorator that adds retry logic and fallback strategies to functions
    
    Args:
        retry_config: Configuration for retry behavior
        retry_on: List of exception types to retry on
        fallback_strategies: List of fallback strategies to try
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    if retry_on is None:
        retry_on = [InferXError, ConnectionError, TimeoutError]
    
    if fallback_strategies is None:
        fallback_strategies = [
            DeviceFallbackStrategy(),
            RuntimeFallbackStrategy(),
            ModelFormatFallbackStrategy()
        ]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    should_retry = any(isinstance(e, exc_type) for exc_type in retry_on)
                    
                    if not should_retry:
                        # Try fallback strategies before giving up
                        for strategy in fallback_strategies:
                            if strategy.can_handle(e):
                                try:
                                    logger.info(f"Attempting fallback strategy: {strategy.__class__.__name__}")
                                    return strategy.execute_fallback(func, *args, **kwargs)
                                except Exception as fallback_error:
                                    logger.warning(f"Fallback strategy failed: {fallback_error}")
                                    last_exception = fallback_error
                        
                        # No fallback worked, raise original exception
                        raise e
                    
                    # Calculate delay for next attempt
                    if attempt < retry_config.max_attempts - 1:  # Don't delay on last attempt
                        delay = retry_config.base_delay
                        
                        if retry_config.exponential_backoff:
                            delay *= (2 ** attempt)
                        
                        delay = min(delay, retry_config.max_delay)
                        
                        if retry_config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                    
                    # Try fallback strategies on last attempt
                    if attempt == retry_config.max_attempts - 1:
                        for strategy in fallback_strategies:
                            if strategy.can_handle(e):
                                try:
                                    logger.info(f"Final attempt with fallback: {strategy.__class__.__name__}")
                                    return strategy.execute_fallback(func, *args, **kwargs)
                                except Exception as fallback_error:
                                    logger.error(f"Final fallback failed: {fallback_error}")
                                    last_exception = fallback_error
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


class GracefulDegradation:
    """Provides graceful degradation capabilities for inference operations"""
    
    def __init__(self):
        self.degraded_mode = False
        self.degradation_reason = None
        self.alternative_outputs = {}
    
    def enable_degraded_mode(self, reason: str):
        """Enable degraded mode with reason"""
        self.degraded_mode = True
        self.degradation_reason = reason
        logger.warning(f"Entering degraded mode: {reason}")
    
    def disable_degraded_mode(self):
        """Disable degraded mode"""
        self.degraded_mode = False
        self.degradation_reason = None
        self.alternative_outputs.clear()
        logger.info("Exiting degraded mode")
    
    def is_degraded(self) -> bool:
        """Check if system is in degraded mode"""
        return self.degraded_mode
    
    def add_alternative_output(self, key: str, value: Any):
        """Add alternative output for degraded mode"""
        self.alternative_outputs[key] = value
    
    def get_degraded_response(self, operation: str) -> Dict[str, Any]:
        """Get response for degraded mode"""
        return {
            "degraded_mode": True,
            "reason": self.degradation_reason,
            "operation": operation,
            "alternative_data": self.alternative_outputs.copy(),
            "suggestions": [
                "Check system resources",
                "Restart the service",
                "Use alternative model or configuration",
                "Contact support if issue persists"
            ]
        }


# Global graceful degradation instance
graceful_degradation = GracefulDegradation()


def with_graceful_degradation(degraded_response_factory: Optional[Callable] = None):
    """Decorator that provides graceful degradation for functions
    
    Args:
        degraded_response_factory: Function to create degraded response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # If we were in degraded mode and this succeeded, exit degraded mode
                if graceful_degradation.is_degraded():
                    graceful_degradation.disable_degraded_mode()
                
                return result
                
            except Exception as e:
                # Enter degraded mode if not already
                if not graceful_degradation.is_degraded():
                    graceful_degradation.enable_degraded_mode(str(e))
                
                # Return degraded response or re-raise
                if degraded_response_factory:
                    return degraded_response_factory(e, *args, **kwargs)
                else:
                    return graceful_degradation.get_degraded_response(func.__name__)
        
        return wrapper
    return decorator


# Convenience function for common retry scenarios
def with_inference_retry():
    """Decorator for inference operations with common retry/fallback strategies"""
    return with_retry(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        retry_on=[ModelError, RuntimeError, ConnectionError],
        fallback_strategies=[
            DeviceFallbackStrategy(),
            RuntimeFallbackStrategy()
        ]
    )


def with_model_loading_retry():
    """Decorator for model loading with format fallback"""
    return with_retry(
        retry_config=RetryConfig(max_attempts=2, base_delay=0.5),
        retry_on=[ModelError],
        fallback_strategies=[
            ModelFormatFallbackStrategy(),
            DeviceFallbackStrategy()
        ]
    )