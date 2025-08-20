"""Configuration management for InferX"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class InferXConfig:
    """Central configuration manager for InferX"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration
        
        Args:
            config_path: Path to configuration file. If None, loads default config hierarchy.
        """
        self._config = {}
        self._load_config_hierarchy(config_path)
    
    def _load_config_hierarchy(self, user_config_path: Optional[Union[str, Path]] = None):
        """Load configuration in priority order
        
        Priority (highest to lowest):
        1. User-specified config file (if provided)
        2. Project local config (./inferx_config.yaml)
        3. User global config (~/.inferx/config.yaml)
        4. Package default config
        """
        # 1. Start with package default config
        self._config = self._load_default_config()
        
        # 2. Load user global config if exists
        user_global_config = self._get_user_global_config_path()
        if user_global_config.exists():
            logger.debug(f"Loading user global config: {user_global_config}")
            self._merge_config_file(user_global_config)
        
        # 3. Load project local config if exists
        project_local_config = Path.cwd() / "inferx_config.yaml"
        if project_local_config.exists():
            logger.debug(f"Loading project local config: {project_local_config}")
            self._merge_config_file(project_local_config)
        
        # 4. Load user-specified config if provided
        if user_config_path:
            user_config_path = Path(user_config_path)
            if user_config_path.exists():
                logger.debug(f"Loading user-specified config: {user_config_path}")
                self._merge_config_file(user_config_path)
            else:
                logger.warning(f"User-specified config file not found: {user_config_path}")
    
    def _get_user_global_config_path(self) -> Path:
        """Get user global config path"""
        home_dir = Path.home()
        return home_dir / ".inferx" / "config.yaml"
    
    def _get_default_config_path(self) -> Path:
        """Get package default config path"""
        # Get the directory where this config.py file is located
        current_dir = Path(__file__).parent
        return current_dir / "configs" / "default.yaml"
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from package config file"""
        default_config_path = self._get_default_config_path()
        
        if not default_config_path.exists():
            logger.error(f"Default config file not found: {default_config_path}")
            return self._get_fallback_config()
        
        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded default config from: {default_config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load default config: {e}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration if default config fails to load"""
        return {
            "model_detection": {
                "yolo_keywords": ["yolo", "yolov", "yolov8", "yolov11"]
            },
            "device_mapping": {
                "auto": "AUTO",
                "cpu": "CPU", 
                "gpu": "GPU"
            },
            "model_defaults": {
                "yolo": {
                    "input_size": 640,
                    "confidence_threshold": 0.25,
                    "nms_threshold": 0.45
                }
            }
        }
    
    def _merge_config_file(self, config_path: Path) -> None:
        """Load and merge configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Merge user config with current config
            self._merge_configs(self._config, user_config)
            logger.info(f"Merged configuration from: {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    
    def _merge_configs(self, default: Dict, user: Dict) -> None:
        """Recursively merge user config with default config"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., "model_defaults.yolo.input_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation
        
        Args:
            key_path: Configuration key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set final value
        config[keys[-1]] = value
    
    def detect_model_type(self, model_path: Path, runtime_hint: Optional[str] = None) -> str:
        """Detect model type based on filename and extension
        
        Args:
            model_path: Path to model file
            runtime_hint: Optional runtime hint (onnx, openvino, etc.)
            
        Returns:
            Detected model type
        """
        file_extension = model_path.suffix.lower()
        model_name = model_path.stem.lower()
        
        # Get keyword lists from config
        yolo_keywords = self.get("model_detection.yolo_keywords", [])
        anomalib_keywords = self.get("model_detection.anomalib_keywords", [])
        classification_keywords = self.get("model_detection.classification_keywords", [])
        
        # Check if it's a YOLO model
        if any(keyword in model_name for keyword in yolo_keywords):
            if file_extension == '.xml':
                return "yolo_openvino"
            elif file_extension == '.onnx':
                return "yolo_onnx"
        
        # Check if it's an anomaly detection model
        if any(keyword in model_name for keyword in anomalib_keywords):
            if file_extension == '.xml':
                return "anomalib_openvino"
            elif file_extension == '.onnx':
                return "anomalib_onnx"
        
        # Check if it's a classification model
        if any(keyword in model_name for keyword in classification_keywords):
            if file_extension == '.xml':
                return "classification_openvino"
            elif file_extension == '.onnx':
                return "classification_onnx"
        
        # Default based on extension and runtime hint
        if file_extension == '.xml':
            return "openvino"
        elif file_extension == '.onnx':
            return "onnx"
        else:
            return runtime_hint or "onnx"
    
    def get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration dictionary
        """
        # Extract base model type (e.g., "yolo" from "yolo_openvino")
        base_type = model_type.split('_')[0]
        return self.get(f"model_defaults.{base_type}", {})
    
    def get_preprocessing_defaults(self, runtime: str) -> Dict[str, Any]:
        """Get default preprocessing configuration for runtime
        
        Args:
            runtime: Runtime type (onnx, openvino, etc.)
            
        Returns:
            Default preprocessing configuration
        """
        return self.get(f"preprocessing_defaults.{runtime}", {})
    
    def get_device_name(self, device: str) -> str:
        """Map device name to runtime-specific name
        
        Args:
            device: Device name
            
        Returns:
            Runtime-specific device name
        """
        return self.get(f"device_mapping.{device.lower()}", device.upper())
    
    def get_runtime_providers(self, runtime: str, device: str) -> List[str]:
        """Get runtime providers for device
        
        Args:
            runtime: Runtime type
            device: Device type
            
        Returns:
            List of providers
        """
        return self.get(f"runtime_preferences.{runtime}.providers.{device}", [])
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()


# Global configuration instance
_global_config = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> InferXConfig:
    """Get global configuration instance
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        InferXConfig instance
    """
    global _global_config
    
    if _global_config is None or config_path is not None:
        _global_config = InferXConfig(config_path)
    
    return _global_config


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to parse configuration file {config_path}: {e}")


def create_user_config_template(output_path: Union[str, Path]) -> None:
    """Create a user configuration template file
    
    Args:
        output_path: Where to save the template
    """
    output_path = Path(output_path)
    
    template = """# InferX User Configuration
# This file overrides default settings for your specific needs
# Uncomment and modify the sections you want to customize

# Model Detection Settings
# model_detection:
#   yolo_keywords:
#     - "yolo"
#     - "yolov8"
#     - "custom_yolo_name"

# Device Preferences  
# device_mapping:
#   auto: "GPU"  # Prefer GPU over CPU when auto is selected

# Performance Settings
# performance_presets:
#   custom_preset:
#     openvino:
#       performance_hint: "LATENCY"
#       num_streams: 1
#     onnx:
#       providers: ["CUDAExecutionProvider"]

# Model-Specific Overrides
# model_defaults:
#   yolo:
#     confidence_threshold: 0.3  # Lower threshold for more detections
#     input_size: 1024           # Higher resolution for better accuracy
#     class_names:               # Custom class names for your model
#       - "my_class_1"
#       - "my_class_2"

# Logging Configuration
# logging:
#   level: "DEBUG"
#   categories:
#     preprocessing: true
#     postprocessing: true

# Output Settings
# output:
#   default_format: "yaml"
#   include_metadata:
#     config_used: true
"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        logger.info(f"Created user config template at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create config template: {e}")
        raise


def setup_user_config_directory() -> Path:
    """Setup user config directory and return path
    
    Returns:
        Path to user config directory
    """
    config_dir = Path.home() / ".inferx"
    config_dir.mkdir(exist_ok=True)
    
    # Create cache directory too
    cache_dir = config_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    return config_dir


def init_user_config() -> None:
    """Initialize user configuration directory and create template"""
    config_dir = setup_user_config_directory()
    config_file = config_dir / "config.yaml"
    
    if not config_file.exists():
        create_user_config_template(config_file)
        print(f"Created user config at: {config_file}")
        print("Edit this file to customize InferX for your needs.")
    else:
        print(f"User config already exists at: {config_file}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of warnings
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check model detection keywords
    if "model_detection" in config:
        for key, keywords in config["model_detection"].items():
            if not isinstance(keywords, list):
                warnings.append(f"model_detection.{key} should be a list")
            elif not all(isinstance(k, str) for k in keywords):
                warnings.append(f"model_detection.{key} should contain only strings")
    
    # Check device mapping
    if "device_mapping" in config:
        for device, mapped in config["device_mapping"].items():
            if not isinstance(mapped, str):
                warnings.append(f"device_mapping.{device} should be a string")
    
    # Check model defaults
    if "model_defaults" in config:
        yolo_config = config["model_defaults"].get("yolo", {})
        if "confidence_threshold" in yolo_config:
            threshold = yolo_config["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                warnings.append("yolo.confidence_threshold should be a number between 0 and 1")
    
    return warnings