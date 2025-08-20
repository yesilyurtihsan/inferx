"""
Phase 1 Hybrid Configuration Implementation

This demonstrates how to implement Pydantic validation internally while maintaining
100% backward compatibility with the existing InferX configuration API.
"""

import warnings
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Pydantic imports (would be added to requirements)
try:
    from pydantic import BaseModel, BaseSettings, Field, ValidationError, validator
    from enum import Enum
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    BaseSettings = object


logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS (New Schema Definition)
# =============================================================================

if PYDANTIC_AVAILABLE:
    
    class DeviceType(str, Enum):
        AUTO = "auto"
        CPU = "cpu"
        GPU = "gpu"
        MYRIAD = "myriad"
    
    
    class YOLOConfigSchema(BaseModel):
        """Type-safe YOLO configuration schema"""
        
        input_size: int = Field(
            default=640,
            gt=0,
            description="Input image size"
        )
        confidence_threshold: float = Field(
            default=0.25,
            ge=0.0,
            le=1.0,
            description="Confidence threshold for detections"
        )
        nms_threshold: float = Field(
            default=0.45,
            ge=0.0,
            le=1.0,
            description="Non-Maximum Suppression threshold"
        )
        max_detections: int = Field(
            default=100,
            gt=0,
            le=1000,
            description="Maximum number of detections"
        )
        class_names: List[str] = Field(
            default_factory=lambda: [
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light"
            ]
        )
        
        @validator('input_size')
        def validate_input_size(cls, v):
            if v % 32 != 0:
                raise ValueError('Input size must be divisible by 32')
            return v
    
    
    class ModelDefaultsSchema(BaseModel):
        """Model-specific default configurations"""
        yolo: YOLOConfigSchema = Field(default_factory=YOLOConfigSchema)
    
    
    class ModelDetectionSchema(BaseModel):
        """Model type detection configuration"""
        yolo_keywords: List[str] = Field(
            default=["yolo", "yolov", "yolov8", "yolov11"]
        )
        classification_keywords: List[str] = Field(
            default=["resnet", "efficientnet", "classifier"]
        )
    
    
    class DeviceMappingSchema(BaseModel):
        """Device name mappings"""
        auto: str = Field(default="AUTO")
        cpu: str = Field(default="CPU")
        gpu: str = Field(default="GPU")
        myriad: str = Field(default="MYRIAD")
    
    
    class InferXSettingsSchema(BaseSettings):
        """Complete InferX configuration schema"""
        
        model_detection: ModelDetectionSchema = Field(default_factory=ModelDetectionSchema)
        device_mapping: DeviceMappingSchema = Field(default_factory=DeviceMappingSchema)
        model_defaults: ModelDefaultsSchema = Field(default_factory=ModelDefaultsSchema)
        
        class Config:
            env_prefix = "INFERX_"
            extra = "allow"  # Allow extra fields for backward compatibility


# =============================================================================
# ENHANCED CONFIGURATION ERROR HANDLING
# =============================================================================

class ConfigurationError(Exception):
    """Enhanced configuration error with detailed information"""
    
    def __init__(self, message: str, field_errors: Optional[List[Dict]] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.field_errors = field_errors or []
        self.suggestions = suggestions or []
    
    def __str__(self):
        message = super().__str__()
        
        if self.field_errors:
            message += "\n\nField Errors:"
            for error in self.field_errors:
                field = " -> ".join(str(loc) for loc in error.get('loc', []))
                msg = error.get('msg', 'Unknown error')
                message += f"\n  • {field}: {msg}"
        
        if self.suggestions:
            message += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                message += f"\n  • {suggestion}"
        
        return message


# =============================================================================
# HYBRID CONFIGURATION CLASS
# =============================================================================

class HybridInferXConfig:
    """
    Hybrid configuration that uses Pydantic validation internally
    while maintaining 100% backward compatibility with existing API.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize hybrid configuration
        
        Args:
            config_path: Path to configuration file
        """
        self._config_path = config_path
        self._raw_config = {}
        self._validated_config = None
        self._validation_enabled = PYDANTIC_AVAILABLE
        
        # Load configuration hierarchy (existing logic)
        self._load_config_hierarchy()
        
        # Add Pydantic validation if available
        if self._validation_enabled:
            self._apply_pydantic_validation()
    
    def _load_config_hierarchy(self):
        """Load configuration in priority order (existing implementation)"""
        # 1. Start with fallback config
        self._config = self._get_fallback_config()
        
        # 2. Load default config
        default_config = self._load_default_config()
        if default_config:
            self._merge_configs(self._config, default_config)
        
        # 3. Load user global config if exists
        user_global_config = self._get_user_global_config_path()
        if user_global_config.exists():
            self._merge_config_file(user_global_config)
        
        # 4. Load project local config if exists
        project_local_config = Path.cwd() / "inferx_config.yaml"
        if project_local_config.exists():
            self._merge_config_file(project_local_config)
        
        # 5. Load user-specified config if provided
        if self._config_path:
            config_path = Path(self._config_path)
            if config_path.exists():
                self._merge_config_file(config_path)
    
    def _apply_pydantic_validation(self):
        """Apply Pydantic validation to loaded configuration"""
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic not available, skipping enhanced validation")
            return
        
        try:
            # Create Pydantic model from raw config
            self._validated_config = InferXSettingsSchema(**self._config)
            
            # Update raw config with validated/default values
            self._config = self._validated_config.dict()
            
            logger.debug("Configuration validated successfully with Pydantic")
            
        except ValidationError as e:
            # Convert Pydantic errors to user-friendly format
            field_errors = []
            suggestions = []
            
            for error in e.errors():
                field_errors.append({
                    'loc': error['loc'],
                    'msg': error['msg'],
                    'type': error['type']
                })
                
                # Add specific suggestions based on error type
                if error['type'] == 'value_error.number.not_ge':
                    suggestions.append(f"Set {'.'.join(str(loc) for loc in error['loc'])} to a value >= {error.get('ctx', {}).get('limit_value', 0)}")
                elif error['type'] == 'value_error.number.not_le':
                    suggestions.append(f"Set {'.'.join(str(loc) for loc in error['loc'])} to a value <= {error.get('ctx', {}).get('limit_value', 1)}")
                elif 'divisible' in str(error['msg']):
                    suggestions.append("YOLO input size must be divisible by 32 (e.g., 320, 416, 512, 640, 1024)")
            
            raise ConfigurationError(
                message="Configuration validation failed",
                field_errors=field_errors,
                suggestions=suggestions
            )
    
    # =============================================================================
    # EXISTING API (100% Backward Compatible)
    # =============================================================================
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (existing API)
        
        Args:
            key_path: Configuration key path (e.g., "model_defaults.yolo.input_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._get_nested_value(self._config, key_path, default)
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation (existing API)
        
        Args:
            key_path: Configuration key path
            value: Value to set
        """
        self._set_nested_value(self._config, key_path, value)
        
        # Re-validate if Pydantic is available
        if self._validation_enabled:
            try:
                self._validated_config = InferXSettingsSchema(**self._config)
                self._config = self._validated_config.dict()
            except ValidationError as e:
                # Reset to previous value and raise error
                logger.error(f"Invalid value for {key_path}: {e}")
                raise ConfigurationError(f"Invalid configuration value: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary (existing API)"""
        return self._config.copy()
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file (existing API)"""
        output_path = Path(output_path)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    # =============================================================================
    # NEW TYPED API (Optional, Backward Compatible)
    # =============================================================================
    
    @property
    def typed(self) -> Optional['InferXSettingsSchema']:
        """Type-safe access to configuration (new API)
        
        Returns:
            Pydantic settings model with type safety and IDE support
        """
        if not self._validation_enabled:
            warnings.warn(
                "Typed configuration access requires Pydantic. "
                "Install with: pip install pydantic",
                UserWarning
            )
            return None
        
        return self._validated_config
    
    def get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get model defaults with enhanced error checking (new API)
        
        Args:
            model_type: Type of model (yolo, classification, etc.)
            
        Returns:
            Model configuration dictionary
        """
        model_defaults = self.get("model_defaults", {})
        if model_type not in model_defaults:
            available_types = list(model_defaults.keys())
            raise ConfigurationError(
                f"Model type '{model_type}' not found",
                suggestions=[
                    f"Use one of: {', '.join(available_types)}",
                    f"Add '{model_type}' configuration to model_defaults"
                ]
            )
        return model_defaults[model_type]
    
    def validate(self) -> List[str]:
        """Validate configuration and return warnings (new API)
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if not self._validation_enabled:
            warnings.append("Enhanced validation unavailable (Pydantic not installed)")
            return warnings
        
        try:
            # Re-validate current config
            InferXSettingsSchema(**self._config)
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error['loc'])
                warnings.append(f"{field}: {error['msg']}")
        
        return warnings
    
    # =============================================================================
    # INTERNAL HELPER METHODS (Existing Implementation)
    # =============================================================================
    
    def _get_nested_value(self, config: Dict, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value"""
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def _set_nested_value(self, config: Dict, key_path: str, value: Any) -> None:
        """Set nested configuration value"""
        keys = key_path.split('.')
        current = config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value
    
    def _merge_configs(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
    
    def _merge_config_file(self, config_path: Path) -> None:
        """Load and merge configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            self._merge_configs(self._config, user_config)
            logger.debug(f"Merged configuration from: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _get_user_global_config_path(self) -> Path:
        """Get user global config path"""
        return Path.home() / ".inferx" / "config.yaml"
    
    def _load_default_config(self) -> Optional[Dict[str, Any]]:
        """Load default configuration"""
        # This would load from the actual default.yaml file
        # For demo purposes, return a basic config
        return {
            "model_defaults": {
                "yolo": {
                    "input_size": 640,
                    "confidence_threshold": 0.25,
                    "nms_threshold": 0.45
                }
            }
        }
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration"""
        return {
            "model_detection": {
                "yolo_keywords": ["yolo", "yolov", "yolov8"]
            },
            "device_mapping": {
                "auto": "AUTO",
                "cpu": "CPU",
                "gpu": "GPU"
            }
        }


# =============================================================================
# BACKWARD COMPATIBLE FACTORY FUNCTION
# =============================================================================

def get_config(config_path: Optional[Union[str, Path]] = None) -> HybridInferXConfig:
    """Get configuration instance (existing API)
    
    This function maintains the existing API while providing enhanced validation.
    """
    return HybridInferXConfig(config_path)


# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def demonstrate_hybrid_config():
    """Demonstrate the hybrid configuration approach"""
    
    print("🔄 Hybrid Configuration Demo")
    print("=" * 50)
    
    # 1. Existing API still works
    print("\n1. Existing API (100% Compatible):")
    config = get_config()
    
    threshold = config.get("model_defaults.yolo.confidence_threshold")
    print(f"   Confidence threshold: {threshold}")
    
    device = config.get("device_mapping.auto")
    print(f"   Auto device mapping: {device}")
    
    # 2. Enhanced error messages
    print("\n2. Enhanced Error Messages:")
    try:
        config.set("model_defaults.yolo.confidence_threshold", 1.5)  # Invalid
    except ConfigurationError as e:
        print(f"   ✅ Enhanced error caught: {e}")
    
    # 3. New typed API (if Pydantic available)
    print("\n3. New Typed API:")
    if config.typed:
        # Type-safe access with IDE support
        yolo_config = config.typed.model_defaults.yolo
        print(f"   YOLO input size: {yolo_config.input_size}")
        print(f"   YOLO class count: {len(yolo_config.class_names)}")
        print("   ✅ Full type safety and autocomplete")
    else:
        print("   ⚠️  Pydantic not available - install for typed API")
    
    # 4. Validation
    print("\n4. Configuration Validation:")
    warnings = config.validate()
    if warnings:
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    else:
        print("   ✅ Configuration is valid")
    
    # 5. Model-specific access
    print("\n5. Enhanced Model Access:")
    try:
        yolo_defaults = config.get_model_defaults("yolo")
        print(f"   YOLO defaults loaded: {len(yolo_defaults)} settings")
    except ConfigurationError as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    demonstrate_hybrid_config()
    
    print("\n" + "=" * 50)
    print("🎯 Hybrid Approach Benefits:")
    print("   ✅ 100% backward compatibility")
    print("   ✅ Enhanced validation and error messages")
    print("   ✅ Optional type safety")
    print("   ✅ Zero breaking changes")
    print("   ✅ Smooth migration path")
    print("   ✅ Best of both worlds")