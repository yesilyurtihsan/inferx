"""
Pydantic Settings Implementation Example for InferX

This demonstrates how InferX configuration could be implemented using Pydantic Settings
with type safety, validation, and environment variable support.
"""

from pydantic import BaseModel, BaseSettings, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any, Literal
from enum import Enum
from pathlib import Path
import yaml
import os


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    MYRIAD = "myriad"
    HDDL = "hddl"
    NPU = "npu"


class RuntimeType(str, Enum):
    AUTO = "auto"
    ONNX = "onnx"
    OPENVINO = "openvino"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OutputFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"


# =============================================================================
# NESTED CONFIGURATION MODELS
# =============================================================================

class YOLOConfig(BaseModel):
    """YOLO model configuration with validation"""
    
    input_size: int = Field(
        default=640,
        gt=0,
        description="Input image size (will be resized to input_size x input_size)"
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
        description="Maximum number of detections to keep"
    )
    class_names: List[str] = Field(
        default_factory=lambda: [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog"
        ],
        description="Class names for YOLO model"
    )
    
    @validator('input_size')
    def validate_input_size(cls, v):
        if v % 32 != 0:
            raise ValueError('Input size must be divisible by 32 for YOLO models')
        return v
    
    @validator('class_names')
    def validate_class_names(cls, v):
        if not v:
            raise ValueError('At least one class name is required')
        if len(v) > 1000:
            raise ValueError('Too many classes (max 1000)')
        return v


class ClassificationConfig(BaseModel):
    """Classification model configuration"""
    
    input_size: List[int] = Field(
        default=[224, 224],
        min_items=2,
        max_items=2,
        description="Input image size [height, width]"
    )
    top_k: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Return top-k predictions"
    )
    normalize: Dict[str, List[float]] = Field(
        default={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        description="Normalization parameters"
    )
    class_names: List[str] = Field(
        default_factory=list,
        description="Class names (empty = auto-detect)"
    )
    
    @validator('input_size')
    def validate_input_size(cls, v):
        if any(size <= 0 for size in v):
            raise ValueError('Input dimensions must be positive')
        return v
    
    @validator('normalize')
    def validate_normalize(cls, v):
        if 'mean' in v and len(v['mean']) != 3:
            raise ValueError('Mean must have 3 values (RGB)')
        if 'std' in v and len(v['std']) != 3:
            raise ValueError('Std must have 3 values (RGB)')
        return v


class ModelDefaults(BaseModel):
    """Model-specific default configurations"""
    
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class ModelDetection(BaseModel):
    """Model type detection keywords"""
    
    yolo_keywords: List[str] = Field(
        default=["yolo", "yolov", "yolov8", "yolov11", "ultralytics"],
        description="Keywords for YOLO model detection"
    )
    classification_keywords: List[str] = Field(
        default=["resnet", "efficientnet", "mobilenet", "vgg", "classifier"],
        description="Keywords for classification model detection"
    )
    anomalib_keywords: List[str] = Field(
        default=["anomalib", "padim", "patchcore", "anomaly"],
        description="Keywords for anomaly detection model detection"
    )


class DeviceMapping(BaseModel):
    """Device name mappings"""
    
    auto: str = Field(default="AUTO", description="Auto device selection")
    cpu: str = Field(default="CPU", description="CPU device")
    gpu: str = Field(default="GPU", description="GPU device")
    cuda: str = Field(default="GPU", description="NVIDIA CUDA GPU")
    opencl: str = Field(default="GPU", description="OpenCL GPU")
    myriad: str = Field(default="MYRIAD", description="Intel Movidius Myriad VPU")
    hddl: str = Field(default="HDDL", description="Intel HDDL")
    npu: str = Field(default="NPU", description="Neural Processing Unit")


class SessionOptions(BaseModel):
    """ONNX Runtime session options"""
    
    graph_optimization_level: Literal["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"] = Field(
        default="ORT_ENABLE_ALL",
        description="Graph optimization level"
    )
    inter_op_num_threads: int = Field(
        default=0,
        ge=0,
        description="Inter-op thread count (0 = auto)"
    )
    intra_op_num_threads: int = Field(
        default=0,
        ge=0,
        description="Intra-op thread count (0 = auto)"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )


class RuntimePreferences(BaseModel):
    """Runtime-specific preferences"""
    
    onnx: Dict[str, Any] = Field(
        default_factory=lambda: {
            "providers": {
                "gpu": ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"],
                "cpu": ["CPUExecutionProvider"],
                "auto": ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
            },
            "session_options": SessionOptions().dict()
        }
    )
    openvino: Dict[str, Any] = Field(
        default_factory=lambda: {
            "performance_hints": {
                "throughput": "THROUGHPUT",
                "latency": "LATENCY"
            }
        }
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""
    
    level: LogLevel = Field(default=LogLevel.INFO, description="Default logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    categories: Dict[str, bool] = Field(
        default_factory=lambda: {
            "model_loading": True,
            "inference_timing": True,
            "preprocessing": False,
            "postprocessing": False,
            "device_info": True
        },
        description="Enable/disable specific logging categories"
    )
    file: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "path": "inferx.log",
            "max_size_mb": 10,
            "backup_count": 3
        },
        description="File logging configuration"
    )


class OutputConfig(BaseModel):
    """Output configuration"""
    
    default_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Default output format"
    )
    float_precision: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Decimal places for floating point numbers"
    )
    include_metadata: Dict[str, bool] = Field(
        default_factory=lambda: {
            "model_info": True,
            "timing_info": True,
            "device_info": True,
            "config_used": False
        },
        description="Include metadata in output"
    )


class AdvancedConfig(BaseModel):
    """Advanced configuration options"""
    
    model_cache: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "cache_dir": "~/.inferx/cache",
            "max_cache_size_gb": 5
        }
    )
    memory: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_memory_pool": True,
            "max_memory_pool_size_mb": 1024
        }
    )
    experimental: Dict[str, bool] = Field(
        default_factory=lambda: {
            "enable_dynamic_batching": False,
            "enable_model_optimization": False,
            "enable_mixed_precision": False
        }
    )


# =============================================================================
# MAIN SETTINGS CLASS
# =============================================================================

class InferXSettings(BaseSettings):
    """
    InferX Configuration using Pydantic Settings
    
    This provides type-safe configuration with automatic validation,
    environment variable support, and comprehensive error messages.
    """
    
    # Core settings
    default_device: DeviceType = Field(
        default=DeviceType.AUTO,
        env="INFERX_DEVICE",
        description="Default inference device"
    )
    default_runtime: RuntimeType = Field(
        default=RuntimeType.AUTO,
        env="INFERX_RUNTIME",
        description="Default runtime engine"
    )
    
    # Configuration sections
    model_detection: ModelDetection = Field(default_factory=ModelDetection)
    device_mapping: DeviceMapping = Field(default_factory=DeviceMapping)
    model_defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    runtime_preferences: RuntimePreferences = Field(default_factory=RuntimePreferences)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    
    class Config:
        env_prefix = "INFERX_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        allow_population_by_field_name = True
        use_enum_values = True
        validate_assignment = True  # Validate on attribute assignment
        
        # JSON Schema configuration
        schema_extra = {
            "title": "InferX Configuration",
            "description": "Complete configuration schema for InferX ML inference toolkit"
        }


# =============================================================================
# YAML CONFIGURATION SOURCE
# =============================================================================

class YamlConfigSettingsSource:
    """Custom settings source for YAML files with hierarchy support"""
    
    def __init__(self, settings_cls):
        self.settings_cls = settings_cls
        self.config_paths = [
            Path.home() / ".inferx" / "config.yaml",  # Global user config
            Path.cwd() / "inferx_config.yaml",       # Project local config
            Path(__file__).parent / "configs" / "default.yaml"  # Default config
        ]
    
    def __call__(self) -> Dict[str, Any]:
        config = {}
        
        # Load in reverse priority order (lowest to highest)
        for config_path in reversed(self.config_paths):
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f) or {}
                    self._merge_config(config, file_config)
                except Exception as e:
                    print(f"Warning: Failed to load config from {config_path}: {e}")
        
        return config
    
    def _merge_config(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target config"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def demonstrate_pydantic_config():
    """Demonstrate the benefits of Pydantic Settings"""
    
    print("🚀 InferX Pydantic Settings Demo")
    print("=" * 50)
    
    # 1. Basic usage with defaults
    print("\n1. Basic Configuration:")
    settings = InferXSettings()
    print(f"   Default device: {settings.default_device}")
    print(f"   YOLO confidence: {settings.model_defaults.yolo.confidence_threshold}")
    print(f"   Log level: {settings.logging.level}")
    
    # 2. Type-safe access with IDE support
    print("\n2. Type-Safe Access:")
    # This would give autocomplete in IDE
    yolo_config = settings.model_defaults.yolo
    print(f"   Input size: {yolo_config.input_size}")
    print(f"   Class count: {len(yolo_config.class_names)}")
    
    # 3. Environment variable override
    print("\n3. Environment Variable Support:")
    os.environ["INFERX_DEVICE"] = "gpu"
    os.environ["INFERX_RUNTIME"] = "onnx"
    settings_with_env = InferXSettings()
    print(f"   Device from ENV: {settings_with_env.default_device}")
    print(f"   Runtime from ENV: {settings_with_env.default_runtime}")
    
    # 4. Validation examples
    print("\n4. Automatic Validation:")
    try:
        # This will fail validation
        invalid_settings = InferXSettings(
            model_defaults={
                "yolo": {
                    "confidence_threshold": 1.5,  # Invalid: > 1.0
                    "input_size": 100  # Invalid: not divisible by 32
                }
            }
        )
    except Exception as e:
        print(f"   ✅ Validation caught errors: {type(e).__name__}")
        # In real usage, you'd get detailed error messages
    
    # 5. JSON Schema generation
    print("\n5. JSON Schema Generation:")
    schema = InferXSettings.schema()
    print(f"   Schema has {len(schema['properties'])} top-level properties")
    print(f"   Title: {schema.get('title', 'N/A')}")
    
    # 6. Export to dict/JSON
    print("\n6. Serialization:")
    config_dict = settings.dict()
    print(f"   Config as dict has {len(config_dict)} sections")
    
    # Clean up environment
    os.environ.pop("INFERX_DEVICE", None)
    os.environ.pop("INFERX_RUNTIME", None)


def compare_with_current_system():
    """Compare Pydantic approach with current system"""
    
    print("\n📊 Comparison with Current System")
    print("=" * 50)
    
    # Current system simulation
    current_config = {
        "model_defaults": {
            "yolo": {
                "confidence_threshold": 0.25,
                "input_size": 640
            }
        }
    }
    
    # Pydantic system
    pydantic_config = InferXSettings()
    
    print("\nAccess Patterns:")
    
    # Current: String-based, no validation
    print("Current System:")
    print(f"   config.get('model_defaults.yolo.confidence_threshold')")
    print("   ❌ No type checking")
    print("   ❌ No autocomplete")
    print("   ❌ Runtime errors only")
    
    # Pydantic: Type-safe, validated
    print("\nPydantic System:")
    print(f"   settings.model_defaults.yolo.confidence_threshold")
    print("   ✅ Type checking")
    print("   ✅ IDE autocomplete")
    print("   ✅ Validation at load time")
    print("   ✅ Detailed error messages")


if __name__ == "__main__":
    demonstrate_pydantic_config()
    compare_with_current_system()
    
    print("\n" + "=" * 50)
    print("🎯 Key Benefits of Pydantic Settings:")
    print("   • Type safety and IDE support")
    print("   • Automatic validation with custom rules")
    print("   • Environment variable integration")
    print("   • JSON Schema generation")
    print("   • Better error messages")
    print("   • Maintainable and self-documenting")