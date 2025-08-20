# InferX Examples

This directory contains practical examples demonstrating InferX features with the new Pydantic Settings system.

## 🚀 Quick Start Examples

### `basic_usage.py`
Comprehensive examples showing:
- Basic ONNX and OpenVINO inference
- YOLO object detection patterns
- Batch processing workflows
- Configuration usage with settings.py
- Device optimization strategies

**Run:** `uv run python examples/basic_usage.py`

### `settings_examples.py` ✨ **NEW**
Modern Pydantic Settings system examples:
- Type-safe configuration loading
- Environment variable integration
- YOLO template validation
- Configuration hierarchy demonstration
- Type safety benefits

**Run:** `uv run python examples/settings_examples.py`

### `yolo_detection_demo.py`
Interactive YOLO detection demonstration:
- ONNX vs OpenVINO runtime comparison
- Visual results with bounding boxes
- Performance timing analysis
- Model type auto-detection

**Run:** `uv run python examples/yolo_detection_demo.py`

## ⚙️ Configuration Examples

### `example_config.yaml`
Complete configuration file example showing:
- Custom model detection patterns
- Device mapping preferences
- Model-specific defaults (YOLO, classification)
- Logging configuration
- Runtime preferences

**Usage:** 
```bash
# Use with InferenceEngine
from inferx import InferenceEngine
engine = InferenceEngine("model.onnx", config_path="examples/example_config.yaml")

# Or with CLI
inferx run model.onnx image.jpg --config examples/example_config.yaml
```

## 🎯 Migration from Old Config System

The examples have been updated to use the new **settings.py** system:

### ✅ **What's New:**
- **Type Safety** - Pydantic validation prevents configuration errors
- **Environment Variables** - Automatic loading with `INFERX_` prefix
- **Template Validation** - Generated templates are automatically validated
- **Better Error Messages** - Clear, actionable validation feedback
- **IDE Support** - Auto-completion and type checking

### 🔄 **Migration Notes:**
- **Old:** `from inferx.config import InferXConfig`
- **New:** `from inferx.settings import get_inferx_settings`
- **Compatibility:** Existing API works without changes
- **Enhanced:** Added validation and type safety

## 📚 Related Documentation

- **CONFIG_GUIDE.md** - Complete configuration guide
- **USAGE.md** - Usage patterns and best practices
- **TODO.md** - Development roadmap and completed features

## 🛠️ Development Examples

For development and testing examples, see:
- `scripts/examples/` - Additional demo files
- `scripts/tests/` - Test utilities
- `scripts/old_examples/` - Archived legacy examples

## 🎯 Usage Patterns

### 1. **Basic Inference**
```python
from inferx import InferenceEngine
engine = InferenceEngine("model.onnx")
result = engine.predict("image.jpg")
```

### 2. **With Configuration**
```python
engine = InferenceEngine("model.xml", config_path="custom.yaml")
result = engine.predict("image.jpg")
```

### 3. **Settings Access**
```python
from inferx.settings import get_inferx_settings
settings = get_inferx_settings()
print(f"YOLO input size: {settings.yolo_input_size}")
```

### 4. **Template Validation**
```python
from inferx.settings import validate_yolo_template_config
config = validate_yolo_template_config("template_config.yaml")
```

## 🚀 Next Steps

1. **Run Examples:** Start with `basic_usage.py` for overview
2. **Try Settings:** Explore `settings_examples.py` for type-safe config  
3. **Create Templates:** Use `inferx template yolo --name my-project`
4. **Configure:** Customize `example_config.yaml` for your needs
5. **Deploy:** Use validated configurations in production