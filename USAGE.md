# InferX Usage Guide

## 🚀 Current State: Functional CLI for ONNX Models

InferX is now **fully functional** for running inference with ONNX models! This guide covers all available features in the current implementation.

## 📦 Installation

### Using UV (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install InferX
git clone <repository-url>
cd inferx

# Install in development mode with all dependencies
uv sync --all-extras

# Verify installation
uv run inferx --version
```

### Using pip (Alternative)

```bash
# From project directory
pip install -e .

# With optional dependencies
pip install -e .[gpu,serve,dev]

# Verify installation
inferx --version
```

### Quick Start with UV

```bash
# All commands use 'uv run' prefix
uv run inferx run model.onnx image.jpg
```

## 🎯 Core Features Available

### ✅ What Works Now:
- **ONNX Runtime inference** - Load and run any ONNX model
- **YOLO object detection** - Run YOLO models with automatic bounding box detection
- **Single image processing** - Process individual images
- **Batch processing** - Process entire folders of images
- **Configuration files** - YAML-based configuration
- **Multiple output formats** - JSON/YAML results export
- **Performance tracking** - Detailed timing information
- **Device selection** - CPU/GPU/auto detection
- **Verbose logging** - Debug and troubleshooting support

### ⚠️ What's Not Yet Implemented:
- Model-specific inferencers (Anomalib)
- FastAPI server (`inferx serve`)
- Docker generation (`inferx docker`)
- Project templates (`inferx init`)
- OpenVINO runtime support

## 🛠️ CLI Commands

### Basic Command Structure
```bash
inferx [OPTIONS] COMMAND [ARGS]...
```

### Global Options
- `--verbose, -v`: Enable detailed logging
- `--version`: Show version and exit
- `--help`: Show help message

## 📊 Running Inference

### 1. Single Image Inference

**Basic usage (UV):**
```bash
uv run inferx run model.onnx image.jpg
```

**YOLO object detection (UV):**
```bash
# YOLO models are automatically detected by filename (containing 'yolo', 'yolov', 'yolov8')
uv run inferx run yolov8n.onnx image.jpg

# Or explicitly specify model type
uv run inferx run custom_model.onnx image.jpg --config config.yaml
```

**With options (UV):**
```bash
uv run inferx run model.onnx image.jpg --device gpu --verbose
```

**Save results (UV):**
```bash
uv run inferx run model.onnx image.jpg --output results.json
```

**Alternative (pip install):**
```bash
inferx run model.onnx image.jpg
```

**Example output (Generic ONNX):**
```
🚀 Starting inference...
   Model: model.onnx
   Input: image.jpg
   Device: auto, Runtime: auto
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running single image inference...
✅ Inference completed in 0.032s

📊 Inference Summary:
   Model type: onnx_generic
   Outputs: 1
   Inference time: 0.032s
```

**Example output (YOLO):**
```
🚀 Starting inference...
   Model: yolov8n.onnx
   Input: image.jpg
   Device: auto, Runtime: auto
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running single image inference...
✅ Inference completed in 0.032s

📊 Inference Summary:
   Model type: yolo
   Detections: 3
   Inference time: 0.032s
```

### 2. Batch Processing

**Process entire folder (UV):**
```bash
uv run inferx run model.onnx photos/
```

**With progress tracking (UV):**
```bash
uv run inferx run model.onnx photos/ --output batch_results.json --verbose
```

**Alternative (pip install):**
```bash
inferx run model.onnx photos/
```

**Example output:**
```
🚀 Starting inference...
   Model: model.onnx
   Input: photos/
   Device: auto, Runtime: auto
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running batch inference on 25 images...
Processing images  [####################################]  100%
✅ Batch processing completed!
   Processed: 25/25 images
   Total time: 0.850s
   Average: 0.034s per image
```

### 3. Configuration Files

**Create config.yaml:**
```yaml
# Model configuration
model:
  type: "onnx"
  
# Runtime settings
runtime:
  engine: "auto"
  device: "auto"
  
# Preprocessing pipeline
preprocessing:
  target_size: [640, 640]
  normalize: true
  color_format: "RGB"
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# Output settings
output:
  format: "json"
```

**Use configuration (UV):**
```bash
uv run inferx run model.onnx image.jpg --config config.yaml
```

**Alternative (pip install):**
```bash
inferx run model.onnx image.jpg --config config.yaml
```

## 🎛️ Available Options

### Device Selection (UV)
```bash
# Automatic device selection (default)
uv run inferx run model.onnx image.jpg --device auto

# Force CPU usage
uv run inferx run model.onnx image.jpg --device cpu

# Force GPU usage (if available)
uv run inferx run model.onnx image.jpg --device gpu
```

### Runtime Selection (UV)
```bash
# Automatic runtime selection (default)
uv run inferx run model.onnx image.jpg --runtime auto

# Force ONNX Runtime
uv run inferx run model.onnx image.jpg --runtime onnx

# OpenVINO (not yet implemented)
uv run inferx run model.onnx image.jpg --runtime openvino
```

### Output Formats (UV)
```bash
# JSON output (default)
uv run inferx run model.onnx image.jpg --output results.json --format json

# YAML output
uv run inferx run model.onnx image.jpg --output results.yaml --format yaml
```

### Verbose Mode (UV)
```bash
# Enable detailed logging
uv run inferx run model.onnx image.jpg --verbose

# See full results in terminal
uv run inferx run model.onnx image.jpg -v
```

## 📄 Output Format

### Single Image Results (Generic ONNX)
```json
{
  "raw_outputs": [...],
  "output_shapes": [[1, 1000]],
  "num_outputs": 1,
  "model_type": "onnx_generic",
  "output_0_shape": [1, 1000],
  "output_0_dtype": "float32",
  "output_0_min": -4.2,
  "output_0_max": 8.1,
  "output_0_mean": 0.05,
  "timing": {
    "model_load_time": 0.245,
    "inference_time": 0.032,
    "total_time": 0.277
  }
}
```

### Single Image Results (YOLO)
```json
{
  "detections": [
    {
      "bbox": [100.5, 150.2, 200.0, 300.0],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "person"
    },
    {
      "bbox": [300.1, 200.5, 150.0, 250.0],
      "confidence": 0.87,
      "class_id": 1,
      "class_name": "bicycle"
    }
  ],
  "num_detections": 2,
  "model_type": "yolo",
  "timing": {
    "model_load_time": 0.245,
    "inference_time": 0.032,
    "total_time": 0.277
  }
}
```

### Batch Processing Results
```json
{
  "batch_summary": {
    "total_images": 25,
    "successful": 25,
    "failed": 0,
    "total_inference_time": 0.850,
    "average_inference_time": 0.034,
    "model_load_time": 0.245
  },
  "results": [
    {
      "file_path": "/path/to/image1.jpg",
      "inference_time": 0.032,
      "raw_outputs": [...],
      "model_type": "onnx_generic"
    },
    ...
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

**1. Model loading fails:**
```bash
❌ Failed to load ONNX model: No such file or directory
```
- Check if model file exists and path is correct
- Ensure model is in ONNX format (.onnx extension)

**2. Image loading fails:**
```bash
❌ Could not load image: /path/to/image.jpg
```
- Verify image file exists and is readable
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp

**3. GPU not available:**
```bash
GPU requested but no GPU providers available, falling back to CPU
```
- Install ONNX Runtime GPU: `pip install onnxruntime-gpu`
- Check CUDA installation and compatibility

### Debug Mode

**Enable verbose logging (UV):**
```bash
uv run inferx run model.onnx image.jpg --verbose
```

**Alternative (pip install):**
```bash
inferx run model.onnx image.jpg --verbose
```

This will show:
- Detailed model loading information
- Provider selection process
- Input/output shapes and types
- Full error stack traces
- Complete results JSON

## 📊 Performance Tips

### 1. Device Optimization (UV)
```bash
# Use GPU for better performance (if available)
uv run inferx run model.onnx image.jpg --device gpu

# Use CPU with specific configuration
uv run inferx run model.onnx image.jpg --device cpu --config cpu_optimized.yaml
```

### 2. Batch Processing (UV)
```bash
# Process multiple images efficiently
uv run inferx run model.onnx photos/ --batch-size 8
```

### 3. Configuration Tuning
```yaml
# config.yaml - Performance optimized
runtime:
  device: "gpu"
  
session_options:
  graph_optimization_level: "ORT_ENABLE_ALL"
  inter_op_num_threads: 4
  intra_op_num_threads: 4
```

## 🔄 Model Requirements

### Supported Model Formats
- **ONNX models** (.onnx files)
- Any model exported from PyTorch, TensorFlow, etc.

### Input Requirements  
- **Image inputs**: Any standard image format
- **Preprocessing**: Automatic resizing, normalization, color format conversion
- **Batch dimension**: Automatically added if needed

### Output Processing
- **Raw outputs**: Numerical arrays from model
- **Basic statistics**: Min, max, mean values
- **Shape information**: Input/output tensor shapes
- **Timing data**: Performance metrics

## 🚀 Example Workflows

### 1. Quick Model Testing (UV)
```bash
# Test a model quickly
uv run inferx run my_model.onnx test_image.jpg --verbose
```

### 2. Batch Evaluation (UV)
```bash
# Process validation dataset
uv run inferx run model.onnx validation_images/ --output validation_results.json
```

### 3. Performance Benchmarking (UV)
```bash
# Benchmark model performance
uv run inferx run model.onnx test_image.jpg --device cpu --verbose
uv run inferx run model.onnx test_image.jpg --device gpu --verbose
```

### 4. Custom Configuration (UV)
```bash
# Use custom preprocessing
uv run inferx run model.onnx image.jpg --config custom_preprocess.yaml --output results.json
```

---

## 🎯 What's Coming Next

**Phase 2 - Model-Specific Support:**
- Anomalib anomaly detection with heatmaps
- Classification models with class predictions

**Phase 3 - Advanced Features:**
- FastAPI server: `inferx serve model.onnx --port 8080`
- OpenVINO runtime support for Intel optimization
- WebUI for easy model testing

**Phase 4 - Production Deployment:**
- Docker container generation: `inferx docker model.onnx --tag mymodel:v1`
- Project templates: `inferx init --template yolo`
- Performance optimizations and monitoring

---

*InferX v1.0 - Making ML inference dead simple! 🚀*