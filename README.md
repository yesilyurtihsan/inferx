# InferX - 4-in-1 ML Inference Toolkit (In Development Stage)

> **"One tool, four ways to deploy your model: Library, CLI, Template, or Full Stack"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-supported-green.svg)](https://onnx.ai/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-supported-blue.svg)](https://docs.openvino.ai/)

## 🎯 Philosophy

**4 ways to use InferX - Choose what fits your needs:**

1. **📦 Library** - Import and use directly in your Python code
2. **⚡ CLI** - Run models directly from command line
3. **🏗️ Template Generator** - Generate ready-to-use project templates
4. **🚢 Full Stack** - Generate API servers and Docker containers

Unlike heavy frameworks, InferX gives you clean, minimal dependency code that you own completely. No framework lock-in, no heavy dependencies.

## 🎯 4 Usage Patterns

### 📦 **1. Library Usage (Import in your code)**
```python
from inferx import InferenceEngine

# Use directly in your Python applications
engine = InferenceEngine("model.onnx", device="gpu")
result = engine.predict("image.jpg")

# Batch processing
results = engine.predict_batch(["img1.jpg", "img2.jpg"])
```

### ⚡ **2. CLI Usage (Command line)**
```bash
# Run inference directly from command line
inferx run model.onnx image.jpg --device gpu

# Batch processing with output
inferx run model.xml images/ --output results.json --runtime openvino

# Device optimization
inferx run model.xml image.jpg --device myriad --runtime openvino
```

### **3. Template Generation (Project scaffolding)**
```bash
# Generate a complete standalone project
inferx template yolo --name my-detector
cd my-detector

# Generate with model file copy
inferx template yolo --name my-detector --model-path /path/to/my/model.onnx

# Generate OpenVINO template
inferx template yolo_openvino --name my-detector --model-path /path/to/my/model.xml

# Project structure:
# ├── pyproject.toml      # Minimal dependencies
# ├── src/inferencer.py   # Your inference code (inherits from InferX YOLOInferencer)
# ├── src/base.py         # Base inferencer class
# ├── config.yaml         # Configuration
# └── models/yolo_model.onnx # Place your model here (or .xml/.bin for OpenVINO)
```

### 🚢 **4. Full Stack Generation (API + Docker)**
```bash
# Start with template
inferx template --model-type yolo --name my-detector
cd my-detector

# Add API server
inferx api
# ├── src/server.py       # Generated FastAPI app
# └── requirements-api.txt # +FastAPI only

# Add Docker deployment
inferx docker
# ├── Dockerfile         # Optimized container
# └── docker-compose.yml # Ready to deploy

# Deploy with Docker
docker-compose up
```

## 🆚 vs Heavy Frameworks

| Framework | Dependencies | Container Size | Approach |
|-----------|-------------|----------------|-----------|
| **InferX** | ONNX Runtime only (~50MB) | ~75MB | Code generation |
| BentoML | Full framework stack | ~900MB | Framework-based |
| TorchServe | PyTorch + dependencies | ~1.2GB | Framework-based |
| TF Serving | TensorFlow | ~800MB | Framework-based |

## 🏗️ Generated Project Structure

When you run `inferx template yolo --name my-detector`:

```
my-detector/                    # Your standalone project
├── pyproject.toml             # UV project with minimal deps
├── src/
│   ├── __init__.py
│   ├── inferencer.py          # YOLO inference implementation (inherits from InferX YOLOInferencer)
│   └── base.py                # Base inferencer class
├── models/
│   └── yolo_model.onnx        # Place your YOLO model here (or .xml/.bin for OpenVINO)
├── config.yaml                # Inference configuration
├── README.md                  # Usage instructions
└── .gitignore                 # Standard Python gitignore
```

When you run `inferx template yolo_openvino --name my-detector`:

```
my-detector/                    # Your standalone project
├── pyproject.toml             # UV project with minimal deps
├── src/
│   ├── __init__.py
│   ├── inferencer.py          # YOLO OpenVINO inference implementation (inherits from InferX YOLOOpenVINOInferencer)
│   └── base.py                # Base inferencer class
├── models/
│   ├── yolo_model.xml         # Place your YOLO OpenVINO model .xml file here
│   └── yolo_model.bin         # Place your YOLO OpenVINO model .bin file here
├── config.yaml                # Inference configuration
├── README.md                  # Usage instructions
└── .gitignore                 # Standard Python gitignore
```

After `inferx api`:
```
my-detector/
├── src/
│   ├── inferencer.py          # Existing
│   ├── base.py                # Existing
│   └── server.py              # Generated FastAPI app
└── requirements-api.txt       # +FastAPI only
```

After `inferx docker`:
```
my-detector/
├── Dockerfile                 # Multi-stage optimized
├── docker-compose.yml         # Ready to deploy
└── .dockerignore             # Build optimization
```

## 📦 Generated Dependencies

### **Template Project** (pyproject.toml)
```toml
[project]
name = "my-detector"
version = "0.1.0"
dependencies = [
    "onnxruntime>=1.16.0",           # ~50MB
    "numpy>=1.24.0",                 # Array operations
    "opencv-python-headless>=4.8.0", # Image processing
]

[project.optional-dependencies]
api = ["fastapi>=0.104.0", "uvicorn>=0.24.0"]  # Only when using API
gpu = ["onnxruntime-gpu>=1.16.0"]               # Only for GPU inference
openvino = ["openvino>=2023.3.0"]               # Intel optimization
```

### **Why Minimal Dependencies?**
- **Production safety**: Fewer dependencies = fewer security vulnerabilities
- **Faster deployment**: Smaller containers, faster startup
- **Cost efficiency**: Less compute resources needed
- **Maintenance**: Easier to update and maintain

## 🚀 Quick Start

### 📥 **Installation**
```bash
# Install from PyPI (when available)
pip install inferx

# Or install from source
git clone https://github.com/yourusername/inferx.git
cd inferx
pip install -e .
```

### 🎯 **Four Usage Patterns**

#### **1. Library Usage (Import in your code)**
```python
from inferx import InferenceEngine

# Use directly in your Python applications
engine = InferenceEngine("model.onnx", device="gpu")
result = engine.predict("image.jpg")
print(result)
```

#### **2. CLI Usage (Command line)**
```bash
# Run inference directly from command line
inferx run model.onnx image.jpg --device gpu

# Batch processing
inferx run model.xml images/ --output results.json --runtime openvino
```

#### **3. Template Generation**
```bash
# Create YOLO detection project
inferx template yolo --name my-detector
cd my-detector

# Project structure:
# ├── src/inference.py    # YOLO inference code
# ├── model.onnx         # Place your model here
# └── pyproject.toml     # Minimal dependencies

# Test inference
uv run python -m src.inference test_image.jpg
```

#### **4. Full Stack Deployment**
```bash
# Start with template
inferx template yolo --name my-detector
cd my-detector

# Add API server
inferx api

# Add Docker deployment
inferx docker

# Start server
uv run python -m src.server

# Or deploy with Docker
docker build -t my-detector:v1 .
docker run -p 8080:8080 my-detector:v1
```

### 🎨 **Available Templates** (Template Generation Pattern)
```bash
# Object detection
inferx template yolo --name my-detector

# Anomaly detection  
inferx template anomaly --name quality-checker

# Image classification
inferx template classification --name image-classifier

# Custom ONNX model
inferx template custom --name my-model
```

## 🚧 Development Status

### ✅ **Currently Available**
- ✅ Basic inference engines (ONNX + OpenVINO)
- ✅ Configuration system
- ✅ CLI structure
- ✅ Testing framework
- ✅ Project examples
- ✅ **Library usage pattern**
- ✅ **CLI usage pattern**

### 🚧 **In Development**
- 🚧 Template generation (`inferx template`)
- 🚧 API generation (`inferx api`) 
- 🚧 Docker generation (`inferx docker`)
- 🚧 Project templates (YOLO, Anomaly, Classification)
- 🚧 **Template + Full Stack usage patterns**

### 📋 **TODO**
See [TODO.md](TODO.md) for detailed development tasks and progress.

## ⚙️ Configuration (Used by All 4 Patterns)

Generated projects include a `config.yaml`:

```yaml
# Model settings
model:
  path: "model.onnx"
  type: "yolo"
  
# Inference settings  
inference:
  device: "auto"        # auto, cpu, gpu
  batch_size: 1
  confidence_threshold: 0.25
  
# Input preprocessing
preprocessing:
  input_size: [640, 640]
  normalize: true
  format: "RGB"
```

## 🎯 Why InferX?

### **4 Flexible Usage Patterns**
```python
# 1. Library - Import and use in your code
from inferx import InferenceEngine
engine = InferenceEngine("model.onnx")
result = engine.predict("image.jpg")

# 2. CLI - Run from command line
# inferx run model.onnx image.jpg

# 3. Template - Generate project structure
# inferx template yolo --name my-detector

# 4. Full Stack - Generate API + Docker
# inferx template yolo --name my-detector
# cd my-detector
# inferx api
# inferx docker
```

### **Problem with Heavy Frameworks**
```python
# BentoML - Framework dependency
import bentoml
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 20},
)
class MyService:
    # Heavy framework, complex setup
```

### **InferX Solution - Clean Code**
```python
# Generated inference.py - No framework dependency
import onnxruntime as ort
import numpy as np

class YOLOInferencer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
    
    def predict(self, image_path: str):
        # Your clean, minimal code
        return results
```

### **Benefits**
- ✅ **You own the code** - No framework lock-in
- ✅ **Minimal dependencies** - Only what you need
- ✅ **Easy to modify** - Standard Python code
- ✅ **Production ready** - UV project structure
- ✅ **Fast deployment** - Small containers
- ✅ **4 usage patterns** - Library, CLI, Template, or Full Stack

## 🤝 Contributing

### ✅ **Current Status**
InferX core inference engines (Library and CLI) are production-ready. Template generation features are in active development.

### 📋 **How to Help**
1. **Test current inference engines** with your ONNX/OpenVINO models
2. **Use the Library and CLI patterns** in your projects and report issues
3. **Suggest template improvements** for different model types  
4. **Contribute code** for template generation features

### 🔧 **Development Setup**
```bash
git clone https://github.com/yourusername/inferx.git
cd inferx
pip install -e .[dev]

# Run tests
python test_runner.py

# See development tasks
cat TODO.md
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**InferX** - Minimal dependency ML inference templates. 🚀

*Give us your model. Get template, API, or Docker container.*
