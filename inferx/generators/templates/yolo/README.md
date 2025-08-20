# YOLO Inference Template

Auto-generated YOLO inference project using InferX.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -e .

# Run inference
python -m src.inferencer path/to/image.jpg
```

## 📁 Project Structure

- `src/inferencer.py` - YOLO inference logic (inherits from InferX YOLOInferencer)
- `config.yaml` - Configuration file
- `models/yolo_model.onnx` - Place your YOLO model here

## 🎯 Usage

1. Place your YOLO model in `models/yolo_model.onnx`
2. Update `config.yaml` with your settings
3. Run inference: `python -m src.inferencer input_image.jpg`

## 🌐 API Server

To add API server:
```bash
inferx api
```

## 🐳 Docker Container

To add Docker container:
```bash
inferx docker
```