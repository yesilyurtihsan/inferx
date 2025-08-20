# InferX Test Data

This directory contains test images and data for InferX examples and testing.

## 📁 Directory Structure

```
data/
├── test_images/           # Test images for YOLO detection
│   ├── street_scene.jpg   # Urban street scene with vehicles
│   ├── person_bicycle.jpg # Person with bicycle
│   └── cars_traffic.jpg   # Traffic scene with multiple cars
└── README.md
```

## 🖼️ Test Images

### `street_scene.jpg`
- **Content**: Urban street scene with cars, traffic
- **Size**: ~800x600 pixels
- **Use Case**: Multi-object detection testing
- **Expected YOLO Detections**: cars, trucks, traffic lights, persons

### `person_bicycle.jpg` 
- **Content**: Person with bicycle
- **Size**: ~800x600 pixels  
- **Use Case**: Person and bicycle detection
- **Expected YOLO Detections**: person, bicycle

### `cars_traffic.jpg`
- **Content**: Traffic scene with multiple vehicles
- **Size**: ~800x600 pixels
- **Use Case**: Vehicle detection in traffic
- **Expected YOLO Detections**: cars, trucks, buses

## 🚀 Usage Examples

### 1. Basic Inference
```bash
# Test single image
inferx run model.onnx data/test_images/street_scene.jpg

# Test all images
inferx run model.onnx data/test_images/ --output results.json
```

### 2. Python API
```python
from inferx import InferenceEngine
engine = InferenceEngine("yolo_model.onnx")

# Single image
result = engine.predict("data/test_images/street_scene.jpg")

# Batch processing
results = engine.predict_batch([
    "data/test_images/street_scene.jpg",
    "data/test_images/person_bicycle.jpg", 
    "data/test_images/cars_traffic.jpg"
])
```

### 3. Example Scripts
```bash
# Run basic usage examples with real images
uv run python examples/basic_usage.py

# YOLO detection demo with real images  
uv run python examples/yolo_detection_demo.py --image data/test_images/street_scene.jpg

# Settings examples
uv run python examples/settings_examples.py
```

## 🎯 Expected Results

### YOLO Detection Classes (COCO)
The test images contain objects from common YOLO/COCO classes:
- **person** - People in the scenes
- **bicycle** - Bicycles
- **car** - Cars and vehicles
- **truck** - Trucks and large vehicles
- **bus** - Buses
- **traffic light** - Traffic lights
- **stop sign** - Stop signs

### Performance Benchmarks
Use these images to test:
- **Inference Speed** - Measure ms per image
- **Detection Accuracy** - Compare with ground truth
- **Device Performance** - CPU vs GPU comparison
- **Runtime Comparison** - ONNX vs OpenVINO

## 📊 Image Properties

| Image | Size (KB) | Dimensions | Objects | Complexity |
|-------|-----------|------------|---------|------------|
| street_scene.jpg | ~137KB | 800x600 | Multiple cars, people | High |
| person_bicycle.jpg | ~67KB | 800x600 | 1 person, 1 bicycle | Medium |
| cars_traffic.jpg | ~137KB | 800x600 | Multiple vehicles | High |

## 🔄 Adding More Test Images

To add more test images:

```bash
# Download from Unsplash (royalty-free)
cd data/test_images
curl -o my_image.jpg "https://images.unsplash.com/photo-ID?w=800&h=600"

# Or add your own images
cp /path/to/your/image.jpg data/test_images/
```

## 📝 Testing Checklist

- [ ] **Detection Accuracy** - Objects correctly detected?
- [ ] **Confidence Scores** - Reasonable confidence values?
- [ ] **Bounding Boxes** - Accurate object localization?
- [ ] **Performance** - Acceptable inference speed?
- [ ] **Memory Usage** - No memory leaks in batch processing?

## 🎨 Image Sources

All test images are from Unsplash (royalty-free):
- High quality, diverse content
- Suitable for object detection testing
- No licensing restrictions for development/testing

## 🚀 Integration with Examples

These images are automatically used by:
- `examples/basic_usage.py` - Demonstrates inference patterns
- `examples/yolo_detection_demo.py` - Visual detection demo
- `examples/settings_examples.py` - Configuration testing
- Template generation testing
- CI/CD pipeline testing (when implemented)