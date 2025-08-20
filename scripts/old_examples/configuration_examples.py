#!/usr/bin/env python3
"""
Configuration Examples

This script demonstrates all the ways to configure InferX,
showing the hierarchical configuration system in action.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferx.settings import get_inferx_settings, validate_yolo_template_config


def example_default_configuration():
    """Example 1: Using default configuration"""
    print("\n🔥 Example 1: Default Configuration")
    print("-" * 50)
    
    try:
        # Load default configuration
        config = InferXConfig()
        
        print("✅ Loaded default configuration")
        print(f"   YOLO keywords: {config.get('model_detection.yolo_keywords', [])[:3]}...")
        print(f"   Default device mapping: {config.get('device_mapping.auto')}")
        print(f"   YOLO input size: {config.get('model_defaults.yolo.input_size')}")
        
        # Show model type detection
        test_files = [
            Path("yolov8n.onnx"),
            Path("yolov8n.xml"), 
            Path("resnet50.onnx"),
            Path("anomaly_model.xml")
        ]
        
        print("\n🔍 Model Type Detection:")
        for file_path in test_files:
            model_type = config.detect_model_type(file_path)
            print(f"   {file_path.name} → {model_type}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_user_config_creation():
    """Example 2: Creating user configuration"""
    print("\n🔥 Example 2: User Configuration Creation")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create user config template
            user_config_path = temp_path / "user_config.yaml"
            create_user_config_template(user_config_path)
            
            print(f"✅ Created user config template: {user_config_path}")
            
            # Read and show part of the template
            with open(user_config_path, 'r') as f:
                lines = f.readlines()[:20]  # First 20 lines
            
            print("\n📄 Template content (first 20 lines):")
            for line in lines:
                print(f"   {line.rstrip()}")
            print("   ...")
            
            print(f"\n💡 Real Usage:")
            print(f"   # Create global user config")
            print(f"   inferx config --init")
            print(f"   ")
            print(f"   # Create project template")
            print(f"   inferx config --template my_config.yaml")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_custom_model_detection():
    """Example 3: Custom model detection patterns"""
    print("\n🔥 Example 3: Custom Model Detection")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create custom config with additional keywords
            custom_config_path = temp_path / "custom_detection.yaml"
            custom_config_content = """
# Custom Model Detection Configuration
model_detection:
  yolo_keywords:
    - "yolo"
    - "yolov8"
    - "my_detector"
    - "vehicle_detector"
    - "person_detector"
  
  classification_keywords:
    - "resnet"
    - "efficientnet"
    - "my_classifier" 
    - "emotion_classifier"
    - "age_predictor"
  
  # Add custom model type
  custom_keywords:
    - "custom_model"
    - "special_detector"
"""
            
            with open(custom_config_path, 'w') as f:
                f.write(custom_config_content)
            
            print(f"✅ Created custom config: {custom_config_path}")
            
            # Load config with custom patterns
            config = InferXConfig(custom_config_path)
            
            # Test custom model detection
            test_models = [
                "my_detector.onnx",
                "vehicle_detector.xml", 
                "emotion_classifier.onnx",
                "age_predictor.xml",
                "standard_yolo.onnx"
            ]
            
            print(f"\n🔍 Custom Model Detection:")
            for model_name in test_models:
                model_path = Path(model_name)
                model_type = config.detect_model_type(model_path)
                print(f"   {model_name} → {model_type}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_device_and_performance_config():
    """Example 4: Device and performance configuration"""
    print("\n🔥 Example 4: Device & Performance Configuration")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create performance-optimized config
            perf_config_path = temp_path / "performance_config.yaml"
            perf_config_content = """
# Performance Optimization Configuration

# Custom device preferences
device_mapping:
  auto: "GPU"          # Prefer GPU over CPU
  fast: "GPU"          # Custom fast device
  edge: "MYRIAD"       # Edge deployment device
  server: "CPU"        # Server deployment

# Performance presets for different scenarios
performance_presets:
  # High throughput for batch processing
  batch_processing:
    openvino:
      performance_hint: "THROUGHPUT"
      num_streams: 0        # Auto-optimize streams
      num_threads: 8        # Use 8 threads
    onnx:
      providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
      session_options:
        graph_optimization_level: "ORT_ENABLE_ALL"
        inter_op_num_threads: 8

  # Low latency for real-time
  real_time:
    openvino:
      performance_hint: "LATENCY"
      num_streams: 1        # Single stream for consistency
      num_threads: 4
    onnx:
      providers: ["CUDAExecutionProvider"]
      session_options:
        graph_optimization_level: "ORT_ENABLE_BASIC"

  # Edge deployment optimized
  edge_deployment:
    openvino:
      performance_hint: "LATENCY"
      device: "MYRIAD"
    onnx:
      providers: ["CPUExecutionProvider"]
      session_options:
        graph_optimization_level: "ORT_ENABLE_EXTENDED"
"""
            
            with open(perf_config_path, 'w') as f:
                f.write(perf_config_content)
            
            print(f"✅ Created performance config: {perf_config_path}")
            
            # Load and test config
            config = InferXConfig(perf_config_path)
            
            # Test device mappings
            device_tests = ["auto", "fast", "edge", "server", "unknown"]
            
            print(f"\n🖥️  Device Mapping Tests:")
            for device in device_tests:
                mapped = config.get_device_name(device)
                print(f"   {device} → {mapped}")
            
            # Show performance presets
            presets = config.get("performance_presets", {})
            print(f"\n⚡ Available Performance Presets:")
            for preset_name in presets.keys():
                print(f"   - {preset_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_model_specific_configuration():
    """Example 5: Model-specific configuration"""
    print("\n🔥 Example 5: Model-Specific Configuration")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create model-specific config
            model_config_path = temp_path / "model_specific.yaml"
            model_config_content = """
# Model-Specific Configuration

model_defaults:
  # High-accuracy YOLO setup
  yolo:
    input_size: 1024              # Higher resolution for accuracy
    confidence_threshold: 0.15    # Lower threshold for more detections
    nms_threshold: 0.35          # Tighter NMS
    max_detections: 200          # More detections allowed
    
    # Custom classes for security camera
    class_names:
      - "person"
      - "vehicle" 
      - "bicycle"
      - "motorcycle"
      - "truck"
      - "bus"
      - "backpack"
      - "handbag"
      - "suitcase"
      - "suspicious_object"

  # Custom classification model
  classification:
    input_size: [384, 384]        # Higher resolution
    top_k: 3                      # Top 3 predictions
    
    # Custom normalization for specific model
    normalize:
      mean: [0.5, 0.5, 0.5]      # Different from ImageNet
      std: [0.5, 0.5, 0.5]
    
    # Custom class names
    class_names:
      - "happy"
      - "sad"
      - "angry"
      - "surprised"
      - "neutral"

  # Anomaly detection setup
  anomalib:
    input_size: [256, 256]
    threshold: 0.3               # Lower threshold for sensitivity
    return_heatmap: true         # Include anomaly heatmaps
    
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

# Runtime-specific preprocessing
preprocessing_defaults:
  openvino:
    target_size: [640, 640]
    normalize: true
    color_format: "RGB"
    maintain_aspect_ratio: true  # Use letterboxing
  
  onnx:
    target_size: [640, 640] 
    normalize: true
    color_format: "RGB"
    maintain_aspect_ratio: true
"""
            
            with open(model_config_path, 'w') as f:
                f.write(model_config_content)
            
            print(f"✅ Created model-specific config: {model_config_path}")
            
            # Load and test config
            config = InferXConfig(model_config_path)
            
            # Test model defaults
            model_types = ["yolo", "classification", "anomalib"]
            
            print(f"\n🎯 Model-Specific Defaults:")
            for model_type in model_types:
                defaults = config.get_model_defaults(model_type)
                print(f"   {model_type}:")
                if "input_size" in defaults:
                    print(f"     Input size: {defaults['input_size']}")
                if "confidence_threshold" in defaults:
                    print(f"     Confidence: {defaults['confidence_threshold']}")
                if "class_names" in defaults:
                    num_classes = len(defaults['class_names'])
                    print(f"     Classes: {num_classes} classes")
                    print(f"     Examples: {defaults['class_names'][:3]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_configuration_validation():
    """Example 6: Configuration validation"""
    print("\n🔥 Example 6: Configuration Validation")
    print("-" * 50)
    
    try:
        # Test valid configuration
        valid_config = {
            "model_detection": {
                "yolo_keywords": ["yolo", "custom_yolo"]
            },
            "device_mapping": {
                "auto": "CPU",
                "fast": "GPU"
            },
            "model_defaults": {
                "yolo": {
                    "confidence_threshold": 0.5,
                    "input_size": 640
                }
            }
        }
        
        print("✅ Testing valid configuration...")
        warnings = validate_config(valid_config)
        if warnings:
            print(f"   Warnings: {warnings}")
        else:
            print("   ✅ Configuration is valid!")
        
        # Test invalid configuration
        invalid_config = {
            "model_detection": {
                "yolo_keywords": "not_a_list"  # Should be list
            },
            "device_mapping": {
                "auto": 123  # Should be string
            },
            "model_defaults": {
                "yolo": {
                    "confidence_threshold": 1.5  # Should be 0-1
                }
            }
        }
        
        print(f"\n❌ Testing invalid configuration...")
        warnings = validate_config(invalid_config)
        if warnings:
            print(f"   Found {len(warnings)} validation issues:")
            for warning in warnings:
                print(f"     - {warning}")
        
        print(f"\n💡 CLI Validation:")
        print(f"   inferx config --validate    # Check current config")
        print(f"   inferx config --show        # View current config")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_configuration_hierarchy():
    """Example 7: Configuration hierarchy demonstration"""
    print("\n🔥 Example 7: Configuration Hierarchy")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create different levels of configuration
            
            # 1. Global config (would be ~/.inferx/config.yaml)
            global_config = temp_path / "global_config.yaml"
            global_config.write_text("""
# Global User Configuration
model_defaults:
  yolo:
    confidence_threshold: 0.25    # Global default
    
device_mapping:
  auto: "CPU"                     # User prefers CPU globally
""")
            
            # 2. Project config (would be ./inferx_config.yaml)
            project_config = temp_path / "project_config.yaml"
            project_config.write_text("""
# Project-Specific Configuration  
model_defaults:
  yolo:
    confidence_threshold: 0.4     # Override global setting
    input_size: 1024              # Project-specific setting
    
performance_presets:
  project_optimized:
    openvino:
      performance_hint: "THROUGHPUT"
""")
            
            # 3. User-specified config
            custom_config = temp_path / "custom_config.yaml"
            custom_config.write_text("""
# Custom Run Configuration
model_defaults:
  yolo:
    confidence_threshold: 0.15    # Override project setting
    class_names: ["person", "vehicle"]  # Custom classes
""")
            
            print("📁 Configuration Hierarchy Simulation:")
            print(f"   Global:  {global_config.name}")
            print(f"   Project: {project_config.name}")  
            print(f"   Custom:  {custom_config.name}")
            
            # Simulate loading in hierarchy order
            configs = {
                "1. Default": InferXConfig(),
                "2. + Global": InferXConfig(),  # Would load global automatically
                "3. + Project": InferXConfig(), # Would load project automatically  
                "4. + Custom": InferXConfig(custom_config)
            }
            
            print(f"\n📊 Configuration Values at Each Level:")
            print(f"   Setting: model_defaults.yolo.confidence_threshold")
            
            # Show how values would change through hierarchy
            values = {
                "1. Default only": 0.25,      # From default config
                "2. + Global": 0.25,          # Global config value  
                "3. + Project": 0.4,          # Project override
                "4. + Custom": 0.15           # Final override
            }
            
            for level, value in values.items():
                print(f"   {level}: {value}")
            
            print(f"\n💡 Hierarchy Priority (highest to lowest):")
            print(f"   1. CLI arguments (--device gpu)")
            print(f"   2. User-specified config (--config custom.yaml)")
            print(f"   3. Project local config (./inferx_config.yaml)")
            print(f"   4. User global config (~/.inferx/config.yaml)")
            print(f"   5. Package defaults (built-in)")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_practical_workflows():
    """Example 8: Practical configuration workflows"""
    print("\n🔥 Example 8: Practical Workflows")
    print("-" * 50)
    
    workflows = [
        {
            "name": "🏠 Personal Setup",
            "description": "Setup InferX for personal use",
            "steps": [
                "inferx config --init",
                "# Edit ~/.inferx/config.yaml for your preferences",
                "inferx config --validate",
                "inferx run model.onnx image.jpg"
            ]
        },
        {
            "name": "🏢 Team Project Setup", 
            "description": "Setup InferX for team collaboration",
            "steps": [
                "cd my_project/",
                "inferx config --template inferx_config.yaml",
                "# Edit inferx_config.yaml for project needs",
                "git add inferx_config.yaml",
                "# Team members get config automatically"
            ]
        },
        {
            "name": "🚀 Production Deployment",
            "description": "Setup InferX for production",
            "steps": [
                "inferx config --template production.yaml",
                "# Configure for high throughput/low latency",
                "inferx run model.xml images/ --config production.yaml",
                "# Benchmark and optimize"
            ]
        },
        {
            "name": "🧪 Model Development",
            "description": "Setup for model development/testing",
            "steps": [
                "inferx config --template dev.yaml",
                "# Configure debug settings, lower thresholds",
                "inferx run experimental_model.onnx test_images/ --config dev.yaml --verbose",
                "# Iterate and improve"
            ]
        },
        {
            "name": "⚡ Performance Optimization",
            "description": "Optimize inference performance",
            "steps": [
                "# Test different configurations",
                "inferx run model.xml image.jpg --device cpu --verbose",
                "inferx run model.xml image.jpg --device gpu --verbose", 
                "# Create optimized config",
                "inferx config --template optimized.yaml",
                "# Use best performing settings"
            ]
        }
    ]
    
    for workflow in workflows:
        print(f"\n{workflow['name']}")
        print(f"   {workflow['description']}")
        for step in workflow['steps']:
            if step.startswith('#'):
                print(f"   {step}")
            else:
                print(f"   $ {step}")


def main():
    """Run all configuration examples"""
    print("🎛️  InferX Configuration Examples")
    print("=" * 60)
    print("This script demonstrates the flexible configuration system")
    
    examples = [
        example_default_configuration,
        example_user_config_creation,
        example_custom_model_detection,
        example_device_and_performance_config,
        example_model_specific_configuration,
        example_configuration_validation,
        example_configuration_hierarchy,
        example_practical_workflows
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
    
    print(f"\n🎯 Configuration Summary")
    print("=" * 50)
    print("✅ InferX Configuration Features:")
    print("   - Hierarchical loading (global → project → custom)")
    print("   - Model type auto-detection with custom keywords")
    print("   - Device mapping and performance presets")
    print("   - Model-specific defaults and overrides")
    print("   - Validation and debugging tools")
    print("   - CLI integration for easy management")
    
    print(f"\n📚 Quick Reference:")
    print("   inferx config --init         # Setup user config")  
    print("   inferx config --template x.yaml  # Create template")
    print("   inferx config --validate     # Check config")
    print("   inferx config --show         # View current config")
    print("   inferx run model.xml image.jpg --config custom.yaml")
    
    print(f"\n📖 See CONFIG_GUIDE.md for complete documentation!")


if __name__ == "__main__":
    main()