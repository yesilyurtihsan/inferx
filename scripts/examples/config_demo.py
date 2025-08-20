#!/usr/bin/env python3
"""
InferX Config System Demo

Bu script InferX'deki config.yaml dosyalarının nasıl çalıştığını
pratik örneklerle gösterir.
"""

import yaml
import tempfile
from pathlib import Path
import sys

# Geçici olarak InferX'i import edebilmek için
sys.path.insert(0, str(Path(__file__).parent))

def create_demo_configs():
    """Demo için geçici config dosyaları oluştur"""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Demo configs created in: {temp_dir}")
    
    # 1. Default Config (System)
    default_config = {
        "model_detection": {
            "yolo_keywords": ["yolo", "yolov8", "ultralytics"],
            "classification_keywords": ["resnet", "efficientnet"]
        },
        "device_mapping": {
            "auto": "AUTO",
            "cpu": "CPU", 
            "gpu": "GPU"
        },
        "model_defaults": {
            "yolo": {
                "confidence_threshold": 0.25,
                "input_size": 640,
                "nms_threshold": 0.45
            }
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    # 2. User Global Config (Personal preferences)
    user_global_config = {
        "device_mapping": {
            "auto": "GPU"  # User prefers GPU by default
        },
        "model_defaults": {
            "yolo": {
                "confidence_threshold": 0.3  # User wants lower threshold
            }
        },
        "logging": {
            "level": "DEBUG"  # User wants verbose logging
        }
    }
    
    # 3. Project Local Config (Project specific)
    project_local_config = {
        "model_defaults": {
            "yolo": {
                "input_size": 1024,  # High resolution for this project
                "class_names": [
                    "car", "truck", "motorcycle", "bus"  # Custom classes
                ]
            }
        },
        "advanced": {
            "model_cache": {
                "cache_dir": "./project_cache"
            }
        }
    }
    
    # 4. Template Config (Generated project baseline)
    template_config = {
        "model": {
            "path": "models/yolo_model.onnx",
            "type": "yolo"
        },
        "inference": {
            "device": "auto",
            "runtime": "auto", 
            "confidence_threshold": 0.25,
            "input_size": 640
        },
        "preprocessing": {
            "target_size": [640, 640],
            "normalize": True,
            "color_format": "RGB",
            "maintain_aspect_ratio": True
        }
    }
    
    # 5. Runtime Config (Specific run)
    runtime_config = {
        "inference": {
            "device": "cpu",  # Force CPU for this test
            "confidence_threshold": 0.5,  # Higher threshold
            "batch_size": 1
        },
        "logging": {
            "level": "WARNING"  # Less verbose for production
        }
    }
    
    # Save configs
    configs = {
        "default.yaml": default_config,
        "user_global.yaml": user_global_config, 
        "project_local.yaml": project_local_config,
        "template.yaml": template_config,
        "runtime.yaml": runtime_config
    }
    
    config_paths = {}
    for name, config in configs.items():
        path = temp_dir / name
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        config_paths[name] = path
        print(f"   ✅ Created {name}")
    
    return temp_dir, config_paths


def merge_configs(base_config, override_config):
    """Recursively merge override_config into base_config"""
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def demonstrate_config_hierarchy(config_paths):
    """Config hierarchy nasıl çalıştığını göster"""
    print("\n🔄 Config Hierarchy Demonstration")
    print("=" * 60)
    
    # Load configs in order
    with open(config_paths["default.yaml"]) as f:
        merged_config = yaml.safe_load(f)
    print("1. 📋 Default Config loaded")
    print(f"   Initial YOLO confidence: {merged_config['model_defaults']['yolo']['confidence_threshold']}")
    print(f"   Initial device mapping: {merged_config['device_mapping']['auto']}")
    print(f"   Initial log level: {merged_config['logging']['level']}")
    
    # Merge user global
    with open(config_paths["user_global.yaml"]) as f:
        user_config = yaml.safe_load(f)
    merge_configs(merged_config, user_config)
    print("\n2. 👤 User Global Config merged")
    print(f"   YOLO confidence: {merged_config['model_defaults']['yolo']['confidence_threshold']} (updated)")
    print(f"   Device mapping: {merged_config['device_mapping']['auto']} (updated)")
    print(f"   Log level: {merged_config['logging']['level']} (updated)")
    
    # Merge project local
    with open(config_paths["project_local.yaml"]) as f:
        project_config = yaml.safe_load(f)
    merge_configs(merged_config, project_config)
    print("\n3. 🏗️  Project Local Config merged") 
    print(f"   YOLO confidence: {merged_config['model_defaults']['yolo']['confidence_threshold']} (unchanged)")
    print(f"   YOLO input size: {merged_config['model_defaults']['yolo']['input_size']} (updated)")
    print(f"   YOLO classes: {len(merged_config['model_defaults']['yolo']['class_names'])} custom classes")
    print(f"   Cache dir: {merged_config['advanced']['model_cache']['cache_dir']}")
    
    # Merge runtime config
    with open(config_paths["runtime.yaml"]) as f:
        runtime_config = yaml.safe_load(f)
    merge_configs(merged_config, runtime_config)
    print("\n4. ⚡ Runtime Config merged (--config runtime.yaml)")
    print(f"   YOLO confidence: {merged_config['model_defaults']['yolo']['confidence_threshold']} (unchanged)")
    print(f"   Inference device: {merged_config['inference']['device']} (overridden)")
    print(f"   Inference confidence: {merged_config['inference']['confidence_threshold']} (runtime specific)")
    print(f"   Log level: {merged_config['logging']['level']} (updated)")
    
    print("\n📊 Final Merged Configuration Summary:")
    print(f"   Device: {merged_config['inference']['device']} (from runtime config)")
    print(f"   Confidence: {merged_config['inference']['confidence_threshold']} (from runtime config)")
    print(f"   Input Size: {merged_config['model_defaults']['yolo']['input_size']} (from project config)")
    print(f"   Log Level: {merged_config['logging']['level']} (from runtime config)")
    
    return merged_config


def demonstrate_template_config(config_paths):
    """Template config'in nasıl kullanıldığını göster"""
    print("\n🎨 Template Config Usage")
    print("=" * 60)
    
    with open(config_paths["template.yaml"]) as f:
        template_config = yaml.safe_load(f)
    
    print("Template config (from generators/templates/yolo/config.yaml):")
    print(f"   Model path: {template_config['model']['path']}")
    print(f"   Model type: {template_config['model']['type']}")
    print(f"   Device: {template_config['inference']['device']}")
    print(f"   Runtime: {template_config['inference']['runtime']}")
    print(f"   Input size: {template_config['inference']['input_size']}")
    
    print("\n🎯 Template Config Purpose:")
    print("   • Generated project için ready-to-use settings")
    print("   • Model-specific optimized defaults")
    print("   • User tarafından customize edilebilir")
    print("   • Project structure'a uygun path'ler")
    
    print("\n💡 Template Generation Flow:")
    print("   1. `inferx template yolo --name my-detector`")
    print("   2. Template config kopyalanır → my-detector/config.yaml")
    print("   3. Model type'a göre customize edilir (XML, ONNX vs.)")
    print("   4. User'a hazır çalışır proje teslim edilir")


def demonstrate_real_world_scenarios():
    """Gerçek dünya kullanım senaryoları"""
    print("\n🌍 Real-World Usage Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Development Setup",
            "description": "Developer kendi environment'ını kurar",
            "steps": [
                "inferx config --init  # Creates ~/.inferx/config.yaml",
                "# Edit global preferences: GPU, DEBUG logging",
                "cd my-project",
                "echo 'model_defaults:\n  yolo:\n    input_size: 1024' > inferx_config.yaml",
                "inferx run model.onnx image.jpg  # Uses merged settings"
            ]
        },
        {
            "name": "Production Deployment", 
            "description": "Production environment için optimize config",
            "steps": [
                "cat > production.yaml << EOF",
                "inference:",
                "  device: 'gpu'",
                "  batch_size: 8",
                "logging:",
                "  level: 'WARNING'",
                "EOF",
                "",
                "inferx run model.xml images/ --config production.yaml"
            ]
        },
        {
            "name": "Template Project",
            "description": "Yeni YOLO projesi başlatma",
            "steps": [
                "inferx template yolo --name vehicle-detector",
                "cd vehicle-detector",
                "# config.yaml otomatik oluştu",
                "# Model path, preprocessing settings hazır",
                "# Sadece model dosyasını koy ve çalıştır"
            ]
        },
        {
            "name": "A/B Testing",
            "description": "Farklı ayarlarla test",
            "steps": [
                "# Conservative config",
                "echo 'inference:\n  confidence_threshold: 0.8' > conservative.yaml",
                "",
                "# Aggressive config", 
                "echo 'inference:\n  confidence_threshold: 0.1' > aggressive.yaml",
                "",
                "inferx run model.onnx test_images/ --config conservative.yaml",
                "inferx run model.onnx test_images/ --config aggressive.yaml"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   {scenario['description']}")
        print("   Steps:")
        for step in scenario['steps']:
            if step.strip():
                print(f"     {step}")
            else:
                print()


def demonstrate_config_debugging():
    """Config debugging teknikleri"""
    print("\n🔍 Config Debugging Techniques") 
    print("=" * 60)
    
    print("1. Show Current Effective Config:")
    print("   inferx config --show")
    print("   # Shows final merged configuration")
    
    print("\n2. Validate Configuration:")
    print("   inferx config --validate")
    print("   # Checks for errors and warnings")
    
    print("\n3. Trace Config Loading:")
    print("   inferx run model.onnx image.jpg --verbose")
    print("   # Shows which configs are loaded and merged")
    
    print("\n4. Override Specific Settings:")
    print("   inferx run model.onnx image.jpg --device cpu --confidence 0.5")
    print("   # CLI args override everything")
    
    print("\n5. Config File Validation:")
    print("   python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"")
    print("   # Check YAML syntax")


def main():
    """Main demo function"""
    print("🚀 InferX Config System Demo")
    print("=" * 60)
    print("Bu demo InferX'deki config.yaml dosyalarının nasıl çalıştığını gösterir\n")
    
    # Create demo configs
    temp_dir, config_paths = create_demo_configs()
    
    # Show contents of each config type
    print("\n📋 Config File Types:")
    for name, path in config_paths.items():
        if "template" not in name:  # Skip template for now
            with open(path) as f:
                config = yaml.safe_load(f)
            print(f"\n{name}:")
            if "default" in name:
                print("   • System-wide defaults")
                print("   • Model detection keywords")
                print("   • Device mappings")
            elif "user_global" in name:
                print("   • User personal preferences")
                print("   • Applies to all projects") 
            elif "project_local" in name:
                print("   • Project-specific settings")
                print("   • Committed to git")
            elif "runtime" in name:
                print("   • Specific run configuration")
                print("   • Temporary overrides")
    
    # Demonstrate hierarchy
    final_config = demonstrate_config_hierarchy(config_paths)
    
    # Demonstrate template usage
    demonstrate_template_config(config_paths)
    
    # Show real-world scenarios
    demonstrate_real_world_scenarios()
    
    # Show debugging techniques
    demonstrate_config_debugging()
    
    print(f"\n🧹 Cleanup: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("🎯 Key Takeaways:")
    print("   • 5-level hierarchy: CLI > Runtime > Project > User > Default")
    print("   • Template configs provide ready-to-use baselines")
    print("   • Default config = system settings, never modify")
    print("   • User global = personal preferences across projects")
    print("   • Project local = project-specific, commit to git")
    print("   • Runtime config = temporary run-specific settings")
    print("   • Merge process preserves inheritance and overrides")


if __name__ == "__main__":
    main()