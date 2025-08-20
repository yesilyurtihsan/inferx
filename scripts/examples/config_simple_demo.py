#!/usr/bin/env python3
"""
InferX Config System Simple Demo
Emoji olmadan Windows-friendly version
"""

import yaml
import tempfile
from pathlib import Path

def create_demo_configs():
    """Demo configs olustur"""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Demo configs created in: {temp_dir}")
    
    # 1. Default Config (System-wide)
    default_config = {
        "model_defaults": {
            "yolo": {
                "confidence_threshold": 0.25,
                "input_size": 640,
                "nms_threshold": 0.45
            }
        },
        "device_mapping": {
            "auto": "AUTO",
            "cpu": "CPU", 
            "gpu": "GPU"
        },
        "logging": {"level": "INFO"}
    }
    
    # 2. User Global Config  
    user_global_config = {
        "model_defaults": {
            "yolo": {
                "confidence_threshold": 0.3  # User prefers lower threshold
            }
        },
        "device_mapping": {
            "auto": "GPU"  # User prefers GPU
        },
        "logging": {"level": "DEBUG"}
    }
    
    # 3. Project Local Config
    project_local_config = {
        "model_defaults": {
            "yolo": {
                "input_size": 1024,  # High resolution for this project
                "class_names": ["car", "truck", "motorcycle", "bus"]
            }
        }
    }
    
    # 4. Template Config (Generated project)
    template_config = {
        "model": {
            "path": "models/yolo_model.onnx",
            "type": "yolo"
        },
        "inference": {
            "device": "auto",
            "confidence_threshold": 0.25,
            "input_size": 640
        }
    }
    
    # 5. Runtime Config (Specific run)
    runtime_config = {
        "inference": {
            "device": "cpu",  # Force CPU for this test
            "confidence_threshold": 0.5
        },
        "logging": {"level": "WARNING"}
    }
    
    # Save configs
    configs = {
        "1_default.yaml": default_config,
        "2_user_global.yaml": user_global_config, 
        "3_project_local.yaml": project_local_config,
        "4_template.yaml": template_config,
        "5_runtime.yaml": runtime_config
    }
    
    config_paths = {}
    for name, config in configs.items():
        path = temp_dir / name
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        config_paths[name] = path
        print(f"   Created {name}")
    
    return temp_dir, config_paths

def merge_configs(base_config, override_config):
    """Recursively merge configs"""
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value

def show_config_hierarchy(config_paths):
    """Config hierarchy demonstration"""
    print("\n" + "="*50)
    print("CONFIG HIERARCHY DEMONSTRATION")
    print("="*50)
    
    # Start with default
    with open(config_paths["1_default.yaml"]) as f:
        merged = yaml.safe_load(f)
    print("1. DEFAULT CONFIG loaded:")
    print(f"   YOLO confidence: {merged['model_defaults']['yolo']['confidence_threshold']}")
    print(f"   Device auto: {merged['device_mapping']['auto']}")
    print(f"   Log level: {merged['logging']['level']}")
    
    # Merge user global
    with open(config_paths["2_user_global.yaml"]) as f:
        user_config = yaml.safe_load(f)
    merge_configs(merged, user_config)
    print("\n2. USER GLOBAL CONFIG merged:")
    print(f"   YOLO confidence: {merged['model_defaults']['yolo']['confidence_threshold']} (UPDATED)")
    print(f"   Device auto: {merged['device_mapping']['auto']} (UPDATED)")
    print(f"   Log level: {merged['logging']['level']} (UPDATED)")
    
    # Merge project local
    with open(config_paths["3_project_local.yaml"]) as f:
        project_config = yaml.safe_load(f)
    merge_configs(merged, project_config)
    print("\n3. PROJECT LOCAL CONFIG merged:")
    print(f"   YOLO confidence: {merged['model_defaults']['yolo']['confidence_threshold']} (unchanged)")
    print(f"   YOLO input size: {merged['model_defaults']['yolo']['input_size']} (UPDATED)")
    print(f"   YOLO classes: {len(merged['model_defaults']['yolo']['class_names'])} custom classes (NEW)")
    
    # Merge runtime config
    with open(config_paths["5_runtime.yaml"]) as f:
        runtime_config = yaml.safe_load(f)
    merge_configs(merged, runtime_config)
    print("\n4. RUNTIME CONFIG merged (--config runtime.yaml):")
    print(f"   Inference device: {merged['inference']['device']} (RUNTIME OVERRIDE)")
    print(f"   Inference confidence: {merged['inference']['confidence_threshold']} (RUNTIME SPECIFIC)")
    print(f"   Log level: {merged['logging']['level']} (UPDATED)")
    
    print("\n" + "="*30)
    print("FINAL EFFECTIVE CONFIG:")
    print("="*30)
    print("Values come from different sources:")
    print(f"   Device: '{merged['inference']['device']}' (from runtime config)")
    print(f"   Confidence: {merged['inference']['confidence_threshold']} (from runtime config)")
    print(f"   Input Size: {merged['model_defaults']['yolo']['input_size']} (from project config)")
    print(f"   Custom Classes: {merged['model_defaults']['yolo']['class_names']} (from project config)")
    print(f"   Log Level: '{merged['logging']['level']}' (from runtime config)")
    
    return merged

def show_template_usage(config_paths):
    """Template config demonstration"""
    print("\n" + "="*50)
    print("TEMPLATE CONFIG USAGE")
    print("="*50)
    
    with open(config_paths["4_template.yaml"]) as f:
        template = yaml.safe_load(f)
    
    print("Template config (from generators/templates/yolo/config.yaml):")
    print(f"   Model path: {template['model']['path']}")
    print(f"   Model type: {template['model']['type']}")
    print(f"   Device: {template['inference']['device']}")
    print(f"   Input size: {template['inference']['input_size']}")
    
    print("\nTemplate Config Purpose:")
    print("   - Generated project icin ready-to-use settings")
    print("   - Model-specific optimized defaults")
    print("   - User tarafindan customize edilebilir")
    print("   - Project structure'a uygun path'ler")
    
    print("\nTemplate Generation Flow:")
    print("   1. `inferx template yolo --name my-detector`")
    print("   2. Template config kopyalanir -> my-detector/config.yaml")
    print("   3. Model type'a gore customize edilir")
    print("   4. User'a hazir calisir proje teslim edilir")

def show_real_scenarios():
    """Real-world scenarios"""
    print("\n" + "="*50)
    print("REAL-WORLD USAGE SCENARIOS")
    print("="*50)
    
    print("1. DEVELOPMENT SETUP:")
    print("   inferx config --init  # Creates ~/.inferx/config.yaml")
    print("   # Edit global preferences: GPU, DEBUG logging")
    print("   cd my-project")
    print("   echo 'model_defaults:' > inferx_config.yaml")
    print("   echo '  yolo:' >> inferx_config.yaml") 
    print("   echo '    input_size: 1024' >> inferx_config.yaml")
    print("   inferx run model.onnx image.jpg  # Uses merged settings")
    
    print("\n2. PRODUCTION DEPLOYMENT:")
    print("   cat > production.yaml << EOF")
    print("   inference:")
    print("     device: 'gpu'")
    print("     batch_size: 8")
    print("   logging:")
    print("     level: 'WARNING'")
    print("   EOF")
    print("   inferx run model.xml images/ --config production.yaml")
    
    print("\n3. TEMPLATE PROJECT:")
    print("   inferx template yolo --name vehicle-detector")
    print("   cd vehicle-detector")
    print("   # config.yaml otomatik olusturuldu")
    print("   # Model path, preprocessing ayarlari hazir")
    print("   # Sadece model dosyasini koy ve calistir")

def main():
    print("INFERX CONFIG SYSTEM DEMO")
    print("="*50)
    print("Bu demo InferX'deki config.yaml dosyalarinin nasil calistigini gosterir")
    
    # Create configs
    temp_dir, config_paths = create_demo_configs()
    
    # Show each config type
    print("\nCONFIG FILE TYPES:")
    config_types = {
        "1_default.yaml": "System-wide defaults, model detection keywords",
        "2_user_global.yaml": "User personal preferences, applies to all projects",
        "3_project_local.yaml": "Project-specific settings, committed to git", 
        "4_template.yaml": "Generated project baseline configuration",
        "5_runtime.yaml": "Specific run configuration, temporary overrides"
    }
    
    for name, desc in config_types.items():
        print(f"   {name}: {desc}")
    
    # Show hierarchy
    final_config = show_config_hierarchy(config_paths)
    
    # Show template usage
    show_template_usage(config_paths)
    
    # Show scenarios
    show_real_scenarios()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleanup: {temp_dir}")
    
    print("\n" + "="*50)
    print("KEY TAKEAWAYS:")
    print("="*50)
    print("- 5-level hierarchy: CLI > Runtime > Project > User > Default")
    print("- Template configs provide ready-to-use baselines")
    print("- Default config = system settings, never modify")
    print("- User global = personal preferences across projects")
    print("- Project local = project-specific, commit to git")
    print("- Runtime config = temporary run-specific settings")
    print("- Merge process preserves inheritance and overrides")

if __name__ == "__main__":
    main()