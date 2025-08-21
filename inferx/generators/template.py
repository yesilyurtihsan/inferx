import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from ..settings import validate_yolo_template_config

class TemplateGenerator:
    def __init__(self):
        """Initialize template generator with paths"""
        self.inferx_root = Path(__file__).parent.parent
        self.templates_dir = self.inferx_root / "generators" / "templates"
    
    def generate(self, model_type: str, project_name: str, **options):
        """
        Generate base template project
        
        Args:
            model_type: Type of model template (yolo, yolo_openvino, anomaly, classification)
            project_name: Name of the project to create
            **options: Additional options (device, runtime, config_path, model_path)
        """
        project_path = Path(project_name)
        project_path.mkdir(exist_ok=True)
        
        # Copy base template files
        self._copy_base_template(model_type, project_path)
        
        # Update config with user options
        self._update_config(project_path, options)
        
        # Copy model file if provided
        if "model_path" in options and options["model_path"]:
            self._copy_model_file(options["model_path"], model_type, project_path)
        
        # Validate generated configuration
        self._validate_generated_config(project_path)
        
        print(f"✅ Created {model_type} template: {project_name}")
    
    def add_api_layer(self, project_path: str):
        """
        Add FastAPI server layer to existing project
        
        Args:
            project_path: Path to existing project
        """
        project_path = Path(project_path)
        
        # Copy server.py from template
        self._copy_server_py(project_path)
        
        # Update pyproject.toml to include API dependencies
        self._update_pyproject_for_api(project_path)
        
        print("✅ Added API layer to project")
    
    def add_docker_layer(self, project_path: str, **options):
        """
        Add Docker container layer to existing project
        
        Args:
            project_path: Path to existing project
            **options: Docker options (tag, optimize, compose)
        """
        project_path = Path(project_path)
        
        # Copy Docker files from template
        self._copy_docker_files(project_path)
        
        print("✅ Added Docker layer to project")
    
    def _copy_base_template(self, model_type: str, project_path: Path):
        """Copy base template files from templates directory"""
        # Determine template source directory
        if model_type == "yolo":
            template_src = self.templates_dir / "yolo"
        else:
            # For other model types, use YOLO template as fallback
            template_src = self.templates_dir / "yolo"
        
        # Copy all files from template directory except src directory
        if template_src.exists():
            for item in template_src.iterdir():
                if item.name != "src":  # src klasörünü kopyalamıyoruz
                    dst_path = project_path / item.name
                    if item.is_dir():
                        self._copy_directory(item, dst_path)
                    else:
                        # Don't copy __pycache__ directories or .pyc files
                        if "__pycache__" not in str(item) and not item.name.endswith(".pyc"):
                            shutil.copy2(item, dst_path)
        else:
            print(f"Warning: Template directory {template_src} not found")
        
        # Create src directory
        src_path = project_path / "src"
        src_path.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = src_path / "__init__.py"
        init_file.write_text("")
        
        # Create models directory
        models_path = project_path / "models"
        models_path.mkdir(exist_ok=True)
        (models_path / ".gitkeep").write_text("# Keep this directory in git")
        
        # Update config.yaml with model-specific settings
        self._update_config_for_model_type(model_type, project_path)
        
        # Create inferencer.py based on model type
        self._create_inferencer_py(model_type, src_path)
    
    def _update_config_for_model_type(self, model_type: str, project_path: Path):
        """Update config.yaml with model-specific settings"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            return
        
        config_content = config_file.read_text()
        
        # Update model path and type based on model_type
        if model_type == "yolo_openvino":
            config_content = config_content.replace(
                'path: "models/yolo_model.onnx"', 
                'path: "models/yolo_model.xml"'
            )
            config_content = config_content.replace(
                'type: "yolo"', 
                'type: "yolo_openvino"'
            )
            # Update runtime if it exists
            if 'runtime: "auto"' in config_content:
                config_content = config_content.replace(
                    'runtime: "auto"', 
                    'runtime: "openvino"'
                )
            # Create placeholder files for OpenVINO
            models_path = project_path / "models"
            (models_path / "yolo_model.xml").write_text("# Place your YOLO OpenVINO model .xml file here")
            (models_path / "yolo_model.bin").write_text("# Place your YOLO OpenVINO model .bin file here")
        elif model_type == "yolo":
            # Already correct for YOLO ONNX, just create placeholder
            models_path = project_path / "models"
            (models_path / "yolo_model.onnx").write_text("# Place your YOLO ONNX model file here")
        else:
            # Generic model
            config_content = config_content.replace(
                'path: "models/yolo_model.onnx"', 
                'path: "models/model.onnx"'
            )
            config_content = config_content.replace(
                'type: "yolo"', 
                'type: "generic"'
            )
            # Create placeholder for generic model
            models_path = project_path / "models"
            (models_path / "model.onnx").write_text("# Place your model file here")
        
        config_file.write_text(config_content)
    
    def _copy_directory(self, src: Path, dst: Path):
        """Recursively copy directory contents"""
        dst.mkdir(exist_ok=True)
        for item in src.iterdir():
            dst_path = dst / item.name
            if item.is_dir():
                self._copy_directory(item, dst_path)
            else:
                # Don't copy __pycache__ directories or .pyc files
                if "__pycache__" not in str(item) and not item.name.endswith(".pyc"):
                    shutil.copy2(item, dst_path)
    
    def _create_inferencer_py(self, model_type: str, src_path: Path):
        """Create self-contained inferencer.py with all dependencies"""
        if model_type == "yolo_openvino":
            self._create_yolo_openvino_inferencer(src_path)
        elif model_type == "yolo":
            self._create_yolo_inferencer(src_path)
        else:
            # Generic inferencer for other model types
            self._create_generic_inferencer(src_path)
    
    def _create_yolo_inferencer(self, src_path: Path):
        """Create YOLO inferencer with all dependencies"""
        # Copy yolo.py and fix import paths
        yolo_source = self.inferx_root / "inferencers" / "yolo.py"
        if yolo_source.exists():
            content = yolo_source.read_text()
            # Fix import paths
            content = content.replace("from .base import BaseInferencer", 
                                    "from base import BaseInferencer")
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from utils import ImageProcessor")
            # Rename class to Inferencer
            content = content.replace("class YOLOInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("base.py", src_path)
            self._copy_dependency("utils.py", src_path)
    
    def _create_yolo_openvino_inferencer(self, src_path: Path):
        """Create YOLO OpenVINO inferencer with all dependencies"""
        # Copy yolo_openvino.py and fix import paths
        yolo_source = self.inferx_root / "inferencers" / "yolo_openvino.py"
        if yolo_source.exists():
            content = yolo_source.read_text()
            # Fix import paths
            content = content.replace("from .base import BaseInferencer", 
                                    "from base import BaseInferencer")
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from utils import ImageProcessor")
            # Rename class to Inferencer
            content = content.replace("class YOLOOpenVINOInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("base.py", src_path)
            self._copy_dependency("utils.py", src_path)
    
    def _create_generic_inferencer(self, src_path: Path):
        """Create generic ONNX inferencer with all dependencies"""
        # Copy onnx_inferencer.py and fix import paths
        onnx_source = self.inferx_root / "inferencers" / "onnx_inferencer.py"
        if onnx_source.exists():
            content = onnx_source.read_text()
            # Fix import paths
            content = content.replace("from .base import BaseInferencer", 
                                    "from base import BaseInferencer")
            content = content.replace("from ..utils import ImageProcessor, preprocess_for_inference", 
                                    "from utils import ImageProcessor, preprocess_for_inference")
            # Rename class to Inferencer
            content = content.replace("class ONNXInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("base.py", src_path)
            self._copy_dependency("utils.py", src_path)
    
    def _copy_dependency(self, dependency_file: str, src_path: Path):
        """Copy dependency file from inferx/inferencers/ to src/"""
        source_file = self.inferx_root / "inferencers" / dependency_file
        target_file = src_path / dependency_file
        if source_file.exists():
            content = source_file.read_text()
            # Fix any relative imports in dependency files
            content = content.replace("from .base import", "from base import")
            content = content.replace("from ..utils import", "from utils import")
            target_file.write_text(content)
    
    def _copy_model_file(self, model_path: str, model_type: str, project_path: Path):
        """Copy model file to template project"""
        model_path = Path(model_path)
        models_dir = project_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        if model_type == "yolo_openvino":
            # For OpenVINO, we need both .xml and .bin files
            if model_path.suffix == ".xml":
                # Copy .xml file
                xml_target = models_dir / "yolo_model.xml"
                shutil.copy2(model_path, xml_target)
                print(f"✅ Copied OpenVINO model XML: {xml_target}")
                
                # Look for corresponding .bin file
                bin_path = model_path.with_suffix(".bin")
                if bin_path.exists():
                    bin_target = models_dir / "yolo_model.bin"
                    shutil.copy2(bin_path, bin_target)
                    print(f"✅ Copied OpenVINO model BIN: {bin_target}")
                else:
                    print("⚠️  Warning: Could not find corresponding .bin file")
            else:
                print("⚠️  Warning: OpenVINO models should have .xml extension")
        else:
            # For ONNX and other models
            if model_type == "yolo":
                target_name = "yolo_model.onnx"
            else:
                target_name = "model.onnx"
            
            target_path = models_dir / target_name
            shutil.copy2(model_path, target_path)
            print(f"✅ Copied model file: {target_path}")
    
    def _copy_server_py(self, project_path: Path):
        """Copy server.py from template"""
        src_path = project_path / "src"
        src_path.mkdir(exist_ok=True)
        
        # Copy server.py from template
        template_server = self.templates_dir / "yolo" / "src" / "server.py"
        if template_server.exists():
            shutil.copy2(template_server, src_path / "server.py")
    
    def _copy_docker_files(self, project_path: Path):
        """Copy Docker files from template"""
        # Copy Dockerfile
        dockerfile_src = self.templates_dir / "yolo" / "Dockerfile"
        if dockerfile_src.exists():
            shutil.copy2(dockerfile_src, project_path / "Dockerfile")
        
        # Copy docker-compose.yml
        compose_src = self.templates_dir / "yolo" / "docker-compose.yml"
        if compose_src.exists():
            shutil.copy2(compose_src, project_path / "docker-compose.yml")
        
        # Copy .dockerignore
        dockerignore_src = self.templates_dir / "yolo" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy2(dockerignore_src, project_path / ".dockerignore")
    
    def _update_config(self, project_path: Path, options: dict):
        """Update config file with user options"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            return
        
        # Read current config
        config_content = config_file.read_text()
        
        # Update device if specified
        if "device" in options and options["device"] != "auto":
            config_content = config_content.replace(
                'device: "auto"', 
                f'device: "{options["device"]}"'
            )
        
        # Update runtime if specified
        if "runtime" in options and options["runtime"] != "auto":
            # Add runtime section if it doesn't exist
            if "runtime:" not in config_content:
                config_content = config_content.replace(
                    "inference:",
                    "inference:\\n  runtime: \"auto\"",
                )
            config_content = config_content.replace(
                'runtime: "auto"',
                f'runtime: "{options["runtime"]}"'
            )
        
        # Write updated config
        config_file.write_text(config_content)
    
    def _update_pyproject_for_api(self, project_path: Path):
        """Update pyproject.toml to include API dependencies"""
        pyproject_file = project_path / "pyproject.toml"
        if not pyproject_file.exists():
            return
        
        content = pyproject_file.read_text()
        # Ensure API dependencies are included
        if "[project.optional-dependencies]" not in content:
            content += '''
[project.optional-dependencies]
api = ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "python-multipart>=0.0.6"]
'''
        elif "api = " not in content:
            content = content.replace(
                "[project.optional-dependencies]",
                '''[project.optional-dependencies]
api = ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "python-multipart>=0.0.6"]'''
            )
        
        pyproject_file.write_text(content)
    
    def _validate_generated_config(self, project_path: Path):
        """Validate generated template configuration with Pydantic"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            print("⚠️ Warning: config.yaml not found, skipping validation")
            return
        
        try:
            validated_config = validate_yolo_template_config(config_file)
            print("✅ YOLO template configuration validated with Pydantic")
            print(f"   Model: {validated_config.model_path}")
            print(f"   Input size: {validated_config.input_size}")
            print(f"   Confidence: {validated_config.confidence_threshold}")
        except ImportError:
            print("⚠️ pydantic-settings not available, skipping validation")
        except Exception as e:
            print(f"⚠️ Configuration validation warning: {e}")
            print("   Template created but config may need adjustment")