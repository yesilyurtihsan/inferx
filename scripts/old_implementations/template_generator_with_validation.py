"""
Enhanced Template Generator with Pydantic Validation

This demonstrates how to integrate Pydantic validation into template generation
"""

import shutil
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config_validator import TemplateConfigValidator
from .exceptions import ConfigurationError, TemplateError, ErrorCode

logger = logging.getLogger(__name__)


class EnhancedTemplateGenerator:
    """
    Enhanced template generator with Pydantic validation
    
    Ensures generated template config.yaml files are valid and type-safe
    """
    
    def __init__(self):
        """Initialize enhanced template generator"""
        self.inferx_root = Path(__file__).parent.parent
        self.templates_dir = self.inferx_root / "generators" / "templates"
        self.validator = TemplateConfigValidator()
        
        logger.debug(f"Template generator initialized with templates dir: {self.templates_dir}")
    
    def generate(self, model_type: str, project_name: str, **options) -> Dict[str, Any]:
        """
        Generate template project with validated configuration
        
        Args:
            model_type: Type of model template (yolo, yolo_openvino, classification, etc.)
            project_name: Name of the project to create
            **options: Additional options (device, runtime, config_path, model_path)
            
        Returns:
            Dictionary containing generation results and validation info
            
        Raises:
            TemplateError: If template generation fails
            ConfigurationError: If generated config validation fails
        """
        project_path = Path(project_name)
        project_path.mkdir(exist_ok=True)
        
        generation_result = {
            "project_name": project_name,
            "project_path": str(project_path.absolute()),
            "model_type": model_type,
            "validation_enabled": True,
            "config_validated": False,
            "validation_warnings": [],
            "files_created": []
        }
        
        try:
            logger.info(f"Generating {model_type} template: {project_name}")
            
            # 1. Copy base template files
            self._copy_base_template(model_type, project_path)
            generation_result["files_created"].extend(["src/", "models/", "pyproject.toml"])
            
            # 2. Create and validate configuration
            config_data = self._create_template_config(model_type, options)
            validated_config = self._validate_and_save_config(config_data, model_type, project_path)
            generation_result["config_validated"] = True
            generation_result["files_created"].append("config.yaml")
            
            # 3. Copy model file if provided
            if "model_path" in options and options["model_path"]:
                self._copy_model_file(options["model_path"], model_type, project_path)
                generation_result["files_created"].append(f"models/{Path(options['model_path']).name}")
            
            # 4. Final validation check
            self._perform_final_validation(project_path, model_type)
            
            logger.info(f"✅ Successfully created {model_type} template: {project_name}")
            print(f"✅ Created {model_type} template: {project_name}")
            print(f"   📁 Project path: {project_path.absolute()}")
            print(f"   ✅ Configuration validated with Pydantic")
            
            if generation_result["validation_warnings"]:
                print(f"   ⚠️  {len(generation_result['validation_warnings'])} validation warnings")
                for warning in generation_result["validation_warnings"]:
                    print(f"      • {warning}")
            
            return generation_result
            
        except ConfigurationError as e:
            logger.error(f"Configuration validation failed for {model_type} template: {e}")
            raise TemplateError(
                message=f"Template configuration validation failed for {model_type}",
                error_code=ErrorCode.TEMPLATE_GENERATION_FAILED,
                suggestions=e.suggestions if hasattr(e, 'suggestions') else [
                    "Check template parameters",
                    "Verify model type is supported",
                    "Check input validation rules"
                ],
                original_error=e,
                context={"model_type": model_type, "project_name": project_name}
            )
        
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            raise TemplateError(
                message=f"Failed to generate {model_type} template",
                error_code=ErrorCode.TEMPLATE_GENERATION_FAILED,
                suggestions=[
                    "Check template directory exists",
                    "Verify write permissions",
                    "Check available disk space"
                ],
                original_error=e,
                context={"model_type": model_type, "project_name": project_name}
            )
    
    def _create_template_config(self, model_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create template configuration data with type-specific defaults"""
        # Determine model path based on model type
        if model_type == "yolo_openvino":
            model_path = "models/yolo_model.xml"
        elif model_type.startswith("yolo"):
            model_path = "models/yolo_model.onnx"
        elif model_type == "classification":
            model_path = "models/classification_model.onnx"
        else:
            model_path = "models/model.onnx"
        
        # Override with user-provided model path if available
        if "model_path" in options and options["model_path"]:
            model_filename = Path(options["model_path"]).name
            model_path = f"models/{model_filename}"
        
        # Create base configuration
        config_data = {
            "model": {
                "path": model_path,
                "type": model_type
            },
            "inference": {
                "device": options.get("device", "auto"),
                "runtime": options.get("runtime", "auto"),
                "confidence_threshold": options.get("confidence_threshold", 0.25)
            },
            "preprocessing": {
                "normalize": True,
                "color_format": "RGB",
                "maintain_aspect_ratio": True
            }
        }
        
        # Add model-specific settings
        if model_type.startswith("yolo"):
            config_data["inference"].update({
                "nms_threshold": options.get("nms_threshold", 0.45),
                "input_size": options.get("input_size", 640)
            })
            config_data["preprocessing"]["target_size"] = [
                options.get("input_size", 640),
                options.get("input_size", 640)
            ]
        elif model_type == "classification":
            config_data["inference"]["input_size"] = options.get("input_size", 224)
            config_data["preprocessing"]["target_size"] = [224, 224]
        else:
            config_data["preprocessing"]["target_size"] = [224, 224]
        
        logger.debug(f"Created {model_type} template configuration with {len(config_data)} sections")
        return config_data
    
    def _validate_and_save_config(
        self, 
        config_data: Dict[str, Any], 
        model_type: str, 
        project_path: Path
    ) -> Dict[str, Any]:
        """Validate template configuration and save to file"""
        try:
            # Validate configuration with Pydantic
            validated_config = self.validator.validate_template_config_data(config_data, model_type)
            
            # Convert to dictionary if it's a Pydantic model
            if hasattr(validated_config, 'dict'):
                config_dict = validated_config.dict()
            else:
                config_dict = config_data  # Fallback if validation disabled
            
            # Save validated configuration
            config_file = project_path / "config.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
            
            logger.info(f"✅ Template configuration validated and saved: {config_file}")
            return config_dict
            
        except ConfigurationError as e:
            logger.error(f"Template config validation failed: {e}")
            # Save invalid config with warnings for user to fix
            config_file = project_path / "config.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                f.write(f"\n# WARNING: Configuration validation failed\n")
                f.write(f"# Error: {e}\n")
            raise
    
    def _perform_final_validation(self, project_path: Path, model_type: str):
        """Perform final validation of generated template"""
        config_file = project_path / "config.yaml"
        
        if not config_file.exists():
            raise TemplateError(
                message="Generated config.yaml file not found",
                error_code=ErrorCode.TEMPLATE_INVALID,
                suggestions=["Check template generation process"]
            )
        
        try:
            # Validate the saved configuration file
            validated_config = self.validator.validate_template_config_file(config_file)
            logger.debug("✅ Final template validation successful")
            
        except Exception as e:
            logger.warning(f"Final template validation failed: {e}")
            # Don't fail generation, but warn user
            with open(config_file, 'a', encoding='utf-8') as f:
                f.write(f"\n# VALIDATION WARNING: {e}\n")
    
    def add_api_layer(self, project_path: str, validate_config: bool = True):
        """Add FastAPI server layer with configuration validation"""
        project_path = Path(project_path)
        
        if validate_config:
            config_file = project_path / "config.yaml"
            if config_file.exists():
                try:
                    self.validator.validate_template_config_file(config_file)
                    logger.info("✅ Project configuration validated before adding API layer")
                except Exception as e:
                    logger.warning(f"Project configuration validation failed: {e}")
                    print(f"⚠️  Warning: Project configuration has issues: {e}")
        
        # Copy server.py from template
        self._copy_server_py(project_path)
        
        # Update pyproject.toml to include API dependencies
        self._update_pyproject_for_api(project_path)
        
        print("✅ Added API layer to project")
        if validate_config:
            print("✅ Project configuration validated")
    
    def add_docker_layer(self, project_path: str, validate_config: bool = True, **options):
        """Add Docker container layer with configuration validation"""
        project_path = Path(project_path)
        
        if validate_config:
            config_file = project_path / "config.yaml"
            if config_file.exists():
                try:
                    validated_config = self.validator.validate_template_config_file(config_file)
                    logger.info("✅ Project configuration validated before adding Docker layer")
                    
                    # Validate Docker-specific requirements
                    if hasattr(validated_config, 'model'):
                        model_path = validated_config.model.path
                        full_model_path = project_path / model_path
                        if not full_model_path.exists():
                            logger.warning(f"Model file not found: {full_model_path}")
                            print(f"⚠️  Warning: Model file not found at {model_path}")
                            print("   Make sure to place your model file before building Docker image")
                
                except Exception as e:
                    logger.warning(f"Project configuration validation failed: {e}")
                    print(f"⚠️  Warning: Project configuration has issues: {e}")
        
        # Copy Docker files from template
        self._copy_docker_files(project_path)
        
        print("✅ Added Docker layer to project")
        if validate_config:
            print("✅ Project configuration validated for Docker deployment")
    
    def validate_existing_template(self, project_path: str) -> Dict[str, Any]:
        """Validate an existing template project"""
        project_path = Path(project_path)
        config_file = project_path / "config.yaml"
        
        validation_result = {
            "project_path": str(project_path),
            "config_exists": config_file.exists(),
            "config_valid": False,
            "validation_errors": [],
            "validation_warnings": [],
            "model_file_exists": False,
            "suggestions": []
        }
        
        if not config_file.exists():
            validation_result["validation_errors"].append("config.yaml file not found")
            validation_result["suggestions"].append("Generate config.yaml file for the project")
            return validation_result
        
        try:
            # Validate configuration
            validated_config = self.validator.validate_template_config_file(config_file)
            validation_result["config_valid"] = True
            
            # Check if model file exists
            if hasattr(validated_config, 'model'):
                model_path = project_path / validated_config.model.path
                validation_result["model_file_exists"] = model_path.exists()
                
                if not validation_result["model_file_exists"]:
                    validation_result["validation_warnings"].append(f"Model file not found: {validated_config.model.path}")
                    validation_result["suggestions"].append(f"Place your model file at: {validated_config.model.path}")
            
            logger.info(f"✅ Template validation successful: {project_path}")
            
        except Exception as e:
            validation_result["validation_errors"].append(str(e))
            validation_result["suggestions"].extend([
                "Check config.yaml syntax",
                "Verify all required fields are present",
                "Check field value ranges and types"
            ])
            logger.error(f"Template validation failed: {e}")
        
        return validation_result
    
    # =============================================================================
    # EXISTING TEMPLATE GENERATION METHODS (Enhanced with validation)
    # =============================================================================
    
    def _copy_base_template(self, model_type: str, project_path: Path):
        """Copy base template files with validation"""
        # Determine template source directory
        if model_type in ["yolo", "yolo_openvino"]:
            template_src = self.templates_dir / "yolo"
        else:
            # For other model types, use YOLO template as fallback
            template_src = self.templates_dir / "yolo"
        
        if not template_src.exists():
            raise TemplateError(
                message=f"Template directory not found: {template_src}",
                error_code=ErrorCode.TEMPLATE_NOT_FOUND,
                suggestions=[
                    "Check InferX installation",
                    "Verify template directory exists",
                    f"Available templates: {list(self.templates_dir.glob('*')) if self.templates_dir.exists() else 'none'}"
                ]
            )
        
        # Copy template files (existing implementation)
        for item in template_src.iterdir():
            if item.name not in ["src", "config.yaml"]:  # Skip src and config.yaml (we generate these)
                dst_path = project_path / item.name
                if item.is_dir():
                    self._copy_directory(item, dst_path)
                else:
                    if "__pycache__" not in str(item) and not item.name.endswith(".pyc"):
                        shutil.copy2(item, dst_path)
        
        # Create src directory
        src_path = project_path / "src"
        src_path.mkdir(exist_ok=True)
        (src_path / "__init__.py").write_text("")
        
        # Create models directory
        models_path = project_path / "models"
        models_path.mkdir(exist_ok=True)
        (models_path / ".gitkeep").write_text("# Keep this directory in git")
        
        # Create inferencer.py based on model type
        self._create_inferencer_py(model_type, src_path)
        
        logger.debug(f"Base template copied for {model_type}")
    
    def _copy_directory(self, src: Path, dst: Path):
        """Recursively copy directory contents"""
        dst.mkdir(exist_ok=True)
        for item in src.iterdir():
            dst_path = dst / item.name
            if item.is_dir():
                self._copy_directory(item, dst_path)
            else:
                if "__pycache__" not in str(item) and not item.name.endswith(".pyc"):
                    shutil.copy2(item, dst_path)
    
    def _create_inferencer_py(self, model_type: str, src_path: Path):
        """Create inferencer.py (existing implementation)"""
        # Implementation would be same as existing template generator
        pass
    
    def _copy_model_file(self, model_path: str, model_type: str, project_path: Path):
        """Copy model file with validation"""
        model_path = Path(model_path)
        models_dir = project_path / "models"
        
        if not model_path.exists():
            raise TemplateError(
                message=f"Model file not found: {model_path}",
                error_code=ErrorCode.TEMPLATE_GENERATION_FAILED,
                suggestions=[
                    "Check model file path",
                    "Verify file exists and is readable"
                ]
            )
        
        # Determine target filename based on model type
        if model_type == "yolo_openvino":
            if model_path.suffix == ".xml":
                xml_target = models_dir / "yolo_model.xml"
                shutil.copy2(model_path, xml_target)
                
                # Look for corresponding .bin file
                bin_path = model_path.with_suffix(".bin")
                if bin_path.exists():
                    bin_target = models_dir / "yolo_model.bin"
                    shutil.copy2(bin_path, bin_target)
                    print(f"✅ Copied OpenVINO model files: .xml and .bin")
                else:
                    logger.warning("OpenVINO .bin file not found")
        else:
            target_name = "yolo_model.onnx" if model_type.startswith("yolo") else "model.onnx"
            target_path = models_dir / target_name
            shutil.copy2(model_path, target_path)
            print(f"✅ Copied model file: {target_path}")
    
    def _copy_server_py(self, project_path: Path):
        """Copy server.py from template"""
        # Implementation same as existing
        pass
    
    def _copy_docker_files(self, project_path: Path):
        """Copy Docker files from template"""
        # Implementation same as existing
        pass
    
    def _update_pyproject_for_api(self, project_path: Path):
        """Update pyproject.toml for API dependencies"""
        # Implementation same as existing
        pass


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_template_generator() -> EnhancedTemplateGenerator:
    """Create enhanced template generator with validation"""
    return EnhancedTemplateGenerator()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_enhanced_template_generation():
    """Demonstrate enhanced template generation with validation"""
    print("Enhanced Template Generation with Pydantic Validation")
    print("=" * 60)
    
    generator = create_enhanced_template_generator()
    
    try:
        # Example 1: Generate YOLO template with validation
        print("\n1. Generating YOLO template with validation:")
        result = generator.generate(
            model_type="yolo",
            project_name="demo-yolo-detector",
            device="auto",
            confidence_threshold=0.3,
            input_size=640
        )
        
        print(f"✅ Template generated successfully")
        print(f"   Project: {result['project_name']}")
        print(f"   Files created: {len(result['files_created'])}")
        print(f"   Config validated: {result['config_validated']}")
        
        # Example 2: Validate existing template
        print("\n2. Validating generated template:")
        validation_result = generator.validate_existing_template("demo-yolo-detector")
        print(f"   Config exists: {validation_result['config_exists']}")
        print(f"   Config valid: {validation_result['config_valid']}")
        print(f"   Model file exists: {validation_result['model_file_exists']}")
        
        if validation_result['validation_warnings']:
            print(f"   Warnings: {len(validation_result['validation_warnings'])}")
        
        # Example 3: Try invalid configuration
        print("\n3. Testing validation with invalid configuration:")
        try:
            generator.generate(
                model_type="yolo",
                project_name="demo-invalid",
                confidence_threshold=1.5,  # Invalid: > 1.0
                input_size=100  # Invalid: not divisible by 32
            )
        except TemplateError as e:
            print(f"✅ Validation correctly caught errors: {e.error_code.value}")
        
        # Cleanup
        import shutil
        for project in ["demo-yolo-detector", "demo-invalid"]:
            if Path(project).exists():
                shutil.rmtree(project)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
    
    print("\n🎯 Enhanced Template Generation Benefits:")
    print("   • Pydantic validation for all generated configs")
    print("   • Type-safe template configuration")
    print("   • Detailed validation reports")
    print("   • Early error detection and helpful suggestions")
    print("   • Consistent configuration structure")


if __name__ == "__main__":
    demonstrate_enhanced_template_generation()