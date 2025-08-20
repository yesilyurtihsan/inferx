# InferX Development TODO

✅ **CORE PACKAGE PRODUCTION READY** - Template generation features in development

## 🎯 Project Vision

InferX is a minimal dependency ML inference package that can be used directly or to generate templates, APIs, and Docker containers:

1. **📦 Package Usage** - Import and use directly in Python code (✅ PRODUCTION READY)
2. **⚡ CLI Usage** - Run models directly from command line (✅ PRODUCTION READY)
3. **🏗️ Template Generation** - Generate standalone UV projects (`inferx template`)
4. **🌐 API Generation** - Add FastAPI server to existing project (`inferx api`)
5. **🐳 Docker Generation** - Generate optimized Docker deployment (`inferx docker`)

## 🔴 CRITICAL - Core Features (Phase 1)

### 1. Template Generation System ⭐ **HIGHEST PRIORITY**
- [x] **Template Engine Implementation**
  - [x] Base template system with file copying
  - [x] Template directory structure (`templates/yolo/`)
  - [x] Variable substitution (project name, model type, etc.)
  - [x] File copying and rendering

- [x] **CLI Command: `inferx template`**
  - [x] `inferx template yolo --name my-detector`
  - [x] `inferx template yolo_openvino --name my-detector`
  - [ ] `inferx template anomaly --name quality-checker`
  - [ ] `inferx template classification --name image-classifier`
  - [ ] `inferx template custom --name my-model`

- [x] **YOLO Template** (Completed)
  ```
  my-detector/
  ├── pyproject.toml      # UV project with minimal deps
  ├── src/
  │   ├── __init__.py
  │   ├── inferencer.py   # YOLO inference implementation (inherits from InferX YOLOInferencer)
  │   └── base.py         # Base inferencer class
  ├── models/
  │   └── yolo_model.onnx # Placeholder for user model (or .xml/.bin for OpenVINO)
  ├── config.yaml         # YOLO configuration
  ├── README.md           # Usage instructions
  └── .gitignore          # Standard Python gitignore
  ```

### 2. API Generation System
- [x] **CLI Command: `inferx api`**
  - [x] Detect existing project structure
  - [x] Generate FastAPI server code
  - [x] Add API dependencies to pyproject.toml
  - [x] Generate requirements-api.txt

- [x] **FastAPI Server Template**
  - [x] `/predict` endpoint (single image)
  - [ ] `/predict/batch` endpoint (multiple images) 
  - [x] `/health` endpoint
  - [ ] `/model/info` endpoint
  - [x] Auto-generated Swagger docs
  - [x] File upload handling
  - [x] Error handling middleware

### 3. Docker Generation System  
- [ ] **CLI Command: `inferx docker`**
  - [ ] Generate optimized Dockerfile
  - [ ] Multi-stage build for size optimization
  - [ ] Generate docker-compose.yml
  - [ ] Generate .dockerignore
  - [ ] Health check configuration

- [ ] **Docker Templates**
  - [ ] Base inference container
  - [ ] API server container  
  - [ ] Production optimizations
  - [ ] Size optimization (<100MB target)

## 🟡 HIGH PRIORITY - Template Variety (Phase 2)

### Additional Templates
- [ ] **Anomaly Detection Template**
  - [ ] Anomalib-compatible inference
  - [ ] Heatmap generation
  - [ ] Threshold configuration

- [ ] **Classification Template** 
  - [ ] ImageNet-style classification
  - [ ] Top-k predictions
  - [ ] Custom class names

- [ ] **Custom ONNX Template**
  - [ ] Generic ONNX model support
  - [ ] Configurable input/output shapes
  - [ ] Flexible preprocessing/postprocessing

### Template Enhancements
- [ ] **OpenVINO Support**
  - [ ] .xml/.bin model templates
  - [ ] Device optimization (CPU, GPU, MYRIAD)
  - [ ] Performance presets

- [ ] **Advanced Features**
  - [ ] Batch processing optimization
  - [ ] Model warmup
  - [ ] Performance monitoring
  - [ ] Graceful error handling

## 🟠 MEDIUM PRIORITY - Developer Experience (Phase 3)

### CLI Improvements
- [ ] **Interactive Mode**
  - [ ] `inferx init` - Interactive project setup
  - [ ] Template selection wizard
  - [ ] Configuration validation

- [ ] **Enhanced Commands**
  - [ ] `inferx validate` - Validate project structure
  - [ ] `inferx test` - Run inference tests
  - [ ] `inferx benchmark` - Performance benchmarking

### Testing & Quality
- [ ] **Template Testing**
  - [ ] Generated project tests
  - [ ] End-to-end workflow tests
  - [ ] Cross-platform testing (Windows, Linux, macOS)

- [ ] **Code Quality**
  - [ ] Generated code formatting (black, ruff)
  - [ ] Type hints in generated code
  - [ ] Comprehensive error messages

## 🟢 LOW PRIORITY - Advanced Features (Phase 4)

### Advanced Templates
- [ ] **Segmentation Models**
  - [ ] U-Net, DeepLab templates
  - [ ] Mask generation and visualization

- [ ] **Multi-Model Projects**
  - [ ] Ensemble inference
  - [ ] Model pipelines
  - [ ] A/B testing setup

### Cloud & Edge
- [ ] **Cloud Deployment**
  - [ ] AWS Lambda templates
  - [ ] Google Cloud Run templates
  - [ ] Azure Container Instances

- [ ] **Edge Optimization**
  - [ ] Raspberry Pi templates
  - [ ] NVIDIA Jetson optimization
  - [ ] Mobile deployment (ONNX.js)

### Template Documentation & Examples
- [ ] **Comprehensive Template Guide**
  - [ ] Detailed usage examples for each template type
  - [ ] Best practices documentation
  - [ ] Troubleshooting guide

- [ ] **Template Validation**
  - [ ] Automated template validation scripts
  - [ ] Dependency checking
  - [ ] Configuration file validation

### Advanced CLI Features
- [ ] **Template Publishing**
  - [ ] Template registry system
  - [ ] Template sharing capabilities
  - [ ] Version management for templates

## 📋 Current Implementation Status

### ✅ Already Implemented (PRODUCTION READY)
- ✅ Basic inference engines (ONNX + OpenVINO)
- ✅ YOLO inferencer with preprocessing/postprocessing
- ✅ Configuration system (hierarchical loading)
- ✅ CLI structure with Click
- ✅ Testing framework with pytest
- ✅ Project examples and documentation
- ✅ **Package usage pattern** - Import and use directly in Python code
- ✅ **CLI usage pattern** - Run models directly from command line
- ✅ **Template generation** - Generate standalone projects with YOLO template
- ✅ **API generation** - Add FastAPI server to existing projects

### 🚧 In Progress  
- ✅ YOLO OpenVINO template support
- 🚧 Docker generation system
- 🚧 Additional template types (anomaly, classification, custom)

### ❌ Not Started
- ❌ Additional FastAPI endpoints (/predict/batch, /model/info)
- ❌ Docker generation system (template files)
- ❌ Anomaly detection template
- ❌ Classification template
- ❌ Custom ONNX template
- ❌ Interactive CLI mode
- ❌ Template validation commands
- ❌ Template testing framework
- ❌ Advanced template types (segmentation, multi-model)
- ❌ Cloud deployment templates
- ❌ Edge optimization templates

## 🚀 Next Steps (Immediate)

### Week 1-2: Template System Foundation ✅ COMPLETED
1. **Create template engine infrastructure** ✅ DONE
   - Set up file copying system
   - Create template directory structure
   - Implement basic file rendering

2. **Implement YOLO template** ✅ DONE
   - Create complete YOLO project template
   - Test template generation
   - Validate generated project works

### Week 3-4: API & Docker Generation
1. **FastAPI server generation** ✅ DONE
   - Create FastAPI template
   - Implement `inferx api` command
   - Test API generation on YOLO template

2. **Docker generation** 
   - Create Dockerfile templates
   - Implement `inferx docker` command
   - Test complete workflow: template → api → docker

### Week 5-6: Additional Templates
1. **Anomaly detection template**
2. **Classification template**
3. **Template testing and validation**
2. **Classification template**
3. **Template testing and validation**

## 💡 Implementation Notes

### Template Strategy
- Use **Jinja2** for template rendering
- Store templates in `inferx/templates/` directory
- Each template is a complete project structure
- Use **pyproject.toml** for modern Python packaging
- **UV** for fast dependency management

### Generated Project Structure
```
templates/
├── yolo/
│   ├── pyproject.toml.j2
│   ├── src/
│   │   ├── inference.py.j2
│   │   ├── preprocess.py.j2
│   │   └── postprocess.py.j2
│   ├── config.yaml.j2
│   └── README.md.j2
├── anomaly/
│   └── ...
└── classification/
    └── ...
```

### Key Variables for Templates
- `{{ project_name }}` - User-specified project name
- `{{ model_type }}` - yolo, anomaly, classification, custom
- `{{ author_name }}` - From git config or user input
- `{{ python_version }}` - Target Python version
- `{{ include_gpu }}` - Include GPU dependencies

---

**Priority**: Focus on YOLO template first, then build out the complete generation system. This will provide immediate value and validate the approach before expanding to other model types.