# InferX Test Suite

This directory contains the test suite for InferX, covering all major components including the new OpenVINO integration and configuration system.

## 🏗️ Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_base_inferencer.py        # Base inferencer tests
├── test_cli.py                    # Original CLI tests
├── test_cli_enhanced.py          # Enhanced CLI tests (OpenVINO + config) 🆕
├── test_config.py                # Configuration system tests 🆕
├── test_openvino_inferencer.py   # OpenVINO inferencer tests 🆕
├── test_runtime.py               # Original runtime tests
├── test_runtime_enhanced.py      # Enhanced runtime tests 🆕
├── test_utils.py                 # Utility function tests
├── test_yolo.py                  # YOLO inferencer tests
└── README.md                     # This file
```

## 🎯 Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests requiring external dependencies
- `@pytest.mark.config` - Configuration system tests 🆕
- `@pytest.mark.openvino` - OpenVINO-specific tests 🆕
- `@pytest.mark.cli` - CLI functionality tests
- `@pytest.mark.slow` - Tests that take longer to run

## 🚀 Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
python test_runner.py

# Run specific test categories
python test_runner.py --unit              # Unit tests only
python test_runner.py --integration       # Integration tests only
python test_runner.py --config           # Configuration tests only
python test_runner.py --openvino         # OpenVINO tests only
python test_runner.py --cli              # CLI tests only

# Run with verbose output
python test_runner.py -v

# Filter tests by name pattern
python test_runner.py -k "test_config"

# Run multiple markers
python test_runner.py -m unit config
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=inferx --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with markers
pytest tests/ -m "config"
pytest tests/ -m "unit and not slow"

# Run with verbose output
pytest tests/ -v
```

## 🧪 Test Coverage

### Core Components
- ✅ **Configuration System**: Hierarchical config loading, validation, model detection
- ✅ **OpenVINO Integration**: Model loading, inference, device optimization
- ✅ **YOLO OpenVINO**: YOLO-specific OpenVINO optimizations
- ✅ **Runtime Engine**: Enhanced model type detection, cross-runtime support
- ✅ **CLI Commands**: OpenVINO device selection, config management commands

### New Test Features 🆕
- **Mock OpenVINO Models**: Test fixtures for .xml/.bin model files
- **Configuration Testing**: Hierarchical config loading and validation
- **Cross-Runtime Testing**: ONNX models on OpenVINO runtime
- **Device Mapping**: Device name mapping and validation
- **CLI Integration**: Config commands, OpenVINO devices, error handling

## 🔧 Test Fixtures

### Model Fixtures
```python
mock_onnx_model_path         # Mock ONNX model file (.onnx)
mock_openvino_model_path     # Mock OpenVINO model files (.xml/.bin)
mock_yolo_onnx_path         # Mock YOLO ONNX model
mock_yolo_openvino_path     # Mock YOLO OpenVINO model
```

### Configuration Fixtures
```python
mock_config                 # Basic configuration dictionary
mock_openvino_config       # OpenVINO-specific configuration
mock_yaml_config           # YAML configuration file
mock_inferx_config         # Full InferX configuration file
```

### Data Fixtures
```python
mock_image_path            # Single test image
batch_image_paths          # Multiple test images for batch testing
test_image_array          # Numpy array image data
mock_inference_result     # Mock inference result structure
```

## 📝 Writing New Tests

### Test Naming Convention
- `test_[component]_[functionality].py` for test files
- `test_[specific_feature]` for test methods
- Use descriptive names that explain what is being tested

### Example Test Structure
```python
@pytest.mark.unit
@pytest.mark.config
class TestConfigFeature:
    """Test configuration feature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        # Act  
        # Assert
        
    def test_error_handling(self):
        """Test error handling"""
        # Test error conditions
```

### Using Mocks
```python
# Mock external dependencies
@patch('openvino.Core')
def test_openvino_functionality(self, mock_ov_core):
    # Setup mock
    mock_core = Mock()
    mock_ov_core.return_value = mock_core
    
    # Test functionality
```

## 🐛 Test Guidelines

### Unit Tests
- Should run fast (< 1 second each)
- Mock external dependencies (OpenVINO, file system)
- Test single components in isolation
- Use `@pytest.mark.unit`

### Integration Tests  
- Can take longer to run
- May use real dependencies (if available)
- Test component interactions
- Use `@pytest.mark.integration`

### Error Testing
- Test error conditions and edge cases
- Verify proper error messages
- Test graceful degradation

### Mocking Strategy
- Mock external libraries (openvino, onnxruntime)
- Use real file system operations with temp directories
- Mock network calls and slow operations

## ⚡ Performance Testing

### Speed Guidelines
- Unit tests: < 1 second each
- Integration tests: < 5 seconds each
- Mark slow tests with `@pytest.mark.slow`

### Memory Testing
- Tests should clean up after themselves
- Use context managers for resources
- Fixtures handle cleanup automatically

## 🔍 Debugging Tests

### Running Single Tests
```bash
# Run specific test
pytest tests/test_config.py::TestInferXConfig::test_model_type_detection -v

# Run with pdb debugger
pytest tests/test_config.py::test_specific_function --pdb
```

### Test Output
```bash
# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Show full diff on assertion failure
pytest tests/ -vv
```

## 🚧 Known Limitations

### OpenVINO Testing
- OpenVINO tests use mocks (real OpenVINO not required for CI)
- Integration tests may require actual OpenVINO installation
- Device-specific tests are mocked (MYRIAD, HDDL testing requires hardware)

### File System Tests
- Use temporary directories for file operations
- Some tests may be platform-specific (Windows paths)

## 🔮 Future Test Additions

### Planned Test Coverage
- [ ] **Anomalib Integration**: When anomaly detection support is added
- [ ] **FastAPI Server**: When server implementation is complete
- [ ] **Docker Generation**: When container generation is implemented
- [ ] **Performance Benchmarks**: Automated performance regression testing
- [ ] **Edge Device Testing**: Raspberry Pi, Jetson specific tests

### Test Infrastructure Improvements
- [ ] **Continuous Integration**: GitHub Actions test automation
- [ ] **Test Reporting**: HTML test reports and coverage dashboards
- [ ] **Performance Monitoring**: Track test execution time trends
- [ ] **Hardware Testing**: Tests with real OpenVINO devices when available

---

*For questions about testing, see the main [CONFIG_GUIDE.md](../CONFIG_GUIDE.md) and project documentation.*