# InferX Scripts & Examples

This directory contains various scripts, examples, and archived implementations.

## Structure

### `/examples/`
Demo and example scripts showing how to use InferX features:
- `pydantic_config_example.py` - Example of Pydantic settings usage
- `config_demo.py` - Configuration system demonstration
- `config_simple_demo.py` - Simple config examples

### `/tests/`
Test scripts and utilities:
- `test_runner.py` - Test runner utilities
- `test_template.py` - Template generation tests  
- `test_error_handling.py` - Error handling tests

### `/old_implementations/`
Archived implementations that were replaced by `settings.py`:
- `config.py.old` - Original config system
- `config_validator.py.old` - Original validation system
- `schemas.py.old` - Original Pydantic schemas
- `hybrid_config_implementation.py` - Hybrid approach attempt
- `runtime_with_validation.py` - Enhanced runtime (replaced)
- `template_generator_with_validation.py` - Enhanced template generator (replaced)

## Usage

These files are kept for reference and examples. The main InferX module uses the clean `settings.py` implementation.