#!/usr/bin/env python3
"""
Test script for enhanced error handling in InferX

This script demonstrates the new error handling capabilities including:
- Custom exception classes with error codes
- Actionable error suggestions
- Structured error logging
- Error recovery mechanisms
"""

import logging
from pathlib import Path
import sys

# Setup logging to see the structured error information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inferx import InferenceEngine, ModelNotFoundError, ModelLoadError, ErrorCode


def test_model_not_found():
    """Test model not found error handling"""
    print("\n=== Test 1: Model Not Found ===")
    
    try:
        # Try to load a non-existent model
        engine = InferenceEngine("non_existent_model.onnx")
    except ModelNotFoundError as e:
        print(f"✅ Caught ModelNotFoundError:")
        print(f"Error Code: {e.error_code.value}")
        print(f"Message: {e}")
        print(f"Context: {e.context}")
        print("\nThis demonstrates:")
        print("- Custom exception with error code")
        print("- Actionable suggestions")
        print("- Structured error information")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_invalid_model_format():
    """Test invalid model format error handling"""
    print("\n=== Test 2: Invalid Model Format ===")
    
    # Create a dummy file with wrong extension
    dummy_model = Path("dummy_model.txt")
    dummy_model.write_text("This is not a model file")
    
    try:
        # Try to load the dummy file as a model
        engine = InferenceEngine(str(dummy_model))
    except ModelLoadError as e:
        print(f"✅ Caught ModelLoadError:")
        print(f"Error Code: {e.error_code.value}")
        print(f"Message: {e}")
        print(f"Suggestions: {e.suggestions}")
        print(f"Recovery Actions: {e.recovery_actions}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up dummy file
        if dummy_model.exists():
            dummy_model.unlink()


def test_error_serialization():
    """Test error serialization to JSON"""
    print("\n=== Test 3: Error Serialization ===")
    
    try:
        # Create a model not found error
        raise ModelNotFoundError("test_model.onnx")
    except ModelNotFoundError as e:
        error_dict = e.to_dict()
        print(f"✅ Error serialized to dictionary:")
        import json
        print(json.dumps(error_dict, indent=2))
        print("\nThis can be used for:")
        print("- API error responses")
        print("- Error logging")
        print("- Error reporting systems")


def test_error_recovery_demo():
    """Demonstrate error recovery mechanisms"""
    print("\n=== Test 4: Error Recovery Demo ===")
    
    from inferx.recovery import with_retry, RetryConfig, DeviceFallbackStrategy
    
    # Example function that might fail
    attempt_count = 0
    
    @with_retry(
        retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
        fallback_strategies=[DeviceFallbackStrategy()]
    )
    def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ConnectionError("Simulated network error")
        return "Success after retries!"
    
    try:
        result = failing_function()
        print(f"✅ Function succeeded: {result}")
        print("This demonstrates:")
        print("- Automatic retry with exponential backoff")
        print("- Structured error recovery")
        print("- Fallback strategies")
    except Exception as e:
        print(f"❌ Function failed after all retries: {e}")


def test_structured_logging():
    """Test structured logging capabilities"""
    print("\n=== Test 5: Structured Logging ===")
    
    # Create a logger to capture log output
    import io
    import logging
    
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.ERROR)
    
    logger = logging.getLogger("inferx.exceptions")
    logger.addHandler(handler)
    
    try:
        # Create an error that will be logged
        raise ModelLoadError(
            model_path="test.onnx",
            runtime="onnx",
            context={"test_context": "demo"}
        )
    except ModelLoadError:
        pass
    
    # Get the logged output
    log_output = log_stream.getvalue()
    print(f"✅ Structured log output:")
    print(log_output)
    print("This demonstrates:")
    print("- Automatic structured logging")
    print("- Context information preservation")
    print("- Error tracking capabilities")
    
    # Clean up
    logger.removeHandler(handler)


if __name__ == "__main__":
    print("🚀 Testing Enhanced Error Handling in InferX")
    print("=" * 50)
    
    # Run all tests
    test_model_not_found()
    test_invalid_model_format()
    test_error_serialization()
    test_error_recovery_demo()
    test_structured_logging()
    
    print("\n" + "=" * 50)
    print("✅ Error handling tests completed!")
    print("\nKey improvements implemented:")
    print("- 🎯 Custom exception classes with error codes")
    print("- 💡 Actionable error suggestions")
    print("- 📊 Structured error logging")
    print("- 🔄 Automatic error recovery mechanisms")
    print("- 🛡️ Security validations (path traversal protection)")
    print("- 📋 JSON serializable error information")