#!/usr/bin/env python3
"""Simple test runner for InferX tests"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_filter=None, markers=None, verbose=False):
    """Run tests with optional filtering"""
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent / "tests"
    cmd.append(str(test_dir))
    
    # Add markers if specified
    if markers:
        if isinstance(markers, str):
            markers = [markers]
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add test filter if specified
    if test_filter:
        cmd.extend(["-k", test_filter])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=inferx", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    # Add color output
    cmd.append("--color=yes")
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run InferX tests")
    parser.add_argument("-k", "--filter", help="Filter tests by pattern")
    parser.add_argument("-m", "--markers", nargs="*", help="Run tests with specific markers")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--config", action="store_true", help="Run only config tests")
    parser.add_argument("--openvino", action="store_true", help="Run only OpenVINO tests")
    parser.add_argument("--cli", action="store_true", help="Run only CLI tests")
    
    args = parser.parse_args()
    
    # Build markers list
    markers = args.markers or []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.config:
        markers.append("config")
    if args.openvino:
        markers.append("openvino")
    if args.cli:
        markers.append("cli")
    
    # Run tests
    result = run_tests(
        test_filter=args.filter,
        markers=markers,
        verbose=args.verbose
    )
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()