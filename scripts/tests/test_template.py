#!/usr/bin/env python3
"""Test script for template generator"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'inferx'))

from inferx.generators.template import TemplateGenerator

def test_template_generation():
    """Test template generation"""
    print("Testing template generation...")
    
    generator = TemplateGenerator()
    try:
        generator.generate("yolo", "test-template-project")
        print("✅ Template generation successful")
        
        # Check if files were created
        expected_files = [
            "test-template-project/pyproject.toml",
            "test-template-project/config.yaml", 
            "test-template-project/README.md",
            "test-template-project/src/__init__.py",
            "test-template-project/src/inferencer.py"
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
                
    except Exception as e:
        print(f"❌ Template generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_template_generation()