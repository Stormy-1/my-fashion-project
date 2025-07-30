#!/usr/bin/env python3
"""
Test script to check if predict module can be imported
"""

import sys
import os

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

try:
    print("\nTrying to import predict module...")
    from predict import process_image_for_recommendations
    print("✅ SUCCESS: process_image_for_recommendations imported successfully")
    print("Function type:", type(process_image_for_recommendations))
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("This might be due to missing dependencies")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.") 