#!/usr/bin/env python3
"""
demo.py - Quick Demo Script for Multimodal Crop Health Detection

A simplified version of main.py for quick testing and demonstration.
"""

import os
import sys
from pathlib import Path

def main():
    """Run the multimodal crop health detection demo"""
    
    print("🌾 Multimodal Crop Health Detection - Demo")
    print("=" * 50)
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ Error: main.py not found!")
        print("Please ensure you're in the correct directory.")
        return 1
    
    # Check for required directories
    required_dirs = [
        "yolov8_insect", 
        "yolov8_disease_seg", 
        "tabnet_insect", 
        "tabnet_disease",
        "test_images"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        print("Please ensure all model directories are present.")
        return 1
    
    # Check for test images
    test_images_dir = Path("test_images")
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if len(image_files) < 2:
        print("⚠️  Warning: Less than 2 test images found.")
        print("Consider adding more test images for better demonstration.")
    
    print("✅ All requirements check passed!")
    print("\n🚀 Running main inference pipeline...")
    print("-" * 50)
    
    # Run the main script
    import subprocess
    try:
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=False, 
                              text=True)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running main.py: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\n🎉 Demo completed successfully!")
    else:
        print(f"\n❌ Demo failed with exit code {exit_code}")
    
    sys.exit(exit_code)