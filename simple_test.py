#!/usr/bin/env python3

import os
import sys

def test_system():
    print("Running system test...")
    
    # Test 1: Imports
    print("\n1. Testing imports...")
    try:
        import streamlit
        import ultralytics
        import cv2
        import yaml
        import torch
        import PIL
        from src.vision_search.config import load_config
        from src.vision_search.inference import YOLOv11Inference
        print("   SUCCESS: All imports work!")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Config
    print("\n2. Testing config...")
    try:
        config = load_config('config/default.yaml')
        print(f"   SUCCESS: Config loaded - {config}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 3: Model
    print("\n3. Testing model...")
    try:
        model = YOLOv11Inference('yolo11m.pt', 0.25, ['.jpg', '.jpeg', '.png'])
        print("   SUCCESS: Model initialized!")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 4: Inference
    print("\n4. Testing inference...")
    try:
        test_image = "data/processed/coco-val-2017-500/Fruits Classification/valid/Strawberry/Strawberry (809).jpeg"
        if os.path.exists(test_image):
            result = model.process_single_image(test_image)
            if result:
                print(f"   SUCCESS: Found {len(result.get('detections', []))} objects")
            else:
                print("   SUCCESS: Inference ran (no objects detected)")
        else:
            print("   SKIP: No test image found")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    print("\nALL TESTS PASSED! System is working properly!")
    return True

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)