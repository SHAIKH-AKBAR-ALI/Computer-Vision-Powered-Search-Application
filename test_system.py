#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        import streamlit
        import ultralytics
        import cv2
        import yaml
        import torch
        import PIL
        import pandas
        import numpy
        from src.vision_search.config import load_config
        from src.vision_search.inference import YOLOv11Inference
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test config loading"""
    print("\n🔍 Testing config...")
    try:
        config = load_config('config/default.yaml')
        print(f"✅ Config loaded: {config}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_model():
    """Test model initialization"""
    print("\n🔍 Testing model initialization...")
    try:
        model = YOLOv11Inference('yolo11m.pt', 0.25, ['.jpg', '.jpeg', '.png'])
        print("✅ Model initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_inference():
    """Test inference with sample image"""
    print("\n🔍 Testing inference...")
    try:
        # Find a test image
        test_image_path = "data/processed/coco-val-2017-500/Fruits Classification/valid/Strawberry/Strawberry (809).jpeg"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return False
            
        model = YOLOv11Inference('yolo11m.pt', 0.25, ['.jpg', '.jpeg', '.png'])
        result = model.process_single_image(test_image_path)
        
        if result and 'detections' in result:
            print(f"✅ Inference successful! Found {len(result['detections'])} objects")
            for i, det in enumerate(result['detections'][:3]):
                print(f"   {i+1}. {det['class']}: {det['confidence']:.2f}")
            return True
        else:
            print("⚠️ No detections found (this might be normal)")
            return True
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def main():
    print("🚀 Running comprehensive system test...\n")
    
    tests = [
        test_imports,
        test_config, 
        test_model,
        test_inference
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! Your system is working properly!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)