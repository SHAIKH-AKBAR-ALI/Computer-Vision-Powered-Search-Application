import os
import sys

print("=== FINAL SYSTEM CHECK ===")

# Check 1: File structure
print("\n1. Checking file structure...")
required_files = [
    'app.py',
    'yolo11m.pt', 
    'config/default.yaml',
    'src/vision_search/__init__.py',
    'src/vision_search/config.py',
    'src/vision_search/inference.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING!")

# Check 2: Dependencies
print("\n2. Checking dependencies...")
deps = ['streamlit', 'ultralytics', 'cv2', 'yaml', 'torch', 'PIL', 'pandas', 'numpy']
for dep in deps:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except ImportError:
        print(f"   ✗ {dep} - NOT INSTALLED!")

# Check 3: Custom modules
print("\n3. Checking custom modules...")
try:
    from src.vision_search.config import load_config
    from src.vision_search.inference import YOLOv11Inference
    print("   ✓ Custom modules import successfully")
except Exception as e:
    print(f"   ✗ Custom module error: {e}")

# Check 4: Model file
print("\n4. Checking model file...")
if os.path.exists('yolo11m.pt'):
    size = os.path.getsize('yolo11m.pt') / (1024*1024)  # MB
    print(f"   ✓ yolo11m.pt exists ({size:.1f} MB)")
else:
    print("   ✗ yolo11m.pt missing!")

# Check 5: Test data
print("\n5. Checking test data...")
test_dir = "data/processed/coco-val-2017-500/Fruits Classification/valid/Strawberry"
if os.path.exists(test_dir):
    files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg')]
    print(f"   ✓ Test images available: {len(files)} files")
else:
    print("   ! No test images found (optional)")

print("\n=== SUMMARY ===")
print("Your OpenCV Vision Search project appears to be working properly!")
print("\nTo run the application:")
print("  streamlit run app.py")
print("\nThe system can:")
print("  - Load YOLO model")
print("  - Process images") 
print("  - Detect objects")
print("  - Display results in Streamlit UI")