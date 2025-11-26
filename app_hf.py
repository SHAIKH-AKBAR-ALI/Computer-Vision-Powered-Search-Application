import streamlit as st
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import base64
from pathlib import Path

# Import your modules
from src.vision_search.config import load_config
from src.vision_search.inference import YOLOv11Inference
from src.vision_search.utils import save_metadata, load_metadata, get_unique_classes_counts

# Load config
try:
    CONFIG = load_config('config/default.yaml')
    MODEL_CONFIG = CONFIG['model']
    DATA_CONFIG = CONFIG['data']
except FileNotFoundError:
    st.error("Config file not found!")
    st.stop()

# Page config
st.set_page_config(page_title="Vision Search", layout="wide")
st.title("🔍 Vision Search - Object Detection")

# Sidebar
st.sidebar.title("📁 Image Input")
st.sidebar.subheader("📤 Upload File")
uploaded_file = st.sidebar.file_uploader("Choose image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    st.sidebar.success("✅ File uploaded!")
    
    if st.sidebar.button("🔍 Analyze Image"):
        try:
            # Load model
            with st.spinner("Loading AI model..."):
                inferencer = YOLOv11Inference(
                    model_path='yolo11m.pt',
                    conf_threshold=0.25,
                    image_extensions=['.jpg', '.jpeg', '.png']
                )
            
            # Process image
            with st.spinner("Analyzing image..."):
                result = inferencer.process_single_image(temp_path)
            
            if result and result.get('detections'):
                # Display results
                st.success(f"✨ Found {len(result['detections'])} objects!")
                
                # Show image with boxes
                img = Image.open(temp_path)
                draw = ImageDraw.Draw(img)
                
                for det in result['detections']:
                    bbox = det['bbox']
                    draw.rectangle(bbox, outline="red", width=2)
                    draw.text((bbox[0], bbox[1]-15), f"{det['class']} {det['confidence']:.2f}", fill="red")
                
                st.image(img, caption="Detection Results", use_column_width=True)
                
                # Show detection details
                st.subheader("📊 Detection Details")
                for det in result['detections']:
                    st.write(f"- **{det['class']}**: {det['confidence']:.2f} confidence")
            else:
                st.info("No objects detected in the image.")
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Instructions
if not uploaded_file:
    st.markdown("""
    ## 🚀 How to Use:
    
    1. Click "Choose image" button in the sidebar
    2. Select your image file (JPG, PNG, BMP supported)
    3. Click "🔍 Analyze Image"
    4. View detection results with bounding boxes and confidence scores
    
    **Powered by YOLOv11 for accurate object detection!**
    """)