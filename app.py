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

def img_to_base64(image):
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Page config
st.set_page_config(page_title="Vision Search", layout="wide")
st.title("🔍 Vision Search - Object Detection")

# Sidebar
st.sidebar.title("📁 Image Input")
option = st.sidebar.radio("Choose method:", ["Upload Image", "Enter Path"])

# Initialize variables
uploaded_file = None
image_path = None

if option == "Upload Image":
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
                
                st.write(f"Debug: Result = {result}")
                
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
                
                # Cleanup
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

else:
    st.sidebar.subheader("📝 Enter Path")
    image_path = st.sidebar.text_input("Image path:", placeholder="D:\\images\\photo.jpg")
    
    if st.sidebar.button("🔍 Analyze Image") and image_path:
        if os.path.exists(image_path):
            try:
                # Load model and process
                with st.spinner("Processing..."):
                    inferencer = YOLOv11Inference(
                        model_path='yolo11m.pt',
                        conf_threshold=0.25,
                        image_extensions=['.jpg', '.jpeg', '.png']
                    )
                    result = inferencer.process_single_image(image_path)
                
                st.write(f"Debug: Result = {result}")
                
                if result and result.get('detections'):
                    st.success(f"✨ Found {len(result['detections'])} objects!")
                    
                    # Show image with boxes
                    img = Image.open(image_path)
                    draw = ImageDraw.Draw(img)
                    
                    for det in result['detections']:
                        bbox = det['bbox']
                        draw.rectangle(bbox, outline="red", width=2)
                        draw.text((bbox[0], bbox[1]-15), f"{det['class']} {det['confidence']:.2f}", fill="red")
                    
                    st.image(img, caption="Detection Results", use_column_width=True)
                    
                    # Show details
                    st.subheader("📊 Detection Details")
                    for det in result['detections']:
                        st.write(f"- **{det['class']}**: {det['confidence']:.2f} confidence")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("File not found!")

# Instructions
if not uploaded_file and not image_path:
    st.markdown("""
    ## 🚀 How to Use:
    
    ### Method 1: Upload Image (Recommended)
    1. Click "Upload Image" in sidebar
    2. Click "Choose image" button
    3. Select your image file
    4. Click "🔍 Analyze Image"
    
    ### Method 2: Enter Path
    1. Click "Enter Path" in sidebar  
    2. Type full path like: `D:\\Pictures\\photo.jpg`
    3. Click "🔍 Analyze Image"
    
    **Supported formats:** JPG, JPEG, PNG, BMP
    """)