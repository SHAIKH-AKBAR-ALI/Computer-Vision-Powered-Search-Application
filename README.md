# Computer Vision Powered Search Application

A powerful object detection web application built with YOLOv11 and Streamlit that allows users to upload images and get real-time object detection results.

## Features

- 🔍 Real-time object detection using YOLOv11
- 📤 Easy image upload interface
- 🎯 Bounding box visualization
- 📊 Confidence scores for detected objects
- 🌐 Web-based interface accessible from any browser

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SHAIKH-AKBAR-ALI/Computer-Vision-Powered-Search-Application.git
cd Computer-Vision-Powered-Search-Application
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model (if not included):
```bash
# The yolo11m.pt model will be downloaded automatically on first run
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open your web browser and go to `http://localhost:8501`
2. Choose "Upload Image" in the sidebar
3. Select an image file (JPG, PNG, BMP supported)
4. Click "🔍 Analyze Image"
5. View the detection results with bounding boxes and confidence scores

## Project Structure

```
├── app.py                 # Main Streamlit application
├── src/
│   └── vision_search/
│       ├── config.py      # Configuration loader
│       ├── inference.py   # YOLO inference engine
│       └── utils.py       # Utility functions
├── config/
│   └── default.yaml       # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technologies Used

- **YOLOv11**: State-of-the-art object detection
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **PIL**: Image processing

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

### Local Development
```bash
streamlit run app.py
```

## License

MIT License - feel free to use this project for your own applications!

## Contributing

Pull requests are welcome! For major changes, please open an issue first.