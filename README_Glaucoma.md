# Glaucoma Detection System

This application has been adapted from the Pick and Place machine project to perform glaucoma detection using computer vision and AI. The system supports both real-time camera analysis and static image analysis using YOLOv11, providing comprehensive glaucoma detection results along with patient and doctor information management.

## Features

### 🏥 Medical Information Management
- **Patient Information Form**: Capture patient details including ID, name, age, gender, phone, and medical history
- **Doctor Information Form**: Record doctor credentials including ID, name, specialization, and hospital/clinic
- **Form Validation**: Ensures all required fields are completed before analysis

### 📷 Eye Imaging System
- **Dual Mode Operation**: Switch between camera stream and image import modes
- **Live Video Feed**: Real-time camera display with detection overlay
- **Camera Selection**: Choose from available cameras with refresh functionality
- **Image Import**: Import eye images from files (JPG, PNG, BMP, TIFF formats)
- **Image Capture**: Capture and analyze specific moments for diagnosis

### 🔍 AI-Powered Detection
- **YOLOv11 Integration**: Uses latest YOLOv11 model for enhanced glaucoma detection
- **Trained Model Support**: Automatically loads your custom trained glaucoma model
- **Live Detection Mode**: Real-time detection with bounding boxes and confidence scores
- **Static Image Analysis**: Analyze imported eye images for glaucoma indicators
- **Intelligent Detection**: Recognizes optic disc, cup, and glaucoma-specific features
- **Result Visualization**: Annotated images with detection results and confidence scores

### 📊 Results Management
- **Detection Results**: Clear indication of glaucoma presence with confidence scores
- **Image Storage**: Automatic saving of analyzed images with timestamps
- **JSON Reports**: Comprehensive reports combining patient, doctor, and detection data
- **Server Integration**: Framework for sending results to external servers

### 🖥️ User Interface
- **Three-Column Layout**: 
  - Left: Patient Information
  - Center: Detection Mode Selection and Controls
  - Right: Doctor Information and Results
- **Mode Selection**: Radio buttons to switch between camera and image modes
- **Adaptive UI**: Interface adapts based on selected detection mode
- **Real-time Logging**: System activity log with timestamps
- **Form Management**: Clear forms functionality for new patients

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or USB camera (for camera mode)
- Eye images in supported formats (for image mode)

### Dependencies

#### For GPU Acceleration (Recommended)
```bash
# Install base requirements
pip install -r requirements_glaucoma.txt

# Replace with CUDA-enabled PyTorch for GPU acceleration
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### For CPU-Only Systems
```bash
# Install requirements with CPU-only PyTorch
pip install -r requirements_glaucoma.txt

# Key dependencies include:
# - ultralytics>=8.0.0 (YOLOv11 support)
# - torch>=2.0.0 (with CUDA support)
# - opencv-python>=4.5.0
# - pillow>=8.3.0
# - tkinter (built-in with Python)
```

### Model Setup
The application automatically detects and loads your trained model:

1. **Trained Model Path**: `model_results/runs/train/glaucoma_yolo11/weights/best.pt`
2. **Fallback Model**: Downloads YOLOv11n if trained model not found
3. **Model Format**: Uses Ultralytics YOLOv11 format (.pt files)

**Training Your Model**: Use the provided `yolov11_glaucoma_detection.ipynb` notebook to train your custom glaucoma detection model.

## Usage

### 1. Starting the Application
```bash
python GlaucomaCam.py
```

### 2. Setup Process
1. **Fill Patient Information**: Complete all required patient details
2. **Fill Doctor Information**: Enter doctor credentials and information
3. **Choose Detection Mode**: Select either "Camera Stream" or "Import Image"

### 3. Detection Modes

#### 📷 Camera Stream Mode
1. **Select Camera**: Choose your camera from the dropdown and click "Apply Camera"
2. **Position Patient**: Ensure proper eye positioning in camera view
3. **Start Live Detection**: Click "Start Live Detection" for real-time monitoring
4. **Capture & Analyze**: When ready, click "Capture & Analyze" for detailed examination

#### 🖼️ Image Import Mode
1. **Import Image**: Click "Import Image" and select an eye image file
2. **Review Image**: Verify the imported image is displayed correctly
3. **Analyze Image**: Click "Analyze Image" to run glaucoma detection
4. **View Results**: Results will be overlaid on the imported image

### 4. Results Review
1. **Detection Outcome**: Check the results panel for glaucoma detection status
2. **Confidence Scores**: Review detection confidence levels
3. **Visual Annotations**: Examine bounding boxes and labels on the image

### 5. Data Management
1. **Review Results**: Examine the detection confidence and findings
2. **Send to Server**: Click "Send to Server" to save/transmit results
3. **Clear Forms**: Use clear buttons to prepare for next patient

## File Structure

```
GlaucomaCam.py                    # Main application file
yolov11_glaucoma_detection.ipynb  # Training notebook
requirements_glaucoma.txt         # Python dependencies
model_results/                    # Training results and models
├── runs/train/glaucoma_yolo11/   # Trained model directory
│   └── weights/best.pt           # Best trained model
├── yolo11n.pt                    # Base YOLOv11 model
results/                          # Analysis results
├── glaucoma_result_*.jpg         # Analyzed images with annotations
├── glaucoma_report_*.json        # Complete examination reports
glaucoma/dataset/                 # Training dataset
├── train/images/                 # Training images
├── train/labels/                 # Training labels
├── val/images/                   # Validation images
├── val/labels/                   # Validation labels
├── test/images/                  # Test images
└── test/labels/                  # Test labels
```

## Output Files

### Images
- **Filename Format**: `glaucoma_result_YYYYMMDD_HHMMSS.jpg`
- **Content**: Original eye image with AI detection annotations
- **Annotations**: Bounding boxes, confidence scores, class labels

### JSON Reports
- **Filename Format**: `glaucoma_report_YYYYMMDD_HHMMSS.json`
- **Content Structure**:
```json
{
  "patient_info": {
    "patient_id": "P001",
    "name": "John Doe",
    "age": "45",
    "gender": "Male",
    "phone": "+1234567890",
    "medical_history": "No previous eye conditions",
    "examination_date": "2024-01-15T10:30:00"
  },
  "doctor_info": {
    "doctor_id": "D001",
    "name": "Dr. Smith",
    "specialization": "Ophthalmologist",
    "hospital": "City Eye Hospital"
  },
  "detection_result": {
    "has_glaucoma": true,
    "confidence": 0.85,
    "detected_features": [...],
    "analysis_timestamp": "2024-01-15T10:35:00"
  },
  "image_path": "results/glaucoma_result_20240115_103500.jpg"
}
```

## Server Integration

The application includes a framework for server communication. To implement:

1. **Update Server Endpoint**: Modify the server URL in `send_to_server()` method
2. **Authentication**: Add necessary API keys or authentication tokens
3. **Error Handling**: Implement proper error handling for network issues

```python
# Example server integration
response = requests.post(
    'http://your-server.com/api/glaucoma-results',
    json=data_package,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

## Model Training

### Using the Provided Notebook

1. **Open Training Notebook**: Launch `yolov11_glaucoma_detection.ipynb`
2. **Dataset Setup**: The notebook uses the included glaucoma dataset
3. **Training Configuration**: YOLOv11n model with 100 epochs, optimized for Jetson Nano
4. **Model Export**: Automatically saves to `model_results/runs/train/glaucoma_yolo11/weights/`

### Training Your Own Model

1. **Dataset Preparation**: 
   - Organize images in `glaucoma/dataset/` structure
   - Create YOLO format annotations (.txt files)
   - Update `data/glaucoma.yaml` with your dataset paths

2. **Training Process**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n.pt')
   results = model.train(data='data/glaucoma.yaml', epochs=100, imgsz=416)
   ```

3. **Model Integration**: The application automatically detects trained models

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera connections
   - Try different camera indices
   - Restart application

2. **Model Loading Failed**
   - Verify model file path
   - Check CUDA availability for GPU inference
   - Ensure model compatibility

3. **Detection Not Working**
   - Verify lighting conditions
   - Check eye positioning
   - Adjust confidence thresholds

4. **Server Communication Errors**
   - Check network connectivity
   - Verify server endpoint
   - Review authentication credentials

## Security and Privacy

- **Data Protection**: Ensure patient data is handled according to medical privacy regulations
- **Secure Transmission**: Use HTTPS for server communication
- **Local Storage**: Implement proper access controls for saved files
- **HIPAA Compliance**: Follow medical data handling guidelines

## Recent Updates (YOLOv11 Migration)

### ✨ New Features
- **Dual Detection Modes**: Switch between camera stream and image import
- **YOLOv11 Integration**: Upgraded from YOLOv7 to latest YOLOv11
- **Enhanced UI**: Mode-specific controls and adaptive interface
- **Improved Detection**: Better accuracy for optic disc and cup detection
- **File Import Support**: JPG, PNG, BMP, TIFF image formats
- **Auto Model Loading**: Automatically detects and loads trained models

### 🔧 Technical Improvements
- **Ultralytics Framework**: Modern YOLO implementation
- **Better Error Handling**: Improved robustness and user feedback
- **Optimized Inference**: Faster processing with YOLOv11 architecture
- **Updated Dependencies**: Latest package versions for better compatibility

## Performance

### YOLOv11 Advantages
- **67% faster inference** than YOLOv8 on edge devices
- **Superior accuracy** for small object detection (ideal for medical imaging)
- **Native Jetson support** with optimized performance
- **Anchor-free architecture** for better generalization

### GPU vs CPU Performance
- **GPU (CUDA)**: 10-50x faster inference, recommended for real-time detection
- **CPU**: Slower but sufficient for occasional analysis
- **Memory**: GPU requires ~2GB VRAM for YOLOv11n model

### Hardware Recommendations
- **GPU**: NVIDIA GTX 1050+ (4GB VRAM) or Jetson Nano/Xavier
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and results
- **CUDA**: Version 11.8+ compatible drivers

## License

This project is a specialized glaucoma detection system using YOLOv11, optimized for deployment on edge devices including Jetson Nano.

## Support

For technical support or questions about the glaucoma detection system, please refer to the original project documentation or contact the development team.

---

**Note**: This system is for research and development purposes. Any medical applications should be validated by qualified medical professionals and comply with appropriate medical device regulations. 