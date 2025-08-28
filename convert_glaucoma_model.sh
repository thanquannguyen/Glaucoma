#!/bin/bash

# ONNX to TensorRT Conversion Script for Glaucoma Detection Model
# This script converts the glaucoma ONNX model to TensorRT format for Jetson Nano

set -e  # Exit on any error

echo "=== Glaucoma Model TensorRT Conversion ==="
echo "This script will convert your ONNX glaucoma model to TensorRT format"
echo ""

# Configuration
ONNX_MODEL="best.onnx"
ENGINE_OUTPUT="best.engine"
PRECISION="fp16"  # Use FP16 for better performance on Jetson Nano
BATCH_SIZE=1
WORKSPACE_MB=512  # Reduced for Jetson Nano memory constraints
INPUT_SHAPE="1 3 640 640"  # Standard YOLO input shape

# Check if model exists
if [ ! -f "$ONNX_MODEL" ]; then
    # Try alternative locations
    if [ -f "model_results/runs/train/glaucoma_yolo11/weights/best.onnx" ]; then
        ONNX_MODEL="model_results/runs/train/glaucoma_yolo11/weights/best.onnx"
        ENGINE_OUTPUT="model_results/runs/train/glaucoma_yolo11/weights/best.engine"
    elif [ -f "model_results/best.onnx" ]; then
        ONNX_MODEL="model_results/best.onnx"
        ENGINE_OUTPUT="model_results/best.engine"
    else
        echo "Error: ONNX model not found!"
        echo "Please ensure one of these files exists:"
        echo "  - best.onnx"
        echo "  - model_results/best.onnx"
        echo "  - model_results/runs/train/glaucoma_yolo11/weights/best.onnx"
        exit 1
    fi
fi

echo "Found ONNX model: $ONNX_MODEL"
echo "Output engine will be: $ENGINE_OUTPUT"
echo ""

# Check if Python conversion script exists
if [ ! -f "convert_onnx_to_tensorrt.py" ]; then
    echo "Error: convert_onnx_to_tensorrt.py not found!"
    echo "Please ensure the conversion script is in the current directory."
    exit 1
fi

# Check TensorRT installation
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" 2>/dev/null || {
    echo "Error: TensorRT not found or not properly installed!"
    echo "Please ensure TensorRT is installed on your Jetson Nano."
    echo "TensorRT should be included with JetPack SDK."
    exit 1
}

# Check PyCUDA installation
python3 -c "import pycuda.driver; print('PyCUDA is available')" 2>/dev/null || {
    echo "Error: PyCUDA not found!"
    echo "Install PyCUDA with: pip3 install pycuda"
    exit 1
}

echo "All dependencies are available. Starting conversion..."
echo ""

# Run conversion
echo "Converting $ONNX_MODEL to $ENGINE_OUTPUT..."
echo "This may take several minutes on Jetson Nano, please wait..."
echo ""

python3 convert_onnx_to_tensorrt.py \
    --onnx "$ONNX_MODEL" \
    --engine "$ENGINE_OUTPUT" \
    --precision "$PRECISION" \
    --batch_size "$BATCH_SIZE" \
    --workspace "$WORKSPACE_MB" \
    --input_shape $INPUT_SHAPE \
    --test \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Conversion Successful! ==="
    echo "TensorRT engine created: $ENGINE_OUTPUT"
    echo ""
    echo "Next steps:"
    echo "1. Copy GlaucomaCam_TensorRT.py to your Jetson Nano"
    echo "2. Copy $ENGINE_OUTPUT to your Jetson Nano"
    echo "3. Install required Python packages on Jetson Nano:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install python3-pip python3-tk"
    echo "   pip3 install opencv-python pillow numpy requests pycuda"
    echo "4. Run the application:"
    echo "   python3 GlaucomaCam_TensorRT.py"
    echo ""
    echo "Make sure the engine file path in GlaucomaCam_TensorRT.py matches your engine location."
else
    echo ""
    echo "=== Conversion Failed! ==="
    echo "Please check the error messages above and try again."
    exit 1
fi
