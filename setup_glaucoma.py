#!/usr/bin/env python3
"""
Setup script for Glaucoma Detection System
This script creates necessary directories and prepares the environment
"""

import os
import sys
import subprocess
import yaml
from datetime import datetime


def create_directories():
    """Create necessary directories for the glaucoma detection system"""
    directories = [
        'results',
        'results/images',
        'results/reports',
        'logs',
        'models',
        'keys'
    ]
    
    print("Creating directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✓ Created: {directory}")
        else:
            print(f"  ○ Exists: {directory}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'PIL',
        'requests',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"  ✓ OpenCV version: {cv2.__version__}")
            elif package == 'torch':
                import torch
                print(f"  ✓ PyTorch version: {torch.__version__}")
                print(f"    CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"    CUDA devices: {torch.cuda.device_count()}")
            elif package == 'yaml':
                import yaml
                print(f"  ✓ PyYAML available")
            else:
                __import__(package)
                print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ Missing: {package}")
    
    return missing_packages


def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if not missing_packages:
        return True
    
    print(f"\nMissing packages detected: {missing_packages}")
    response = input("Would you like to install them? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Please install the missing packages manually using:")
        print("pip install -r requirements_glaucoma.txt")
        return False
    
    try:
        print("Installing packages...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_glaucoma.txt'
        ])
        print("  ✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error installing packages: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file if it doesn't exist"""
    config_file = 'config_glaucoma_local.yaml'
    
    if os.path.exists(config_file):
        print(f"  ○ Config file already exists: {config_file}")
        return
    
    print(f"Creating local configuration file: {config_file}")
    
    try:
        # Load the template config
        with open('config_glaucoma.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify for local development
        config['server']['enabled'] = False
        config['development']['debug_mode'] = True
        config['logging']['level'] = 'DEBUG'
        
        # Save local config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"  ✓ Created: {config_file}")
        
    except Exception as e:
        print(f"  ✗ Error creating config: {e}")


def check_model_file():
    """Check if model file exists"""
    print("\nChecking model files...")
    
    model_paths = [
        'runs_v7tiny100epoch/train/exp/weights/best.pt',
        'models/glaucoma_v1.pt',
        'models/glaucoma_v2.pt'
    ]
    
    model_found = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"  ✓ Found model: {model_path}")
            model_found = True
        else:
            print(f"  ○ Model not found: {model_path}")
    
    if not model_found:
        print("\n⚠️  WARNING: No model files found!")
        print("   Please ensure you have a trained glaucoma detection model.")
        print("   You can:")
        print("   1. Train a new model using the existing training scripts")
        print("   2. Download a pre-trained model")
        print("   3. Use the existing component detection model for testing")


def check_camera():
    """Check if camera is available"""
    print("\nChecking camera availability...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("  ✓ Camera 0 is working")
                height, width = frame.shape[:2]
                print(f"    Resolution: {width}x{height}")
            else:
                print("  ⚠️  Camera 0 detected but cannot read frames")
            cap.release()
        else:
            print("  ✗ Camera 0 not available")
        
        # Check for additional cameras
        camera_count = 0
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_count += 1
                print(f"  ✓ Camera {i} available")
                cap.release()
        
        if camera_count > 0:
            print(f"  Total cameras found: {camera_count + 1}")
        
    except Exception as e:
        print(f"  ✗ Error checking camera: {e}")


def create_test_data():
    """Create sample test data"""
    print("\nCreating test data...")
    
    # Create a simple test report
    test_report = {
        "patient_info": {
            "patient_id": "P001",
            "name": "Test Patient",
            "age": "45",
            "gender": "Male",
            "phone": "+1234567890",
            "medical_history": "No previous eye conditions",
            "examination_date": datetime.now().isoformat()
        },
        "doctor_info": {
            "doctor_id": "D001", 
            "name": "Dr. Test",
            "specialization": "Ophthalmologist",
            "hospital": "Test Hospital"
        },
        "detection_result": {
            "has_glaucoma": False,
            "confidence": 0.0,
            "detected_features": [],
            "analysis_timestamp": datetime.now().isoformat()
        },
        "image_path": "results/test_image.jpg"
    }
    
    test_file = 'results/reports/test_report.json'
    try:
        import json
        with open(test_file, 'w') as f:
            json.dump(test_report, f, indent=2)
        print(f"  ✓ Created test report: {test_file}")
    except Exception as e:
        print(f"  ✗ Error creating test data: {e}")


def main():
    """Main setup function"""
    print("=" * 50)
    print("Glaucoma Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️  WARNING: Python 3.7+ is recommended")
    
    # Create directories
    create_directories()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Install missing packages
    if missing_packages:
        success = install_dependencies(missing_packages)
        if not success:
            print("\nSetup incomplete. Please install missing dependencies manually.")
            return
    
    # Create configuration
    create_sample_config()
    
    # Check model files
    check_model_file()
    
    # Check camera
    check_camera()
    
    # Create test data
    create_test_data()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Ensure you have a trained glaucoma detection model")
    print("2. Update the model path in config_glaucoma_local.yaml")
    print("3. Run the application: python GlaucomaCam.py")
    print("4. Test with your camera and sample data")
    print("\nFor more information, see README_Glaucoma.md")


if __name__ == "__main__":
    main() 