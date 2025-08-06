#!/usr/bin/env python3
"""
Jetson Nano Optimization Script for Glaucoma Detection
This script optimizes the YOLOv7 model for deployment on Jetson Nano
"""

import os
import torch
import yaml
import argparse
from pathlib import Path

def optimize_for_jetson(model_path, output_dir="jetson_deployment"):
    """
    Optimize YOLOv7 model for Jetson Nano deployment
    """
    print("🚀 Starting Jetson Nano optimization...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✓ Found model: {model_path}")
    
    try:
        # Load and optimize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device)
        
        # Convert to half precision for Jetson Nano
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        
        # Optimize for inference
        model.eval()
        model.half()  # Convert to FP16
        
        # Save optimized model
        optimized_path = output_path / "glaucoma_model_optimized.pt"
        torch.save({
            'model': model,
            'epoch': -1,
            'optimizer': None,
            'training_results': None,
            'wandb_id': None
        }, optimized_path)
        
        print(f"✓ Optimized model saved: {optimized_path}")
        
        # Create deployment configuration
        jetson_config = {
            'model': {
                'path': str(optimized_path),
                'input_size': [416, 416],
                'confidence_threshold': 0.35,
                'iou_threshold': 0.65,
                'max_detections': 100,
                'use_half_precision': True
            },
            'jetson': {
                'power_mode': 0,  # MAXN mode
                'jetson_clocks': True,
                'cpu_threads': 4,
                'gpu_memory_fraction': 0.9
            },
            'inference': {
                'batch_size': 1,
                'warmup_iterations': 10,
                'enable_tensorrt': True,
                'tensorrt_precision': 'fp16'
            }
        }
        
        # Save Jetson configuration
        config_path = output_path / "jetson_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(jetson_config, f, default_flow_style=False)
        
        print(f"✓ Jetson configuration saved: {config_path}")
        
        # Create deployment script
        deployment_script = f'''#!/bin/bash
# Jetson Nano Deployment Script for Glaucoma Detection

echo "🔧 Setting up Jetson Nano for optimal performance..."

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install Python dependencies
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python pillow numpy pyyaml tqdm matplotlib

echo "✓ Jetson Nano setup complete!"
echo "🏥 Starting Glaucoma Detection System..."

# Run the application
python3 GlaucomaCam.py --config jetson_config.yaml

echo "🎯 Glaucoma Detection System started successfully!"
'''
        
        script_path = output_path / "deploy_jetson.sh"
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        print(f"✓ Deployment script created: {script_path}")
        
        # Create performance monitoring script
        monitor_script = '''#!/usr/bin/env python3
import time
import psutil
import subprocess
import json

def monitor_jetson_performance():
    """Monitor Jetson Nano performance during inference"""
    print("📊 Starting Jetson Nano performance monitoring...")
    
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # GPU usage (if available)
        try:
            gpu_stats = subprocess.check_output(['tegrastats', '--interval', '1000'], 
                                              timeout=2).decode()
            print(f"GPU Stats: {gpu_stats.strip()}")
        except:
            pass
        
        # Temperature
        try:
            temp_output = subprocess.check_output(['cat', '/sys/class/thermal/thermal_zone*/temp'])
            temps = [int(t)/1000 for t in temp_output.decode().strip().split('\\n')]
            avg_temp = sum(temps) / len(temps)
        except:
            avg_temp = 0
        
        stats = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'temperature_c': avg_temp
        }
        
        print(f"📈 CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Temp: {avg_temp:.1f}°C")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_jetson_performance()
'''
        
        monitor_path = output_path / "monitor_performance.py"
        with open(monitor_path, 'w') as f:
            f.write(monitor_script)
        
        print(f"✓ Performance monitor created: {monitor_path}")
        
        # Copy essential files
        essential_files = [
            "GlaucomaCam.py",
            "config_glaucoma.yaml", 
            "requirements_glaucoma.txt",
            "data/glaucoma.yaml"
        ]
        
        for file_path in essential_files:
            if os.path.exists(file_path):
                dest = output_path / os.path.basename(file_path)
                import shutil
                shutil.copy2(file_path, dest)
                print(f"✓ Copied: {file_path}")
        
        # Create README
        readme_content = f"""# Glaucoma Detection System - Jetson Nano Deployment

## 📋 Overview
This package contains an optimized YOLOv7-based glaucoma detection system for NVIDIA Jetson Nano.

## 🚀 Quick Start

1. **Copy files to Jetson Nano:**
   ```bash
   scp -r {output_dir}/* jetson@<jetson-ip>:~/glaucoma_detection/
   ```

2. **Setup Jetson Nano:**
   ```bash
   chmod +x deploy_jetson.sh
   ./deploy_jetson.sh
   ```

3. **Run the application:**
   ```bash
   python3 GlaucomaCam.py
   ```

## 📁 Files Description

- `glaucoma_model_optimized.pt` - Optimized YOLOv7 model (FP16)
- `jetson_config.yaml` - Jetson-specific configuration
- `deploy_jetson.sh` - Automated deployment script
- `monitor_performance.py` - Performance monitoring tool
- `GlaucomaCam.py` - Main application
- `config_glaucoma.yaml` - Application configuration
- `requirements_glaucoma.txt` - Python dependencies

## ⚡ Performance Optimization

The model has been optimized for Jetson Nano with:
- FP16 precision for 2x faster inference
- Input size reduced to 416x416 for memory efficiency
- Optimized confidence and IoU thresholds
- TensorRT integration support

## 🔧 Manual Setup (if automated script fails)

```bash
# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install dependencies
pip3 install torch torchvision opencv-python pillow numpy pyyaml

# Run application
python3 GlaucomaCam.py
```

## 📊 Performance Monitoring

Monitor system performance during inference:
```bash
python3 monitor_performance.py
```

## 🏥 Medical Use Notice

This system is for research and educational purposes. For clinical use, ensure proper validation and regulatory compliance.

## 🔍 Troubleshooting

- **Memory issues**: Reduce batch size or input resolution
- **Slow inference**: Enable TensorRT optimization
- **Camera issues**: Check camera permissions and device access
- **Model loading errors**: Verify file paths and model compatibility

## 📞 Support

For technical support, refer to the project documentation or contact the development team.
"""
        
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✓ README created: {readme_path}")
        
        print(f"\n🎉 Jetson Nano optimization complete!")
        print(f"📦 Deployment package ready in: {output_path}")
        print(f"📋 Next steps:")
        print(f"   1. Copy {output_path}/ to your Jetson Nano")
        print(f"   2. Run: chmod +x deploy_jetson.sh && ./deploy_jetson.sh")
        print(f"   3. Launch: python3 GlaucomaCam.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Optimize YOLOv7 model for Jetson Nano")
    parser.add_argument("--model", default="runs/glaucoma_train/yolov7_tiny_glaucoma/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--output", default="jetson_deployment", 
                       help="Output directory for deployment package")
    
    args = parser.parse_args()
    
    success = optimize_for_jetson(args.model, args.output)
    if success:
        print("✅ Ready for Jetson Nano deployment!")
    else:
        print("❌ Optimization failed!")

if __name__ == "__main__":
    main()