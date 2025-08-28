#!/usr/bin/env python3
"""
ONNX to TensorRT Conversion Script for Jetson Nano
This script converts YOLO ONNX models to TensorRT engines optimized for Jetson Nano

Usage:
    python convert_onnx_to_tensorrt.py --onnx model.onnx --engine model.engine --precision fp16
    
Requirements:
    - TensorRT (should be pre-installed on Jetson Nano with JetPack)
    - pycuda
    - numpy
"""

import argparse
import os
import sys
import logging
import time
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure TensorRT and PyCUDA are installed on your Jetson Nano")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXToTensorRTConverter:
    """Convert ONNX models to TensorRT engines optimized for Jetson Nano"""
    
    def __init__(self, onnx_path, engine_path, precision='fp16', max_batch_size=1, 
                 max_workspace_size=1 << 30, input_shape=None):
        """
        Initialize the converter
        
        Args:
            onnx_path (str): Path to ONNX model
            engine_path (str): Path to save TensorRT engine
            precision (str): Precision mode ('fp32', 'fp16', 'int8')
            max_batch_size (int): Maximum batch size
            max_workspace_size (int): Maximum workspace size in bytes
            input_shape (tuple): Input shape (batch, channels, height, width)
        """
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.max_workspace_size = max_workspace_size
        self.input_shape = input_shape or (1, 3, 416, 416)
        
        # Validate inputs
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        if self.precision not in ['fp32', 'fp16', 'int8']:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        # Setup TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
    def build_engine(self):
        """Build TensorRT engine from ONNX model"""
        logger.info(f"Building TensorRT engine from {self.onnx_path}")
        logger.info(f"Precision: {self.precision}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        logger.info(f"Input shape: {self.input_shape}")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX model
        logger.info("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(f"Parser error {error}: {parser.get_error(error)}")
                return None
        
        logger.info("ONNX model parsed successfully")
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        # Set precision
        if self.precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, falling back to FP32")
        elif self.precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled")
                # Note: INT8 calibration would be needed here for optimal performance
                logger.warning("INT8 calibration not implemented - using default calibration")
            else:
                logger.warning("INT8 not supported on this platform, falling back to FP32")
        
        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Get input tensor name (assuming first input is the main input)
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        logger.info(f"Input tensor name: {input_name}")
        
        # Set dynamic shapes (min, optimal, max)
        batch_size, channels, height, width = self.input_shape
        profile.set_shape(input_name, 
                         (1, channels, height, width),          # min
                         (batch_size, channels, height, width), # opt
                         (self.max_batch_size, channels, height, width))  # max
        
        config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building TensorRT engine... This may take several minutes on Jetson Nano")
        start_time = time.time()
        
        engine = builder.build_engine(network, config)
        
        build_time = time.time() - start_time
        logger.info(f"Engine built successfully in {build_time:.2f} seconds")
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return None
        
        return engine
    
    def save_engine(self, engine):
        """Save TensorRT engine to file"""
        logger.info(f"Saving engine to {self.engine_path}")
        
        # Create directory if it doesn't exist
        engine_dir = os.path.dirname(self.engine_path)
        if engine_dir:  # Only create directory if there's a directory component
            os.makedirs(engine_dir, exist_ok=True)
        
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"Engine saved successfully: {self.engine_path}")
        
        # Print engine info
        self.print_engine_info(engine)
    
    def print_engine_info(self, engine):
        """Print information about the built engine"""
        logger.info("=== Engine Information ===")
        logger.info(f"Max batch size: {engine.max_batch_size}")
        logger.info(f"Number of bindings: {engine.num_bindings}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            
            logger.info(f"Binding {i}: {name}")
            logger.info(f"  Shape: {shape}")
            logger.info(f"  Type: {dtype}")
            logger.info(f"  Is Input: {is_input}")
        
        # Get engine size
        engine_size = len(engine.serialize()) / (1024 * 1024)  # MB
        logger.info(f"Engine size: {engine_size:.2f} MB")
    
    def convert(self):
        """Main conversion method"""
        try:
            # Build engine
            engine = self.build_engine()
            if engine is None:
                return False
            
            # Save engine
            self.save_engine(engine)
            
            logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False

def test_engine(engine_path, input_shape=(1, 3, 416, 416)):
    """Test the converted TensorRT engine"""
    logger.info(f"Testing engine: {engine_path}")
    
    try:
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            logger.error("Failed to load engine")
            return False
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Prepare test input
        input_data = np.random.rand(*input_shape).astype(np.float32)
        
        # Allocate buffers
        bindings = []
        for i in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(i)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(i):
                np.copyto(host_mem, input_data.ravel())
                cuda.memcpy_htod(device_mem, host_mem)
        
        # Run inference
        stream = cuda.Stream()
        start_time = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        inference_time = time.time() - start_time
        
        logger.info(f"Test inference successful! Time: {inference_time*1000:.2f}ms")
        return True
        
    except Exception as e:
        logger.error(f"Engine test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine for Jetson Nano")
    parser.add_argument('--onnx', required=True, help='Path to input ONNX model')
    parser.add_argument('--engine', required=True, help='Path to output TensorRT engine')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'],
                       help='Precision mode (default: fp16)')
    parser.add_argument('--batch_size', type=int, default=1, help='Maximum batch size (default: 1)')
    parser.add_argument('--workspace', type=int, default=1024, help='Max workspace size in MB (default: 1024)')
    parser.add_argument('--input_shape', nargs=4, type=int, default=[1, 3, 416, 416],
                       help='Input shape as batch channels height width (default: 1 3 416 416)')
    parser.add_argument('--test', action='store_true', help='Test the converted engine after conversion')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert workspace to bytes
    workspace_size = args.workspace * 1024 * 1024
    
    logger.info("=== ONNX to TensorRT Conversion ===")
    logger.info(f"Input ONNX: {args.onnx}")
    logger.info(f"Output Engine: {args.engine}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Workspace: {args.workspace} MB")
    logger.info(f"Input Shape: {args.input_shape}")
    
    # Create converter
    converter = ONNXToTensorRTConverter(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        max_batch_size=args.batch_size,
        max_workspace_size=workspace_size,
        input_shape=tuple(args.input_shape)
    )
    
    # Convert
    success = converter.convert()
    
    if success and args.test:
        logger.info("Running engine test...")
        test_engine(args.engine, tuple(args.input_shape))
    
    if success:
        logger.info("Conversion completed successfully!")
        logger.info(f"You can now use the engine file: {args.engine}")
        logger.info("Copy this engine file to your Jetson Nano and use it with GlaucomaCam_TensorRT.py")
    else:
        logger.error("Conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
