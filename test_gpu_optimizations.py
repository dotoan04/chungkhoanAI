#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU optimizations for training
"""

import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np
from train import setup_gpu

def test_gpu_setup():
    """Test GPU setup with config"""
    print("Testing GPU setup...")

    # Test config
    test_cfg = {
        "gpu": {
            "memory_limit": 14336,  # 14GB
            "mixed_precision": True,
            "memory_growth": True
        }
    }

    # Test GPU setup
    gpu_enabled = setup_gpu(test_cfg)

    if gpu_enabled:
        print("GPU setup successful!")

        # Check mixed precision
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.global_policy()
        print(f"Mixed precision policy: {policy}")

        # Check available memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"GPU memory growth enabled: {tf.config.experimental.get_memory_growth(gpus[0])}")
    else:
        print("GPU setup failed, using CPU")

def test_memory_usage():
    """Test memory usage patterns"""
    print("\nTesting memory usage...")

    # Create some dummy data
    batch_size = 32
    seq_length = 60
    n_features = 50

    # Simulate training data
    X = np.random.randn(1000, seq_length, n_features).astype(np.float32)
    y = np.random.randn(1000,).astype(np.float32)

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Memory usage: X={X.nbytes / 1024**2:.1f}MB, y={y.nbytes / 1024**2:.1f}MB")

    # Test with TensorFlow
    import tensorflow as tf

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset created with batch_size={batch_size}")

    # Test memory clearing
    print("Testing memory clearing...")
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    print("Memory cleared")

def main():
    """Main test function"""
    test_gpu_setup()
    test_memory_usage()

    print("\nGPU optimization tests completed!")
    print("Ready for Colab training with optimized settings.")

if __name__ == "__main__":
    main()
