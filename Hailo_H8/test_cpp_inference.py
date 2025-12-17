#!/usr/bin/env python3
"""
Test script for C++ Hailo inference module
"""
import sys
import os
import cv2
import time
import glob

# Add module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hailo_yolov8_cpp

# Paths
HEF_PATH = "/home/amap/Project/FITO2026/MLCC_Index/Hailo_H8/fito_best.hef"
TEST_IMAGE_DIR = "/home/amap/Project/FITO2026/MLCC_Index/images/Side_Left"

def main():
    print("=" * 60)
    print("Testing C++ Hailo Inference Module")
    print("=" * 60)

    # Create inference engine
    print(f"\nCreating HailoYolov8Seg with HEF: {HEF_PATH}")
    engine = hailo_yolov8_cpp.HailoYolov8Seg(HEF_PATH, 640, 544)

    # Set thresholds
    engine.set_conf_threshold(0.75)
    engine.set_iou_threshold(0.45)
    print(f"Conf threshold: {engine.get_conf_threshold()}")
    print(f"IOU threshold: {engine.get_iou_threshold()}")

    # Initialize
    print("\nInitializing...")
    if not engine.initialize():
        print("Failed to initialize!")
        return 1

    print("Initialized successfully!")

    # Find test images
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.bmp"))
    if not image_files:
        print(f"No images found in {TEST_IMAGE_DIR}")
        engine.cleanup()
        return 1

    print(f"\nFound {len(image_files)} images")

    # Test on first image
    test_image_path = image_files[0]
    print(f"\nTesting on: {os.path.basename(test_image_path)}")

    image = cv2.imread(test_image_path)
    if image is None:
        print("Failed to load image")
        engine.cleanup()
        return 1

    print(f"Image shape: {image.shape}")

    # Run inference
    print("\nRunning inference...")
    result = engine.infer(image)

    print(f"\n=== Results ===")
    print(f"Detections: {result.count}")
    print(f"Inference time: {result.inference_time_ms:.1f} ms")
    print(f"Total time: {result.total_time_ms:.1f} ms")

    for i, det in enumerate(result.detections):
        print(f"  [{i}] box=({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}), "
              f"score={det.score:.3f}")

    # Test dict conversion
    print("\n=== Dict Output ===")
    result_dict = engine.infer_dict(image)
    print(f"Boxes: {result_dict['boxes']}")
    print(f"Scores: {result_dict['scores']}")
    print(f"Count: {result_dict['count']}")
    print(f"Inference time: {result_dict['inference_time_ms']:.1f} ms")

    # Benchmark
    print("\n=== Benchmark (10 iterations) ===")
    times = []
    for i in range(10):
        result = engine.infer(image)
        times.append(result.total_time_ms)

    print(f"Average total time: {sum(times)/len(times):.1f} ms")
    print(f"Min: {min(times):.1f} ms, Max: {max(times):.1f} ms")

    # Cleanup
    engine.cleanup()
    print("\nTest completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
