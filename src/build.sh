#!/bin/bash
# Build script for hailo_yolov8_cpp module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building hailo_yolov8_cpp module ==="

# Clean and create build directory
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Module location: ${SCRIPT_DIR}/../Hailo_H8/hailo_yolov8_cpp*.so"
ls -la "${SCRIPT_DIR}/../Hailo_H8/"*.so 2>/dev/null || echo "Module not found in output directory"
