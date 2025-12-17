#!/usr/bin/env python3
"""
Projection-based MLCC Index Detector - Python wrapper

This module provides a Python interface to the C++ projection index detector.
Falls back to a pure Python implementation if the C++ module is not available.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Try to import C++ module
try:
    import projection_index_cpp as cpp_detector
    HAS_CPP_MODULE = True
except ImportError:
    HAS_CPP_MODULE = False
    print("Warning: C++ projection_index_cpp module not found, using Python fallback")


@dataclass
class IndexInfo:
    """Detected Index information"""
    center_y: int
    top_y: int
    bottom_y: int
    confidence: float


class ProjectionIndexDetector:
    """
    Projection-based Index detector

    Uses derivative-based method to detect Index positions from horizontal projection.
    """

    def __init__(self, use_cpp: bool = True):
        """
        Initialize detector

        Args:
            use_cpp: Use C++ implementation if available (default: True)
        """
        self.use_cpp = use_cpp and HAS_CPP_MODULE

        if self.use_cpp:
            self._detector = cpp_detector.ProjectionIndexDetector()
        else:
            self._detector = None

        # Parameters (for Python fallback)
        self.smoothing_sigma = 5.0
        self.derivative_threshold = 0.1
        self.min_index_height = 50
        self.max_index_height = 300
        self.index_pitch = 0.0
        self.pitch_tolerance = 0.33

        # Debug data
        self._projection = None
        self._smoothed_projection = None
        self._derivative = None

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect indices from image

        Args:
            image: BGR image (numpy array)

        Returns:
            Dictionary with detection results:
            - 'indices': List of IndexInfo objects
            - 'count': Number of detected indices
            - 'process_time_ms': Processing time in milliseconds
            - 'cutting_lines': List of Y coordinates for cutting
        """
        if self.use_cpp:
            return self._detect_cpp(image)
        else:
            return self._detect_python(image)

    def _detect_cpp(self, image: np.ndarray) -> Dict:
        """C++ implementation"""
        # Ensure contiguous array
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        result = self._detector.detect(image)

        indices = []
        for idx in result.indices:
            indices.append(IndexInfo(
                center_y=idx.center_y,
                top_y=idx.top_y,
                bottom_y=idx.bottom_y,
                confidence=idx.confidence
            ))

        # Store debug data
        self._projection = self._detector.get_projection()
        self._smoothed_projection = self._detector.get_smoothed_projection()
        self._derivative = self._detector.get_derivative()

        return {
            'indices': indices,
            'count': len(indices),
            'process_time_ms': result.process_time_ms,
            'cutting_lines': [idx.center_y for idx in indices]
        }

    def _detect_python(self, image: np.ndarray) -> Dict:
        """Pure Python fallback implementation - Simple threshold method"""
        start_time = time.time()

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape

        # Compute horizontal projection (normalized by width)
        projection = np.sum(gray, axis=1).astype(np.float32) / w

        # Store debug data
        self._projection = projection
        self._smoothed_projection = projection  # No smoothing needed
        self._derivative = np.gradient(projection)

        # Simple threshold: (mean + max) / 2
        proj_mean = np.mean(projection)
        proj_max = np.max(projection)
        threshold = (proj_mean + proj_max) / 2

        # Find regions above threshold
        above = projection > threshold
        indices = []

        # Find contiguous regions
        in_region = False
        top_y = 0

        for y in range(h):
            if above[y] and not in_region:
                # Region start
                in_region = True
                top_y = y
            elif not above[y] and in_region:
                # Region end
                in_region = False
                bottom_y = y - 1
                height = bottom_y - top_y + 1

                if self.min_index_height <= height <= self.max_index_height:
                    center_y = (top_y + bottom_y) // 2
                    peak_value = np.max(projection[top_y:bottom_y + 1])
                    confidence = peak_value / proj_max if proj_max > 0 else 0

                    indices.append(IndexInfo(
                        center_y=center_y,
                        top_y=top_y,
                        bottom_y=bottom_y,
                        confidence=confidence
                    ))

        # Handle region at end of image
        if in_region:
            bottom_y = h - 1
            height = bottom_y - top_y + 1
            if self.min_index_height <= height <= self.max_index_height:
                center_y = (top_y + bottom_y) // 2
                peak_value = np.max(projection[top_y:bottom_y + 1])
                confidence = peak_value / proj_max if proj_max > 0 else 0
                indices.append(IndexInfo(
                    center_y=center_y,
                    top_y=top_y,
                    bottom_y=bottom_y,
                    confidence=confidence
                ))

        # Validate pitch if enabled
        if self.index_pitch > 0 and len(indices) > 1:
            indices = self._validate_pitch(indices)

        process_time = (time.time() - start_time) * 1000

        return {
            'indices': indices,
            'count': len(indices),
            'process_time_ms': process_time,
            'cutting_lines': [idx.center_y for idx in indices],
            'threshold': threshold
        }

    def _validate_pitch(self, indices: List[IndexInfo]) -> List[IndexInfo]:
        """Validate index pitch"""
        min_pitch = self.index_pitch * (1.0 - self.pitch_tolerance)
        max_pitch = self.index_pitch * (1.0 + self.pitch_tolerance)

        validated = []

        # Find first valid pair
        first_valid = -1
        for i in range(len(indices) - 1):
            pitch = indices[i + 1].center_y - indices[i].center_y
            if min_pitch <= pitch <= max_pitch:
                first_valid = i
                break

        if first_valid >= 0:
            validated.append(indices[first_valid])
            for i in range(first_valid + 1, len(indices)):
                pitch = indices[i].center_y - validated[-1].center_y
                if min_pitch <= pitch <= max_pitch:
                    validated.append(indices[i])
        elif indices:
            validated.append(indices[0])

        return validated

    # Parameter setters
    def set_smoothing_sigma(self, sigma: float):
        self.smoothing_sigma = sigma
        if self.use_cpp:
            self._detector.set_smoothing_sigma(sigma)

    def set_derivative_threshold(self, threshold: float):
        self.derivative_threshold = threshold
        if self.use_cpp:
            self._detector.set_derivative_threshold(threshold)

    def set_min_index_height(self, height: int):
        self.min_index_height = height
        if self.use_cpp:
            self._detector.set_min_index_height(height)

    def set_max_index_height(self, height: int):
        self.max_index_height = height
        if self.use_cpp:
            self._detector.set_max_index_height(height)

    def set_index_pitch(self, pitch: float):
        self.index_pitch = pitch
        if self.use_cpp:
            self._detector.set_index_pitch(pitch)

    def set_pitch_tolerance(self, tolerance: float):
        self.pitch_tolerance = tolerance
        if self.use_cpp:
            self._detector.set_pitch_tolerance(tolerance)

    # Debug data access
    def get_projection(self) -> Optional[np.ndarray]:
        return self._projection

    def get_smoothed_projection(self) -> Optional[np.ndarray]:
        return self._smoothed_projection

    def get_derivative(self) -> Optional[np.ndarray]:
        return self._derivative

    def draw_results(self, image: np.ndarray, result: Dict,
                     show_lines: bool = True,
                     show_boundaries: bool = True) -> np.ndarray:
        """
        Draw detection results on image

        Args:
            image: Original BGR image
            result: Detection result from detect()
            show_lines: Draw cutting lines (center Y)
            show_boundaries: Draw top/bottom boundaries

        Returns:
            Image with drawn results
        """
        img = image.copy()
        h, w = img.shape[:2]

        for idx in result['indices']:
            if show_lines:
                # Draw center line (cutting line) - Red
                cv2.line(img, (0, idx.center_y), (w, idx.center_y), (0, 0, 255), 4)

            if show_boundaries:
                # Draw top boundary - Green
                cv2.line(img, (0, idx.top_y), (w, idx.top_y), (0, 255, 0), 2)
                # Draw bottom boundary - Blue
                cv2.line(img, (0, idx.bottom_y), (w, idx.bottom_y), (255, 0, 0), 2)

            # Draw label
            label = f"Y:{idx.center_y} ({idx.confidence:.2f})"
            cv2.putText(img, label, (10, idx.center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return img


# Test code
if __name__ == "__main__":
    import sys

    print(f"C++ module available: {HAS_CPP_MODULE}")

    if len(sys.argv) < 2:
        print("Usage: python projection_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    detector = ProjectionIndexDetector(use_cpp=HAS_CPP_MODULE)
    result = detector.detect(image)

    print(f"Detected {result['count']} indices in {result['process_time_ms']:.2f} ms")
    for idx in result['indices']:
        print(f"  Index: Y={idx.center_y}, top={idx.top_y}, bottom={idx.bottom_y}, conf={idx.confidence:.2f}")

    # Show result
    result_img = detector.draw_results(image, result)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
