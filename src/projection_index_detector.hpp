/**
 * @file projection_index_detector.hpp
 * @brief Projection-based MLCC Index detector using derivative method
 *
 * Algorithm:
 * 1. Grayscale conversion
 * 2. Horizontal projection calculation
 * 3. Gaussian smoothing (noise reduction)
 * 4. 1st derivative calculation
 * 5. Zero-crossing detection (boundary points)
 * 6. Index center calculation from boundary pairs
 * 7. Index Pitch validation (optional)
 */

#ifndef PROJECTION_INDEX_DETECTOR_HPP
#define PROJECTION_INDEX_DETECTOR_HPP

#include <vector>
#include <cstdint>

namespace projection_detector {

/**
 * @brief Detected Index information
 */
struct IndexInfo {
    int center_y;           // Index center Y coordinate
    int top_y;              // Index top boundary
    int bottom_y;           // Index bottom boundary
    float confidence;       // Detection confidence (0-1)
};

/**
 * @brief Detection result
 */
struct DetectionResult {
    std::vector<IndexInfo> indices;
    float process_time_ms;
    int image_width;
    int image_height;
};

/**
 * @brief Projection-based Index detector
 */
class ProjectionIndexDetector {
public:
    ProjectionIndexDetector();
    ~ProjectionIndexDetector() = default;

    /**
     * @brief Detect indices from image
     * @param image_data BGR image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @return Detection result
     */
    DetectionResult detect(const uint8_t* image_data, int width, int height);

    // Parameter setters
    void setSmoothingSigma(float sigma) { smoothing_sigma_ = sigma; }
    void setDerivativeThreshold(float threshold) { derivative_threshold_ = threshold; }
    void setMinIndexHeight(int height) { min_index_height_ = height; }
    void setMaxIndexHeight(int height) { max_index_height_ = height; }
    void setIndexPitch(float pitch) { index_pitch_ = pitch; }
    void setPitchTolerance(float tolerance) { pitch_tolerance_ = tolerance; }
    void setUseROI(bool use) { use_roi_ = use; }
    void setROI(int x1, int y1, int x2, int y2);

    // Parameter getters
    float getSmoothingSigma() const { return smoothing_sigma_; }
    float getDerivativeThreshold() const { return derivative_threshold_; }
    int getMinIndexHeight() const { return min_index_height_; }
    int getMaxIndexHeight() const { return max_index_height_; }
    float getIndexPitch() const { return index_pitch_; }
    float getPitchTolerance() const { return pitch_tolerance_; }

    // Debug data access
    const std::vector<float>& getProjection() const { return projection_; }
    const std::vector<float>& getSmoothedProjection() const { return smoothed_projection_; }
    const std::vector<float>& getDerivative() const { return derivative_; }

    /**
     * @brief Compute vertical edge using Sobel X filter
     * @param image_data BGR image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @param scale_factor Downscale factor (e.g., 4 = 1/4 size)
     * @return Edge image (grayscale, original size), process_time_ms
     */
    std::pair<std::vector<uint8_t>, float> computeVerticalEdge(
        const uint8_t* image_data, int width, int height, int scale_factor = 4);

private:
    // Parameters
    float smoothing_sigma_ = 5.0f;          // Gaussian smoothing sigma
    float derivative_threshold_ = 0.1f;      // Derivative threshold (relative to max)
    int min_index_height_ = 50;              // Minimum index height (pixels)
    int max_index_height_ = 300;             // Maximum index height (pixels)
    float index_pitch_ = 0.0f;               // Expected index pitch (0 = disabled)
    float pitch_tolerance_ = 0.33f;          // Pitch tolerance (33%)

    // ROI
    bool use_roi_ = false;
    int roi_x1_ = 0, roi_y1_ = 0, roi_x2_ = 0, roi_y2_ = 0;

    // Debug data
    std::vector<float> projection_;
    std::vector<float> smoothed_projection_;
    std::vector<float> derivative_;

    // Internal methods
    void computeHorizontalProjection(const uint8_t* gray_data, int width, int height);
    void gaussianSmooth(int kernel_size);
    void computeDerivative();
    std::vector<int> findZeroCrossings(float threshold);
    std::vector<IndexInfo> pairBoundaries(const std::vector<int>& crossings);
    std::vector<IndexInfo> findRegionsAboveThreshold(float threshold, float proj_max);
    void validatePitch(std::vector<IndexInfo>& indices);
};

} // namespace projection_detector

#endif // PROJECTION_INDEX_DETECTOR_HPP
