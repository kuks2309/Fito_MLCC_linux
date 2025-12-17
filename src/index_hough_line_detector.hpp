/**
 * @file index_hough_line_detector.hpp
 * @brief Hough Line Transform for vertical line detection (-15° ~ +15°)
 *
 * Used for detecting alignment of MLCC Index (bright horizontal bands)
 * by finding near-vertical lines in edge images.
 */

#ifndef INDEX_HOUGH_LINE_DETECTOR_HPP
#define INDEX_HOUGH_LINE_DETECTOR_HPP

#include <vector>
#include <cstdint>
#include <utility>

namespace hough_detector {

/**
 * @brief Detected line information
 */
struct LineInfo {
    float rho;              // Distance from origin
    float theta;            // Angle in radians
    float angle_deg;        // Angle in degrees (0 = vertical)
    int votes;              // Accumulator votes
    int x1, y1, x2, y2;     // Line endpoints for visualization
};

/**
 * @brief Hough Line detection result
 */
struct HoughResult {
    std::vector<LineInfo> lines;
    float process_time_ms;
    float mean_angle_deg;   // Mean angle of detected lines
    bool is_aligned;        // True if mean angle is within tolerance
};

/**
 * @brief Hough Line detector for near-vertical lines
 */
class IndexHoughLineDetector {
public:
    IndexHoughLineDetector();
    ~IndexHoughLineDetector() = default;

    /**
     * @brief Detect near-vertical lines using Hough Transform
     * @param edge_data Grayscale edge image (from Sobel)
     * @param width Image width
     * @param height Image height
     * @return Detection result with lines and alignment info
     */
    HoughResult detect(const uint8_t* edge_data, int width, int height);

    /**
     * @brief Detect lines from BGR image (includes edge detection)
     * @param image_data BGR image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @param scale_factor Downscale factor for processing
     * @return Detection result
     */
    HoughResult detectFromImage(const uint8_t* image_data, int width, int height, int scale_factor = 4);

    // Parameter setters
    void setAngleRange(float min_deg, float max_deg) {
        min_angle_deg_ = min_deg;
        max_angle_deg_ = max_deg;
    }
    void setEdgeThreshold(int threshold) { edge_threshold_ = threshold; }
    void setAccumulatorThreshold(int threshold) { accumulator_threshold_ = threshold; }
    void setRhoResolution(float rho) { rho_resolution_ = rho; }
    void setThetaResolution(float theta_deg) { theta_resolution_deg_ = theta_deg; }
    void setMaxLines(int max_lines) { max_lines_ = max_lines; }

    // Parameter getters
    float getMinAngleDeg() const { return min_angle_deg_; }
    float getMaxAngleDeg() const { return max_angle_deg_; }
    int getEdgeThreshold() const { return edge_threshold_; }
    int getAccumulatorThreshold() const { return accumulator_threshold_; }

private:
    // Parameters
    float min_angle_deg_ = -15.0f;      // Minimum angle from vertical (degrees)
    float max_angle_deg_ = 15.0f;       // Maximum angle from vertical (degrees)
    int edge_threshold_ = 30;            // Edge pixel threshold
    int accumulator_threshold_ = 30;     // Minimum votes for line detection
    float rho_resolution_ = 1.0f;        // Rho resolution in pixels
    float theta_resolution_deg_ = 1.0f;  // Theta resolution in degrees
    int max_lines_ = 50;                 // Maximum number of lines to return

    // Internal methods
    void computeAccumulator(const uint8_t* edge_data, int width, int height,
                           std::vector<int>& accumulator, int& num_rho, int& num_theta,
                           float& theta_min, float& theta_step, float& rho_max);
    std::vector<LineInfo> extractLines(const std::vector<int>& accumulator,
                                       int num_rho, int num_theta,
                                       float theta_min, float theta_step,
                                       float rho_max, int width, int height);
};

} // namespace hough_detector

#endif // INDEX_HOUGH_LINE_DETECTOR_HPP
