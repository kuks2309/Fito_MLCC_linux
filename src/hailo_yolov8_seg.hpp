/**
 * @file hailo_yolov8_seg.hpp
 * @brief Hailo-8 YOLOv8-seg inference engine with C++ API
 *
 * Full inference pipeline using Hailo C++ SDK
 * - HEF model loading
 * - Preprocessing
 * - Inference
 * - Post-processing (DFL decoding, NMS, mask generation)
 */

#ifndef HAILO_YOLOV8_SEG_HPP
#define HAILO_YOLOV8_SEG_HPP

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <cstdint>

namespace hailo_yolov8 {

/**
 * @brief Detection result structure
 */
struct Detection {
    float x1, y1, x2, y2;       // Bounding box (original image coordinates)
    float score;                 // Confidence score
    int class_id;               // Class ID
    std::vector<uint8_t> mask;  // Binary mask (original image size)
    int mask_width;
    int mask_height;
};

/**
 * @brief Inference result structure
 */
struct InferenceResult {
    std::vector<Detection> detections;
    int original_width;
    int original_height;
    float inference_time_ms;    // Pure inference time
    float total_time_ms;        // Total processing time
};

/**
 * @brief YOLOv8-seg inference engine using Hailo-8
 */
class HailoYolov8Seg {
public:
    /**
     * @brief Constructor
     * @param hef_path Path to HEF model file
     * @param input_height Model input height (default: 640)
     * @param input_width Model input width (default: 544)
     */
    HailoYolov8Seg(const std::string& hef_path,
                   int input_height = 640,
                   int input_width = 544);

    ~HailoYolov8Seg();

    // Disable copy
    HailoYolov8Seg(const HailoYolov8Seg&) = delete;
    HailoYolov8Seg& operator=(const HailoYolov8Seg&) = delete;

    /**
     * @brief Initialize the inference engine
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Release resources
     */
    void cleanup();

    /**
     * @brief Check if engine is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Run inference on an image
     * @param image_data BGR image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @return Inference result
     */
    InferenceResult infer(const uint8_t* image_data, int width, int height);

    /**
     * @brief Set confidence threshold
     */
    void setConfThreshold(float threshold) { conf_threshold_ = threshold; }

    /**
     * @brief Set IOU threshold for NMS
     */
    void setIouThreshold(float threshold) { iou_threshold_ = threshold; }

    /**
     * @brief Get confidence threshold
     */
    float getConfThreshold() const { return conf_threshold_; }

    /**
     * @brief Get IOU threshold
     */
    float getIouThreshold() const { return iou_threshold_; }

private:
    // Configuration
    std::string hef_path_;
    int input_height_;
    int input_width_;
    float conf_threshold_ = 0.25f;
    float iou_threshold_ = 0.45f;

    // State
    bool initialized_ = false;

    // Hailo resources (PIMPL pattern)
    struct HailoImpl;
    std::unique_ptr<HailoImpl> impl_;

    // Internal methods
    std::vector<uint8_t> preprocess(const uint8_t* image_data,
                                     int width, int height);

    void processOutput(const std::vector<std::pair<std::string, std::vector<float>>>& outputs,
                       int orig_width, int orig_height,
                       std::vector<Detection>& detections);

    // DFL decoding
    float dflDecode(const float* data, int offset);

    // NMS
    void nms(std::vector<Detection>& detections);

    // Mask generation
    void generateMask(Detection& det,
                      const float* proto,
                      const float* mask_coeffs,
                      int proto_h, int proto_w, int proto_c,
                      int orig_width, int orig_height);

    // Sigmoid
    static float sigmoid(float x);
};

} // namespace hailo_yolov8

#endif // HAILO_YOLOV8_SEG_HPP
