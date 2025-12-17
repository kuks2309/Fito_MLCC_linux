/**
 * @file hailo_yolov8_seg.cpp
 * @brief Hailo-8 YOLOv8-seg inference engine implementation
 */

#include "hailo_yolov8_seg.hpp"
#include <hailo/hailort.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <unordered_map>

namespace hailo_yolov8 {

using namespace hailort;

// Scale structure for multi-scale output
struct ScaleInfo {
    std::string bbox_name;
    std::string cls_name;
    std::string mask_name;
    int stride;
};

// PIMPL implementation
struct HailoYolov8Seg::HailoImpl {
    std::unique_ptr<VDevice> vdevice;
    std::shared_ptr<ConfiguredNetworkGroup> network_group;
    std::unique_ptr<Hef> hef;
    std::string input_name;
    std::vector<std::string> output_names;
    std::string model_prefix;

    // Persistent VStreams (created once, reused for every inference)
    std::vector<InputVStream> input_vstreams;
    std::vector<OutputVStream> output_vstreams;

    // Persistent activated network (unique_ptr - abstract class)
    std::unique_ptr<ActivatedNetworkGroup> activated_network;
};

HailoYolov8Seg::HailoYolov8Seg(const std::string& hef_path,
                               int input_height,
                               int input_width)
    : hef_path_(hef_path)
    , input_height_(input_height)
    , input_width_(input_width)
    , impl_(std::make_unique<HailoImpl>())
{
}

HailoYolov8Seg::~HailoYolov8Seg() {
    cleanup();
}

bool HailoYolov8Seg::initialize() {
    if (initialized_) {
        return true;
    }

    try {
        std::cout << "Loading HEF: " << hef_path_ << std::endl;

        // Load HEF
        auto hef_exp = Hef::create(hef_path_);
        if (!hef_exp) {
            std::cerr << "Failed to load HEF: " << hef_exp.status() << std::endl;
            return false;
        }
        impl_->hef = std::make_unique<Hef>(hef_exp.release());

        // Create VDevice with scheduler disabled
        hailo_vdevice_params_t vdevice_params = {};
        vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
        vdevice_params.device_count = 1;

        auto vdevice_exp = VDevice::create(vdevice_params);
        if (!vdevice_exp) {
            std::cerr << "Failed to create VDevice: " << vdevice_exp.status() << std::endl;
            return false;
        }
        impl_->vdevice = vdevice_exp.release();

        // Configure network group
        auto configure_params = impl_->vdevice->create_configure_params(*(impl_->hef));
        if (!configure_params) {
            std::cerr << "Failed to create configure params" << std::endl;
            return false;
        }

        auto network_groups = impl_->vdevice->configure(*(impl_->hef), configure_params.value());
        if (!network_groups) {
            std::cerr << "Failed to configure network: " << network_groups.status() << std::endl;
            return false;
        }

        if (network_groups->empty()) {
            std::cerr << "No network groups found" << std::endl;
            return false;
        }

        impl_->network_group = std::move(network_groups->at(0));

        // Get input/output info
        auto input_vstream_infos = impl_->hef->get_input_vstream_infos();
        if (!input_vstream_infos || input_vstream_infos->empty()) {
            std::cerr << "Failed to get input vstream info" << std::endl;
            return false;
        }
        impl_->input_name = input_vstream_infos->at(0).name;

        auto output_vstream_infos = impl_->hef->get_output_vstream_infos();
        if (!output_vstream_infos) {
            std::cerr << "Failed to get output vstream info" << std::endl;
            return false;
        }

        // Find model prefix from output names
        for (const auto& info : *output_vstream_infos) {
            std::string name(info.name);
            impl_->output_names.push_back(name);
            if (name.find("conv44") != std::string::npos) {
                size_t pos = name.rfind("/conv44");
                if (pos != std::string::npos) {
                    impl_->model_prefix = name.substr(0, pos);
                }
            }
        }

        if (impl_->model_prefix.empty()) {
            std::cerr << "Could not determine model prefix" << std::endl;
            return false;
        }

        std::cout << "Model prefix: " << impl_->model_prefix << std::endl;

        // Create persistent VStreams (only once)
        auto input_params = impl_->network_group->make_input_vstream_params(
            true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!input_params) {
            std::cerr << "Failed to create input params" << std::endl;
            return false;
        }

        auto output_params = impl_->network_group->make_output_vstream_params(
            true, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!output_params) {
            std::cerr << "Failed to create output params" << std::endl;
            return false;
        }

        auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(
            *impl_->network_group, *input_params);
        if (!input_vstreams_exp) {
            std::cerr << "Failed to create input vstreams" << std::endl;
            return false;
        }
        impl_->input_vstreams = input_vstreams_exp.release();

        auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(
            *impl_->network_group, *output_params);
        if (!output_vstreams_exp) {
            std::cerr << "Failed to create output vstreams" << std::endl;
            return false;
        }
        impl_->output_vstreams = output_vstreams_exp.release();

        // Activate network (keep active for all inferences)
        auto activated_network_exp = impl_->network_group->activate();
        if (!activated_network_exp) {
            std::cerr << "Failed to activate network" << std::endl;
            return false;
        }
        impl_->activated_network = activated_network_exp.release();

        std::cout << "Hailo-8 initialized successfully (VStreams persistent)" << std::endl;

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void HailoYolov8Seg::cleanup() {
    initialized_ = false;

    // Clear VStreams first
    impl_->input_vstreams.clear();
    impl_->output_vstreams.clear();

    // Deactivate network
    impl_->activated_network.reset();

    impl_->network_group.reset();
    impl_->vdevice.reset();
    impl_->hef.reset();
    std::cout << "Hailo-8 cleanup done" << std::endl;
}

std::vector<uint8_t> HailoYolov8Seg::preprocess(const uint8_t* image_data,
                                                 int width, int height) {
    // Simple resize using bilinear interpolation
    std::vector<uint8_t> resized(input_height_ * input_width_ * 3);

    float scale_x = static_cast<float>(width) / input_width_;
    float scale_y = static_cast<float>(height) / input_height_;

    for (int y = 0; y < input_height_; ++y) {
        for (int x = 0; x < input_width_; ++x) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; ++c) {
                // BGR to RGB conversion
                int src_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;

                float v00 = image_data[(y0 * width + x0) * 3 + src_c];
                float v01 = image_data[(y0 * width + x1) * 3 + src_c];
                float v10 = image_data[(y1 * width + x0) * 3 + src_c];
                float v11 = image_data[(y1 * width + x1) * 3 + src_c];

                float value = v00 * (1 - dx) * (1 - dy) +
                              v01 * dx * (1 - dy) +
                              v10 * (1 - dx) * dy +
                              v11 * dx * dy;

                resized[(y * input_width_ + x) * 3 + c] = static_cast<uint8_t>(value);
            }
        }
    }

    return resized;
}

float HailoYolov8Seg::sigmoid(float x) {
    x = std::max(-500.0f, std::min(500.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float HailoYolov8Seg::dflDecode(const float* data, int offset) {
    // Find max for numerical stability
    float max_val = data[offset];
    for (int i = 1; i < 16; ++i) {
        max_val = std::max(max_val, data[offset + i]);
    }

    // Softmax and weighted sum
    float sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float exp_val = std::exp(data[offset + i] - max_val);
        sum += exp_val;
        weighted_sum += exp_val * i;
    }

    return weighted_sum / sum;
}

void HailoYolov8Seg::nms(std::vector<Detection>& detections) {
    if (detections.empty()) return;

    // Sort by score descending
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            // Calculate IoU
            float x1 = std::max(detections[i].x1, detections[j].x1);
            float y1 = std::max(detections[i].y1, detections[j].y1);
            float x2 = std::min(detections[i].x2, detections[j].x2);
            float y2 = std::min(detections[i].y2, detections[j].y2);

            float inter_w = std::max(0.0f, x2 - x1);
            float inter_h = std::max(0.0f, y2 - y1);
            float inter_area = inter_w * inter_h;

            float area_i = (detections[i].x2 - detections[i].x1) *
                           (detections[i].y2 - detections[i].y1);
            float area_j = (detections[j].x2 - detections[j].x1) *
                           (detections[j].y2 - detections[j].y1);

            float iou = inter_area / (area_i + area_j - inter_area + 1e-6f);

            if (iou > iou_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(result);
}

void HailoYolov8Seg::generateMask(Detection& det,
                                   const float* proto,
                                   const float* mask_coeffs,
                                   int proto_h, int proto_w, int proto_c,
                                   int orig_width, int orig_height) {
    // Step 1: Compute mask at proto resolution (160x136) - FAST
    // Only compute within the box region in proto coords
    float scale_orig_to_proto_x = static_cast<float>(proto_w) / orig_width;
    float scale_orig_to_proto_y = static_cast<float>(proto_h) / orig_height;

    int proto_box_x1 = std::max(0, static_cast<int>(det.x1 * scale_orig_to_proto_x));
    int proto_box_y1 = std::max(0, static_cast<int>(det.y1 * scale_orig_to_proto_y));
    int proto_box_x2 = std::min(proto_w, static_cast<int>(det.x2 * scale_orig_to_proto_x) + 1);
    int proto_box_y2 = std::min(proto_h, static_cast<int>(det.y2 * scale_orig_to_proto_y) + 1);

    // Compute mask values only in box region at proto resolution
    std::vector<float> proto_mask(proto_h * proto_w, 0.0f);
    for (int py = proto_box_y1; py < proto_box_y2; ++py) {
        for (int px = proto_box_x1; px < proto_box_x2; ++px) {
            float sum = 0.0f;
            const float* proto_ptr = proto + (py * proto_w + px) * proto_c;
            for (int c = 0; c < proto_c; ++c) {
                sum += proto_ptr[c] * mask_coeffs[c];
            }
            proto_mask[py * proto_w + px] = sigmoid(sum);
        }
    }

    // Step 2: Resize to original image size (only box region)
    int orig_box_x1 = std::max(0, static_cast<int>(det.x1));
    int orig_box_y1 = std::max(0, static_cast<int>(det.y1));
    int orig_box_x2 = std::min(orig_width, static_cast<int>(det.x2));
    int orig_box_y2 = std::min(orig_height, static_cast<int>(det.y2));

    det.mask.resize(orig_width * orig_height, 0);
    det.mask_width = orig_width;
    det.mask_height = orig_height;

    float scale_proto_to_orig_x = static_cast<float>(proto_w) / orig_width;
    float scale_proto_to_orig_y = static_cast<float>(proto_h) / orig_height;

    for (int y = orig_box_y1; y < orig_box_y2; ++y) {
        for (int x = orig_box_x1; x < orig_box_x2; ++x) {
            int px = static_cast<int>(x * scale_proto_to_orig_x);
            int py = static_cast<int>(y * scale_proto_to_orig_y);
            px = std::min(px, proto_w - 1);
            py = std::min(py, proto_h - 1);

            if (proto_mask[py * proto_w + px] > 0.5f) {
                det.mask[y * orig_width + x] = 255;
            }
        }
    }
}

void HailoYolov8Seg::processOutput(
    const std::vector<std::pair<std::string, std::vector<float>>>& outputs,
    int orig_width, int orig_height,
    std::vector<Detection>& detections)
{
    // Build output map
    std::unordered_map<std::string, const std::vector<float>*> output_map;
    for (const auto& output : outputs) {
        output_map[output.first] = &output.second;
    }

    // Scale definitions
    std::vector<ScaleInfo> scales = {
        {impl_->model_prefix + "/conv44", impl_->model_prefix + "/conv45",
         impl_->model_prefix + "/conv46", 8},
        {impl_->model_prefix + "/conv60", impl_->model_prefix + "/conv61",
         impl_->model_prefix + "/conv62", 16},
        {impl_->model_prefix + "/conv73", impl_->model_prefix + "/conv74",
         impl_->model_prefix + "/conv75", 32}
    };

    std::string proto_name = impl_->model_prefix + "/conv48";

    // Get proto output
    auto proto_it = output_map.find(proto_name);
    if (proto_it == output_map.end()) {
        std::cerr << "Proto output not found" << std::endl;
        return;
    }
    const std::vector<float>& proto_data = *(proto_it->second);

    // Proto dimensions: (H, W, C) = (160, 136, 32) for 640x544 input
    int proto_h = input_height_ / 4;  // 160
    int proto_w = input_width_ / 4;   // 136
    int proto_c = 32;

    float scale_x = static_cast<float>(orig_width) / input_width_;
    float scale_y = static_cast<float>(orig_height) / input_height_;

    // Temporary storage for detections with mask coefficients
    struct TempDetection {
        Detection det;
        std::vector<float> mask_coeffs;
    };
    std::vector<TempDetection> temp_detections;

    // Process each scale
    for (const auto& scale : scales) {
        auto bbox_it = output_map.find(scale.bbox_name);
        auto cls_it = output_map.find(scale.cls_name);
        auto mask_it = output_map.find(scale.mask_name);

        if (bbox_it == output_map.end() || cls_it == output_map.end() ||
            mask_it == output_map.end()) {
            continue;
        }

        const std::vector<float>& bbox_data = *(bbox_it->second);
        const std::vector<float>& cls_data = *(cls_it->second);
        const std::vector<float>& mask_data = *(mask_it->second);

        int feat_h = input_height_ / scale.stride;
        int feat_w = input_width_ / scale.stride;
        int num_classes = 1;  // Single class for MLCC index
        int num_mask_coeffs = 32;

        for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {
                int idx = y * feat_w + x;

                // Get class score (assuming single class)
                float cls_score = cls_data[idx * num_classes];

                if (cls_score < conf_threshold_) continue;

                // DFL decode bbox
                const float* bbox_ptr = bbox_data.data() + idx * 64;
                float dfl_left = dflDecode(bbox_ptr, 0);
                float dfl_top = dflDecode(bbox_ptr, 16);
                float dfl_right = dflDecode(bbox_ptr, 32);
                float dfl_bottom = dflDecode(bbox_ptr, 48);

                float cx = (x + 0.5f) * scale.stride;
                float cy = (y + 0.5f) * scale.stride;

                float x1 = (cx - dfl_left * scale.stride) * scale_x;
                float y1 = (cy - dfl_top * scale.stride) * scale_y;
                float x2 = (cx + dfl_right * scale.stride) * scale_x;
                float y2 = (cy + dfl_bottom * scale.stride) * scale_y;

                TempDetection temp;
                temp.det.x1 = x1;
                temp.det.y1 = y1;
                temp.det.x2 = x2;
                temp.det.y2 = y2;
                temp.det.score = cls_score;
                temp.det.class_id = 0;

                // Store mask coefficients
                temp.mask_coeffs.resize(num_mask_coeffs);
                const float* mask_ptr = mask_data.data() + idx * num_mask_coeffs;
                std::copy(mask_ptr, mask_ptr + num_mask_coeffs, temp.mask_coeffs.begin());

                temp_detections.push_back(std::move(temp));
            }
        }
    }

    // Extract detections for NMS
    std::vector<Detection> nms_detections;
    for (auto& temp : temp_detections) {
        nms_detections.push_back(temp.det);
    }

    // Apply NMS
    nms(nms_detections);

    // Generate masks only for surviving detections
    for (auto& det : nms_detections) {
        // Find matching temp detection
        for (auto& temp : temp_detections) {
            if (std::abs(temp.det.x1 - det.x1) < 1.0f &&
                std::abs(temp.det.y1 - det.y1) < 1.0f &&
                std::abs(temp.det.score - det.score) < 0.001f) {

                generateMask(det, proto_data.data(), temp.mask_coeffs.data(),
                             proto_h, proto_w, proto_c, orig_width, orig_height);
                break;
            }
        }
        detections.push_back(det);
    }
}

InferenceResult HailoYolov8Seg::infer(const uint8_t* image_data,
                                       int width, int height) {
    InferenceResult result;
    result.original_width = width;
    result.original_height = height;

    if (!initialized_) {
        std::cerr << "Engine not initialized" << std::endl;
        return result;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    try {
        // Preprocess
        std::vector<uint8_t> input_data = preprocess(image_data, width, height);

        // Run inference using persistent VStreams (no creation overhead)
        auto infer_start = std::chrono::high_resolution_clock::now();

        // Write input to persistent input vstream
        auto write_status = impl_->input_vstreams[0].write(
            MemoryView(input_data.data(), input_data.size()));
        if (write_status != HAILO_SUCCESS) {
            std::cerr << "Failed to write input" << std::endl;
            return result;
        }

        // Read outputs from persistent output vstreams
        std::vector<std::pair<std::string, std::vector<float>>> outputs;
        for (auto& output_vstream : impl_->output_vstreams) {
            auto frame_size = output_vstream.get_frame_size();
            std::vector<float> output_data(frame_size / sizeof(float));
            auto read_status = output_vstream.read(
                MemoryView(output_data.data(), frame_size));
            if (read_status != HAILO_SUCCESS) {
                std::cerr << "Failed to read output" << std::endl;
                return result;
            }
            outputs.emplace_back(output_vstream.name(), std::move(output_data));
        }

        auto infer_end = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<float, std::milli>(
            infer_end - infer_start).count();

        // Post-process
        processOutput(outputs, width, height, result.detections);

    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<float, std::milli>(
        total_end - total_start).count();

    return result;
}

} // namespace hailo_yolov8
