/**
 * @file projection_index_detector.cpp
 * @brief Projection-based MLCC Index detector implementation
 */

#include "projection_index_detector.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <climits>

namespace projection_detector {

ProjectionIndexDetector::ProjectionIndexDetector() = default;

void ProjectionIndexDetector::setROI(int x1, int y1, int x2, int y2) {
    roi_x1_ = x1;
    roi_y1_ = y1;
    roi_x2_ = x2;
    roi_y2_ = y2;
}

DetectionResult ProjectionIndexDetector::detect(const uint8_t* image_data, int width, int height) {
    auto start_time = std::chrono::high_resolution_clock::now();

    DetectionResult result;
    result.image_width = width;
    result.image_height = height;

    // Step 1: Convert BGR to grayscale (inline for speed)
    std::vector<uint8_t> gray(width * height);
    for (int i = 0; i < width * height; ++i) {
        // BGR to Gray: 0.114*B + 0.587*G + 0.299*R
        int idx = i * 3;
        gray[i] = static_cast<uint8_t>(
            0.114f * image_data[idx] +
            0.587f * image_data[idx + 1] +
            0.299f * image_data[idx + 2]
        );
    }

    // Step 2: Compute horizontal projection
    computeHorizontalProjection(gray.data(), width, height);

    // Step 3: Simple threshold method - (mean + max) / 2
    float proj_sum = 0.0f;
    float proj_max = 0.0f;
    for (const auto& p : projection_) {
        proj_sum += p;
        proj_max = std::max(proj_max, p);
    }
    float proj_mean = proj_sum / projection_.size();
    float threshold = (proj_mean + proj_max) / 2.0f;

    // Step 4: Find contiguous regions above threshold
    result.indices = findRegionsAboveThreshold(threshold, proj_max);

    // Step 5: Validate pitch if enabled
    if (index_pitch_ > 0 && result.indices.size() > 1) {
        validatePitch(result.indices);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.process_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    return result;
}

void ProjectionIndexDetector::computeHorizontalProjection(const uint8_t* gray_data, int width, int height) {
    projection_.resize(height);

    int start_x = use_roi_ ? std::max(0, roi_x1_) : 0;
    int end_x = use_roi_ ? std::min(width, roi_x2_) : width;
    int start_y = use_roi_ ? std::max(0, roi_y1_) : 0;
    int end_y = use_roi_ ? std::min(height, roi_y2_) : height;

    float scale = 1.0f / (end_x - start_x);

    for (int y = 0; y < height; ++y) {
        if (y < start_y || y >= end_y) {
            projection_[y] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        const uint8_t* row = gray_data + y * width;
        for (int x = start_x; x < end_x; ++x) {
            sum += row[x];
        }
        projection_[y] = sum * scale;  // Normalize by width
    }
}

void ProjectionIndexDetector::gaussianSmooth(int kernel_size) {
    int half = kernel_size / 2;
    int height = static_cast<int>(projection_.size());
    smoothed_projection_.resize(height);

    // Create Gaussian kernel
    std::vector<float> kernel(kernel_size);
    float sigma_sq_2 = 2.0f * smoothing_sigma_ * smoothing_sigma_;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        float x = static_cast<float>(i - half);
        kernel[i] = std::exp(-x * x / sigma_sq_2);
        sum += kernel[i];
    }
    // Normalize kernel
    for (auto& k : kernel) {
        k /= sum;
    }

    // Apply convolution
    for (int y = 0; y < height; ++y) {
        float value = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int idx = y + k - half;
            if (idx < 0) idx = 0;
            if (idx >= height) idx = height - 1;
            value += projection_[idx] * kernel[k];
        }
        smoothed_projection_[y] = value;
    }
}

void ProjectionIndexDetector::computeDerivative() {
    int height = static_cast<int>(smoothed_projection_.size());
    derivative_.resize(height);

    // Central difference (except at boundaries)
    derivative_[0] = smoothed_projection_[1] - smoothed_projection_[0];
    for (int y = 1; y < height - 1; ++y) {
        derivative_[y] = (smoothed_projection_[y + 1] - smoothed_projection_[y - 1]) * 0.5f;
    }
    derivative_[height - 1] = smoothed_projection_[height - 1] - smoothed_projection_[height - 2];
}

std::vector<int> ProjectionIndexDetector::findZeroCrossings(float threshold) {
    // This function now finds local extrema instead of zero crossings
    // Returns: positive values for local maxima, negative values for local minima
    std::vector<int> extrema;
    int height = static_cast<int>(derivative_.size());

    for (int y = 1; y < height - 1; ++y) {
        float curr = derivative_[y];
        float prev = derivative_[y - 1];
        float next = derivative_[y + 1];

        // Local maximum (Index top boundary)
        if (curr > threshold && curr > prev && curr > next) {
            extrema.push_back(y);  // Positive = local max
        }
        // Local minimum (Index bottom boundary)
        if (curr < -threshold && curr < prev && curr < next) {
            extrema.push_back(-y);  // Negative = local min
        }
    }

    return extrema;
}

std::vector<IndexInfo> ProjectionIndexDetector::pairBoundaries(const std::vector<int>& extrema) {
    std::vector<IndexInfo> indices;

    // Separate maxima and minima
    std::vector<int> local_max, local_min;
    for (int e : extrema) {
        if (e > 0) local_max.push_back(e);
        else local_min.push_back(-e);
    }

    // Pair maxima (top) with minima (bottom)
    std::vector<bool> used_min(local_min.size(), false);

    for (int max_y : local_max) {
        int best_idx = -1;
        int best_dist = INT_MAX;

        for (size_t i = 0; i < local_min.size(); ++i) {
            if (used_min[i]) continue;
            int min_y = local_min[i];
            if (min_y > max_y) {  // Minimum must be after maximum
                int dist = min_y - max_y;
                if (dist >= min_index_height_ && dist <= max_index_height_) {
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx = static_cast<int>(i);
                    }
                }
            }
        }

        if (best_idx >= 0) {
            used_min[best_idx] = true;
            int top_y = max_y;
            int bottom_y = local_min[best_idx];

            IndexInfo info;
            info.top_y = top_y;
            info.bottom_y = bottom_y;
            info.center_y = (top_y + bottom_y) / 2;

            // Confidence based on peak height
            float peak_value = 0.0f;
            for (int y = top_y; y <= bottom_y && y < static_cast<int>(smoothed_projection_.size()); ++y) {
                peak_value = std::max(peak_value, smoothed_projection_[y]);
            }
            float max_proj = *std::max_element(smoothed_projection_.begin(), smoothed_projection_.end());
            info.confidence = (max_proj > 0) ? peak_value / max_proj : 0.0f;

            indices.push_back(info);
        }
    }

    // Sort by Y coordinate
    std::sort(indices.begin(), indices.end(),
              [](const IndexInfo& a, const IndexInfo& b) {
                  return a.center_y < b.center_y;
              });

    return indices;
}

void ProjectionIndexDetector::validatePitch(std::vector<IndexInfo>& indices) {
    if (indices.size() < 2 || index_pitch_ <= 0) return;

    float min_pitch = index_pitch_ * (1.0f - pitch_tolerance_);
    float max_pitch = index_pitch_ * (1.0f + pitch_tolerance_);

    std::vector<IndexInfo> validated;

    // Find first valid pair
    int first_valid = -1;
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        float pitch = static_cast<float>(indices[i + 1].center_y - indices[i].center_y);
        if (pitch >= min_pitch && pitch <= max_pitch) {
            first_valid = static_cast<int>(i);
            break;
        }
    }

    if (first_valid >= 0) {
        // Start from first valid
        validated.push_back(indices[first_valid]);

        // Add subsequent indices that match pitch
        for (size_t i = first_valid + 1; i < indices.size(); ++i) {
            float pitch = static_cast<float>(indices[i].center_y - validated.back().center_y);
            if (pitch >= min_pitch && pitch <= max_pitch) {
                validated.push_back(indices[i]);
            }
        }
    } else if (!indices.empty()) {
        // No valid pitch found, return first index only
        validated.push_back(indices[0]);
    }

    indices = std::move(validated);
}

std::vector<IndexInfo> ProjectionIndexDetector::findRegionsAboveThreshold(float threshold, float proj_max) {
    std::vector<IndexInfo> indices;
    int height = static_cast<int>(projection_.size());

    bool in_region = false;
    int top_y = 0;

    for (int y = 0; y < height; ++y) {
        if (projection_[y] > threshold && !in_region) {
            // Region start
            in_region = true;
            top_y = y;
        } else if (projection_[y] <= threshold && in_region) {
            // Region end
            in_region = false;
            int bottom_y = y - 1;
            int region_height = bottom_y - top_y + 1;

            if (region_height >= min_index_height_ && region_height <= max_index_height_) {
                IndexInfo info;
                info.top_y = top_y;
                info.bottom_y = bottom_y;
                info.center_y = (top_y + bottom_y) / 2;

                // Find peak value in region
                float peak_value = 0.0f;
                for (int py = top_y; py <= bottom_y; ++py) {
                    peak_value = std::max(peak_value, projection_[py]);
                }
                info.confidence = (proj_max > 0) ? peak_value / proj_max : 0.0f;

                indices.push_back(info);
            }
        }
    }

    // Handle region at end of image
    if (in_region) {
        int bottom_y = height - 1;
        int region_height = bottom_y - top_y + 1;

        if (region_height >= min_index_height_ && region_height <= max_index_height_) {
            IndexInfo info;
            info.top_y = top_y;
            info.bottom_y = bottom_y;
            info.center_y = (top_y + bottom_y) / 2;

            float peak_value = 0.0f;
            for (int py = top_y; py <= bottom_y; ++py) {
                peak_value = std::max(peak_value, projection_[py]);
            }
            info.confidence = (proj_max > 0) ? peak_value / proj_max : 0.0f;

            indices.push_back(info);
        }
    }

    return indices;
}

std::pair<std::vector<uint8_t>, float> ProjectionIndexDetector::computeVerticalEdge(
    const uint8_t* image_data, int width, int height, int scale_factor) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Convert BGR to grayscale
    std::vector<uint8_t> gray(width * height);
    for (int i = 0; i < width * height; ++i) {
        int idx = i * 3;
        gray[i] = static_cast<uint8_t>(
            0.114f * image_data[idx] +
            0.587f * image_data[idx + 1] +
            0.299f * image_data[idx + 2]
        );
    }

    // Step 2: Downscale image (1/scale_factor)
    int small_w = width / scale_factor;
    int small_h = height / scale_factor;
    std::vector<uint8_t> small(small_w * small_h);

    // Simple area averaging for downscale
    for (int sy = 0; sy < small_h; ++sy) {
        for (int sx = 0; sx < small_w; ++sx) {
            int sum = 0;
            for (int dy = 0; dy < scale_factor; ++dy) {
                for (int dx = 0; dx < scale_factor; ++dx) {
                    int src_y = sy * scale_factor + dy;
                    int src_x = sx * scale_factor + dx;
                    sum += gray[src_y * width + src_x];
                }
            }
            small[sy * small_w + sx] = static_cast<uint8_t>(sum / (scale_factor * scale_factor));
        }
    }

    // Step 3: Apply Sobel X filter (3x3 kernel)
    // Sobel X kernel: [-1 0 1]
    //                 [-2 0 2]
    //                 [-1 0 1]
    std::vector<int16_t> sobel_result(small_w * small_h, 0);

    for (int y = 1; y < small_h - 1; ++y) {
        for (int x = 1; x < small_w - 1; ++x) {
            int gx = -small[(y - 1) * small_w + (x - 1)] + small[(y - 1) * small_w + (x + 1)]
                   - 2 * small[y * small_w + (x - 1)] + 2 * small[y * small_w + (x + 1)]
                   - small[(y + 1) * small_w + (x - 1)] + small[(y + 1) * small_w + (x + 1)];
            sobel_result[y * small_w + x] = static_cast<int16_t>(gx);
        }
    }

    // Step 4: Convert to absolute value and normalize to uint8
    std::vector<uint8_t> edge_small(small_w * small_h);
    for (int i = 0; i < small_w * small_h; ++i) {
        int abs_val = std::abs(sobel_result[i]);
        edge_small[i] = static_cast<uint8_t>(std::min(255, abs_val));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float process_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Step 5: Upscale back to original size (bilinear interpolation)
    std::vector<uint8_t> edge_full(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float src_x = static_cast<float>(x) / scale_factor;
            float src_y = static_cast<float>(y) / scale_factor;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, small_w - 1);
            int y1 = std::min(y0 + 1, small_h - 1);

            float fx = src_x - x0;
            float fy = src_y - y0;

            float v00 = edge_small[y0 * small_w + x0];
            float v10 = edge_small[y0 * small_w + x1];
            float v01 = edge_small[y1 * small_w + x0];
            float v11 = edge_small[y1 * small_w + x1];

            float value = v00 * (1 - fx) * (1 - fy) +
                         v10 * fx * (1 - fy) +
                         v01 * (1 - fx) * fy +
                         v11 * fx * fy;

            edge_full[y * width + x] = static_cast<uint8_t>(value);
        }
    }

    return {edge_full, process_time_ms};
}

} // namespace projection_detector
