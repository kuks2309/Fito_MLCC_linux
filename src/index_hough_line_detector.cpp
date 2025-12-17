/**
 * @file index_hough_line_detector.cpp
 * @brief Hough Line Transform implementation for vertical line detection
 */

#include "index_hough_line_detector.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace hough_detector {

IndexHoughLineDetector::IndexHoughLineDetector() = default;

HoughResult IndexHoughLineDetector::detect(const uint8_t* edge_data, int width, int height) {
    auto start_time = std::chrono::high_resolution_clock::now();

    HoughResult result;
    result.is_aligned = false;
    result.mean_angle_deg = 0.0f;

    // Compute Hough accumulator
    std::vector<int> accumulator;
    int num_rho, num_theta;
    float theta_min, theta_step, rho_max;

    computeAccumulator(edge_data, width, height, accumulator,
                      num_rho, num_theta, theta_min, theta_step, rho_max);

    // Extract lines from accumulator
    result.lines = extractLines(accumulator, num_rho, num_theta,
                                theta_min, theta_step, rho_max, width, height);

    // Calculate mean angle
    if (!result.lines.empty()) {
        float sum_angle = 0.0f;
        for (const auto& line : result.lines) {
            sum_angle += line.angle_deg;
        }
        result.mean_angle_deg = sum_angle / result.lines.size();

        // Check if aligned (mean angle within ±5°)
        result.is_aligned = std::abs(result.mean_angle_deg) < 5.0f;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.process_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    return result;
}

HoughResult IndexHoughLineDetector::detectFromImage(const uint8_t* image_data, int width, int height, int scale_factor) {
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

    // Step 2: Downscale image
    int small_w = width / scale_factor;
    int small_h = height / scale_factor;
    std::vector<uint8_t> small(small_w * small_h);

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

    // Step 3: Sobel X for vertical edges
    std::vector<uint8_t> edge(small_w * small_h, 0);
    for (int y = 1; y < small_h - 1; ++y) {
        for (int x = 1; x < small_w - 1; ++x) {
            int gx = -small[(y - 1) * small_w + (x - 1)] + small[(y - 1) * small_w + (x + 1)]
                   - 2 * small[y * small_w + (x - 1)] + 2 * small[y * small_w + (x + 1)]
                   - small[(y + 1) * small_w + (x - 1)] + small[(y + 1) * small_w + (x + 1)];
            edge[y * small_w + x] = static_cast<uint8_t>(std::min(255, std::abs(gx)));
        }
    }

    // Step 4: Run Hough detection on edge image
    HoughResult result = detect(edge.data(), small_w, small_h);

    // Scale line endpoints back to original size
    for (auto& line : result.lines) {
        line.x1 *= scale_factor;
        line.y1 *= scale_factor;
        line.x2 *= scale_factor;
        line.y2 *= scale_factor;
        line.rho *= scale_factor;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.process_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    return result;
}

void IndexHoughLineDetector::computeAccumulator(const uint8_t* edge_data, int width, int height,
                                                std::vector<int>& accumulator, int& num_rho, int& num_theta,
                                                float& theta_min, float& theta_step, float& rho_max) {
    const float PI = 3.14159265358979f;

    // Standard Hough Transform: rho = x*cos(theta) + y*sin(theta)
    // - theta = 0°:  rho = x  → VERTICAL line at x = rho
    // - theta = 90°: rho = y  → HORIZONTAL line at y = rho
    //
    // For near-vertical lines (±15° from vertical), use theta around 0°
    // -15° to +15° means lines tilted slightly from perfect vertical
    float theta_min_rad = min_angle_deg_ * PI / 180.0f;  // -15° = -0.262 rad
    float theta_max_rad = max_angle_deg_ * PI / 180.0f;  // +15° = +0.262 rad
    float theta_step_rad = theta_resolution_deg_ * PI / 180.0f;

    theta_min = theta_min_rad;
    theta_step = theta_step_rad;

    num_theta = static_cast<int>((theta_max_rad - theta_min_rad) / theta_step_rad) + 1;
    rho_max = std::sqrt(static_cast<float>(width * width + height * height));
    num_rho = static_cast<int>(2 * rho_max / rho_resolution_) + 1;

    accumulator.assign(num_rho * num_theta, 0);

    // Pre-compute cos/sin values
    std::vector<float> cos_theta(num_theta);
    std::vector<float> sin_theta(num_theta);
    for (int t = 0; t < num_theta; ++t) {
        float theta = theta_min_rad + t * theta_step_rad;
        cos_theta[t] = std::cos(theta);
        sin_theta[t] = std::sin(theta);
    }

    // Vote in accumulator
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edge_data[y * width + x] >= edge_threshold_) {
                for (int t = 0; t < num_theta; ++t) {
                    float rho = x * cos_theta[t] + y * sin_theta[t];
                    int rho_idx = static_cast<int>((rho + rho_max) / rho_resolution_);
                    if (rho_idx >= 0 && rho_idx < num_rho) {
                        accumulator[rho_idx * num_theta + t]++;
                    }
                }
            }
        }
    }
}

std::vector<LineInfo> IndexHoughLineDetector::extractLines(const std::vector<int>& accumulator,
                                                           int num_rho, int num_theta,
                                                           float theta_min, float theta_step,
                                                           float rho_max, int width, int height) {
    const float PI = 3.14159265358979f;
    std::vector<LineInfo> lines;

    // Find peaks in accumulator
    std::vector<std::pair<int, int>> peaks;  // (votes, index)

    for (int r = 0; r < num_rho; ++r) {
        for (int t = 0; t < num_theta; ++t) {
            int votes = accumulator[r * num_theta + t];
            if (votes >= accumulator_threshold_) {
                // Check if local maximum (3x3 neighborhood)
                bool is_max = true;
                for (int dr = -1; dr <= 1 && is_max; ++dr) {
                    for (int dt = -1; dt <= 1 && is_max; ++dt) {
                        if (dr == 0 && dt == 0) continue;
                        int nr = r + dr;
                        int nt = t + dt;
                        if (nr >= 0 && nr < num_rho && nt >= 0 && nt < num_theta) {
                            if (accumulator[nr * num_theta + nt] > votes) {
                                is_max = false;
                            }
                        }
                    }
                }
                if (is_max) {
                    peaks.push_back({votes, r * num_theta + t});
                }
            }
        }
    }

    // Sort by votes (descending)
    std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Extract top lines
    int num_lines = std::min(static_cast<int>(peaks.size()), max_lines_);
    for (int i = 0; i < num_lines; ++i) {
        int idx = peaks[i].second;
        int r = idx / num_theta;
        int t = idx % num_theta;

        float rho = r * rho_resolution_ - rho_max;
        float theta = theta_min + t * theta_step;

        LineInfo line;
        line.rho = rho;
        line.theta = theta;
        line.angle_deg = theta * 180.0f / PI;  // Angle from vertical (0° = vertical)
        line.votes = peaks[i].first;

        // Calculate line endpoints for near-vertical lines (theta near 0°)
        // Line equation: x*cos(theta) + y*sin(theta) = rho
        // Solve for x: x = (rho - y*sin(theta)) / cos(theta)
        //
        // For theta near 0°: cos(theta) ≈ 1, sin(theta) ≈ small
        // So x ≈ rho - y*sin(theta)  (nearly vertical line at x ≈ rho)
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        // For near-vertical lines, calculate x at y=0 and y=height-1
        // cos(theta) is always > 0.96 for |theta| <= 15°, so division is safe
        line.y1 = 0;
        line.x1 = static_cast<int>((rho - line.y1 * sin_t) / cos_t);
        line.y2 = height - 1;
        line.x2 = static_cast<int>((rho - line.y2 * sin_t) / cos_t);

        // Clamp to image bounds
        line.x1 = std::max(0, std::min(width - 1, line.x1));
        line.x2 = std::max(0, std::min(width - 1, line.x2));
        line.y1 = std::max(0, std::min(height - 1, line.y1));
        line.y2 = std::max(0, std::min(height - 1, line.y2));

        lines.push_back(line);
    }

    return lines;
}

} // namespace hough_detector
