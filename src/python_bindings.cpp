/**
 * @file python_bindings.cpp
 * @brief pybind11 bindings for HailoYolov8Seg
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hailo_yolov8_seg.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hailo_yolov8_cpp, m) {
    m.doc() = "Hailo-8 YOLOv8-seg inference engine (C++ implementation)";

    // Detection structure
    py::class_<hailo_yolov8::Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("x1", &hailo_yolov8::Detection::x1)
        .def_readwrite("y1", &hailo_yolov8::Detection::y1)
        .def_readwrite("x2", &hailo_yolov8::Detection::x2)
        .def_readwrite("y2", &hailo_yolov8::Detection::y2)
        .def_readwrite("score", &hailo_yolov8::Detection::score)
        .def_readwrite("class_id", &hailo_yolov8::Detection::class_id)
        .def_property_readonly("mask_numpy", [](const hailo_yolov8::Detection& d) {
            if (d.mask.empty()) {
                return py::array_t<uint8_t>();
            }
            // Return as numpy array with proper shape
            std::vector<ssize_t> shape = {d.mask_height, d.mask_width};
            std::vector<ssize_t> strides = {
                static_cast<ssize_t>(d.mask_width * sizeof(uint8_t)),
                static_cast<ssize_t>(sizeof(uint8_t))
            };
            return py::array_t<uint8_t>(shape, strides, d.mask.data());
        })
        .def("get_box", [](const hailo_yolov8::Detection& d) {
            return py::make_tuple(d.x1, d.y1, d.x2, d.y2);
        });

    // InferenceResult structure
    py::class_<hailo_yolov8::InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readonly("detections", &hailo_yolov8::InferenceResult::detections)
        .def_readonly("original_width", &hailo_yolov8::InferenceResult::original_width)
        .def_readonly("original_height", &hailo_yolov8::InferenceResult::original_height)
        .def_readonly("inference_time_ms", &hailo_yolov8::InferenceResult::inference_time_ms)
        .def_readonly("total_time_ms", &hailo_yolov8::InferenceResult::total_time_ms)
        .def_property_readonly("count", [](const hailo_yolov8::InferenceResult& r) {
            return r.detections.size();
        })
        // Compatibility with existing Python code
        .def("to_dict", [](const hailo_yolov8::InferenceResult& r) {
            py::list boxes, scores, class_ids, masks;
            for (const auto& det : r.detections) {
                boxes.append(py::make_tuple(det.x1, det.y1, det.x2, det.y2));
                scores.append(det.score);
                class_ids.append(det.class_id);

                if (!det.mask.empty()) {
                    std::vector<ssize_t> shape = {det.mask_height, det.mask_width};
                    std::vector<ssize_t> strides = {
                        static_cast<ssize_t>(det.mask_width * sizeof(uint8_t)),
                        static_cast<ssize_t>(sizeof(uint8_t))
                    };
                    // Create a copy for Python ownership
                    auto mask_array = py::array_t<uint8_t>(shape);
                    auto buf = mask_array.mutable_data();
                    std::memcpy(buf, det.mask.data(), det.mask.size());
                    // Convert to bool array
                    py::array_t<bool> bool_mask(shape);
                    auto bool_buf = bool_mask.mutable_data();
                    for (size_t i = 0; i < det.mask.size(); ++i) {
                        bool_buf[i] = det.mask[i] > 0;
                    }
                    masks.append(bool_mask);
                } else {
                    masks.append(py::none());
                }
            }

            py::dict result;
            result["boxes"] = boxes;
            result["scores"] = scores;
            result["class_ids"] = class_ids;
            result["masks"] = masks;
            result["original_size"] = py::make_tuple(r.original_height, r.original_width);
            result["count"] = r.detections.size();
            result["inference_time_ms"] = r.inference_time_ms;
            result["total_time_ms"] = r.total_time_ms;
            return result;
        });

    // Main inference class
    py::class_<hailo_yolov8::HailoYolov8Seg>(m, "HailoYolov8Seg")
        .def(py::init<const std::string&, int, int>(),
             py::arg("hef_path"),
             py::arg("input_height") = 640,
             py::arg("input_width") = 544)
        .def("initialize", &hailo_yolov8::HailoYolov8Seg::initialize)
        .def("cleanup", &hailo_yolov8::HailoYolov8Seg::cleanup)
        .def("is_initialized", &hailo_yolov8::HailoYolov8Seg::isInitialized)
        .def("set_conf_threshold", &hailo_yolov8::HailoYolov8Seg::setConfThreshold)
        .def("set_iou_threshold", &hailo_yolov8::HailoYolov8Seg::setIouThreshold)
        .def("get_conf_threshold", &hailo_yolov8::HailoYolov8Seg::getConfThreshold)
        .def("get_iou_threshold", &hailo_yolov8::HailoYolov8Seg::getIouThreshold)
        .def("infer", [](hailo_yolov8::HailoYolov8Seg& self,
                         py::array_t<uint8_t> image) {
            // Expect BGR image in HWC format
            auto buf = image.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Image must be HWC format with 3 channels (BGR)");
            }

            int height = buf.shape[0];
            int width = buf.shape[1];

            return self.infer(static_cast<uint8_t*>(buf.ptr), width, height);
        }, py::arg("image"))
        // Convenience method that returns dict for compatibility
        .def("infer_dict", [](hailo_yolov8::HailoYolov8Seg& self,
                              py::array_t<uint8_t> image) {
            auto buf = image.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Image must be HWC format with 3 channels (BGR)");
            }

            int height = buf.shape[0];
            int width = buf.shape[1];

            auto result = self.infer(static_cast<uint8_t*>(buf.ptr), width, height);

            // Convert to dict
            py::list boxes, scores, class_ids, masks;
            for (const auto& det : result.detections) {
                boxes.append(py::make_tuple(det.x1, det.y1, det.x2, det.y2));
                scores.append(det.score);
                class_ids.append(det.class_id);

                if (!det.mask.empty()) {
                    std::vector<ssize_t> shape = {det.mask_height, det.mask_width};
                    py::array_t<bool> bool_mask(shape);
                    auto bool_buf = bool_mask.mutable_data();
                    for (size_t i = 0; i < det.mask.size(); ++i) {
                        bool_buf[i] = det.mask[i] > 0;
                    }
                    masks.append(bool_mask);
                } else {
                    masks.append(py::none());
                }
            }

            py::dict dict_result;
            dict_result["boxes"] = boxes;
            dict_result["scores"] = scores;
            dict_result["class_ids"] = class_ids;
            dict_result["masks"] = masks;
            dict_result["original_size"] = py::make_tuple(result.original_height, result.original_width);
            dict_result["count"] = result.detections.size();
            dict_result["inference_time_ms"] = result.inference_time_ms;
            dict_result["total_time_ms"] = result.total_time_ms;
            return dict_result;
        }, py::arg("image"));
}
