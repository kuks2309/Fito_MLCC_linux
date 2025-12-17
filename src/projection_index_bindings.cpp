/**
 * @file projection_index_bindings.cpp
 * @brief pybind11 bindings for ProjectionIndexDetector
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "projection_index_detector.hpp"

namespace py = pybind11;
using namespace projection_detector;

PYBIND11_MODULE(projection_index_cpp, m) {
    m.doc() = "Projection-based MLCC Index detector (C++)";

    // IndexInfo structure
    py::class_<IndexInfo>(m, "IndexInfo")
        .def(py::init<>())
        .def_readwrite("center_y", &IndexInfo::center_y)
        .def_readwrite("top_y", &IndexInfo::top_y)
        .def_readwrite("bottom_y", &IndexInfo::bottom_y)
        .def_readwrite("confidence", &IndexInfo::confidence);

    // DetectionResult structure
    py::class_<DetectionResult>(m, "DetectionResult")
        .def(py::init<>())
        .def_readwrite("indices", &DetectionResult::indices)
        .def_readwrite("process_time_ms", &DetectionResult::process_time_ms)
        .def_readwrite("image_width", &DetectionResult::image_width)
        .def_readwrite("image_height", &DetectionResult::image_height);

    // ProjectionIndexDetector class
    py::class_<ProjectionIndexDetector>(m, "ProjectionIndexDetector")
        .def(py::init<>())

        // Main detect function with numpy array support
        .def("detect", [](ProjectionIndexDetector& self, py::array_t<uint8_t> image) {
            py::buffer_info buf = image.request();

            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Input must be a 3-channel BGR image (H x W x 3)");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

            return self.detect(data, width, height);
        }, py::arg("image"), "Detect indices from BGR image")

        // Parameter setters
        .def("set_smoothing_sigma", &ProjectionIndexDetector::setSmoothingSigma)
        .def("set_derivative_threshold", &ProjectionIndexDetector::setDerivativeThreshold)
        .def("set_min_index_height", &ProjectionIndexDetector::setMinIndexHeight)
        .def("set_max_index_height", &ProjectionIndexDetector::setMaxIndexHeight)
        .def("set_index_pitch", &ProjectionIndexDetector::setIndexPitch)
        .def("set_pitch_tolerance", &ProjectionIndexDetector::setPitchTolerance)
        .def("set_use_roi", &ProjectionIndexDetector::setUseROI)
        .def("set_roi", &ProjectionIndexDetector::setROI)

        // Parameter getters
        .def("get_smoothing_sigma", &ProjectionIndexDetector::getSmoothingSigma)
        .def("get_derivative_threshold", &ProjectionIndexDetector::getDerivativeThreshold)
        .def("get_min_index_height", &ProjectionIndexDetector::getMinIndexHeight)
        .def("get_max_index_height", &ProjectionIndexDetector::getMaxIndexHeight)
        .def("get_index_pitch", &ProjectionIndexDetector::getIndexPitch)
        .def("get_pitch_tolerance", &ProjectionIndexDetector::getPitchTolerance)

        // Debug data access (return as numpy arrays)
        .def("get_projection", [](const ProjectionIndexDetector& self) {
            const auto& proj = self.getProjection();
            return py::array_t<float>(proj.size(), proj.data());
        })
        .def("get_smoothed_projection", [](const ProjectionIndexDetector& self) {
            const auto& proj = self.getSmoothedProjection();
            return py::array_t<float>(proj.size(), proj.data());
        })
        .def("get_derivative", [](const ProjectionIndexDetector& self) {
            const auto& deriv = self.getDerivative();
            return py::array_t<float>(deriv.size(), deriv.data());
        })

        // Vertical edge detection using Sobel X filter
        .def("compute_vertical_edge", [](ProjectionIndexDetector& self,
                                         py::array_t<uint8_t> image,
                                         int scale_factor) {
            py::buffer_info buf = image.request();

            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Input must be a 3-channel BGR image (H x W x 3)");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

            auto [edge_data, process_time_ms] = self.computeVerticalEdge(data, width, height, scale_factor);

            // Return as numpy array (H x W grayscale)
            py::array_t<uint8_t> result({height, width});
            std::memcpy(result.mutable_data(), edge_data.data(), edge_data.size());

            return py::make_tuple(result, process_time_ms);
        }, py::arg("image"), py::arg("scale_factor") = 4,
           "Compute vertical edge using Sobel X filter. Returns (edge_image, process_time_ms)");
}
