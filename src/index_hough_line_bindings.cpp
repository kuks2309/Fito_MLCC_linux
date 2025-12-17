/**
 * @file index_hough_line_bindings.cpp
 * @brief pybind11 bindings for IndexHoughLineDetector
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "index_hough_line_detector.hpp"

namespace py = pybind11;
using namespace hough_detector;

PYBIND11_MODULE(index_hough_line_cpp, m) {
    m.doc() = "Hough Line detector for vertical lines (C++)";

    // LineInfo structure
    py::class_<LineInfo>(m, "LineInfo")
        .def(py::init<>())
        .def_readwrite("rho", &LineInfo::rho)
        .def_readwrite("theta", &LineInfo::theta)
        .def_readwrite("angle_deg", &LineInfo::angle_deg)
        .def_readwrite("votes", &LineInfo::votes)
        .def_readwrite("x1", &LineInfo::x1)
        .def_readwrite("y1", &LineInfo::y1)
        .def_readwrite("x2", &LineInfo::x2)
        .def_readwrite("y2", &LineInfo::y2);

    // HoughResult structure
    py::class_<HoughResult>(m, "HoughResult")
        .def(py::init<>())
        .def_readwrite("lines", &HoughResult::lines)
        .def_readwrite("process_time_ms", &HoughResult::process_time_ms)
        .def_readwrite("mean_angle_deg", &HoughResult::mean_angle_deg)
        .def_readwrite("is_aligned", &HoughResult::is_aligned);

    // IndexHoughLineDetector class
    py::class_<IndexHoughLineDetector>(m, "IndexHoughLineDetector")
        .def(py::init<>())

        // Detect from edge image
        .def("detect", [](IndexHoughLineDetector& self, py::array_t<uint8_t> edge_image) {
            py::buffer_info buf = edge_image.request();

            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be a 2D grayscale edge image (H x W)");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

            return self.detect(data, width, height);
        }, py::arg("edge_image"), "Detect lines from grayscale edge image")

        // Detect from BGR image (includes edge detection)
        .def("detect_from_image", [](IndexHoughLineDetector& self,
                                     py::array_t<uint8_t> image,
                                     int scale_factor) {
            py::buffer_info buf = image.request();

            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("Input must be a 3-channel BGR image (H x W x 3)");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

            return self.detectFromImage(data, width, height, scale_factor);
        }, py::arg("image"), py::arg("scale_factor") = 4,
           "Detect lines from BGR image (includes edge detection)")

        // Parameter setters
        .def("set_angle_range", &IndexHoughLineDetector::setAngleRange,
             py::arg("min_deg"), py::arg("max_deg"),
             "Set angle range from vertical (-15 to +15 degrees default)")
        .def("set_edge_threshold", &IndexHoughLineDetector::setEdgeThreshold)
        .def("set_accumulator_threshold", &IndexHoughLineDetector::setAccumulatorThreshold)
        .def("set_rho_resolution", &IndexHoughLineDetector::setRhoResolution)
        .def("set_theta_resolution", &IndexHoughLineDetector::setThetaResolution)
        .def("set_max_lines", &IndexHoughLineDetector::setMaxLines)

        // Parameter getters
        .def("get_min_angle_deg", &IndexHoughLineDetector::getMinAngleDeg)
        .def("get_max_angle_deg", &IndexHoughLineDetector::getMaxAngleDeg)
        .def("get_edge_threshold", &IndexHoughLineDetector::getEdgeThreshold)
        .def("get_accumulator_threshold", &IndexHoughLineDetector::getAccumulatorThreshold);
}
