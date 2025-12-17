cmake_minimum_required(VERSION 3.14)
project(hailo_yolov8_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Hailo SDK paths
set(HAILO_INCLUDE_DIR "/usr/include/hailo")
set(HAILO_LIB_DIR "/usr/lib")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HAILO_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Link directories
link_directories(${HAILO_LIB_DIR})

# ==========================================
# Target 1: Hailo YOLOv8 Segmentation Module
# ==========================================
set(HAILO_SOURCES
    hailo_yolov8_seg.cpp
    python_bindings.cpp
)

pybind11_add_module(hailo_yolov8_cpp ${HAILO_SOURCES})

target_link_libraries(hailo_yolov8_cpp PRIVATE
    hailort
)

set_target_properties(hailo_yolov8_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../Hailo_H8
)

# ==========================================
# Target 2: Projection Index Detector Module
# ==========================================
set(PROJECTION_SOURCES
    projection_index_detector.cpp
    projection_index_bindings.cpp
)

pybind11_add_module(projection_index_cpp ${PROJECTION_SOURCES})

# No external dependencies, pure C++
set_target_properties(projection_index_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
)

# ==========================================
# Target 3: Hough Line Detector Module
# ==========================================
set(HOUGH_SOURCES
    index_hough_line_detector.cpp
    index_hough_line_bindings.cpp
)

pybind11_add_module(index_hough_line_cpp ${HOUGH_SOURCES})

set_target_properties(index_hough_line_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
)

# Install targets
install(TARGETS hailo_yolov8_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Hailo_H8
)

install(TARGETS projection_index_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
)

install(TARGETS index_hough_line_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
)
