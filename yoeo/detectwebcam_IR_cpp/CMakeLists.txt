# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.10)
project(detectwebcam_IR_cpp)
set (TARGET_NAME "detectwebcam_IR_cpp")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

# Find OpenCV components if exist
find_package(OpenCV REQUIRED)
message(${OpenCV_DIR})
find_package(OpenVINO REQUIRED PATHS /opt/intel/openvino_2022.3.0/runtime/)
message(${OpenVINO_DIR})

add_executable(${TARGET_NAME} detectwebcam_IR.cpp) 

target_link_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime PRIVATE ${OpenCV_LIBRARIES} ie_samples_utils)
