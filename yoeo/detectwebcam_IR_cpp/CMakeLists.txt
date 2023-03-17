cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
project(detectwebcam_IR)

find_package(OpenVINO REQUIRED PATHS /opt/intel/openvino_2022.3.0/runtime/)
# message(${OpenVINO_LIBRARIES})
find_package(ngraph REQUIRED PATHS /opt/intel/openvino_2022.3.0/runtime/)
add_executable(detectwebcam_IR detectwebcam_IR.cpp)

target_link_libraries(detectwebcam_IR PRIVATE openvino::runtime "${ngraph_LIBRARIES}")