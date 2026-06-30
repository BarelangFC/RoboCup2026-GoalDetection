cmake_minimum_required(VERSION 3.10)
project(goal-detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find TensorRT (JetPack 4.6: /usr/lib/aarch64-linux-gnu, /usr/include/aarch64-linux-gnu)
set(TENSORRT_ROOT /usr)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS ${TENSORRT_ROOT}/include/aarch64-linux-gnu
        ${TENSORRT_ROOT}/include)
find_library(TENSORRT_LIB nvinfer
  PATHS ${TENSORRT_ROOT}/lib/aarch64-linux-gnu
        ${TENSORRT_ROOT}/lib)

# Find CUDA (already on Jetson)
find_package(CUDA REQUIRED)

# Find OpenCV (system install on JP4.6)
find_package(OpenCV REQUIRED)

add_executable(goal-detector
  src/main.cpp
  src/trt_infer.cpp
  src/yolo.cpp
  src/geometry.cpp
  src/network.cpp
  src/camera.cpp
  src/goal_sender.cpp)

target_include_directories(goal-detector PRIVATE
  src
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(goal-detector
  ${TENSORRT_LIB}
  nvinfer_plugin
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  pthread)

# Optimize for Jetson Nano aarch64
target_compile_options(goal-detector PRIVATE -O3 -march=armv8-a -mtune=cortex-a57)
set_target_properties(goal-detector PROPERTIES LINK_FLAGS "-O3")
