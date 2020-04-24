project(cublas_test)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV 4 REQUIRED)

find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )

add_executable(trt_ssd main.cpp)
target_link_libraries(trt_ssd
    ${OpenCV_LIBS}
    ${CUBLAS_LIB}
    ${NVINFER}
    ${NVONNXPARSER} 
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
  )
