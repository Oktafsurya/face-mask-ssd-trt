cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(SSD-FaceMask-TensorRT)

find_package(Threads)
find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(TensorRT REQUIRED)

# TARGETS --------------------------------------------------------------------------------------------------------------
add_executable(onnx2trt_converter onnx2trt_converter.cpp)
target_include_directories(onnx2trt_converter PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(onnx2trt_converter PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
