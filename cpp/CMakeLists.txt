cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(SSD-FaceMask-TensorRT)

set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu")

list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()


find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_path(TENSORRT_PARSER_INCLUDE_DIR NvOnnxParser.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_PARSER_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNX_PARSER nvonnxparser
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)


set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNX_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

find_package(Threads)
find_package(CUDA REQUIRED)
find_library(NVINFER NAMES libnvinfer.so)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

if(NVINFER)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVONNXPARSERS: ${NVONNXPARSERS}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

find_library(LIBNVINFER nvinfer /usr/lib/x86_64-linux-gnu/)
find_library(LIBNVONNXPARSERS nvonnxparser /usr/lib/x86_64-linux-gnu/)

#find_package(OpenCV REQUIRED)
#find_package(TensorRT REQUIRED)

# TARGETS --------------------------------------------------------------------------------------------------------------
add_executable(onnx2trt_converter onnx2trt_converter.cpp)
target_include_directories(onnx2trt_converter PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(onnx2trt_converter ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} ${NVINFER_LIBRARY})
target_link_libraries(onnx2trt_converter ${LIBNVINFER})
target_link_libraries(onnx2trt_converter ${LIBNVONNXPARSERS})
