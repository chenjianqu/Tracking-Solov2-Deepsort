cmake_minimum_required(VERSION 3.20)
project(Tracking_Solov2_Deepsort)

set(CMAKE_CXX_STANDARD 17)





find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})
message("spdlog version:${spdlog_VERSION}")
message("spdlog dir:${spdlog_DIR}")

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV DIR:  ${OpenCV_DIR}")
message("OpenCV OpenCV_VERSION:  ${OpenCV_VERSION}")


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIR})

#Libtorch
set(CMAKE_PREFIX_PATH "/usr/local/lib;/home/chen/app/libtorch")

find_package(TorchVision REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

#CUDA
find_package(CUDA 10.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(TensorRT_LIBS nvinfer nvonnxparser nvinfer_plugin)
include_directories(${TensorRT_INCLUDE_DIRS})


aux_source_directory(InstanceSegment InstanceSegmentation_SOURCES)
aux_source_directory(InstanceTracking InstanceTracking_SOURCES)

add_executable(tracking tracking.cpp parameters.cpp utils.cpp  ${InstanceSegmentation_SOURCES} ${InstanceTracking_SOURCES})
target_link_libraries(tracking  ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} )



add_executable(build_model build_model.cpp parameters.cpp InstanceSegment/logger.cpp)
target_link_libraries(build_model  ${TensorRT_LIBS} pthread ${OpenCV_LIBRARIES})


