cmake_minimum_required(VERSION 3.0.0)
project(RtreeClassifier VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    include
    utils
    ${PCL_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

add_library(KITTIHelper utils/kitti_helper.cpp)
target_link_libraries(KITTIHelper
${PCL_LIBRARIES})

add_library(RForest src/RForest.cpp)
target_link_libraries(RForest ${OpenCV_LIBS})

add_executable(demo src/main.cpp)
target_link_libraries(demo 
KITTIHelper)
