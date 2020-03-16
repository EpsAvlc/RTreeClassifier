cmake_minimum_required(VERSION 3.0.0)
project(RtreeClassifier VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(PCL REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
)

add_library(KITTIHelper utils/kitti_helper.cpp)
target_link_libraries(KITTIHelper
${PCL_LIBRARIES})

add_executable(demo src/main.cpp)
target_link_libraries(demo 
KITTIHelper)
