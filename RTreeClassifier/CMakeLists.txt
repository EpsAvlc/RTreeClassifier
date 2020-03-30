cmake_minimum_required(VERSION 3.8.0)
project(RtreeClassifier VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_MODULE_PATH {CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

add_library(EVDescriptor src/descriptors/EVDescriptor.cpp)
target_link_libraries(EVDescriptor ${PCL_LIBRARIES})

add_library(ESFDescriptor src/descriptors/ESFDescriptor.cpp)
target_link_libraries(ESFDescriptor ${PCL_LIBRARIES})

add_executable(demo src/main.cpp)
target_link_libraries(demo 
KITTIHelper
EVDescriptor
RForest
glog
)

add_executable(demo_train_test demo/demo_train_test.cpp)
target_link_libraries(demo_train_test 
KITTIHelper
EVDescriptor
ESFDescriptor
RForest
glog
)

add_executable(demo_predict demo/demo_predict.cpp)
target_link_libraries(demo_predict 
KITTIHelper
EVDescriptor
RForest
glog
)

