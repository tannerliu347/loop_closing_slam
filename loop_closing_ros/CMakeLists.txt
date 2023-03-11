cmake_minimum_required(VERSION 3.0.2)
project(loop_closure)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package (glog 0.6.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Sophus REQUIRED)
find_package(fbow REQUIRED)
find_package(yaml-cpp REQUIRED)

#find_package(pcl REQUIRED)
#remove later debug purpose

include_directories(${PROJECT_SOURCE_DIR}/include)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}

)

## Declare a C++ library
list(APPEND thirdparty_libraries
${OpenCV_LIBS} 
${Boost_LIBRARIES}
boost_system
Sophus::Sophus
glog::glog
${YAML_CPP_LIBRARIES}
)
add_library(loopClosure SHARED ${PROJECT_SOURCE_DIR}/src/LoopClosingTool.cpp ${PROJECT_SOURCE_DIR}/src/landmark_manager.cpp
${PROJECT_SOURCE_DIR}/src/util.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)

target_link_libraries( loopClosure  PUBLIC
${thirdparty_libraries}
)

# add_executable(LoopClosingMain ${PROJECT_SOURCE_DIR}/src/LoopClosingMain.cpp)
# add_dependencies(LoopClosingMain ${catkin_EXPORTED_TARGETS})
# target_link_libraries(LoopClosingMain PUBLIC loopClosure ${fbow_LIBS})

