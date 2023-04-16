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
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Boost 1.65 REQUIRED )
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED)
#find_package(pcl REQUIRED)
#remove later debug purpose

include_directories(${PROJECT_SOURCE_DIR}/include)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${YAML_CPP_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${fbow_INCLUDE_DIRS}
  gtsam
  gtsam_unstable
)

## Declare a C++ library
list(APPEND thirdparty_libraries
${OpenCV_LIBS} 
${Boost_LIBRARIES}
${PCL_LIBRARIES}
boost_system
Sophus::Sophus
glog::glog
${YAML_CPP_LIBRARIES}
${fbow_LIBS}
)
add_library(loopClosure SHARED ${PROJECT_SOURCE_DIR}/src/LoopClosingTool.cpp 
${PROJECT_SOURCE_DIR}/src/landmark_manager.cpp
${PROJECT_SOURCE_DIR}/src/util.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp
${PROJECT_SOURCE_DIR}/src/RegistrationTool.cpp)

target_link_libraries( loopClosure  PUBLIC
${thirdparty_libraries}
)

# add_executable(LoopClosingMain ${PROJECT_SOURCE_DIR}/src/LoopClosingMain.cpp)
# add_dependencies(LoopClosingMain ${catkin_EXPORTED_TARGETS})
# target_link_libraries(LoopClosingMain PUBLIC loopClosure ${fbow_LIBS})

