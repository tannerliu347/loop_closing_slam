set(PROJECT_NAME LoopClosure)

cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release) 
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
#remove later debug purpose
find_package(IcecreamCpp)
include_directories(${IcecreamCpp_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/src/LoopClosingTool.cpp)
target_link_libraries( ${PROJECT_NAME}  PUBLIC ${OpenCV_LIBS} ${DBoW3_LIBS} ${Boost_LIBRARIES}
                                             boost_system)

add_executable(test ${PROJECT_SOURCE_DIR}/src/test.cpp)
target_link_libraries(test PUBLIC ${PROJECT_NAME})