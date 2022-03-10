set(PROJECT_NAME CvoLoopClosure)

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
find_package(UnifiedCvo REQUIRED) 
find_package(DBoW3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/src/CvoLoopClosing.cpp
                                   ${PROJECT_SOURCE_DIR}/src/KeyFrame.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC UnifiedCvo::cvo_utils_lib
                                             UnifiedCvo::lie_group_utils
                                             UnifiedCvo::cvo_gpu_img_lib 
                                             UnifiedCvo::cvo_irls_lib
                                             UnifiedCvo::elas
                                             ${Boost_LIBRARIES}
                                             boost_system)

add_executable(kitti_test ${PROJECT_SOURCE_DIR}/src/kitti_test.cpp)
target_link_libraries(kitti_test PUBLIC ${PROJECT_NAME}
                                        UnifiedCvo::elas
                                        UnifiedCvo::kitti
                                        ${OpenCV_LIBS}
                                        ${DBoW3_LIBS})