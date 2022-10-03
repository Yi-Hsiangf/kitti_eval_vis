cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(kitti_vis)

find_package(PCL 1.12.1 REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)

set(VTK_DIR "/home/vivid/projects/VTK-build/")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (kitti_vis kitti_vis.cpp)
target_link_libraries(kitti_vis yaml-cpp)

target_link_libraries(kitti_vis ${VTK_LIBRARIES})
target_link_libraries (kitti_vis ${PCL_LIBRARIES})
