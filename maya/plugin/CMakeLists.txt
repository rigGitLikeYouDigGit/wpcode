
cmake_minimum_required(VERSION 3.0)
set(MAYA_VERSION 2026)

set(CMAKE_MODULE_PATH "C:/Program Files/CMake/share/cmake-3.31/Modules")
set(EIGEN3_INCLUDE_DIR "C:/Users/ed/Documents/GitHub/eigen-5.0.1")
project(wpplugin)
find_package(Maya REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../output/Maya${MAYA_VERSION})
include_directories("C:/Users/ed/Documents/GitHub/wpcode/maya/plugin/src")
include_directories(${MAYA_INCLUDE_DIR})

# set up PCH
add_library(std_pch INTERFACE)
target_precompile_headers(std_pch INTERFACE
    <vector>
    <string>
    <map>
    <unordered_map>
    <iostream>
    <fstream>
    <sstream>
    <cmath>
)

add_library(eigen_pch INTERFACE)
target_include_directories(eigen_pch INTERFACE ${EIGEN3_INCLUDE_DIR})
target_precompile_headers(eigen_pch INTERFACE
    <Eigen/Core>
    <Eigen/Dense>
    <Eigen/Geometry>
    # Add more Eigen modules as needed:
    # <Eigen/Sparse>
    # <Eigen/LU>
    # <Eigen/QR>
)

add_library(maya_pch INTERFACE)
target_precompile_headers(maya_pch INTERFACE
    # Maya SDK headers (slow to compile)
    <maya/MFnPlugin.h>
    <maya/MPxCommand.h>
    <maya/MGlobal.h>
    <maya/MFnMesh.h>
    <maya/MItMeshVertex.h>
    <maya/MPointArray.h>
    
)




file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h" "src/lib/*.cpp" "src/lib/*.h" 
	"src/deformer/*.h" "src/deformer/*.cpp" "src/*/*.cpp" "src/*/*.h" "src/*/*/*.cpp" "src/*/*/*.h"
)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

#include_directories("F:/wp/code/cpp/src")

#link_directories(${MAYA_LIBRARY_DIR})

#target_link_libraries(${PROJECT_NAME} ${MAYA_LIBRARIES})

MAYA_PLUGIN(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    std_pch 
    eigen_pch
    maya_pch 
    ${MAYA_LIBRARIES})


install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
