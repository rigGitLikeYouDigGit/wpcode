cmake_minimum_required(VERSION 3.0)
set(MAYA_VERSION 2026)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "C:/Program Files/CMake/share/cmake-3.31/Modules")
set(EIGEN3_INCLUDE_DIR "C:/Users/ed/Documents/GitHub/eigen-5.0.1")
project(wpplugin)

find_package(Maya REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../output/Maya${MAYA_VERSION})

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
)

add_library(maya_pch INTERFACE)
target_include_directories(maya_pch INTERFACE ${MAYA_INCLUDE_DIR})
target_precompile_headers(maya_pch INTERFACE
    <maya/MFnPlugin.h>
    <maya/MPxCommand.h>
    <maya/MGlobal.h>
    <maya/MFnMesh.h>
    <maya/MFnNurbsCurve.h>
    <maya/MItMeshVertex.h>

    <maya/MVector.h>
    <maya/MPoint.h>
    <maya/MMatrix.h>
    <maya/MObject.h>
    <maya/MPointArray.h>
)
 
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
    "src/node/*.h"
    "src/node/*.cpp"
    "src/**/*.h"
    "src/**/*.cpp"
)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# IMPORTANT: Use target_include_directories instead of include_directories
target_include_directories(${PROJECT_NAME} PRIVATE
    "C:/Users/ed/Documents/GitHub/wpcode/maya/plugin/src"
    ${MAYA_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

MAYA_PLUGIN(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    std_pch 
    eigen_pch
    maya_pch 
    ${MAYA_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
