cmake_minimum_required(VERSION 3.0)
set(MAYA_VERSION 2026)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "C:/Program Files/CMake/share/cmake-3.31/Modules")

# Use top-level include directory for Eigen and all shared headers
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

project(wpplugin)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Maya REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../output/Maya${MAYA_VERSION})

if(MSVC)
    add_compile_options(/diagnostics:caret)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-ftemplate-backtrace-limit=0)
endif()

# set up PCH for standard library
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

# set up PCH for Eigen - use Eigen from top-level include directory
add_library(eigen_pch INTERFACE)
target_include_directories(eigen_pch INTERFACE ${EIGEN3_INCLUDE_DIR})
target_precompile_headers(eigen_pch INTERFACE
    <Eigen/Core>
    <Eigen/Dense>
    <Eigen/Geometry>
)

# set up PCH for Maya
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

# ============================================================================
# MAIN WPPLUGIN - Existing plugin
# ============================================================================
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
    "src/node/*.h"
    "src/node/*.cpp"
    "src/**/*.h"
    "src/**/*.cpp"
)

# Remove strata plugin.cpp from main plugin sources
list(FILTER SOURCE_FILES EXCLUDE REGEX "src/strata/plugin\\.cpp$")

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} "src/strata/evalGraph.h")

target_include_directories(${PROJECT_NAME} PRIVATE
    "src"
    ${MAYA_INCLUDE_DIR}
    ${SHARED_INCLUDE_DIR}
)

MAYA_PLUGIN(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    std_pch 
    eigen_pch
    maya_pch 
    ${MAYA_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)

# ============================================================================
# STRATA PLUGIN - Separate plugin (strata.mll)
# ============================================================================

# Collect Strata source files
file(GLOB_RECURSE STRATA_CORE_FILES CONFIGURE_DEPENDS
    "src/strata/stratacore/*.cpp"
    "src/strata/stratacore/*.h"
)

file(GLOB_RECURSE STRATA_OP_FILES CONFIGURE_DEPENDS
    "src/strata/strataop/*.cpp"
    "src/strata/strataop/*.h"
)

file(GLOB_RECURSE STRATA_MAYA_FILES CONFIGURE_DEPENDS
    "src/strata/mayaNode/*.cpp"
    "src/strata/mayaNode/*.h"
)

file(GLOB_RECURSE STRATA_SUPPORT_FILES CONFIGURE_DEPENDS
    "src/strata/bezier/*.cpp"
    "src/strata/bezier/*.h"
    "src/strata/exp/*.cpp"
    "src/strata/exp/*.h"
)

# Combine all Strata sources + plugin entry point
set(STRATA_PLUGIN_SOURCES
    src/strata/plugin.cpp
    ${STRATA_CORE_FILES}
    ${STRATA_OP_FILES}
    ${STRATA_MAYA_FILES}
    ${STRATA_SUPPORT_FILES}
)

# Create strata plugin library
add_library(strata SHARED ${STRATA_PLUGIN_SOURCES} "src/strata/evalGraph.h")

# Set output name to strata.mll
set_target_properties(strata PROPERTIES
    OUTPUT_NAME "strata"
    PREFIX ""
    SUFFIX ".mll"
)

target_include_directories(strata PRIVATE
    "src"
    "src/strata"
    ${MAYA_INCLUDE_DIR}
    ${SHARED_INCLUDE_DIR}
)

MAYA_PLUGIN(strata)

target_link_libraries(strata PRIVATE
    std_pch
    eigen_pch
    maya_pch
    ${MAYA_LIBRARIES}
)

# Install strata plugin
install(TARGETS strata ${MAYA_TARGET_TYPE} DESTINATION plug-ins)

message(STATUS "Configured two plugins:")
message(STATUS "  - wpplugin.mll (main plugin)")
message(STATUS "  - strata.mll (Strata plugin)")
message(STATUS "Shared include directory: ${SHARED_INCLUDE_DIR}")
