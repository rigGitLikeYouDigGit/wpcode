#cmake_minimum_required(VERSION 2.6)
#cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.0)
set(MAYA_VERSION 2023)

set(CMAKE_MODULE_PATH "C:/Program Files/CMake/share/cmake-3.20/Modules")

project(strata)
find_package(Maya REQUIRED)
#find_package(Eigen3 REQUIRED)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/../output/Maya${MAYA_VERSION})

set(CMAKE_ARCJOVE)

#option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
#find_package(LIBIGL REQUIRED QUIET)


file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h" "src/lib/*.cpp" "src/lib/*.h" 
	"src/deformer/*.h" "src/deformer/*.cpp" "src/*/*.cpp" "src/*/*.h" "src/*/*/*.cpp" "src/*/*/*.h"
)

#include_directories($ENV{WP_ROOT}/code/cpp/wpshared)
#file(GLOB SOURCE_FILES "F:/wp/code/cpp/wpshared")
include_directories("F:/wp/code/cpp/src")
include_directories(${MAYA_INCLUDE_DIR})

#include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/armadillo/include)

link_directories(${MAYA_LIBRARY_DIR})
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${MAYA_LIBRARIES})

MAYA_PLUGIN(${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
