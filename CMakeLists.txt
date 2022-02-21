# Copyright(c) 2020 H.D. Wobben
# Distributed under the MIT License (http://opensource.org/licenses/MIT)

cmake_minimum_required(VERSION 3.2)

include(cmake/utils.cmake)

project(ExcitationDiffusion LANGUAGES CXX C)

#---------------------------------------------------------------------------------------
# Set default build to release
#---------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#---------------------------------------------------------------------------------------
# Compiler config
#---------------------------------------------------------------------------------------
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_EXTENSIONS  OFF)

if(CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
	set(CMAKE_CXX_EXTENSIONS ON)
endif()

#---------------------------------------------------------------------------------------
# Build programs
#---------------------------------------------------------------------------------------

include(dependencies/dependencies.cmake)
find_package(Threads REQUIRED) 

#-----------------------
# nise_diff
#-----------------------
add_executable(nise_diff 

    src/nise_diff.cpp
    src/random.cpp
    src/utils.cpp)

target_include_directories(nise_diff PUBLIC 

    ${CMAKE_CURRENT_LIST_DIR}/include
    ${json_SOURCE_DIR}/include)

target_link_libraries(nise_diff PUBLIC Eigen3::Eigen Threads::Threads)
_enable_warnings(nise_diff)
set_target_properties(nise_diff PROPERTIES DEBUG_POSTFIX d)

#-----------------------
# kubo_diff_int
#-----------------------
add_executable(kubo_diff_int 

    src/kubo_diff_int.cpp
    src/random.cpp
    src/utils.cpp)

target_include_directories(kubo_diff_int PUBLIC 

    ${CMAKE_CURRENT_LIST_DIR}/include
    ${json_SOURCE_DIR}/include)

target_link_libraries(kubo_diff_int PUBLIC Eigen3::Eigen Threads::Threads)
_enable_warnings(kubo_diff_int)
set_target_properties(kubo_diff_int PROPERTIES DEBUG_POSTFIX d)