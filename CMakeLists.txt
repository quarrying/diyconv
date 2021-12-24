cmake_minimum_required(VERSION 2.8)
project(conv2d)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")

file(GLOB SRCS "${PROJECT_SOURCE_DIR}/conv2d.cpp")
add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME})
