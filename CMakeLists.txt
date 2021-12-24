cmake_minimum_required(VERSION 2.8)
project(diyconv)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
endif()


file(GLOB SRCS "${PROJECT_SOURCE_DIR}/conv_impl.cpp")
add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME})
