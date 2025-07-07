cmake_minimum_required(VERSION 3.14)
project("llama-demo" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 简单的多层感知器示例
add_executable(simple-mlp simple-mlp.cpp)
target_link_libraries(simple-mlp PRIVATE pthread)

# 增强版 MLP（使用你的 forward、backward 和 optimizer 组件）
add_executable(enhanced-mlp enhanced-mlp.cpp)
target_link_libraries(enhanced-mlp PRIVATE pthread)
