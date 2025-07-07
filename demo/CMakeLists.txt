cmake_minimum_required(VERSION 3.12)
project(llama_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 設置 GGML 路徑
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ggml)

# 包含目錄
include_directories(${GGML_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

# 查找 GGML 庫
find_library(GGML_LIB ggml PATHS ${GGML_DIR}/src NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_DIR}/src/ggml-cpu NO_DEFAULT_PATH)

if(NOT GGML_LIB)
    message(FATAL_ERROR "GGML library not found. Please build GGML first.")
endif()

if(NOT GGML_CPU_LIB)
    message(FATAL_ERROR "GGML CPU library not found. Please build GGML first.")
endif()

# 設置庫列表
set(GGML_LIBRARIES ${GGML_LIB} ${GGML_CPU_LIB})

# GPU支援檢測
option(GGML_USE_CUDA "Enable CUDA support" OFF)
option(GGML_USE_METAL "Enable Metal support" OFF)

# CUDA支援
if(GGML_USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        find_library(GGML_CUDA_LIB ggml-cuda PATHS ${GGML_DIR}/src/ggml-cuda NO_DEFAULT_PATH)
        if(GGML_CUDA_LIB)
            add_definitions(-DGGML_USE_CUDA)
            list(APPEND GGML_LIBRARIES ${GGML_CUDA_LIB})
            message(STATUS "CUDA support enabled")
        else()
            message(WARNING "CUDA requested but ggml-cuda library not found")
        endif()
    else()
        message(WARNING "CUDA requested but CUDA toolkit not found")
    endif()
endif()

# Metal支援 (macOS)
if(GGML_USE_METAL AND APPLE)
    find_library(GGML_METAL_LIB ggml-metal PATHS ${GGML_DIR}/src/ggml-metal NO_DEFAULT_PATH)
    if(GGML_METAL_LIB)
        add_definitions(-DGGML_USE_METAL)
        list(APPEND GGML_LIBRARIES ${GGML_METAL_LIB})
        # 添加Metal框架
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        list(APPEND GGML_LIBRARIES ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        message(STATUS "Metal support enabled")
    else()
        message(WARNING "Metal requested but ggml-metal library not found")
    endif()
endif()

# 源文件
set(SOURCES
    ../src/llama-forward.cpp
    ../src/llama-backward.cpp
    ../src/llama-optimizer.cpp
    ../src/simple-neural-model.cpp
)

# 簡單MLP演示
add_executable(simple-mlp simple-mlp.cpp ${SOURCES})
target_link_libraries(simple-mlp ${GGML_LIBRARIES})

# 增強MLP演示
add_executable(enhanced-mlp enhanced-mlp.cpp ${SOURCES})
target_link_libraries(enhanced-mlp ${GGML_LIBRARIES})

# 使用src組件的增強MLP演示
add_executable(enhanced-mlp-with-src enhanced-mlp-with-src.cpp ${SOURCES})
target_link_libraries(enhanced-mlp-with-src ${GGML_LIBRARIES})

# GPU神經網路演示
add_executable(gpu-neural-demo gpu-neural-demo.cpp)
target_link_libraries(gpu-neural-demo PRIVATE ggml simple-neural-model)
target_compile_features(gpu-neural-demo PRIVATE cxx_std_11)

# GPT-2 Training Demo
add_executable(gpt2-train gpt2-train.cpp)
target_link_libraries(gpt2-train PRIVATE ggml)
target_compile_features(gpt2-train PRIVATE cxx_std_11)

# Simple GPT-2 Training Demo
add_executable(simple-gpt2-train simple-gpt2-train.cpp)
target_link_libraries(simple-gpt2-train PRIVATE ggml)
target_compile_features(simple-gpt2-train PRIVATE cxx_std_11)

# Text Generation Training Demo
add_executable(train-text-generation train-text-generation.cpp)
target_link_libraries(train-text-generation PRIVATE ggml)
target_compile_features(train-text-generation PRIVATE cxx_std_11)

# 如果有 CUDA 支持
if(GGML_CUDA)
    target_compile_definitions(gpt2-train PRIVATE GGML_USE_CUDA)
    target_compile_definitions(simple-gpt2-train PRIVATE GGML_USE_CUDA)
    target_compile_definitions(train-text-generation PRIVATE GGML_USE_CUDA)
endif()

# 如果有 Metal 支持
if(GGML_METAL)
    target_compile_definitions(gpt2-train PRIVATE GGML_USE_METAL)
    target_compile_definitions(simple-gpt2-train PRIVATE GGML_USE_METAL)
    target_compile_definitions(train-text-generation PRIVATE GGML_USE_METAL)
endif()

# 編譯標誌
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(simple-mlp PRIVATE -g -O0 -Wall -Wextra)
    target_compile_options(enhanced-mlp PRIVATE -g -O0 -Wall -Wextra)
    target_compile_options(enhanced-mlp-with-src PRIVATE -g -O0 -Wall -Wextra)
    target_compile_options(gpu-neural-demo PRIVATE -g -O0 -Wall -Wextra)
else()
    target_compile_options(simple-mlp PRIVATE -O3 -DNDEBUG)
    target_compile_options(enhanced-mlp PRIVATE -O3 -DNDEBUG)
    target_compile_options(enhanced-mlp-with-src PRIVATE -O3 -DNDEBUG)
    target_compile_options(gpu-neural-demo PRIVATE -O3 -DNDEBUG)
endif()

# 打印配置信息
message(STATUS "=== LLAMA Demo Configuration ===")
message(STATUS "GGML Directory: ${GGML_DIR}")
message(STATUS "GGML Libraries: ${GGML_LIBRARIES}")
message(STATUS "CUDA Support: ${GGML_USE_CUDA}")
message(STATUS "Metal Support: ${GGML_USE_METAL}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===============================")
