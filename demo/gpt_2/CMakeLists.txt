cmake_minimum_required(VERSION 3.12)
project(GPT2Demo)

# 設置 C++ 標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 編譯選項
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# 源文件
set(SOURCES
    embeddings.cpp
    layer_norm.cpp
    attention.cpp
    feedforward.cpp
    transformer_block.cpp
    model.cpp
    data_loader.cpp
    trainer.cpp
    main.cpp
)

# 創建可執行文件
add_executable(gpt2_demo ${SOURCES})

# 添加數學庫鏈接（需要用到 exp, sqrt 等數學函數）
target_link_libraries(gpt2_demo m)

# 編譯信息
message(STATUS "=== GPT-2 Demo 構建配置 ===")
message(STATUS "C++ 標準: ${CMAKE_CXX_STANDARD}")
message(STATUS "編譯選項: ${CMAKE_CXX_FLAGS}")
message(STATUS "源文件數量: ${SOURCES}")
message(STATUS "============================")

# 安裝規則（可選）
install(TARGETS gpt2_demo DESTINATION bin)

# 複製數據文件到構建目錄
configure_file(${CMAKE_SOURCE_DIR}/training_data.txt ${CMAKE_BINARY_DIR}/training_data.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/vocab.txt ${CMAKE_BINARY_DIR}/vocab.txt COPYONLY)