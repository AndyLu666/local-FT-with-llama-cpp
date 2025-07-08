#!/bin/bash

echo "=== 模組化 GPT-2 Demo 構建腳本 ==="

# 創建構建目錄
if [ ! -d "build" ]; then
    echo "創建構建目錄..."
    mkdir build
fi

cd build

# 配置 CMake
echo "配置 CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "CMake 配置失敗！"
    exit 1
fi

# 編譯
echo "開始編譯..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "編譯失敗！"
    exit 1
fi

echo "編譯成功！"
echo "可執行文件位於: build/gpt2_demo"
echo ""
echo "運行程序："
echo "  cd build && ./gpt2_demo"
echo ""
echo "清理構建："
echo "  rm -rf build"