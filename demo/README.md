# LLAMA.cpp 神經網路訓練演示

這個演示項目展示了如何使用 GGML 庫進行神經網路訓練，支持 CPU 和 GPU 加速。

## 🚀 功能特性

- ✅ **多後端支援**: CPU、CUDA、Metal
- ✅ **模組化設計**: 前向傳播、反向傳播、優化器分離
- ✅ **GPU加速**: 自動檢測並使用可用的GPU
- ✅ **性能測試**: 詳細的性能基準測試
- ✅ **模型保存/加載**: 完整的模型持久化支援

## 📋 系統需求

### 基本需求
- CMake 3.12+
- C++17 編譯器
- 已編譯的 GGML 庫

### GPU 支援需求

#### CUDA (NVIDIA GPU)
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# 或下載並安裝 CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads
```

#### Metal (Apple GPU)
- macOS 10.13+
- 支援 Metal 的 Apple 設備

## 🔧 編譯指南

### 1. 編譯 GGML 庫

首先需要編譯 GGML 庫：

```bash
cd ggml
mkdir build && cd build

# CPU 版本
cmake ..
make -j$(nproc)

# CUDA 版本
cmake -DGGML_USE_CUDA=ON ..
make -j$(nproc)

# Metal 版本 (macOS)
cmake -DGGML_USE_METAL=ON ..
make -j$(nproc)
```

### 2. 編譯演示程序

```bash
cd demo
mkdir build && cd build

# CPU 版本
cmake ..
make -j$(nproc)

# CUDA 版本
cmake -DGGML_USE_CUDA=ON ..
make -j$(nproc)

# Metal 版本 (macOS)
cmake -DGGML_USE_METAL=ON ..
make -j$(nproc)
```

## 🎮 使用方法

### 基本演示程序

```bash
# 簡單 MLP (手動實現)
./bin/simple-mlp

# 增強 MLP (模組化設計)
./bin/enhanced-mlp

# 使用 src 組件的版本
./bin/enhanced-mlp-with-src
```

### GPU 加速演示

```bash
# CPU 執行
./bin/gpu-neural-demo

# GPU 執行 (自動檢測)
./bin/gpu-neural-demo --gpu

# 指定 CUDA 後端
./bin/gpu-neural-demo --gpu --backend cuda

# 指定 Metal 後端 (macOS)
./bin/gpu-neural-demo --gpu --backend metal

# 指定 GPU 設備
./bin/gpu-neural-demo --gpu --device 0

# 顯示幫助
./bin/gpu-neural-demo --help
```

## 📊 性能比較

### 典型性能數據 (僅供參考)

| 後端 | 單樣本前向傳播 | 批量前向傳播 (200樣本) | 記憶體使用 |
|------|---------------|----------------------|-----------|
| CPU  | ~50-100 μs    | ~10-20 ms           | ~1-2 MB   |
| CUDA | ~20-50 μs     | ~5-10 ms            | ~2-4 MB   |
| Metal| ~30-60 μs     | ~7-15 ms            | ~2-3 MB   |

*實際性能取決於硬體配置和模型大小*

## 🏗️ 架構設計

### 核心組件

```
src/
├── simple-neural-model.h/cpp    # GPU支援的神經網路模型
├── llama-forward.h/cpp          # 前向傳播組件
├── llama-backward.h/cpp         # 反向傳播組件
└── llama-optimizer.h/cpp        # 優化器組件
```

### GPU 後端架構

```
GGML 後端系統
├── CPU 後端 (ggml-cpu)
├── CUDA 後端 (ggml-cuda)
├── Metal 後端 (ggml-metal)
└── 統一接口 (ggml-backend)
```

## 🐛 故障排除

### 常見問題

#### 1. GGML 庫未找到
```
CMake Error: GGML library not found
```
**解決方案**: 確保先編譯 GGML 庫
```bash
cd ggml && mkdir build && cd build && cmake .. && make
```

#### 2. CUDA 支援未啟用
```
CUDA requested but CUDA toolkit not found
```
**解決方案**: 安裝 CUDA Toolkit 並設置環境變數
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 3. Metal 支援問題 (macOS)
```
Metal requested but ggml-metal library not found
```
**解決方案**: 確保在 macOS 上編譯並啟用 Metal 支援
```bash
cmake -DGGML_USE_METAL=ON ..
```

#### 4. 記憶體不足
```
Failed to allocate GPU memory
```
**解決方案**: 
- 減少批次大小
- 使用較小的模型
- 檢查 GPU 記憶體使用情況

### 調試技巧

#### 啟用詳細日誌
```bash
export GGML_LOG_LEVEL=DEBUG
./bin/gpu-neural-demo --gpu
```

#### 檢查 GPU 狀態
```bash
# NVIDIA GPU
nvidia-smi

# 檢查 CUDA 安裝
nvcc --version
```

## 📚 進階使用

### 自定義模型參數

```cpp
SimpleNeuralModel::ModelParams params;
params.input_size = 8;      // 輸入維度
params.hidden_size = 32;    // 隱藏層維度
params.output_size = 5;     // 輸出維度
params.use_gpu = true;      // 啟用 GPU
params.backend_type = "cuda"; // 指定後端
params.gpu_device = 0;      // GPU 設備 ID

SimpleNeuralModel model(params);
```

### 性能調優建議

1. **批次大小**: 增加批次大小可以提高 GPU 利用率
2. **記憶體管理**: 重複使用計算圖以減少記憶體分配
3. **數據類型**: 考慮使用 FP16 以節省記憶體（需要硬體支援）
4. **異步執行**: 使用異步 API 重疊計算和數據傳輸

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

### 開發環境設置
```bash
git clone <your-repo>
cd llama.cpp
git submodule update --init --recursive
```

### 代碼風格
- 使用 C++17 標準
- 遵循現有的命名慣例
- 添加適當的註釋和文檔

## 📄 許可證

本項目遵循 MIT 許可證。詳見 LICENSE 文件。

## 🙏 致謝

- [GGML](https://github.com/ggerganov/ggml) - 機器學習張量庫
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLaMA 模型推理引擎 