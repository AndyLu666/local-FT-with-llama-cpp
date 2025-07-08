# GPT-2 訓練演示

這是一個基於 llama.cpp 和 GGML 的模組化 GPT-2 實現和訓練演示項目。

## 特性

- **模組化設計**: 將 GPT-2 的各個組件拆分為獨立的模組
- **基於 GGML**: 使用 GGML 作為底層計算框架
- **多種模型尺寸**: 支持 Small (117M)、Medium (345M) 和 Large (762M) 參數
- **完整的訓練流程**: 包含數據加載、模型訓練、檢查點保存等
- **文本生成**: 支持多種採樣策略的文本生成

## 項目結構

```
gpt2/
├── config.h              # 模型配置和超參數
├── embeddings.h          # 詞嵌入和位置嵌入
├── layer_norm.h           # 層歸一化
├── attention.h            # 多頭注意力機制
├── feedforward.h          # 前饋網絡
├── transformer_block.h    # Transformer 塊
├── model.h               # GPT-2 模型主體
├── trainer.h             # 訓練器
├── data_loader.h         # 數據加載器
├── main.cpp              # 主訓練程序
├── example_usage.cpp     # 使用示例
├── CMakeLists.txt        # 構建配置
├── build.sh              # 構建腳本
├── README.md             # 項目說明 (本文件)
├── ARCHITECTURE.md       # 架構詳細說明
└── QUICKSTART.md         # 快速開始指南
```

## 快速開始

### 1. 構建項目

```bash
# 賦予構建腳本執行權限
chmod +x build.sh

# 構建項目
./build.sh

# 或者手動構建
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. 運行示例

```bash
# 運行使用示例
./build/gpt2-example
```

### 3. 訓練模型

```bash
# 準備訓練數據
echo "從前有一座山，山上有一座廟，廟裡有個老和尚在講故事..." > data.txt

# 開始訓練
./build/gpt2-train --data data.txt --model-size small --epochs 10
```

## 模型配置

### Small (117M 參數)
- 嵌入維度: 768
- 注意力頭數: 12
- Transformer 層數: 12

### Medium (345M 參數)
- 嵌入維度: 1024
- 注意力頭數: 16
- Transformer 層數: 24

### Large (762M 參數)
- 嵌入維度: 1280
- 注意力頭數: 20
- Transformer 層數: 36

## 使用方法

### 訓練選項

```bash
./gpt2-train [選項]

選項:
  --data <file>         訓練數據文件路径
  --vocab <file>        詞彙表文件路径
  --model-size <size>   模型大小 (small/medium/large)
  --epochs <num>        訓練輪數 (默認: 10)
  --batch-size <num>    批次大小 (默認: 32)
  --learning-rate <lr>  學習率 (默認: 3e-4)
  --output <dir>        輸出目录 (默認: ./output)
  --resume <checkpoint> 從檢查點恢復訓練
  --help               顯示幫助信息
```

### 編程接口

```cpp
#include "config.h"
#include "model.h"
#include "trainer.h"
#include "data_loader.h"

using namespace gpt2;

// 創建配置
GPT2Config config = configs::small();

// 創建模型
auto model = std::make_shared<GPT2Model>(config);

// 創建數據加載器
auto dataloader = std::make_shared<DataLoader>(config);
dataloader->load_text_file("data.txt");

// 創建訓練器
auto trainer = std::make_unique<Trainer>(config, model, dataloader);

// 開始訓練
trainer->train();
```

## 依賴項目

- **CMake** (>= 3.14)
- **C++ 編譯器** (支持 C++17)
- **GGML** (包含在 llama.cpp 中)
- **OpenMP** (可選，用於並行加速)
- **CUDA** (可選，用於 GPU 加速)

## 架構說明

詳細的架構說明請參考 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速開始

詳細的快速開始指南請參考 [QUICKSTART.md](QUICKSTART.md)。

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 許可證

本項目遵循與 llama.cpp 相同的許可證。

## 致謝

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 基礎框架
- [GGML](https://github.com/ggerganov/ggml) - 機器學習張量庫
- OpenAI GPT-2 - 原始模型架構