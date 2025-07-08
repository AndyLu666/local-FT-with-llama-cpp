# GPT-2 快速開始指南

本指南將幫助您快速上手 GPT-2 訓練演示項目。

## 系統要求

### 最低要求
- **作業系統**: Linux, macOS, or Windows (WSL)
- **編譯器**: GCC 7+ 或 Clang 8+ (支援 C++17)
- **CMake**: 3.14 或更高版本
- **記憶體**: 至少 4GB RAM
- **磁碟空間**: 至少 2GB 可用空間

### 推薦配置
- **CPU**: 多核心處理器 (8 核心或更多)
- **記憶體**: 16GB+ RAM
- **GPU**: NVIDIA GPU with CUDA support (可選)

## 安裝步驟

### 1. 克隆 llama.cpp 項目

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### 2. 導航到 GPT-2 演示目錄

```bash
cd demo/gpt2
```

### 3. 構建項目

#### 使用構建腳本 (推薦)

```bash
# 賦予執行權限
chmod +x build.sh

# 構建項目
./build.sh

# 如果要運行測試
./build.sh --test
```

#### 手動構建

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..
```

### 4. 驗證安裝

```bash
# 運行示例程序
./build/gpt2-example
```

如果看到類似以下輸出，說明安裝成功：

```
GPT-2 使用示例
================

=== 模型配置示例 ===

Small (117M) 參數:
  嵌入維度: 768
  注意力頭數: 12
  ...
```

## 第一次訓練

### 1. 準備訓練數據

創建一個簡單的文本文件：

```bash
cat > training_data.txt << 'EOF'
從前有一座山，山上有一座廟，廟裡有個老和尚在講故事。
他講的故事是什麼呢？從前有一座山，山上有一座廟，廟裡有個老和尚在講故事。
故事很簡單，但很有意思。老和尚每天都在講這個故事。
小和尚們都很喜歡聽這個故事，因為故事裡面有很多有趣的內容。
EOF
```

### 2. 創建詞彙表 (可選)

```bash
cat > vocab.txt << 'EOF'
<pad>
<unk>
<bos>
<eos>
從前
有
一
座
山
，
上
廟
裡
個
老
和尚
在
講
故事
。
他
的
是
什麼
呢
？
很
簡單
但
意思
每天
都
這個
小
們
喜歡
聽
因為
面
多
趣
內容
EOF
```

### 3. 開始訓練

```bash
./build/gpt2-train \
    --data training_data.txt \
    --vocab vocab.txt \
    --model-size small \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --output ./output
```

### 4. 查看訓練結果

訓練完成後，模型會保存在 `./output/` 目錄中。

## 常見用例

### 訓練小型模型 (測試用)

```bash
./build/gpt2-train \
    --data small_dataset.txt \
    --model-size small \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-4
```

### 訓練中型模型

```bash
./build/gpt2-train \
    --data large_dataset.txt \
    --model-size medium \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-4
```

### 從檢查點恢復訓練

```bash
./build/gpt2-train \
    --data training_data.txt \
    --resume ./output/checkpoint_epoch_5.bin \
    --epochs 10
```

## 程式碼示例

### 基本使用

```cpp
#include "config.h"
#include "model.h"
#include "data_loader.h"

using namespace gpt2;

int main() {
    // 1. 創建配置
    GPT2Config config = configs::small();
    
    // 2. 創建模型
    auto model = std::make_unique<GPT2Model>(config);
    
    // 3. 創建數據加載器
    auto dataloader = std::make_unique<DataLoader>(config);
    dataloader->load_text_file("data.txt");
    
    // 4. 初始化 GGML 上下文
    size_t ctx_size = 1024 * 1024 * 256; // 256MB
    void* ctx_data = malloc(ctx_size);
    ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = ctx_data,
        .no_alloc = false
    };
    ggml_context* ctx = ggml_init(params);
    
    // 5. 初始化模型權重
    model->init_weights(ctx);
    
    // 6. 清理
    ggml_free(ctx);
    free(ctx_data);
    
    return 0;
}
```

### 文本生成

```cpp
// 準備輸入
std::vector<int> input_ids = {1, 2, 3}; // token IDs
std::string input_text = "從前有一座山";

// 生成文本
std::vector<int> generated = model->generate(
    ctx,
    input_ids,
    100,    // max_length
    0.8f,   // temperature
    50,     // top_k
    0.9f    // top_p
);

// 輸出結果
for (int token_id : generated) {
    std::cout << token_id << " ";
}
```

### 自定義配置

```cpp
// 創建自定義配置
GPT2Config custom_config;
custom_config.n_embd = 512;
custom_config.n_head = 8;
custom_config.n_layer = 6;
custom_config.batch_size = 16;
custom_config.learning_rate = 1e-4f;

// 使用自定義配置
auto model = std::make_unique<GPT2Model>(custom_config);
```

## 常見問題

### Q: 編譯時出現 "ggml.h not found" 錯誤

**A**: 確保您在 `llama.cpp` 項目的 `demo/gpt2` 目錄中運行構建命令。項目依賴於上級目錄的 GGML 庫。

### Q: 訓練時記憶體不足

**A**: 嘗試以下解決方案：
- 減小批次大小 (`--batch-size`)
- 使用更小的模型 (`--model-size small`)
- 減少序列長度

### Q: 訓練速度很慢

**A**: 可以嘗試：
- 啟用 OpenMP: 確保安裝了 OpenMP 並重新編譯
- 使用 GPU: 如果有 NVIDIA GPU，使用 `--enable-cuda` 編譯
- 增加批次大小以提高 GPU 利用率

### Q: 如何查看訓練進度？

**A**: 訓練器會自動輸出訓練統計信息，包括：
- 損失值
- 困惑度 (Perplexity)
- 學習率
- 已用時間

### Q: 生成的文本質量不好

**A**: 可能的原因和解決方案：
- 訓練數據不足：使用更多、更高質量的訓練數據
- 訓練不充分：增加訓練輪數 (`--epochs`)
- 學習率不當：嘗試調整學習率 (`--learning-rate`)
- 模型太小：使用更大的模型 (`--model-size`)

## 性能優化建議

### 1. 編譯優化

```bash
# 啟用所有優化選項
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=ON

# 如果有 CUDA GPU
cmake .. -DLLAMA_CUBLAS=ON

# 如果有 Apple Silicon
cmake .. -DLLAMA_METAL=ON
```

### 2. 運行時優化

```bash
# 設置 OpenMP 線程數
export OMP_NUM_THREADS=8

# 設置 CPU 親和性 (Linux)
taskset -c 0-7 ./build/gpt2-train --data data.txt
```

### 3. 記憶體優化

- 使用較小的批次大小
- 設置適當的上下文大小
- 定期清理不必要的張量

## 下一步

現在您已經成功運行了 GPT-2 演示，可以嘗試：

1. **實驗不同的超參數**：學習率、批次大小、模型尺寸
2. **使用真實數據集**：下載公開的文本數據集進行訓練
3. **自定義模型架構**：修改代碼以實驗不同的架構
4. **實現新功能**：添加新的採樣策略或訓練技巧
5. **性能分析**：使用 profiler 分析性能瓶頸

更多詳細信息請參考：
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架構詳細說明
- [README.md](README.md) - 項目概述
- [llama.cpp 文檔](../../README.md) - 主項目文檔