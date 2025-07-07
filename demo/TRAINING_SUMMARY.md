# GGML 訓練功能總結

本文檔總結了在 llama.cpp 中實現的訓練功能和示例。

## 已完成的工作

### 1. 基礎神經網絡組件 (src/ 目錄)

- **llama-forward.h/cpp**: 前向傳播實現
- **llama-backward.h/cpp**: 反向傳播和梯度計算
- **llama-optimizer.h/cpp**: 參數優化（SGD with gradient clipping）
- **simple-neural-model.h/cpp**: 支持 GPU 的神經網絡模型類

### 2. 訓練示例 (demo/ 目錄)

#### a) simple-mlp.cpp
- 純 C++ 手動實現的 3 層 MLP
- 展示基本的前向/反向傳播原理
- 批量梯度下降訓練

#### b) enhanced-mlp.cpp
- 模塊化的神經網絡設計
- 清晰的架構分離
- 易於擴展的結構

#### c) enhanced-mlp-with-src.cpp
- 使用 src/ 目錄下的組件
- 展示如何集成各個模塊
- 完整的訓練流程

#### d) gpu-neural-demo.cpp
- GPU 加速的神經網絡訓練
- 支持 CUDA 和 Metal 後端
- 自動後端檢測和切換

#### e) gpt2-train.cpp
- GPT-2 模型結構實現
- 展示如何構建 Transformer 架構
- 使用 GGML 的張量操作

#### f) simple-gpt2-train.cpp
- 簡化的 GPT-2 訓練示例
- 使用現有的 GGML 功能
- 展示基本的訓練循環

#### g) train-text-generation.cpp
- 完整的文本生成模型訓練
- 字符級別的 tokenizer
- 包含文本生成功能

## GGML 已支持的訓練功能

### 1. 自動微分
- `ggml_build_backward_expand()` - 自動構建反向傳播圖
- 支持大部分基本運算的梯度計算

### 2. 已實現的反向傳播算子
- `GGML_OP_CROSS_ENTROPY_LOSS_BACK` - 交叉熵損失
- `GGML_OP_SOFT_MAX_BACK` - Softmax
- `GGML_OP_RMS_NORM_BACK` - RMS Normalization
- `GGML_OP_SILU_BACK` - SiLU 激活函數
- `GGML_OP_ROPE_BACK` - RoPE 位置編碼
- `GGML_OP_GET_ROWS_BACK` - 嵌入查找
- `GGML_OP_REPEAT_BACK` - Repeat 操作
- `GGML_OP_IM2COL_BACK` - Im2Col (卷積)
- `GGML_OP_POOL_2D_BACK` - 2D 池化

### 3. 優化器
- `GGML_OP_OPT_STEP_ADAMW` - AdamW 優化器

## 尚需實現的功能

為了完整支持 GPT-2 等大型模型的訓練，還需要：

### 1. 缺失的反向傳播算子
- `GGML_OP_NORM_BACK` - Layer Normalization 反向傳播
- `GGML_OP_GELU_BACK` - GELU 激活函數反向傳播
- 完善 `GGML_OP_MUL_MAT` 的反向傳播

### 2. 訓練優化功能
- 梯度累積
- 混合精度訓練
- 梯度檢查點
- 分佈式訓練支持

### 3. 數據處理
- 高效的數據加載器
- 動態批處理
- 數據增強

## 使用指南

### 編譯示例

```bash
cd demo
mkdir build
cd build
cmake ..
make
```

### 運行訓練

```bash
# 簡單 MLP
./simple-mlp

# GPU 加速訓練
./gpu-neural-demo

# 文本生成訓練
./train-text-generation

# GPT-2 訓練（需要實現額外算子）
./simple-gpt2-train
```

## 性能考慮

1. **內存使用**：訓練需要存儲梯度和優化器狀態，內存需求是推理的 3-4 倍
2. **計算效率**：使用 GPU 後端可以顯著加速訓練
3. **批處理大小**：根據可用內存調整批處理大小
4. **梯度累積**：對於大模型，使用梯度累積來模擬更大的批次

## 限制和注意事項

1. **模型規模**：當前實現適合小到中等規模的模型
2. **訓練穩定性**：大模型訓練可能需要更複雜的技術（如梯度裁剪、學習率調度）
3. **後端支持**：並非所有算子都支持所有後端（CUDA/Metal/CPU）

## 未來發展方向

1. 實現更多的反向傳播算子
2. 添加更多優化器（SGD、RMSprop 等）
3. 支持更複雜的模型架構（如 BERT、T5）
4. 改進訓練效率和穩定性
5. 添加模型保存和加載功能

## 參考資源

- [GGML GitHub](https://github.com/ggerganov/ggml)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPT-2 論文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 結論

雖然 GGML 主要設計用於推理，但它已經具備了基本的訓練功能。通過實現額外的反向傳播算子和優化技術，可以支持更複雜的模型訓練。本項目展示了如何使用現有功能進行神經網絡訓練，並為未來的改進提供了基礎。 