# GPT-2 架構詳細說明

本文檔詳細說明了模組化 GPT-2 實現的架構設計和各個組件的功能。

## 整體架構

```
┌─────────────────┐
│   Text Input    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│   Data Loader   │ ← data_loader.h/.cpp
└─────────┬───────┘
          │
┌─────────▼───────┐
│   Embeddings    │ ← embeddings.h/.cpp
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Transformer     │
│    Blocks       │ ← transformer_block.h/.cpp
│   (x N layers)  │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  Layer Norm     │ ← layer_norm.h/.cpp
└─────────┬───────┘
          │
┌─────────▼───────┐
│   LM Head       │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  Text Output    │
└─────────────────┘
```

## 核心組件

### 1. 配置模組 (`config.h`)

定義了 GPT-2 模型的所有超參數和配置選項：

```cpp
struct GPT2Config {
    // 模型基本參數
    int32_t vocab_size = 50257;      // 詞彙表大小
    int32_t n_ctx = 1024;            // 上下文長度
    int32_t n_embd = 768;            // 嵌入維度
    int32_t n_head = 12;             // 注意力頭數
    int32_t n_layer = 12;            // Transformer 層數
    
    // 訓練參數
    float learning_rate = 3e-4f;     // 學習率
    float weight_decay = 0.1f;       // 權重衰減
    // ...
};
```

**設計原則**:
- 集中管理所有配置參數
- 提供預設的模型尺寸配置
- 支援運行時參數調整

### 2. 嵌入層 (`embeddings.h/.cpp`)

處理詞嵌入和位置嵌入：

```cpp
class Embeddings {
private:
    ggml_tensor* wte;  // 詞嵌入權重 [vocab_size, n_embd]
    ggml_tensor* wpe;  // 位置嵌入權重 [n_positions, n_embd]
    
public:
    ggml_tensor* forward(ggml_context* ctx, 
                        ggml_tensor* input_ids, 
                        ggml_tensor* position_ids = nullptr);
};
```

**功能**:
- 將 token IDs 轉換為嵌入向量
- 添加位置編碼信息
- 支援自動位置 ID 生成

### 3. 層歸一化 (`layer_norm.h/.cpp`)

實現 Layer Normalization：

```cpp
class LayerNorm {
private:
    ggml_tensor* weight;  // 縮放參數 [normalized_shape]
    ggml_tensor* bias;    // 偏移參數 [normalized_shape]
    
public:
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input);
};
```

**特性**:
- 穩定訓練過程
- 減少內部協變量偏移
- 支援可學習的縮放和偏移參數

### 4. 多頭注意力 (`attention.h/.cpp`)

實現多頭自注意力機制：

```cpp
class MultiHeadAttention {
private:
    ggml_tensor* c_attn_w;  // QKV 線性層權重
    ggml_tensor* c_attn_b;  // QKV 線性層偏置
    ggml_tensor* c_proj_w;  // 輸出投影權重
    ggml_tensor* c_proj_b;  // 輸出投影偏置
    
public:
    ggml_tensor* forward(ggml_context* ctx, 
                        ggml_tensor* hidden_states,
                        ggml_tensor* attention_mask = nullptr);
};
```

**計算流程**:
1. 線性變換生成 Query、Key、Value
2. 拆分多個注意力頭
3. 計算縮放點積注意力
4. 合併多頭結果
5. 輸出投影

**特性**:
- 支援因果掩碼 (causal mask)
- 可選的鍵值快取 (KV cache)
- 注意力權重縮放

### 5. 前饋網絡 (`feedforward.h/.cpp`)

實現 MLP (Multi-Layer Perceptron)：

```cpp
class FeedForward {
private:
    ggml_tensor* c_fc_w;    // 第一個線性層權重
    ggml_tensor* c_fc_b;    // 第一個線性層偏置
    ggml_tensor* c_proj_w;  // 第二個線性層權重
    ggml_tensor* c_proj_b;  // 第二個線性層偏置
    
public:
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* hidden_states);
};
```

**計算流程**:
1. 線性變換：`hidden_size → intermediate_size`
2. GELU 激活函數
3. Dropout (訓練時)
4. 線性變換：`intermediate_size → hidden_size`

### 6. Transformer 塊 (`transformer_block.h/.cpp`)

組合注意力和前饋網絡：

```cpp
class TransformerBlock {
private:
    std::unique_ptr<LayerNorm> ln_1;        // 第一個層歸一化
    std::unique_ptr<MultiHeadAttention> attn;  // 多頭注意力
    std::unique_ptr<LayerNorm> ln_2;        // 第二個層歸一化
    std::unique_ptr<FeedForward> mlp;       // 前饋網絡
    
public:
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* hidden_states);
};
```

**計算流程** (Pre-LayerNorm):
```
x → LayerNorm → Attention → Residual → LayerNorm → MLP → Residual → output
```

### 7. GPT-2 模型 (`model.h/.cpp`)

整合所有組件：

```cpp
class GPT2Model {
private:
    std::unique_ptr<Embeddings> embeddings;              // 嵌入層
    std::vector<std::unique_ptr<TransformerBlock>> h;     // Transformer 塊
    std::unique_ptr<LayerNorm> ln_f;                     // 最終層歸一化
    ggml_tensor* lm_head;                                // 語言模型頭部
    
public:
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input_ids);
    std::vector<int> generate(ggml_context* ctx, 
                             const std::vector<int>& input_ids);
};
```

**前向傳播流程**:
1. Token 嵌入 + 位置嵌入
2. 通過 N 個 Transformer 塊
3. 最終層歸一化
4. 語言模型頭部 (線性層) → Logits

### 8. 數據加載器 (`data_loader.h/.cpp`)

處理訓練數據：

```cpp
class DataLoader {
private:
    std::vector<int> tokenized_data_;              // 分詞後的數據
    std::vector<std::string> vocab_;               // 詞彙表
    std::unordered_map<std::string, int> token_to_id_;  // 詞到ID映射
    
public:
    std::unique_ptr<Batch> get_next_batch();
    std::vector<int> tokenize(const std::string& text);
};
```

**功能**:
- 文本分詞和編碼
- 批次數據生成
- 序列填充和截斷
- 注意力掩碼生成

### 9. 訓練器 (`trainer.h/.cpp`)

管理訓練流程：

```cpp
class Trainer {
private:
    std::shared_ptr<GPT2Model> model_;
    std::shared_ptr<DataLoader> dataloader_;
    std::unique_ptr<OptimizerState> optimizer_state_;
    
public:
    void train();
    float train_step(const Batch& batch);
    bool save_checkpoint(const std::string& filepath);
};
```

**功能**:
- Adam 優化器實現
- 梯度計算和反向傳播
- 學習率調度
- 檢查點保存和恢復
- 訓練統計和日誌

## 記憶體管理

### GGML 上下文管理

```cpp
// 創建上下文
size_t ctx_size = calculate_memory_requirements();
void* ctx_data = malloc(ctx_size);
ggml_init_params params = {
    .mem_size = ctx_size,
    .mem_buffer = ctx_data,
    .no_alloc = false
};
ggml_context* ctx = ggml_init(params);

// 使用後清理
ggml_free(ctx);
free(ctx_data);
```

### 張量生命週期

- 權重張量：在模型初始化時創建，訓練結束時釋放
- 中間張量：在前向/反向傳播時創建，計算完成後自動釋放
- 梯度張量：在反向傳播時創建，參數更新後釋放

## 性能優化

### 1. 記憶體優化
- 使用 GGML 的記憶體池管理
- 避免不必要的張量複製
- 支援原地操作 (in-place operations)

### 2. 計算優化
- 利用 SIMD 指令集 (AVX, NEON)
- OpenMP 多線程並行
- 可選的 CUDA GPU 加速

### 3. 數據流優化
- 批次處理減少計算開銷
- 數據預取和流水線處理
- 記憶體對齊優化

## 擴展性設計

### 模組化架構
- 每個組件都是獨立的類
- 清晰的接口定義
- 易於單元測試和調試

### 配置驅動
- 通過配置文件控制模型行為
- 支援運行時參數調整
- 便於實驗和超參數搜索

### 插件式設計
- 可以輕鬆添加新的激活函數
- 支援不同的注意力機制
- 可擴展的採樣策略

## 與原始 GPT-2 的差異

### 相同點
- 使用相同的 Transformer 架構
- Pre-LayerNorm 配置
- GELU 激活函數
- 相同的注意力機制

### 差異點
- 使用 GGML 而非 PyTorch/TensorFlow
- C++ 實現而非 Python
- 模組化設計便於理解和修改
- 針對 CPU 推理優化

## 未來改進方向

1. **模型壓縮**: 支援量化和剪枝
2. **分散式訓練**: 多 GPU/多節點訓練
3. **更多採樣策略**: 支援更多文本生成方法
4. **模型並行**: 支援大模型的層間並行
5. **動態批處理**: 自適應批次大小優化