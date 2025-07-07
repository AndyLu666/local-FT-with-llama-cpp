# GPT-2 訓練實現指南

本指南說明如何在 llama.cpp/GGML 中實現 GPT-2 訓練功能。

## 現有功能

GGML 已經實現了以下訓練相關的算子：

### 已實現的反向傳播算子
1. **GGML_OP_CROSS_ENTROPY_LOSS_BACK** - 交叉熵損失反向傳播
2. **GGML_OP_SOFT_MAX_BACK** - Softmax 反向傳播
3. **GGML_OP_RMS_NORM_BACK** - RMS Normalization 反向傳播
4. **GGML_OP_SILU_BACK** - SiLU 激活函數反向傳播
5. **GGML_OP_ROPE_BACK** - RoPE 位置編碼反向傳播
6. **GGML_OP_GET_ROWS_BACK** - 嵌入查找反向傳播
7. **GGML_OP_OPT_STEP_ADAMW** - AdamW 優化器

### 自動微分支持
- `ggml_build_backward_expand()` - 自動構建反向傳播圖
- 基本算術運算（add, mul, div 等）的梯度已自動實現

## 需要實現的功能

為了完整支持 GPT-2 訓練，還需要實現以下功能：

### 1. Layer Normalization 反向傳播

```cpp
// 需要在 ggml.h 中添加
GGML_OP_NORM_BACK,

// 在 ggml.c 中實現
struct ggml_tensor * ggml_norm_back(
    struct ggml_context * ctx,
    struct ggml_tensor  * grad,    // 輸出梯度
    struct ggml_tensor  * input,   // 原始輸入
    float                 eps) {
    
    struct ggml_tensor * result = ggml_dup_tensor(ctx, input);
    
    result->op     = GGML_OP_NORM_BACK;
    result->src[0] = grad;
    result->src[1] = input;
    
    float params[] = { eps };
    ggml_set_op_params(result, params, sizeof(params));
    
    return result;
}
```

### 2. GELU 反向傳播

```cpp
// 需要在 ggml.h 中添加
GGML_OP_GELU_BACK,

// 在 ggml.c 中實現
struct ggml_tensor * ggml_gelu_back(
    struct ggml_context * ctx,
    struct ggml_tensor  * grad,    // 輸出梯度
    struct ggml_tensor  * input) { // 原始輸入
    
    struct ggml_tensor * result = ggml_dup_tensor(ctx, input);
    
    result->op     = GGML_OP_GELU_BACK;
    result->src[0] = grad;
    result->src[1] = input;
    
    return result;
}
```

### 3. 矩陣乘法反向傳播

這是最複雜的部分，需要計算兩個輸入的梯度：

```cpp
// 在 ggml_compute_backward() 中添加
case GGML_OP_MUL_MAT: {
    // grad_a = grad @ b^T
    if (src0_needs_grads) {
        struct ggml_tensor* grad_a = ggml_mul_mat(ctx, 
            ggml_transpose(ctx, src1), grad);
        ggml_add_or_set(ctx, cgraph, isrc0, grad_a);
    }
    
    // grad_b = a^T @ grad
    if (src1_needs_grads) {
        struct ggml_tensor* grad_b = ggml_mul_mat(ctx,
            ggml_transpose(ctx, src0), grad);
        ggml_add_or_set(ctx, cgraph, isrc1, grad_b);
    }
} break;
```

## 訓練流程實現

### 1. 模型定義

```cpp
struct GPT2Model {
    // 模型配置
    int n_vocab;
    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
    
    // 模型參數（需要設置 GGML_TENSOR_FLAG_PARAM）
    struct ggml_tensor* wte;      // token embeddings
    struct ggml_tensor* wpe;      // position embeddings
    // ... 其他參數
};
```

### 2. 前向傳播

```cpp
struct ggml_tensor* gpt2_forward(
    struct ggml_context* ctx,
    GPT2Model& model,
    struct ggml_tensor* input_ids
) {
    // 1. Embeddings
    auto tok_emb = ggml_get_rows(ctx, model.wte, input_ids);
    auto pos_emb = ggml_get_rows(ctx, model.wpe, positions);
    auto x = ggml_add(ctx, tok_emb, pos_emb);
    
    // 2. Transformer layers
    for (int i = 0; i < model.n_layer; ++i) {
        // Layer norm
        x = ggml_norm(ctx, x, eps);
        
        // Self-attention
        // ...
        
        // FFN
        // ...
    }
    
    // 3. Final layer norm and output projection
    x = ggml_norm(ctx, x, eps);
    auto logits = ggml_mul_mat(ctx, model.wte, x);
    
    return logits;
}
```

### 3. 訓練循環

```cpp
void train_gpt2(GPT2Model& model) {
    // 創建優化器參數
    auto adamw_params = create_adamw_params(learning_rate);
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        for (auto& batch : dataloader) {
            // 創建計算圖上下文
            auto ctx = ggml_init(params);
            
            // 前向傳播
            auto logits = gpt2_forward(ctx, model, batch.input_ids);
            
            // 計算損失
            auto loss = ggml_cross_entropy_loss(ctx, logits, batch.labels);
            ggml_set_loss(loss);
            
            // 構建計算圖
            auto gf = ggml_new_graph_custom(ctx, size, true);
            ggml_build_forward_expand(gf, loss);
            
            // 前向計算
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            
            // 構建反向圖
            ggml_build_backward_expand(ctx, gf, nullptr);
            
            // 反向計算
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            
            // 更新參數
            update_parameters(model, gf, adamw_params);
            
            ggml_free(ctx);
        }
    }
}
```

## 優化建議

### 1. 梯度累積
對於大模型，可能需要梯度累積來模擬更大的批次大小：

```cpp
// 累積多個小批次的梯度
for (int i = 0; i < gradient_accumulation_steps; ++i) {
    // 前向和反向傳播
    // 累積梯度而不是立即更新
}
// 更新參數
```

### 2. 混合精度訓練
使用 F16 進行前向傳播，F32 進行參數更新：

```cpp
// 將模型權重轉換為 F16
auto weight_f16 = ggml_cpy(ctx, weight_f32, 
    ggml_new_tensor(..., GGML_TYPE_F16));

// 前向傳播使用 F16
// 梯度計算和參數更新使用 F32
```

### 3. 梯度檢查點
減少內存使用：

```cpp
// 標記某些張量為檢查點
ggml_set_checkpoint(tensor);

// 在反向傳播時重新計算
```

## 實現步驟

1. **第一步**：實現缺失的反向傳播算子
   - NORM_BACK
   - GELU_BACK
   - 完善 MUL_MAT 的反向傳播

2. **第二步**：創建完整的 GPT-2 模型結構
   - 正確的注意力機制
   - 位置編碼
   - Layer normalization

3. **第三步**：實現數據加載和預處理
   - Tokenization
   - Batching
   - Padding

4. **第四步**：優化和調試
   - 梯度裁剪
   - 學習率調度
   - 驗證集評估

## 示例代碼

完整的示例代碼請參考：
- `demo/simple-gpt2-train.cpp` - 簡化的訓練示例
- `demo/gpt2-train.cpp` - 完整的訓練實現

## 注意事項

1. **內存管理**：訓練需要大量內存，注意及時釋放不需要的張量
2. **數值穩定性**：使用適當的初始化和正則化技術
3. **性能優化**：利用 GPU 加速和並行計算

## 參考資源

- [GPT-2 論文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GGML 文檔](https://github.com/ggerganov/ggml)
- [llama.cpp 訓練示例](https://github.com/ggerganov/llama.cpp/tree/master/examples/train-text-from-scratch) 