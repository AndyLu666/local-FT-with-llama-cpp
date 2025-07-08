#pragma once

#include <cstdint>

struct GPT2Config {
    // 模型基本參數
    int32_t vocab_size = 50257;      // 詞彙表大小
    int32_t n_ctx = 1024;            // 上下文長度
    int32_t n_embd = 768;            // 嵌入維度
    int32_t n_head = 12;             // 注意力頭數
    int32_t n_layer = 12;            // Transformer 層數
    int32_t n_positions = 1024;      // 最大位置編碼數量
    
    // 激活函數和 Dropout
    float attn_pdrop = 0.1f;         // 注意力 Dropout
    float resid_pdrop = 0.1f;        // 殘差連接 Dropout
    float embd_pdrop = 0.1f;         // 嵌入 Dropout
    
    // 訓練參數
    float learning_rate = 3e-4f;     // 學習率
    float weight_decay = 0.1f;       // 權重衰減
    float beta1 = 0.9f;              // Adam beta1
    float beta2 = 0.999f;            // Adam beta2
    float eps = 1e-8f;               // Adam epsilon
    
    // 批次和序列參數
    int32_t batch_size = 8;          // 批次大小
    int32_t seq_length = 512;        // 序列長度
    int32_t grad_accumulation_steps = 1;  // 梯度累積步數
    
    // 模型尺寸預設
    static GPT2Config gpt2_small() {
        GPT2Config config;
        config.n_embd = 768;
        config.n_head = 12;
        config.n_layer = 12;
        return config;
    }
    
    static GPT2Config gpt2_medium() {
        GPT2Config config;
        config.n_embd = 1024;
        config.n_head = 16;
        config.n_layer = 24;
        return config;
    }
    
    static GPT2Config gpt2_large() {
        GPT2Config config;
        config.n_embd = 1280;
        config.n_head = 20;
        config.n_layer = 36;
        return config;
    }
    
    // 便利函數
    int32_t head_dim() const {
        return n_embd / n_head;
    }
    
    int32_t intermediate_size() const {
        return 4 * n_embd;
    }
}; 