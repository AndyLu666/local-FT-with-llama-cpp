#pragma once

#include "embeddings.h"  // 使用 Tensor 定義
#include "config.h"
#include <memory>
#include <vector>

class MultiHeadAttention {
private:
    std::unique_ptr<Tensor> c_attn_w_;  // QKV 線性層權重 [n_embd, 3*n_embd]
    std::unique_ptr<Tensor> c_attn_b_;  // QKV 線性層偏置 [3*n_embd]
    std::unique_ptr<Tensor> c_proj_w_;  // 輸出投影權重 [n_embd, n_embd]
    std::unique_ptr<Tensor> c_proj_b_;  // 輸出投影偏置 [n_embd]
    
    GPT2Config config_;
    
    void initialize_weights();
    
    // 內部計算函數
    std::unique_ptr<Tensor> apply_causal_mask(const Tensor& scores) const;
    std::unique_ptr<Tensor> softmax(const Tensor& input, int dim) const;
    std::unique_ptr<Tensor> linear_transform(const Tensor& input, 
                                           const Tensor& weight, 
                                           const Tensor& bias) const;
    
public:
    explicit MultiHeadAttention(const GPT2Config& config);
    ~MultiHeadAttention() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const Tensor& hidden_states);
    
    // 調試信息
    void print_info() const;
    
private:
    // 輔助函數
    void split_heads(const Tensor& tensor, Tensor& output, int seq_len) const;
    void merge_heads(const Tensor& tensor, Tensor& output, int seq_len) const;
    void scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    Tensor& output, int seq_len) const;
}; 