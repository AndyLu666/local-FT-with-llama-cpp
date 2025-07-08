#pragma once

#include "embeddings.h"  // 使用 Tensor 定義
#include "config.h"
#include <memory>

class FeedForward {
private:
    std::unique_ptr<Tensor> c_fc_w_;    // 第一個線性層權重 [n_embd, intermediate_size]
    std::unique_ptr<Tensor> c_fc_b_;    // 第一個線性層偏置 [intermediate_size]
    std::unique_ptr<Tensor> c_proj_w_;  // 第二個線性層權重 [intermediate_size, n_embd]
    std::unique_ptr<Tensor> c_proj_b_;  // 第二個線性層偏置 [n_embd]
    
    GPT2Config config_;
    
    void initialize_weights();
    
    // 內部計算函數
    std::unique_ptr<Tensor> linear_transform(const Tensor& input, 
                                           const Tensor& weight, 
                                           const Tensor& bias) const;
    std::unique_ptr<Tensor> gelu_activation(const Tensor& input) const;
    
public:
    explicit FeedForward(const GPT2Config& config);
    ~FeedForward() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const Tensor& hidden_states);
    
    // 調試信息
    void print_info() const;
}; 