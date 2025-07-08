#pragma once

#include "layer_norm.h"
#include "attention.h"
#include "feedforward.h"
#include "config.h"
#include <memory>

class TransformerBlock {
private:
    std::unique_ptr<LayerNorm> ln_1_;        // 第一個層歸一化
    std::unique_ptr<MultiHeadAttention> attn_;  // 多頭注意力
    std::unique_ptr<LayerNorm> ln_2_;        // 第二個層歸一化
    std::unique_ptr<FeedForward> mlp_;       // 前饋網絡
    
    GPT2Config config_;
    
    // 殘差連接
    std::unique_ptr<Tensor> add_residual(const Tensor& input, const Tensor& residual) const;
    
public:
    explicit TransformerBlock(const GPT2Config& config);
    ~TransformerBlock() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const Tensor& hidden_states);
    
    // 調試信息
    void print_info() const;
}; 