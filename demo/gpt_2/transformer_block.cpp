#include "transformer_block.h"
#include <iostream>
#include <cassert>

TransformerBlock::TransformerBlock(const GPT2Config& config) : config_(config) {
    ln_1_ = std::make_unique<LayerNorm>(config_.n_embd);
    attn_ = std::make_unique<MultiHeadAttention>(config_);
    ln_2_ = std::make_unique<LayerNorm>(config_.n_embd);
    mlp_ = std::make_unique<FeedForward>(config_);
}

std::unique_ptr<Tensor> TransformerBlock::forward(const Tensor& hidden_states) {
    // Pre-LayerNorm 架構
    // x → LayerNorm → Attention → Residual → LayerNorm → MLP → Residual → output
    
    // 1. 第一個 LayerNorm + 注意力 + 殘差連接
    auto ln1_output = ln_1_->forward(hidden_states);
    auto attn_output = attn_->forward(*ln1_output);
    auto residual1 = add_residual(hidden_states, *attn_output);
    
    // 2. 第二個 LayerNorm + MLP + 殘差連接
    auto ln2_output = ln_2_->forward(*residual1);
    auto mlp_output = mlp_->forward(*ln2_output);
    auto residual2 = add_residual(*residual1, *mlp_output);
    
    return residual2;
}

std::unique_ptr<Tensor> TransformerBlock::add_residual(const Tensor& input, const Tensor& residual) const {
    assert(input.shape == residual.shape);
    
    auto output = std::make_unique<Tensor>(input.shape);
    
    for (int i = 0; i < input.data.size(); ++i) {
        output->data[i] = input.data[i] + residual.data[i];
    }
    
    return output;
}

void TransformerBlock::print_info() const {
    std::cout << "TransformerBlock Info:" << std::endl;
    std::cout << "  架構: Pre-LayerNorm" << std::endl;
    std::cout << "  組件: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual" << std::endl;
    std::cout << "  嵌入維度: " << config_.n_embd << std::endl;
    
    std::cout << "\n  子組件詳情:" << std::endl;
    attn_->print_info();
    mlp_->print_info();
} 