#include "attention.h"
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

MultiHeadAttention::MultiHeadAttention(const GPT2Config& config) : config_(config) {
    c_attn_w_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, 3 * config_.n_embd});
    c_attn_b_ = std::make_unique<Tensor>(std::vector<int>{3 * config_.n_embd});
    c_proj_w_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, config_.n_embd});
    c_proj_b_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd});
    initialize_weights();
}

void MultiHeadAttention::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    // 初始化所有權重
    for (float& val : c_attn_w_->data) val = dist(gen);
    for (float& val : c_proj_w_->data) val = dist(gen);
    
    // 初始化偏置為 0
    for (float& val : c_attn_b_->data) val = 0.0f;
    for (float& val : c_proj_b_->data) val = 0.0f;
}

std::unique_ptr<Tensor> MultiHeadAttention::forward(const Tensor& hidden_states) {
    int seq_len = hidden_states.shape[0];
    int n_embd = hidden_states.shape[1];
    assert(n_embd == config_.n_embd);
    
    // 1. 線性變換生成 QKV
    auto qkv = linear_transform(hidden_states, *c_attn_w_, *c_attn_b_);
    
    // 2. 分離 Q, K, V
    auto q = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    auto k = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    auto v = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < config_.n_embd; ++j) {
            q->at({i, j}) = qkv->at({i, j});
            k->at({i, j}) = qkv->at({i, j + config_.n_embd});
            v->at({i, j}) = qkv->at({i, j + 2 * config_.n_embd});
        }
    }
    
    // 3. 重塑為多頭形式 [seq_len, n_head, head_dim]
    int head_dim = config_.head_dim();
    auto q_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_head, head_dim});
    auto k_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_head, head_dim});
    auto v_heads = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_head, head_dim});
    
    split_heads(*q, *q_heads, seq_len);
    split_heads(*k, *k_heads, seq_len);
    split_heads(*v, *v_heads, seq_len);
    
    // 4. 計算注意力
    auto attn_output = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_head, head_dim});
    scaled_dot_product_attention(*q_heads, *k_heads, *v_heads, *attn_output, seq_len);
    
    // 5. 合併多頭
    auto merged = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    merge_heads(*attn_output, *merged, seq_len);
    
    // 6. 輸出投影
    auto output = linear_transform(*merged, *c_proj_w_, *c_proj_b_);
    
    return output;
}

void MultiHeadAttention::split_heads(const Tensor& tensor, Tensor& output, int seq_len) const {
    int head_dim = config_.head_dim();
    
    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < config_.n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                int input_idx = i * config_.n_embd + h * head_dim + d;
                output.at({i, h, d}) = tensor.data[input_idx];
            }
        }
    }
}

void MultiHeadAttention::merge_heads(const Tensor& tensor, Tensor& output, int seq_len) const {
    int head_dim = config_.head_dim();
    
    for (int i = 0; i < seq_len; ++i) {
        for (int h = 0; h < config_.n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                int output_idx = i * config_.n_embd + h * head_dim + d;
                output.data[output_idx] = tensor.at({i, h, d});
            }
        }
    }
}

void MultiHeadAttention::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                                     Tensor& output, int seq_len) const {
    int head_dim = config_.head_dim();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // 計算每個頭的注意力
    for (int h = 0; h < config_.n_head; ++h) {
        // 計算注意力分數
        auto scores = std::make_unique<Tensor>(std::vector<int>{seq_len, seq_len});
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q.at({i, h, d}) * k.at({j, h, d});
                }
                scores->at({i, j}) = score * scale;
            }
        }
        
        // 應用因果掩碼
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                scores->at({i, j}) = -1e9f;  // 掩碼未來位置
            }
        }
        
        // Softmax
        for (int i = 0; i < seq_len; ++i) {
            float max_score = -1e9f;
            for (int j = 0; j <= i; ++j) {
                max_score = std::max(max_score, scores->at({i, j}));
            }
            
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                scores->at({i, j}) = std::exp(scores->at({i, j}) - max_score);
                sum += scores->at({i, j});
            }
            
            for (int j = 0; j <= i; ++j) {
                scores->at({i, j}) /= sum;
            }
        }
        
        // 應用注意力權重到 V
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float weighted_sum = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    weighted_sum += scores->at({i, j}) * v.at({j, h, d});
                }
                output.at({i, h, d}) = weighted_sum;
            }
        }
    }
}

std::unique_ptr<Tensor> MultiHeadAttention::linear_transform(const Tensor& input, 
                                                           const Tensor& weight, 
                                                           const Tensor& bias) const {
    int seq_len = input.shape[0];
    int input_dim = input.shape[1];
    int output_dim = weight.shape[1];
    
    auto output = std::make_unique<Tensor>(std::vector<int>{seq_len, output_dim});
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            float sum = bias.data[j];
            for (int k = 0; k < input_dim; ++k) {
                sum += input.at({i, k}) * weight.at({k, j});
            }
            output->at({i, j}) = sum;
        }
    }
    
    return output;
}

void MultiHeadAttention::print_info() const {
    std::cout << "MultiHeadAttention Info:" << std::endl;
    std::cout << "  注意力頭數: " << config_.n_head << std::endl;
    std::cout << "  每頭維度: " << config_.head_dim() << std::endl;
    std::cout << "  嵌入維度: " << config_.n_embd << std::endl;
    std::cout << "  QKV 權重維度: " << config_.n_embd << " x " << 3 * config_.n_embd << std::endl;
} 