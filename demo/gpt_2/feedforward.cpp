#include "feedforward.h"
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>

FeedForward::FeedForward(const GPT2Config& config) : config_(config) {
    int intermediate_size = config_.intermediate_size();
    
    c_fc_w_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, intermediate_size});
    c_fc_b_ = std::make_unique<Tensor>(std::vector<int>{intermediate_size});
    c_proj_w_ = std::make_unique<Tensor>(std::vector<int>{intermediate_size, config_.n_embd});
    c_proj_b_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd});
    
    initialize_weights();
}

void FeedForward::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    // 初始化權重
    for (float& val : c_fc_w_->data) val = dist(gen);
    for (float& val : c_proj_w_->data) val = dist(gen);
    
    // 初始化偏置為 0
    for (float& val : c_fc_b_->data) val = 0.0f;
    for (float& val : c_proj_b_->data) val = 0.0f;
}

std::unique_ptr<Tensor> FeedForward::forward(const Tensor& hidden_states) {
    // 1. 第一個線性層: hidden_size → intermediate_size
    auto intermediate = linear_transform(hidden_states, *c_fc_w_, *c_fc_b_);
    
    // 2. GELU 激活函數
    auto activated = gelu_activation(*intermediate);
    
    // 3. 第二個線性層: intermediate_size → hidden_size
    auto output = linear_transform(*activated, *c_proj_w_, *c_proj_b_);
    
    return output;
}

std::unique_ptr<Tensor> FeedForward::linear_transform(const Tensor& input, 
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

std::unique_ptr<Tensor> FeedForward::gelu_activation(const Tensor& input) const {
    auto output = std::make_unique<Tensor>(input.shape);
    
    // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    for (int i = 0; i < input.data.size(); ++i) {
        float x = input.data[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_val = std::tanh(inner);
        output->data[i] = 0.5f * x * (1.0f + tanh_val);
    }
    
    return output;
}

void FeedForward::print_info() const {
    int intermediate_size = config_.intermediate_size();
    std::cout << "FeedForward Info:" << std::endl;
    std::cout << "  輸入維度: " << config_.n_embd << std::endl;
    std::cout << "  中間層維度: " << intermediate_size << std::endl;
    std::cout << "  輸出維度: " << config_.n_embd << std::endl;
    std::cout << "  激活函數: GELU" << std::endl;
} 