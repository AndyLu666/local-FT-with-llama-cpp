#include "layer_norm.h"
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>

LayerNorm::LayerNorm(int normalized_shape, float eps) 
    : normalized_shape_(normalized_shape), eps_(eps) {
    weight_ = std::make_unique<Tensor>(std::vector<int>{normalized_shape});
    bias_ = std::make_unique<Tensor>(std::vector<int>{normalized_shape});
    initialize_weights();
}

void LayerNorm::initialize_weights() {
    // 初始化縮放參數為 1
    for (float& val : weight_->data) {
        val = 1.0f;
    }
    
    // 初始化偏移參數為 0
    for (float& val : bias_->data) {
        val = 0.0f;
    }
}

std::unique_ptr<Tensor> LayerNorm::forward(const Tensor& input) {
    assert(input.shape.back() == normalized_shape_);
    
    auto output = std::make_unique<Tensor>(input.shape);
    
    // 計算每個樣本的均值和方差
    int batch_size = 1;
    for (int i = 0; i < input.shape.size() - 1; ++i) {
        batch_size *= input.shape[i];
    }
    
    for (int b = 0; b < batch_size; ++b) {
        // 計算均值
        float sum = 0.0f;
        for (int i = 0; i < normalized_shape_; ++i) {
            int idx = b * normalized_shape_ + i;
            sum += input.data[idx];
        }
        float mean = sum / normalized_shape_;
        
        // 計算方差
        float var_sum = 0.0f;
        for (int i = 0; i < normalized_shape_; ++i) {
            int idx = b * normalized_shape_ + i;
            float diff = input.data[idx] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / normalized_shape_;
        float std_dev = std::sqrt(variance + eps_);
        
        // 歸一化並應用縮放和偏移
        for (int i = 0; i < normalized_shape_; ++i) {
            int idx = b * normalized_shape_ + i;
            float normalized = (input.data[idx] - mean) / std_dev;
            output->data[idx] = normalized * weight_->data[i] + bias_->data[i];
        }
    }
    
    return output;
}

float LayerNorm::compute_mean(const Tensor& input, int dim_start) const {
    float sum = 0.0f;
    int count = 0;
    for (int i = dim_start; i < dim_start + normalized_shape_; ++i) {
        sum += input.data[i];
        count++;
    }
    return sum / count;
}

float LayerNorm::compute_variance(const Tensor& input, int dim_start, float mean) const {
    float var_sum = 0.0f;
    int count = 0;
    for (int i = dim_start; i < dim_start + normalized_shape_; ++i) {
        float diff = input.data[i] - mean;
        var_sum += diff * diff;
        count++;
    }
    return var_sum / count;
}

void LayerNorm::print_info() const {
    std::cout << "LayerNorm Info:" << std::endl;
    std::cout << "  歸一化維度: " << normalized_shape_ << std::endl;
    std::cout << "  Epsilon: " << eps_ << std::endl;
    std::cout << "  權重範圍: " << weight_->data.front() << " ~ " << weight_->data.back() << std::endl;
} 