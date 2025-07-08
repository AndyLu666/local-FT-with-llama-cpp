#pragma once

#include "embeddings.h"  // 使用 Tensor 定義
#include "config.h"
#include <memory>

class LayerNorm {
private:
    std::unique_ptr<Tensor> weight_;  // 縮放參數 [normalized_shape]
    std::unique_ptr<Tensor> bias_;    // 偏移參數 [normalized_shape]
    int normalized_shape_;
    float eps_;
    
    void initialize_weights();
    
public:
    LayerNorm(int normalized_shape, float eps = 1e-5f);
    ~LayerNorm() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const Tensor& input);
    
    // 調試信息
    void print_info() const;
    
private:
    // 輔助函數
    float compute_mean(const Tensor& input, int dim_start) const;
    float compute_variance(const Tensor& input, int dim_start, float mean) const;
}; 