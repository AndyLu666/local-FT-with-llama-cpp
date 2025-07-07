#pragma once
#include <vector>
#include "simple-neural-model.h"

// 优化器组件
class LlamaOptimizer {
public:
    LlamaOptimizer(SimpleNeuralModel* model, float learning_rate);

    // 执行一次参数更新
    void step();
    
    // 使用给定梯度执行参数更新
    void step(const SimpleNeuralModel::GradientInfo& gradients);

    // 清空梯度
    void zero_grad();

private:
    SimpleNeuralModel* simple_model_;
    float learning_rate_;
    const float max_grad_ = 1.0f; // 梯度裁剪阈值
    
    // 权重更新函数
    void update_weights(
        std::vector<std::vector<float>>& weights,
        const std::vector<std::vector<float>>& gradients
    );
    
    void update_bias(
        std::vector<float>& bias,
        const std::vector<float>& gradients
    );
}; 