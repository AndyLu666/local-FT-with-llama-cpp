#include "llama-optimizer.h"
#include <iostream>
#include <algorithm>

LlamaOptimizer::LlamaOptimizer(SimpleNeuralModel* model, float learning_rate)
    : simple_model_(model), learning_rate_(learning_rate) {}

void LlamaOptimizer::step() {
    if (simple_model_) {
        // 获取模型中存储的梯度
        const auto& gradients = simple_model_->get_gradients();
        step(gradients);
    }
}

void LlamaOptimizer::step(const SimpleNeuralModel::GradientInfo& gradients) {
    std::cout << "执行参数更新..." << std::endl;
    
    if (!simple_model_) {
        std::cerr << "错误：没有绑定SimpleNeuralModel" << std::endl;
        return;
    }
    
    // 获取当前权重
    auto weights1 = simple_model_->get_weights1();
    auto bias1 = simple_model_->get_bias1();
    auto weights2 = simple_model_->get_weights2();
    auto bias2 = simple_model_->get_bias2();
    
    // 更新参数
    update_weights(weights1, gradients.weights1_grad);
    update_bias(bias1, gradients.bias1_grad);
    update_weights(weights2, gradients.weights2_grad);
    update_bias(bias2, gradients.bias2_grad);
    
    // 将更新后的参数设置回模型
    simple_model_->update_weights1(weights1);
    simple_model_->update_bias1(bias1);
    simple_model_->update_weights2(weights2);
    simple_model_->update_bias2(bias2);
}

void LlamaOptimizer::zero_grad() {
    if (simple_model_) {
        simple_model_->zero_grad();
    }
    std::cout << "清空梯度..." << std::endl;
}

void LlamaOptimizer::update_weights(
    std::vector<std::vector<float>>& weights,
    const std::vector<std::vector<float>>& gradients
) {
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            float grad = gradients[i][j];
            // 梯度裁剪
            grad = std::max(-max_grad_, std::min(max_grad_, grad));
            // SGD更新：w = w - η × grad
            weights[i][j] -= learning_rate_ * grad;
        }
    }
}

void LlamaOptimizer::update_bias(
    std::vector<float>& bias,
    const std::vector<float>& gradients
) {
    for (int i = 0; i < bias.size(); ++i) {
        float grad = gradients[i];
        // 梯度裁剪
        grad = std::max(-max_grad_, std::min(max_grad_, grad));
        // SGD更新：b = b - η × grad
        bias[i] -= learning_rate_ * grad;
    }
} 