#include "llama-backward.h"
#include <iostream>

SimpleNeuralModel::GradientInfo LlamaBackward::backward(
    SimpleNeuralModel* model,
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& target
) {
    std::cout << "执行反向传播，计算梯度..." << std::endl;
    
    SimpleNeuralModel::GradientInfo gradients;
    int batch_size = input[0].size();
    
    // 获取模型配置和中间结果
    const auto& config = model->get_config();
    const auto& hidden_pre_relu = model->get_hidden_pre_relu();
    const auto& hidden_post_relu = model->get_hidden_post_relu();
    const auto& output = model->get_output();
    const auto& weights2 = model->get_weights2();
    
    // 计算输出层梯度（softmax + cross-entropy）
    auto output_grad = std::vector<std::vector<float>>(config.output_size, std::vector<float>(batch_size));
    for (int i = 0; i < config.output_size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
            output_grad[i][j] = (output[i][j] - target[i][j]) / batch_size;
        }
    }
    
    // 计算第二层权重梯度
    gradients.weights2_grad.resize(config.output_size, std::vector<float>(config.hidden_size, 0.0f));
    for (int i = 0; i < config.output_size; ++i) {
        for (int k = 0; k < config.hidden_size; ++k) {
            for (int j = 0; j < batch_size; ++j) {
                gradients.weights2_grad[i][k] += output_grad[i][j] * hidden_post_relu[k][j];
            }
        }
    }
    
    // 计算第二层偏置梯度
    gradients.bias2_grad.resize(config.output_size, 0.0f);
    for (int i = 0; i < config.output_size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
            gradients.bias2_grad[i] += output_grad[i][j];
        }
    }
    
    // 计算隐藏层梯度
    auto hidden_grad = std::vector<std::vector<float>>(config.hidden_size, std::vector<float>(batch_size, 0.0f));
    for (int k = 0; k < config.hidden_size; ++k) {
        for (int j = 0; j < batch_size; ++j) {
            for (int i = 0; i < config.output_size; ++i) {
                hidden_grad[k][j] += weights2[i][k] * output_grad[i][j];
            }
        }
    }
    
    // 应用ReLU梯度
    for (int i = 0; i < config.hidden_size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
            if (hidden_pre_relu[i][j] <= 0) {
                hidden_grad[i][j] = 0.0f;
            }
        }
    }
    
    // 计算第一层权重梯度
    gradients.weights1_grad.resize(config.hidden_size, std::vector<float>(config.input_size, 0.0f));
    for (int i = 0; i < config.hidden_size; ++i) {
        for (int k = 0; k < config.input_size; ++k) {
            for (int j = 0; j < batch_size; ++j) {
                gradients.weights1_grad[i][k] += hidden_grad[i][j] * input[k][j];
            }
        }
    }
    
    // 计算第一层偏置梯度
    gradients.bias1_grad.resize(config.hidden_size, 0.0f);
    for (int i = 0; i < config.hidden_size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
            gradients.bias1_grad[i] += hidden_grad[i][j];
        }
    }
    
    return gradients;
}

 