#pragma once
#include <vector>
#include "simple-neural-model.h"

// 反向传播组件
class LlamaBackward {
public:
    // 使用简化模型的反向传播
    static SimpleNeuralModel::GradientInfo backward(
        SimpleNeuralModel* model,
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& target
    );
}; 