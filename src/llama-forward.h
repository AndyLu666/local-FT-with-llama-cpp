#pragma once
#include <vector>
#include "simple-neural-model.h"

// 前向传播组件
class LlamaForward {
public:
    // 使用简化模型的前向传播
    static std::vector<std::vector<float>> forward(
        SimpleNeuralModel* model, 
        const std::vector<std::vector<float>>& input
    );
}; 