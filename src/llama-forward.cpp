#include "llama-forward.h"
#include <iostream>

std::vector<std::vector<float>> LlamaForward::forward(
    SimpleNeuralModel* model, 
    const std::vector<std::vector<float>>& input
) {
    std::cout << "执行前向传播，输入 batch 大小: " << input[0].size() << std::endl;
    
    if (model) {
        return model->forward(input);
    }
    
    return {};
} 

 