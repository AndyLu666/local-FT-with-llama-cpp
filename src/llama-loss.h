#pragma once
#include <vector>

class LlamaLoss {
public:
    LlamaLoss();
    // 計算損失，輸入模型輸出和標籤，返回損失值
    float compute(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target);
}; 