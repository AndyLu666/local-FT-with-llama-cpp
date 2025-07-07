#pragma once
#include <vector>
#include <memory>
#include <functional>

class AutogradTensor {
public:
    std::vector<std::vector<float>> data;   // 數值
    std::vector<std::vector<float>> grad;   // 梯度
    // 計算圖：父節點
    std::vector<std::shared_ptr<AutogradTensor>> parents;
    // 反向傳播函數
    std::function<void(AutogradTensor&)> backward_fn;

    AutogradTensor(const std::vector<std::vector<float>>& data_);
    void backward();
    void zero_grad();
};

// 支持加法、乘法等靜態操作
std::shared_ptr<AutogradTensor> autograd_add(const std::shared_ptr<AutogradTensor>& a, const std::shared_ptr<AutogradTensor>& b);
std::shared_ptr<AutogradTensor> autograd_mul(const std::shared_ptr<AutogradTensor>& a, const std::shared_ptr<AutogradTensor>& b); 