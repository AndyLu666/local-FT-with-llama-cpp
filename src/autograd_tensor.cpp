#include "autograd_tensor.h"
#include <algorithm>
#include <cassert>

AutogradTensor::AutogradTensor(const std::vector<std::vector<float>>& data_)
    : data(data_), grad(data_.size(), std::vector<float>(data_[0].size(), 0.0f)) {}

void AutogradTensor::backward() {
    // 如果是 loss，初始化自身梯度為 1
    for (auto& row : grad) {
        std::fill(row.begin(), row.end(), 1.0f);
    }
    if (backward_fn) backward_fn(*this);
    for (auto& p : parents) {
        if (p) p->backward();
    }
}

void AutogradTensor::zero_grad() {
    for (auto& row : grad) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
}

std::shared_ptr<AutogradTensor> autograd_add(const std::shared_ptr<AutogradTensor>& a, const std::shared_ptr<AutogradTensor>& b) {
    assert(a->data.size() == b->data.size() && a->data[0].size() == b->data[0].size());
    std::vector<std::vector<float>> out(a->data.size(), std::vector<float>(a->data[0].size()));
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = 0; j < out[i].size(); ++j)
            out[i][j] = a->data[i][j] + b->data[i][j];
    auto result = std::make_shared<AutogradTensor>(out);
    result->parents = {a, b};
    result->backward_fn = [a, b](AutogradTensor& self) {
        for (size_t i = 0; i < self.grad.size(); ++i)
            for (size_t j = 0; j < self.grad[i].size(); ++j) {
                a->grad[i][j] += self.grad[i][j];
                b->grad[i][j] += self.grad[i][j];
            }
    };
    return result;
}

std::shared_ptr<AutogradTensor> autograd_mul(const std::shared_ptr<AutogradTensor>& a, const std::shared_ptr<AutogradTensor>& b) {
    assert(a->data.size() == b->data.size() && a->data[0].size() == b->data[0].size());
    std::vector<std::vector<float>> out(a->data.size(), std::vector<float>(a->data[0].size()));
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = 0; j < out[i].size(); ++j)
            out[i][j] = a->data[i][j] * b->data[i][j];
    auto result = std::make_shared<AutogradTensor>(out);
    result->parents = {a, b};
    result->backward_fn = [a, b](AutogradTensor& self) {
        for (size_t i = 0; i < self.grad.size(); ++i)
            for (size_t j = 0; j < self.grad[i].size(); ++j) {
                a->grad[i][j] += b->data[i][j] * self.grad[i][j];
                b->grad[i][j] += a->data[i][j] * self.grad[i][j];
            }
    };
    return result;
} 