#include "llama-loss.h"
#include <cmath>
#include <algorithm>

LlamaLoss::LlamaLoss() {}

float LlamaLoss::compute(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    // TODO: 這裡以 MSE 為例，實際可根據需要實現交叉熵等
    float loss = 0.0f;
    size_t n = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            float diff = output[i][j] - target[i][j];
            loss += diff * diff;
            ++n;
        }
    }
    return n > 0 ? loss / n : 0.0f;
}

float LlamaLoss::cross_entropy(const std::vector<float> &logits, const std::vector<int> &labels) {
    if (labels.empty() || logits.empty()) return 0.0f;
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float l : logits) sum_exp += std::exp(l - max_logit);
    float log_prob = logits[labels[0]] - max_logit - std::log(sum_exp);
    return -log_prob;
} 