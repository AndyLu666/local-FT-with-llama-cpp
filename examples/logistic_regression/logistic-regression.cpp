#include <iostream>
#include <vector>
#include <random>
#include <cmath>

struct Sample {
    std::vector<float> x;
    int y;
};

static inline float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

int main(int /*argc*/, char ** /*argv*/) {
    constexpr int n_samples  = 1000;   // 訓練樣本數
    constexpr int n_features = 2;      // 特徵維度
    constexpr int epochs     = 1000;   // 訓練輪數
    constexpr float lr       = 0.1f;   // 學習率

    // 隨機數生成器與噪聲
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.2f);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 真實參數，用來產生可分資料
    const std::vector<float> w_true = {2.0f, -3.0f};
    const float b_true = 0.5f;

    // 生成訓練資料
    std::vector<Sample> data;
    data.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> x(n_features);
        for (int j = 0; j < n_features; ++j) {
            x[j] = dist(rng);
            x[j] += noise(rng); // 加點噪聲
        }

        float z = b_true;
        for (int j = 0; j < n_features; ++j) {
            z += w_true[j] * x[j];
        }
        float p = sigmoid(z);
        float u = std::uniform_real_distribution<float>()(rng);
        int y = u < p ? 1 : 0;

        data.push_back({x, y});
    }

    // 參數初始化
    std::vector<float> w(n_features, 0.0f);
    float b = 0.0f;

    // 訓練
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> grad_w(n_features, 0.0f);
        float grad_b = 0.0f;
        float loss = 0.0f;

        for (const auto &sample : data) {
            // forward
            float z = b;
            for (int j = 0; j < n_features; ++j) {
                z += w[j] * sample.x[j];
            }
            float y_hat = sigmoid(z);

            // loss (binary cross entropy)
            loss += - (sample.y * std::log(y_hat + 1e-7f) + (1 - sample.y) * std::log(1 - y_hat + 1e-7f));

            // gradient
            float error = y_hat - sample.y; // dL/dz
            for (int j = 0; j < n_features; ++j) {
                grad_w[j] += error * sample.x[j];
            }
            grad_b += error;
        }

        // 平均梯度與損失
        for (int j = 0; j < n_features; ++j) {
            grad_w[j] /= n_samples;
        }
        grad_b /= n_samples;
        loss /= n_samples;

        // 參數更新
        for (int j = 0; j < n_features; ++j) {
            w[j] -= lr * grad_w[j];
        }
        b -= lr * grad_b;

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << "\tLoss: " << loss << std::endl;
        }
    }

    // 評估訓練集準確率
    int correct = 0;
    for (const auto &sample : data) {
        float z = b;
        for (int j = 0; j < n_features; ++j) {
            z += w[j] * sample.x[j];
        }
        int pred = sigmoid(z) > 0.5f ? 1 : 0;
        if (pred == sample.y) ++correct;
    }
    float acc = static_cast<float>(correct) / n_samples * 100.0f;
    std::cout << "Training accuracy: " << acc << "%" << std::endl;

    std::cout << "Learned parameters: w = [" << w[0] << ", " << w[1] << "], b = " << b << std::endl;

    return 0;
} 