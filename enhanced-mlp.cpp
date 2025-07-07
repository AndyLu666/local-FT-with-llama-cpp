#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <memory>

// 基于你的架构设计的简化版本

// ===== 前向传播组件 =====
class LlamaForward {
public:
    // 执行前向传播
    static std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& weights1,
        const std::vector<float>& bias1,
        const std::vector<std::vector<float>>& weights2,
        const std::vector<float>& bias2
    ) {
        std::cout << "执行前向传播，输入 batch 大小: " << input[0].size() << std::endl;
        
        // 第一层：线性变换 + ReLU
        auto hidden = linear_forward(input, weights1, bias1);
        auto activated = relu_forward(hidden);
        
        // 第二层：线性变换 + Softmax
        auto output = linear_forward(activated, weights2, bias2);
        auto softmax_output = softmax_forward(output);
        
        return softmax_output;
    }

private:
    static std::vector<std::vector<float>> linear_forward(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& weights,
        const std::vector<float>& bias
    ) {
        int output_size = weights.size();
        int batch_size = input[0].size();
        
        std::vector<std::vector<float>> result(output_size, std::vector<float>(batch_size, 0.0f));
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < batch_size; ++j) {
                result[i][j] = bias[i];
                for (int k = 0; k < weights[i].size(); ++k) {
                    result[i][j] += weights[i][k] * input[k][j];
                }
            }
        }
        
        return result;
    }
    
    static std::vector<std::vector<float>> relu_forward(
        const std::vector<std::vector<float>>& input
    ) {
        auto result = input;
        for (auto& row : result) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
        return result;
    }
    
    static std::vector<std::vector<float>> softmax_forward(
        const std::vector<std::vector<float>>& input
    ) {
        auto result = input;
        int batch_size = input[0].size();
        
        for (int j = 0; j < batch_size; ++j) {
            // 找到最大值以避免数值溢出
            float max_val = input[0][j];
            for (int i = 1; i < input.size(); ++i) {
                max_val = std::max(max_val, input[i][j]);
            }
            
            // 计算 exp 和 sum
            float sum = 0.0f;
            for (int i = 0; i < input.size(); ++i) {
                result[i][j] = std::exp(input[i][j] - max_val);
                sum += result[i][j];
            }
            
            // 正则化
            for (int i = 0; i < input.size(); ++i) {
                result[i][j] /= sum;
            }
        }
        
        return result;
    }
};

// ===== 反向传播组件 =====
class LlamaBackward {
public:
    struct GradientInfo {
        std::vector<std::vector<float>> weights1_grad;
        std::vector<float> bias1_grad;
        std::vector<std::vector<float>> weights2_grad;
        std::vector<float> bias2_grad;
    };
    
    static GradientInfo backward(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& hidden_pre_relu,
        const std::vector<std::vector<float>>& hidden_post_relu,
        const std::vector<std::vector<float>>& output,
        const std::vector<std::vector<float>>& target,
        const std::vector<std::vector<float>>& weights1,
        const std::vector<std::vector<float>>& weights2
    ) {
        std::cout << "执行反向传播，计算梯度..." << std::endl;
        
        GradientInfo gradients;
        int batch_size = input[0].size();
        
        // 输出层梯度（softmax + cross-entropy）
        auto output_grad = compute_output_gradient(output, target, batch_size);
        
        // 计算 weights2 和 bias2 的梯度
        gradients.weights2_grad = compute_weight_gradient(output_grad, hidden_post_relu);
        gradients.bias2_grad = compute_bias_gradient(output_grad);
        
        // 计算隐藏层梯度
        auto hidden_grad = compute_hidden_gradient(output_grad, weights2);
        auto hidden_grad_pre_relu = apply_relu_gradient(hidden_grad, hidden_pre_relu);
        
        // 计算 weights1 和 bias1 的梯度
        gradients.weights1_grad = compute_weight_gradient(hidden_grad_pre_relu, input);
        gradients.bias1_grad = compute_bias_gradient(hidden_grad_pre_relu);
        
        return gradients;
    }

private:
    static std::vector<std::vector<float>> compute_output_gradient(
        const std::vector<std::vector<float>>& output,
        const std::vector<std::vector<float>>& target,
        int batch_size
    ) {
        auto grad = output;
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < output[i].size(); ++j) {
                grad[i][j] = (output[i][j] - target[i][j]) / batch_size;
            }
        }
        return grad;
    }
    
    static std::vector<std::vector<float>> compute_weight_gradient(
        const std::vector<std::vector<float>>& output_grad,
        const std::vector<std::vector<float>>& input
    ) {
        int output_size = output_grad.size();
        int input_size = input.size();
        
        std::vector<std::vector<float>> weight_grad(output_size, std::vector<float>(input_size, 0.0f));
        
        for (int i = 0; i < output_size; ++i) {
            for (int k = 0; k < input_size; ++k) {
                for (int j = 0; j < output_grad[i].size(); ++j) {
                    weight_grad[i][k] += output_grad[i][j] * input[k][j];
                }
            }
        }
        
        return weight_grad;
    }
    
    static std::vector<float> compute_bias_gradient(
        const std::vector<std::vector<float>>& output_grad
    ) {
        std::vector<float> bias_grad(output_grad.size(), 0.0f);
        
        for (int i = 0; i < output_grad.size(); ++i) {
            for (int j = 0; j < output_grad[i].size(); ++j) {
                bias_grad[i] += output_grad[i][j];
            }
        }
        
        return bias_grad;
    }
    
    static std::vector<std::vector<float>> compute_hidden_gradient(
        const std::vector<std::vector<float>>& output_grad,
        const std::vector<std::vector<float>>& weights
    ) {
        int hidden_size = weights[0].size();
        int batch_size = output_grad[0].size();
        
        std::vector<std::vector<float>> hidden_grad(hidden_size, std::vector<float>(batch_size, 0.0f));
        
        for (int k = 0; k < hidden_size; ++k) {
            for (int j = 0; j < batch_size; ++j) {
                for (int i = 0; i < weights.size(); ++i) {
                    hidden_grad[k][j] += weights[i][k] * output_grad[i][j];
                }
            }
        }
        
        return hidden_grad;
    }
    
    static std::vector<std::vector<float>> apply_relu_gradient(
        const std::vector<std::vector<float>>& grad,
        const std::vector<std::vector<float>>& pre_relu
    ) {
        auto result = grad;
        for (int i = 0; i < grad.size(); ++i) {
            for (int j = 0; j < grad[i].size(); ++j) {
                if (pre_relu[i][j] <= 0) {
                    result[i][j] = 0.0f;
                }
            }
        }
        return result;
    }
};

// ===== 优化器组件 =====
class LlamaOptimizer {
public:
    LlamaOptimizer(float learning_rate) : learning_rate_(learning_rate) {}
    
    void step(
        std::vector<std::vector<float>>& weights1,
        std::vector<float>& bias1,
        std::vector<std::vector<float>>& weights2,
        std::vector<float>& bias2,
        const LlamaBackward::GradientInfo& gradients
    ) {
        std::cout << "执行参数更新..." << std::endl;
        
        // 更新 weights1
        update_weights(weights1, gradients.weights1_grad);
        
        // 更新 bias1
        update_bias(bias1, gradients.bias1_grad);
        
        // 更新 weights2
        update_weights(weights2, gradients.weights2_grad);
        
        // 更新 bias2
        update_bias(bias2, gradients.bias2_grad);
    }
    
    void zero_grad() {
        std::cout << "清空梯度..." << std::endl;
        // 在这个实现中，梯度每次都重新计算，所以不需要显式清空
    }

private:
    float learning_rate_;
    const float max_grad_ = 1.0f; // 梯度裁剪阈值
    
    void update_weights(
        std::vector<std::vector<float>>& weights,
        const std::vector<std::vector<float>>& gradients
    ) {
        for (int i = 0; i < weights.size(); ++i) {
            for (int j = 0; j < weights[i].size(); ++j) {
                float grad = gradients[i][j];
                // 梯度裁剪
                grad = std::max(-max_grad_, std::min(max_grad_, grad));
                weights[i][j] -= learning_rate_ * grad;
            }
        }
    }
    
    void update_bias(
        std::vector<float>& bias,
        const std::vector<float>& gradients
    ) {
        for (int i = 0; i < bias.size(); ++i) {
            float grad = gradients[i];
            // 梯度裁剪
            grad = std::max(-max_grad_, std::min(max_grad_, grad));
            bias[i] -= learning_rate_ * grad;
        }
    }
};

// ===== 增强版多层感知器 =====
class EnhancedMLP {
private:
    std::vector<std::vector<float>> weights1_, weights2_;
    std::vector<float> bias1_, bias2_;
    int input_size_, hidden_size_, output_size_;
    std::unique_ptr<LlamaOptimizer> optimizer_;
    
public:
    EnhancedMLP(int input_size, int hidden_size, int output_size, float learning_rate)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {
        
        // 初始化权重
        init_weights();
        
        // 建立优化器
        optimizer_ = std::make_unique<LlamaOptimizer>(learning_rate);
    }
    
    void init_weights() {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        // 初始化 weights1 (hidden_size x input_size)
        weights1_.resize(hidden_size_, std::vector<float>(input_size_));
        for (auto& row : weights1_) {
            for (auto& val : row) {
                val = dist(rng) * 0.5f;
            }
        }
        
        // 初始化 bias1
        bias1_.resize(hidden_size_, 0.0f);
        
        // 初始化 weights2 (output_size x hidden_size)
        weights2_.resize(output_size_, std::vector<float>(hidden_size_));
        for (auto& row : weights2_) {
            for (auto& val : row) {
                val = dist(rng) * 0.5f;
            }
        }
        
        // 初始化 bias2
        bias2_.resize(output_size_, 0.0f);
    }
    
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input) {
        return LlamaForward::forward(input, weights1_, bias1_, weights2_, bias2_);
    }
    
    void train_step(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& target
    ) {
        // 前向传播（保存中间结果用于反向传播）
        auto hidden_pre_relu = linear_transform(input, weights1_, bias1_);
        auto hidden_post_relu = apply_relu(hidden_pre_relu);
        auto output_pre_softmax = linear_transform(hidden_post_relu, weights2_, bias2_);
        auto output = apply_softmax(output_pre_softmax);
        
        // 反向传播
        auto gradients = LlamaBackward::backward(
            input, hidden_pre_relu, hidden_post_relu, output, target,
            weights1_, weights2_
        );
        
        // 参数更新
        optimizer_->step(weights1_, bias1_, weights2_, bias2_, gradients);
    }
    
    std::pair<float, float> evaluate(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& target
    ) {
        auto output = forward(input);
        
        float loss = 0.0f;
        int correct = 0;
        int batch_size = input[0].size();
        
        for (int j = 0; j < batch_size; ++j) {
            // 计算交叉熵损失
            for (int i = 0; i < output_size_; ++i) {
                if (target[i][j] > 0.5f) {
                    loss -= std::log(output[i][j] + 1e-7f);
                }
            }
            
            // 计算准确率
            int pred_class = 0;
            int true_class = 0;
            float max_prob = -1.0f;
            
            for (int i = 0; i < output_size_; ++i) {
                if (output[i][j] > max_prob) {
                    max_prob = output[i][j];
                    pred_class = i;
                }
                if (target[i][j] > 0.5f) {
                    true_class = i;
                }
            }
            
            if (pred_class == true_class) {
                correct++;
            }
        }
        
        loss /= batch_size;
        float accuracy = (float)correct / batch_size;
        
        return {loss, accuracy};
    }

private:
    std::vector<std::vector<float>> linear_transform(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& weights,
        const std::vector<float>& bias
    ) {
        int output_size = weights.size();
        int batch_size = input[0].size();
        
        std::vector<std::vector<float>> result(output_size, std::vector<float>(batch_size, 0.0f));
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < batch_size; ++j) {
                result[i][j] = bias[i];
                for (int k = 0; k < weights[i].size(); ++k) {
                    result[i][j] += weights[i][k] * input[k][j];
                }
            }
        }
        
        return result;
    }
    
    std::vector<std::vector<float>> apply_relu(const std::vector<std::vector<float>>& input) {
        auto result = input;
        for (auto& row : result) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
        return result;
    }
    
    std::vector<std::vector<float>> apply_softmax(const std::vector<std::vector<float>>& input) {
        auto result = input;
        int batch_size = input[0].size();
        
        for (int j = 0; j < batch_size; ++j) {
            float max_val = input[0][j];
            for (int i = 1; i < input.size(); ++i) {
                max_val = std::max(max_val, input[i][j]);
            }
            
            float sum = 0.0f;
            for (int i = 0; i < input.size(); ++i) {
                result[i][j] = std::exp(input[i][j] - max_val);
                sum += result[i][j];
            }
            
            for (int i = 0; i < input.size(); ++i) {
                result[i][j] /= sum;
            }
        }
        
        return result;
    }
};

// ===== 数据生成函数 =====
void generate_dataset(
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y,
    int n_samples, int n_features, int n_classes
) {
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    
    for (int i = 0; i < n_samples; ++i) {
        int label = i % n_classes;
        float angle = 2.0f * M_PI * label / n_classes;
        float radius = 2.0f;
        
        // 生成特征
        X[0][i] = radius * std::cos(angle) + noise(rng);
        X[1][i] = radius * std::sin(angle) + noise(rng);
        for (int j = 2; j < n_features; ++j) {
            X[j][i] = noise(rng);
        }
        
        // 设定标签（one-hot）
        for (int j = 0; j < n_classes; ++j) {
            Y[j][i] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

// ===== 主程序 =====
int main() {
    // 设定参数
    const int n_samples = 1000;
    const int n_features = 4;
    const int n_hidden = 16;
    const int n_classes = 3;
    const int n_epochs = 500;
    const float learning_rate = 0.001f;
    const int batch_size = 32;
    
    // 生成数据
    std::vector<std::vector<float>> X_train(n_features, std::vector<float>(n_samples));
    std::vector<std::vector<float>> Y_train(n_classes, std::vector<float>(n_samples));
    generate_dataset(X_train, Y_train, n_samples, n_features, n_classes);
    
    // 建立增强版模型
    EnhancedMLP model(n_features, n_hidden, n_classes, learning_rate);
    
    // 训练
    std::cout << "开始训练增强版多层感知器...\n";
    std::cout << "使用您的 forward、backward 和 optimizer 组件\n";
    std::cout << "样本数: " << n_samples << ", 特征数: " << n_features 
              << ", 隐藏层大小: " << n_hidden << ", 类别数: " << n_classes << "\n";
    std::cout << "批次大小: " << batch_size << ", 学习率: " << learning_rate << "\n";
    std::cout << "========================================\n";
    
    std::mt19937 rng(123);
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        // 打乱索引
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int n_batches = (n_samples + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < n_batches; ++batch) {
            int start = batch * batch_size;
            int end = std::min(start + batch_size, n_samples);
            int actual_batch_size = end - start;
            
            // 准备批次数据
            std::vector<std::vector<float>> X_batch(n_features, std::vector<float>(actual_batch_size));
            std::vector<std::vector<float>> Y_batch(n_classes, std::vector<float>(actual_batch_size));
            
            for (int i = 0; i < actual_batch_size; ++i) {
                int idx = indices[start + i];
                for (int j = 0; j < n_features; ++j) {
                    X_batch[j][i] = X_train[j][idx];
                }
                for (int j = 0; j < n_classes; ++j) {
                    Y_batch[j][i] = Y_train[j][idx];
                }
            }
            
            // 训练步骤
            model.train_step(X_batch, Y_batch);
            
            // 评估
            auto [loss, acc] = model.evaluate(X_batch, Y_batch);
            epoch_loss += loss * actual_batch_size;
            epoch_acc += acc * actual_batch_size;
        }
        
        epoch_loss /= n_samples;
        epoch_acc /= n_samples;
        
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << ": Loss = " << std::fixed << std::setprecision(4) << epoch_loss 
                      << ", Accuracy = " << std::fixed << std::setprecision(1) << epoch_acc * 100 << "%\n";
        }
    }
    
    std::cout << "========================================\n";
    std::cout << "训练完成！使用了您的组件架构\n";
    
    return 0;
} 