#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <fstream>

// 引用您在src目录中编写的组件
#include "../src/simple-neural-model.h"
#include "../src/llama-forward.h"
#include "../src/llama-backward.h"
#include "../src/llama-optimizer.h"

// ===== 使用您的src组件的增强版多层感知器 =====
class EnhancedMLPWithSrc {
private:
    std::unique_ptr<SimpleNeuralModel> model_;
    std::unique_ptr<LlamaForward> forward_;
    std::unique_ptr<LlamaBackward> backward_;
    std::unique_ptr<LlamaOptimizer> optimizer_;
    
public:
    EnhancedMLPWithSrc(int input_size, int hidden_size, int output_size, float learning_rate) {
        // 创建模型配置
        SimpleNeuralModel::NetworkConfig config;
        config.input_size = input_size;
        config.hidden_size = hidden_size;
        config.output_size = output_size;
        config.learning_rate = learning_rate;
        
        // 创建模型和组件
        model_ = std::make_unique<SimpleNeuralModel>(config);
        forward_ = std::make_unique<LlamaForward>();
        backward_ = std::make_unique<LlamaBackward>();
        optimizer_ = std::make_unique<LlamaOptimizer>(model_.get(), learning_rate);
    }
    
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input) {
        // 调用前向传播函数
        return LlamaForward::forward(model_.get(), input);
    }
    
    void train_step(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& target
    ) {
        // 1. 前向传播
        auto output = LlamaForward::forward(model_.get(), input);
        
        // 2. 反向传播计算梯度
        auto gradients = LlamaBackward::backward(model_.get(), input, target);
        
        // 3. 更新网络参数
        optimizer_->step(gradients);
    }
    
    std::pair<float, float> evaluate(
        const std::vector<std::vector<float>>& input,
        const std::vector<std::vector<float>>& target
    ) {
        auto output = forward(input);
        
        float loss = 0.0f;
        int correct = 0;
        int batch_size = input[0].size();
        const auto& config = model_->get_config();
        
        for (int j = 0; j < batch_size; ++j) {
            // 计算交叉熵损失
            for (int i = 0; i < config.output_size; ++i) {
                if (target[i][j] > 0.5f) {
                    loss -= std::log(output[i][j] + 1e-7f);
                }
            }
            
            // 计算准确率
            int pred_class = 0;
            int true_class = 0;
            float max_prob = -1.0f;
            
            for (int i = 0; i < config.output_size; ++i) {
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
    
    void save_model(const std::string& filepath) {
        model_->save_model(filepath);
    }
    
    bool load_model(const std::string& filepath) {
        return model_->load_model(filepath);
    }
    
    // 获取模型信息
    void print_model_info() {
        const auto& config = model_->get_config();
        std::cout << "模型架构: " << config.input_size << " → " 
                  << config.hidden_size << " → " << config.output_size << std::endl;
        std::cout << "学习率: " << config.learning_rate << std::endl;
        std::cout << "使用组件:" << std::endl;
        std::cout << "  - llama-forward: 前向传播" << std::endl;
        std::cout << "  - llama-backward: 反向传播" << std::endl;
        std::cout << "  - llama-optimizer: 参数优化" << std::endl;
        std::cout << "  - simple-neural-model: 模型管理" << std::endl;
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
    
    // 创建神经网络模型
    EnhancedMLPWithSrc model(n_features, n_hidden, n_classes, learning_rate);
    
    // 打印模型信息
    std::cout << "========================================\n";
    std::cout << "创建增强版多层感知器\n";
    model.print_model_info();
    std::cout << "训练数据: " << n_samples << "个样本, " 
              << n_features << "个特征, " << n_classes << "个类别\n";
    std::cout << "训练参数: " << n_epochs << "个epoch, 批次大小 " << batch_size 
              << ", 学习率 " << learning_rate << "\n";
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
            
            // 执行训练步骤
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
    std::cout << "训练完成！使用的组件：\n";
    std::cout << "✅ LlamaForward: 前向传播\n";
    std::cout << "✅ LlamaBackward: 反向传播\n";
    std::cout << "✅ LlamaOptimizer: 参数优化\n";
    std::cout << "✅ SimpleNeuralModel: 模型管理\n";
    
    // 保存训练好的模型
    std::string model_path = "enhanced_mlp_with_src_model.bin";
    model.save_model(model_path);
    
    // 测试模型加载
    std::cout << "\n测试模型保存和加载功能...\n";
    EnhancedMLPWithSrc test_model(n_features, n_hidden, n_classes, learning_rate);
    if (test_model.load_model(model_path)) {
        // 使用加载的模型进行一次评估
        std::vector<std::vector<float>> X_test(n_features, std::vector<float>(100));
        std::vector<std::vector<float>> Y_test(n_classes, std::vector<float>(100));
        generate_dataset(X_test, Y_test, 100, n_features, n_classes);
        
        auto [test_loss, test_acc] = test_model.evaluate(X_test, Y_test);
        std::cout << "加载模型的测试性能: Loss = " << std::fixed << std::setprecision(4) << test_loss 
                  << ", Accuracy = " << std::fixed << std::setprecision(1) << test_acc * 100 << "%\n";
    }
    
    std::cout << "模型文件已保存在: " << model_path << std::endl;
    
    return 0;
}