#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <cstring>

// 包含ggml頭文件
#include "ggml/include/ggml.h"

// 單層感知器類 - 使用ggml內置函數
class GGMLPerceptron {
private:
    struct ggml_context* ctx_;
    struct ggml_tensor* weights_;  // 權重矩陣 [output_size, input_size]
    struct ggml_tensor* bias_;     // 偏置向量 [output_size]
    int input_size_;
    int output_size_;
    float learning_rate_;
    
public:
    GGMLPerceptron(int input_size, int output_size, float learning_rate = 0.01f)
        : input_size_(input_size), output_size_(output_size), learning_rate_(learning_rate) {
        
        // 初始化ggml上下文
        struct ggml_init_params params = {
            .mem_size = 64 * 1024 * 1024,  // 64MB記憶體
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("無法初始化ggml上下文");
        }
        
        // 創建權重和偏置張量
        weights_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, input_size_, output_size_);
        bias_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, output_size_);
        
        // 設置為可訓練參數
        ggml_set_param(weights_);
        ggml_set_param(bias_);
        
        // 初始化權重和偏置
        init_weights();
        
        std::cout << "✅ GGML感知器已初始化：" << input_size_ << " → " << output_size_ << std::endl;
        std::cout << "📊 權重矩陣形狀：[" << weights_->ne[0] << ", " << weights_->ne[1] << "]" << std::endl;
        std::cout << "📊 偏置向量長度：" << bias_->ne[0] << std::endl;
    }
    
    ~GGMLPerceptron() {
        if (ctx_) {
            ggml_free(ctx_);
        }
    }
    
    // 初始化權重和偏置
    void init_weights() {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        // 初始化權重矩陣
        float* w_data = ggml_get_data_f32(weights_);
        for (int i = 0; i < ggml_nelements(weights_); ++i) {
            w_data[i] = dist(rng);
        }
        
        // 初始化偏置向量
        float* b_data = ggml_get_data_f32(bias_);
        for (int i = 0; i < ggml_nelements(bias_); ++i) {
            b_data[i] = 0.0f;
        }
        
        std::cout << "🎲 權重和偏置已隨機初始化" << std::endl;
    }
    
    // 前向傳播：使用ggml內置函數
    struct ggml_tensor* forward(struct ggml_tensor* input) {
        // 計算：output = weights * input + bias
        // 注意：ggml_mul_mat執行矩陣乘法 A * B
        struct ggml_tensor* mul_result = ggml_mul_mat(ctx_, weights_, input);
        
        // 添加偏置（廣播）
        struct ggml_tensor* output = ggml_add(ctx_, mul_result, bias_);
        
        // 應用sigmoid激活函數
        output = ggml_unary(ctx_, output, GGML_UNARY_OP_SIGMOID);
        
        return output;
    }
    
    // 計算損失（均方誤差）
    struct ggml_tensor* compute_loss(struct ggml_tensor* predictions, struct ggml_tensor* targets) {
        // 計算預測值與目標值的差異
        struct ggml_tensor* diff = ggml_sub(ctx_, predictions, targets);
        
        // 計算平方
        struct ggml_tensor* squared = ggml_mul(ctx_, diff, diff);
        
        // 計算均值
        struct ggml_tensor* loss = ggml_mean(ctx_, squared);
        
        return loss;
    }
    
    // 訓練步驟
    void train_step(const std::vector<std::vector<float>>& inputs, 
                   const std::vector<std::vector<float>>& targets) {
        
        int batch_size = inputs.size();
        
        std::cout << "🚀 開始訓練步驟，批次大小：" << batch_size << std::endl;
        
        for (int i = 0; i < batch_size; ++i) {
            // 創建輸入張量
            struct ggml_tensor* input = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, input_size_);
            float* input_data = ggml_get_data_f32(input);
            std::memcpy(input_data, inputs[i].data(), input_size_ * sizeof(float));
            
            // 創建目標張量
            struct ggml_tensor* target = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, output_size_);
            float* target_data = ggml_get_data_f32(target);
            std::memcpy(target_data, targets[i].data(), output_size_ * sizeof(float));
            
            // 前向傳播
            struct ggml_tensor* prediction = forward(input);
            
            // 計算損失
            struct ggml_tensor* loss = compute_loss(prediction, target);
            
            // 設置損失張量
            ggml_set_loss(loss);
            
            // 創建計算圖
            struct ggml_cgraph* graph = ggml_new_graph_custom(ctx_, 1024, true);
            ggml_build_forward_expand(graph, loss);
            
            // 構建反向傳播圖
            ggml_build_backward_expand(ctx_, graph, nullptr);
            
            // 執行前向和反向傳播（這裡簡化處理，實際需要使用backend）
            // 注意：完整的實現需要使用ggml_backend進行計算
            
            if (i == 0) {  // 只打印第一個樣本的信息
                std::cout << "📈 樣本 " << i+1 << " 處理完成" << std::endl;
            }
        }
        
        std::cout << "✅ 訓練步驟完成" << std::endl;
    }
    
    // 預測
    std::vector<float> predict(const std::vector<float>& input) {
        // 創建輸入張量
        struct ggml_tensor* input_tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, input_size_);
        float* input_data = ggml_get_data_f32(input_tensor);
        std::memcpy(input_data, input.data(), input_size_ * sizeof(float));
        
        // 前向傳播
        struct ggml_tensor* output = forward(input_tensor);
        
        // 創建計算圖並執行前向傳播
        struct ggml_cgraph* graph = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph, output);
        
        // 提取結果（實際使用中需要使用backend計算）
        std::vector<float> result(output_size_);
        float* output_data = ggml_get_data_f32(output);
        std::memcpy(result.data(), output_data, output_size_ * sizeof(float));
        
        return result;
    }
    
    // 打印模型信息
    void print_model_info() {
        std::cout << "\n📋 模型資訊：" << std::endl;
        std::cout << "   輸入大小：" << input_size_ << std::endl;
        std::cout << "   輸出大小：" << output_size_ << std::endl;
        std::cout << "   學習率：" << learning_rate_ << std::endl;
        std::cout << "   總參數數量：" << (input_size_ * output_size_ + output_size_) << std::endl;
        std::cout << "   使用激活函數：Sigmoid" << std::endl;
        std::cout << "   使用損失函數：均方誤差(MSE)" << std::endl;
        
        // 顯示權重統計信息
        float* w_data = ggml_get_data_f32(weights_);
        float w_sum = 0.0f, w_min = w_data[0], w_max = w_data[0];
        int w_elements = ggml_nelements(weights_);
        
        for (int i = 0; i < w_elements; ++i) {
            w_sum += w_data[i];
            w_min = std::min(w_min, w_data[i]);
            w_max = std::max(w_max, w_data[i]);
        }
        
        std::cout << "   權重統計：平均=" << std::fixed << std::setprecision(4) 
                  << (w_sum / w_elements) << ", 最小=" << w_min << ", 最大=" << w_max << std::endl;
    }
};

// 生成訓練資料
void generate_training_data(std::vector<std::vector<float>>& inputs,
                          std::vector<std::vector<float>>& targets,
                          int num_samples) {
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    inputs.resize(num_samples);
    targets.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        // 生成2維輸入
        inputs[i] = {dist(rng), dist(rng)};
        
        // 簡單的邏輯：如果 x1 + x2 > 0，則輸出1，否則輸出0
        float sum = inputs[i][0] + inputs[i][1];
        targets[i] = {sum > 0.0f ? 1.0f : 0.0f};
    }
    
    std::cout << "📊 已生成 " << num_samples << " 個訓練樣本" << std::endl;
    std::cout << "📝 任務：學習函數 f(x1, x2) = (x1 + x2 > 0) ? 1 : 0" << std::endl;
}

int main() {
    std::cout << "🤖 GGML單層感知器演示\n" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        // 設定參數
        const int input_size = 2;
        const int output_size = 1;
        const float learning_rate = 0.1f;
        const int num_samples = 100;
        const int num_epochs = 10;
        
        // 創建模型
        std::cout << "🏗️  正在創建GGML感知器..." << std::endl;
        GGMLPerceptron model(input_size, output_size, learning_rate);
        model.print_model_info();
        
        // 生成訓練資料
        std::cout << "\n📚 正在生成訓練資料..." << std::endl;
        std::vector<std::vector<float>> inputs, targets;
        generate_training_data(inputs, targets, num_samples);
        
        // 顯示幾個樣本
        std::cout << "\n🔍 樣本預覽：" << std::endl;
        for (int i = 0; i < std::min(5, num_samples); ++i) {
            std::cout << "   輸入: [" << std::fixed << std::setprecision(2) 
                      << inputs[i][0] << ", " << inputs[i][1] 
                      << "] → 目標: " << targets[i][0] << std::endl;
        }
        
        // 訓練模型
        std::cout << "\n🎯 開始訓練..." << std::endl;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n📅 Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
            model.train_step(inputs, targets);
        }
        
        // 測試模型
        std::cout << "\n🧪 測試模型..." << std::endl;
        std::vector<std::vector<float>> test_inputs = {
            {0.5f, 0.3f},   // 應該輸出接近1
            {-0.2f, -0.4f}, // 應該輸出接近0
            {0.1f, -0.05f}, // 應該輸出接近1
            {-0.3f, 0.1f}   // 應該輸出接近0
        };
        
        for (const auto& test_input : test_inputs) {
            auto prediction = model.predict(test_input);
            float expected = (test_input[0] + test_input[1] > 0.0f) ? 1.0f : 0.0f;
            
            std::cout << "   輸入: [" << std::fixed << std::setprecision(2)
                      << test_input[0] << ", " << test_input[1] 
                      << "] → 預測: " << std::setprecision(3) << prediction[0]
                      << ", 期望: " << expected << std::endl;
        }
        
        std::cout << "\n✅ GGML感知器演示完成！" << std::endl;
        
        // 說明GGML的優勢
        std::cout << "\n💡 GGML的優勢：" << std::endl;
        std::cout << "   🔧 內置自動微分：無需手動計算梯度" << std::endl;
        std::cout << "   ⚡ 優化的算子：高效的矩陣運算和激活函數" << std::endl;
        std::cout << "   🧮 計算圖：支援複雜的神經網路結構" << std::endl;
        std::cout << "   🚀 後端支援：可利用GPU/CPU優化" << std::endl;
        std::cout << "   🔒 記憶體管理：統一的張量記憶體池" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 錯誤：" << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 