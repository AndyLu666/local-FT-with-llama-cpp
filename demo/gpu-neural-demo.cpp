#include "../src/simple-neural-model.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// 生成合成數據集
void generate_synthetic_data(std::vector<std::vector<float>>& inputs, 
                           std::vector<int>& labels, 
                           int num_samples = 1000) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    inputs.resize(num_samples);
    labels.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        inputs[i].resize(4);
        for (int j = 0; j < 4; ++j) {
            inputs[i][j] = dist(gen);
        }
        
        // 簡單的分類規則
        float sum = inputs[i][0] + inputs[i][1] + inputs[i][2] + inputs[i][3];
        if (sum < -2.0f) {
            labels[i] = 0;
        } else if (sum > 2.0f) {
            labels[i] = 2;
        } else {
            labels[i] = 1;
        }
    }
}

// 計算準確率
float calculate_accuracy(const std::vector<std::vector<float>>& predictions,
                        const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        int predicted_class = 0;
        float max_prob = predictions[i][0];
        for (size_t j = 1; j < predictions[i].size(); ++j) {
            if (predictions[i][j] > max_prob) {
                max_prob = predictions[i][j];
                predicted_class = j;
            }
        }
        if (predicted_class == labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size() * 100.0f;
}

int main(int argc, char* argv[]) {
    std::cout << "=== GPU神經網路訓練演示 ===" << std::endl;
    
    // 解析命令行參數
    bool use_gpu = false;
    std::string backend_type = "auto";
    int gpu_device = 0;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend_type = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            gpu_device = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [選項]" << std::endl;
            std::cout << "選項:" << std::endl;
            std::cout << "  --gpu              使用GPU加速" << std::endl;
            std::cout << "  --backend TYPE     指定後端類型 (auto, cpu, cuda, metal)" << std::endl;
            std::cout << "  --device ID        指定GPU設備ID (默認: 0)" << std::endl;
            std::cout << "  --help, -h         顯示此幫助信息" << std::endl;
            return 0;
        }
    }
    
    try {
        // 配置模型參數
        SimpleNeuralModel::ModelParams params;
        params.input_size = 4;
        params.hidden_size = 16;
        params.output_size = 3;
        params.use_gpu = use_gpu;
        params.backend_type = backend_type;
        params.gpu_device = gpu_device;
        
        std::cout << "\n📋 模型配置:" << std::endl;
        std::cout << "  輸入維度: " << params.input_size << std::endl;
        std::cout << "  隱藏維度: " << params.hidden_size << std::endl;
        std::cout << "  輸出維度: " << params.output_size << std::endl;
        std::cout << "  使用GPU: " << (use_gpu ? "是" : "否") << std::endl;
        std::cout << "  後端類型: " << backend_type << std::endl;
        if (use_gpu) {
            std::cout << "  GPU設備: " << gpu_device << std::endl;
        }
        
        // 創建模型
        std::cout << "\n🚀 初始化模型..." << std::endl;
        auto start_init = std::chrono::high_resolution_clock::now();
        
        SimpleNeuralModel model(params);
        
        auto end_init = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
        
        std::cout << "✅ 模型初始化完成 (" << init_time.count() << "ms)" << std::endl;
        std::cout << "🖥️  使用後端: " << model.get_backend_name() << std::endl;
        std::cout << "💾 記憶體使用: " << model.get_memory_usage() / 1024.0 / 1024.0 << " MB" << std::endl;
        
        // 生成訓練數據
        std::cout << "\n📊 生成合成數據集..." << std::endl;
        std::vector<std::vector<float>> train_inputs, test_inputs;
        std::vector<int> train_labels, test_labels;
        
        generate_synthetic_data(train_inputs, train_labels, 800);
        generate_synthetic_data(test_inputs, test_labels, 200);
        
        std::cout << "  訓練樣本: " << train_inputs.size() << std::endl;
        std::cout << "  測試樣本: " << test_inputs.size() << std::endl;
        
        // 測試前向傳播性能
        std::cout << "\n⚡ 性能測試..." << std::endl;
        
        // 單樣本前向傳播
        auto start_single = std::chrono::high_resolution_clock::now();
        auto single_result = model.forward(train_inputs[0]);
        auto end_single = std::chrono::high_resolution_clock::now();
        auto single_time = std::chrono::duration_cast<std::chrono::microseconds>(end_single - start_single);
        
        std::cout << "  單樣本前向傳播: " << single_time.count() << " μs" << std::endl;
        
        // 批量前向傳播
        auto start_batch = std::chrono::high_resolution_clock::now();
        auto batch_results = model.forward_batch(test_inputs);
        auto end_batch = std::chrono::high_resolution_clock::now();
        auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch);
        
        std::cout << "  批量前向傳播 (" << test_inputs.size() << " 樣本): " << batch_time.count() << " ms" << std::endl;
        std::cout << "  平均每樣本: " << static_cast<float>(batch_time.count()) / test_inputs.size() << " ms" << std::endl;
        
        // 計算初始準確率
        float initial_accuracy = calculate_accuracy(batch_results, test_labels);
        std::cout << "\n📈 初始測試準確率: " << initial_accuracy << "%" << std::endl;
        
        // 計算損失
        float initial_loss = model.compute_loss(batch_results, test_labels);
        std::cout << "📉 初始測試損失: " << initial_loss << std::endl;
        
        // 演示訓練步驟（簡化版本）
        std::cout << "\n🎯 演示訓練步驟..." << std::endl;
        for (int epoch = 0; epoch < 5; ++epoch) {
            std::cout << "Epoch " << (epoch + 1) << "/5:" << std::endl;
            
            // 這裡只是演示，實際的反向傳播需要更複雜的實現
            model.backward_and_update(train_inputs, train_labels, 0.001f);
            
            // 測試當前性能
            auto current_predictions = model.forward_batch(test_inputs);
            float current_accuracy = calculate_accuracy(current_predictions, test_labels);
            float current_loss = model.compute_loss(current_predictions, test_labels);
            
            std::cout << "  準確率: " << current_accuracy << "%" << std::endl;
            std::cout << "  損失: " << current_loss << std::endl;
        }
        
        // 保存模型
        std::string model_filename = "gpu_neural_model_" + model.get_backend_name() + ".bin";
        std::cout << "\n💾 保存模型..." << std::endl;
        if (model.save_model(model_filename)) {
            std::cout << "✅ 模型已保存到: " << model_filename << std::endl;
        }
        
        // 測試模型加載
        std::cout << "\n🔄 測試模型加載..." << std::endl;
        SimpleNeuralModel loaded_model(params);
        if (loaded_model.load_model(model_filename)) {
            auto loaded_predictions = loaded_model.forward_batch(test_inputs);
            float loaded_accuracy = calculate_accuracy(loaded_predictions, test_labels);
            std::cout << "✅ 加載模型的測試準確率: " << loaded_accuracy << "%" << std::endl;
        }
        
        std::cout << "\n🎉 演示完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 錯誤: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 