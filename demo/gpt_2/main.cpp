#include "config.h"
#include "model.h"
#include "data_loader.h"
#include "trainer.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>

void print_banner() {
    std::cout << "======================================" << std::endl;
    std::cout << "       模組化 GPT-2 演示程序        " << std::endl;
    std::cout << "      基於 ARCHITECTURE.md 設計      " << std::endl;
    std::cout << "======================================" << std::endl;
}

void demonstrate_architecture(const GPT2Config& config) {
    std::cout << "\n=== 架構演示 ===" << std::endl;
    
    // 創建模型並顯示架構信息
    auto model = std::make_shared<GPT2Model>(config);
    model->print_info();
    model->print_architecture();
}

void demonstrate_data_loading(const GPT2Config& config) {
    std::cout << "\n=== 數據加載演示 ===" << std::endl;
    
    auto dataloader = std::make_shared<DataLoader>(config);
    
    if (dataloader->initialize("training_data.txt", "vocab.txt")) {
        dataloader->print_info();
        dataloader->print_sample_data(3);
    } else {
        std::cerr << "數據加載失敗！" << std::endl;
    }
}

void demonstrate_training(const GPT2Config& config) {
    std::cout << "\n=== 訓練演示 ===" << std::endl;
    
    // 創建模型和數據加載器
    auto model = std::make_shared<GPT2Model>(config);
    auto dataloader = std::make_shared<DataLoader>(config);
    
    if (!dataloader->initialize("training_data.txt", "vocab.txt")) {
        std::cerr << "數據加載失敗，跳過訓練演示" << std::endl;
        return;
    }
    
    // 創建訓練器
    auto trainer = std::make_unique<Trainer>(model, dataloader, config);
    
    // 開始訓練（只訓練1個epoch作為演示）
    trainer->train(1);
    
    std::cout << "訓練演示完成" << std::endl;
}

void demonstrate_generation(const GPT2Config& config) {
    std::cout << "\n=== 文本生成演示 ===" << std::endl;
    
    // 創建模型和數據加載器
    auto model = std::make_shared<GPT2Model>(config);
    auto dataloader = std::make_shared<DataLoader>(config);
    
    if (!dataloader->initialize("training_data.txt", "vocab.txt")) {
        std::cerr << "數據加載失敗，跳過生成演示" << std::endl;
        return;
    }
    
    // 準備輸入序列
    std::vector<std::string> prompts = {
        "古寺",
        "山中 寺廟",
        "禪師 說法"
    };
    
    for (const auto& prompt : prompts) {
        std::cout << "\n提示: \"" << prompt << "\"" << std::endl;
        
        // 分詞
        auto input_ids = dataloader->tokenize(prompt);
        
        if (input_ids.empty()) {
            std::cout << "無法分詞，跳過" << std::endl;
            continue;
        }
        
        // 生成文本
        auto generated_ids = model->generate(input_ids, 10, 1.0f);
        
        // 轉換回文本
        std::string generated_text = dataloader->detokenize(generated_ids);
        std::cout << "生成: \"" << generated_text << "\"" << std::endl;
    }
}

void demonstrate_module_details() {
    std::cout << "\n=== 模組詳細演示 ===" << std::endl;
    
    GPT2Config config = GPT2Config::gpt2_small();
    
    // 演示各個模組
    std::cout << "\n1. 配置模組:" << std::endl;
    std::cout << "  使用 GPT-2 Small 配置" << std::endl;
    std::cout << "  嵌入維度: " << config.n_embd << std::endl;
    std::cout << "  注意力頭數: " << config.n_head << std::endl;
    std::cout << "  每頭維度: " << config.head_dim() << std::endl;
    std::cout << "  中間層維度: " << config.intermediate_size() << std::endl;
    
    std::cout << "\n2. 張量操作演示:" << std::endl;
    Tensor test_tensor({3, 4});
    test_tensor.at({0, 0}) = 1.0f;
    test_tensor.at({1, 2}) = 2.5f;
    test_tensor.at({2, 3}) = 3.7f;
    std::cout << "  創建 3x4 張量，設置部分值" << std::endl;
    std::cout << "  張量[0,0] = " << test_tensor.at({0, 0}) << std::endl;
    std::cout << "  張量[1,2] = " << test_tensor.at({1, 2}) << std::endl;
    std::cout << "  張量[2,3] = " << test_tensor.at({2, 3}) << std::endl;
    
    std::cout << "\n3. 嵌入層演示:" << std::endl;
    Embeddings embeddings(config);
    embeddings.print_info();
    
    std::cout << "\n4. 層歸一化演示:" << std::endl;
    LayerNorm layer_norm(config.n_embd);
    layer_norm.print_info();
}

int main() {
    try {
        print_banner();
        
        // 使用較小的配置進行演示
        GPT2Config config;
        config.vocab_size = 100;        // 較小的詞彙表
        config.n_embd = 64;             // 較小的嵌入維度
        config.n_head = 4;              // 較少的注意力頭
        config.n_layer = 3;             // 較少的層數
        config.n_ctx = 128;             // 較短的上下文
        config.seq_length = 32;         // 較短的序列長度
        config.batch_size = 2;          // 較小的批次
        
        std::cout << "\n使用簡化配置進行演示:" << std::endl;
        std::cout << "  詞彙表大小: " << config.vocab_size << std::endl;
        std::cout << "  嵌入維度: " << config.n_embd << std::endl;
        std::cout << "  Transformer 層數: " << config.n_layer << std::endl;
        
        // 1. 展示架構
        demonstrate_architecture(config);
        
        // 2. 展示模組詳情
        demonstrate_module_details();
        
        // 3. 展示數據加載
        demonstrate_data_loading(config);
        
        // 4. 展示訓練過程
        demonstrate_training(config);
        
        // 5. 展示文本生成
        demonstrate_generation(config);
        
        std::cout << "\n=== 演示完成 ===" << std::endl;
        std::cout << "所有模組均按照 ARCHITECTURE.md 設計實現" << std::endl;
        std::cout << "包含完整的 Transformer 架構和訓練流程" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "程序執行出錯: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 