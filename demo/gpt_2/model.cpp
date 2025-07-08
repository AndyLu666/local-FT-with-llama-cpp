#include "model.h"
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

GPT2Model::GPT2Model(const GPT2Config& config) : config_(config) {
    // 初始化嵌入層
    embeddings_ = std::make_unique<Embeddings>(config_);
    
    // 初始化 Transformer 塊
    h_.reserve(config_.n_layer);
    for (int i = 0; i < config_.n_layer; ++i) {
        h_.push_back(std::make_unique<TransformerBlock>(config_));
    }
    
    // 初始化最終層歸一化
    ln_f_ = std::make_unique<LayerNorm>(config_.n_embd);
    
    // 初始化語言模型頭部
    initialize_lm_head();
}

void GPT2Model::initialize_lm_head() {
    // 語言模型頭部通常與詞嵌入權重共享
    // 這裡為了簡化，創建獨立的權重
    lm_head_ = std::make_unique<Tensor>(std::vector<int>{config_.n_embd, config_.vocab_size});
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    for (float& val : lm_head_->data) {
        val = dist(gen);
    }
}

std::unique_ptr<Tensor> GPT2Model::forward(const std::vector<int>& input_ids) {
    // 1. 嵌入層
    auto hidden_states = embeddings_->forward(input_ids);
    
    // 2. 通過所有 Transformer 塊
    for (auto& block : h_) {
        hidden_states = block->forward(*hidden_states);
    }
    
    // 3. 最終層歸一化
    auto ln_output = ln_f_->forward(*hidden_states);
    
    // 4. 語言模型頭部
    auto logits = apply_lm_head(*ln_output);
    
    return logits;
}

std::unique_ptr<Tensor> GPT2Model::apply_lm_head(const Tensor& hidden_states) {
    int seq_len = hidden_states.shape[0];
    int n_embd = hidden_states.shape[1];
    assert(n_embd == config_.n_embd);
    
    auto logits = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.vocab_size});
    
    // 矩陣乘法: [seq_len, n_embd] × [n_embd, vocab_size] = [seq_len, vocab_size]
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < config_.vocab_size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n_embd; ++k) {
                sum += hidden_states.at({i, k}) * lm_head_->at({k, j});
            }
            logits->at({i, j}) = sum;
        }
    }
    
    return logits;
}

std::vector<int> GPT2Model::generate(const std::vector<int>& input_ids, 
                                    int max_new_tokens,
                                    float temperature) {
    std::vector<int> generated = input_ids;
    
    for (int step = 0; step < max_new_tokens; ++step) {
        // 前向傳播
        auto logits = forward(generated);
        
        // 取最後一個時間步的 logits
        Tensor last_logits(std::vector<int>{config_.vocab_size});
        int last_idx = generated.size() - 1;
        for (int i = 0; i < config_.vocab_size; ++i) {
            last_logits.data[i] = logits->at({last_idx, i});
        }
        
        // 採樣下一個 token
        int next_token = sample_from_logits(last_logits, temperature);
        generated.push_back(next_token);
        
        // 限制序列長度
        if (generated.size() >= config_.n_ctx) {
            break;
        }
    }
    
    return generated;
}

int GPT2Model::sample_from_logits(const Tensor& logits, float temperature) {
    std::vector<float> probs(config_.vocab_size);
    
    // 應用溫度縮放並計算 softmax
    float max_logit = *std::max_element(logits.data.begin(), logits.data.end());
    float sum = 0.0f;
    
    for (int i = 0; i < config_.vocab_size; ++i) {
        float scaled_logit = (logits.data[i] - max_logit) / temperature;
        probs[i] = std::exp(scaled_logit);
        sum += probs[i];
    }
    
    // 歸一化
    for (float& prob : probs) {
        prob /= sum;
    }
    
    // 隨機採樣
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float random_val = dist(gen);
    float cumulative = 0.0f;
    
    for (int i = 0; i < config_.vocab_size; ++i) {
        cumulative += probs[i];
        if (random_val <= cumulative) {
            return i;
        }
    }
    
    return config_.vocab_size - 1;  // 回退
}

void GPT2Model::print_info() const {
    std::cout << "\n=== GPT-2 模型信息 ===" << std::endl;
    std::cout << "詞彙表大小: " << config_.vocab_size << std::endl;
    std::cout << "嵌入維度: " << config_.n_embd << std::endl;
    std::cout << "注意力頭數: " << config_.n_head << std::endl;
    std::cout << "Transformer 層數: " << config_.n_layer << std::endl;
    std::cout << "上下文長度: " << config_.n_ctx << std::endl;
    
    // 計算參數數量
    size_t total_params = 0;
    
    // 嵌入層參數
    total_params += config_.vocab_size * config_.n_embd;  // 詞嵌入
    total_params += config_.n_positions * config_.n_embd; // 位置嵌入
    
    // Transformer 塊參數
    for (int i = 0; i < config_.n_layer; ++i) {
        // 注意力層
        total_params += config_.n_embd * 3 * config_.n_embd; // QKV 權重
        total_params += 3 * config_.n_embd;                  // QKV 偏置
        total_params += config_.n_embd * config_.n_embd;     // 輸出投影權重
        total_params += config_.n_embd;                      // 輸出投影偏置
        
        // 前饋網絡
        int intermediate_size = config_.intermediate_size();
        total_params += config_.n_embd * intermediate_size;  // 第一層權重
        total_params += intermediate_size;                   // 第一層偏置
        total_params += intermediate_size * config_.n_embd;  // 第二層權重
        total_params += config_.n_embd;                      // 第二層偏置
        
        // 層歸一化（兩個）
        total_params += 2 * config_.n_embd;                 // 權重
        total_params += 2 * config_.n_embd;                 // 偏置
    }
    
    // 最終層歸一化
    total_params += config_.n_embd;  // 權重
    total_params += config_.n_embd;  // 偏置
    
    // 語言模型頭部
    total_params += config_.n_embd * config_.vocab_size;
    
    std::cout << "總參數數量: " << total_params << " (" << total_params / 1000000.0f << "M)" << std::endl;
}

void GPT2Model::print_architecture() const {
    std::cout << "\n=== GPT-2 架構詳情 ===" << std::endl;
    
    std::cout << "\n1. 嵌入層:" << std::endl;
    embeddings_->print_info();
    
    std::cout << "\n2. Transformer 塊 (共 " << config_.n_layer << " 層):" << std::endl;
    if (!h_.empty()) {
        h_[0]->print_info();
        if (config_.n_layer > 1) {
            std::cout << "  ... (其餘 " << config_.n_layer - 1 << " 層結構相同)" << std::endl;
        }
    }
    
    std::cout << "\n3. 最終層歸一化:" << std::endl;
    ln_f_->print_info();
    
    std::cout << "\n4. 語言模型頭部:" << std::endl;
    std::cout << "  權重維度: " << config_.n_embd << " x " << config_.vocab_size << std::endl;
} 