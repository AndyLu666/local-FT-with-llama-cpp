#include "embeddings.h"
#include <random>
#include <iostream>
#include <cassert>

// Tensor 方法實現
float& Tensor::at(const std::vector<int>& indices) {
    assert(indices.size() == shape.size());
    int idx = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data[idx];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    assert(indices.size() == shape.size());
    int idx = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data[idx];
}

int Tensor::size() const {
    int size = 1;
    for (int dim : shape) size *= dim;
    return size;
}

// Embeddings 實現
Embeddings::Embeddings(const GPT2Config& config) : config_(config) {
    wte_ = std::make_unique<Tensor>(std::vector<int>{config_.vocab_size, config_.n_embd});
    wpe_ = std::make_unique<Tensor>(std::vector<int>{config_.n_positions, config_.n_embd});
    initialize_weights();
}

void Embeddings::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    // 初始化詞嵌入權重
    for (float& val : wte_->data) {
        val = dist(gen);
    }
    
    // 初始化位置嵌入權重
    for (float& val : wpe_->data) {
        val = dist(gen);
    }
}

std::unique_ptr<Tensor> Embeddings::forward(const std::vector<int>& input_ids, 
                                           const std::vector<int>& position_ids) {
    int seq_len = input_ids.size();
    auto output = std::make_unique<Tensor>(std::vector<int>{seq_len, config_.n_embd});
    
    // 生成位置 IDs（如果未提供）
    std::vector<int> pos_ids = position_ids;
    if (pos_ids.empty()) {
        pos_ids.resize(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            pos_ids[i] = i;
        }
    }
    
    // 計算詞嵌入 + 位置嵌入
    for (int i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];
        int pos_id = pos_ids[i];
        
        for (int j = 0; j < config_.n_embd; ++j) {
            float word_emb = wte_->at({token_id, j});
            float pos_emb = wpe_->at({pos_id, j});
            output->at({i, j}) = word_emb + pos_emb;
        }
    }
    
    return output;
}

void Embeddings::print_info() const {
    std::cout << "Embeddings Info:" << std::endl;
    std::cout << "  詞嵌入維度: " << config_.vocab_size << " x " << config_.n_embd << std::endl;
    std::cout << "  位置嵌入維度: " << config_.n_positions << " x " << config_.n_embd << std::endl;
    std::cout << "  詞嵌入權重範圍: " << wte_->data.front() << " ~ " << wte_->data.back() << std::endl;
} 