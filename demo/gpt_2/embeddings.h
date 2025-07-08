#pragma once

#include "config.h"
#include <vector>
#include <memory>

// 簡化的張量結構
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;
    
    Tensor(const std::vector<int>& s) : shape(s) {
        int size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0f);
    }
    
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    int size() const;
};

class Embeddings {
private:
    std::unique_ptr<Tensor> wte_;  // 詞嵌入權重 [vocab_size, n_embd]
    std::unique_ptr<Tensor> wpe_;  // 位置嵌入權重 [n_positions, n_embd]
    GPT2Config config_;
    
    void initialize_weights();
    
public:
    explicit Embeddings(const GPT2Config& config);
    ~Embeddings() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const std::vector<int>& input_ids, 
                                   const std::vector<int>& position_ids = {});
    
    // 獲取權重（用於語言模型頭）
    const Tensor& get_word_embeddings() const { return *wte_; }
    
    // 調試信息
    void print_info() const;
}; 