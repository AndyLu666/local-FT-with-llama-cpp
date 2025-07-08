#pragma once

#include "embeddings.h"
#include "transformer_block.h"
#include "layer_norm.h"
#include "config.h"
#include <memory>
#include <vector>
#include <string>

class GPT2Model {
private:
    std::unique_ptr<Embeddings> embeddings_;              // 嵌入層
    std::vector<std::unique_ptr<TransformerBlock>> h_;     // Transformer 塊
    std::unique_ptr<LayerNorm> ln_f_;                     // 最終層歸一化
    std::unique_ptr<Tensor> lm_head_;                     // 語言模型頭部權重
    
    GPT2Config config_;
    
    void initialize_lm_head();
    std::unique_ptr<Tensor> apply_lm_head(const Tensor& hidden_states);
    
public:
    explicit GPT2Model(const GPT2Config& config);
    ~GPT2Model() = default;
    
    // 前向傳播
    std::unique_ptr<Tensor> forward(const std::vector<int>& input_ids);
    
    // 文本生成
    std::vector<int> generate(const std::vector<int>& input_ids, 
                             int max_new_tokens = 20,
                             float temperature = 1.0f);
    
    // 採樣函數
    int sample_from_logits(const Tensor& logits, float temperature = 1.0f);
    
    // 調試信息
    void print_info() const;
    void print_architecture() const;
    
    // 獲取配置
    const GPT2Config& get_config() const { return config_; }
}; 