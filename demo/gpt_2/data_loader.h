#pragma once

#include "config.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

struct Batch {
    std::vector<int> input_ids;      // 輸入 token IDs
    std::vector<int> labels;         // 標籤（用於計算損失）
    int seq_length;                  // 序列長度
    
    Batch(int seq_len) : seq_length(seq_len) {
        input_ids.reserve(seq_len);
        labels.reserve(seq_len);
    }
};

class DataLoader {
private:
    std::vector<int> tokenized_data_;              // 分詞後的數據
    std::vector<std::string> vocab_;               // 詞彙表
    std::unordered_map<std::string, int> token_to_id_;  // 詞到ID映射
    std::unordered_map<int, std::string> id_to_token_;  // ID到詞映射
    
    GPT2Config config_;
    size_t current_position_;                      // 當前數據位置
    
    // 特殊 token IDs
    int pad_token_id_;
    int unk_token_id_;
    int bos_token_id_;
    int eos_token_id_;
    
    void load_vocabulary(const std::string& vocab_file);
    void load_text_data(const std::string& data_file);
    void setup_special_tokens();
    
public:
    explicit DataLoader(const GPT2Config& config);
    ~DataLoader() = default;
    
    // 初始化方法
    bool initialize(const std::string& data_file, const std::string& vocab_file);
    
    // 數據生成
    std::unique_ptr<Batch> get_next_batch();
    bool has_next_batch() const;
    void reset();
    
    // 分詞和編碼
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& token_ids) const;
    
    // 詞彙表信息
    int get_vocab_size() const { return vocab_.size(); }
    const std::string& get_token(int id) const;
    int get_token_id(const std::string& token) const;
    
    // 特殊 token 獲取
    int pad_token() const { return pad_token_id_; }
    int unk_token() const { return unk_token_id_; }
    int bos_token() const { return bos_token_id_; }
    int eos_token() const { return eos_token_id_; }
    
    // 調試信息
    void print_info() const;
    void print_sample_data(int num_samples = 5) const;
}; 