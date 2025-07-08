#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>

DataLoader::DataLoader(const GPT2Config& config) 
    : config_(config), current_position_(0) {
    setup_special_tokens();
}

void DataLoader::setup_special_tokens() {
    pad_token_id_ = 0;
    unk_token_id_ = 1;
    bos_token_id_ = 2;
    eos_token_id_ = 3;
}

bool DataLoader::initialize(const std::string& data_file, const std::string& vocab_file) {
    try {
        load_vocabulary(vocab_file);
        load_text_data(data_file);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "數據加載器初始化失敗: " << e.what() << std::endl;
        return false;
    }
}

void DataLoader::load_vocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        throw std::runtime_error("無法打開詞彙表文件: " + vocab_file);
    }
    
    vocab_.clear();
    token_to_id_.clear();
    id_to_token_.clear();
    
    std::string token;
    int id = 0;
    
    while (std::getline(file, token)) {
        if (!token.empty()) {
            vocab_.push_back(token);
            token_to_id_[token] = id;
            id_to_token_[id] = token;
            ++id;
        }
    }
    
    file.close();
    
    std::cout << "載入詞彙表: " << vocab_.size() << " 個詞彙" << std::endl;
}

void DataLoader::load_text_data(const std::string& data_file) {
    std::ifstream file(data_file);
    if (!file.is_open()) {
        throw std::runtime_error("無法打開數據文件: " + data_file);
    }
    
    tokenized_data_.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            auto tokens = tokenize(line);
            tokenized_data_.insert(tokenized_data_.end(), tokens.begin(), tokens.end());
            tokenized_data_.push_back(eos_token_id_);  // 添加行結束符
        }
    }
    
    file.close();
    current_position_ = 0;
    
    std::cout << "載入訓練數據: " << tokenized_data_.size() << " 個 tokens" << std::endl;
}

std::vector<int> DataLoader::tokenize(const std::string& text) {
    std::vector<int> token_ids;
    std::istringstream iss(text);
    std::string token;
    
    // 簡單的空格分詞
    while (iss >> token) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id_);  // 未知詞
        }
    }
    
    return token_ids;
}

std::string DataLoader::detokenize(const std::vector<int>& token_ids) const {
    std::string text;
    
    for (int id : token_ids) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            if (!text.empty()) text += " ";
            text += it->second;
        }
    }
    
    return text;
}

std::unique_ptr<Batch> DataLoader::get_next_batch() {
    if (!has_next_batch()) {
        return nullptr;
    }
    
    auto batch = std::make_unique<Batch>(config_.seq_length);
    
    // 複製序列數據
    for (int i = 0; i < config_.seq_length && current_position_ + i < tokenized_data_.size(); ++i) {
        batch->input_ids.push_back(tokenized_data_[current_position_ + i]);
        
        // 標籤是下一個 token（用於語言建模）
        if (current_position_ + i + 1 < tokenized_data_.size()) {
            batch->labels.push_back(tokenized_data_[current_position_ + i + 1]);
        } else {
            batch->labels.push_back(eos_token_id_);
        }
    }
    
    // 填充到指定長度
    while (batch->input_ids.size() < config_.seq_length) {
        batch->input_ids.push_back(pad_token_id_);
        batch->labels.push_back(pad_token_id_);
    }
    
    current_position_ += config_.seq_length;
    
    return batch;
}

bool DataLoader::has_next_batch() const {
    return current_position_ < tokenized_data_.size();
}

void DataLoader::reset() {
    current_position_ = 0;
}

const std::string& DataLoader::get_token(int id) const {
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    static const std::string unk = "<unk>";
    return unk;
}

int DataLoader::get_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return unk_token_id_;
}

void DataLoader::print_info() const {
    std::cout << "\n=== 數據加載器信息 ===" << std::endl;
    std::cout << "詞彙表大小: " << vocab_.size() << std::endl;
    std::cout << "訓練數據大小: " << tokenized_data_.size() << " tokens" << std::endl;
    std::cout << "序列長度: " << config_.seq_length << std::endl;
    std::cout << "批次大小: " << config_.batch_size << std::endl;
    std::cout << "當前位置: " << current_position_ << std::endl;
    
    std::cout << "\n特殊 tokens:" << std::endl;
    std::cout << "  PAD: " << pad_token_id_ << " (" << get_token(pad_token_id_) << ")" << std::endl;
    std::cout << "  UNK: " << unk_token_id_ << " (" << get_token(unk_token_id_) << ")" << std::endl;
    std::cout << "  BOS: " << bos_token_id_ << " (" << get_token(bos_token_id_) << ")" << std::endl;
    std::cout << "  EOS: " << eos_token_id_ << " (" << get_token(eos_token_id_) << ")" << std::endl;
}

void DataLoader::print_sample_data(int num_samples) const {
    std::cout << "\n=== 數據樣本 ===" << std::endl;
    
    for (int i = 0; i < num_samples && i * config_.seq_length < tokenized_data_.size(); ++i) {
        std::cout << "樣本 " << i + 1 << ":" << std::endl;
        
        std::vector<int> sample_ids;
        for (int j = 0; j < config_.seq_length && i * config_.seq_length + j < tokenized_data_.size(); ++j) {
            sample_ids.push_back(tokenized_data_[i * config_.seq_length + j]);
        }
        
        std::cout << "  Token IDs: ";
        for (int id : sample_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  文本: " << detokenize(sample_ids) << std::endl;
        std::cout << std::endl;
    }
} 