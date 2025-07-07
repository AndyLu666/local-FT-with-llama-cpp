#pragma once
#include <vector>
#include <string>

// 假設每個 batch 包含輸入和標籤
struct LlamaBatch {
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> target;
};

// 简单的数据集类，负责加载和批量提供数据
class LlamaDataset {
public:
    LlamaDataset(const std::string &data_path);

    // 获取总样本数
    size_t size() const;

    // 返回所有 batch 的簡單接口，後續可優化為迭代器
    std::vector<LlamaBatch> batches(size_t batch_size) const;

private:
    std::vector<LlamaBatch> data_;
    void load_data(const std::string &data_path);
}; 