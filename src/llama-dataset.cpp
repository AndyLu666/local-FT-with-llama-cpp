#include "llama-dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

LlamaDataset::LlamaDataset(const std::string &data_path) {
    load_data(data_path);
}

void LlamaDataset::load_data(const std::string &data_path) {
    std::ifstream fin(data_path);
    std::string line;
    while (std::getline(fin, line)) {
        auto pos = line.find('|');
        if (pos == std::string::npos) continue;
        std::vector<int> input, label;
        std::stringstream ss1(line.substr(0, pos)), ss2(line.substr(pos + 1));
        int token;
        while (ss1 >> token) input.push_back(token);
        while (ss2 >> token) label.push_back(token);
        data_inputs_.push_back(input);
        data_labels_.push_back(label);
    }
    std::cout << "数据加载完成，样本数: " << data_inputs_.size() << std::endl;
}

size_t LlamaDataset::size() const {
    return data_inputs_.size();
}

void LlamaDataset::get_batch(size_t start, size_t batch_size, std::vector<std::vector<int>> &inputs, std::vector<std::vector<int>> &labels) {
    inputs.clear();
    labels.clear();
    for (size_t i = start; i < start + batch_size && i < data_inputs_.size(); ++i) {
        inputs.push_back(data_inputs_[i]);
        labels.push_back(data_labels_[i]);
    }
}

std::vector<LlamaBatch> LlamaDataset::batches(size_t batch_size) const {
    std::vector<LlamaBatch> result;
    for (size_t i = 0; i < data_inputs_.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, data_inputs_.size());
        std::vector<std::vector<int>> batch_inputs(data_inputs_.begin() + i, data_inputs_.begin() + end);
        std::vector<std::vector<int>> batch_labels(data_labels_.begin() + i, data_labels_.begin() + end);
        result.push_back(LlamaBatch(batch_inputs, batch_labels));
    }
    return result;
} 