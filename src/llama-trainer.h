#pragma once
#include "llama.h"
#include "llama-dataset.h"
#include "llama-forward.h"
#include "llama-loss.h"
#include "llama-backward.h"
#include "llama-optimizer.h"
#include <string>

class LlamaModel;
class LlamaDataset;
class LlamaOptimizer;
class LlamaLoss;

// 训练主控类
class LlamaTrainer {
public:
    LlamaTrainer(LlamaModel* model, LlamaDataset* dataset, LlamaOptimizer* optimizer, LlamaLoss* loss);

    // 加载训练数据
    bool load_data(const std::string &train_path, const std::string &valid_path);

    // 训练一个 epoch
    void train_epoch(int epoch, int batch_size);

    // 验证
    float validate();

    // 训练主循环
    void train(int epochs);

    // 保存模型
    void save_model(const std::string &path);

    void eval();

private:
    LlamaModel* model_;
    LlamaDataset* dataset_;
    LlamaOptimizer* optimizer_;
    LlamaLoss* loss_;
    // 你可以根据需要添加更多成员变量
}; 