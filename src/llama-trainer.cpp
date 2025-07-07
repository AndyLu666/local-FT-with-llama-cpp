#include "llama-trainer.h"
#include <iostream>

LlamaTrainer::LlamaTrainer(LlamaModel* model, LlamaDataset* dataset, LlamaOptimizer* optimizer, LlamaLoss* loss)
    : model_(model), dataset_(dataset), optimizer_(optimizer), loss_(loss) {}

void LlamaTrainer::train(int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // TODO: 遍歷數據集的 batch
        // for (auto& batch : dataset_->batches()) {
        //     auto output = model_->forward(batch.input);
        //     auto loss_val = loss_->compute(output, batch.target);
        //     model_->zero_grad();
        //     loss_val.backward();
        //     optimizer_->step();
        // }
    }
}

void LlamaTrainer::eval() {
    // TODO: 評估模式
}

void LlamaTrainer::save_model(const std::string &path) {
    // TODO: 实现模型保存
    std::cout << "保存模型到: " << path << std::endl;
} 