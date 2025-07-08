#pragma once

#include "model.h"
#include "data_loader.h"
#include "config.h"
#include <memory>
#include <string>

struct OptimizerState {
    std::vector<float> m_weights;  // 一階動量
    std::vector<float> v_weights;  // 二階動量
    int step_count;                // 步數計數
    
    OptimizerState() : step_count(0) {}
};

struct TrainingStats {
    float loss;
    float learning_rate;
    int step;
    int epoch;
    
    TrainingStats() : loss(0.0f), learning_rate(0.0f), step(0), epoch(0) {}
};

class Trainer {
private:
    std::shared_ptr<GPT2Model> model_;
    std::shared_ptr<DataLoader> dataloader_;
    std::unique_ptr<OptimizerState> optimizer_state_;
    
    GPT2Config config_;
    int current_step_;
    int current_epoch_;
    
    // 優化器相關
    float compute_loss(const Tensor& logits, const std::vector<int>& labels);
    void compute_gradients(const Tensor& logits, const std::vector<int>& labels);
    void adam_step();
    float get_learning_rate(int step) const;
    
    // 模擬函數（由於沒有真實的反向傳播）
    void simulate_gradient_computation();
    void simulate_parameter_update();
    
public:
    Trainer(std::shared_ptr<GPT2Model> model, 
           std::shared_ptr<DataLoader> dataloader,
           const GPT2Config& config);
    ~Trainer() = default;
    
    // 訓練方法
    void train(int num_epochs = 1);
    TrainingStats train_step(const Batch& batch);
    
    // 評估方法
    float evaluate();
    
    // 檢查點管理
    bool save_checkpoint(const std::string& filepath);
    bool load_checkpoint(const std::string& filepath);
    
    // 調試和監控
    void print_training_info() const;
    void log_step(const TrainingStats& stats);
    
    // 獲取當前狀態
    int get_current_step() const { return current_step_; }
    int get_current_epoch() const { return current_epoch_; }
}; 