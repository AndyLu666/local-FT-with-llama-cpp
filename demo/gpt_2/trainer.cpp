#include "trainer.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

Trainer::Trainer(std::shared_ptr<GPT2Model> model, 
                std::shared_ptr<DataLoader> dataloader,
                const GPT2Config& config)
    : model_(model), dataloader_(dataloader), config_(config),
      current_step_(0), current_epoch_(0) {
    
    optimizer_state_ = std::make_unique<OptimizerState>();
    
    // 初始化優化器狀態（模擬）
    // 在真實實現中，這裡會為所有模型參數初始化動量
    size_t total_params = config_.vocab_size * config_.n_embd * 2; // 簡化估計
    optimizer_state_->m_weights.resize(total_params, 0.0f);
    optimizer_state_->v_weights.resize(total_params, 0.0f);
}

void Trainer::train(int num_epochs) {
    std::cout << "\n=== 開始訓練 ===" << std::endl;
    print_training_info();
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        current_epoch_ = epoch;
        dataloader_->reset();
        
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;
        
        float epoch_loss = 0.0f;
        int batch_count = 0;
        
        while (dataloader_->has_next_batch()) {
            auto batch = dataloader_->get_next_batch();
            if (!batch) break;
            
            TrainingStats stats = train_step(*batch);
            epoch_loss += stats.loss;
            batch_count++;
            
            // 每10步記錄一次
            if (current_step_ % 10 == 0) {
                log_step(stats);
            }
        }
        
        float avg_loss = epoch_loss / batch_count;
        std::cout << "Epoch " << epoch + 1 << " 平均損失: " << avg_loss << std::endl;
        
        // 保存檢查點
        if ((epoch + 1) % 5 == 0) {
            std::string checkpoint_path = "checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
            save_checkpoint(checkpoint_path);
        }
    }
    
    std::cout << "\n訓練完成！" << std::endl;
}

TrainingStats Trainer::train_step(const Batch& batch) {
    TrainingStats stats;
    stats.step = ++current_step_;
    stats.epoch = current_epoch_;
    stats.learning_rate = get_learning_rate(current_step_);
    
    // 1. 前向傳播
    auto logits = model_->forward(batch.input_ids);
    
    // 2. 計算損失
    stats.loss = compute_loss(*logits, batch.labels);
    
    // 3. 模擬梯度計算
    simulate_gradient_computation();
    
    // 4. 模擬參數更新
    simulate_parameter_update();
    
    return stats;
}

float Trainer::compute_loss(const Tensor& logits, const std::vector<int>& labels) {
    // 計算交叉熵損失（簡化版本）
    float total_loss = 0.0f;
    int valid_tokens = 0;
    
    int seq_len = logits.shape[0];
    int vocab_size = logits.shape[1];
    
    for (int i = 0; i < seq_len && i < labels.size(); ++i) {
        int label = labels[i];
        
        // 跳過填充 token
        if (label == dataloader_->pad_token()) {
            continue;
        }
        
        // 計算 softmax
        float max_logit = -1e9f;
        for (int j = 0; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits.at({i, j}));
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; ++j) {
            sum_exp += std::exp(logits.at({i, j}) - max_logit);
        }
        
        float log_softmax = logits.at({i, label}) - max_logit - std::log(sum_exp);
        total_loss -= log_softmax;
        valid_tokens++;
    }
    
    return valid_tokens > 0 ? total_loss / valid_tokens : 0.0f;
}

void Trainer::simulate_gradient_computation() {
    // 這裡模擬梯度計算過程
    // 在真實實現中，會通過反向傳播計算所有參數的梯度
    std::cout << "  計算梯度中..." << std::endl;
}

void Trainer::simulate_parameter_update() {
    // 模擬 Adam 優化器更新參數
    optimizer_state_->step_count++;
    
    // 模擬參數更新（在真實實現中會更新所有模型參數）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.001f);
    
    // 添加一些隨機噪聲來模擬參數變化
    for (int i = 0; i < std::min(100, (int)optimizer_state_->m_weights.size()); ++i) {
        optimizer_state_->m_weights[i] += noise(gen);
        optimizer_state_->v_weights[i] += noise(gen) * noise(gen);
    }
    
    std::cout << "  更新參數中..." << std::endl;
}

float Trainer::get_learning_rate(int step) const {
    // 簡單的學習率衰減
    float base_lr = config_.learning_rate;
    float decay_factor = 0.95f;
    int decay_steps = 100;
    
    int decay_count = step / decay_steps;
    return base_lr * std::pow(decay_factor, decay_count);
}

float Trainer::evaluate() {
    std::cout << "評估模型中..." << std::endl;
    
    float total_loss = 0.0f;
    int batch_count = 0;
    
    dataloader_->reset();
    
    while (dataloader_->has_next_batch() && batch_count < 10) { // 只評估幾個批次
        auto batch = dataloader_->get_next_batch();
        if (!batch) break;
        
        auto logits = model_->forward(batch->input_ids);
        float loss = compute_loss(*logits, batch->labels);
        total_loss += loss;
        batch_count++;
    }
    
    return batch_count > 0 ? total_loss / batch_count : 0.0f;
}

bool Trainer::save_checkpoint(const std::string& filepath) {
    std::cout << "保存檢查點到: " << filepath << std::endl;
    
    // 模擬保存檢查點
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "無法創建檢查點文件: " << filepath << std::endl;
        return false;
    }
    
    // 寫入基本信息
    file.write(reinterpret_cast<const char*>(&current_step_), sizeof(current_step_));
    file.write(reinterpret_cast<const char*>(&current_epoch_), sizeof(current_epoch_));
    file.write(reinterpret_cast<const char*>(&optimizer_state_->step_count), sizeof(optimizer_state_->step_count));
    
    file.close();
    std::cout << "檢查點保存成功" << std::endl;
    return true;
}

bool Trainer::load_checkpoint(const std::string& filepath) {
    std::cout << "載入檢查點從: " << filepath << std::endl;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "無法打開檢查點文件: " << filepath << std::endl;
        return false;
    }
    
    // 讀取基本信息
    file.read(reinterpret_cast<char*>(&current_step_), sizeof(current_step_));
    file.read(reinterpret_cast<char*>(&current_epoch_), sizeof(current_epoch_));
    file.read(reinterpret_cast<char*>(&optimizer_state_->step_count), sizeof(optimizer_state_->step_count));
    
    file.close();
    std::cout << "檢查點載入成功" << std::endl;
    return true;
}

void Trainer::print_training_info() const {
    std::cout << "訓練配置:" << std::endl;
    std::cout << "  學習率: " << config_.learning_rate << std::endl;
    std::cout << "  批次大小: " << config_.batch_size << std::endl;
    std::cout << "  序列長度: " << config_.seq_length << std::endl;
    std::cout << "  權重衰減: " << config_.weight_decay << std::endl;
    std::cout << "  Adam Beta1: " << config_.beta1 << std::endl;
    std::cout << "  Adam Beta2: " << config_.beta2 << std::endl;
}

void Trainer::log_step(const TrainingStats& stats) {
    std::cout << "步驟 " << stats.step 
              << " | Epoch " << stats.epoch 
              << " | 損失: " << stats.loss 
              << " | LR: " << stats.learning_rate << std::endl;
} 