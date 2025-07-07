#pragma once
#include <vector>
#include <memory>
#include <string>

// 前向聲明ggml結構
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;

/**
 * 簡化的神經網絡模型類
 * 支持CPU和GPU後端
 */
class SimpleNeuralModel {
public:
    struct ModelParams {
        int input_size = 4;
        int hidden_size = 16;
        int output_size = 3;
        bool use_gpu = false;      // 新增：是否使用GPU
        int gpu_device = 0;        // 新增：GPU設備ID
        std::string backend_type = "auto"; // 新增：後端類型 ("auto", "cpu", "cuda", "metal")
    };

    SimpleNeuralModel(const ModelParams& params);
    ~SimpleNeuralModel();

    // 前向傳播
    std::vector<float> forward(const std::vector<float>& input);
    
    // 批量前向傳播
    std::vector<std::vector<float>> forward_batch(const std::vector<std::vector<float>>& inputs);
    
    // 計算損失
    float compute_loss(const std::vector<std::vector<float>>& predictions, 
                      const std::vector<int>& labels);
    
    // 反向傳播和參數更新
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
                           const std::vector<int>& labels,
                           float learning_rate = 0.001f);
    
    // 保存和加載模型
    bool save_model(const std::string& filename);
    bool load_model(const std::string& filename);
    
    // 獲取模型信息
    ModelParams get_params() const { return params_; }
    std::string get_backend_name() const;
    size_t get_memory_usage() const;

private:
    ModelParams params_;
    
    // GGML上下文和後端
    struct ggml_context* ctx_weights_;      // 權重上下文
    struct ggml_context* ctx_compute_;      // 計算上下文
    struct ggml_backend* backend_;          // 後端（CPU/GPU）
    struct ggml_backend_buffer* buffer_weights_; // 權重緩衝區
    struct ggml_gallocr* allocr_;          // 內存分配器
    
    // 權重張量
    struct ggml_tensor* w1_;    // 第一層權重
    struct ggml_tensor* b1_;    // 第一層偏置
    struct ggml_tensor* w2_;    // 第二層權重
    struct ggml_tensor* b2_;    // 第二層偏置
    
    // 內部方法
    bool init_backend();
    bool init_weights();
    void free_resources();
    
    // 構建計算圖
    struct ggml_cgraph* build_forward_graph(struct ggml_tensor* input);
    struct ggml_cgraph* build_backward_graph(struct ggml_tensor* input, 
                                           struct ggml_tensor* target);
    
    // 權重初始化
    void xavier_init(struct ggml_tensor* tensor);
    void zero_init(struct ggml_tensor* tensor);
}; 