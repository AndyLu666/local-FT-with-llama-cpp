#include "simple-neural-model.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

SimpleNeuralModel::SimpleNeuralModel(const ModelParams& params) 
    : params_(params), ctx_weights_(nullptr), ctx_compute_(nullptr), 
      backend_(nullptr), buffer_weights_(nullptr), allocr_(nullptr),
      w1_(nullptr), b1_(nullptr), w2_(nullptr), b2_(nullptr) {
    
    if (!init_backend()) {
        throw std::runtime_error("Failed to initialize backend");
    }
    
    if (!init_weights()) {
        throw std::runtime_error("Failed to initialize weights");
    }
    
    std::cout << "SimpleNeuralModel initialized with " << get_backend_name() 
              << " backend" << std::endl;
}

SimpleNeuralModel::~SimpleNeuralModel() {
    free_resources();
}

bool SimpleNeuralModel::init_backend() {
    // 初始化後端
    bool gpu_available = false;
    
    if (params_.use_gpu && params_.backend_type != "cpu") {
#ifdef GGML_USE_CUDA
        if (params_.backend_type == "auto" || params_.backend_type == "cuda") {
            std::cout << "嘗試初始化CUDA後端..." << std::endl;
            backend_ = ggml_backend_cuda_init(params_.gpu_device);
            if (backend_) {
                gpu_available = true;
                std::cout << "✅ CUDA後端初始化成功" << std::endl;
            } else {
                std::cout << "❌ CUDA後端初始化失敗" << std::endl;
            }
        }
#endif

#ifdef GGML_USE_METAL
        if (!gpu_available && (params_.backend_type == "auto" || params_.backend_type == "metal")) {
            std::cout << "嘗試初始化Metal後端..." << std::endl;
            backend_ = ggml_backend_metal_init();
            if (backend_) {
                gpu_available = true;
                std::cout << "✅ Metal後端初始化成功" << std::endl;
            } else {
                std::cout << "❌ Metal後端初始化失敗" << std::endl;
            }
        }
#endif
    }
    
    // 如果GPU不可用，回退到CPU
    if (!backend_) {
        std::cout << "使用CPU後端" << std::endl;
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            std::cerr << "❌ CPU後端初始化失敗" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool SimpleNeuralModel::init_weights() {
    // 創建權重上下文
    size_t ctx_size = 4 * ggml_tensor_overhead(); // 4個張量
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    ctx_weights_ = ggml_init(ctx_params);
    if (!ctx_weights_) {
        std::cerr << "❌ 創建權重上下文失敗" << std::endl;
        return false;
    }
    
    // 創建權重張量
    w1_ = ggml_new_tensor_2d(ctx_weights_, GGML_TYPE_F32, params_.input_size, params_.hidden_size);
    b1_ = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, params_.hidden_size);
    w2_ = ggml_new_tensor_2d(ctx_weights_, GGML_TYPE_F32, params_.hidden_size, params_.output_size);
    b2_ = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, params_.output_size);
    
    if (!w1_ || !b1_ || !w2_ || !b2_) {
        std::cerr << "❌ 創建權重張量失敗" << std::endl;
        return false;
    }
    
    // 設置張量名稱
    ggml_set_name(w1_, "w1");
    ggml_set_name(b1_, "b1");
    ggml_set_name(w2_, "w2");
    ggml_set_name(b2_, "b2");
    
    // 分配後端緩衝區
    buffer_weights_ = ggml_backend_alloc_ctx_tensors(ctx_weights_, backend_);
    if (!buffer_weights_) {
        std::cerr << "❌ 分配權重緩衝區失敗" << std::endl;
        return false;
    }
    
    // 初始化權重值
    xavier_init(w1_);
    zero_init(b1_);
    xavier_init(w2_);
    zero_init(b2_);
    
    // 創建內存分配器
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocr_) {
        std::cerr << "❌ 創建內存分配器失敗" << std::endl;
        return false;
    }
    
    return true;
}

void SimpleNeuralModel::xavier_init(struct ggml_tensor* tensor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    int fan_in = tensor->ne[0];
    int fan_out = tensor->ne[1];
    float std_dev = std::sqrt(2.0f / (fan_in + fan_out));
    
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);
    
    for (size_t i = 0; i < n_elements; ++i) {
        data[i] = dist(gen);
    }
    
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

void SimpleNeuralModel::zero_init(struct ggml_tensor* tensor) {
    size_t n_elements = ggml_nelements(tensor);
    std::vector<float> zeros(n_elements, 0.0f);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, ggml_nbytes(tensor));
}

std::vector<float> SimpleNeuralModel::forward(const std::vector<float>& input) {
    // 創建計算上下文
    size_t ctx_size = ggml_tensor_overhead() * 10; // 估計需要的張量數量
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    ctx_compute_ = ggml_init(ctx_params);
    if (!ctx_compute_) {
        throw std::runtime_error("創建計算上下文失敗");
    }
    
    // 創建輸入張量
    struct ggml_tensor* input_tensor = ggml_new_tensor_1d(ctx_compute_, GGML_TYPE_F32, params_.input_size);
    
    // 構建前向計算圖
    struct ggml_cgraph* gf = build_forward_graph(input_tensor);
    
    // 分配計算緩衝區
    ggml_gallocr_alloc_graph(allocr_, gf);
    
    // 設置輸入數據
    ggml_backend_tensor_set(input_tensor, input.data(), 0, ggml_nbytes(input_tensor));
    
    // 執行計算
    ggml_backend_graph_compute(backend_, gf);
    
    // 獲取結果
    struct ggml_tensor* output_tensor = ggml_graph_node(gf, -1); // 最後一個節點
    std::vector<float> result(params_.output_size);
    ggml_backend_tensor_get(output_tensor, result.data(), 0, ggml_nbytes(output_tensor));
    
    // 清理計算上下文
    ggml_free(ctx_compute_);
    ctx_compute_ = nullptr;
    
    return result;
}

struct ggml_cgraph* SimpleNeuralModel::build_forward_graph(struct ggml_tensor* input) {
    struct ggml_cgraph* gf = ggml_new_graph(ctx_compute_);
    
    // 第一層: input * w1 + b1
    struct ggml_tensor* hidden = ggml_mul_mat(ctx_compute_, w1_, input);
    hidden = ggml_add(ctx_compute_, hidden, b1_);
    
    // ReLU激活
    hidden = ggml_relu(ctx_compute_, hidden);
    
    // 第二層: hidden * w2 + b2
    struct ggml_tensor* output = ggml_mul_mat(ctx_compute_, w2_, hidden);
    output = ggml_add(ctx_compute_, output, b2_);
    
    // Softmax
    output = ggml_soft_max(ctx_compute_, output);
    
    ggml_build_forward_expand(gf, output);
    
    return gf;
}

std::vector<std::vector<float>> SimpleNeuralModel::forward_batch(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> results;
    results.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        results.push_back(forward(input));
    }
    
    return results;
}

float SimpleNeuralModel::compute_loss(const std::vector<std::vector<float>>& predictions, 
                                    const std::vector<int>& labels) {
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int label = labels[i];
        if (label >= 0 && label < static_cast<int>(predictions[i].size())) {
            // 交叉熵損失：-log(p_correct)
            float p_correct = std::max(predictions[i][label], 1e-10f);
            total_loss -= std::log(p_correct);
        }
    }
    
    return total_loss / predictions.size();
}

void SimpleNeuralModel::backward_and_update(const std::vector<std::vector<float>>& inputs,
                                           const std::vector<int>& labels,
                                           float learning_rate) {
    // 簡化的梯度下降實現
    // 在實際應用中，這裡應該使用GGML的自動微分功能
    
    std::cout << "執行反向傳播和參數更新，學習率: " << learning_rate << std::endl;
    
    // 這裡可以添加實際的梯度計算和參數更新邏輯
    // 由於GGML的自動微分API比較複雜，這裡先用簡化版本
}

bool SimpleNeuralModel::save_model(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ 無法打開文件進行保存: " << filename << std::endl;
        return false;
    }
    
    // 保存模型參數
    file.write(reinterpret_cast<const char*>(&params_), sizeof(params_));
    
    // 保存權重數據
    size_t w1_size = ggml_nbytes(w1_);
    size_t b1_size = ggml_nbytes(b1_);
    size_t w2_size = ggml_nbytes(w2_);
    size_t b2_size = ggml_nbytes(b2_);
    
    std::vector<float> w1_data(ggml_nelements(w1_));
    std::vector<float> b1_data(ggml_nelements(b1_));
    std::vector<float> w2_data(ggml_nelements(w2_));
    std::vector<float> b2_data(ggml_nelements(b2_));
    
    ggml_backend_tensor_get(w1_, w1_data.data(), 0, w1_size);
    ggml_backend_tensor_get(b1_, b1_data.data(), 0, b1_size);
    ggml_backend_tensor_get(w2_, w2_data.data(), 0, w2_size);
    ggml_backend_tensor_get(b2_, b2_data.data(), 0, b2_size);
    
    file.write(reinterpret_cast<const char*>(w1_data.data()), w1_size);
    file.write(reinterpret_cast<const char*>(b1_data.data()), b1_size);
    file.write(reinterpret_cast<const char*>(w2_data.data()), w2_size);
    file.write(reinterpret_cast<const char*>(b2_data.data()), b2_size);
    
    file.close();
    std::cout << "✅ 模型已保存到: " << filename << std::endl;
    return true;
}

bool SimpleNeuralModel::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ 無法打開文件進行加載: " << filename << std::endl;
        return false;
    }
    
    // 加載模型參數
    ModelParams loaded_params;
    file.read(reinterpret_cast<char*>(&loaded_params), sizeof(loaded_params));
    
    // 檢查參數兼容性
    if (loaded_params.input_size != params_.input_size ||
        loaded_params.hidden_size != params_.hidden_size ||
        loaded_params.output_size != params_.output_size) {
        std::cerr << "❌ 模型參數不匹配" << std::endl;
        return false;
    }
    
    // 加載權重數據
    size_t w1_size = ggml_nbytes(w1_);
    size_t b1_size = ggml_nbytes(b1_);
    size_t w2_size = ggml_nbytes(w2_);
    size_t b2_size = ggml_nbytes(b2_);
    
    std::vector<float> w1_data(ggml_nelements(w1_));
    std::vector<float> b1_data(ggml_nelements(b1_));
    std::vector<float> w2_data(ggml_nelements(w2_));
    std::vector<float> b2_data(ggml_nelements(b2_));
    
    file.read(reinterpret_cast<char*>(w1_data.data()), w1_size);
    file.read(reinterpret_cast<char*>(b1_data.data()), b1_size);
    file.read(reinterpret_cast<char*>(w2_data.data()), w2_size);
    file.read(reinterpret_cast<char*>(b2_data.data()), b2_size);
    
    ggml_backend_tensor_set(w1_, w1_data.data(), 0, w1_size);
    ggml_backend_tensor_set(b1_, b1_data.data(), 0, b1_size);
    ggml_backend_tensor_set(w2_, w2_data.data(), 0, w2_size);
    ggml_backend_tensor_set(b2_, b2_data.data(), 0, b2_size);
    
    file.close();
    std::cout << "✅ 模型已從文件加載: " << filename << std::endl;
    return true;
}

std::string SimpleNeuralModel::get_backend_name() const {
    if (!backend_) return "None";
    
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(backend_)) {
        return "CUDA";
    }
#endif

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(backend_)) {
        return "Metal";
    }
#endif
    
    return "CPU";
}

size_t SimpleNeuralModel::get_memory_usage() const {
    size_t total_size = 0;
    
    if (buffer_weights_) {
        total_size += ggml_backend_buffer_get_size(buffer_weights_);
    }
    
    if (allocr_) {
        total_size += ggml_gallocr_get_buffer_size(allocr_, 0);
    }
    
    return total_size;
}

void SimpleNeuralModel::free_resources() {
    if (allocr_) {
        ggml_gallocr_free(allocr_);
        allocr_ = nullptr;
    }
    
    if (buffer_weights_) {
        ggml_backend_buffer_free(buffer_weights_);
        buffer_weights_ = nullptr;
    }
    
    if (ctx_weights_) {
        ggml_free(ctx_weights_);
        ctx_weights_ = nullptr;
    }
    
    if (ctx_compute_) {
        ggml_free(ctx_compute_);
        ctx_compute_ = nullptr;
    }
    
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
} 