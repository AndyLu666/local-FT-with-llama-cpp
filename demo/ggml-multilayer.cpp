#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <cstring>

// 使用ggml的多层神经网络
class GGMLMultiLayerNet {
private:
    struct ggml_context* ctx_;
    
    // 网络层参数
    struct ggml_tensor* W1_;  // 第一层权重 [hidden_size, input_size]
    struct ggml_tensor* b1_;  // 第一层偏置 [hidden_size]
    struct ggml_tensor* W2_;  // 第二层权重 [output_size, hidden_size]  
    struct ggml_tensor* b2_;  // 第二层偏置 [output_size]
    
    int input_size_;
    int hidden_size_;
    int output_size_;
    float learning_rate_;
    
public:
    GGMLMultiLayerNet(int input_size, int hidden_size, int output_size, float learning_rate = 0.01f)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), learning_rate_(learning_rate) {
        
        // 初始化ggml上下文
        struct ggml_init_params params = {
            .mem_size = 128 * 1024 * 1024,  // 128MB内存
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        ctx_ = ggml_init(params);
        if (!ctx_) {
            throw std::runtime_error("无法初始化ggml上下文");
        }
        
        // 创建网络层张量
        W1_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, input_size_, hidden_size_);
        b1_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, hidden_size_);
        W2_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hidden_size_, output_size_);
        b2_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, output_size_);
        
        // 设置为可训练参数
        ggml_set_param(W1_);
        ggml_set_param(b1_);
        ggml_set_param(W2_);
        ggml_set_param(b2_);
        
        // 初始化权重
        init_weights();
        
        std::cout << "🚀 GGML多层神经网络已创建！" << std::endl;
        std::cout << "📊 网络架构：" << input_size_ << " → " << hidden_size_ << " → " << output_size_ << std::endl;
        std::cout << "⚙️  激活函数：ReLU → Softmax" << std::endl;
        std::cout << "🎯 可训练参数：" << get_param_count() << " 个" << std::endl;
    }
    
    ~GGMLMultiLayerNet() {
        if (ctx_) {
            ggml_free(ctx_);
        }
    }
    
    void init_weights() {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        // 初始化第一层权重和偏置
        float* w1_data = ggml_get_data_f32(W1_);
        for (int i = 0; i < ggml_nelements(W1_); ++i) {
            w1_data[i] = dist(rng);
        }
        
        float* b1_data = ggml_get_data_f32(b1_);
        for (int i = 0; i < ggml_nelements(b1_); ++i) {
            b1_data[i] = 0.0f;
        }
        
        // 初始化第二层权重和偏置
        float* w2_data = ggml_get_data_f32(W2_);
        for (int i = 0; i < ggml_nelements(W2_); ++i) {
            w2_data[i] = dist(rng);
        }
        
        float* b2_data = ggml_get_data_f32(b2_);
        for (int i = 0; i < ggml_nelements(b2_); ++i) {
            b2_data[i] = 0.0f;
        }
        
        std::cout << "🎲 权重已随机初始化" << std::endl;
    }
    
    // 多层前向传播：使用ggml内置函数链式构建
    struct ggml_tensor* forward(struct ggml_tensor* input) {
        // 第一层：线性变换 + ReLU激活
        struct ggml_tensor* z1 = ggml_mul_mat(ctx_, W1_, input);  // W1 * input
        z1 = ggml_add(ctx_, z1, b1_);                             // + b1
        struct ggml_tensor* a1 = ggml_relu(ctx_, z1);             // ReLU激活
        
        // 第二层：线性变换 + Softmax激活  
        struct ggml_tensor* z2 = ggml_mul_mat(ctx_, W2_, a1);     // W2 * a1
        z2 = ggml_add(ctx_, z2, b2_);                             // + b2
        struct ggml_tensor* output = ggml_soft_max(ctx_, z2);     // Softmax激活
        
        return output;
    }
    
    // 计算交叉熵损失：使用ggml内置损失函数
    struct ggml_tensor* compute_loss(struct ggml_tensor* logits, struct ggml_tensor* targets) {
        // 使用ggml内置的交叉熵损失函数
        return ggml_cross_entropy_loss(ctx_, logits, targets);
    }
    
    // 训练步骤：展示ggml的自动微分能力
    void train_step(const std::vector<float>& input_data, const std::vector<float>& target_data) {
        // 创建输入张量
        struct ggml_tensor* input = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, input_size_);
        float* input_ptr = ggml_get_data_f32(input);
        std::memcpy(input_ptr, input_data.data(), input_size_ * sizeof(float));
        
        // 创建目标张量
        struct ggml_tensor* target = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, output_size_);
        float* target_ptr = ggml_get_data_f32(target);
        std::memcpy(target_ptr, target_data.data(), output_size_ * sizeof(float));
        
        // 前向传播
        struct ggml_tensor* prediction = forward(input);
        
        // 计算损失
        struct ggml_tensor* loss = compute_loss(prediction, target);
        ggml_set_loss(loss);
        
        // 创建计算图（支持自动微分）
        struct ggml_cgraph* graph = ggml_new_graph_custom(ctx_, 1024, true);
        ggml_build_forward_expand(graph, loss);
        
        // 构建反向传播图（自动计算梯度！）
        ggml_build_backward_expand(ctx_, graph, nullptr);
        
        // 这里展示了ggml的计算图结构
        std::cout << "📈 计算图已构建，包含 " << ggml_graph_n_nodes(graph) << " 个节点" << std::endl;
        
        // 注意：完整的训练需要backend来执行计算和梯度更新
        // 这里主要展示ggml可以构建任意复杂的多层网络结构
    }
    
    int get_param_count() const {
        return ggml_nelements(W1_) + ggml_nelements(b1_) + 
               ggml_nelements(W2_) + ggml_nelements(b2_);
    }
    
    void print_architecture() {
        std::cout << "\n🏗️  网络架构详情：" << std::endl;
        std::cout << "   输入层: " << input_size_ << " 个神经元" << std::endl;
        std::cout << "   隐藏层: " << hidden_size_ << " 个神经元 (ReLU激活)" << std::endl;
        std::cout << "   输出层: " << output_size_ << " 个神经元 (Softmax激活)" << std::endl;
        std::cout << "   总参数: " << get_param_count() << " 个" << std::endl;
        
        std::cout << "\n⚡ GGML内置函数使用：" << std::endl;
        std::cout << "   ✅ ggml_mul_mat()   - 矩阵乘法" << std::endl;
        std::cout << "   ✅ ggml_add()       - 矩阵加法" << std::endl;
        std::cout << "   ✅ ggml_relu()      - ReLU激活函数" << std::endl;
        std::cout << "   ✅ ggml_soft_max()  - Softmax激活函数" << std::endl;
        std::cout << "   ✅ ggml_cross_entropy_loss() - 交叉熵损失函数" << std::endl;
        std::cout << "   ✅ ggml_build_backward_expand() - 自动微分" << std::endl;
    }
};

int main() {
    std::cout << "🧠 GGML多层神经网络演示\n" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // 创建一个4-16-3的多层网络
        GGMLMultiLayerNet network(4, 16, 3, 0.01f);
        network.print_architecture();
        
        // 演示训练步骤
        std::cout << "\n🎯 演示训练过程..." << std::endl;
        
        std::vector<float> input = {0.5f, -0.2f, 0.8f, -0.1f};
        std::vector<float> target = {0.0f, 1.0f, 0.0f};  // one-hot编码
        
        std::cout << "📝 输入数据: [" << input[0] << ", " << input[1] 
                  << ", " << input[2] << ", " << input[3] << "]" << std::endl;
        std::cout << "📝 目标输出: [" << target[0] << ", " << target[1] 
                  << ", " << target[2] << "]" << std::endl;
        
        network.train_step(input, target);
        
        std::cout << "\n💡 关键点解释：" << std::endl;
        std::cout << "   🔹 GGML完全支持多层神经网络！" << std::endl;
        std::cout << "   🔹 可以链式调用多个ggml函数构建复杂网络" << std::endl;
        std::cout << "   🔹 支持各种激活函数：ReLU, Sigmoid, Tanh, GELU等" << std::endl;
        std::cout << "   🔹 内置自动微分，无需手动计算梯度" << std::endl;
        std::cout << "   🔹 支持任意深度的网络结构" << std::endl;
        
        std::cout << "\n🤔 那为什么Enhanced-MLP不用ggml？" << std::endl;
        std::cout << "   答案：这只是演示选择！Enhanced-MLP主要展示" << std::endl;
        std::cout << "   软件架构设计，而不是性能优化。" << std::endl;
        std::cout << "   如果要追求性能，完全可以用ggml重写Enhanced-MLP！" << std::endl;
        
        std::cout << "\n✅ 演示完成！GGML的强大功能展示无遗！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 