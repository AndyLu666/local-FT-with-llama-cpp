#include "ggml/include/ggml.h"
#include "ggml/include/ggml-alloc.h"
#include "ggml/include/ggml-backend.h"
#include "ggml/src/ggml-cpu/ggml-cpu.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

class GGMLMultiLayerLogisticRegression {
private:
    // GGML上下文
    struct ggml_context* ctx_static;
    struct ggml_context* ctx_compute;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    
    // 网络结构
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;
    
    // 网络参数（张量）
    struct ggml_tensor* W1;  // 第一层权重 [input_size, hidden_size]
    struct ggml_tensor* b1;  // 第一层偏置 [hidden_size]
    struct ggml_tensor* W2;  // 第二层权重 [hidden_size, output_size]
    struct ggml_tensor* b2;  // 第二层偏置 [output_size]
    
    // 输入输出张量
    struct ggml_tensor* input;   // 输入 [batch_size, input_size]
    struct ggml_tensor* target;  // 目标 [batch_size, output_size]
    struct ggml_tensor* output;  // 网络输出
    struct ggml_tensor* loss;    // 损失
    
    // 计算图
    struct ggml_cgraph* forward_graph;
    struct ggml_cgraph* backward_graph;
    
public:
    GGMLMultiLayerLogisticRegression(int input_size, int hidden_size, int output_size, int batch_size = 32)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), batch_size(batch_size) {
        
        // 初始化CPU后端
        backend = ggml_backend_cpu_init();
        
        // 创建静态上下文（用于参数）
        {
            struct ggml_init_params params = {
                .mem_size = 16 * 1024 * 1024,  // 16MB
                .mem_buffer = nullptr,
                .no_alloc = true,
            };
            ctx_static = ggml_init(params);
        }
        
        // 创建计算上下文（用于前向传播）
        {
            struct ggml_init_params params = {
                .mem_size = 32 * 1024 * 1024,  // 32MB
                .mem_buffer = nullptr,
                .no_alloc = true,
            };
            ctx_compute = ggml_init(params);
        }
        
        // 创建参数张量
        createParameters();
        
        // 分配内存
        allocateMemory();
        
        // 初始化参数
        initializeParameters();
        
        // 构建计算图
        buildComputeGraph();
    }
    
    ~GGMLMultiLayerLogisticRegression() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backward_graph) ggml_free(backward_graph->ctx);
        if (ctx_compute) ggml_free(ctx_compute);
        if (ctx_static) ggml_free(ctx_static);
        if (backend) ggml_backend_free(backend);
    }
    
private:
    void createParameters() {
        // 第一层权重和偏置
        W1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, input_size, hidden_size);
        b1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, hidden_size);
        ggml_set_param(W1);  // 标记为可训练参数
        ggml_set_param(b1);
        ggml_set_name(W1, "W1");
        ggml_set_name(b1, "b1");
        
        // 第二层权重和偏置
        W2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, hidden_size, output_size);
        b2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, output_size);
        ggml_set_param(W2);  // 标记为可训练参数
        ggml_set_param(b2);
        ggml_set_name(W2, "W2");
        ggml_set_name(b2, "b2");
        
        // 输入和目标张量
        input = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, input_size, batch_size);
        target = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, output_size, batch_size);
        ggml_set_name(input, "input");
        ggml_set_name(target, "target");
    }
    
    void allocateMemory() {
        // 为所有张量分配内存
        buffer = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    }
    
    void initializeParameters() {
        // Xavier初始化权重
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // 初始化W1 (Xavier)
        float w1_scale = std::sqrt(2.0f / (input_size + hidden_size));
        std::normal_distribution<float> w1_dist(0.0f, w1_scale);
        std::vector<float> w1_data(input_size * hidden_size);
        for (auto& w : w1_data) w = w1_dist(gen);
        ggml_backend_tensor_set(W1, w1_data.data(), 0, ggml_nbytes(W1));
        
        // 初始化b1为零
        std::vector<float> b1_data(hidden_size, 0.0f);
        ggml_backend_tensor_set(b1, b1_data.data(), 0, ggml_nbytes(b1));
        
        // 初始化W2 (Xavier)
        float w2_scale = std::sqrt(2.0f / (hidden_size + output_size));
        std::normal_distribution<float> w2_dist(0.0f, w2_scale);
        std::vector<float> w2_data(hidden_size * output_size);
        for (auto& w : w2_data) w = w2_dist(gen);
        ggml_backend_tensor_set(W2, w2_data.data(), 0, ggml_nbytes(W2));
        
        // 初始化b2为零
        std::vector<float> b2_data(output_size, 0.0f);
        ggml_backend_tensor_set(b2, b2_data.data(), 0, ggml_nbytes(b2));
    }
    
    void buildComputeGraph() {
        // 前向传播：使用纯GGML内置函数
        // 第一层：z1 = input * W1 + b1
        struct ggml_tensor* z1 = ggml_add(ctx_compute,
            ggml_mul_mat(ctx_compute, W1, input), 
            ggml_repeat(ctx_compute, b1, ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, hidden_size, batch_size))
        );
        
        // ReLU激活：a1 = relu(z1)
        struct ggml_tensor* a1 = ggml_unary(ctx_compute, z1, GGML_UNARY_OP_RELU);
        
        // 第二层：z2 = a1 * W2 + b2
        struct ggml_tensor* z2 = ggml_add(ctx_compute,
            ggml_mul_mat(ctx_compute, W2, a1),
            ggml_repeat(ctx_compute, b2, ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, output_size, batch_size))
        );
        
        // Sigmoid激活（用于逻辑回归）：output = sigmoid(z2)
        output = ggml_unary(ctx_compute, z2, GGML_UNARY_OP_SIGMOID);
        ggml_set_name(output, "output");
        
        // 交叉熵损失：使用GGML内置的交叉熵损失函数
        loss = ggml_cross_entropy_loss(ctx_compute, output, target);
        ggml_set_name(loss, "loss");
        ggml_set_loss(loss);  // 标记为损失张量
        
        // 构建前向传播图
        forward_graph = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(forward_graph, loss);
        
        // 构建反向传播图（用于训练）
        backward_graph = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(backward_graph, loss);
        ggml_build_backward_expand(ctx_compute, backward_graph, nullptr);
    }
    
public:
    // 前向传播
    float forward(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y) {
        // 设置输入数据
        std::vector<float> input_data(input_size * batch_size);
        std::vector<float> target_data(output_size * batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_size; ++i) {
                input_data[b * input_size + i] = X[b][i];
            }
            for (int o = 0; o < output_size; ++o) {
                target_data[b * output_size + o] = y[b][o];
            }
        }
        
        ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(target, target_data.data(), 0, ggml_nbytes(target));
        
        // 执行前向传播
        ggml_backend_graph_compute(backend, forward_graph);
        
        // 获取损失值
        float loss_value;
        ggml_backend_tensor_get(loss, &loss_value, 0, sizeof(float));
        
        return loss_value;
    }
    
    // 训练一步（前向+反向传播）
    void trainStep(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y, float learning_rate = 0.01f) {
        // 前向传播
        forward(X, y);
        
        // 反向传播：GGML自动计算梯度
        ggml_backend_graph_compute(backend, backward_graph);
        
        // 手动更新参数（简单SGD）
        updateParameters(learning_rate);
    }
    
    // 预测
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& X) {
        // 创建虚拟目标（预测时不使用）
        std::vector<std::vector<float>> dummy_y(batch_size, std::vector<float>(output_size, 0.0f));
        
        // 前向传播
        forward(X, dummy_y);
        
        // 获取输出
        std::vector<float> output_data(output_size * batch_size);
        ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
        
        std::vector<std::vector<float>> predictions(batch_size, std::vector<float>(output_size));
        for (int b = 0; b < batch_size; ++b) {
            for (int o = 0; o < output_size; ++o) {
                predictions[b][o] = output_data[b * output_size + o];
            }
        }
        
        return predictions;
    }
    
private:
    void updateParameters(float learning_rate) {
        // 获取梯度并更新参数
        updateParameter(W1, learning_rate);
        updateParameter(b1, learning_rate);
        updateParameter(W2, learning_rate);
        updateParameter(b2, learning_rate);
    }
    
    void updateParameter(struct ggml_tensor* param, float learning_rate) {
        // 获取参数的梯度
        struct ggml_tensor* grad = ggml_graph_get_grad(backward_graph, param);
        if (!grad) return;
        
        size_t param_size = ggml_nelements(param);
        std::vector<float> param_data(param_size);
        std::vector<float> grad_data(param_size);
        
        // 获取当前参数值和梯度
        ggml_backend_tensor_get(param, param_data.data(), 0, ggml_nbytes(param));
        ggml_backend_tensor_get(grad, grad_data.data(), 0, ggml_nbytes(grad));
        
        // SGD更新：param = param - learning_rate * grad
        for (size_t i = 0; i < param_size; ++i) {
            param_data[i] -= learning_rate * grad_data[i];
        }
        
        // 写回更新后的参数
        ggml_backend_tensor_set(param, param_data.data(), 0, ggml_nbytes(param));
    }
};

// 生成示例数据（XOR问题）
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateXORData(int batch_size) {
    std::vector<std::vector<float>> X(batch_size, std::vector<float>(2));
    std::vector<std::vector<float>> y(batch_size, std::vector<float>(1));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 3);
    
    for (int i = 0; i < batch_size; ++i) {
        int pattern = dis(gen);
        switch (pattern) {
            case 0: X[i] = {0.0f, 0.0f}; y[i] = {0.0f}; break;
            case 1: X[i] = {0.0f, 1.0f}; y[i] = {1.0f}; break;
            case 2: X[i] = {1.0f, 0.0f}; y[i] = {1.0f}; break;
            case 3: X[i] = {1.0f, 1.0f}; y[i] = {0.0f}; break;
        }
    }
    
    return {X, y};
}

int main() {
    std::cout << "🚀 GGML多层逻辑回归演示\n";
    std::cout << "===============================\n\n";
    
    // 网络参数
    const int input_size = 2;
    const int hidden_size = 8;
    const int output_size = 1;
    const int batch_size = 4;
    const int epochs = 1000;
    const float learning_rate = 0.1f;
    
    std::cout << "网络结构: " << input_size << " → " << hidden_size << " → " << output_size << "\n";
    std::cout << "问题: XOR逻辑回归\n";
    std::cout << "批大小: " << batch_size << "\n";
    std::cout << "训练轮数: " << epochs << "\n\n";
    
    // 创建模型
    GGMLMultiLayerLogisticRegression model(input_size, hidden_size, output_size, batch_size);
    
    // 训练
    std::cout << "开始训练...\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto [X, y] = generateXORData(batch_size);
        
        model.trainStep(X, y, learning_rate);
        
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            float loss = model.forward(X, y);
            std::cout << "Epoch " << std::setw(4) << epoch << " | Loss: " << std::fixed << std::setprecision(6) << loss << "\n";
        }
    }
    
    // 测试
    std::cout << "\n测试结果:\n";
    std::cout << "输入 → 输出 (期望)\n";
    std::cout << "-------------------\n";
    
    std::vector<std::vector<float>> test_cases = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    };
    std::vector<float> expected = {0.0f, 1.0f, 1.0f, 0.0f};
    
    for (int i = 0; i < 4; ++i) {
        std::vector<std::vector<float>> X_test(batch_size, test_cases[i]);
        std::vector<std::vector<float>> y_dummy(batch_size, std::vector<float>(1, 0.0f));
        
        auto predictions = model.predict(X_test);
        float pred = predictions[0][0];
        
        std::cout << "(" << test_cases[i][0] << ", " << test_cases[i][1] << ") → " 
                  << std::fixed << std::setprecision(3) << pred 
                  << " (" << expected[i] << ")" 
                  << (std::abs(pred - expected[i]) < 0.5 ? " ✓" : " ✗") << "\n";
    }
    
    std::cout << "\n✅ 演示完成！这个多层逻辑回归完全使用GGML内置函数实现。\n";
    std::cout << "🔧 使用的GGML函数:\n";
    std::cout << "   - ggml_mul_mat() - 矩阵乘法\n";
    std::cout << "   - ggml_add() - 加法\n";
    std::cout << "   - ggml_unary(RELU/SIGMOID) - 激活函数\n";
    std::cout << "   - ggml_cross_entropy_loss() - 损失函数\n";
    std::cout << "   - ggml_build_backward_expand() - 自动微分\n";
    std::cout << "   - ggml_backend_graph_compute() - 图执行\n";
    
    return 0;
} 