#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <algorithm>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// 生成數據集
static void generate_dataset(
    float * x_data,
    float * y_data,
    int n_samples,
    int n_features,
    int n_classes) {
    
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    
    for (int i = 0; i < n_samples; ++i) {
        // 根據類別生成不同的模式
        int label = i % n_classes;
        
        // 計算基礎位置（不同類別在不同位置）
        float angle = 2.0f * M_PI * label / n_classes;
        float radius = 2.0f;
        
        // 生成特徵
        for (int j = 0; j < n_features; ++j) {
            if (j == 0) {
                x_data[i * n_features + j] = radius * cosf(angle) + noise(rng);
            } else if (j == 1) {
                x_data[i * n_features + j] = radius * sinf(angle) + noise(rng);
            } else {
                x_data[i * n_features + j] = noise(rng); // 其他特徵為噪聲
            }
        }
        
        // 設定 one-hot 編碼的標籤
        for (int j = 0; j < n_classes; ++j) {
            y_data[i * n_classes + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

int main() {
    // 設定參數
    const int n_samples = 1000;
    const int n_features = 4;
    const int n_hidden = 16;
    const int n_classes = 3;
    const int n_epochs = 500;
    const float learning_rate = 0.01f;
    const int batch_size = 32;
    
    // 初始化 ggml
    struct ggml_init_params params = {
        .mem_size   = 256*1024*1024, // 256 MB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }
    
    // 建立模型參數
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_features, n_hidden);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_hidden);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hidden, n_classes);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_classes);
    
    // 設定參數名稱
    ggml_set_name(w1, "w1");
    ggml_set_name(b1, "b1");
    ggml_set_name(w2, "w2");
    ggml_set_name(b2, "b2");
    
    // 標記為可訓練參數
    ggml_set_param(w1);
    ggml_set_param(b1);
    ggml_set_param(w2);
    ggml_set_param(b2);
    
    // 初始化權重
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        float * w1_data = ggml_get_data_f32(w1);
        float * b1_data = ggml_get_data_f32(b1);
        float * w2_data = ggml_get_data_f32(w2);
        float * b2_data = ggml_get_data_f32(b2);
        
        for (int i = 0; i < ggml_nelements(w1); ++i) {
            w1_data[i] = dist(rng);
        }
        for (int i = 0; i < ggml_nelements(b1); ++i) {
            b1_data[i] = 0.0f;
        }
        for (int i = 0; i < ggml_nelements(w2); ++i) {
            w2_data[i] = dist(rng);
        }
        for (int i = 0; i < ggml_nelements(b2); ++i) {
            b2_data[i] = 0.0f;
        }
    }
    
    // 建立輸入張量
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_features, batch_size);
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_classes, batch_size);
    
    ggml_set_name(x, "x");
    ggml_set_name(y, "y");
    ggml_set_input(x);
    ggml_set_input(y);
    
    // 建立前向傳播圖
    // 第一層：線性變換 + ReLU
    struct ggml_tensor * z1 = ggml_mul_mat(ctx, w1, x);
    z1 = ggml_add(ctx, z1, ggml_repeat(ctx, b1, z1));
    struct ggml_tensor * a1 = ggml_relu(ctx, z1);
    
    // 第二層：線性變換
    struct ggml_tensor * z2 = ggml_mul_mat(ctx, w2, a1);
    z2 = ggml_add(ctx, z2, ggml_repeat(ctx, b2, z2));
    
    // Softmax 輸出
    struct ggml_tensor * output = ggml_soft_max(ctx, z2);
    
    // 交叉熵損失
    struct ggml_tensor * loss = ggml_cross_entropy_loss(ctx, output, y);
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);
    
    // 生成訓練資料
    float * x_train = (float *)malloc(n_samples * n_features * sizeof(float));
    float * y_train = (float *)malloc(n_samples * n_classes * sizeof(float));
    
    generate_dataset(x_train, y_train, n_samples, n_features, n_classes);
    
    // 建立計算圖
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_cgraph * gb = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    
    ggml_build_forward_expand(gf, loss);
    ggml_graph_cpy(gf, gb);
    ggml_build_backward_expand(ctx, gb, NULL);
    
    // 訓練循環
    printf("開始訓練多層感知器...\n");
    printf("樣本數: %d, 特徵數: %d, 隱藏層大小: %d, 類別數: %d\n", 
           n_samples, n_features, n_hidden, n_classes);
    printf("批次大小: %d, 學習率: %.4f\n", batch_size, learning_rate);
    printf("----------------------------------------\n");
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int n_correct = 0;
        int n_batches = (n_samples + batch_size - 1) / batch_size;
        
        // 打亂資料順序
        std::vector<int> indices(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));
        
        for (int batch = 0; batch < n_batches; ++batch) {
            int batch_start = batch * batch_size;
            int batch_end = std::min(batch_start + batch_size, n_samples);
            int actual_batch_size = batch_end - batch_start;
            
            // 複製批次資料
            float * x_data = ggml_get_data_f32(x);
            float * y_data = ggml_get_data_f32(y);
            
            for (int i = 0; i < actual_batch_size; ++i) {
                int idx = indices[batch_start + i];
                for (int j = 0; j < n_features; ++j) {
                    x_data[i * n_features + j] = x_train[idx * n_features + j];
                }
                for (int j = 0; j < n_classes; ++j) {
                    y_data[i * n_classes + j] = y_train[idx * n_classes + j];
                }
            }
            
            // 前向傳播
            ggml_graph_reset(gb);
            ggml_graph_compute_with_ctx(ctx, gb, 1);
            
            // 累積損失
            epoch_loss += ggml_get_f32_1d(loss, 0) * actual_batch_size;
            
            // 計算準確率
            float * output_data = ggml_get_data_f32(output);
            for (int i = 0; i < actual_batch_size; ++i) {
                int pred_class = 0;
                int true_class = 0;
                float max_prob = -1.0f;
                
                for (int j = 0; j < n_classes; ++j) {
                    if (output_data[i * n_classes + j] > max_prob) {
                        max_prob = output_data[i * n_classes + j];
                        pred_class = j;
                    }
                    if (y_data[i * n_classes + j] > 0.5f) {
                        true_class = j;
                    }
                }
                
                if (pred_class == true_class) {
                    n_correct++;
                }
            }
            
            // 反向傳播和參數更新
            struct ggml_tensor * gw1 = ggml_graph_get_grad(gb, w1);
            struct ggml_tensor * gb1 = ggml_graph_get_grad(gb, b1);
            struct ggml_tensor * gw2 = ggml_graph_get_grad(gb, w2);
            struct ggml_tensor * gb2 = ggml_graph_get_grad(gb, b2);
            
            // 手動實現 SGD 更新
            if (gw1 && gb1 && gw2 && gb2) {
                float * w1_data = ggml_get_data_f32(w1);
                float * b1_data = ggml_get_data_f32(b1);
                float * w2_data = ggml_get_data_f32(w2);
                float * b2_data = ggml_get_data_f32(b2);
                
                float * gw1_data = ggml_get_data_f32(gw1);
                float * gb1_data = ggml_get_data_f32(gb1);
                float * gw2_data = ggml_get_data_f32(gw2);
                float * gb2_data = ggml_get_data_f32(gb2);
                
                // 更新權重
                for (int i = 0; i < ggml_nelements(w1); ++i) {
                    w1_data[i] -= learning_rate * gw1_data[i];
                }
                for (int i = 0; i < ggml_nelements(b1); ++i) {
                    b1_data[i] -= learning_rate * gb1_data[i];
                }
                for (int i = 0; i < ggml_nelements(w2); ++i) {
                    w2_data[i] -= learning_rate * gw2_data[i];
                }
                for (int i = 0; i < ggml_nelements(b2); ++i) {
                    b2_data[i] -= learning_rate * gb2_data[i];
                }
            }
        }
        
        // 計算平均損失和準確率
        epoch_loss /= n_samples;
        float accuracy = (float)n_correct / n_samples * 100.0f;
        
        if (epoch % 50 == 0) {
            printf("Epoch %3d: Loss = %.4f, Accuracy = %.1f%%\n", 
                   epoch, epoch_loss, accuracy);
        }
    }
    
    printf("----------------------------------------\n");
    printf("訓練完成！\n");
    
    // 清理資源
    free(x_train);
    free(y_train);
    ggml_free(ctx);
    
    return 0;
} 