#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

// 簡化的 GPT-2 配置
struct SimpleGPT2Config {
    int n_vocab = 100;    // 詞彙表大小
    int n_ctx = 32;       // 上下文長度
    int n_embd = 128;     // 嵌入維度
    int n_head = 4;       // 注意力頭數
    int n_layer = 2;      // 層數
    float eps = 1e-5f;    // Layer norm epsilon
};

// 創建簡單的 GPT-2 模型
struct SimpleGPT2 {
    SimpleGPT2Config config;
    struct ggml_context* ctx;
    
    // 模型參數
    struct ggml_tensor* wte;      // token embeddings [n_vocab, n_embd]
    struct ggml_tensor* wpe;      // position embeddings [n_ctx, n_embd]
    
    // Transformer 層參數
    std::vector<struct ggml_tensor*> ln1_w;     // [n_layer][n_embd]
    std::vector<struct ggml_tensor*> ln1_b;     // [n_layer][n_embd]
    
    std::vector<struct ggml_tensor*> attn_qkv_w;  // [n_layer][n_embd, 3*n_embd]
    std::vector<struct ggml_tensor*> attn_qkv_b;  // [n_layer][3*n_embd]
    std::vector<struct ggml_tensor*> attn_proj_w; // [n_layer][n_embd, n_embd]
    std::vector<struct ggml_tensor*> attn_proj_b; // [n_layer][n_embd]
    
    std::vector<struct ggml_tensor*> ln2_w;     // [n_layer][n_embd]
    std::vector<struct ggml_tensor*> ln2_b;     // [n_layer][n_embd]
    
    std::vector<struct ggml_tensor*> ffn_up_w;   // [n_layer][n_embd, 4*n_embd]
    std::vector<struct ggml_tensor*> ffn_up_b;   // [n_layer][4*n_embd]
    std::vector<struct ggml_tensor*> ffn_down_w; // [n_layer][4*n_embd, n_embd]
    std::vector<struct ggml_tensor*> ffn_down_b; // [n_layer][n_embd]
    
    struct ggml_tensor* ln_f_w;   // [n_embd]
    struct ggml_tensor* ln_f_b;   // [n_embd]
    
    SimpleGPT2(struct ggml_context* ctx, const SimpleGPT2Config& cfg) 
        : ctx(ctx), config(cfg) {
        init_params();
    }
    
    void init_params() {
        // Embeddings
        wte = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_embd, config.n_vocab);
        wpe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_embd, config.n_ctx);
        ggml_set_param(wte);
        ggml_set_param(wpe);
        
        // 初始化層參數
        ln1_w.resize(config.n_layer);
        ln1_b.resize(config.n_layer);
        attn_qkv_w.resize(config.n_layer);
        attn_qkv_b.resize(config.n_layer);
        attn_proj_w.resize(config.n_layer);
        attn_proj_b.resize(config.n_layer);
        ln2_w.resize(config.n_layer);
        ln2_b.resize(config.n_layer);
        ffn_up_w.resize(config.n_layer);
        ffn_up_b.resize(config.n_layer);
        ffn_down_w.resize(config.n_layer);
        ffn_down_b.resize(config.n_layer);
        
        for (int i = 0; i < config.n_layer; ++i) {
            // Layer norm 1
            ln1_w[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ln1_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ggml_set_param(ln1_w[i]);
            ggml_set_param(ln1_b[i]);
            
            // Attention
            attn_qkv_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_embd, 3*config.n_embd);
            attn_qkv_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*config.n_embd);
            attn_proj_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_embd, config.n_embd);
            attn_proj_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ggml_set_param(attn_qkv_w[i]);
            ggml_set_param(attn_qkv_b[i]);
            ggml_set_param(attn_proj_w[i]);
            ggml_set_param(attn_proj_b[i]);
            
            // Layer norm 2
            ln2_w[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ln2_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ggml_set_param(ln2_w[i]);
            ggml_set_param(ln2_b[i]);
            
            // FFN
            ffn_up_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_embd, 4*config.n_embd);
            ffn_up_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*config.n_embd);
            ffn_down_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4*config.n_embd, config.n_embd);
            ffn_down_b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
            ggml_set_param(ffn_up_w[i]);
            ggml_set_param(ffn_up_b[i]);
            ggml_set_param(ffn_down_w[i]);
            ggml_set_param(ffn_down_b[i]);
        }
        
        // Final layer norm
        ln_f_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
        ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
        ggml_set_param(ln_f_w);
        ggml_set_param(ln_f_b);
    }
    
    // 隨機初始化參數
    void randomize_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        auto init_tensor = [&](struct ggml_tensor* t, float scale) {
            std::normal_distribution<float> dist(0.0f, scale);
            float* data = (float*)t->data;
            int64_t n = ggml_nelements(t);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(gen);
            }
        };
        
        // 初始化 embeddings
        init_tensor(wte, 0.02f);
        init_tensor(wpe, 0.02f);
        
        // 初始化層參數
        for (int i = 0; i < config.n_layer; ++i) {
            // Layer norms 初始化為 1 和 0
            std::fill_n((float*)ln1_w[i]->data, config.n_embd, 1.0f);
            std::fill_n((float*)ln1_b[i]->data, config.n_embd, 0.0f);
            std::fill_n((float*)ln2_w[i]->data, config.n_embd, 1.0f);
            std::fill_n((float*)ln2_b[i]->data, config.n_embd, 0.0f);
            
            // 權重初始化
            init_tensor(attn_qkv_w[i], 0.02f);
            std::fill_n((float*)attn_qkv_b[i]->data, 3*config.n_embd, 0.0f);
            init_tensor(attn_proj_w[i], 0.02f / std::sqrt(2.0f * config.n_layer));
            std::fill_n((float*)attn_proj_b[i]->data, config.n_embd, 0.0f);
            
            init_tensor(ffn_up_w[i], 0.02f);
            std::fill_n((float*)ffn_up_b[i]->data, 4*config.n_embd, 0.0f);
            init_tensor(ffn_down_w[i], 0.02f / std::sqrt(2.0f * config.n_layer));
            std::fill_n((float*)ffn_down_b[i]->data, config.n_embd, 0.0f);
        }
        
        // Final layer norm
        std::fill_n((float*)ln_f_w->data, config.n_embd, 1.0f);
        std::fill_n((float*)ln_f_b->data, config.n_embd, 0.0f);
    }
};

// 簡單的前向傳播
struct ggml_tensor* simple_gpt2_forward(
    SimpleGPT2& model,
    struct ggml_context* ctx,
    struct ggml_tensor* input_ids  // [seq_len, batch_size]
) {
    const auto& cfg = model.config;
    const int seq_len = input_ids->ne[0];
    const int batch_size = input_ids->ne[1];
    
    // 1. Token embeddings
    struct ggml_tensor* tok_emb = ggml_get_rows(ctx, model.wte, input_ids);
    
    // 2. Position embeddings
    struct ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        ((int32_t*)positions->data)[i] = i;
    }
    struct ggml_tensor* pos_emb = ggml_get_rows(ctx, model.wpe, positions);
    
    // 3. 合併 embeddings
    struct ggml_tensor* x = ggml_add(ctx, tok_emb, pos_emb);
    
    // 4. Transformer 層
    for (int layer = 0; layer < cfg.n_layer; ++layer) {
        struct ggml_tensor* residual = x;
        
        // Layer norm 1
        x = ggml_norm(ctx, x, cfg.eps);
        x = ggml_mul(ctx, x, model.ln1_w[layer]);
        x = ggml_add(ctx, x, model.ln1_b[layer]);
        
        // Self-attention
        struct ggml_tensor* qkv = ggml_mul_mat(ctx, model.attn_qkv_w[layer], x);
        qkv = ggml_add(ctx, qkv, model.attn_qkv_b[layer]);
        
        // 簡化的注意力計算（為了演示）
        // 實際實現需要正確的 reshape 和 permute
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx, model.attn_proj_w[layer], qkv);
        attn_out = ggml_add(ctx, attn_out, model.attn_proj_b[layer]);
        
        // Residual connection
        x = ggml_add(ctx, residual, attn_out);
        residual = x;
        
        // Layer norm 2
        x = ggml_norm(ctx, x, cfg.eps);
        x = ggml_mul(ctx, x, model.ln2_w[layer]);
        x = ggml_add(ctx, x, model.ln2_b[layer]);
        
        // FFN
        struct ggml_tensor* ffn = ggml_mul_mat(ctx, model.ffn_up_w[layer], x);
        ffn = ggml_add(ctx, ffn, model.ffn_up_b[layer]);
        ffn = ggml_gelu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, model.ffn_down_w[layer], ffn);
        ffn = ggml_add(ctx, ffn, model.ffn_down_b[layer]);
        
        // Residual connection
        x = ggml_add(ctx, residual, ffn);
    }
    
    // 5. Final layer norm
    x = ggml_norm(ctx, x, cfg.eps);
    x = ggml_mul(ctx, x, model.ln_f_w);
    x = ggml_add(ctx, x, model.ln_f_b);
    
    // 6. 輸出 logits
    struct ggml_tensor* logits = ggml_mul_mat(ctx, model.wte, x);
    
    return logits;
}

// 訓練循環
void train_simple_gpt2() {
    std::cout << "簡單 GPT-2 訓練示例" << std::endl;
    std::cout << "===================" << std::endl;
    
    // 配置
    SimpleGPT2Config config;
    config.n_vocab = 50;
    config.n_ctx = 16;
    config.n_embd = 64;
    config.n_head = 4;
    config.n_layer = 2;
    
    // 訓練參數
    const int batch_size = 4;
    const int seq_length = 8;
    const int n_epochs = 5;
    const float learning_rate = 1e-3f;
    
    // 創建參數上下文
    size_t params_size = 16*1024*1024; // 16MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ params_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx_params = ggml_init(params);
    
    // 創建模型
    SimpleGPT2 model(ctx_params, config);
    model.randomize_params();
    
    std::cout << "模型參數已初始化" << std::endl;
    
    // 創建優化器參數
    struct ggml_tensor* adamw_params = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, 7);
    float* adamw_data = (float*)adamw_params->data;
    adamw_data[0] = learning_rate; // alpha
    adamw_data[1] = 0.9f;          // beta1
    adamw_data[2] = 0.999f;        // beta2
    adamw_data[3] = 1e-8f;         // eps
    adamw_data[4] = 0.01f;         // weight decay
    adamw_data[5] = 1.0f;          // adam_pf
    adamw_data[6] = 1.0f;          // adam_eps
    
    // 為每個參數創建動量和速度緩衝區
    std::vector<struct ggml_tensor*> params_list;
    std::vector<struct ggml_tensor*> momentum;
    std::vector<struct ggml_tensor*> velocity;
    
    // 收集所有參數
    for (int i = 0; i < ggml_graph_n_nodes(ctx_params); ++i) {
        struct ggml_tensor* t = ggml_get_tensor(ctx_params, std::to_string(i).c_str());
        if (t && (t->flags & GGML_TENSOR_FLAG_PARAM)) {
            params_list.push_back(t);
            momentum.push_back(ggml_dup_tensor(ctx_params, t));
            velocity.push_back(ggml_dup_tensor(ctx_params, t));
            ggml_set_zero(momentum.back());
            ggml_set_zero(velocity.back());
        }
    }
    
    std::cout << "開始訓練..." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, config.n_vocab - 1);
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        float total_loss = 0.0f;
        int n_batches = 10;
        
        for (int batch = 0; batch < n_batches; ++batch) {
            // 創建計算上下文
            size_t compute_size = 32*1024*1024; // 32MB
            struct ggml_init_params compute_params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context* ctx_compute = ggml_init(compute_params);
            
            // 生成隨機數據
            std::vector<int32_t> input_data(batch_size * seq_length);
            std::vector<int32_t> target_data(batch_size * seq_length);
            
            for (int i = 0; i < batch_size * seq_length; ++i) {
                input_data[i] = dist(gen);
                target_data[i] = dist(gen);
            }
            
            // 創建輸入張量
            struct ggml_tensor* input_ids = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, seq_length, batch_size);
            memcpy(input_ids->data, input_data.data(), input_data.size() * sizeof(int32_t));
            
            struct ggml_tensor* targets = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, seq_length * batch_size);
            memcpy(targets->data, target_data.data(), target_data.size() * sizeof(int32_t));
            
            // 前向傳播
            struct ggml_tensor* logits = simple_gpt2_forward(model, ctx_compute, input_ids);
            
            // Reshape logits
            logits = ggml_reshape_2d(ctx_compute, logits, config.n_vocab, seq_length * batch_size);
            
            // 計算損失
            struct ggml_tensor* loss = ggml_cross_entropy_loss(ctx_compute, logits, targets);
            ggml_set_loss(loss);
            
            // 構建前向圖
            struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_compute, 1024, true);
            ggml_build_forward_expand(gf, loss);
            
            // 計算前向傳播
            ggml_graph_compute_with_ctx(ctx_compute, gf, 1);
            
            // 獲取損失值
            float loss_val = ((float*)loss->data)[0];
            total_loss += loss_val;
            
            // 構建反向圖
            ggml_build_backward_expand(ctx_compute, gf, nullptr);
            
            // 計算反向傳播
            ggml_graph_compute_with_ctx(ctx_compute, gf, 1);
            
            // 應用梯度更新（使用 AdamW）
            for (size_t i = 0; i < params_list.size(); ++i) {
                struct ggml_tensor* param = params_list[i];
                struct ggml_tensor* grad = ggml_graph_get_grad(gf, param);
                
                if (grad) {
                    // 創建優化器步驟
                    struct ggml_tensor* opt_step = ggml_opt_step_adamw(
                        ctx_compute, param, grad, momentum[i], velocity[i], adamw_params);
                    
                    // 執行優化步驟
                    ggml_graph_compute_with_ctx(ctx_compute, ggml_new_graph_custom(ctx_compute, 1, false), 1);
                }
            }
            
            if (batch % 5 == 0) {
                std::cout << "Epoch " << epoch << ", Batch " << batch 
                         << ", Loss: " << loss_val << std::endl;
            }
            
            ggml_free(ctx_compute);
        }
        
        std::cout << "Epoch " << epoch << " 完成，平均損失: " 
                  << total_loss / n_batches << std::endl;
    }
    
    std::cout << "訓練完成！" << std::endl;
    
    ggml_free(ctx_params);
}

int main() {
    std::cout << "GGML GPT-2 訓練示例" << std::endl;
    std::cout << "==================" << std::endl;
    
    // 檢查 GGML 版本和功能
    std::cout << "檢查 GGML 功能..." << std::endl;
    
    // 運行訓練
    train_simple_gpt2();
    
    return 0;
} 