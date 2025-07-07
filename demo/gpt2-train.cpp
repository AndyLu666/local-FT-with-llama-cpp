#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <fstream>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

// GPT-2 模型參數
struct GPT2Config {
    int n_vocab = 50257;      // 詞彙表大小
    int n_ctx = 1024;         // 上下文長度
    int n_embd = 768;         // 嵌入維度
    int n_head = 12;          // 注意力頭數
    int n_layer = 12;         // 層數
    int n_ff = 3072;          // 前饋網絡維度
    float eps = 1e-5f;        // Layer norm epsilon
};

// GPT-2 模型
class GPT2Model {
public:
    GPT2Config config;
    
    // 模型參數
    struct ggml_tensor* wte;      // token embeddings
    struct ggml_tensor* wpe;      // position embeddings
    
    // 每層的參數
    std::vector<struct ggml_tensor*> ln1_g;     // layer norm 1 gain
    std::vector<struct ggml_tensor*> ln1_b;     // layer norm 1 bias
    std::vector<struct ggml_tensor*> ln2_g;     // layer norm 2 gain
    std::vector<struct ggml_tensor*> ln2_b;     // layer norm 2 bias
    
    std::vector<struct ggml_tensor*> c_attn_w;  // attention weights
    std::vector<struct ggml_tensor*> c_attn_b;  // attention bias
    std::vector<struct ggml_tensor*> c_proj_w;  // projection weights
    std::vector<struct ggml_tensor*> c_proj_b;  // projection bias
    
    std::vector<struct ggml_tensor*> c_fc_w;    // FFN weights
    std::vector<struct ggml_tensor*> c_fc_b;    // FFN bias
    std::vector<struct ggml_tensor*> c_proj2_w; // FFN projection weights
    std::vector<struct ggml_tensor*> c_proj2_b; // FFN projection bias
    
    struct ggml_tensor* ln_f_g;   // final layer norm gain
    struct ggml_tensor* ln_f_b;   // final layer norm bias
    
    // 上下文和後端
    struct ggml_context* ctx_params;
    struct ggml_backend* backend;
    struct ggml_backend_buffer* buffer;
    
    GPT2Model(const GPT2Config& cfg) : config(cfg) {
        // 初始化後端
        backend = nullptr;
#ifdef GGML_USE_CUDA
        if (ggml_backend_cuda_init(0) != nullptr) {
            backend = ggml_backend_cuda_init(0);
            std::cout << "使用 CUDA 後端" << std::endl;
        }
#endif
#ifdef GGML_USE_METAL
        if (backend == nullptr && ggml_backend_metal_init() != nullptr) {
            backend = ggml_backend_metal_init();
            std::cout << "使用 Metal 後端" << std::endl;
        }
#endif
        if (backend == nullptr) {
            backend = ggml_backend_cpu_init();
            std::cout << "使用 CPU 後端" << std::endl;
        }
        
        // 計算參數大小
        size_t params_size = 0;
        params_size += config.n_vocab * config.n_embd * sizeof(float); // wte
        params_size += config.n_ctx * config.n_embd * sizeof(float);   // wpe
        
        params_size += config.n_layer * config.n_embd * 2 * sizeof(float); // ln1
        params_size += config.n_layer * config.n_embd * 2 * sizeof(float); // ln2
        
        params_size += config.n_layer * config.n_embd * 3 * config.n_embd * sizeof(float); // c_attn_w
        params_size += config.n_layer * 3 * config.n_embd * sizeof(float);                 // c_attn_b
        params_size += config.n_layer * config.n_embd * config.n_embd * sizeof(float);     // c_proj_w
        params_size += config.n_layer * config.n_embd * sizeof(float);                     // c_proj_b
        
        params_size += config.n_layer * config.n_embd * config.n_ff * sizeof(float);   // c_fc_w
        params_size += config.n_layer * config.n_ff * sizeof(float);                   // c_fc_b
        params_size += config.n_layer * config.n_ff * config.n_embd * sizeof(float);   // c_proj2_w
        params_size += config.n_layer * config.n_embd * sizeof(float);                 // c_proj2_b
        
        params_size += config.n_embd * 2 * sizeof(float); // ln_f
        
        // 創建參數上下文
        struct ggml_init_params params = {
            /*.mem_size   =*/ params_size + 1024*1024, // 額外 1MB
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        
        ctx_params = ggml_init(params);
        
        // 初始化參數
        init_params();
        
        // 分配後端緩衝區
        buffer = ggml_backend_alloc_ctx_tensors(ctx_params, backend);
        
        // 隨機初始化權重
        randomize_params();
    }
    
    ~GPT2Model() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx_params) ggml_free(ctx_params);
        if (backend) ggml_backend_free(backend);
    }
    
private:
    void init_params() {
        // Token and position embeddings
        wte = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_embd, config.n_vocab);
        wpe = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_embd, config.n_ctx);
        
        ggml_set_name(wte, "wte");
        ggml_set_name(wpe, "wpe");
        ggml_set_param(wte);
        ggml_set_param(wpe);
        
        // 初始化每層參數
        ln1_g.resize(config.n_layer);
        ln1_b.resize(config.n_layer);
        ln2_g.resize(config.n_layer);
        ln2_b.resize(config.n_layer);
        
        c_attn_w.resize(config.n_layer);
        c_attn_b.resize(config.n_layer);
        c_proj_w.resize(config.n_layer);
        c_proj_b.resize(config.n_layer);
        
        c_fc_w.resize(config.n_layer);
        c_fc_b.resize(config.n_layer);
        c_proj2_w.resize(config.n_layer);
        c_proj2_b.resize(config.n_layer);
        
        for (int i = 0; i < config.n_layer; ++i) {
            // Layer norm 1
            ln1_g[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ln1_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ggml_set_name(ln1_g[i], ("ln1_g_" + std::to_string(i)).c_str());
            ggml_set_name(ln1_b[i], ("ln1_b_" + std::to_string(i)).c_str());
            ggml_set_param(ln1_g[i]);
            ggml_set_param(ln1_b[i]);
            
            // Layer norm 2
            ln2_g[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ln2_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ggml_set_name(ln2_g[i], ("ln2_g_" + std::to_string(i)).c_str());
            ggml_set_name(ln2_b[i], ("ln2_b_" + std::to_string(i)).c_str());
            ggml_set_param(ln2_g[i]);
            ggml_set_param(ln2_b[i]);
            
            // Attention
            c_attn_w[i] = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_embd, 3*config.n_embd);
            c_attn_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, 3*config.n_embd);
            ggml_set_name(c_attn_w[i], ("c_attn_w_" + std::to_string(i)).c_str());
            ggml_set_name(c_attn_b[i], ("c_attn_b_" + std::to_string(i)).c_str());
            ggml_set_param(c_attn_w[i]);
            ggml_set_param(c_attn_b[i]);
            
            c_proj_w[i] = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_embd, config.n_embd);
            c_proj_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ggml_set_name(c_proj_w[i], ("c_proj_w_" + std::to_string(i)).c_str());
            ggml_set_name(c_proj_b[i], ("c_proj_b_" + std::to_string(i)).c_str());
            ggml_set_param(c_proj_w[i]);
            ggml_set_param(c_proj_b[i]);
            
            // FFN
            c_fc_w[i] = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_embd, config.n_ff);
            c_fc_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_ff);
            ggml_set_name(c_fc_w[i], ("c_fc_w_" + std::to_string(i)).c_str());
            ggml_set_name(c_fc_b[i], ("c_fc_b_" + std::to_string(i)).c_str());
            ggml_set_param(c_fc_w[i]);
            ggml_set_param(c_fc_b[i]);
            
            c_proj2_w[i] = ggml_new_tensor_2d(ctx_params, GGML_TYPE_F32, config.n_ff, config.n_embd);
            c_proj2_b[i] = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
            ggml_set_name(c_proj2_w[i], ("c_proj2_w_" + std::to_string(i)).c_str());
            ggml_set_name(c_proj2_b[i], ("c_proj2_b_" + std::to_string(i)).c_str());
            ggml_set_param(c_proj2_w[i]);
            ggml_set_param(c_proj2_b[i]);
        }
        
        // Final layer norm
        ln_f_g = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
        ln_f_b = ggml_new_tensor_1d(ctx_params, GGML_TYPE_F32, config.n_embd);
        ggml_set_name(ln_f_g, "ln_f_g");
        ggml_set_name(ln_f_b, "ln_f_b");
        ggml_set_param(ln_f_g);
        ggml_set_param(ln_f_b);
    }
    
    void randomize_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Xavier initialization
        auto init_tensor = [&](struct ggml_tensor* t, float scale) {
            std::normal_distribution<float> dist(0.0f, scale);
            float* data = (float*)t->data;
            int64_t n = ggml_nelements(t);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(gen);
            }
        };
        
        // 初始化嵌入層
        init_tensor(wte, std::sqrt(2.0f / config.n_embd));
        init_tensor(wpe, std::sqrt(2.0f / config.n_embd));
        
        // 初始化每層參數
        for (int i = 0; i < config.n_layer; ++i) {
            // Layer norm 初始化為 1 和 0
            std::fill_n((float*)ln1_g[i]->data, config.n_embd, 1.0f);
            std::fill_n((float*)ln1_b[i]->data, config.n_embd, 0.0f);
            std::fill_n((float*)ln2_g[i]->data, config.n_embd, 1.0f);
            std::fill_n((float*)ln2_b[i]->data, config.n_embd, 0.0f);
            
            // Attention 和 FFN 權重
            init_tensor(c_attn_w[i], std::sqrt(2.0f / config.n_embd));
            std::fill_n((float*)c_attn_b[i]->data, 3*config.n_embd, 0.0f);
            
            init_tensor(c_proj_w[i], std::sqrt(2.0f / config.n_embd) / std::sqrt(2.0f * config.n_layer));
            std::fill_n((float*)c_proj_b[i]->data, config.n_embd, 0.0f);
            
            init_tensor(c_fc_w[i], std::sqrt(2.0f / config.n_embd));
            std::fill_n((float*)c_fc_b[i]->data, config.n_ff, 0.0f);
            
            init_tensor(c_proj2_w[i], std::sqrt(2.0f / config.n_ff) / std::sqrt(2.0f * config.n_layer));
            std::fill_n((float*)c_proj2_b[i]->data, config.n_embd, 0.0f);
        }
        
        // Final layer norm
        std::fill_n((float*)ln_f_g->data, config.n_embd, 1.0f);
        std::fill_n((float*)ln_f_b->data, config.n_embd, 0.0f);
    }
};

// 前向傳播
struct ggml_tensor* gpt2_forward(
    GPT2Model& model,
    struct ggml_context* ctx,
    struct ggml_tensor* input_ids,
    bool training = false
) {
    const auto& cfg = model.config;
    const int n_seq = input_ids->ne[0];
    const int n_batch = input_ids->ne[1];
    
    // Token embeddings
    struct ggml_tensor* tok_emb = ggml_get_rows(ctx, model.wte, input_ids);
    
    // Position embeddings
    struct ggml_tensor* positions = ggml_arange(ctx, 0, n_seq, 1);
    struct ggml_tensor* pos_emb = ggml_get_rows(ctx, model.wpe, positions);
    
    // 添加 embeddings
    struct ggml_tensor* x = ggml_add(ctx, tok_emb, pos_emb);
    
    // 通過每個 transformer 層
    for (int i = 0; i < cfg.n_layer; ++i) {
        // Layer norm 1
        struct ggml_tensor* ln1 = ggml_norm(ctx, x, cfg.eps);
        ln1 = ggml_mul(ctx, ln1, model.ln1_g[i]);
        ln1 = ggml_add(ctx, ln1, model.ln1_b[i]);
        
        // Self-attention
        struct ggml_tensor* qkv = ggml_mul_mat(ctx, model.c_attn_w[i], ln1);
        qkv = ggml_add(ctx, qkv, model.c_attn_b[i]);
        
        // 分割 Q, K, V
        struct ggml_tensor* q = ggml_view_3d(ctx, qkv, cfg.n_embd/cfg.n_head, cfg.n_head, n_seq*n_batch, 
                                            qkv->nb[1], qkv->nb[1]*cfg.n_embd/cfg.n_head, 0);
        struct ggml_tensor* k = ggml_view_3d(ctx, qkv, cfg.n_embd/cfg.n_head, cfg.n_head, n_seq*n_batch,
                                            qkv->nb[1], qkv->nb[1]*cfg.n_embd/cfg.n_head, cfg.n_embd*qkv->nb[1]);
        struct ggml_tensor* v = ggml_view_3d(ctx, qkv, cfg.n_embd/cfg.n_head, cfg.n_head, n_seq*n_batch,
                                            qkv->nb[1], qkv->nb[1]*cfg.n_embd/cfg.n_head, 2*cfg.n_embd*qkv->nb[1]);
        
        // Reshape for attention
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 1, 2, 0, 3);
        
        // Attention
        struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_scale(ctx, kq, 1.0f/std::sqrt(float(cfg.n_embd/cfg.n_head)));
        
        // Causal mask
        kq = ggml_diag_mask_inf(ctx, kq, 0);
        
        // Softmax
        kq = ggml_soft_max(ctx, kq);
        
        // Apply attention to values
        struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        
        // Reshape back
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        kqv = ggml_cont(ctx, kqv);
        kqv = ggml_reshape_2d(ctx, kqv, cfg.n_embd, n_seq*n_batch);
        
        // Projection
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx, model.c_proj_w[i], kqv);
        attn_out = ggml_add(ctx, attn_out, model.c_proj_b[i]);
        
        // Residual connection
        x = ggml_add(ctx, x, attn_out);
        
        // Layer norm 2
        struct ggml_tensor* ln2 = ggml_norm(ctx, x, cfg.eps);
        ln2 = ggml_mul(ctx, ln2, model.ln2_g[i]);
        ln2 = ggml_add(ctx, ln2, model.ln2_b[i]);
        
        // FFN
        struct ggml_tensor* ffn = ggml_mul_mat(ctx, model.c_fc_w[i], ln2);
        ffn = ggml_add(ctx, ffn, model.c_fc_b[i]);
        ffn = ggml_gelu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, model.c_proj2_w[i], ffn);
        ffn = ggml_add(ctx, ffn, model.c_proj2_b[i]);
        
        // Residual connection
        x = ggml_add(ctx, x, ffn);
    }
    
    // Final layer norm
    x = ggml_norm(ctx, x, cfg.eps);
    x = ggml_mul(ctx, x, model.ln_f_g);
    x = ggml_add(ctx, x, model.ln_f_b);
    
    // 輸出 logits
    struct ggml_tensor* logits = ggml_mul_mat(ctx, model.wte, x);
    
    return logits;
}

// 主訓練循環
void train_gpt2() {
    // 創建小型 GPT-2 配置
    GPT2Config config;
    config.n_vocab = 1000;   // 小詞彙表
    config.n_ctx = 128;      // 短上下文
    config.n_embd = 256;     // 小嵌入維度
    config.n_head = 8;       // 少注意力頭
    config.n_layer = 4;      // 少層數
    config.n_ff = 1024;      // 小 FFN
    
    std::cout << "初始化 GPT-2 模型..." << std::endl;
    GPT2Model model(config);
    
    // 訓練參數
    const int batch_size = 8;
    const int seq_length = 64;
    const int n_epochs = 10;
    const float learning_rate = 1e-4f;
    
    // 創建訓練數據（簡單的序列預測任務）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, config.n_vocab - 1);
    
    std::cout << "開始訓練..." << std::endl;
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        float total_loss = 0.0f;
        int n_batches = 100;
        
        for (int batch = 0; batch < n_batches; ++batch) {
            // 創建計算圖上下文
            size_t compute_size = 256*1024*1024; // 256 MB
            struct ggml_init_params params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context* ctx_compute = ggml_init(params);
            
            // 生成隨機輸入數據
            std::vector<int32_t> input_data(batch_size * seq_length);
            std::vector<int32_t> target_data(batch_size * seq_length);
            
            for (int i = 0; i < batch_size * seq_length; ++i) {
                input_data[i] = dist(gen);
                target_data[i] = (input_data[i] + 1) % config.n_vocab; // 簡單的 +1 預測任務
            }
            
            // 創建輸入張量
            struct ggml_tensor* input_ids = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, seq_length, batch_size);
            memcpy(input_ids->data, input_data.data(), input_data.size() * sizeof(int32_t));
            
            struct ggml_tensor* targets = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, seq_length, batch_size);
            memcpy(targets->data, target_data.data(), target_data.size() * sizeof(int32_t));
            
            // 前向傳播
            struct ggml_tensor* logits = gpt2_forward(model, ctx_compute, input_ids, true);
            
            // 計算損失
            struct ggml_tensor* loss = ggml_cross_entropy_loss(ctx_compute, 
                ggml_reshape_2d(ctx_compute, logits, config.n_vocab, seq_length * batch_size),
                ggml_reshape_1d(ctx_compute, targets, seq_length * batch_size));
            
            ggml_set_loss(loss);
            
            // 構建計算圖
            struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_compute, 2048, true);
            ggml_build_forward_expand(gf, loss);
            
            // 分配計算緩衝區
            ggml_gallocr* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            ggml_gallocr_alloc_graph(alloc, gf);
            
            // 複製輸入數據到後端
            ggml_backend_tensor_set(input_ids, input_data.data(), 0, input_data.size() * sizeof(int32_t));
            ggml_backend_tensor_set(targets, target_data.data(), 0, target_data.size() * sizeof(int32_t));
            
            // 執行前向傳播
            ggml_backend_graph_compute(model.backend, gf);
            
            // 獲取損失值
            float loss_val;
            ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));
            total_loss += loss_val;
            
            // 構建反向傳播圖
            ggml_build_backward_expand(ctx_compute, gf, nullptr);
            
            // TODO: 應用梯度更新
            // 這裡需要實現優化器步驟
            
            // 清理
            ggml_gallocr_free(alloc);
            ggml_free(ctx_compute);
            
            if (batch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Batch " << batch << "/" << n_batches 
                         << ", Loss: " << loss_val << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch << " completed. Average loss: " << total_loss / n_batches << std::endl;
    }
    
    std::cout << "訓練完成！" << std::endl;
}

int main() {
    std::cout << "GPT-2 訓練示例" << std::endl;
    std::cout << "===============" << std::endl;
    
    train_gpt2();
    
    return 0;
} 