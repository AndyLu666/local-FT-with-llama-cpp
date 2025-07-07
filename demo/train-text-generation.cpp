#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

// 簡單的文本生成模型配置
struct TextGenConfig {
    int vocab_size = 256;     // ASCII 字符
    int context_size = 64;    // 上下文長度
    int embed_dim = 128;      // 嵌入維度
    int n_heads = 4;          // 注意力頭數
    int n_layers = 2;         // 層數
    int ff_dim = 512;         // FFN 維度
    float dropout = 0.1f;     // Dropout 率
};

// 簡單的 Tokenizer（字符級別）
class CharTokenizer {
public:
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<unsigned char>(c));
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int token : tokens) {
            if (token >= 0 && token < 256) {
                text += static_cast<char>(token);
            }
        }
        return text;
    }
};

// 文本生成模型
class TextGenModel {
public:
    TextGenConfig config;
    struct ggml_context* ctx;
    struct ggml_backend* backend;
    struct ggml_backend_buffer* buffer;
    
    // 模型參數
    struct ggml_tensor* token_emb;    // [vocab_size, embed_dim]
    struct ggml_tensor* pos_emb;      // [context_size, embed_dim]
    
    // Transformer 層
    struct {
        struct ggml_tensor* ln1_w;     // [embed_dim]
        struct ggml_tensor* ln1_b;     // [embed_dim]
        struct ggml_tensor* qkv_w;     // [embed_dim, 3*embed_dim]
        struct ggml_tensor* qkv_b;     // [3*embed_dim]
        struct ggml_tensor* proj_w;    // [embed_dim, embed_dim]
        struct ggml_tensor* proj_b;    // [embed_dim]
        struct ggml_tensor* ln2_w;     // [embed_dim]
        struct ggml_tensor* ln2_b;     // [embed_dim]
        struct ggml_tensor* ff1_w;     // [embed_dim, ff_dim]
        struct ggml_tensor* ff1_b;     // [ff_dim]
        struct ggml_tensor* ff2_w;     // [ff_dim, embed_dim]
        struct ggml_tensor* ff2_b;     // [embed_dim]
    } *layers;
    
    struct ggml_tensor* ln_out_w;     // [embed_dim]
    struct ggml_tensor* ln_out_b;     // [embed_dim]
    struct ggml_tensor* head_w;       // [embed_dim, vocab_size]
    struct ggml_tensor* head_b;       // [vocab_size]
    
    TextGenModel(const TextGenConfig& cfg) : config(cfg) {
        // 初始化後端
        backend = ggml_backend_cpu_init();
        
        // 計算參數大小
        size_t param_size = 0;
        param_size += config.vocab_size * config.embed_dim * sizeof(float); // token_emb
        param_size += config.context_size * config.embed_dim * sizeof(float); // pos_emb
        
        // 每層參數
        param_size += config.n_layers * (
            config.embed_dim * 2 * sizeof(float) +              // ln1
            config.embed_dim * 3 * config.embed_dim * sizeof(float) + // qkv
            3 * config.embed_dim * sizeof(float) +              // qkv_b
            config.embed_dim * config.embed_dim * sizeof(float) + // proj
            config.embed_dim * sizeof(float) +                  // proj_b
            config.embed_dim * 2 * sizeof(float) +              // ln2
            config.embed_dim * config.ff_dim * sizeof(float) +  // ff1
            config.ff_dim * sizeof(float) +                     // ff1_b
            config.ff_dim * config.embed_dim * sizeof(float) +  // ff2
            config.embed_dim * sizeof(float)                    // ff2_b
        );
        
        param_size += config.embed_dim * 2 * sizeof(float); // ln_out
        param_size += config.embed_dim * config.vocab_size * sizeof(float); // head
        param_size += config.vocab_size * sizeof(float); // head_b
        
        // 創建上下文
        struct ggml_init_params params = {
            /*.mem_size   =*/ param_size + 1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        
        ctx = ggml_init(params);
        
        // 初始化參數
        init_params();
        
        // 分配緩衝區
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        
        // 隨機初始化
        randomize_params();
    }
    
    ~TextGenModel() {
        delete[] layers;
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    
private:
    void init_params() {
        // Embeddings
        token_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, config.vocab_size);
        pos_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, config.context_size);
        ggml_set_param(token_emb);
        ggml_set_param(pos_emb);
        
        // Layers
        layers = new decltype(layers[0])[config.n_layers];
        
        for (int i = 0; i < config.n_layers; ++i) {
            auto& layer = layers[i];
            
            layer.ln1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            layer.ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            ggml_set_param(layer.ln1_w);
            ggml_set_param(layer.ln1_b);
            
            layer.qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, 3*config.embed_dim);
            layer.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*config.embed_dim);
            ggml_set_param(layer.qkv_w);
            ggml_set_param(layer.qkv_b);
            
            layer.proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, config.embed_dim);
            layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            ggml_set_param(layer.proj_w);
            ggml_set_param(layer.proj_b);
            
            layer.ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            layer.ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            ggml_set_param(layer.ln2_w);
            ggml_set_param(layer.ln2_b);
            
            layer.ff1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, config.ff_dim);
            layer.ff1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.ff_dim);
            ggml_set_param(layer.ff1_w);
            ggml_set_param(layer.ff1_b);
            
            layer.ff2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.ff_dim, config.embed_dim);
            layer.ff2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
            ggml_set_param(layer.ff2_w);
            ggml_set_param(layer.ff2_b);
        }
        
        // Output
        ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
        ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embed_dim);
        head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.embed_dim, config.vocab_size);
        head_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.vocab_size);
        
        ggml_set_param(ln_out_w);
        ggml_set_param(ln_out_b);
        ggml_set_param(head_w);
        ggml_set_param(head_b);
    }
    
    void randomize_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        auto init_tensor = [&](struct ggml_tensor* t, float scale) {
            std::normal_distribution<float> dist(0.0f, scale);
            float* data = (float*)ggml_backend_tensor_get_data(t);
            int64_t n = ggml_nelements(t);
            std::vector<float> init_data(n);
            for (int64_t i = 0; i < n; ++i) {
                init_data[i] = dist(gen);
            }
            ggml_backend_tensor_set(t, init_data.data(), 0, n * sizeof(float));
        };
        
        // 初始化 embeddings
        init_tensor(token_emb, 0.02f);
        init_tensor(pos_emb, 0.02f);
        
        // 初始化層
        for (int i = 0; i < config.n_layers; ++i) {
            auto& layer = layers[i];
            
            // Layer norms
            std::vector<float> ones(config.embed_dim, 1.0f);
            std::vector<float> zeros(config.embed_dim, 0.0f);
            
            ggml_backend_tensor_set(layer.ln1_w, ones.data(), 0, ones.size() * sizeof(float));
            ggml_backend_tensor_set(layer.ln1_b, zeros.data(), 0, zeros.size() * sizeof(float));
            ggml_backend_tensor_set(layer.ln2_w, ones.data(), 0, ones.size() * sizeof(float));
            ggml_backend_tensor_set(layer.ln2_b, zeros.data(), 0, zeros.size() * sizeof(float));
            
            // Weights
            init_tensor(layer.qkv_w, 0.02f);
            init_tensor(layer.proj_w, 0.02f / std::sqrt(2.0f * config.n_layers));
            init_tensor(layer.ff1_w, 0.02f);
            init_tensor(layer.ff2_w, 0.02f / std::sqrt(2.0f * config.n_layers));
            
            // Biases
            std::vector<float> zeros_qkv(3 * config.embed_dim, 0.0f);
            std::vector<float> zeros_ff(config.ff_dim, 0.0f);
            
            ggml_backend_tensor_set(layer.qkv_b, zeros_qkv.data(), 0, zeros_qkv.size() * sizeof(float));
            ggml_backend_tensor_set(layer.proj_b, zeros.data(), 0, zeros.size() * sizeof(float));
            ggml_backend_tensor_set(layer.ff1_b, zeros_ff.data(), 0, zeros_ff.size() * sizeof(float));
            ggml_backend_tensor_set(layer.ff2_b, zeros.data(), 0, zeros.size() * sizeof(float));
        }
        
        // Output layer
        std::vector<float> ones(config.embed_dim, 1.0f);
        std::vector<float> zeros_out(config.vocab_size, 0.0f);
        
        ggml_backend_tensor_set(ln_out_w, ones.data(), 0, ones.size() * sizeof(float));
        ggml_backend_tensor_set(ln_out_b, ones.data(), 0, ones.size() * sizeof(float));
        init_tensor(head_w, 0.02f);
        ggml_backend_tensor_set(head_b, zeros_out.data(), 0, zeros_out.size() * sizeof(float));
    }
};

// 前向傳播
struct ggml_tensor* text_gen_forward(
    TextGenModel& model,
    struct ggml_context* ctx,
    struct ggml_tensor* input_ids,  // [seq_len, batch_size]
    bool use_cache = false
) {
    const auto& cfg = model.config;
    const int seq_len = input_ids->ne[0];
    const int batch_size = input_ids->ne[1];
    
    // Token embeddings
    struct ggml_tensor* tok_emb = ggml_get_rows(ctx, model.token_emb, input_ids);
    
    // Position embeddings
    struct ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    std::vector<int32_t> pos_data(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(positions, pos_data.data(), 0, seq_len * sizeof(int32_t));
    
    struct ggml_tensor* pos_emb = ggml_get_rows(ctx, model.pos_emb, positions);
    
    // Combine embeddings
    struct ggml_tensor* x = ggml_add(ctx, tok_emb, pos_emb);
    
    // Transformer layers
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        auto& l = model.layers[layer];
        struct ggml_tensor* residual = x;
        
        // Layer norm 1
        x = ggml_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, l.ln1_w);
        x = ggml_add(ctx, x, l.ln1_b);
        
        // Self-attention (simplified)
        struct ggml_tensor* qkv = ggml_mul_mat(ctx, l.qkv_w, x);
        qkv = ggml_add(ctx, qkv, l.qkv_b);
        
        // Project back
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx, l.proj_w, qkv);
        attn_out = ggml_add(ctx, attn_out, l.proj_b);
        
        // Residual
        x = ggml_add(ctx, residual, attn_out);
        residual = x;
        
        // Layer norm 2
        x = ggml_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, l.ln2_w);
        x = ggml_add(ctx, x, l.ln2_b);
        
        // FFN
        struct ggml_tensor* ff = ggml_mul_mat(ctx, l.ff1_w, x);
        ff = ggml_add(ctx, ff, l.ff1_b);
        ff = ggml_gelu(ctx, ff);
        ff = ggml_mul_mat(ctx, l.ff2_w, ff);
        ff = ggml_add(ctx, ff, l.ff2_b);
        
        // Residual
        x = ggml_add(ctx, residual, ff);
    }
    
    // Output layer norm
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, model.ln_out_w);
    x = ggml_add(ctx, x, model.ln_out_b);
    
    // Logits
    struct ggml_tensor* logits = ggml_mul_mat(ctx, model.head_w, x);
    logits = ggml_add(ctx, logits, model.head_b);
    
    return logits;
}

// 訓練函數
void train_text_generation() {
    std::cout << "文本生成模型訓練" << std::endl;
    std::cout << "================" << std::endl;
    
    // 配置
    TextGenConfig config;
    config.vocab_size = 256;
    config.context_size = 32;
    config.embed_dim = 64;
    config.n_heads = 4;
    config.n_layers = 2;
    config.ff_dim = 256;
    
    // 創建模型
    TextGenModel model(config);
    CharTokenizer tokenizer;
    
    // 訓練文本
    std::string training_text = 
        "The quick brown fox jumps over the lazy dog. "
        "To be or not to be, that is the question. "
        "All that glitters is not gold. "
        "A journey of a thousand miles begins with a single step.";
    
    // 準備訓練數據
    auto tokens = tokenizer.encode(training_text);
    std::cout << "訓練文本長度: " << tokens.size() << " tokens" << std::endl;
    
    // 訓練參數
    const int batch_size = 4;
    const int seq_length = 16;
    const int n_epochs = 10;
    const float learning_rate = 1e-3f;
    
    // 創建 AdamW 參數
    std::vector<float> adamw_data = {
        learning_rate,  // alpha
        0.9f,          // beta1
        0.999f,        // beta2
        1e-8f,         // eps
        0.01f,         // weight decay
        1.0f,          // adam_pf
        1.0f           // adam_eps
    };
    
    std::cout << "\n開始訓練..." << std::endl;
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        float total_loss = 0.0f;
        int n_batches = 0;
        
        // 滑動窗口生成批次
        for (size_t i = 0; i + seq_length + 1 < tokens.size(); i += seq_length) {
            // 創建計算上下文
            size_t compute_size = 64*1024*1024; // 64MB
            struct ggml_init_params compute_params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ false,
            };
            
            struct ggml_context* ctx_compute = ggml_init(compute_params);
            
            // 準備批次數據
            std::vector<int32_t> input_data;
            std::vector<int32_t> target_data;
            
            for (int b = 0; b < batch_size && i + b * seq_length + seq_length + 1 < tokens.size(); ++b) {
                for (int j = 0; j < seq_length; ++j) {
                    input_data.push_back(tokens[i + b * seq_length + j]);
                    target_data.push_back(tokens[i + b * seq_length + j + 1]);
                }
            }
            
            int actual_batch_size = input_data.size() / seq_length;
            if (actual_batch_size == 0) break;
            
            // 創建輸入張量
            struct ggml_tensor* input_ids = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, 
                                                              seq_length, actual_batch_size);
            struct ggml_tensor* targets = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, 
                                                            seq_length * actual_batch_size);
            
            ggml_backend_tensor_set(input_ids, input_data.data(), 0, 
                                   input_data.size() * sizeof(int32_t));
            ggml_backend_tensor_set(targets, target_data.data(), 0, 
                                   target_data.size() * sizeof(int32_t));
            
            // 前向傳播
            struct ggml_tensor* logits = text_gen_forward(model, ctx_compute, input_ids);
            
            // Reshape logits
            logits = ggml_reshape_2d(ctx_compute, logits, config.vocab_size, 
                                    seq_length * actual_batch_size);
            
            // 計算損失
            struct ggml_tensor* loss = ggml_cross_entropy_loss(ctx_compute, logits, targets);
            ggml_set_loss(loss);
            
            // 構建計算圖
            struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_compute, 2048, true);
            ggml_build_forward_expand(gf, loss);
            
            // 分配計算緩衝區
            ggml_gallocr* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            ggml_gallocr_alloc_graph(alloc, gf);
            
            // 執行前向傳播
            ggml_backend_graph_compute(model.backend, gf);
            
            // 獲取損失值
            float loss_val;
            ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));
            total_loss += loss_val;
            n_batches++;
            
            // 構建反向傳播圖
            ggml_build_backward_expand(ctx_compute, gf, nullptr);
            
            // 執行反向傳播
            ggml_backend_graph_compute(model.backend, gf);
            
            // TODO: 應用梯度更新
            // 這裡需要實現參數更新邏輯
            
            ggml_gallocr_free(alloc);
            ggml_free(ctx_compute);
        }
        
        if (n_batches > 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << n_epochs 
                     << ", 平均損失: " << total_loss / n_batches << std::endl;
        }
    }
    
    std::cout << "\n訓練完成！" << std::endl;
    
    // 生成一些文本
    std::cout << "\n生成文本示例:" << std::endl;
    std::string prompt = "The quick";
    auto prompt_tokens = tokenizer.encode(prompt);
    
    std::cout << "提示: " << prompt << std::endl;
    std::cout << "生成: " << prompt;
    
    // 簡單的貪婪解碼
    for (int i = 0; i < 20; ++i) {
        // 創建計算上下文
        size_t compute_size = 16*1024*1024;
        struct ggml_init_params compute_params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        
        struct ggml_context* ctx_compute = ggml_init(compute_params);
        
        // 準備輸入
        int context_len = std::min((int)prompt_tokens.size(), config.context_size);
        std::vector<int32_t> input_data(prompt_tokens.end() - context_len, prompt_tokens.end());
        
        struct ggml_tensor* input_ids = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, context_len, 1);
        ggml_backend_tensor_set(input_ids, input_data.data(), 0, input_data.size() * sizeof(int32_t));
        
        // 前向傳播
        struct ggml_tensor* logits = text_gen_forward(model, ctx_compute, input_ids);
        
        // 構建計算圖
        struct ggml_cgraph* gf = ggml_new_graph(ctx_compute);
        ggml_build_forward_expand(gf, logits);
        
        // 分配和計算
        ggml_gallocr* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_graph_compute(model.backend, gf);
        
        // 獲取最後一個位置的 logits
        std::vector<float> last_logits(config.vocab_size);
        ggml_backend_tensor_get(logits, last_logits.data(), 
                               (context_len - 1) * config.vocab_size * sizeof(float), 
                               config.vocab_size * sizeof(float));
        
        // 找到最大值
        int next_token = std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin();
        
        // 添加到序列
        prompt_tokens.push_back(next_token);
        std::cout << tokenizer.decode({next_token});
        std::cout.flush();
        
        ggml_gallocr_free(alloc);
        ggml_free(ctx_compute);
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "GGML 文本生成訓練示例" << std::endl;
    std::cout << "=====================" << std::endl;
    
    train_text_generation();
    
    return 0;
} 