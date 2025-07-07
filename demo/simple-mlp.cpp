#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

// 簡單的矩陣類
class Matrix {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<float>(cols, 0.0f));
    }
    
    // Xavier 初始化
    void xavier_init(std::mt19937& rng) {
        float scale = std::sqrt(2.0f / (rows + cols));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dist(rng) * 0.5f;  // 進一步縮小初始化值
            }
        }
    }
    
    // 矩陣乘法
    Matrix matmul(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    // 轉置
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
    
    // 元素加法
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    // 標量乘法
    Matrix operator*(float scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    // 更新權重（帶梯度裁剪）
    void update(const Matrix& gradient, float learning_rate) {
        const float max_grad = 1.0f;  // 梯度裁剪閾值
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float grad = gradient.data[i][j];
                // 梯度裁剪
                if (grad > max_grad) grad = max_grad;
                if (grad < -max_grad) grad = -max_grad;
                data[i][j] -= learning_rate * grad;
            }
        }
    }
};

// 激活函數
float relu(float x) {
    return std::max(0.0f, x);
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Softmax
Matrix softmax(const Matrix& x) {
    Matrix result(x.rows, x.cols);
    for (int i = 0; i < x.rows; ++i) {
        float max_val = *std::max_element(x.data[i].begin(), x.data[i].end());
        float sum = 0.0f;
        
        for (int j = 0; j < x.cols; ++j) {
            result.data[i][j] = std::exp(x.data[i][j] - max_val);
            sum += result.data[i][j];
        }
        
        for (int j = 0; j < x.cols; ++j) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}

// 多層感知器類
class MLP {
private:
    Matrix W1, b1, W2, b2;
    int input_size, hidden_size, output_size;
    
public:
    MLP(int in_size, int hid_size, int out_size) 
        : input_size(in_size), hidden_size(hid_size), output_size(out_size),
          W1(hid_size, in_size), b1(hid_size, 1),
          W2(out_size, hid_size), b2(out_size, 1) {
        
        std::mt19937 rng(42);
        W1.xavier_init(rng);
        W2.xavier_init(rng);
    }
    
    // 前向傳播
    std::tuple<Matrix, Matrix, Matrix, Matrix> forward(const Matrix& X) {
        // 第一層
        Matrix z1 = W1.matmul(X);
        for (int i = 0; i < z1.rows; ++i) {
            for (int j = 0; j < z1.cols; ++j) {
                z1.data[i][j] += b1.data[i][0];
            }
        }
        
        // ReLU 激活
        Matrix a1 = z1;
        for (int i = 0; i < a1.rows; ++i) {
            for (int j = 0; j < a1.cols; ++j) {
                a1.data[i][j] = relu(a1.data[i][j]);
            }
        }
        
        // 第二層
        Matrix z2 = W2.matmul(a1);
        for (int i = 0; i < z2.rows; ++i) {
            for (int j = 0; j < z2.cols; ++j) {
                z2.data[i][j] += b2.data[i][0];
            }
        }
        
        // Softmax
        Matrix output = softmax(z2);
        
        return {z1, a1, z2, output};
    }
    
    // 反向傳播
    void backward(const Matrix& X, const Matrix& Y, float learning_rate) {
        int batch_size = X.cols;
        
        // 前向傳播
        auto [z1, a1, z2, output] = forward(X);
        
        // 計算輸出層梯度（softmax with cross-entropy 的梯度是 output - Y）
        Matrix dz2 = output;
        for (int i = 0; i < dz2.rows; ++i) {
            for (int j = 0; j < dz2.cols; ++j) {
                dz2.data[i][j] = (dz2.data[i][j] - Y.data[i][j]) / batch_size;
            }
        }
        
        // 計算 W2 和 b2 的梯度
        Matrix dW2 = dz2.matmul(a1.transpose());
        Matrix db2(output_size, 1);
        for (int i = 0; i < output_size; ++i) {
            db2.data[i][0] = 0.0f;
            for (int j = 0; j < batch_size; ++j) {
                db2.data[i][0] += dz2.data[i][j];
            }
        }
        
        // 計算隱藏層梯度
        Matrix da1 = W2.transpose().matmul(dz2);
        Matrix dz1 = da1;
        for (int i = 0; i < dz1.rows; ++i) {
            for (int j = 0; j < dz1.cols; ++j) {
                dz1.data[i][j] *= relu_derivative(z1.data[i][j]);
            }
        }
        
        // 計算 W1 和 b1 的梯度
        Matrix dW1 = dz1.matmul(X.transpose());
        Matrix db1(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            db1.data[i][0] = 0.0f;
            for (int j = 0; j < batch_size; ++j) {
                db1.data[i][0] += dz1.data[i][j];
            }
        }
        
        // 更新權重
        W1.update(dW1, learning_rate);
        b1.update(db1, learning_rate);
        W2.update(dW2, learning_rate);
        b2.update(db2, learning_rate);
    }
    
    // 計算損失和準確率
    std::pair<float, float> evaluate(const Matrix& X, const Matrix& Y) {
        auto [_, __, ___, output] = forward(X);
        
        float loss = 0.0f;
        int correct = 0;
        int batch_size = X.cols;
        
        for (int j = 0; j < batch_size; ++j) {
            // 交叉熵損失
            for (int i = 0; i < output_size; ++i) {
                if (Y.data[i][j] > 0.5f) {
                    loss -= std::log(output.data[i][j] + 1e-7f);
                }
            }
            
            // 準確率
            int pred_class = 0;
            int true_class = 0;
            float max_prob = -1.0f;
            
            for (int i = 0; i < output_size; ++i) {
                if (output.data[i][j] > max_prob) {
                    max_prob = output.data[i][j];
                    pred_class = i;
                }
                if (Y.data[i][j] > 0.5f) {
                    true_class = i;
                }
            }
            
            if (pred_class == true_class) {
                correct++;
            }
        }
        
        loss /= batch_size;
        float accuracy = (float)correct / batch_size;
        
        return {loss, accuracy};
    }
};

// 生成資料集
void generate_dataset(Matrix& X, Matrix& Y, int n_samples, int n_features, int n_classes) {
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    
    for (int i = 0; i < n_samples; ++i) {
        int label = i % n_classes;
        float angle = 2.0f * M_PI * label / n_classes;
        float radius = 2.0f;
        
        // 生成特徵
        X.data[0][i] = radius * std::cos(angle) + noise(rng);
        X.data[1][i] = radius * std::sin(angle) + noise(rng);
        for (int j = 2; j < n_features; ++j) {
            X.data[j][i] = noise(rng);
        }
        
        // 設定標籤（one-hot）
        for (int j = 0; j < n_classes; ++j) {
            Y.data[j][i] = (j == label) ? 1.0f : 0.0f;
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
    const float learning_rate = 0.001f;  // 進一步降低學習率
    const int batch_size = 32;
    
    // 生成資料
    Matrix X_train(n_features, n_samples);
    Matrix Y_train(n_classes, n_samples);
    generate_dataset(X_train, Y_train, n_samples, n_features, n_classes);
    
    // 建立模型
    MLP model(n_features, n_hidden, n_classes);
    
    // 訓練
    std::cout << "開始訓練多層感知器...\n";
    std::cout << "樣本數: " << n_samples << ", 特徵數: " << n_features 
              << ", 隱藏層大小: " << n_hidden << ", 類別數: " << n_classes << "\n";
    std::cout << "批次大小: " << batch_size << ", 學習率: " << learning_rate << "\n";
    std::cout << "----------------------------------------\n";
    
    std::mt19937 rng(123);
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        // 打亂索引
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int n_batches = (n_samples + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < n_batches; ++batch) {
            int start = batch * batch_size;
            int end = std::min(start + batch_size, n_samples);
            int actual_batch_size = end - start;
            
            // 準備批次資料
            Matrix X_batch(n_features, actual_batch_size);
            Matrix Y_batch(n_classes, actual_batch_size);
            
            for (int i = 0; i < actual_batch_size; ++i) {
                int idx = indices[start + i];
                for (int j = 0; j < n_features; ++j) {
                    X_batch.data[j][i] = X_train.data[j][idx];
                }
                for (int j = 0; j < n_classes; ++j) {
                    Y_batch.data[j][i] = Y_train.data[j][idx];
                }
            }
            
            // 訓練步驟
            model.backward(X_batch, Y_batch, learning_rate);
            
            // 評估
            auto [loss, acc] = model.evaluate(X_batch, Y_batch);
            epoch_loss += loss * actual_batch_size;
            epoch_acc += acc * actual_batch_size;
        }
        
        epoch_loss /= n_samples;
        epoch_acc /= n_samples;
        
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << ": Loss = " << std::fixed << std::setprecision(4) << epoch_loss 
                      << ", Accuracy = " << std::fixed << std::setprecision(1) << epoch_acc * 100 << "%\n";
        }
    }
    
    std::cout << "----------------------------------------\n";
    std::cout << "訓練完成！\n";
    
    return 0;
} 