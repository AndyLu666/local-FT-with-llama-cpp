# Logistic Regression 範例 (llama-logistic-regression)

此範例示範如何在 `llama.cpp` 專案中加入並編譯一個最簡單的機器學習模型 —— **邏輯回歸 (Logistic Regression)**。

特點：
1. 全部使用 ISO C++17 標準庫完成，**不依賴** 任何外部數值或深度學習函式庫。
2. 自動隨機生成 2 維可分資料集，並使用交叉熵損失 (Binary Cross-Entropy) 與 Batch Gradient Descent 進行訓練。
3. 編譯後可直接執行：
   ```bash
   ./bin/llama-logistic-regression
   ```
4. 執行過程會顯示每 100 epoch 的 Loss，以及最終訓練集準確率與學得的參數。

## 編譯

在專案根目錄：
```bash
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-logistic-regression -j
```

或是直接
```bash
cmake --build build -j
```
（若你已經建立好 `build/` 資料夾並啟用範例編譯）。

## 執行

```bash
./build/bin/llama-logistic-regression
```

你應該會看到類似以下輸出：
```
Epoch 0     Loss: 0.69314
Epoch 100   Loss: 0.20543
Epoch 200   Loss: 0.14715
...
Training accuracy: 95.4%
Learned parameters: w = [2.01, -3.02], b = 0.49
``` 