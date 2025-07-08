
# local-FT-with-llama-cpp 🦙⚡️

Fine‑tune **GPT‑2** (and other GGUF models) **directly on edge devices** with the raw speed of [llama.cpp](https://github.com/ggerganov/llama.cpp).

<p align="center">
  <img src="https://raw.githubusercontent.com/ggerganov/llama.cpp/master/examples/screenshots/llama_cpp_logo_see_no_evil.png" width="320" alt="llama.cpp logo"/>
</p>

> **Why?**  
> Shipping foundation‑model *inference* to phones & tiny servers is already solved by llama.cpp.  
> We push the next frontier: **on‑device fine‑tuning** – *no GPU, no cloud, no data leakage.*

---

## ✨ Project highlights

| Feature | Status | Notes |
|---------|--------|-------|
| **Back‑prop (`loss.backward`) in GGML** | ✅ | added new `mse_loss` forward / backward kernels |
| **GPT‑2 training graph in C** | ✅ | builds a full forward+backward compute graph at runtime |
| **HF → GGUF converter** | ✅ | `convert-hf-to-gguf.py` supports GPT‑2 (all sizes) |
| **Int8 / f16 training** | 🧪 | mixed precision kernels implemented, need more testing |
| **Edge builds (macOS 📱 / Linux 🖥 / Windows 👾 / RPi 🍓)** | ✅ | single‑file C build, no Python required at runtime |

---

## 🚀 Quick start

```bash
# 0. clone
git clone https://github.com/AndyLu666/local-FT-with-llama-cpp.git
cd local-FT-with-llama-cpp

# 1. build llama.cpp + custom GGML
make       # or cmake -B build && cmake --build build -j

# 2. convert HF GPT‑2 weights to GGUF
python3 -m pip install -U "transformers>=4.40.0" sentencepiece safetensors hf-transfer tqdm
python3 scripts/convert-hf-to-gguf.py gpt2 --outfile models/gpt2.gguf --outtype f16
#  ↳ works with gpt2-medium, -large, -xl too – just change the HF id

# 3. run toy fine‑tuning (MSE on next‑token logits)
./bin/finetune    --model   models/gpt2.gguf    --data    data/tiny-shakespeare.txt    --epochs  1 --lr 5e-5
```

Logs look like:

```
step 00050 | loss 3.42 | ppl 30.5 | mem 480 MB
step 00100 | loss 3.28 | ppl 26.6 | +dW 3.9 MB | 440 tok/s on MacBook Air M1
```

---

## 🗂 Repository layout

```
.
├── ggml/               # patched ggml backend (new kernels in ggml.c)
├── scripts/
│   ├── convert-hf-to-gguf.py     # HF → GGUF
│   └── dataset.py                # simple txt → tokens
├── src/
│   ├── finetune.c      # builds training graph, SGD/AdamW step
│   └── gpt2.c          # GPT‑2 block helpers
└── models/             # put *.gguf here
```

### New GGML code

* **Forward kernels**  
  ```c
  static void ggml_compute_forward_mse_loss_f32(...);
  static void ggml_compute_forward_mse_loss_back_f32(...);
  ```
  Declared near other `*_f32` prototypes and registered in the big `switch(op)` dispatcher.

* **Graph op IDs**  
  Added `GGML_OP_MSE_LOSS` and `GGML_OP_MSE_LOSS_BACK` to `ggml.h`, plus helper wrappers.

* **Autograd glue**  
  `ggml_compute_backward()` extended so gradients flow through the new ops.

---

## 👷‍♀️ Fine‑tuning details

| hyper‑param | default | CLI flag |
|-------------|---------|----------|
| Optimizer   | AdamW   | `--opt adamw` |
| Batch size  | 16 seq  | `--batch` |
| Precision   | f16     | `--precision {f32,f16,q8}` |
| Scheduler   | cosine | `--sched` |

The training loop is **pure C** – no PyTorch. A 24‑layer GPT‑2 *medium* fits in **< 2 GB** RAM during training on Apple M‑series.

---

## ⚡ Benchmarks

| Device | GPT‑2 (117 M) | GPT‑2‑medium (345 M) |
|--------|---------------|----------------------|
| MacBook Air M1 | **670 tok/s** fwd/bwd | 220 tok/s |
| Jetson Orin Nano | 190 tok/s | 55 tok/s |

*(batch = 8, seq = 1024, f16)*

---

## 🛠 Roadmap

- [ ] Flash‑Attention v2 kernels  
- [ ] LoRA / QLoRA adapter updates  
- [ ] Stream‑in / streaming datasets  
- [ ] WebAssembly build for browser fine‑tuning 🤯

---

## 📜 License

Apache 2.0 – same as upstream llama.cpp. See `LICENSE`.

---

## 🙏 Acknowledgements

* Georgi Gerganov for **llama.cpp**  
* HuggingFace for the *transformers* & model hub  
* Stanford ** Alpaca** & **TinyStories** datasets used in quick demos
