# 🚀 LightLLM

A lightweight, custom **124M Parameter Decoder-Only Causal Transformer LLM** built completely from scratch in PyTorch, featuring PyTorch 2.0 FlashAttention, FP16 Automatic Mixed Precision (AMP), and GPU acceleration.

---

## ✨ Features

- 🧠 **Decoder-Only Transformer Architecture:** 12 Layers, 12 Attention Heads, 768 Embedding Dimension (~124M Parameters).
- ⚡ **GPU Accelerated Training:** PyTorch FP16 AMP Mixed Precision (`torch.amp.autocast`) + FlashAttention (`F.scaled_dot_product_attention`) running on CUDA GPUs.
- 💬 **Instruction Tuning & Interactive Chat:** Interactive CLI interface ([chat.py](file:///d:/LightLLM/chat.py)) for real-time inference with temperature and top-k sampling.
- 📊 **Custom Data Pipeline:** Binary memory-mapped data pipeline ([prepare_data.py](file:///d:/LightLLM/prepare_data.py)) supporting instruction datasets and custom Q&A injection.
- 🎯 **Optimized Memory Usage:** Train 100M+ parameter models using ~2.8 GB VRAM on consumer GPUs (e.g. NVIDIA RTX 4050).

---

## 🛠️ Project Structure

```text
LightLLM/
├── lightllm/
│   ├── config.py       # LightLLMConfig (124M parameter architecture)
│   ├── model.py        # Transformer, FlashAttention & MLP blocks
│   ├── tokenizer.py    # GPT-2 BPE Tokenizer wrapper
│   └── utils.py        # Helper utilities & parameter formatting
├── prepare_data.py     # Download, format & tokenize dataset to binary maps
├── train.py            # High-performance PyTorch AMP training loop
├── chat.py             # Interactive CLI chat interface
├── check_model.py      # Model parameter counter & diagnostics
└── pyproject.toml      # Project dependencies & package configuration
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/RABNEER/LightLLM.git
cd LightLLM
uv venv .venv-gpu --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --python .venv-gpu
uv pip install numpy tiktoken tqdm --python .venv-gpu
```

### 2. Prepare Dataset
Tokenize dataset to binary memory maps (`train.bin` & `val.bin`):
```bash
.venv-gpu\Scripts\python.exe prepare_data.py
```

### 3. Train Model on GPU
Launch PyTorch FP16 AMP training on GPU:
```bash
.venv-gpu\Scripts\python.exe train.py
```

### 4. Chat with Model
Interactive Q&A chat session:
```bash
.venv-gpu\Scripts\python.exe chat.py
```

---

## 👨‍💻 Author
Built with PyTorch by **RABNEER**.
