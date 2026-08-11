<div align="center">

# 🚀 LightLLM

### *A Lightweight 124M Parameter Decoder-Only Causal Transformer Language Model Built from Scratch in PyTorch*

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" />
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20Causal-blue?style=for-the-badge" alt="Decoder-Only" />
  <img src="https://img.shields.io/badge/Parameters-123.65M-success?style=for-the-badge" alt="124M Params" />
  <img src="https://img.shields.io/badge/Ollama-ranveer%2Flightllm-FF9800?style=for-the-badge&logo=ollama" alt="Ollama" />
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Author-Ranveer%20Kumar-9C27B0?style=for-the-badge" alt="Author" />
</p>

```bash
# Run and chat with LightLLM via Ollama in 1 command:
ollama run ranveer/lightllm
```

---

</div>

## 📖 Overview

**LightLLM** is an open-source, 123.65-million parameter causal autoregressive language model engineered completely from scratch in PyTorch. 

Designed for high efficiency on consumer hardware, LightLLM incorporates **PyTorch 2.0 FlashAttention**, **FP16 Automatic Mixed Precision (AMP)**, and a custom binary memory-mapped data pipeline. Pretrained for 50,000 optimization iterations down to **`0.0162` cross-entropy loss**, LightLLM delivers coherent English instruction following and conversational capabilities with a lightweight footprint (~2.8 GB VRAM for full training).

> 📄 **Official 13-Page Research Paper**: [`paper/lightllm_paper.pdf`](paper/lightllm_paper.pdf)  
> 🦙 **Run with Ollama**: `ollama run ranveer/lightllm`

---

## ✨ Architectural Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **123,652,608** (~124M) | Full FP32 / FP16 precision weights |
| **Layers ($L$)** | **12** | Transformer Decoder Blocks |
| **Attention Heads ($H$)** | **12** | Multi-Head Self-Attention ($d_{head} = 64$) |
| **Embedding Dimension ($d_{model}$)** | **768** | Latent hidden representation size |
| **Context Length ($T_{max}$)** | **512 / 1024** | Maximum token sequence length |
| **Vocabulary Size ($V$)** | **50,257** | GPT-2 Byte-Pair Encoding (BPE) |
| **Attention Mechanism** | **Fused FlashAttention** | `torch.nn.functional.scaled_dot_product_attention` |
| **Normalization** | **RMSNorm / LayerNorm** | Pre-Layer Normalization ($\epsilon = 10^{-5}$) |
| **Activation Function** | **GELU (Approximate)** | Gaussian Error Linear Unit in MLP |

---

## 🛠️ Project Structure

```text
LightLLM/
├── lightllm/
│   ├── __init__.py         # Package exports & model constructors
│   ├── config.py           # LightLLMConfig dataclass
│   ├── model.py            # Transformer decoder blocks & Fused FlashAttention
│   ├── tokenizer.py        # GPT-2 BPE Tokenizer wrapper
│   └── utils.py            # Memory telemetry & parameter counter
├── paper/
│   ├── lightllm_paper.pdf  # Publication-grade 13-Page Research Monograph
│   ├── lightllm_paper.tex  # LaTeX source code
│   └── generate_pdf.py     # PDF build engine
├── prepare_data.py         # Binary memory-mapped dataset preparation
├── train.py                # High-performance PyTorch FP16 AMP training loop
├── chat.py                 # Interactive terminal chat CLI
├── test_inference.py       # Standalone generation & prompt testing
├── convert_to_gguf.py      # Standalone GGUF exporter (with BPE merges)
├── export_hf.py            # Hugging Face format exporter
├── Modelfile               # Ollama model definition
└── pyproject.toml          # Package configuration
```

---

## 🚀 Quick Start

### 1. Run with Ollama (Fastest)
Run the model directly using Ollama CLI:
```bash
ollama run ranveer/lightllm
```

---

### 2. Local Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/RABNEER/LightLLM.git
cd LightLLM
pip install -e .
```

---

### 3. Interactive Terminal Chat
Launch the interactive terminal chat session:
```bash
python chat.py
```

---

### 4. Prepare Dataset & Train from Scratch

#### Step A: Prepare Dataset
Tokenize text data into memory-mapped binary files (`train.bin` & `val.bin`):
```bash
python prepare_data.py
```

#### Step B: Launch GPU Pretraining
Launch the PyTorch FP16 AMP accelerated training loop:
```bash
python train.py
```

---

### 5. Python API Usage

```python
import torch
from lightllm import LightLLM, LightLLMConfig, Tokenizer

# 1. Initialize tokenizer and model
tokenizer = Tokenizer()
model = LightLLM(LightLLMConfig())

# 2. Load trained weights
checkpoint = torch.load("out/checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# 3. Generate text
prompt = "User: Explain the theory of relativity in simple terms.\nAssistant:"
tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

with torch.no_grad():
    output_tokens = model.generate(tokens, max_new_tokens=100, temperature=0.7)

print(tokenizer.decode(output_tokens[0].tolist()))
```

---

## 📜 Citation

If you use LightLLM in your research, please cite our monograph:

```bibtex
@article{kumar2026lightllm,
  title={LightLLM: A Depth-Invariant Layer-Streaming Causal Transformer Architecture for Lossless Full-Precision Pretraining and Inference on Constrained Hardware},
  author={Kumar, Ranveer},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/RABNEER/LightLLM}
}
```

---

## 👨‍💻 Author & Maintainer
**Ranveer Kumar**  
*Independent AI Researcher*  
GitHub: [@RABNEER](https://github.com/RABNEER) | Repository: [LightLLM](https://github.com/RABNEER/LightLLM)
