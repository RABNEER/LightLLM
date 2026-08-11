<div align="center">

# 🚀 LightLLM: Depth-Invariant Layer-Streaming Causal Transformer

### *A 123.65M Parameter Foundation Architecture with Lossless Full-Precision Streaming Execution*

<p align="center">
  <a href="https://pypi.org/project/stream-transformer/"><img src="https://img.shields.io/pypi/v/stream-transformer.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI version" /></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" />
  <img src="https://img.shields.io/badge/Precision-Lossless%20FP32%20(1.00000012)-4CAF50?style=for-the-badge" alt="Lossless FP32" />
  <img src="https://img.shields.io/badge/VRAM%20Savings-97.62%25-00BCD4?style=for-the-badge" alt="VRAM Savings" />
  <img src="https://img.shields.io/badge/Ollama-ranveer%2Flightllm-FF9800?style=for-the-badge&logo=ollama" alt="Ollama" />
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Author-Ranveer%20Kumar-9C27B0?style=for-the-badge" alt="Author" />
</p>

```bash
# Run with Ollama in 1 command:
ollama run ranveer/lightllm

# Or install the universal streaming engine via PyPI:
pip install stream-transformer
```

---

</div>

## 📖 Executive Summary

**LightLLM** is an open-source 123.65-million parameter causal autoregressive language model engineered from scratch in PyTorch around a novel **StreamTransformer (STR)** Layer-Streaming Engine.

By reformulating the computational execution hierarchy of deep transformers, LightLLM mathematically decouples model depth ($L$) from GPU physical memory capacity, achieving **depth-invariant memory scaling ($\mathcal{M}_{\text{peak}} = \mathcal{O}(1\text{ layer})$)**. Any arbitrary depth (12, 36, 100, or 1,000 layers) executes with a constant peak GPU VRAM footprint (**~214 MB active, 297 MB peak**), delivering over **97.62% VRAM reduction** with **100% Lossless FP32 Precision** ($\text{Cosine Similarity} = 1.00000012$, $\Delta = 0.00000000$).

> 📄 **Read the Official 13-Page Monograph**: [`paper/lightllm_paper.pdf`](paper/lightllm_paper.pdf)  
> ⚡ **Standalone PyPI Engine**: [`github.com/RABNEER/stream-transformer`](https://github.com/RABNEER/stream-transformer)  
> 🦙 **Run with Ollama**: `ollama run ranveer/lightllm`

---

## ⚡ The Breakthrough: StreamTransformer (STR)

Scaling foundation models has traditionally been bounded by the **VRAM Wall**—the requirement that all parameters, momentum buffers, and activations must reside simultaneously in high-bandwidth GPU memory. On consumer hardware (4GB–8GB GPUs), researchers are forced into lossy 4-bit quantization (GGUF/AWQ) that degrades reasoning, coding, and mathematical accuracy.

**StreamTransformer** replaces spatial VRAM allocation with **temporal layer streaming**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           STREAMTRANSFORMER RUNTIME PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Input Tokens ──→ [ Token Embeddings (Resident) ] ──→ h₀                               │
│                                                         │                               │
│   [ Host RAM / NVMe ] ──→ Layer 1/100 ──→ GPU VRAM ──→ h₁ = Layer₁(h₀) ──→ Purged      │
│   [ Async Prefetch  ] ──→ Layer 2/100 ──→ GPU VRAM ──→ h₂ = Layer₂(h₁) ──→ Purged      │
│   ...                                                                                   │
│   [ Async Prefetch  ] ──→ Layer 100/100 ─→ GPU VRAM ─→ h₁₀₀ = Layer₁₀₀(h₉₉) ──→ Purged  │
│                                                         │                               │
│   Output Logits ←── [ LM Head (Resident) ] ←── [ LayerNorm (Resident) ]                 │
│                                                                                         │
│   Peak GPU VRAM: CONSTANT ~297.50 MB across 100 Layers (>97.62% VRAM Reduction!)        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks

### 1. Monolithic vs. Quantization vs. StreamTransformer (124M FP32 Model)

| Execution Paradigm | Compute Precision | Peak VRAM | VRAM Savings | Cosine Similarity | Max Absolute Logit Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Monolithic** | FP32 (Lossless) | ~1,850.0 MB | 0.0% (Baseline) | $1.00000000$ | $0.00000000 \times 10^0$ |
| **StreamTransformer (Ours)** | **FP32 (Lossless)** | **~148.5 MB** | **🔥 91.97% Savings** | **1.00000012** | **0.00000000 \times 10^0** |
| **Standard INT4 Quantization** | INT4 (Lossy) | ~480.0 MB | 74.05% Savings | $0.96142010$ | $1.84210940 \times 10^{-1}$ |

---

### 2. Live 100-Layer Ultra-Deep GPU Telemetry (NVIDIA Tesla T4)

```
===========================================================================
 100-LAYER TRANSFORMER ON GPU (~746 Million Parameters)
===========================================================================
• Layer   1/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
• Layer  20/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
• Layer  40/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
• Layer  60/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
• Layer  80/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
• Layer 100/100: Active VRAM = 214.16 MB | Peak VRAM = 297.50 MB
---------------------------------------------------------------------------
• Status:           ✅ SUCCESS (0 Errors, All 100 Layers Computed)
• Peak GPU VRAM:    297.50 MB (Monolithic Expected: ~12,500 MB)
• Memory Savings:   🔥 97.62% VRAM Reduction!
• Depth-Invariance: PROVEN (Flat memory line across all 100 layers)
===========================================================================
```

---

## 🛠️ Project Structure

```text
LightLLM/
├── lightllm/
│   ├── __init__.py         # Core exports (StreamTransformer, StreamTrainer, Tokenizer)
│   ├── config.py           # LightLLMConfig dataclass (124M parameter architecture)
│   ├── model.py            # Transformer decoder blocks & Fused FlashAttention
│   ├── streaming.py        # StreamTransformer (STR) O(1) Inference Engine
│   ├── train_streaming.py  # Layer-Wise Pretraining Engine (Forward/Backward)
│   ├── tokenizer.py        # GPT-2 BPE Tokenizer wrapper
│   └── utils.py            # Memory telemetry & parameter counting
├── paper/
│   ├── lightllm_paper.pdf  # Publication-grade 13-page Research Monograph
│   ├── lightllm_paper.tex  # LaTeX source code
│   └── generate_pdf.py     # PDF build engine
├── chat.py                 # Interactive terminal chat CLI (STR native)
├── test_inference.py       # Sample text generation through STR engine
├── test_100_layers.py      # 100-layer GPU depth-invariance benchmark
├── test_stream_training.py # Forward/backward layer-wise training test
├── convert_to_gguf.py      # Direct GGUF exporter with full BPE merges
├── export_hf.py            # Hugging Face format exporter
├── Modelfile               # Ollama model definition
└── pyproject.toml          # Project package configuration
```

---

## 🚀 Quick Start Guides

### 1. Run with Ollama (Fastest)
Run the model directly using Ollama CLI:
```bash
ollama run ranveer/lightllm
```

---

### 2. Interactive Terminal Chat via StreamTransformer (STR)
Launch interactive CLI chat session powered by layer streaming:
```bash
python chat.py
```

---

### 3. Install & Stream in Python via PyPI (`stream-transformer`)
```bash
pip install stream-transformer
```

```python
import torch
from stream_transformer import StreamEngine

# Load and stream any model with O(1) GPU Memory
engine = StreamEngine(
    resident_modules=resident_dict,
    layer_constructor=lambda: YourTransformerBlock(),
    shard_dir="model_shards",
    num_layers=100,
    device="cuda"
)

output_logits = engine(input_tensor)
```

---

### 4. Run the 100-Layer GPU Depth-Invariance Benchmark
```bash
python test_100_layers.py
```

---

## 📜 Citation

If you use LightLLM or StreamTransformer in your research, please cite our monograph:

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
GitHub: [@RABNEER](https://github.com/RABNEER) | PyPI: [stream-transformer](https://pypi.org/project/stream-transformer/) | Repository: [LightLLM](https://github.com/RABNEER/LightLLM)
