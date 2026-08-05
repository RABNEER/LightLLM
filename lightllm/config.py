from dataclasses import dataclass

@dataclass
class LightLLMConfig:
    block_size: int = 512  # context window size (512 tokens)
    vocab_size: int = 50257 # GPT-2 vocab size
    n_layer: int = 12      # 12 Transformer layers (~124M parameters)
    n_head: int = 12       # 12 Attention heads
    n_embd: int = 768      # 768 Embedding dimension (GPT-2 Small)
    dropout: float = 0.0   # 0.0 for faster GPU training
    bias: bool = False     # False for fast compute
