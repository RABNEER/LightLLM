"""
lightllm/streaming.py – Production Layer-Streaming Engine for LightLLM
Decouples model depth (L) from GPU VRAM, achieving O(1) memory scaling.
Author: Ranveer Kumar
"""

import os
import time
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from lightllm.model import Block
from lightllm.config import LightLLMConfig

class StreamTransformer:
    """
    Native Layer-Streaming Transformer Engine for LightLLM.
    Keeps only embeddings and active layer in GPU VRAM, achieving O(1) depth memory scaling.
    """
    def __init__(self, config: LightLLMConfig, shard_dir: str = "model_shards", device: str = None, prefetch: bool = True):
        self.config = config
        self.shard_dir = shard_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prefetch = prefetch and (self.device == 'cuda')
        os.makedirs(shard_dir, exist_ok=True)
        
        # Resident modules: Token & positional embeddings and final norm
        self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(self.device)
        self.wpe = nn.Embedding(config.block_size, config.n_embd).to(self.device)
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias).to(self.device)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(self.device)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Thread pool for asynchronous PCIe prefetching
        self.executor = ThreadPoolExecutor(max_workers=1) if self.prefetch else None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, shard_dir: str = "model_shards", device: str = None, prefetch: bool = True):
        """
        Shards a monolithic checkpoint into discrete layer files and initializes the streaming engine.
        """
        print(f"[STREAMING] Initializing StreamTransformer from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        state_dict = checkpoint['model']
        
        # Strip DataParallel 'module.' prefix if present
        cleaned_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        instance = cls(config, shard_dir, device, prefetch=prefetch)
        
        # Load resident embeddings
        instance.wte.weight.data.copy_(cleaned_dict['transformer.wte.weight'])
        instance.wpe.weight.data.copy_(cleaned_dict['transformer.wpe.weight'])
        if config.bias and instance.ln_f.weight is not None:
            instance.ln_f.weight.data.copy_(cleaned_dict['transformer.ln_f.weight'])
            instance.ln_f.bias.data.copy_(cleaned_dict['transformer.ln_f.bias'])
        
        # Shard each transformer block individually to disk
        print(f"[STREAMING] Sharding {config.n_layer} transformer layers to '{shard_dir}/'...")
        for i in range(config.n_layer):
            block = Block(config)
            block_state = {}
            prefix = f'transformer.h.{i}.'
            for k, v in cleaned_dict.items():
                if k.startswith(prefix):
                    block_state[k[len(prefix):]] = v
            block.load_state_dict(block_state)
            
            shard_path = os.path.join(shard_dir, f"layer_{i}.pt")
            torch.save(block.state_dict(), shard_path)
            
        print(f"[STREAMING] Successfully sharded {config.n_layer} layers. O(1) Memory Engine active!")
        return instance

    def _load_layer_state(self, layer_idx: int):
        shard_path = os.path.join(self.shard_dir, f"layer_{layer_idx}.pt")
        state = torch.load(shard_path, map_location='cpu', weights_only=False)
        if self.device == 'cuda' and torch.cuda.is_available():
            for k in state.keys():
                state[k] = state[k].pin_memory()
        return state

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing one transformer block at a time.
        GPU VRAM remains strictly constant regardless of depth L.
        """
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        
        # 1. Resident input embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        # 2. Sequential Layer Streaming with Optional Prefetching
        prefetch_future = None
        if self.prefetch and self.config.n_layer > 1:
            prefetch_future = self.executor.submit(self._load_layer_state, 1)
            
        for i in range(self.config.n_layer):
            if self.prefetch and i > 0 and prefetch_future is not None:
                block_state = prefetch_future.result()
                prefetch_future = None
            else:
                block_state = self._load_layer_state(i)
                
            # Launch prefetch of the next layer
            if self.prefetch and (i + 1 < self.config.n_layer):
                prefetch_future = self.executor.submit(self._load_layer_state, i + 1)
                
            # Materialize block on running device
            block = Block(self.config).to(self.device)
            # Transfer state dict to device
            device_state = {k: v.to(self.device, non_blocking=True) for k, v in block_state.items()}
            block.load_state_dict(device_state)
            block.eval()
            
            with torch.no_grad():
                x = block(x)
                
            # Immediate VRAM eviction
            del block
            del block_state
            del device_state
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
        # 3. Final normalization & output logits
        x = self.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive text generation using native layer streaming.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
