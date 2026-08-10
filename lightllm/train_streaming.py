"""
lightllm/train_streaming.py – Layer-Wise Streaming Training Engine with CPU Activation Stashing
Author: Ranveer Kumar (Independent AI Researcher)

Implements O(1) VRAM Pretraining & Fine-Tuning:
1. Forward Pass: Streams layers 0 -> L-1, stashing boundary activations (h_l) to CPU Host RAM.
2. Backward Pass: Streams layers L-1 -> 0 in reverse, recomputing activations on-the-fly,
   computing layer gradients, executing in-place fused AdamW updates, and immediately evicting VRAM.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightllm.config import LightLLMConfig
from lightllm.model import Block

class StreamTrainer:
    def __init__(self, config: LightLLMConfig, shard_dir: str = "train_shards", device: str = "cpu", lr: float = 6e-4):
        self.config = config
        self.shard_dir = shard_dir
        self.device = device
        self.lr = lr
        os.makedirs(shard_dir, exist_ok=True)
        
        # 1. Resident Parameters in GPU/Device Memory
        self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(device)
        self.wpe = nn.Embedding(config.block_size, config.n_embd).to(device)
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias).to(device)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(device)
        
        # Weight tying
        self.lm_head.weight = self.wte.weight
        
        # Resident optimizer for embeddings and final LayerNorm
        resident_params = list(self.wte.parameters()) + list(self.wpe.parameters()) + list(self.ln_f.parameters())
        self.resident_optimizer = torch.optim.AdamW(resident_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        
        # 2. Initialize Layer Shards on Disk/RAM and AdamW States per Layer
        self.adam_states = {} # Layer-wise momentum buffers stored on CPU
        self._init_layer_shards()

    def _init_layer_shards(self):
        """Initializes individual layer parameter files and CPU AdamW buffers."""
        for l in range(self.config.n_layer):
            shard_path = os.path.join(self.shard_dir, f"train_layer_{l}.pt")
            if not os.path.exists(shard_path):
                block = Block(self.config)
                torch.save(block.state_dict(), shard_path)
            
            # Initialize CPU momentum buffers for each layer (m and v = 0)
            if l not in self.adam_states:
                dummy_block = Block(self.config)
                self.adam_states[l] = {
                    'step': 0,
                    'exp_avg': {name: torch.zeros_like(param) for name, param in dummy_block.named_parameters()},
                    'exp_avg_sq': {name: torch.zeros_like(param) for name, param in dummy_block.named_parameters()},
                }

    def _adamw_layer_step(self, block: Block, layer_idx: int, lr: float, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        """Executes an in-place AdamW update for a single layer on GPU and syncs buffers to CPU."""
        state = self.adam_states[layer_idx]
        state['step'] += 1
        step = state['step']
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        with torch.no_grad():
            for name, param in block.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                
                # Apply decoupled weight decay
                if weight_decay != 0 and param.dim() >= 2:
                    param.data.mul_(1.0 - lr * weight_decay)
                
                # Load momentum from CPU to GPU
                exp_avg = state['exp_avg'][name].to(param.device)
                exp_avg_sq = state['exp_avg_sq'][name].to(param.device)
                
                # Update biased 1st and 2nd moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Save updated momentum back to CPU RAM
                state['exp_avg'][name].copy_(exp_avg.cpu())
                state['exp_avg_sq'][name].copy_(exp_avg_sq.cpu())
                del exp_avg, exp_avg_sq

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        """
        Executes one full streaming forward-backward pretraining step.
        Peak GPU VRAM is strictly O(1 layer).
        """
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        self.resident_optimizer.zero_grad()
        
        # -------------------------------------------------------------
        # 1. FORWARD PASS WITH CPU BOUNDARY ACTIVATION STASHING
        # -------------------------------------------------------------
        cpu_stashes = [] # Stashes boundary activations h_l to CPU RAM
        
        # Resident Embedding Forward
        h_0 = self.wte(x) + self.wpe(pos)
        current_h = h_0
        
        for l in range(self.config.n_layer):
            # Save boundary input to CPU System RAM
            cpu_stashes.append(current_h.detach().cpu())
            
            # Load Block l into GPU VRAM
            shard_path = os.path.join(self.shard_dir, f"train_layer_{l}.pt")
            block_state = torch.load(shard_path, map_location=self.device, weights_only=False)
            block = Block(self.config).to(self.device)
            block.load_state_dict(block_state)
            block.eval()
            
            # Compute layer output
            with torch.no_grad():
                current_h = block(current_h)
                
            # Evict Block l from GPU VRAM immediately
            del block, block_state
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
        # Resident LayerNorm and LM Head Forward (with gradient tracking)
        h_final = current_h.detach().requires_grad_(True)
        normed = self.ln_f(h_final)
        logits = self.lm_head(normed)
        
        # Cross-Entropy Loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss_val = loss.item()
        
        # Backprop through resident head and final LayerNorm
        loss.backward()
        delta_incoming = h_final.grad # Incoming gradient dL/dh_final
        
        # -------------------------------------------------------------
        # 2. REVERSE BACKWARD PASS WITH ON-THE-FLY RECOMPUTATION
        # -------------------------------------------------------------
        for l in reversed(range(self.config.n_layer)):
            shard_path = os.path.join(self.shard_dir, f"train_layer_{l}.pt")
            block_state = torch.load(shard_path, map_location=self.device, weights_only=False)
            block = Block(self.config).to(self.device)
            block.load_state_dict(block_state)
            block.train()
            
            # Fetch stashed boundary from CPU RAM into GPU
            h_in = cpu_stashes[l].to(self.device).requires_grad_(True)
            
            # Recompute layer forward pass to build micro-autograd graph
            h_out = block(h_in)
            
            # Backward pass through this single layer
            h_out.backward(delta_incoming)
            delta_incoming = h_in.grad.detach() # Incoming gradient for layer l-1
            
            # Apply In-Place AdamW Step directly on this layer
            self._adamw_layer_step(block, layer_idx=l, lr=self.lr)
            
            # Save updated weights back to shard file on disk
            torch.save(block.state_dict(), shard_path)
            
            # Evict layer, gradients, and micro-graph from GPU VRAM immediately
            del block, block_state, h_in, h_out
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
        # -------------------------------------------------------------
        # 3. BACKPROP THROUGH RESIDENT EMBEDDINGS
        # -------------------------------------------------------------
        h_0.backward(delta_incoming)
        self.resident_optimizer.step()
        
        return loss_val
