import os
import time
import math
import numpy as np
import torch
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig

# -----------------------------------------------------------------------------
# Configuration
batch_size = 8        # FP16 AMP batch size for 124M model (~1.8GB VRAM used)
max_iters = 5000      # 5,000 steps (~3-4 minutes on GPU)
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0       # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True       # whether to decay the learning rate
warmup_iters = 200    # how many steps to warm up for
lr_decay_iters = 5000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5         # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system setup
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Auto-select CUDA GPU if available, fallback to CPU
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
out_dir = 'out'
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loader (Simple Memory Map)
def get_batch(split):
    data = np.memmap(f'{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Initialization
config = LightLLMConfig()
model = LightLLM(config).to(device)

# Multi-GPU support (Dual T4 on Kaggle = 32GB total VRAM!)
if device == 'cuda' and torch.cuda.device_count() > 1:
    print(f"[INFO] Multi-GPU Mode: Utilizing all {torch.cuda.device_count()} GPUs (Dual T4 DataParallel)!")
    model = torch.nn.DataParallel(model)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# learning rate schedule (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# FP16 Automatic Mixed Precision (AMP) for GPU memory optimization & Tensor Core acceleration
scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

# -----------------------------------------------------------------------------
# Training Loop
print(f"[INFO] Starting training on {device} (AMP Mixed Precision enabled)...")
iter_num = 0
best_val_loss = 1e9

t0 = time.time()
while iter_num <= max_iters:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % 100 == 0:
        model.eval()
        with torch.no_grad():
            losses = torch.zeros(10)
            for k in range(10):
                X, Y = get_batch('val')
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                    logits, loss = model(X, Y)
                    if loss.dim() > 0:
                        loss = loss.mean()
                losses[k] = loss.item()
            val_loss = losses.mean()
            print(f"step {iter_num}: val loss {val_loss:.4f}, lr {lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    raw_model = model.module if hasattr(model, 'module') else model
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"[SAVE] Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))
        model.train()

    # forward backward update with Mixed Precision
    X, Y = get_batch('train')
    
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
        logits, loss = model(X, Y)
        if loss.dim() > 0:
            loss = loss.mean() # Average loss across Dual T4 GPUs
    
    scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    scaler.step(optimizer)
    scaler.update()

    # timing and logging
    if iter_num % 10 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1

# Always save final model checkpoint at completion (unwrapped for single-GPU inference)
raw_model = model.module if hasattr(model, 'module') else model
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'iter_num': iter_num - 1,
    'best_val_loss': best_val_loss,
}
print(f"[SAVE] Saving final trained model checkpoint to {out_dir}/checkpoint.pt...")
torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))

print("\n[SUCCESS] Training complete!")
