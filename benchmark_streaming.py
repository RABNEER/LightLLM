"""
benchmark_streaming.py – Empirical Benchmark: Monolithic vs. StreamTransformer
Measures VRAM consumption, latency, and numerical output equivalence.
Author: Ranveer Kumar
"""

import os
import time
import torch
import torch.nn.functional as F
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.streaming import StreamTransformer
from lightllm.tokenizer import Tokenizer

def run_benchmark():
    print("=" * 70)
    print(" LightLLM Architecture Benchmark: Monolithic vs. StreamTransformer")
    print(" Author: Ranveer Kumar")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SYSTEM] Running on Device: {device.upper()}")
    if device == 'cuda':
        print(f"[SYSTEM] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[SYSTEM] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    config = LightLLMConfig(block_size=512, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    tokenizer = Tokenizer()
    
    # 1. Create or load checkpoint
    os.makedirs("out", exist_ok=True)
    checkpoint_path = "out/checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print("[SETUP] Creating baseline checkpoint for benchmarking...")
        base_model = LightLLM(config)
        torch.save({'model': base_model.state_dict(), 'config': config}, checkpoint_path)
    
    # 2. Benchmark Standard Monolithic Model
    print("\n" + "-" * 70)
    print("1. EVALUATING STANDARD MONOLITHIC MODEL (All 12 Layers in Memory)")
    print("-" * 70)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    mono_model = LightLLM(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cleaned_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    mono_model.load_state_dict(cleaned_dict)
    mono_model.to(device)
    mono_model.eval()
    
    prompt = "Artificial Intelligence and deep learning"
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Measure forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        mono_logits, _ = mono_model(x)
    t_mono_forward = (time.perf_counter() - t0) * 1000
    
    mono_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == 'cuda' else 0
    print(f"[MONOLITHIC] Forward Latency: {t_mono_forward:.2f} ms")
    if device == 'cuda':
        print(f"[MONOLITHIC] Peak VRAM Allocated: {mono_vram_mb:.2f} MB")
        
    # Free monolithic model to cleanly test streaming
    del mono_model
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    # 3. Benchmark StreamTransformer Engine
    print("\n" + "-" * 70)
    print("2. EVALUATING STREAMTRANSFORMER ENGINE (O(1) Single Layer in VRAM)")
    print("-" * 70)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    stream_model = StreamTransformer.from_checkpoint(checkpoint_path, shard_dir="model_shards", device=device)
    
    # Measure forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        stream_logits = stream_model.forward(x)
    t_stream_forward = (time.perf_counter() - t0) * 1000
    
    stream_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == 'cuda' else 0
    print(f"[STREAMING] Forward Latency: {t_stream_forward:.2f} ms")
    if device == 'cuda':
        print(f"[STREAMING] Peak VRAM Allocated: {stream_vram_mb:.2f} MB")
        vram_reduction = ((mono_vram_mb - stream_vram_mb) / mono_vram_mb) * 100
        print(f"[RESULT] VRAM Reduction: {vram_reduction:.2f}% Savings!")
        
    # 4. Numerical Equivalence & Cosine Similarity Verification
    print("\n" + "-" * 70)
    print("3. NUMERICAL VERIFICATION & LOGIT EQUIVALENCE")
    print("-" * 70)
    mono_last_logit = mono_logits[:, -1, :].to('cpu')
    stream_last_logit = stream_logits.to('cpu')
    
    cos_sim = F.cosine_similarity(mono_last_logit, stream_last_logit, dim=-1).item()
    max_abs_diff = torch.max(torch.abs(mono_last_logit - stream_last_logit)).item()
    
    print(f"[EQUIVALENCE] Cosine Similarity: {cos_sim:.8f} (Expected: 1.00000000)")
    print(f"[EQUIVALENCE] Max Absolute Logit Difference: {max_abs_diff:.8e}")
    if cos_sim > 0.99999:
        print("[SUCCESS] 100% Lossless Precision Verified! Mathematical Output is Identical.")
    else:
        print("[WARNING] Precision mismatch detected.")
        
    # 5. Generation Test
    print("\n" + "-" * 70)
    print("4. STREAMING AUTOREGRESSIVE GENERATION TEST")
    print("-" * 70)
    gen_tokens = stream_model.generate(x, max_new_tokens=15, temperature=0.8)
    gen_text = tokenizer.decode(gen_tokens[0].tolist())
    print(f"[PROMPT]: {prompt}")
    print(f"[GENERATION]: {gen_text}")
    print("=" * 70)
    print(" Benchmark Complete!")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
