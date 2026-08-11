"""
test_inference.py – StreamTransformer (STR) Inference Verification
Author: Ranveer Kumar
"""

import os
import sys
import torch
from lightllm.streaming import StreamTransformer
from lightllm.tokenizer import Tokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def test_inference():
    print("=" * 65)
    print(" 🚀 LIGHTLLM: STREAMTRANSFORMER (STR) INFERENCE TEST")
    print(" Technology: Layer-Streaming O(1) Memory Engine")
    print(" Author: Ranveer Kumar (Independent AI Researcher)")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE]: {device.upper()}")
    
    checkpoint_path = "out/checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return
        
    # Initialize StreamTransformer from checkpoint
    print(f"\n[STREAMING]: Loading model through StreamTransformer engine...")
    model = StreamTransformer.from_checkpoint(checkpoint_path, shard_dir="model_shards", device=device)
    tokenizer = Tokenizer()
    
    prompts = [
        "Artificial Intelligence is",
        "The purpose of education is",
        "In a faraway galaxy",
    ]
    
    print("\n" + "=" * 65)
    print(" GENERATING TEXT SAMPLES THROUGH STREAMTRANSFORMER (STR)")
    print("=" * 65)
    
    for prompt in prompts:
        formatted = f"User: {prompt}\nAssistant:"
        input_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=40)
            
        new_tokens = generated_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens.cpu().tolist())
        output_text = output_text.split("<|endoftext|>")[0].strip()
        
        print(f"\n[PROMPT]: '{prompt}'")
        print(f"[STR GENERATION]:\n{output_text}")
        print("-" * 65)

if __name__ == "__main__":
    test_inference()
