import os
import sys
import torch
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.tokenizer import Tokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def test_inference():
    print("=" * 60)
    print(" [LIGHTLLM] INFERENCE DEMO WITH TRAINED KAGGLE CHECKPOINT")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE]: {device.upper()}")
    
    # 1. Load Checkpoint
    checkpoint_path = "out/checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return
        
    print(f"[LOADING]: Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', LightLLMConfig())
    model = LightLLM(config)
    
    # Handle state_dict if trained under DataParallel
    state_dict = checkpoint['model']
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            unwrapped_state_dict[k[7:]] = v
        else:
            unwrapped_state_dict[k] = v
            
    model.load_state_dict(unwrapped_state_dict, strict=False)
    model.to(device)
    model.eval()
    tokenizer = Tokenizer()
    
    print("[STATUS]: Model loaded successfully!")
    if 'best_val_loss' in checkpoint:
        print(f"[CHECKPOINT METRIC]: Best Val Loss = {checkpoint['best_val_loss']}")
        
    # 2. Test Prompts
    prompts = [
        "Artificial Intelligence is",
        "The purpose of education is",
        "In a faraway galaxy",
    ]
    
    print("\n" + "=" * 60)
    print(" GENERATING TEXT SAMPLES FROM TRAINED MODEL")
    print("=" * 60)
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.8, top_k=40)
        output_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n[PROMPT]: '{prompt}'")
        print(f"[GENERATION]:\n{output_text}")
        print("-" * 60)

if __name__ == "__main__":
    test_inference()
