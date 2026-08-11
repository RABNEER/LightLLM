"""
chat.py – Interactive Terminal Chat for LightLLM powered by StreamTransformer (STR)
Author: Ranveer Kumar
"""

import os
import sys
import argparse
import torch
from lightllm.config import LightLLMConfig
from lightllm.model import LightLLM
from lightllm.streaming import StreamTransformer
from lightllm.tokenizer import Tokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def chat():
    parser = argparse.ArgumentParser(description="LightLLM Interactive Chat")
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "monolithic"],
                        help="Execution engine: 'streaming' (StreamTransformer O(1) VRAM) or 'monolithic'")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token sampling")
    parser.add_argument("--max_tokens", type=int, default=80, help="Maximum generated tokens per reply")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" 🤖 LIGHTLLM INTERACTIVE CHAT")
    print(f" Technology: StreamTransformer (STR) Engine [Mode: {args.mode.upper()}]")
    print(" Author: Ranveer Kumar (Independent AI Researcher)")
    print("=" * 65)
    print(" Type any message to chat with your trained model. Type 'quit' to exit.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join('out', 'checkpoint.pt')

    # Load Model
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Checkpoint not found at '{checkpoint_path}'. Initializing random weights.")
        config = LightLLMConfig()
        model = StreamTransformer(config, device=device) if args.mode == "streaming" else LightLLM(config).to(device)
    else:
        if args.mode == "streaming":
            print(f"⚡ [STR ENGINE]: Loading and sharding checkpoint with O(1) Memory Engine...")
            model = StreamTransformer.from_checkpoint(checkpoint_path, shard_dir="model_shards", device=device)
            print("✅ StreamTransformer active! Peak VRAM decoupled from model depth.")
        else:
            print("📥 [MONOLITHIC]: Loading all layers directly into memory...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint.get('config', LightLLMConfig())
            model = LightLLM(config)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            print("✅ Monolithic model loaded successfully.")

    tokenizer = Tokenizer()
    print("=" * 65 + "\n")

    # Chat loop
    while True:
        try:
            prompt = input("👤 YOU: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
                
            if not prompt.strip():
                continue

            formatted_prompt = f"User: {prompt}\nAssistant:"
            input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)
            
            print(f"🤖 LightLLM [STR] is thinking...", end="", flush=True)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids, 
                    max_new_tokens=args.max_tokens, 
                    temperature=args.temp,
                    top_k=args.top_k
                )
            
            new_tokens = generated_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens.cpu().tolist())
            response = response.split("<|endoftext|>")[0].strip()
            
            print("\r" + " " * 45 + "\r", end="") 
            print(f"🤖 LIGHTLLM: {response}\n")
            print("-" * 65)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()
