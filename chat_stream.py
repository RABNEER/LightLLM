import os
import sys
import time
import torch
from lightllm.streaming import StreamTransformer
from lightllm.tokenizer import Tokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def chat_stream():
    print("\n" + "=" * 65)
    print(" 🚀 LIGHTLLM INTERACTIVE CHAT [STREAMTRANSFORMER ENGINE (STR)]")
    print(" Architecture: Depth-Invariant O(1) VRAM Layer-Streaming")
    print(" Author: Ranveer Kumar")
    print("=" * 65)
    print(" Type any prompt/question. Type 'quit' or 'exit' to leave.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join('out', 'checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at '{checkpoint_path}'.")
        return

    # 1. Initialize StreamTransformer Engine directly from Checkpoint
    print("⚡ Initializing StreamTransformer from trained checkpoint...")
    stream_model = StreamTransformer.from_checkpoint(
        checkpoint_path, 
        shard_dir="model_shards", 
        device=device,
        prefetch=(device == 'cuda')
    )
    tokenizer = Tokenizer()
    print("=" * 65 + "\n")

    # 2. Interactive Chat Loop
    while True:
        try:
            prompt = input("👤 YOU: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
                
            if not prompt.strip():
                continue

            # Format for instruction tuning
            formatted_prompt = f"User: {prompt}\nAssistant:"
            input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)
            
            print("🤖 StreamTransformer is streaming layers...", end="", flush=True)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = stream_model.generate(
                    input_ids, 
                    max_new_tokens=60, 
                    temperature=0.7,
                    top_k=40
                )
            gen_duration = (time.perf_counter() - t0)
            
            new_tokens = generated_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens.cpu().tolist())
            response = response.split("<|endoftext|>")[0].strip()
            
            # Clean up line
            print("\r" + " " * 45 + "\r", end="") 
            print(f"🤖 STR RESPONSE ({len(new_tokens)} tokens in {gen_duration:.2f}s):\n{response}\n")
            print("-" * 65)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat_stream()
