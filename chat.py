import torch
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.tokenizer import Tokenizer
import sys

def chat():
    print("\n" + "="*50)
    print("✨ LIGHTLLM INTERACTIVE CHAT ✨")
    print("="*50)
    print("Type your prompt and press Enter. Type 'quit' to exit.\n")
    print("NOTE: Since the model isn't trained yet, output will be random.")
    print("="*50 + "\n")

    # 1. Setup
    import os
    config = LightLLMConfig()
    model = LightLLM(config)
    tokenizer = Tokenizer()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LOAD CHECKPOINT (This connects the brain to the script!)
    checkpoint_path = os.path.join('out', 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print("📥 Loading trained checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Using strict=False in case minor config mismatches occurred
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"✅ Loaded checkpoint (Trained for {checkpoint.get('iter_num', '???')} steps)")
    else:
        print("⚠️ Warning: No checkpoint found in out/checkpoint.pt. Generating with random weights!")

    model.to(device)
    model.eval()

    # 2. Interaction Loop
    while True:
        try:
            prompt = input("👤 PROMPT: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
                
            if not prompt.strip():
                continue

            # FORMAT FOR INSTRUCTION TUNING
            formatted_prompt = f"User: {prompt}\nAssistant:"
            
            # Tokenize input
            input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long).to(device)
            
            print("🤖 LightLLM is thinking...", end="", flush=True)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids, 
                    max_new_tokens=60, 
                    temperature=0.2,  # Low temperature for exact math and factual accuracy
                    top_k=5
                )
            
            # Slice off the input prompt length from the generated tokens
            new_tokens = generated_ids[0][input_ids.shape[1]:]
            
            # Decode only the newly generated text
            response = tokenizer.decode(new_tokens.cpu().tolist())
            
            # Stop the response at <|endoftext|> so it doesn't hallucinate a new User prompt
            response = response.split("<|endoftext|>")[0].strip()

            # Clean up the "Thinking..." line
            print("\r" + " " * 30 + "\r", end="") 
            
            print(f"🤖 RESPONSE: {response}\n")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()
