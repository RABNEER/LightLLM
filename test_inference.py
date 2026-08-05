import torch
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.tokenizer import Tokenizer

def test_inference():
    print("🚀 Initializing LightLLM for inference test...")
    
    # 1. Setup
    config = LightLLMConfig()
    model = LightLLM(config)
    tokenizer = Tokenizer()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set to evaluation mode
    
    # 2. Prepare Input
    prompt = "The future of AI is"
    print(f"\n🔹 Input Prompt: '{prompt}'")
    
    # Encode prompt to tokens
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # 3. Generate
    print("🔹 Generating 20 tokens (expecting gibberish since untrained)...")
    with torch.no_grad():
        # Temperature=1.0 for random sampling
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=1.0)
    
    # 4. Decode and Print
    output_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\n✅ Generation Successful!")
    print("-" * 30)
    print(f"Output: {output_text}")
    print("-" * 30)
    print("\nNote: The output is random because the model weights are initialized to random values. Once you train it on a dataset, it will generate meaningful sentences!")

if __name__ == "__main__":
    test_inference()
