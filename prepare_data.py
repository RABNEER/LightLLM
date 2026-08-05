import os
import urllib.request
import json
import numpy as np
from lightllm.tokenizer import Tokenizer

def prepare():
    print("[INFO] Preparing instruction-tuning data for LightLLM...")
    
    # 1. Download Alpaca dataset
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    filepath = "alpaca_data.json"
    if not os.path.exists(filepath):
        print("[INFO] Downloading Alpaca instruction dataset (might take a moment)...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print("[INFO] Dataset already downloaded.")
        
    # 2. Parse JSON and Format
    print("[INFO] Formatting instructions...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_text = ""
    
    # --- A. Inject Explicit Basic Greetings & Math Data ---
    custom_qa = [
        # Basic Greetings
        ("hello", "Hello! How are you doing today?"),
        ("Hello", "Hello! How are you?"),
        ("hi", "Hi there! How can I help you today?"),
        ("Hi", "Hello! How can I help you?"),
        ("how are you", "I am doing great! How are you?"),
        ("How are you?", "I am doing great! How can I assist you today?"),
        ("what is your name", "My name is LightLLM, a custom AI model."),
        ("What is your name?", "I am LightLLM, a 10M parameter neural network model!"),
        
        # Arithmetic / Basic Math
        ("2+2", "4"),
        ("2 + 2", "4"),
        ("What is 2+2?", "4"),
        ("What is 2 + 2?", "4"),
        ("2+3", "5"),
        ("2 + 3", "5"),
        ("5+5", "10"),
        ("5 + 5", "10"),
        ("10+10", "20"),
        ("10 + 10", "20"),
        ("1+1", "2"),
        ("1 + 1", "2"),
        ("3+3", "6"),
        ("3 + 3", "6"),
        ("4+4", "8"),
        ("4 + 4", "8"),
    ]
    
    # Generate full addition table (0..15) repeated to make it prominent
    for a in range(16):
        for b in range(16):
            custom_qa.append((f"{a}+{b}", f"{a+b}"))
            custom_qa.append((f"{a} + {b}", f"{a+b}"))
            custom_qa.append((f"What is {a} + {b}?", f"The answer is {a+b}."))

    # Repeat custom Q&A to ensure high frequency in training data
    for _ in range(15):
        for user_q, bot_a in custom_qa:
            formatted_text += f"User: {user_q}\nAssistant: {bot_a}<|endoftext|>\n\n"
    
    # --- B. General Alpaca Instructions ---
    subset = data[:3000]
    for item in subset:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            instruction = f"{instruction}\nInput: {input_text}"
            
        formatted_text += f"User: {instruction}\nAssistant: {output}<|endoftext|>\n\n"
        
    # 3. Tokenize
    print("[INFO] Tokenizing dataset...")
    tokenizer = Tokenizer()
    
    # Encode all text
    all_tokens = tokenizer.encode(formatted_text, allowed_special={'<|endoftext|>'})

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # 4. Save to binary files
    n = len(all_tokens)
    train_data = all_tokens[:int(n*0.9)]
    val_data = all_tokens[int(n*0.9):]
    
    print(f"[INFO] Saving to train.bin ({len(train_data)} tokens) and val.bin ({len(val_data)} tokens)...")
    train_data.tofile('train.bin')
    val_data.tofile('val.bin')
    
    print("\n[SUCCESS] Instruction data preparation complete!")

if __name__ == "__main__":
    prepare()
