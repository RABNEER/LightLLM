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
    
    # --- A. Inject Explicit Basic Greetings, Facts, Coding & Math Data ---
    custom_qa = [
        # Basic Greetings
        ("hello", "Hello! How are you doing today?"),
        ("Hello", "Hello! How are you?"),
        ("hi", "Hi there! How can I help you today?"),
        ("Hi", "Hello! How can I help you?"),
        ("how are you", "I am doing great! How are you?"),
        ("How are you?", "I am doing great! How can I assist you today?"),
        ("what is your name", "My name is LightLLM, a custom AI model."),
        ("What is your name?", "I am LightLLM, a 124M parameter neural network model!"),
        ("who created you", "I was created and trained from scratch by RABNEER!"),
        ("Who created LightLLM?", "LightLLM was created and trained from scratch by RABNEER!"),

        # General Knowledge Facts
        ("what is an apple", "An apple is a sweet, round edible fruit produced by an apple tree."),
        ("what is apple", "An apple is a sweet edible fruit produced by an apple tree."),
        ("What is an apple?", "An apple is a sweet, edible fruit produced by an apple tree."),
        ("what is python", "Python is a popular high-level programming language known for readability and powerful AI libraries."),
        ("what is an llm", "An LLM is a Large Language Model built using Transformer neural network architectures."),
        ("what is ai", "AI stands for Artificial Intelligence, enabling computers to perform tasks like learning, reasoning, and speech."),
        ("what is gravity", "Gravity is a fundamental force of nature that pulls objects with mass toward one another."),
        ("what is the capital of france", "The capital of France is Paris."),
        ("what is 2+2", "4"),
        ("what is 5+5", "10"),

        # Basic Math Addition, Subtraction & Multiplication
        ("2+2", "4"),
        ("2 + 2", "4"),
        ("5+5", "10"),
        ("5 + 5", "10"),
        ("10+10", "20"),
        ("10 + 10", "20"),
    ]
    
    # Generate full addition table (0..30)
    for a in range(31):
        for b in range(31):
            custom_qa.append((f"{a}+{b}", f"{a+b}"))
            custom_qa.append((f"{a} + {b}", f"{a+b}"))

    # Generate full multiplication table (0..15)
    for a in range(16):
        for b in range(16):
            custom_qa.append((f"{a}*{b}", f"{a*b}"))
            custom_qa.append((f"{a} * {b}", f"{a*b}"))
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
