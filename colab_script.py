# LightLLM Training Notebook for Google Colab
# Run this notebook on a Google Colab GPU (T4 / A100) instance!

# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Clone repository & install dependencies
!git clone https://github.com/RABNEER/LightLLM.git
%cd LightLLM
!pip install torch numpy tiktoken tqdm

# Cell 3: Prepare Dataset (with custom arithmetic & greetings)
!python prepare_data.py

# Cell 4: Train LightLLM on Colab GPU
!python train.py

# Cell 5: Interactive Generation Test
import torch
from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.tokenizer import Tokenizer

config = LightLLMConfig()
model = LightLLM(config)
tokenizer = Tokenizer()
checkpoint = torch.load('out/checkpoint.pt', map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=False)
model.to('cuda').eval()

def generate_answer(prompt):
    formatted = f"User: {prompt}\nAssistant:"
    ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long).to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, temperature=0.2, top_k=5)
    return tokenizer.decode(out[0].tolist()).split('<|endoftext|>')[0]

print("Q: hello ->", generate_answer("hello"))
print("Q: 2+2 ->", generate_answer("2+2"))
print("Q: 5+5 ->", generate_answer("5+5"))
