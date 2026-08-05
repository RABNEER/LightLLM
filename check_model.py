from lightllm.model import LightLLM
from lightllm.config import LightLLMConfig
from lightllm.utils import format_params

def main():
    config = LightLLMConfig()
    model = LightLLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print("-" * 30)
    print(f"Model Configuration: {config}")
    print("-" * 30)
    print(f"Total Parameters:      {format_params(total_params)}")
    print(f"Trainable Parameters:  {format_params(trainable_params)}")
    print(f"Non-Embedding Params:  {format_params(non_emb_params)}")
    print("-" * 30)
    
    if total_params >= 9.5e6 and total_params <= 11e6:
        print("[SUCCESS] LightLLM is in the target 10M range!")
    else:
        print(f"[WARNING] Model size is {format_params(total_params)}, aim for ~10M.")

if __name__ == "__main__":
    main()
