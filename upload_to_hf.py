import os
import torch
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(repo_id="RABNEER/LightLLM-124M", token=None):
    """
    Uploads trained LightLLM model weights and configuration to Hugging Face Hub.
    """
    print(f"[INFO] Uploading LightLLM weights to Hugging Face Hub ({repo_id})...")
    
    api = HfApi(token=token)
    
    # 1. Create repo on Hugging Face if it doesn't exist
    create_repo(repo_id=repo_id, exist_ok=True, token=token)
    print(f"[SUCCESS] Repository created/verified: https://huggingface.co/{repo_id}")
    
    # 2. Check checkpoint existence
    checkpoint_path = "out/checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")
        
    # 3. Upload model checkpoint
    print("[INFO] Uploading checkpoint.pt...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="pytorch_model.bin",
        repo_id=repo_id,
        token=token,
    )
    
    # 4. Upload README.md model card
    if os.path.exists("README.md"):
        print("[INFO] Uploading README.md Model Card...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        
    print(f"\n🎉 SUCCESS! Your model is live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # If HF_TOKEN environment variable is set or passed via login:
    hf_token = os.environ.get("HF_TOKEN")
    upload_to_huggingface(token=hf_token)
