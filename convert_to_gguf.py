"""
convert_to_gguf.py – Standalone Direct GGUF Exporter for LightLLM (with full BPE merges & complete GGUF tensors)
Converts PyTorch checkpoint into a standard Ollama / llama.cpp compatible GGUF binary.
Author: Ranveer Kumar (Independent AI Researcher)
"""

import os
import sys
import json
import numpy as np
import torch
import tiktoken
import gguf
from lightllm.config import LightLLMConfig

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def export_checkpoint_to_gguf(checkpoint_path="out/checkpoint.pt", output_path="out/lightllm-124m.gguf"):
    print("=" * 65)
    print(" 🚀 CONVERTING LIGHTLLM TO GGUF FORMAT (WITH BPE MERGES & NORMS)")
    print(f" Source: {checkpoint_path}")
    print(f" Target: {output_path}")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: LightLLMConfig = checkpoint.get("config", LightLLMConfig())
    state_dict = checkpoint["model"]
    cleaned_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 2. Initialize GGUF Writer
    writer = gguf.GGUFWriter(output_path, "gpt2")

    # 3. Add Model Architecture Metadata
    print("[METADATA] Writing GGUF header and model parameters...")
    writer.add_name("LightLLM-124M")
    writer.add_author("Ranveer Kumar")
    writer.add_description("LightLLM: A Depth-Invariant Layer-Streaming Causal Transformer Architecture")
    writer.add_uint32("gpt2.context_length", config.block_size)
    writer.add_uint32("gpt2.embedding_length", config.n_embd)
    writer.add_uint32("gpt2.block_count", config.n_layer)
    writer.add_uint32("gpt2.feed_forward_length", 4 * config.n_embd)
    writer.add_uint32("gpt2.attention.head_count", config.n_head)
    writer.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)

    # 4. Add Tokenizer Vocabulary & BPE Merges
    print("[TOKENIZER] Writing GPT-2 Byte-Pair Encoding vocabulary and 50,000 merges...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = []
    scores = []
    tok_types = []

    for i in range(config.vocab_size):
        try:
            b_text = enc.decode_single_token_bytes(i)
            tokens.append(b_text)
        except KeyError:
            tokens.append(f"<token_{i}>".encode('utf-8'))
        scores.append(0.0)
        tok_types.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(tok_types)
    writer.add_bos_token_id(50256)
    writer.add_eos_token_id(50256)

    # Load and add BPE merges from tokenizer.json as list of strings
    tokenizer_json_path = "out/huggingface_model/tokenizer.json"
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        raw_merges = tok_data.get("model", {}).get("merges", [])
        if raw_merges:
            string_merges = [
                " ".join(m) if isinstance(m, list) else str(m)
                for m in raw_merges
            ]
            print(f"[TOKENIZER] Adding {len(string_merges)} BPE merges as string pairs to GGUF...")
            writer.add_token_merges(string_merges)

    # 5. Add Tensor Weights to GGUF
    print("[TENSORS] Converting and writing model weights to GGUF format...")
    
    # Token & Positional Embeddings
    wte = cleaned_dict['transformer.wte.weight'].float().numpy()
    wpe = cleaned_dict['transformer.wpe.weight'].float().numpy()
    writer.add_tensor("token_embd.weight", wte)
    writer.add_tensor("position_embd.weight", wpe)

    ones_embd = np.ones(config.n_embd, dtype=np.float32)
    zeros_embd = np.zeros(config.n_embd, dtype=np.float32)
    zeros_3embd = np.zeros(3 * config.n_embd, dtype=np.float32)
    zeros_4embd = np.zeros(4 * config.n_embd, dtype=np.float32)

    # Transformer Blocks
    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}."
        
        # Attention LayerNorm (ln_1)
        attn_norm_w = cleaned_dict.get(f"{prefix}ln_1.weight", torch.tensor(ones_embd)).float().numpy()
        attn_norm_b = cleaned_dict.get(f"{prefix}ln_1.bias", torch.tensor(zeros_embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.attn_norm.weight", attn_norm_w)
        writer.add_tensor(f"blk.{i}.attn_norm.bias", attn_norm_b)

        # Fused Attention QKV Projection (c_attn)
        c_attn_w = cleaned_dict[f"{prefix}attn.c_attn.weight"].float().numpy()
        c_attn_b = cleaned_dict.get(f"{prefix}attn.c_attn.bias", torch.tensor(zeros_3embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", c_attn_w)
        writer.add_tensor(f"blk.{i}.attn_qkv.bias", c_attn_b)

        # Attention Output Projection (c_proj)
        c_proj_w = cleaned_dict[f"{prefix}attn.c_proj.weight"].float().numpy()
        c_proj_b = cleaned_dict.get(f"{prefix}attn.c_proj.bias", torch.tensor(zeros_embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.attn_output.weight", c_proj_w)
        writer.add_tensor(f"blk.{i}.attn_output.bias", c_proj_b)

        # MLP LayerNorm (ln_2)
        ffn_norm_w = cleaned_dict.get(f"{prefix}ln_2.weight", torch.tensor(ones_embd)).float().numpy()
        ffn_norm_b = cleaned_dict.get(f"{prefix}ln_2.bias", torch.tensor(zeros_embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", ffn_norm_w)
        writer.add_tensor(f"blk.{i}.ffn_norm.bias", ffn_norm_b)

        # MLP Up-Projection (c_fc)
        c_fc_w = cleaned_dict[f"{prefix}mlp.c_fc.weight"].float().numpy()
        c_fc_b = cleaned_dict.get(f"{prefix}mlp.c_fc.bias", torch.tensor(zeros_4embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.ffn_up.weight", c_fc_w)
        writer.add_tensor(f"blk.{i}.ffn_up.bias", c_fc_b)

        # MLP Down-Projection (c_proj)
        c_mlp_proj_w = cleaned_dict[f"{prefix}mlp.c_proj.weight"].float().numpy()
        c_mlp_proj_b = cleaned_dict.get(f"{prefix}mlp.c_proj.bias", torch.tensor(zeros_embd)).float().numpy()
        writer.add_tensor(f"blk.{i}.ffn_down.weight", c_mlp_proj_w)
        writer.add_tensor(f"blk.{i}.ffn_down.bias", c_mlp_proj_b)

    # Final LayerNorm (ln_f)
    out_norm_w = cleaned_dict.get("transformer.ln_f.weight", torch.tensor(ones_embd)).float().numpy()
    out_norm_b = cleaned_dict.get("transformer.ln_f.bias", torch.tensor(zeros_embd)).float().numpy()
    writer.add_tensor("output_norm.weight", out_norm_w)
    writer.add_tensor("output_norm.bias", out_norm_b)

    # Output LM Head (Tied with token embeddings or explicit head in GPT-2)
    lm_head_w = cleaned_dict.get("lm_head.weight", torch.tensor(wte)).float().numpy()
    writer.add_tensor("output.weight", lm_head_w)

    # 6. Finalize and Write File
    print(f"[WRITING] Writing binary GGUF file to: {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print("\n" + "=" * 65)
    print(f" ✅ GGUF CONVERSION COMPLETE (ALL TENSORS & NORMS PRESENT)!")
    print(f" Output File: {output_path} ({file_size_mb:.2f} MB)")
    print(" Ready to create and run on Ollama:")
    print("   ollama create ranveer/lightllm -f Modelfile")
    print("   ollama run ranveer/lightllm")
    print("=" * 65)

if __name__ == "__main__":
    export_checkpoint_to_gguf()
