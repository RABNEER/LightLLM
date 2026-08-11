"""
convert_to_gguf.py – Standalone Direct GGUF Exporter for LightLLM
Converts PyTorch checkpoint into a standard Ollama / llama.cpp compatible GGUF binary.
Author: Ranveer Kumar (Independent AI Researcher)
"""

import os
import sys
import numpy as np
import torch
import tiktoken
import gguf
from lightllm.config import LightLLMConfig

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def export_checkpoint_to_gguf(checkpoint_path="out/checkpoint.pt", output_path="out/lightllm-124m.gguf"):
    print("=" * 65)
    print(" 🚀 CONVERTING LIGHTLLM TO GGUF FORMAT (FOR OLLAMA)")
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

    # 4. Add Tokenizer Vocabulary
    print("[TOKENIZER] Writing GPT-2 Byte-Pair Encoding vocabulary...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = []
    scores = []
    tok_types = []

    # Populate vocabulary
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

    # 5. Add Tensor Weights to GGUF
    print("[TENSORS] Converting and writing model weights to GGUF format...")
    
    # Token & Positional Embeddings
    wte = cleaned_dict['transformer.wte.weight'].float().numpy()
    wpe = cleaned_dict['transformer.wpe.weight'].float().numpy()
    writer.add_tensor("token_embd.weight", wte)
    writer.add_tensor("position_embd.weight", wpe)

    # Transformer Blocks
    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}."
        
        # Attention LayerNorm (ln_1)
        if f"{prefix}ln_1.weight" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.attn_norm.weight", cleaned_dict[f"{prefix}ln_1.weight"].float().numpy())
        if f"{prefix}ln_1.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.attn_norm.bias", cleaned_dict[f"{prefix}ln_1.bias"].float().numpy())

        # Fused Attention QKV Projection (c_attn)
        # Note: Conv1D/Linear weight in PyTorch is [in_features, out_features]
        c_attn_w = cleaned_dict[f"{prefix}attn.c_attn.weight"].float().numpy()
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", c_attn_w)
        if f"{prefix}attn.c_attn.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.attn_qkv.bias", cleaned_dict[f"{prefix}attn.c_attn.bias"].float().numpy())

        # Attention Output Projection (c_proj)
        c_proj_w = cleaned_dict[f"{prefix}attn.c_proj.weight"].float().numpy()
        writer.add_tensor(f"blk.{i}.attn_output.weight", c_proj_w)
        if f"{prefix}attn.c_proj.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.attn_output.bias", cleaned_dict[f"{prefix}attn.c_proj.bias"].float().numpy())

        # MLP LayerNorm (ln_2)
        if f"{prefix}ln_2.weight" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.ffn_norm.weight", cleaned_dict[f"{prefix}ln_2.weight"].float().numpy())
        if f"{prefix}ln_2.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.ffn_norm.bias", cleaned_dict[f"{prefix}ln_2.bias"].float().numpy())

        # MLP Up-Projection (c_fc)
        c_fc_w = cleaned_dict[f"{prefix}mlp.c_fc.weight"].float().numpy()
        writer.add_tensor(f"blk.{i}.ffn_up.weight", c_fc_w)
        if f"{prefix}mlp.c_fc.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.ffn_up.bias", cleaned_dict[f"{prefix}mlp.c_fc.bias"].float().numpy())

        # MLP Down-Projection (c_proj)
        c_mlp_proj_w = cleaned_dict[f"{prefix}mlp.c_proj.weight"].float().numpy()
        writer.add_tensor(f"blk.{i}.ffn_down.weight", c_mlp_proj_w)
        if f"{prefix}mlp.c_proj.bias" in cleaned_dict:
            writer.add_tensor(f"blk.{i}.ffn_down.bias", cleaned_dict[f"{prefix}mlp.c_proj.bias"].float().numpy())

    # Final LayerNorm (ln_f)
    if "transformer.ln_f.weight" in cleaned_dict:
        writer.add_tensor("output_norm.weight", cleaned_dict["transformer.ln_f.weight"].float().numpy())
    if "transformer.ln_f.bias" in cleaned_dict:
        writer.add_tensor("output_norm.bias", cleaned_dict["transformer.ln_f.bias"].float().numpy())

    # Output LM Head (Tied with token embeddings in GPT-2)
    writer.add_tensor("output.weight", wte)

    # 6. Finalize and Write File
    print(f"[WRITING] Writing binary GGUF file to: {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print("\n" + "=" * 65)
    print(f" ✅ GGUF CONVERSION COMPLETE!")
    print(f" Output File: {output_path} ({file_size_mb:.2f} MB)")
    print(" Ready to create and run on Ollama:")
    print("   ollama create rabneer/lightllm -f Modelfile")
    print("   ollama run rabneer/lightllm")
    print("=" * 65)

if __name__ == "__main__":
    export_checkpoint_to_gguf()
