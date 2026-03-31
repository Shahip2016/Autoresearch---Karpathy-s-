import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train import GPT, BLOCK_SIZE, DEVICE

def visualize_attention(prompt, model_path, meta_path, out_file):
    # Load meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Encode prompt
    start_ids = [stoi.get(c, stoi.get(' ', 0)) for c in prompt]
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Load model
    model = GPT().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    # Fix keys if model was compiled
    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Forward pass with attention weights
    with torch.no_grad():
        logits, _, weights = model(x, return_attn=True)
    
    # weights shape: (n_layers, batch, n_heads, seq_len, seq_len)
    weights = weights.cpu().numpy()
    n_layers, _, n_heads, t, _ = weights.shape
    
    # Create plot
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 4, n_layers * 4))
    if n_layers == 1: axes = [axes]
    
    chars = [itos[i] for i in start_ids]
    
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h] if n_layers > 1 else axes[h]
            im = ax.imshow(weights[l, 0, h], cmap='viridis')
            ax.set_xticks(range(t))
            ax.set_yticks(range(t))
            ax.set_xticklabels(chars, rotation=90)
            ax.set_yticklabels(chars)
            if l == 0:
                ax.set_title(f'Head {h+1}')
            if h == 0:
                ax.set_ylabel(f'Layer {l+1}')
    
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    print(f"Attention map saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Attention Weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--model", type=str, default="out/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--meta", type=str, default="data/meta.pkl", help="Path to meta.pkl")
    parser.add_argument("--out", type=str, default="attention_map.png", help="Output PNG file")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}. Run training first.")
    elif not os.path.exists(args.meta):
        # Try to find another meta file
        data_dir = os.path.dirname(args.meta)
        metas = [f for f in os.listdir(data_dir) if f.endswith('meta.pkl')]
        if metas:
            args.meta = os.path.join(data_dir, metas[0])
            print(f"Found meta file: {args.meta}")
            visualize_attention(args.prompt, args.model, args.meta, args.out)
        else:
            print(f"Error: Meta file not found at {args.meta}.")
    else:
        visualize_attention(args.prompt, args.model, args.meta, args.out)
