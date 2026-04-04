import torch
import torch.nn as nn
import argparse
import sys
from train import GPT, N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, VOCAB_SIZE

def visualize_model(save_path=None):
    if save_path:
        original_stdout = sys.stdout
        sys.stdout = open(save_path, 'w')
        
    print(f"\n{'='*60}")
    print(" GPT ARCHITECTURE VISUALIZER")
    print(f"{'='*60}")
    
    model = GPT()
    
    print(f"Model Configuration:")
    print(f"  Vocab Size:    {VOCAB_SIZE}")
    print(f"  Block Size:    {BLOCK_SIZE}")
    print(f"  Embed Dim:     {N_EMBD}")
    print(f"  Layers:        {N_LAYER}")
    print(f"  Heads:         {N_HEAD}")
    print(f"  Head Dim:      {N_EMBD // N_HEAD}")
    print(f"{'='*60}")

    total_params = 0
    
    # Header
    print(f"{'Layer (type)':<30} | {'Parameters':>15}")
    print("-" * 48)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:<30} | {num_params:>15,}")
            
    print("-" * 48)
    print(f"{'TOTAL':<30} | {total_params:>15,}")
    print(f"{'='*60}")

    # Generate Mermaid snippet
    print("\nMermaid Architecture Diagram:")
    print("```mermaid")
    print("graph TD")
    print("  Input([Input Tokens]) --> TokenEmb[Token Embedding]")
    print(f"  TokenEmb --> BlockLoop{{x{N_LAYER} Blocks}}")
    print("  subgraph TransformerBlock [Transformer Block]")
    print("    direction TB")
    print("    LN1[RMSNorm 1] --> Attn[Causal Self-Attention]")
    print("    Attn --> Add1((+))")
    print("    Add1 --> LN2[RMSNorm 2]")
    print("    LN2 --> MLP[SwiGLU MLP]")
    print("    MLP --> Add2((+))")
    print("  end")
    print("  BlockLoop --> TransformerBlock")
    print("  TransformerBlock --> LN_F[Final RMSNorm]")
    print("  LN_F --> Head[LM Head]")
    print("  Head --> Output([Next Token Logits])")
    print("```")

    if save_path:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GPT Architecture")
    parser.add_argument("--save", type=str, default="", help="Optional path to save standard output")
    args = parser.parse_args()
    
    visualize_model(args.save)
