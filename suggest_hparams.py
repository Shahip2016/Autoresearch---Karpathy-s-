import torch
from train import GPT, N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, VOCAB_SIZE

def suggest_hparams(vram_gb=4.0):
    print(f"\n{'='*60}")
    print(f" HYPERPARAMETER SUGGESTER (Target: {vram_gb}GB VRAM)")
    print(f"{'='*60}")
    
    model = GPT()
    params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory
    # Model weights (FP32 = 4 bytes, actual training often uses mixed precision but gradients/optimizer states are FP32)
    # Adam optimizer: 8 bytes per parameter (m and v)
    # Gradients: 4 bytes per parameter
    # Total per parameter: 4 + 8 + 4 = 16 bytes
    model_mem_mb = (params * 16) / (1024 * 1024)
    
    # Activation memory (rough estimate for transformer)
    # B * T * D * Layers * 2 (approx for SwiGLU + Attention)
    # Let's target a safe 70% VRAM usage for training to avoid OOM
    target_vram_mb = vram_gb * 1024 * 0.7 
    available_for_activations = target_vram_mb - model_mem_mb
    
    # Simple heuristic for batch size
    # Activation size per sample (MB)
    act_per_sample_mb = (BLOCK_SIZE * N_EMBD * N_LAYER * 2 * 4) / (1024 * 1024)
    
    suggested_batch = max(1, int(available_for_activations / act_per_sample_mb))
    # Round down to nearest power of 2 for efficiency
    if suggested_batch >= 16:
        suggested_batch = (suggested_batch // 16) * 16
    elif suggested_batch >= 8:
        suggested_batch = 8
    elif suggested_batch >= 4:
        suggested_batch = 4
    
    # Effective batch size target should be around 64-128 for stability
    target_eff_batch = 64
    suggested_grad_accum = max(1, target_eff_batch // suggested_batch)
    
    # LR Scaling (Square root scaling is a common heuristic)
    base_lr = 3e-4
    base_batch = 64
    eff_batch = suggested_batch * suggested_grad_accum
    suggested_lr = base_lr * (eff_batch / base_batch)**0.5

    print(f"Model Parameters: {params:,}")
    print(f"Estimated Model Memory: {model_mem_mb:.1f} MB")
    print(f"Available for Activations: {available_for_activations:.1f} MB")
    print(f"\nRecommended Configuration:")
    print(f"  --batch-size           {suggested_batch}")
    print(f"  GRAD_ACCUM_STEPS       {suggested_grad_accum}")
    print(f"  Effective Batch Size   {eff_batch}")
    print(f"  --learning-rate        {suggested_lr:.2e}")
    print(f"{'='*60}")
    
    print("\nTo use these, run:")
    print(f"python train.py --batch-size {suggested_batch} --learning-rate {suggested_lr:.2e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vram', type=float, default=4.0, help='Available VRAM in GB')
    args = parser.parse_args()
    suggest_hparams(args.vram)
