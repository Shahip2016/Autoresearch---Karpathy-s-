"""
profile.py — VRAM Profiler
Profiles the currently configured model in `train.py` to estimate memory
requirements, check if it fits within the 4GB budget, and perform a peak
memory runtime test if CUDA is available.

Usage:
    python profile.py
    python profile.py --batch-size 32
    python profile.py --block-size 1024
"""

import os
import sys
import argparse
import inspect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
# Import directly from train.py to ensure we profile the exact architecture
import train as t


def format_bytes(b):
    """Format bytes into MB/GB string."""
    mb = b / (1024 ** 2)
    if mb > 1000:
        return f"{b / (1024 ** 3):.2f} GB"
    return f"{mb:.1f} MB"


def patch_train_hyperparams(new_bs, new_block, new_accum):
    """Dynamically overwrite hyperparameters in the loaded `train` module."""
    if new_bs is not None:
        t.BATCH_SIZE = new_bs
    if new_block is not None:
        t.BLOCK_SIZE = new_block
    if new_accum is not None:
        t.GRAD_ACCUM_STEPS = new_accum


def profile_static_memory():
    """Calculate theoretical memory requirements (Model + Optimizer + Gradients)."""
    model = t.GPT()
    param_count = model.count_parameters()
    
    # 1. Model Weights
    # If AMP is on, model stays in FP32 usually, and casts to FP16 in forward.
    # We assume FP32 weights for safety (4 bytes/param).
    bytes_per_param = 4
    model_bytes = param_count * bytes_per_param
    
    # 2. Gradients
    # gradients are FP32
    grad_bytes = param_count * bytes_per_param
    
    # 3. Optimizer State (AdamW)
    # Adam needs 2 states per param (momentum, variance) = 8 bytes/param
    opt_bytes = param_count * 8
    
    # 4. Total Static
    static_bytes = model_bytes + grad_bytes + opt_bytes
    
    print("\n--- Static Memory Profile ---")
    print(f"Parameters:        {param_count:,}")
    print(f"Model Weights:     {format_bytes(model_bytes)}")
    print(f"Gradients:         {format_bytes(grad_bytes)}")
    print(f"Optimizer (Adam):  {format_bytes(opt_bytes)}")
    print("-" * 29)
    print(f"Total Static:      {format_bytes(static_bytes)}")
    
    return param_count, static_bytes


def profile_runtime_memory():
    """Run a dummy forward/backward pass to measure actual VRAM via torch.cuda."""
    if not torch.cuda.is_available():
        print("\n--- Runtime Memory Profile ---")
        print("CUDA not available. Cannot measure real VRAM usage.")
        return None

    print("\n--- Runtime Memory Profile (CUDA) ---")
    
    # Reset stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    baseline = torch.cuda.memory_allocated()
    
    # Move model to device
    model = t.GPT().to('cuda')
    
    # Setup dummy optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=t.USE_AMP)
    
    # Dummy data
    X = torch.randint(0, t.VOCAB_SIZE, (t.BATCH_SIZE, t.BLOCK_SIZE)).to('cuda')
    Y = torch.randint(0, t.VOCAB_SIZE, (t.BATCH_SIZE, t.BLOCK_SIZE)).to('cuda')
    
    # Forward + Backward
    with torch.amp.autocast('cuda', enabled=t.USE_AMP, dtype=t.DTYPE):
        logits, loss = model(X, Y)
    
    scaler.scale(loss).backward()
    
    # Optimizer step (initializes Adam states)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak VRAM during FW/BW: {format_bytes(peak_bytes)}")
    
    # Clean up so we don't hold on to memory if called iteratively
    del model, optimizer, scaler, X, Y, logits, loss
    torch.cuda.empty_cache()
    
    return peak_bytes


def main():
    parser = argparse.ArgumentParser(description="AutoResearch VRAM Profiler")
    parser.add_argument('--batch-size', type=int, default=None, help='Override BATCH_SIZE (micro batch)')
    parser.add_argument('--block-size', type=int, default=None, help='Override BLOCK_SIZE (context window)')
    parser.add_argument('--accum-steps', type=int, default=None, help='Override GRAD_ACCUM_STEPS')
    args = parser.parse_args()

    patch_train_hyperparams(args.batch_size, args.block_size, args.accum_steps)

    print("=" * 60)
    print(" AutoResearch VRAM Profiler")
    print("=" * 60)
    
    print("\n[Current Configuration]")
    print(f"Device:           {t.DEVICE}")
    print(f"Mixed Precision:  {'Enabled (FP16/BF16)' if t.USE_AMP else 'Disabled (FP32)'}")
    print(f"Model: {t.N_LAYER} layers, {t.N_HEAD} heads, {t.N_EMBD} embd")
    print(f"Micro Batch:      {t.BATCH_SIZE}")
    print(f"Context (Block):  {t.BLOCK_SIZE}")
    print(f"Grad Accum steps: {t.GRAD_ACCUM_STEPS}")

    # 1. Static estimation
    _, static_bytes = profile_static_memory()

    # 2. Runtime measurement
    peak_vram = profile_runtime_memory()
    
    # 3. Budget Check (4GB)
    budget_bytes = 4 * 1024 * 1024 * 1024
    
    print("\n" + "=" * 60)
    print(" Verdict: 4GB GPU Budget Check")
    print("=" * 60)
    
    if peak_vram is not None:
        total = peak_vram
        source_str = f"Measure ({format_bytes(peak_vram)})"
    else:
        # If no cuda, guess activation memory (very rough heuristic: 2x static)
        # Sequence len * batch size * embd size * layers is the real math, but
        # this is just a fallback.
        total = static_bytes * 2.5
        source_str = f"Estimate ({format_bytes(static_bytes)} static + ~activations)"
    
    pct = (total / budget_bytes) * 100
    
    print(f"Memory: {source_str}")
    print(f"Budget: {format_bytes(budget_bytes)}")
    print(f"Usage:  {pct:.1f}%")
    
    if total <= budget_bytes:
        print("\n✅ PASS: This model should fit in 4GB VRAM.")
    else:
        print("\n❌ FAIL: This model will likely Out-Of-Memory (OOM) on a 4GB GPU.")
        print("   Consider reducing --batch-size, --block-size, N_LAYER, or N_EMBD.")

if __name__ == "__main__":
    main()
