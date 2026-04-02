"""
benchmark.py — Performance Benchmarking Suite
Measures throughput (tokens/sec) and latency of the AutoResearch GPT model.
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F

# Add script dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import GPT, DEVICE, BLOCK_SIZE, DTYPE, USE_AMP, VOCAB_SIZE

def benchmark_throughput(model, batch_size=16, seq_len=512, iters=10):
    """Measures tokens per second throughput."""
    # Warmup
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(DEVICE)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(3):
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=DTYPE):
                model(x, y)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()
        
    # Main benchmark loop
    for _ in range(iters):
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=DTYPE):
            model(x, y)
            
    if DEVICE == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    else:
        elapsed_time = time.time() - start_time
        
    total_tokens = batch_size * seq_len * iters
    tokens_per_sec = total_tokens / elapsed_time
    latency_per_iter = (elapsed_time / iters) * 1000.0  # ms
    
    return tokens_per_sec, latency_per_iter

def main():
    print("=" * 60)
    print("  AutoResearch Performance Benchmarking")
    print("=" * 60)
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Mixed Precision: {USE_AMP}")
    print(f"[*] Model Configuration: {DTYPE}")
    
    # Initialize model
    print("[*] Initializing model...")
    model = GPT().to(DEVICE)
    model.eval()
    
    param_count = model.count_parameters()
    print(f"[+] Model parameters: {param_count:,}")
    
    batch_sizes = [1, 4, 8, 16]
    seq_lengths = [128, 256, 512]
    
    results = {
        "metadata": {
            "device": DEVICE,
            "param_count": param_count,
            "amp_enabled": USE_AMP
        },
        "trials": []
    }
    
    print("-" * 60)
    print(f"{'Batch':<10} | {'Seq Len':<10} | {'Tokens/Sec':<15} | {'Latency (ms)':<15}")
    print("-" * 60)
    
    for bs in batch_sizes:
        for sl in seq_lengths:
            try:
                # Use smaller iterations for CPU to not take too long
                iters = 20 if DEVICE == 'cuda' else 5
                tps, lat = benchmark_throughput(model, batch_size=bs, seq_len=sl, iters=iters)
                
                print(f"{bs:<10} | {sl:<10} | {tps:<15.0f} | {lat:<15.2f}")
                
                results["trials"].append({
                    "batch_size": bs,
                    "seq_len": sl,
                    "tokens_per_sec": tps,
                    "latency_ms": lat
                })
            except Exception as e:
                print(f"{bs:<10} | {sl:<10} | FAILED ({e})")
                
    # Save results
    output_path = os.path.join(SCRIPT_DIR, 'out', 'benchmark_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("-" * 60)
    print(f"[+] Benchmark results saved to {output_path}")

if __name__ == "__main__":
    main()
