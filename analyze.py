"""
analyze.py — Experiment Analysis Tool.
Parses results.tsv and provides a summary of all experiments,
including best runs, trends, and improvement statistics.
"""

import os
import sys

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.tsv')

def load_results():
    """Load and parse results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first!")
        return []

    results = []
    with open(RESULTS_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                results.append({
                    'run': line_num,
                    'timestamp': parts[0],
                    'val_loss': float(parts[1]),
                    'val_bpb': float(parts[2]),
                    'params': int(parts[3]) if len(parts) > 3 else None,
                })
    return results

def analyze():
    """Print a comprehensive analysis of experiment results."""
    results = load_results()
    if not results:
        return

    print("=" * 70)
    print("  AutoResearch — Experiment Analysis")
    print("=" * 70)
    print(f"\n  Total experiments: {len(results)}\n")

    # Table header
    print(f"  {'Run':>4}  {'Timestamp':<20}  {'Val Loss':>10}  {'Val BPB':>10}  {'Params':>12}  {'Status'}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")

    best_bpb = float('inf')
    best_run = None
    improvements = 0

    for r in results:
        is_best = r['val_bpb'] < best_bpb
        if is_best:
            best_bpb = r['val_bpb']
            best_run = r
            improvements += 1
            status = "★ BEST"
        else:
            status = ""

        params_str = f"{r['params']:,}" if r['params'] else "N/A"
        print(f"  {r['run']:>4}  {r['timestamp']:<20}  {r['val_loss']:>10.6f}  {r['val_bpb']:>10.6f}  {params_str:>12}  {status}")

    # Summary
    print(f"\n{'─' * 70}")
    print(f"\n  Summary:")
    print(f"    Best val_bpb:    {best_bpb:.6f} (Run #{best_run['run']})")
    print(f"    Worst val_bpb:   {max(r['val_bpb'] for r in results):.6f}")
    print(f"    Improvements:    {improvements}/{len(results)} runs")

    if len(results) >= 2:
        first_bpb = results[0]['val_bpb']
        total_improvement = first_bpb - best_bpb
        pct_improvement = (total_improvement / first_bpb) * 100
        print(f"    Total improvement: {total_improvement:.6f} ({pct_improvement:.2f}%)")

    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    analyze()
