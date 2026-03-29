import os
import shutil
import glob

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.tsv')
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
BEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best')

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
                    'iter': int(parts[4]) if len(parts) > 4 else None,
                })
    return results

def track_best_model(best_run):
    """Archive the code and weights of the best performing model."""
    if not best_run:
        return
    
    os.makedirs(BEST_DIR, exist_ok=True)
    print(f"\n[*] Archiving best model (Run #{best_run['run']}, Iteration {best_run['iter']})")
    
    # 1. Copy the best weights
    best_weights = os.path.join(OUT_DIR, 'best_model.pt')
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, os.path.join(BEST_DIR, 'best_model.pt'))
        print(f"  [+] Copied best_model.pt to {BEST_DIR}")
    
    # 2. Find and copy the best train.py from history
    if best_run['iter'] is not None:
        pattern = os.path.join(HISTORY_DIR, f"train_*_iter{best_run['iter']}_*.py")
        matches = glob.glob(pattern)
        if matches:
            # Sort by modification time to get the 'kept' or most recent version
            matches.sort(key=os.path.getmtime, reverse=True)
            best_script = matches[0]
            shutil.copy2(best_script, os.path.join(BEST_DIR, 'best_train.py'))
            print(f"  [+] Copied {os.path.basename(best_script)} to {BEST_DIR}/best_train.py")
        else:
            print(f"  [-] Could not find matching script in history for iteration {best_run['iter']}.")

def analyze(track_best=True, csv_export=None):
    """Print a comprehensive analysis of experiment results and optionally export to CSV."""
    results = load_results()
    if not results:
        return

    print("=" * 75)
    print("  AutoResearch — Experiment Analysis")
    print("=" * 75)
    print(f"\n  Total experiments: {len(results)}\n")

    # Table header
    print(f"  {'Run':>4}  {'Iter':>4}  {'Timestamp':<20}  {'Val Loss':>10}  {'Val BPB':>10}  {'Status'}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}")

    best_bpb = float('inf')
    best_run = None
    improvements = 0

    for r in results:
        is_best = r['val_bpb'] < (best_bpb - 1e-7) # Slight buffer for float comparison
        if is_best:
            best_bpb = r['val_bpb']
            best_run = r
            improvements += 1
            status = "★ BEST"
        else:
            status = ""

        iter_str = f"{r['iter']}" if r['iter'] is not None else "N/A"
        print(f"  {r['run']:>4}  {iter_str:>4}  {r['timestamp']:<20}  {r['val_loss']:>10.6f}  {r['val_bpb']:>10.6f}  {status}")

    # Summary
    print(f"\n{'─' * 75}")
    print(f"\n  Summary:")
    print(f"    Best val_bpb:    {best_bpb:.6f} (Run #{best_run['run']}, Iter {best_run['iter']})")
    print(f"    Improvements:    {improvements}/{len(results)} runs")

    if track_best:
        track_best_model(best_run)
    
    if csv_export:
        import csv
        try:
            with open(csv_export, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[+] Exported results to {csv_export}")
        except Exception as e:
            print(f"\n[-] CSV Export failed: {e}")

    print(f"\n{'=' * 75}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoResearch Analysis & Export")
    parser.add_argument("--csv", type=str, help="Path to export results as CSV (e.g., results.csv)")
    parser.add_argument("--no-track", action="store_true", help="Disable best model tracking/archiving")
    args = parser.parse_args()
    
    analyze(track_best=not args.no_track, csv_export=args.csv)


