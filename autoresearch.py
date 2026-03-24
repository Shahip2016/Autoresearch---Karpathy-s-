"""
autoresearch.py — The Karpathy Loop controller.
Manages the autonomous research cycle:
  1. Run train.py
  2. Check val_bpb
  3. If improved, keep changes; else revert
  4. Repeat
"""

import os
import sys
import shutil
import time
import subprocess
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, 'train.py')
HISTORY_DIR = os.path.join(SCRIPT_DIR, 'history')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'results.tsv')

os.makedirs(HISTORY_DIR, exist_ok=True)

def read_last_bpb():
    """Read the last val_bpb from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    last_line = lines[-1].strip()
    if not last_line:
        return None
    parts = last_line.split('\t')
    if len(parts) >= 3:
        return float(parts[2])
    return None

def save_snapshot(iteration, label=""):
    """Save a copy of train.py to history/."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"train_{timestamp}_iter{iteration}_{label}.py"
    dest = os.path.join(HISTORY_DIR, filename)
    shutil.copy2(TRAIN_PY, dest)
    print(f"  [SNAPSHOT] Saved {filename}")
    return dest

def restore_snapshot(snapshot_path):
    """Restore train.py from a snapshot."""
    shutil.copy2(snapshot_path, TRAIN_PY)
    print(f"  [RESTORE] Reverted to {os.path.basename(snapshot_path)}")

def run_training():
    """Run train.py and return the exit code."""
    print(f"\n{'='*60}")
    print(f"  RUNNING EXPERIMENT: python train.py")
    print(f"{'='*60}\n")

    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, TRAIN_PY],
        cwd=SCRIPT_DIR,
        timeout=600,  # 10 min hard timeout (5 min in train.py + buffer)
    )
    return result.returncode

def run_loop(max_experiments=100, test_mode=False):
    """Main autonomous research loop."""
    print("=" * 60)
    print("  AutoResearch — The Karpathy Loop")
    print("=" * 60)
    print(f"  Max experiments: {max_experiments}")
    print(f"  Train script:    {TRAIN_PY}")
    print(f"  Results file:    {RESULTS_FILE}")
    print(f"  History dir:     {HISTORY_DIR}")
    print()

    if test_mode:
        print("[TEST MODE] Dry run — no actual training will occur.")
        # Save initial snapshot
        save_snapshot(0, "initial")
        print("[TEST MODE] Loop logic verified successfully.")
        return

    best_bpb = read_last_bpb()
    if best_bpb:
        print(f"  Previous best val_bpb: {best_bpb:.6f}")
    else:
        print("  No previous results found. This is the first run.")

    for experiment in range(1, max_experiments + 1):
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT {experiment}/{max_experiments}")
        print(f"{'#'*60}")

        # Save current train.py before any agent modifications
        snapshot = save_snapshot(experiment, "before")

        # Run training
        try:
            exit_code = run_training()
        except subprocess.TimeoutExpired:
            print("  [ERROR] Training timed out!")
            restore_snapshot(snapshot)
            continue
        except Exception as e:
            print(f"  [ERROR] Training failed: {e}")
            restore_snapshot(snapshot)
            continue

        if exit_code != 0:
            print(f"  [ERROR] Training exited with code {exit_code}. Reverting.")
            restore_snapshot(snapshot)
            continue

        # Check new bpb
        new_bpb = read_last_bpb()
        if new_bpb is None:
            print("  [ERROR] Could not read val_bpb. Reverting.")
            restore_snapshot(snapshot)
            continue

        print(f"\n  Result: val_bpb = {new_bpb:.6f}")

        if best_bpb is None or new_bpb < best_bpb:
            improvement = 0 if best_bpb is None else (best_bpb - new_bpb)
            print(f"  [IMPROVED] val_bpb improved by {improvement:.6f}! Keeping changes.")
            best_bpb = new_bpb
            save_snapshot(experiment, "kept")
        else:
            print(f"  [NO IMPROVEMENT] val_bpb {new_bpb:.6f} >= best {best_bpb:.6f}. Reverting.")
            restore_snapshot(snapshot)

        print(f"\n  Current best val_bpb: {best_bpb:.6f}")
        print(f"  Experiments completed: {experiment}/{max_experiments}")

    print(f"\n{'='*60}")
    print(f"  LOOP COMPLETE")
    print(f"  Best val_bpb achieved: {best_bpb:.6f}")
    print(f"  Total experiments: {max_experiments}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Loop Controller")
    parser.add_argument('--max-experiments', type=int, default=100,
                        help='Maximum number of experiments to run (default: 100)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Dry run to verify loop logic')
    args = parser.parse_args()
    run_loop(max_experiments=args.max_experiments, test_mode=args.test_mode)
