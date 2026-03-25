"""
diff_experiments.py — Experiment Diff Tool
Compares two hyperparameter/architecture configurations in `train.py`
snapshots saved in the `history/` directory.

Usage:
    python diff_experiments.py --list
    python diff_experiments.py history/train_2026..._iter5.py history/train_2026..._iter6.py
"""

import os
import sys
import argparse
import glob

try:
    import colorama
    colorama.init()
    RED = colorama.Fore.RED
    GREEN = colorama.Fore.GREEN
    RESET = colorama.Style.RESET_ALL
except ImportError:
    RED = ''
    GREEN = ''
    RESET = ''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(SCRIPT_DIR, 'history')


def list_snapshots():
    if not os.path.exists(HISTORY_DIR):
        print("No history directory found.")
        return []
    
    files = glob.glob(os.path.join(HISTORY_DIR, "train_*.py"))
    files.sort(key=os.path.getmtime)
    
    print("\n--- Available History Snapshots ---")
    if not files:
         print("No snapshots found.")
    for i, f in enumerate(files):
         name = os.path.basename(f)
         print(f"  [{i}] {name}")
    print("-----------------------------------")
    return files


def extract_hyperparams(filepath):
    """
    Extract lines between the typical hyperparameters section comment blocks
    in train.py to just compare the relevant config, not the whole file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_idx = 0
    end_idx = len(lines)
    
    # Try to isolate the hyperparam block if we can find the markers
    for i, line in enumerate(lines):
        if 'Hyperparameters' in line:
            start_idx = i
        if 'Data Loading' in line or 'Model:' in line:
            end_idx = i
            break
            
    # As a fallback or if we want to diff the whole thing, we just take the lines
    # Let's actually diff the whole file but filter out boilerplate and imports for brevity
    # Or actually, the most robust way is to just do a line-by-line diff of the whole file.
    return lines


def diff_files(file1, file2):
    import difflib
    
    if not os.path.exists(file1):
        print(f"Error: {file1} not found")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found")
        return

    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)
    
    lines1 = extract_hyperparams(file1)
    lines2 = extract_hyperparams(file2)
    
    print(f"\nComparing {name1} -> {name2}\n")
    
    diff = difflib.ndiff(lines1, lines2)
    
    changes = 0
    for line in diff:
        line = line.rstrip("\n")
        if line.startswith('- '):
            print(f"{RED}{line}{RESET}")
            changes += 1
        elif line.startswith('+ '):
            print(f"{GREEN}{line}{RESET}")
            changes += 1
        elif line.startswith('? '):
            pass # ignore diff hints line
    
    if changes == 0:
        print("No changes found between files.")
    else:
        print(f"\nFound {changes} changed lines.")


def main():
    parser = argparse.ArgumentParser(description="Compare AutoResearch train.py snapshots")
    parser.add_argument('file1', nargs='?', help='First snapshot file (older)')
    parser.add_argument('file2', nargs='?', help='Second snapshot file (newer)')
    parser.add_argument('--list', action='store_true', help='List available snapshots')
    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    if not args.file1 or not args.file2:
        print("Please provide two files to diff, or use --list.")
        print("Usage: python diff_experiments.py file1.py file2.py")
        sys.exit(1)
        
    diff_files(args.file1, args.file2)

if __name__ == "__main__":
    main()
