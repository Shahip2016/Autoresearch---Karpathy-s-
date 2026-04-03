import os
import glob
import argparse

def clean_checkpoints(directory, keep=3):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
        
    checkpoints = glob.glob(os.path.join(directory, '*.pt'))
    checkpoints.sort(key=os.path.getmtime)
    
    if len(checkpoints) > keep:
        to_delete = checkpoints[:-keep]
        for ckpt in to_delete:
            os.remove(ckpt)
            print(f"Deleted old checkpoint: {ckpt}")
        print(f"Kept newest {keep} checkpoints.")
    else:
        print(f"Only found {len(checkpoints)} checkpoints, keeping all.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean old model checkpoints.")
    parser.add_argument('--dir', type=str, default='out', help='Directory containing checkpoints')
    parser.add_argument('--keep', type=int, default=3, help='Number of newest checkpoints to keep')
    args = parser.parse_args()
    
    clean_checkpoints(args.dir, args.keep)
