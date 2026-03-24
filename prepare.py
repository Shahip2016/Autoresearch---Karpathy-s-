import os
import requests
import numpy as np
from tqdm import tqdm

# Settings
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
INPUT_FILE_PATH = os.path.join(DATA_DIR, 'input.txt')

def download_data():
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(INPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def prepare_data():
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Simple character-level tokenization for 4GB GPU efficiency
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    
    # Mappings
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Train/Val split
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # Encode to numpy
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
    val_ids.tofile(os.path.join(DATA_DIR, 'val.bin'))
    
    # Save meta information
    import pickle
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Preparation complete. Files saved in 'data/' directory.")

if __name__ == "__main__":
    download_data()
    prepare_data()
