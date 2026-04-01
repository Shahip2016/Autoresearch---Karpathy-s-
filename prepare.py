import os
import requests
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from collections import Counter

# Available datasets
DATASETS = {
    'tinyshakespeare': "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    'wikitext2': "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    'tinystories': "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    'wikitext103': "https://raw.githubusercontent.com/lexnederbragt/python-course/master/data/wikitext-103-train-subset.txt",
}

def download_data(dataset_name, data_dir):
    input_file_path = os.path.join(data_dir, f'input_{dataset_name}.txt')
    if not os.path.exists(input_file_path):
        url = DATASETS.get(dataset_name)
        if not url:
            print(f"Error: Dataset '{dataset_name}' not found in registry.")
            return None
        
        print(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(input_file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dataset_name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"Dataset '{dataset_name}' already exists at {input_file_path}.")
    return input_file_path

def prepare_data(input_file_path, data_dir, dataset_name, max_chars=None):
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        if max_chars:
            data = f.read(max_chars)
        else:
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
    
    # Use dataset prefix if not default
    prefix = "" if dataset_name == 'tinyshakespeare' else f"{dataset_name}_"
    train_ids.tofile(os.path.join(data_dir, f'{prefix}train.bin'))
    val_ids.tofile(os.path.join(data_dir, f'{prefix}val.bin'))
    
    # Save meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_name = 'meta.pkl' if dataset_name == 'tinyshakespeare' else f'{dataset_name}_meta.pkl'
    with open(os.path.join(data_dir, meta_name), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Preparation complete. Files saved in 'data/' with prefix '{prefix}'.")

def print_stats(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read()
    
    print(f"\n{'='*30}")
    print(f" DATASET PROFILER: {os.path.basename(input_file_path)}")
    print(f"{'='*30}")
    print(f"Total characters: {len(data):,}")
    
    chars = Counter(data)
    unique_chars = sorted(chars.keys())
    print(f"Unique characters: {len(unique_chars)}")
    
    # Frequency analysis
    most_common = chars.most_common(10)
    least_common = chars.most_common()[:-11:-1]
    
    print("\nTop 10 Most Frequent:")
    for char, count in most_common:
        repr_char = repr(char)
        print(f"  {repr_char:10} : {count:8,}")
        
    print("\nTop 10 Least Frequent:")
    for char, count in least_common:
        repr_char = repr(char)
        print(f"  {repr_char:10} : {count:8,}")
        
    # Density / Entropy-like stats
    whitespace = chars.get(' ', 0) + chars.get('\n', 0) + chars.get('\t', 0)
    print(f"\nWhitespace density: {whitespace / len(data):.2%}")
    print(f"{'='*30}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for AutoResearch")
    parser.add_argument('--dataset', type=str, default='tinyshakespeare', 
                        choices=list(DATASETS.keys()), help='Dataset to download and prepare')
    parser.add_argument('--max-chars', type=int, default=None, help='Limit the dataset size for faster preparation/training')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics during preparation')
    parser.add_argument('--stats-only', action='store_true', help='Only show dataset statistics and exit')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    path = download_data(args.dataset, data_dir)
    if not path:
        exit(1)
        
    if args.stats_only:
        print_stats(path)
    else:
        if args.stats:
            print_stats(path)
        prepare_data(path, data_dir, args.dataset, args.max_chars)

