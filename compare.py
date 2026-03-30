"""
compare.py — Side-by-Side Model Comparison Tool.
Loads two checkpoints (e.g., current vs best) and generates text from both for comparison.
"""

import os
import sys
import argparse
import pickle
import torch
from colorama import init, Fore, Style

# Initialize colorama
init()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import GPT, DEVICE, BLOCK_SIZE  # noqa: E402

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'out')
BEST_DIR = os.path.join(SCRIPT_DIR, 'best')

def load_meta(dataset='tinyshakespeare'):
    prefix = "" if dataset == 'tinyshakespeare' else f"{dataset}_"
    meta_path = os.path.join(DATA_DIR, f'{prefix}meta.pkl')
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found.")
        sys.exit(1)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    model = GPT()
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def generate_text(model, itos, prompt_ids, max_tokens, temperature, top_k):
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    tokens = output_ids[0].tolist()
    return ''.join([itos[t] for t in tokens])

def main():
    parser = argparse.ArgumentParser(description="Compare two AutoResearch GPT models.")
    parser.add_argument('--model-a', type=str, default=os.path.join(OUT_DIR, 'final_model.pt'),
                        help='Path to Model A (default: out/final_model.pt)')
    parser.add_argument('--model-b', type=str, default=os.path.join(BEST_DIR, 'best_model.pt'),
                        help='Path to Model B (default: best/best_model.pt)')
    parser.add_argument('--prompt', type=str, default="Once upon a time,",
                        help='Prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=150,
                        help='Max new tokens (default: 150)')
    parser.add_argument('--temp', type=float, default=0.8,
                        help='Temperature (default: 0.8)')
    parser.add_argument('--dataset', type=str, default='tinyshakespeare',
                        help='Dataset to use for meta (default: tinyshakespeare)')
    args = parser.parse_args()

    stoi, itos = load_meta(args.dataset)
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f" 🔬 AutoResearch Model Comparison")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}Prompt:{Style.RESET_ALL} \"{args.prompt}\"")
    print(f"{Fore.YELLOW}Max Tokens:{Style.RESET_ALL} {args.max_tokens}, {Fore.YELLOW}Temp:{Style.RESET_ALL} {args.temp}\n")

    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=DEVICE)

    print(f"{Fore.GREEN}[Model A]: {os.path.basename(args.model_a)}{Style.RESET_ALL}")
    model_a = load_model(args.model_a)
    if model_a:
        out_a = generate_text(model_a, itos, prompt_ids, args.max_tokens, args.temp, 40)
        print(f'"{out_a}"\n')
    else:
        print("  (Failed to load Model A)\n")

    print(f"{Fore.MAGENTA}[Model B]: {os.path.basename(args.model_b)}{Style.RESET_ALL}")
    model_b = load_model(args.model_b)
    if model_b:
        out_b = generate_text(model_b, itos, prompt_ids, args.max_tokens, args.temp, 40)
        print(f'"{out_b}"\n')
    else:
        print("  (Failed to load Model B)\n")

    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

if __name__ == '__main__':
    main()
