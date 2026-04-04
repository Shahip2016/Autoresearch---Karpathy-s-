"""
chat.py — Interactive Research Assistant
A premium CLI experience for interacting with trained AutoResearch models.
"""

import os
import sys
import time
import pickle
import argparse
import torch
import torch.nn.functional as F
from colorama import init, Fore, Style

# Initialize colorama
init()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import GPT, DEVICE, BLOCK_SIZE

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def typing_print(text, delay=0.01, color=Fore.GREEN):
    print(color, end='', flush=True)
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print(Style.RESET_ALL)

def main(args):
    clear_screen()
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  AutoResearch Interactive Assistant{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Load metadata
    meta_path = os.path.join(SCRIPT_DIR, 'data', 'meta.pkl')
    if not os.path.exists(meta_path):
        print(f"{Fore.RED}Error: data/meta.pkl not found. Run prepare.py first.{Style.RESET_ALL}")
        return
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Load best model
    ckpt_path = os.path.join(SCRIPT_DIR, 'out', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        # try final_model if best doesn't exist
        ckpt_path = os.path.join(SCRIPT_DIR, 'out', 'final_model.pt')
        
    if not os.path.exists(ckpt_path):
        print(f"{Fore.RED}Error: No model checkpoint found in out/. Train a model first.{Style.RESET_ALL}")
        return
        
    print(f"[*] Loading model from {Fore.YELLOW}{ckpt_path}{Style.RESET_ALL}...")
    model = GPT()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    # Handle both new and old checkpoint formats
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"[+] Model ready. (Param count: {model.count_parameters():,})")
    print(f"{Fore.DIM}Commands: /exit to quit, /clear to clear screen{Style.RESET_ALL}")
    print("-" * 60)

    context_ids = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    
    if args.system_prompt:
        sys_ids = []
        for ch in args.system_prompt:
            if ch in stoi:
                sys_ids.append(stoi[ch])
        if sys_ids:
            context_ids = torch.cat((context_ids, torch.tensor([sys_ids], dtype=torch.long, device=DEVICE)), dim=1)
            print(f"{Fore.DIM}[System Prompt Loaded]{Style.RESET_ALL}")

    
    while True:
        try:
            print(f"\n{Fore.MAGENTA}USER > {Style.RESET_ALL}", end='')
            user_input = input().strip()
            
            if not user_input:
                continue
            if user_input.lower() == '/exit':
                break
            if user_input.lower() == '/clear':
                clear_screen()
                continue

            # Encode user input as prompt
            new_ids = []
            for ch in user_input:
                if ch in stoi:
                    new_ids.append(stoi[ch])
                else:
                    # Ignore chars not in vocab
                    continue
            
            if not new_ids:
                print(f"{Fore.YELLOW}Warning: None of the characters in your input were in the model vocabulary.{Style.RESET_ALL}")
                continue
                
            context_ids = torch.cat((context_ids, torch.tensor([new_ids], dtype=torch.long, device=DEVICE)), dim=1)
            # Clip context to BLOCK_SIZE
            context_ids = context_ids[:, -BLOCK_SIZE:]
            
            print(f"{Fore.GREEN}GPT  > {Style.RESET_ALL}", end='', flush=True)
            
            # Generate one character at a time for the "typing" effect
            temperature = args.temperature
            top_k = args.top_k
            top_p = args.top_p
            
            generated_text = ""
            # Generate up to 200 tokens
            for _ in range(200):
                # crop to last BLOCK_SIZE tokens
                idx_cond = context_ids[:, -BLOCK_SIZE:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Decode and print
                char = itos[idx_next.item()]
                print(char, end='', flush=True)
                generated_text += char
                
                # Update context
                context_ids = torch.cat((context_ids, idx_next), dim=1)
                
                # Stop if it generates a newline (end of response)
                if char == '\n':
                    break
                    
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Interactive Chat")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (nucleus) sampling")
    parser.add_argument("--system-prompt", type=str, default="", help="Initial system prompt context")
    args = parser.parse_args()
    main(args)
