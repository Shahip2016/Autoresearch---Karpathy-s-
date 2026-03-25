"""
generate.py — Text Generation CLI.
Load a trained GPT checkpoint and generate text interactively or from a prompt.

Usage:
    python generate.py                          # interactive mode
    python generate.py --prompt "ROMEO:"        # generate from prompt
    python generate.py --checkpoint out/best_model.pt --max-tokens 500
    python generate.py --prompt "To be" --save output.txt
"""

import os
import sys
import argparse
import pickle
import torch

# ---------------------------------------------------------------------------
# We import model + hyperparams directly from train.py so the architecture
# always stays in sync — no copy-paste drift.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import GPT, DEVICE, BLOCK_SIZE  # noqa: E402

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'out')
DEFAULT_CHECKPOINT = os.path.join(OUT_DIR, 'best_model.pt')


def load_meta():
    """Load character-level tokenizer mappings from meta.pkl."""
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    if not os.path.exists(meta_path):
        print("ERROR: data/meta.pkl not found. Run `python prepare.py` first.")
        sys.exit(1)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']


def load_model(checkpoint_path):
    """Instantiate the GPT model and load weights from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.isdir(OUT_DIR):
            for f in os.listdir(OUT_DIR):
                if f.endswith('.pt'):
                    print(f"  out/{f}")
        else:
            print("  (none — run training first)")
        sys.exit(1)

    model = GPT()
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Parameters: {model.count_parameters():,}")
    return model


def encode_prompt(prompt, stoi):
    """Encode a string prompt to a tensor of token ids."""
    try:
        ids = [stoi[ch] for ch in prompt]
    except KeyError as e:
        print(f"WARNING: Character {e} not in vocabulary, skipping.")
        ids = [stoi[ch] for ch in prompt if ch in stoi]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def generate_text(model, itos, prompt_ids, max_tokens, temperature, top_k):
    """Generate text from the model and return the decoded string."""
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    tokens = output_ids[0].tolist()
    return ''.join([itos[t] for t in tokens])


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained AutoResearch GPT checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "ROMEO:"
  python generate.py --checkpoint out/final_model.pt --max-tokens 500
  python generate.py --prompt "To be" --temperature 1.0 --top-k 50
  python generate.py --prompt "JULIET:" --save output.txt
        """,
    )
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to model checkpoint (default: out/best_model.pt)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt to start generation (default: empty/interactive)')
    parser.add_argument('--max-tokens', type=int, default=300,
                        help='Maximum number of tokens to generate (default: 300)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling (default: 40)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--save', type=str, default=None,
                        help='Save generated text to a file')
    args = parser.parse_args()

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(args.seed)

    # Load tokenizer & model
    stoi, itos = load_meta()
    model = load_model(args.checkpoint)

    # Encode prompt
    if args.prompt:
        prompt_ids = encode_prompt(args.prompt, stoi)
        print(f"Prompt: \"{args.prompt}\"")
    else:
        # Empty context — start from a zero token
        prompt_ids = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        print("Prompt: (empty — free generation)")

    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    print(f"{'─' * 60}")

    # Generate
    text = generate_text(model, itos, prompt_ids, args.max_tokens, args.temperature, args.top_k)
    print(text)

    # Save if requested
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\n{'─' * 60}")
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
