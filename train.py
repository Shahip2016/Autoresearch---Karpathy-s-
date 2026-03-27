"""
train.py — The ONLY file the AI agent is allowed to edit.
Contains the full GPT model, optimizer, and training loop.
Optimized for 4GB VRAM GPUs.
"""

import os
import math
import time
import pickle
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # For reproducibility

# =============================================================================
# Hyperparameters — 4GB GPU Optimized
# =============================================================================

# Model
VOCAB_SIZE = 65  # char-level tokenizer on tinyshakespeare
BLOCK_SIZE = 512  # context length (reduced from 1024/2048 for VRAM)
N_LAYER = 6       # number of transformer layers (reduced from 12)
N_HEAD = 6        # number of attention heads
N_EMBD = 384      # embedding dimension (reduced from 768)
DROPOUT = 0.1

# Training
BATCH_SIZE = 16           # micro batch size
GRAD_ACCUM_STEPS = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 64
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 100
MAX_ITERS = 5000          # fallback max iters
MAX_RUNTIME = 300         # 5 minutes time budget (seconds)
EVAL_INTERVAL = 50
EVAL_ITERS = 20
PATIENCE = 5              # Early stopping patience in terms of evaluations
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
USE_AMP = DEVICE == 'cuda'

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Data Loading
# =============================================================================

def get_batch(split):
    """Load a batch of data from memmap files."""
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(DATA_DIR, fname), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# =============================================================================
# Model: GPT with RMSNorm (lightweight, good for small GPUs)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.head_dim = N_EMBD // N_HEAD
        self.n_head = N_HEAD
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.rotary = RotaryEmbedding(self.head_dim, BLOCK_SIZE)
        self.q_ln = RMSNorm(self.head_dim)
        self.k_ln = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply QK-Norm
        q = self.q_ln(q)
        k = self.k_ln(k)

        # Apply RoPE
        cos, sin = self.rotary(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use scaled_dot_product_attention for memory efficiency
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """SwiGLU MLP — more parameter efficient than standard FFN."""
    def __init__(self):
        super().__init__()
        hidden_dim = int(N_EMBD * 8 / 3)
        # Round to nearest multiple of 64 for GPU efficiency
        hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(N_EMBD, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, N_EMBD, bias=False)
        self.w3 = nn.Linear(N_EMBD, hidden_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(N_EMBD)
        self.mlp = MLP()
        self.ls_1 = nn.Parameter(torch.ones(N_EMBD) * 0.1)
        self.ls_2 = nn.Parameter(torch.ones(N_EMBD) * 0.1)

    def forward(self, x):
        x = x + self.ls_1 * self.attn(self.ln_1(x))
        x = x + self.ls_2 * self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * N_LAYER))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.tok_emb(idx)
        x = self.dropout(tok_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.1)
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================================================================
# Learning Rate Scheduler — Cosine with Warmup
# =============================================================================

def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# =============================================================================
# BPB (Bits Per Byte) — The Metric of Truth
# =============================================================================

def compute_bpb(val_loss):
    """Convert validation cross-entropy loss to Bits Per Byte."""
    # For char-level tokenizer: 1 token = 1 byte (approx)
    bpb = val_loss / math.log(2)
    return bpb

# =============================================================================
# Training Loop
# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("autoresearch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=DTYPE):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train():
    # Update globals with argparse args if needed
    global BATCH_SIZE, LEARNING_RATE, MAX_ITERS
    parser = argparse.ArgumentParser(description="AutoResearch Training — 4GB GPU Optimized")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Micro batch size")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS, help="Fallback max iters")
    args, unknown = parser.parse_known_args()
    
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MAX_ITERS = args.max_iters

    logger.info("=" * 60)
    logger.info("AutoResearch Training — 4GB GPU Optimized")
    logger.info("=" * 60)

    # Load meta
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        logger.info(f"Vocab size: {meta['vocab_size']}")

    # Create model
    model = GPT().to(DEVICE)
    param_count = model.count_parameters()
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Estimated model size: {param_count * 4 / 1024 / 1024:.1f} MB (FP32)")
    logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")
    logger.info(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"Time budget: {MAX_RUNTIME}s")

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Warning: torch.compile failed: {e}")

    # Optimizer — AdamW
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(BETA1, BETA2), fused=DEVICE=='cuda')

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # Timing
    if DEVICE == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()

    from tqdm import tqdm
    best_val_loss = float('inf')
    results = []
    patience_counter = 0

    pbar = tqdm(range(MAX_ITERS), desc="Training")
    for iter_num in pbar:
        # Check time budget
        if DEVICE == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = time.time() - t0

        if elapsed >= MAX_RUNTIME:
            logger.info(f"\n[TIME] Budget of {MAX_RUNTIME}s reached at iter {iter_num}. Stopping.")
            break

        # Learning rate schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch('train')
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=DTYPE):
                _, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Evaluation
        if iter_num > 0 and iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            train_loss = losses['train']
            val_loss = losses['val']
            val_bpb = compute_bpb(val_loss)

            pbar.write(f"  iter {iter_num:5d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f} | lr {lr:.2e} | time {elapsed:.1f}s")
            pbar.set_postfix({'val_bpb': f'{val_bpb:.4f}', 'lr': f'{lr:.2e}'})

            results.append({
                'iter': iter_num,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bpb': val_bpb,
                'elapsed': elapsed,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_model.pt'))
                pbar.write(f"    -> New best val_loss! Saved checkpoint.")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    pbar.write(f"\n[EARLY STOPPING] Validation loss stopped improving. Patience ({PATIENCE}) reached.")
                    break

    # Final evaluation
    losses = estimate_loss(model)
    final_val_loss = losses['val']
    final_bpb = compute_bpb(final_val_loss)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FINAL | val_loss {final_val_loss:.4f} | val_bpb {final_bpb:.4f}")
    logger.info(f"{'=' * 60}")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'final_model.pt'))

    import json
    # Write results to TSV
    results_path = os.path.join(os.path.dirname(__file__), 'results.tsv')
    with open(results_path, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{final_val_loss:.6f}\t{final_bpb:.6f}\t{param_count}\n")

    # Write full history to JSON
    json_path = os.path.join(OUT_DIR, 'loss_history.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Generate sample text
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        itos = meta['itos']
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        model.eval()
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=200)
        text = ''.join([itos[i] for i in generated[0].tolist()])
        logger.info(f"\n--- Generated Sample ---\n{text}\n")

    return final_bpb

if __name__ == "__main__":
    train()
