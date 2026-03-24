# AutoResearch — 4GB GPU Edition 🔬

> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An autonomous AI research framework that runs ML experiments while you sleep — optimized for 4GB VRAM GPUs.

## The Idea

An AI agent iteratively modifies `train.py`, runs a 5-minute training experiment, checks the **val_bpb** (validation bits per byte) metric, and keeps improvements while reverting failures. Overnight, it can run ~100 experiments autonomously.

## Architecture (4GB GPU Optimized)

| Component | Spec |
|---|---|
| Model | 6-layer GPT, 384 dim, 6 heads (~15M params) |
| Attention | Scaled Dot-Product with RoPE |
| Normalization | RMSNorm |
| MLP | SwiGLU |
| Precision | FP16 mixed precision |
| Context | 512 tokens |
| Optimizer | AdamW with cosine LR schedule |

## Files

| File | Description |
|---|---|
| `prepare.py` | Downloads TinyShakespeare, creates train/val splits, builds tokenizer. **Never modified by the agent.** |
| `train.py` | The **only file** the AI agent edits. Contains the full GPT model, optimizer, and training loop. |
| `program.md` | Plain English instructions for the AI agent — defines rules, constraints, and strategy. |
| `autoresearch.py` | The "Karpathy Loop" controller. Runs experiments, tracks results, keeps/reverts changes. |
| `analyze.py` | Experiment analysis tool — parses `results.tsv` and shows improvement statistics. |
| `dashboard.html` | Interactive web dashboard for visualizing experiment progress with charts. |

## Quick Start

```bash
# 1. Prepare data
python prepare.py

# 2. Run a single training experiment
python train.py

# 3. Start the autonomous loop (100 experiments)
python autoresearch.py --max-experiments 100

# 4. Analyze results
python analyze.py

# 5. View dashboard
# Open dashboard.html in your browser, then load results.tsv
```

## How It Works

```
┌─────────────────────────────────────────────┐
│           The Karpathy Loop                 │
│                                             │
│   1. Agent reads program.md                 │
│   2. Agent modifies train.py                │
│   3. Run 5-min training experiment          │
│   4. Check val_bpb in results.tsv           │
│   5. If improved → keep changes             │
│      If not → revert to previous version    │
│   6. Repeat                                 │
└─────────────────────────────────────────────┘
```

## Extra Features (Beyond Original)

- **🖥️ Web Dashboard** — Real-time experiment visualization with Chart.js
- **📊 Analysis Tool** — Statistical summary of all experiments
- **🔒 VRAM Safety** — Optimized defaults to prevent OOM on 4GB GPUs
- **💾 History Snapshots** — Every version of `train.py` is saved in `history/`
- **🎯 SwiGLU + RoPE** — Modern architecture choices for better parameter efficiency

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU)
- numpy, requests, tqdm, matplotlib

## License

MIT
