# AutoResearch Program

You are an AI research agent. Your job is to improve the training of a small GPT language model by iteratively modifying `train.py`.

## Rules

1. **You may ONLY modify `train.py`.** Do not modify `prepare.py` or any other file.
8. **The primary metric is `val_bpb`** (validation bits per byte). Lower is better.
9. **Secondary metrics include `acc1`** (Top-1 accuracy) and `acc5` (Top-5 accuracy).
10. **Each experiment has a fixed time budget** of 5 minutes (300 seconds). The training loop will stop automatically.
11. **After each run**, check the output for the FINAL metrics summary.
12. **If val_bpb improves** (decreases), keep the change. Otherwise, revert.
13. **Results are tracked** in `results.tsv`. Columns: `timestamp`, `val_loss`, `val_bpb`, `param_count`, `iteration`, `lr`, `tokens`, `acc1`.
14. **Reproducibility matters.** Metadata is logged to `experiments.json` and `out/config.json`.
15. **Think step by step.** Before making a change, write a brief hypothesis about why it might help.

## What You Can Change in train.py

Everything is fair game:
- Model architecture (layers, heads, embedding size, MLP type)
- Hyperparameters (learning rate, batch size, warmup, weight decay)
- Optimizer settings
- Training loop logic
- Data loading strategy
- Regularization (dropout, weight decay)
- Normalization (RMSNorm, LayerNorm)
- Position encoding
- Activation functions

## Constraints

- The model MUST fit in 4GB of GPU VRAM (model + optimizer + gradients + activations).
- Keep `VOCAB_SIZE = 65` (character-level tokenizer).
- Keep `MAX_RUNTIME = 300` (5 minute time budget).
- Do NOT change the BPB calculation formula.
- Ensure `results.tsv` output includes the new `acc1` column as the 8th tab-separated value.

## Strategy Tips

- Start with safe, well-known improvements (e.g., tuning learning rate).
- Try one change at a time so you can isolate what works.
- If a change hurts, revert it immediately.
- Try advanced features: **Grouped Query Attention (GQA)**, **Sliding Window Attention**, or **Weight Tying**.
- Experiment with **Activation Functions**: SwiGLU is current, but try GeGLU or ReGLU.
- Test **Normalization**: RMSNorm vs LayerNorm vs DeepNorm.
- Try **Register Tokens**: Add few dummy tokens to the sequence for the model to "think".
- Monitor VRAM with `profile.py` if you make large architectural changes.
- Use `visualize_attention.py --prompt "To be or not to be"` to see if heads are collapsing.

## Current Architecture

- 6-layer GPT with RMSNorm, RoPE, SwiGLU MLP
- ~15M parameters
- Character-level tokenizer (vocab_size=65)
- AdamW optimizer with cosine LR schedule
- Mixed precision (FP16) training
