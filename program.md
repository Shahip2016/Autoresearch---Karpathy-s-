# AutoResearch Program

You are an AI research agent. Your job is to improve the training of a small GPT language model by iteratively modifying `train.py`.

## Rules

1. **You may ONLY modify `train.py`.** Do not modify `prepare.py` or any other file.
2. **The metric is `val_bpb`** (validation bits per byte). Lower is better.
3. **Each experiment has a fixed time budget** of 5 minutes (300 seconds). The training loop will stop automatically.
4. **After each run**, check the output for the FINAL val_bpb score.
5. **If val_bpb improves** (decreases), keep the change. Otherwise, revert.
6. **Results are tracked** in `results.tsv`. Each line contains: timestamp, val_loss, val_bpb, param_count.
7. **Think step by step.** Before making a change, write a brief hypothesis about why it might help.

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
- Do NOT change the results.tsv output format.

## Strategy Tips

- Start with safe, well-known improvements (e.g., tuning learning rate).
- Try one change at a time so you can isolate what works.
- If a change hurts, revert it immediately.
- Consider: learning rate warmup, cosine decay, gradient accumulation, architecture tweaks.
- Monitor VRAM usage — if you OOM, reduce batch size or model size.

## Current Architecture

- 6-layer GPT with RMSNorm, RoPE, SwiGLU MLP
- ~15M parameters
- Character-level tokenizer (vocab_size=65)
- AdamW optimizer with cosine LR schedule
- Mixed precision (FP16) training
