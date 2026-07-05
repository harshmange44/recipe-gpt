"""
train.py — Training loop for Recipe GPT
========================================

What this script does:
  1. Detects best available device (MPS > CUDA > CPU)
  2. Loads tokenized data from data/processed/
  3. Initializes the GPT model from config.py
  4. Runs a training loop with:
       - Cosine LR schedule with linear warmup
       - Periodic validation loss evaluation (stable, averaged over many batches)
       - Gradient clipping (prevents training instability)
       - Checkpointing (saves on best val loss)
       - Loss logging to JSON (for visualize.py)
  5. Prints progress and timing

Usage:
  python train.py                         # default hyperparameters
  python train.py --max_iters 2000        # quick test run
  python train.py --batch_size 32         # if running out of memory
  python train.py --device cpu            # force CPU (slower but safe)
"""

import os
import sys
import time
import math
import json
import pickle
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch

# Add parent dir to path so we can import model.py / config.py
sys.path.insert(0, os.path.dirname(__file__))

from config import GPTConfig, TrainConfig
from model import GPT


# ── Device detection ──────────────────────────────────────────────────────────

def get_device(preferred: str = 'auto') -> str:
    """
    Detect the best available training device.

    Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA) > CPU
    MPS gives 3–5x speedup over CPU for this model size.
    """
    if preferred != 'auto':
        return preferred
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """
    Load tokenized binary files as memory-mapped numpy arrays.

    np.memmap reads from disk on demand instead of loading everything into RAM.
    For small datasets (~2MB) this doesn't matter much, but it's the correct
    pattern for larger datasets (production uses 40GB+ OpenWebText this way).

    Note: We recreate the memmap object every call to avoid a known Python
    memory leak with persistent memmap objects in loops.
    """
    train_path = os.path.join(data_dir, 'train.bin')
    val_path   = os.path.join(data_dir, 'val.bin')
    meta_path  = os.path.join(data_dir, 'meta.pkl')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            f"Run 'python data/prepare.py' first."
        )

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    return train_path, val_path, meta


def get_batch(data_path: str, block_size: int, batch_size: int, device: str):
    """
    Sample a random mini-batch of (inputs, targets) from the dataset.

    Key insight: targets are inputs shifted left by 1 position.
    This means each sequence contains block_size separate training examples:
      - position 0 predicts token 1 given token 0
      - position 1 predicts token 2 given tokens 0-1
      - position t predicts token t+1 given tokens 0..t

    So one batch of (B, T) actually has B*T training examples packed in!

    Args:
        data_path:  path to .bin file (re-opened each call to avoid memmap leak)
        block_size: context length (T)
        batch_size: number of sequences (B)
        device:     'mps', 'cuda', or 'cpu'
    """
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    # Sample random starting positions (ensure we have room for T+1 tokens)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)


# ── Loss evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_path: str,
    val_path: str,
    cfg: GPTConfig,
    train_cfg: TrainConfig,
    device: str,
) -> dict:
    """
    Estimate validation and training loss by averaging over many batches.

    Why not just use the last training batch's loss?
      The training loss is noisy — it's computed on ONE random batch.
      Averaging over eval_iters=200 batches gives a much more stable estimate.

    @torch.no_grad() disables gradient computation entirely:
      - Saves memory (no activation storage for backprop)
      - Faster forward pass
    model.eval() disables dropout (so we evaluate the full model, not a random
    subnetwork), then model.train() re-enables it.
    """
    results = {}
    model.eval()

    for split, data_path in [('train', train_path), ('val', val_path)]:
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(data_path, cfg.block_size, train_cfg.batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        results[split] = losses.mean().item()

    model.train()
    return results


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Three phases:
    ┌──────────────────────────────────────────────────────┐
    │  Phase 1: Linear warmup (0 → max_lr)                │
    │    Steps 0 → warmup_iters                           │
    │    Why: random init produces noisy gradients.        │
    │    Starting slow prevents large destructive updates. │
    │                                                      │
    │  Phase 2: Cosine decay (max_lr → min_lr)            │
    │    Steps warmup_iters → lr_decay_iters              │
    │    Why: smooth decay avoids abrupt drops that        │
    │    can destabilize an otherwise-stable training run. │
    │                                                      │
    │  Phase 3: Constant min_lr                           │
    │    Steps lr_decay_iters → ∞                         │
    │    Why: never fully stop — model can still refine    │
    │    rare patterns at low learning rate.               │
    └──────────────────────────────────────────────────────┘
    """
    max_lr = cfg.learning_rate
    min_lr = cfg.min_lr
    wi     = cfg.warmup_iters
    di     = cfg.lr_decay_iters

    if it < wi:
        # Linear warmup: lr = max_lr * (it+1) / (wi+1)
        return max_lr * (it + 1) / (wi + 1)
    if it > di:
        # Past decay schedule: stay at min_lr
        return min_lr
    # Cosine decay in between
    decay_ratio = (it - wi) / (di - wi)              # 0.0 → 1.0
    coeff       = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1.0 → 0.0
    return min_lr + coeff * (max_lr - min_lr)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(gpt_cfg: GPTConfig, train_cfg: TrainConfig):
    """Main training function."""

    # Setup
    torch.manual_seed(train_cfg.seed)
    device = get_device(train_cfg.device)
    print(f"\n{'='*55}")
    print(f"  Recipe GPT — Training")
    print(f"{'='*55}")
    print(f"  Device:     {device}")
    print(f"  Max iters:  {train_cfg.max_iters:,}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Block size: {gpt_cfg.block_size}")

    os.makedirs(train_cfg.out_dir, exist_ok=True)

    # Load data
    train_path, val_path, meta = load_data(train_cfg.data_dir)
    print(f"\n  Dataset:    {meta['n_recipes']:,} recipes")
    print(f"  Train:      {meta['train_tokens']:,} tokens")
    print(f"  Val:        {meta['val_tokens']:,} tokens")
    print(f"  Vocab:      {meta['vocab_size']} unique chars")
    print(f"  Baseline:   loss = {meta['baseline_loss']:.4f} (random model)")

    # Set vocab size from data
    gpt_cfg.vocab_size = meta['vocab_size']

    # Build model
    model = GPT(gpt_cfg).to(device)

    # Optimizer (separated weight decay groups)
    optimizer = model.configure_optimizers(train_cfg)

    # Compile (optional, skip on MPS — not fully supported)
    if train_cfg.compile:
        if device == 'mps':
            print("WARNING: torch.compile is not stable on MPS. Skipping.")
        else:
            print("Compiling model with torch.compile() (this takes ~1 minute)...")
            model = torch.compile(model)

    # ── State ────────────────────────────────────────────────────────────────
    loss_log      = {'iter': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    iter_num      = 0
    t0            = time.time()

    print(f"\n{'─'*55}")
    print(f"{'Step':>8}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print(f"{'─'*55}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while iter_num <= train_cfg.max_iters:

        # ① Set learning rate for this iteration
        lr = get_lr(iter_num, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ② Periodic validation + checkpointing
        if iter_num % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, train_path, val_path, gpt_cfg, train_cfg, device)
            train_loss = losses['train']
            val_loss   = losses['val']

            print(f"{iter_num:>8,}  {train_loss:>12.4f}  {val_loss:>10.4f}  {lr:>10.2e}")

            # Save to loss log JSON (available for visualization mid-training)
            loss_log['iter'].append(iter_num)
            loss_log['train_loss'].append(train_loss)
            loss_log['val_loss'].append(val_loss)
            loss_log['lr'].append(lr)

            log_path = os.path.join(train_cfg.out_dir, 'loss_log.json')
            with open(log_path, 'w') as f:
                json.dump(loss_log, f, indent=2)

            # Save checkpoint if val loss improved
            improved = val_loss < best_val_loss
            if improved or train_cfg.always_save_checkpoint:
                if improved:
                    best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        'model':        model.state_dict(),
                        'optimizer':    optimizer.state_dict(),
                        'gpt_config':   asdict(gpt_cfg),
                        'train_config': asdict(train_cfg),
                        'iter_num':     iter_num,
                        'best_val_loss': best_val_loss,
                        'loss_log':     loss_log,
                        'meta':         meta,
                    }
                    ckpt_path = os.path.join(train_cfg.out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_path)
                    if improved:
                        print(f"          → ✓ Checkpoint saved (new best: {best_val_loss:.4f})")

        if iter_num == train_cfg.max_iters:
            break

        # ③ Sample a training batch
        X, Y = get_batch(train_path, gpt_cfg.block_size, train_cfg.batch_size, device)

        # ④ Forward pass: compute loss
        logits, loss = model(X, Y)

        # ⑤ Backward pass: compute gradients
        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True is faster than zeroing: it deallocates the gradient
        # tensors entirely instead of writing zeros, saving a GPU kernel launch.
        loss.backward()

        # ⑥ Gradient clipping
        # If the gradient L2 norm exceeds grad_clip, scale all gradients down
        # proportionally. This prevents rare "spikes" where a bad batch causes
        # destructively large weight updates.
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # ⑦ Optimizer step: update weights
        optimizer.step()

        # ⑧ Timing log (every log_interval steps)
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms per step
        if iter_num % train_cfg.log_interval == 0 and iter_num % train_cfg.eval_interval != 0:
            print(f"{iter_num:>8,}  {loss.item():>12.4f}  {'─':>10}  {lr:>10.2e}  ({dt:.0f}ms)")
        t0 = t1

        iter_num += 1

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Perplexity:    {math.exp(best_val_loss):.2f}")
    print(f"")
    print(f"  Checkpoint: {train_cfg.out_dir}/ckpt.pt")
    print(f"  Loss log:   {train_cfg.out_dir}/loss_log.json")
    print(f"{'='*55}")
    print(f"\nNext steps:")
    print(f"  Generate recipes: python sample.py")
    print(f"  Visualize training: python visualize.py\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Recipe GPT')

    # Model architecture
    parser.add_argument('--block_size',   type=int,   default=256)
    parser.add_argument('--n_layer',      type=int,   default=6)
    parser.add_argument('--n_head',       type=int,   default=6)
    parser.add_argument('--n_embd',       type=int,   default=384)
    parser.add_argument('--dropout',      type=float, default=0.2)

    # Training
    parser.add_argument('--max_iters',    type=int,   default=5000)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--learning_rate',type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int,   default=100)

    # System
    parser.add_argument('--device',       type=str,   default='auto')
    parser.add_argument('--compile',      action='store_true', default=False)
    parser.add_argument('--data_dir',     type=str,   default='data/processed')
    parser.add_argument('--out_dir',      type=str,   default='out')
    parser.add_argument('--seed',         type=int,   default=1337)

    args = parser.parse_args()

    gpt_cfg = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        device=args.device,
        compile=args.compile,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
    )

    train(gpt_cfg, train_cfg)
