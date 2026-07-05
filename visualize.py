"""
visualize.py — Training visualization for Recipe GPT
=====================================================

Three plots for understanding and showcasing what the model learned:

  1. Loss curves      — train + val loss over training steps
                        with perplexity annotations and LR schedule
  2. Attention heatmaps — attention weights across all layers and heads
                        for a given prompt
  3. Char distribution  — probability over next characters at a given position
                        (shows what the model "expects" to come next)

Usage:
  python visualize.py                                    # all plots, default prompt
  python visualize.py --plot loss                        # just loss curves
  python visualize.py --plot attention --prompt "TITLE: Pasta"
  python visualize.py --plot distribution --prompt "TITLE: Banana Bread\nINGREDIENTS:"
  python visualize.py --checkpoint out/ckpt.pt --out_dir plots/
"""

import os
import sys
import json
import math
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from config import GPTConfig
from model import GPT


# ── Style ──────────────────────────────────────────────────────────────────────

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi':       150,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
})

BLUE   = '#2196F3'
RED    = '#EF5350'
GREEN  = '#4CAF50'
PURPLE = '#9C27B0'
GREY   = '#9E9E9E'


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run 'python train.py' first, then come back."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    gpt_cfg = GPTConfig(**checkpoint['gpt_config'])
    model = GPT(gpt_cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint, gpt_cfg


# ── Plot 1: Loss Curves ────────────────────────────────────────────────────────

def plot_loss_curves(checkpoint: dict, out_dir: str):
    """
    Plot training and validation loss over training iterations.

    What to look for:
      - Both curves should decrease from ~ln(vocab_size) baseline
      - A gap between train_loss and val_loss indicates some overfitting
        (normal for small datasets — val_loss is what matters for generation)
      - val_loss bottoming out = model has learned all it can from this data
      - The LR schedule subplot shows when warmup ends and cosine decay starts
    """
    loss_log = checkpoint.get('loss_log', {})

    if not loss_log or not loss_log.get('iter'):
        # Try loading from JSON file directly
        log_path = os.path.join(os.path.dirname(checkpoint.get('out_dir', '.')), 'loss_log.json')
        if os.path.exists('out/loss_log.json'):
            with open('out/loss_log.json') as f:
                loss_log = json.load(f)
        else:
            print("No loss log found in checkpoint. Train for at least one eval_interval first.")
            return

    iters      = loss_log['iter']
    train_loss = loss_log['train_loss']
    val_loss   = loss_log['val_loss']
    lrs        = loss_log.get('lr', [])

    meta          = checkpoint['meta']
    vocab_size    = meta['vocab_size']
    baseline      = meta['baseline_loss']
    best_val      = checkpoint['best_val_loss']
    best_val_iter = iters[int(np.argmin(val_loss))]

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35}
    )

    # ── Loss subplot ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(iters, train_loss, label='Train loss', color=BLUE,   linewidth=2.0)
    ax.plot(iters, val_loss,   label='Val loss',   color=RED,    linewidth=2.0, linestyle='--')
    ax.axhline(baseline, color=GREY, linestyle=':', linewidth=1.5,
               label=f'Random baseline: ln({vocab_size}) = {baseline:.2f}')
    ax.axvline(best_val_iter, color=GREEN, linestyle='-.', linewidth=1.5, alpha=0.8,
               label=f'Best val: {best_val:.4f} (iter {best_val_iter:,})')

    ax.set_xlabel('Training iteration')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('Recipe GPT — Training Progress', fontweight='bold', pad=12)
    ax.legend(loc='upper right')

    # Shade the gap between train and val (overfitting region)
    if len(iters) > 1:
        ax.fill_between(iters, train_loss, val_loss,
                        alpha=0.08, color=RED,
                        label='_Generalization gap')

    # Annotate final values
    final_ppl = math.exp(val_loss[-1])
    ax.annotate(
        f"Val loss: {val_loss[-1]:.3f}\n"
        f"Perplexity: {final_ppl:.1f}x\n"
        f"({final_ppl:.1f} equally likely\nnext chars on avg)",
        xy=(iters[-1], val_loss[-1]),
        xytext=(max(iters) * 0.55, val_loss[-1] + (baseline - val_loss[-1]) * 0.4),
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#BDBDBD', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2),
    )

    # Add perplexity labels on right axis (secondary y-axis)
    ax2 = ax.twinx()
    ax2.set_ylabel('Perplexity = exp(loss)', color='#795548', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#795548')
    y_min, y_max = ax.get_ylim()
    ax2.set_ylim(math.exp(max(0.01, y_min)), math.exp(y_max))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # ── LR schedule subplot ───────────────────────────────────────────────────
    if lrs:
        ax_lr = axes[1]
        ax_lr.plot(iters, lrs, color=PURPLE, linewidth=1.8)
        ax_lr.set_xlabel('Iteration')
        ax_lr.set_ylabel('Learning rate')
        ax_lr.set_title('Learning Rate Schedule (cosine with linear warmup)')
        ax_lr.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # Annotate warmup end
        train_cfg = checkpoint.get('train_config', {})
        warmup_iters = train_cfg.get('warmup_iters', None)
        if warmup_iters and warmup_iters in iters:
            ax_lr.axvline(warmup_iters, color=GREY, linestyle=':', alpha=0.7)
            ax_lr.text(warmup_iters + max(iters)*0.01,
                       max(lrs) * 0.95, 'warmup ends',
                       fontsize=8, color=GREY)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.show()


# ── Plot 2: Attention Heatmaps ─────────────────────────────────────────────────

def plot_attention_heatmaps(model: GPT, checkpoint: dict, prompt: str, out_dir: str):
    """
    Visualize attention weight matrices across all layers and heads.

    Each cell [i, j] in a heatmap shows how much position i attended to
    position j. Brighter = more attention weight.

    What to look for in recipe attention:
      - Diagonal: tokens always attend to themselves (identity pattern)
      - "INGREDIENTS:" tokens attending back to "TITLE:" (cross-reference)
      - Punctuation marks like "," and "\n" drawing attention (structural anchors)
      - Later layers showing more "global" patterns (structured reasoning)
      - Earlier layers showing more local patterns (adjacent chars)

    This is one of the most compelling visuals for a blog — it shows the
    model has genuinely learned the structure of recipes, not just memorized.
    """
    device = next(model.parameters()).device
    meta = checkpoint['meta']
    stoi = meta['stoi']
    itos = meta['itos']

    encode = lambda s: [stoi.get(c, stoi.get(' ', 0)) for c in s]

    ids = encode(prompt)
    T   = len(ids)

    if T == 0:
        print("Prompt is empty after encoding. Check that all characters are in vocab.")
        return

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Run forward pass — attention weights are stored as side effect
    attn_weights = model.get_attention_weights(x)  # dict: layer_i_head_j → (T, T)

    gpt_cfg = GPTConfig(**checkpoint['gpt_config'])
    n_layer = gpt_cfg.n_layer
    n_head  = gpt_cfg.n_head

    fig, axes = plt.subplots(
        n_layer, n_head,
        figsize=(n_head * 2.8, n_layer * 2.8),
        squeeze=False,
    )

    # Truncate prompt for title display
    prompt_display = prompt.replace('\n', '↵')
    short_prompt = prompt_display[:45] + '...' if len(prompt_display) > 45 else prompt_display

    fig.suptitle(
        f'Attention Weights — All {n_layer} Layers × {n_head} Heads\n'
        f'Prompt: "{short_prompt}"',
        fontsize=12, fontweight='bold', y=1.01
    )

    # Character labels for axes (show every nth char to avoid crowding)
    chars = [c if c.isprintable() else '↵' for c in prompt]
    step  = max(1, T // 16)

    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            ax  = axes[layer_idx][head_idx]
            key = f'layer_{layer_idx}_head_{head_idx}'
            w   = attn_weights.get(key, np.zeros((T, T)))

            sns.heatmap(
                w,
                ax=ax,
                cmap='Blues',
                vmin=0,
                vmax=w.max() if w.max() > 0 else 1.0,
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
            )

            # Row labels (left column only): layer names
            if head_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=9, rotation=90, labelpad=4)

            # Column labels (top row only): head names
            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=9, pad=4)

            # Border styling: highlight middle layers
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor('#BDBDBD')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'attention_heatmaps.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {out_path}")
    plt.show()


# ── Plot 3: Next-Character Distribution ────────────────────────────────────────

def plot_char_distribution(model: GPT, checkpoint: dict, prompt: str, out_dir: str,
                           top_n: int = 30):
    """
    Show the model's probability distribution over the next character.

    This answers: "After seeing this prompt, what characters does the model
    think are most likely to come next, and with what confidence?"

    For a well-trained model on recipes:
      - After "TITLE: ", you'd expect letters (A-Z, a-z) to dominate
      - After "INGREDIENTS: ", same (ingredient names start with letters)
      - After "\n", you'd expect "INGREDIENTS" or "INSTRUCTIONS" or "TITLE"
        (the model learned the recipe format!)
      - After "1 cup ", the model might strongly predict letters (ingredient names)
    """
    device = next(model.parameters()).device
    meta = checkpoint['meta']
    stoi = meta['stoi']
    itos = meta['itos']

    encode = lambda s: [stoi.get(c, stoi.get(' ', 0)) for c in s]

    ids = encode(prompt)
    if not ids:
        print("Empty prompt after encoding.")
        return

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x)
        # Probability distribution for the next character after the prompt
        probs = torch.softmax(logits[0, -1, :], dim=-1).cpu().numpy()

    # Sort by probability (descending), take top N
    sorted_indices = np.argsort(probs)[::-1][:top_n]
    sorted_chars   = []
    for i in sorted_indices:
        ch = itos.get(i, '?')
        if ch == '\n':   display = '↵'
        elif ch == ' ':  display = '·'
        elif ch == '\t': display = '→'
        else:            display = ch if ch.isprintable() else '?'
        sorted_chars.append(display)
    sorted_probs = probs[sorted_indices]

    fig, ax = plt.subplots(figsize=(13, 5))

    # Color bars: top 5 = strong blue, rest = light blue
    colors = [BLUE if i < 5 else '#90CAF9' for i in range(top_n)]
    bars   = ax.bar(range(top_n), sorted_probs, color=colors, edgecolor='white',
                    linewidth=0.5)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(sorted_chars, fontsize=10)
    ax.set_ylabel('Probability')
    prompt_display = prompt.replace('\n', '↵')
    ax.set_title(
        f'Next-Character Probability Distribution\n'
        f'After: "{prompt_display[-40:]}"',
        fontweight='bold', pad=12
    )

    # Value annotations on bars
    for bar, prob in zip(bars, sorted_probs):
        if prob > 0.015:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{prob:.2f}',
                ha='center', va='bottom', fontsize=7.5, color='#424242'
            )

    # Entropy annotation (measure of uncertainty)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    max_entropy = math.log(meta['vocab_size'])
    ax.text(
        0.97, 0.92,
        f"Entropy: {entropy:.2f} / {max_entropy:.2f}\n"
        f"(lower = more confident)",
        transform=ax.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#BDBDBD', alpha=0.9)
    )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'char_distribution.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = ('mps'  if torch.backends.mps.is_available()  else
              'cuda' if torch.cuda.is_available()           else 'cpu')

    print(f"\nVisualizing: {args.checkpoint}")
    model, checkpoint, gpt_cfg = load_checkpoint(args.checkpoint, device)
    print(f"  Loaded — iter {checkpoint['iter_num']:,}, "
          f"val_loss {checkpoint['best_val_loss']:.4f}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot in ('all', 'loss'):
        print("Plotting loss curves...")
        plot_loss_curves(checkpoint, args.out_dir)

    if args.plot in ('all', 'attention'):
        print(f"Plotting attention heatmaps (prompt: '{args.prompt[:30]}...')...")
        plot_attention_heatmaps(model, checkpoint, args.prompt, args.out_dir)

    if args.plot in ('all', 'distribution'):
        print(f"Plotting char distribution (prompt: '{args.prompt[:30]}...')...")
        plot_char_distribution(model, checkpoint, args.prompt, args.out_dir)

    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Recipe GPT training')

    parser.add_argument('--checkpoint', type=str,  default='out/ckpt.pt')
    parser.add_argument('--plot',       type=str,  default='all',
                        choices=['all', 'loss', 'attention', 'distribution'])
    parser.add_argument('--prompt',     type=str,
                        default='TITLE: Banana Bread\nINGREDIENTS:',
                        help='Prompt for attention and distribution plots')
    parser.add_argument('--out_dir',    type=str,  default='plots')
    parser.add_argument('--top_n',      type=int,  default=30,
                        help='Number of top chars to show in distribution plot')

    args = parser.parse_args()
    main(args)
