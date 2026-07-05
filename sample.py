"""
sample.py — Generate recipes from a trained Recipe GPT checkpoint
=================================================================

Usage:
  python sample.py                                    # default settings
  python sample.py --prompt "TITLE: Spicy"           # steer generation
  python sample.py --temperature 0.7 --top_k 40      # less random output
  python sample.py --temperature 1.2 --top_k None    # more creative output
  python sample.py --num_samples 5 --max_new_tokens 400

Temperature guide:
  0.5 → Very focused, likely to repeat patterns ("safe")
  0.8 → Good balance: coherent structure, some creativity (recommended)
  1.0 → Raw model probabilities, no adjustment
  1.2 → More adventurous, may invent stranger ingredients/combinations
  1.5+ → Often incoherent but occasionally surreal and funny

Top-k guide:
  10  → Very conservative, picks from likely chars only
  40  → Balanced (default)
  100 → Wider variety
  None → Sample from full vocabulary (can generate rare/weird chars)
"""

import os
import sys
import argparse
import pickle
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import GPTConfig
from model import GPT


def get_device() -> str:
    if torch.backends.mps.is_available():   return 'mps'
    if torch.cuda.is_available():           return 'cuda'
    return 'cpu'


def load_checkpoint(ckpt_path: str, device: str):
    """
    Load model and vocab from a checkpoint file.

    The checkpoint stores:
      - model weights (state_dict)
      - gpt_config (as a plain dict — no import dependency)
      - meta (vocab stoi/itos, dataset stats)
      - best_val_loss, iter_num, loss_log
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run 'python train.py' first."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Reconstruct config from saved dict
    gpt_cfg = GPTConfig(**checkpoint['gpt_config'])

    # Rebuild model and load weights
    model = GPT(gpt_cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    meta = checkpoint['meta']
    print(f"  Iter:      {checkpoint['iter_num']:,}")
    print(f"  Val loss:  {checkpoint['best_val_loss']:.4f}")
    print(f"  Vocab:     {meta['vocab_size']} chars")

    return model, meta, gpt_cfg


def generate(args):
    device = get_device()
    print(f"\nDevice: {device}")

    model, meta, gpt_cfg = load_checkpoint(args.checkpoint, device)

    stoi = meta['stoi']
    itos = meta['itos']

    encode = lambda s: [stoi.get(c, stoi.get(' ', 0)) for c in s]
    decode = lambda ids: ''.join(itos.get(i, '?') for i in ids)

    # Build prompt tokens
    prompt_ids = encode(args.prompt)
    if not prompt_ids:
        # Default: start from newline (beginning of a recipe)
        prompt_ids = encode('\n')

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    top_k = args.top_k if args.top_k > 0 else None

    print(f"\nGenerating {args.num_samples} recipe(s)...")
    print(f"  Prompt:      '{args.prompt}'")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k:       {top_k}")
    print(f"  Max tokens:  {args.max_new_tokens}")

    print(f"\n{'═'*60}")

    with torch.no_grad():
        for i in range(args.num_samples):
            output = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
            generated_ids = output[0].tolist()
            generated_text = decode(generated_ids)

            # Find the END marker to cleanly terminate the recipe
            end_marker = '<END>'
            end_pos = generated_text.find(end_marker)
            if end_pos != -1:
                generated_text = generated_text[:end_pos + len(end_marker)]

            print(f"\n{'─'*60}")
            print(f"  Recipe {i+1} of {args.num_samples}")
            print(f"{'─'*60}")
            print(generated_text)

    print(f"\n{'═'*60}")
    print("Done! Tip: try --temperature 0.7 for more coherent output,")
    print("or --temperature 1.2 for more creative (and chaotic) recipes.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate recipes from trained GPT')

    parser.add_argument(
        '--checkpoint', type=str, default='out/ckpt.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt', type=str, default='TITLE:',
        help='Starting text to condition generation on'
    )
    parser.add_argument(
        '--num_samples', type=int, default=3,
        help='Number of recipes to generate'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=300,
        help='Max characters to generate per recipe'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='Sampling temperature (lower=safer, higher=more creative)'
    )
    parser.add_argument(
        '--top_k', type=int, default=40,
        help='Top-k sampling (0 to disable)'
    )

    args = parser.parse_args()
    generate(args)
