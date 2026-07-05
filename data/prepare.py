"""
data/prepare.py — Download, clean, and tokenize recipe dataset
==============================================================

What this script does:
  1. Download recipe data from HuggingFace datasets (no API key needed)
  2. Clean: filter bad records, strip HTML, normalize whitespace
  3. Format each recipe in Option A minimal format:
       TITLE: Chocolate Chip Cookies
       INGREDIENTS: 2 cups flour, 1 cup butter, 1 cup sugar
       INSTRUCTIONS: Preheat oven to 375F. Mix butter and sugar. ...
       <END>
  4. Concatenate all recipes into one long string (same as Tiny Shakespeare)
  5. Build character vocabulary: unique chars → stoi/itos mappings
  6. Encode: entire text → uint16 integer array
  7. Split 90% train / 10% val
  8. Save: train.bin, val.bin, meta.pkl

Output (in data/processed/):
  train.bin  — numpy uint16 array, ~1.8M tokens (90%)
  val.bin    — numpy uint16 array, ~200K tokens (10%)
  meta.pkl   — {vocab_size, stoi, itos, n_recipes, stats}

Usage:
  python data/prepare.py
  python data/prepare.py --n_recipes 5000   # fewer recipes for quick test
"""

import os
import re
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'processed')
N_RECIPES   = 20_000   # how many recipes to use from the full dataset
MIN_CHARS   = 150      # skip recipes shorter than this (too sparse to learn from)
MAX_CHARS   = 1_800    # skip very long recipes (outliers distort char distribution)
TRAIN_RATIO = 0.9      # 90% train, 10% val

# ── Dataset loading ────────────────────────────────────────────────────────────

def try_load_dataset(n_recipes: int):
    """
    Try to download a recipe dataset from HuggingFace.
    Falls back through multiple sources in case one is unavailable.

    Returns: (records list, schema string)
    """
    from datasets import load_dataset

    # Attempt 1: recipe_nlg (the standard benchmark dataset, 2.2M recipes)
    # We take a slice so we don't download the full 2GB.
    candidates = [
        ("recipe_nlg",               "train",   "recipe_nlg"),
        ("Shengtao/recipe",          "train",   "shengtao"),
        ("m3hrdadfi/recipe_nlg_lite","train",   "recipe_nlg"),
    ]

    for dataset_name, split, schema in candidates:
        try:
            print(f"Trying: {dataset_name} ...")
            # Take a slice to avoid full download when possible
            slice_split = f"{split}[:{n_recipes * 2}]"  # 2x buffer for filtering
            ds = load_dataset(dataset_name, split=slice_split, trust_remote_code=True)
            print(f"  ✓ Loaded {len(ds):,} records from '{dataset_name}'")
            return list(ds), schema
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    raise RuntimeError(
        "Could not download any recipe dataset.\n"
        "Make sure you have internet access and 'datasets' installed:\n"
        "  pip install datasets"
    )


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove HTML artifacts and normalize whitespace."""
    if not text:
        return ''
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace (collapse multiple spaces, strip leading/trailing)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_fields(record: dict, schema: str):
    """
    Extract (title, ingredients, instructions) from a raw dataset record.

    Different HuggingFace datasets use different field names — this handles both.
    """
    if schema == 'recipe_nlg':
        title        = record.get('title', '')
        ingredients  = record.get('ingredients', [])
        instructions = record.get('directions', [])  # list of step strings
    elif schema == 'shengtao':
        title        = record.get('title', '') or record.get('name', '')
        ingredients  = record.get('ingredients', [])
        instructions = record.get('instructions', '') or record.get('directions', [])
    else:
        title        = record.get('title', '')
        ingredients  = record.get('ingredients', [])
        instructions = record.get('directions', record.get('instructions', []))

    return title, ingredients, instructions


def format_recipe(title, ingredients, instructions) -> str:
    """
    Format one recipe into our canonical Option A minimal format.

    TITLE: <title>
    INGREDIENTS: <ing1>, <ing2>, <ing3>
    INSTRUCTIONS: <step1>. <step2>. ...
    <END>

    Returns None if the recipe is missing critical fields.
    """
    title = clean_text(str(title or ''))
    if not title:
        return None

    # Ingredients: normalize to a comma-separated string
    if isinstance(ingredients, list):
        ingredients = ', '.join(
            clean_text(str(i)) for i in ingredients if str(i).strip()
        )
    else:
        ingredients = clean_text(str(ingredients or ''))
    if not ingredients:
        return None

    # Instructions: normalize to a single string with period-separated steps
    if isinstance(instructions, list):
        instructions = ' '.join(
            clean_text(str(step)) for step in instructions if str(step).strip()
        )
    else:
        instructions = clean_text(str(instructions or ''))
    if not instructions:
        return None

    recipe = (
        f"TITLE: {title}\n"
        f"INGREDIENTS: {ingredients}\n"
        f"INSTRUCTIONS: {instructions}\n"
        f"<END>\n\n"
    )
    return recipe


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_records(records: list, schema: str, n_recipes: int) -> list:
    """Convert raw dataset records to formatted recipe strings."""
    recipes  = []
    skipped  = 0

    for record in tqdm(records, desc="Formatting recipes"):
        title, ingredients, instructions = extract_fields(record, schema)
        formatted = format_recipe(title, ingredients, instructions)

        if formatted is None:
            skipped += 1
            continue

        # Filter by length: too short = not enough to learn from;
        # too long = outliers with unusual structure
        if not (MIN_CHARS <= len(formatted) <= MAX_CHARS):
            skipped += 1
            continue

        recipes.append(formatted)

        if len(recipes) >= n_recipes:
            break

    print(f"  Kept: {len(recipes):,} recipes  |  Skipped: {skipped:,}")
    return recipes


def build_vocab(text: str) -> tuple:
    """
    Build a character-level vocabulary from the full text.

    Returns:
        vocab_size: int — number of unique characters
        stoi: dict — character → integer index
        itos: dict — integer index → character
    """
    chars     = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, stoi, itos


def encode(text: str, stoi: dict) -> list:
    """Encode a string to a list of integer token ids."""
    return [stoi[ch] for ch in text if ch in stoi]


def main(n_recipes: int = N_RECIPES):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Recipe GPT — Data Preparation")
    print(f"{'='*55}")
    print(f"  Target: {n_recipes:,} recipes")
    print(f"  Output: {OUTPUT_DIR}\n")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    records, schema = try_load_dataset(n_recipes)

    # ── Step 2: Process + format ──────────────────────────────────────────────
    print(f"\nFormatting to Option A (TITLE/INGREDIENTS/INSTRUCTIONS/<END>)...")
    recipes = process_records(records, schema, n_recipes)

    if len(recipes) < 500:
        print(f"ERROR: Only got {len(recipes)} valid recipes (need ≥ 500).")
        print("Try with a larger source dataset or check the data fields.")
        sys.exit(1)

    # ── Step 3: Concatenate ───────────────────────────────────────────────────
    full_text = ''.join(recipes)
    print(f"\nDataset stats:")
    print(f"  Recipes:          {len(recipes):,}")
    print(f"  Total characters: {len(full_text):,}")

    # Show a sample recipe so we can verify format looks right
    print(f"\nSample recipe preview:")
    print(f"{'─'*50}")
    print(recipes[0][:400])
    print(f"{'─'*50}")

    # ── Step 4: Build vocabulary ──────────────────────────────────────────────
    vocab_size, stoi, itos = build_vocab(full_text)
    print(f"\nCharacter vocabulary:")
    print(f"  Vocab size: {vocab_size}")
    printable_chars = ''.join(c if c.isprintable() else '?' for c in sorted(set(full_text)))
    print(f"  Characters: {printable_chars[:80]}{'...' if vocab_size > 80 else ''}")

    # ── Step 5: Encode ────────────────────────────────────────────────────────
    print(f"\nEncoding {len(full_text):,} characters to integers...")
    ids = encode(full_text, stoi)
    print(f"  Total tokens: {len(ids):,}")

    # Baseline loss for a random model: -log(1/vocab_size) = log(vocab_size)
    baseline_loss = np.log(vocab_size)
    print(f"\nExpected loss for random model: ln({vocab_size}) = {baseline_loss:.4f}")
    print(f"(A well-trained model should reach ~1.5–2.0 on this dataset)")

    # ── Step 6: Train/val split ───────────────────────────────────────────────
    n = int(TRAIN_RATIO * len(ids))
    train_ids = np.array(ids[:n],  dtype=np.uint16)
    val_ids   = np.array(ids[n:],  dtype=np.uint16)
    print(f"\nSplit ({TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}):")
    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val:   {len(val_ids):,} tokens")

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    train_path = os.path.join(OUTPUT_DIR, 'train.bin')
    val_path   = os.path.join(OUTPUT_DIR, 'val.bin')
    meta_path  = os.path.join(OUTPUT_DIR, 'meta.pkl')

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        'vocab_size':    vocab_size,
        'stoi':          stoi,
        'itos':          itos,
        'n_recipes':     len(recipes),
        'total_chars':   len(full_text),
        'train_tokens':  len(train_ids),
        'val_tokens':    len(val_ids),
        'baseline_loss': float(baseline_loss),
        'sample_recipe': recipes[0],
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  train.bin: {os.path.getsize(train_path) / 1024:.0f} KB")
    print(f"  val.bin:   {os.path.getsize(val_path) / 1024:.0f} KB")
    print(f"  meta.pkl")
    print(f"\n✓ Data preparation complete! Run 'python train.py' next.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare recipe dataset for GPT training')
    parser.add_argument(
        '--n_recipes', type=int, default=N_RECIPES,
        help=f'Number of recipes to use (default: {N_RECIPES})'
    )
    args = parser.parse_args()
    main(n_recipes=args.n_recipes)
