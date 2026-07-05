"""
config.py — Hyperparameters for Recipe GPT

Two dataclasses:
  GPTConfig   — defines the model architecture
  TrainConfig — defines the training procedure

Why dataclasses?
  Clean, self-documenting, type-checked, and can be serialized to dict
  (via dataclasses.asdict()) for safe checkpoint storage — no import
  dependency issues when loading old checkpoints.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPTConfig:
    """
    Architecture hyperparameters for the GPT model.

    These define the model's capacity (how many parameters it has)
    and the relationships between its components.
    """

    # --- Sequence ---
    block_size: int = 256
    # The maximum number of tokens the model can "see" at once (context window).
    # Increasing this allows the model to use longer context, but attention
    # cost scales as O(block_size^2) — quadratic in memory and time.
    # 256 chars ≈ ~2–3 recipe lines: enough to learn structure, fast to train.

    vocab_size: Optional[int] = None
    # Set at runtime from data/processed/meta.pkl after prepare.py runs.
    # For char-level recipe data, expect ~90–100 unique characters.

    # --- Architecture ---
    n_layer: int = 6
    # Number of transformer blocks stacked on top of each other.
    # Each block adds one round of "attention + FFN". More layers = deeper
    # reasoning capacity, but also longer to train and more prone to overfitting.

    n_head: int = 6
    # Number of attention heads per block. Must divide n_embd evenly.
    # Each head specializes in different attention patterns (positional,
    # semantic, structural, etc.).

    n_embd: int = 384
    # Embedding dimension: the "width" of the model. Every token is
    # represented as a vector of this size throughout the network.
    # 384 / 6 heads = 64 per head (standard head_size for this scale).

    dropout: float = 0.2
    # Probability of randomly zeroing activations during training.
    # Acts as regularization — prevents co-adaptation of neurons.
    # Set to 0.0 at inference time (model.eval() handles this automatically).

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}). "
            f"Each head gets n_embd // n_head = {self.n_embd} // {self.n_head} dimensions."
        )

    @property
    def head_size(self) -> int:
        """Dimensionality of each attention head's Q/K/V projections."""
        return self.n_embd // self.n_head

    def describe(self):
        """Print a human-readable summary of the model configuration."""
        # Rough parameter count estimate
        # Token emb: vocab_size * n_embd
        # Pos emb:   block_size * n_embd
        # Per block: ~4 * n_embd^2 * 2 (attn + ffn)
        # LM head:   n_embd * vocab_size
        if self.vocab_size:
            emb = (self.vocab_size + self.block_size) * self.n_embd
            blocks = self.n_layer * (4 * self.n_embd ** 2 * 2)
            head = self.n_embd * self.vocab_size
            approx_params = emb + blocks + head
            print(f"  Approx params: {approx_params/1e6:.1f}M")
        print(f"  Layers:        {self.n_layer}")
        print(f"  Heads:         {self.n_head} × {self.head_size} dim each")
        print(f"  Embedding:     {self.n_embd}")
        print(f"  Context:       {self.block_size} chars")
        print(f"  Dropout:       {self.dropout}")


@dataclass
class TrainConfig:
    """
    Training procedure hyperparameters.

    These control HOW the model learns, not what it looks like.
    """

    # --- Paths ---
    data_dir: str = 'data/processed'
    # Where prepare.py saved train.bin, val.bin, meta.pkl

    out_dir: str = 'out'
    # Where checkpoints and loss logs are saved

    # --- Training loop ---
    max_iters: int = 5000
    # Total gradient update steps. With batch_size=64 and ~1.8M train tokens,
    # one pass through data ≈ 1.8M / (64 * 256) ≈ 110 steps.
    # 5000 steps ≈ ~45 passes through the data (healthy for small datasets).

    batch_size: int = 64
    # Number of independent sequences processed in parallel per step.
    # Each sequence is block_size=256 chars → 64*256 = 16,384 chars/step.

    # --- Learning rate schedule ---
    learning_rate: float = 1e-3
    # Peak learning rate. 1e-3 is higher than GPT-2's 6e-4 because our
    # model is smaller — smaller models can tolerate larger LRs.

    warmup_iters: int = 100
    # LR linearly ramps from 0 → learning_rate over this many steps.
    # Prevents large destructive updates when model weights are random.

    lr_decay_iters: int = 5000
    # LR decays (cosine) from learning_rate → min_lr over this many steps.
    # Set equal to max_iters: keep decaying until training ends.

    min_lr: float = 1e-4
    # Floor learning rate. Chinchilla recommendation: ~learning_rate / 10.
    # Never fully stop learning — long-tail patterns still need small updates.

    # --- Regularization ---
    weight_decay: float = 0.1
    # L2 penalty applied to weight matrices (not biases or LayerNorm params).
    # Shrinks large weights toward zero, reducing overfitting.

    grad_clip: float = 1.0
    # Maximum gradient L2 norm. If gradients exceed this, scale them down.
    # Prevents rare "gradient explosions" that can destabilize training.

    beta1: float = 0.9
    # AdamW first moment decay (momentum). Standard value.

    beta2: float = 0.99
    # AdamW second moment decay. Slightly higher than default (0.999) because
    # our dataset is small — fewer unique gradient patterns to estimate variance.

    # --- Evaluation ---
    eval_interval: int = 250
    # How many training steps between loss evaluations.
    # Each eval uses eval_iters batches for a stable loss estimate.

    eval_iters: int = 200
    # Number of batches averaged per loss estimate.
    # More → less noisy estimate, but slower evaluation.

    log_interval: int = 10
    # Print training loss every N steps (cheap — just the current batch loss).

    always_save_checkpoint: bool = False
    # If True: save after every eval (safe but uses disk space).
    # If False: only save when val_loss improves (default for small datasets).

    # --- System ---
    device: str = 'auto'
    # 'auto' → detects MPS (Apple Silicon) > CUDA > CPU automatically.
    # Override: 'cpu', 'mps', 'cuda', 'cuda:0', etc.

    compile: bool = False
    # torch.compile() (PyTorch 2.0+) can give 2x speedup on CUDA.
    # Disabled by default — not stable on MPS, and adds complexity.

    seed: int = 1337
    # Random seed for reproducibility (Karpathy's signature seed).
