"""
model.py — Recipe GPT: Character-Level Language Model
======================================================

Architecture: Decoder-only Transformer (GPT family)
Dataset:      Cooking recipes (char-level tokenization)

References:
  [1] Vaswani et al., "Attention Is All You Need" (2017) — original Transformer
  [2] Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
  [3] Karpathy, "Let's build GPT: from scratch, in code, spelled out" (2023)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig, TrainConfig


# =============================================================================
# Head — Single Self-Attention Head
# =============================================================================

class Head(nn.Module):
    """
    One head of causal self-attention.

    Self-attention is the mechanism that allows each token to "look at"
    other tokens and decide how much to incorporate their information.

    The core idea: every token emits three vectors:
      Q (query)  — "What kind of information am I looking for?"
      K (key)    — "What kind of information do I have?"
      V (value)  — "What information do I actually pass along?"

    The attention weight between token i and token j is:
        score(i,j) = softmax( Q_i · K_j / sqrt(d_k) )

    The output at position i is: sum over j of score(i,j) * V_j

    Intuition with recipes:
      When the model sees "INSTRUCTIONS:", it might attend strongly to
      the TITLE and INGREDIENTS tokens — learning "what am I making?"
      and "what do I have?" before generating cooking steps.

    "Causal" means each token can only attend to itself and past tokens —
    it cannot look into the future. This makes generation possible.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hs = config.head_size  # dimension of Q, K, V projections

        # Three learned linear projections — no bias (standard in transformers)
        # These learn WHAT each token looks for, contains, and communicates.
        self.key   = nn.Linear(config.n_embd, hs, bias=False)
        self.query = nn.Linear(config.n_embd, hs, bias=False)
        self.value = nn.Linear(config.n_embd, hs, bias=False)

        # Causal mask: lower-triangular matrix of ones.
        #
        # Example (T=4):
        #   [[1, 0, 0, 0],
        #    [1, 1, 0, 0],
        #    [1, 1, 1, 0],
        #    [1, 1, 1, 1]]
        #
        # Position 0 can only see itself.
        # Position 2 can see positions 0, 1, 2 — not 3.
        #
        # register_buffer: NOT a learnable parameter, but moves to GPU with
        # .to(device) and is saved/loaded with the model's state_dict.
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

        # Store the last computed attention weights (used by visualize.py)
        self.last_attn_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — batch of sequences, each token as a C-dim vector

        Returns:
            out: (B, T, head_size) — attention-weighted value vectors
        """
        B, T, C = x.shape

        k = self.key(x)    # (B, T, hs) — "what I contain"
        q = self.query(x)  # (B, T, hs) — "what I'm looking for"

        # ── Scaled dot-product attention scores ──────────────────────────────
        # Each score[b, t_q, t_k] = how much position t_q attends to t_k.
        # Shape: (B, T, hs) @ (B, hs, T) → (B, T, T)
        #
        # Why scale by 1/sqrt(head_size)?
        #   Without scaling, dot products grow in magnitude as head_size grows,
        #   pushing softmax into a near-one-hot distribution (vanishing gradients).
        #   Dividing by sqrt(d_k) keeps the variance of dot products at ~1.
        #   See [1] §3.2.1.
        scale = k.shape[-1] ** -0.5           # = 1 / sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * scale # (B, T, T)

        # ── Causal masking ───────────────────────────────────────────────────
        # Positions where tril == 0 are "future" tokens — set their scores
        # to -inf so softmax assigns them 0 weight.
        # [:T, :T] handles the case where T < block_size (e.g. short prompts).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # ── Normalize to probabilities ────────────────────────────────────────
        # Softmax along dim=-1: for each query position, the attention weights
        # over all key positions sum to 1.
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Store for visualization (detached — no gradient needed)
        self.last_attn_weights = wei.detach()

        # ── Weighted aggregation ──────────────────────────────────────────────
        # The output at each position is a weighted combination of Value vectors.
        v   = self.value(x)    # (B, T, hs)
        out = wei @ v          # (B, T, T) @ (B, T, hs) → (B, T, hs)
        return out


# =============================================================================
# MultiHeadAttention — Parallel Attention Heads
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel, outputs concatenated.

    Why multiple heads?
      A single attention head can only learn one type of relationship.
      Multiple heads allow the model to simultaneously attend to:
        - Syntactic structure (subject-verb agreement)
        - Semantic similarity (ingredient types)
        - Positional patterns (nearby tokens)
        - Structural markers (TITLE: → INGREDIENTS: → INSTRUCTIONS:)
        ... and more

    Each head has head_size = n_embd // n_head dimensions.
    Concatenating all heads gives back n_embd dimensions total.

    The output projection (self.proj) mixes information across heads,
    allowing them to "collaborate" on the final representation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # Output projection: (n_head * head_size, n_embd) = (n_embd, n_embd)
        self.proj  = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head: (B, T, n_embd) → (B, T, head_size)
        # Concatenate along feature dim: → (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Mix head outputs and apply dropout
        out = self.dropout(self.proj(out))
        return out

    def get_attn_weights(self) -> torch.Tensor:
        """
        Return the last-computed attention weights from all heads.
        Shape: (n_head, B, T, T)
        Requires a forward pass to have happened first.
        """
        weights = [h.last_attn_weights for h in self.heads
                   if h.last_attn_weights is not None]
        if not weights:
            return None
        return torch.stack(weights, dim=0)


# =============================================================================
# FeedForward — Position-wise FFN
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN).

    After self-attention (tokens "communicate"), the FFN lets each token
    independently "think" about what it gathered — processing its updated
    representation without interaction with other positions.

    Structure:
      Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout

    The 4x inner expansion comes from the original Transformer paper [1].
    It gives the model a wider "thinking space" before projecting back.

    GELU vs ReLU:
      ReLU hard-zeros all negative inputs (non-differentiable at 0).
      GELU (Gaussian Error Linear Unit) is a smooth approximation:
        GELU(x) ≈ x * Φ(x)   where Φ is the Gaussian CDF
      This allows small gradient signals from slightly-negative inputs,
      which empirically improves transformer training. Used in GPT-2, BERT, etc.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # expand
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),  # compress back
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applied independently at each position: (B, T, C) → (B, T, C)
        return self.net(x)


# =============================================================================
# Block — One Transformer Layer
# =============================================================================

class Block(nn.Module):
    """
    One complete transformer block.

    Karpathy's framing: "communication followed by computation."
      1. Self-attention  → tokens exchange information with each other
      2. Feed-forward   → each token processes what it gathered

    Two critical architectural choices in every block:

    ① Residual connections: x = x + sublayer(x)
       Without residuals, gradients in deep networks vanish during
       backpropagation (each layer multiplies gradients by its Jacobian,
       which is often < 1). Residuals add a direct path from input to
       output: ∂L/∂x_l = ∂L/∂x_L + (sum of block derivatives)
       — the gradient always has at least one non-vanishing path.

    ② Pre-LayerNorm (not Post-LN as in the original paper [1])
       LayerNorm is applied BEFORE each sublayer, not after.
       Benefits:
         - At initialization, the residual update is near zero →
           the whole block starts as an approximate identity function
         - More stable gradient flow (less "gradient explosion" at the
           residual branches early in training)
       Modern GPT variants (GPT-2, GPT-3, ...) all use Pre-LN.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN → attention → residual add
        x = x + self.attn(self.ln1(x))
        # Pre-LN → FFN → residual add
        x = x + self.ffwd(self.ln2(x))
        return x


# =============================================================================
# GPT — Full Language Model
# =============================================================================

class GPT(nn.Module):
    """
    Full decoder-only GPT language model.

    Forward pass data flow:
      token_ids (B, T)
        ↓ token_embedding table (vocab_size, n_embd)
      tok_emb (B, T, n_embd)   — "what is this token?"
        +
      pos_emb (T, n_embd)      — "where in the sequence is it?"
        ↓ n_layer × Block
      x (B, T, n_embd)         — contextualized representations
        ↓ LayerNorm
        ↓ lm_head Linear (n_embd → vocab_size)
      logits (B, T, vocab_size) — unnormalized next-token scores
        ↓ cross_entropy (if training)
      loss (scalar)

    Token embeddings:
      Each of the ~90 characters maps to a learned 384-dim vector.
      Initially random; the model learns that e.g. 'a' and 'A' are
      semantically similar (after training).

    Position embeddings:
      Each position 0..255 maps to a learned 384-dim vector.
      The model learns to encode "I am early in the title" vs
      "I am deep in the instructions" as distinct patterns.
      (This is "learned positional encoding" — sinusoidal is the
       alternative from [1] but learned works just as well in practice.)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None, (
            "vocab_size must be set before creating GPT. "
            "Run data/prepare.py first and load meta.pkl."
        )
        self.config = config

        # Core components
        self.token_embedding    = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop               = nn.Dropout(config.dropout)
        self.blocks             = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f   = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ── Weight initialization ──────────────────────────────────────────────
        # Initialize all Linear and Embedding weights to N(0, 0.02).
        # std=0.02 is small enough that:
        #   - Outputs start near zero (prevents saturation)
        #   - Gradients flow freely at initialization
        # This is standard for GPT-scale models (GPT-2 used std=0.02).
        self.apply(self._init_weights)

        # Special: scale DOWN residual projection weights.
        # Each Block adds its output to the residual stream. Without scaling,
        # variance accumulates: after L blocks, variance ≈ L * var(one_block).
        # Scaling by 1/sqrt(2*L) keeps total variance ≈ constant regardless of depth.
        # The factor of 2 accounts for TWO residual additions per block (attn + ffn).
        for name, param in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

        # Count and display parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"  Recipe GPT initialized")
        print(f"{'='*50}")
        print(f"  Parameters:    {n_params/1e6:.2f}M")
        print(f"  Layers:        {config.n_layer}")
        print(f"  Heads:         {config.n_head} × {config.head_size} dim")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Context size:  {config.block_size} chars")
        print(f"  Vocab size:    {config.vocab_size}")
        print(f"{'='*50}\n")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        Args:
            idx:     (B, T) — integer token indices in [0, vocab_size)
            targets: (B, T) — integer token indices, shifted left by 1
                     If provided: compute and return cross-entropy loss.
                     If None: return logits only (inference mode).

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar tensor if targets given, else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}. "
            f"Crop your input to {self.config.block_size} tokens max."
        )
        device = idx.device

        # ── Embeddings ────────────────────────────────────────────────────────
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)

        # Create position indices [0, 1, 2, ..., T-1] for this batch
        pos     = torch.arange(T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)  # (T, n_embd), broadcast over B

        # Combine: the model sees "token identity + where it appears"
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # ── Transformer blocks ────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)  # each block: communicate (attn) + compute (ffn)

        # ── Final layer norm + language model head ────────────────────────────
        x = self.ln_f(x)                 # (B, T, n_embd) — normalize last layer
        logits = self.lm_head(x)         # (B, T, vocab_size) — unnormalized scores

        # ── Loss (training only) ──────────────────────────────────────────────
        loss = None
        if targets is not None:
            # cross_entropy expects (N, C) and (N,) — flatten batch and time dims
            # logits: (B, T, vocab_size) → (B*T, vocab_size)
            # targets: (B, T) → (B*T,)
            #
            # cross_entropy = -log P(correct_token | context)
            # averaged over all B*T positions.
            #
            # Baseline (random model): -log(1/vocab_size) = log(vocab_size) ≈ 4.5
            # A well-trained model gets this down to ~1.5–1.8 on recipes.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def configure_optimizers(self, train_cfg: TrainConfig) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with separate weight decay groups.

        Weight decay (L2 regularization) shrinks weights toward zero,
        preventing overfitting. BUT it should only apply to weight matrices:

          DECAY:    2D+ tensors (Linear.weight, Embedding.weight)
                    → these are the "knowledge" matrices; regularize them
          NO DECAY: 1D tensors (Linear.bias, LayerNorm.weight, LayerNorm.bias)
                    → these are scale/shift params; decaying them is harmful

        Also: Adam + L2 ≠ AdamW.
          Standard Adam with L2 penalty couples the decay to the adaptive
          learning rate. AdamW decouples them, applying decay independently.
          This is the "correct" way (Loshchilov & Hutter, 2017).
        """
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': train_cfg.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {n_decay:,} params with weight decay, "
              f"{n_nodecay:,} params without")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens one at a time.

        Process:
          1. Forward pass → logits at last position
          2. Optionally apply temperature (scale logits)
          3. Optionally apply top-k filtering (zero out low-probability tokens)
          4. Sample from the resulting distribution
          5. Append sampled token, repeat

        Args:
            idx:            (B, T) starting context (encoded prompt)
            max_new_tokens: how many chars to generate
            temperature:    < 1.0 → sharper/safer, > 1.0 → more creative/random
            top_k:          if set, restrict sampling to the top-k most likely chars

        Temperature intuition:
          logits / T < 1  → probabilities more concentrated (model is "confident")
          logits / T > 1  → probabilities more spread (model is "uncertain")
          logits / T → 0  → argmax (always pick most likely char)
          logits / T → ∞  → uniform (completely random)

        Top-k intuition:
          top_k=1  → greedy (deterministic)
          top_k=40 → sample from the 40 most likely chars (balanced)
          top_k=None → sample from all vocab (can generate rare chars)
        """
        self.eval()  # disable dropout during generation

        for _ in range(max_new_tokens):
            # Crop context to block_size — model can't handle longer inputs
            idx_cond = idx[:, -self.config.block_size:]

            # Forward pass (no targets → no loss computation)
            logits, _ = self(idx_cond)

            # Focus on the LAST position: this predicts the NEXT character
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering: zero out everything outside the top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[:, [-1]] = the k-th largest value (threshold)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert to probabilities via softmax
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size), sums to 1

            # Sample from the distribution (not greedy — allows creativity)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token and continue
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx  # (B, T + max_new_tokens)

    @torch.no_grad()
    def get_attention_weights(self, idx: torch.Tensor) -> dict:
        """
        Run a forward pass and capture attention weights from every layer and head.

        Used by visualize.py to generate attention heatmaps.

        Returns:
            dict with keys 'layer_{i}_head_{j}' → numpy array of shape (T, T)
            Each array[t_query, t_key] = how much position t_query attended to t_key.
        """
        self.eval()

        # Crop to context window
        idx_cond = idx[:, -self.config.block_size:]
        T = idx_cond.shape[1]

        # Forward pass (attention weights are stored in each Head as a side effect)
        _ = self(idx_cond)

        # Collect weights from all layers and heads
        weights = {}
        for layer_idx, block in enumerate(self.blocks):
            head_stack = block.attn.get_attn_weights()  # (n_head, B, T, T) or None
            if head_stack is None:
                continue
            for head_idx in range(self.config.n_head):
                key = f'layer_{layer_idx}_head_{head_idx}'
                # Take first batch item → (T, T) numpy array
                weights[key] = head_stack[head_idx, 0].cpu().numpy()

        return weights
