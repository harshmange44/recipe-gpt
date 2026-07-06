# Recipe GPT — Architecture Reference

> A complete map of the model: every class, every tensor shape, every design decision.
> Read this alongside `recipe-gpt/model.py`.

---

## Big Picture

Recipe GPT is a **decoder-only transformer** — the same family as GPT-2 and GPT-3.

- **Input**: a sequence of characters (e.g. `T I T L E : ` ...)
- **Output**: a probability distribution over the next character at every position
- **Training signal**: cross-entropy loss (how surprised was the model by the actual next char?)
- **Inference**: repeatedly sample the next character and append it

---

## Data Flow (End to End)

```
Raw text: "TITLE: Mac and Cheese\nINGREDIENTS: 2 cups..."
               ↓ character → integer (stoi map)
Token IDs: [84, 73, 84, 76, 69, 58, ...] — dtype uint16, shape (B, T)
               ↓ token_embedding  (vocab_size=110, n_embd=384)
               + position_embedding (block_size=256, n_embd=384)
               ↓ dropout
            x : (B, T, 384)
               ↓
    ┌──────────────────────────────┐
    │  Transformer Block  ×6       │
    │  ┌──────────────────────┐    │
    │  │ LayerNorm(384)       │    │
    │  │ MultiHeadAttention   │    │
    │  │   6 heads × 64 dim   │    │
    │  └──────────────────────┘    │
    │  + residual                  │
    │  ┌──────────────────────┐    │
    │  │ LayerNorm(384)       │    │
    │  │ FeedForward          │    │
    │  │   384 → 1536 → 384   │    │
    │  └──────────────────────┘    │
    │  + residual                  │
    └──────────────────────────────┘
               ↓
            x : (B, T, 384)
               ↓ LayerNorm (final)
               ↓ Linear (384 → 110)
           logits : (B, T, 110)
               ↓ cross_entropy vs targets
            loss : scalar
```

---

## Components

### 1. Tokenizer (char-level)

Not a class — lives in `data/prepare.py` and `meta.pkl`.

```python
vocab = sorted(set(full_text))   # 110 unique chars
stoi  = { ch: i for i, ch in enumerate(vocab) }   # char → int
itos  = { i: ch for i, ch in enumerate(vocab) }   # int → char
```

| Property | Value |
|----------|-------|
| Vocab size | 110 chars |
| Includes | A-Z, a-z, 0-9, punctuation, `¼ ½ ¾ ;` (from recipe fractions) |
| Baseline loss | `ln(110) = 4.70` |

---

### 2. Embeddings (`GPT.__init__`)

```
token_embedding:    nn.Embedding(110, 384)   →  "what is this character?"
position_embedding: nn.Embedding(256, 384)   →  "where is this character?"
```

Both are learned tables. At position `t` with character `c`:
```
x[t] = token_embedding[c] + position_embedding[t]
```

**Why add, not concatenate?**  
Adding preserves dimensionality (stays at 384). The model learns to disentangle identity from position through training.

---

### 3. Head — Single Attention Head

**File**: `model.py` · **Class**: `Head`

```
Input:  (B, T, 384)

Q = Linear(384, 64)  →  (B, T, 64)   "what am I looking for?"
K = Linear(384, 64)  →  (B, T, 64)   "what do I contain?"
V = Linear(384, 64)  →  (B, T, 64)   "what do I communicate?"

scores = Q @ K^T / sqrt(64)           (B, T, T)   raw affinities
scores = masked_fill(future=-inf)      (B, T, T)   causal mask
weights = softmax(scores, dim=-1)      (B, T, T)   sums to 1 per row
output  = weights @ V                  (B, T, 64)
```

**The causal mask** — lower triangular, blocks future tokens:
```
T=4 example:
[[1, 0, 0, 0],    ← position 0 sees only itself
 [1, 1, 0, 0],    ← position 1 sees 0,1
 [1, 1, 1, 0],    ← position 2 sees 0,1,2
 [1, 1, 1, 1]]    ← position 3 sees all
```
Positions with 0 become `-inf` before softmax → 0 weight after softmax.

**The scale factor `sqrt(64)`**: dot products grow as `d_k` grows. Dividing keeps softmax from saturating (going too peaked), which would starve gradients.

---

### 4. MultiHeadAttention

**File**: `model.py` · **Class**: `MultiHeadAttention`

```
6 × Head  →  each produces (B, T, 64)
cat(dim=-1)  →  (B, T, 384)
Linear(384, 384)  →  (B, T, 384)   output projection
```

**Why 6 heads?** Each head specializes. In a trained recipe model:
- Some heads learn **local patterns** (adjacent characters form words)
- Some heads learn **structural patterns** (`TITLE:` attends back when generating `INGREDIENTS:`)
- Some heads track **ingredient context** during `INSTRUCTIONS:`

---

### 5. FeedForward

**File**: `model.py` · **Class**: `FeedForward`

```
Linear(384, 1536)   expand 4×
GELU()
Linear(1536, 384)   compress back
Dropout(0.2)
```

Applied **independently at every position** — no cross-token interaction here.

**Role**: After attention (tokens *communicate*), FFN lets each token *think* about what it gathered without involving its neighbors.

**GELU vs ReLU**: GELU is smooth at 0 — small negative inputs still receive gradient. Empirically better for transformers. Used in GPT-2, BERT, LLaMA.

---

### 6. Block

**File**: `model.py` · **Class**: `Block`

```python
x = x + self.attn(self.ln1(x))   # attention + residual
x = x + self.ffwd(self.ln2(x))   # FFN + residual
```

**Pre-LayerNorm** (applied before sublayer, not after): more training-stable than the original "Attention Is All You Need" paper's post-LN. Used in all modern GPT variants.

**Residual connections** (`x = x + sublayer(x)`): gradient flows directly through addition, bypassing sublayers. Prevents vanishing gradients in 6-layer-deep network.

---

### 7. GPT — Full Model

**File**: `model.py` · **Class**: `GPT`

```
token_embedding:    Embedding(110, 384)       40,320 params
position_embedding: Embedding(256, 384)       98,304 params
blocks × 6:                                   ~10.3M params
  ├── ln1 + ln2:        LayerNorm(384) × 2   1,536 params each
  ├── MultiHeadAttention:                    ~590K params per block
  │   ├── 6 × Head (Q, K, V)  3 × (384×64)
  │   └── proj                 (384×384)
  └── FeedForward:                           ~1.18M params per block
      ├── fc1  (384×1536)
      └── fc2  (1536×384)
ln_f:               LayerNorm(384)            768 params
lm_head:            Linear(384, 110)          42,240 params
─────────────────────────────────────────────────────
Total:                                        ~10.82M parameters
```

---

## Tensor Shapes — Quick Reference

| Stage | Shape | Notes |
|-------|-------|-------|
| Input token IDs | `(B, T)` | B=64, T=256 during training |
| Token embedding | `(B, T, 384)` | lookup from 110-row table |
| Position embedding | `(T, 384)` | broadcast-added to all B |
| After embedding | `(B, T, 384)` | input to transformer blocks |
| Q / K / V per head | `(B, T, 64)` | 64 = 384 / 6 heads |
| Attention scores | `(B, T, T)` | every pair, causal masked |
| Attention weights | `(B, T, T)` | after softmax, rows sum to 1 |
| Head output | `(B, T, 64)` | weighted sum of values |
| Multi-head concat | `(B, T, 384)` | 6 × 64 |
| FFN intermediate | `(B, T, 1536)` | 4× expansion |
| FFN output | `(B, T, 384)` | back to model dim |
| Final logits | `(B, T, 110)` | one distribution per position |
| For loss (flat) | `(B×T, 110)` vs `(B×T,)` | cross_entropy input format |

---

## Weight Initialization

Two rules, both important:

**Rule 1**: All Linear and Embedding weights initialized to `N(0, 0.02)`.  
Small std means activations start near zero → gradients flow without saturation.

**Rule 2**: Residual projection weights scaled by `1 / sqrt(2 × n_layer)`.  
Each Block adds its output to the residual stream. Without scaling, variance grows as `O(n_layer)`. This keeps it constant regardless of depth. The `2` accounts for two residual additions per block (attention + FFN).

---

## Optimizer: AdamW with Weight Decay Groups

```python
# 2D tensors (weight matrices) → apply L2 regularization
decay_params   = [p for p in params if p.dim() >= 2]

# 1D tensors (biases, LayerNorm scale/shift) → NO regularization
nodecay_params = [p for p in params if p.dim() < 2]
```

**Why separate?** Decaying a bias toward zero is actively harmful (biases encode useful offsets). LayerNorm parameters set the output distribution scale — regularizing them doesn't help.

---

## Design Decisions vs Paper Defaults

| Decision | Original Paper | This Model | Why |
|----------|---------------|------------|-----|
| LayerNorm position | Post-LN | **Pre-LN** | More stable training |
| Activation | ReLU | **GELU** | Smoother gradients, better empirical results |
| Positional encoding | Sinusoidal | **Learned** | Works as well, simpler to implement |
| Attention impl | Manual loops | **Per-head classes** | Clearer for learning; allows weight capture for visualization |
| Flash Attention | N/A | **Not used** | Incompatible with attention weight capture for heatmaps |
| Weight tying | N/A | **Not used** | Kept separate for clarity (nanoGPT does tie them) |
