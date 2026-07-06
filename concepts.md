# Concepts Glossary

> Core ML/transformer concepts used in this project, explained simply.
> No assumed background beyond basic programming.

---

## Loss & Metrics

### Cross-Entropy Loss

The training signal. Measures "how surprised was the model by the correct answer?"

```
loss = -log P(correct_next_char | context)
```

- If the model assigns probability 1.0 to the correct char → loss = 0 (perfect)
- If the model assigns probability 0.01 → loss = 4.6 (very surprised)
- **Baseline** (random model, uniform over 110 chars): `-log(1/110) = ln(110) ≈ 4.70`

All token positions contribute — one batch of 64 sequences × 256 positions = 16,384 individual loss values, averaged into one scalar.

### Perplexity

`perplexity = exp(loss)`

More interpretable than loss. Answers: "how many equally likely options does the model think there are?"

| Loss | Perplexity | Interpretation |
|------|-----------|----------------|
| 4.70 | 110× | Pure random — considers all 110 chars equally likely |
| 2.30 | 10× | Narrowed to ~10 plausible chars |
| 1.00 | 2.7× | Very confident, ~2-3 real options |
| 0.49 | 1.6× | Our trained model — highly confident |

### Train vs Val Loss

**Train loss**: measured on training data (data the model has seen).  
**Val loss**: measured on held-out validation data (data the model has NOT trained on).

The gap between them is the **generalization gap**:
- Small gap (ours: ~0.03): model is learning rules, not memorizing
- Large gap: **overfitting** — model has memorized training examples but fails on new ones

---

## Architecture

### Embedding

A lookup table: each integer maps to a vector.

```python
nn.Embedding(110, 384)   # 110 chars, each maps to a 384-dim vector
```

Initially random. Through training, the model learns that e.g. `'a'` and `'e'` should have similar vectors (both are vowels in similar contexts).

**Token embedding**: what is this character?  
**Position embedding**: where in the sequence is it?  
Both are summed together — the model learns to separate these signals during training.

### Self-Attention

The mechanism that lets each token "look at" other tokens and decide how much to incorporate their information.

Three roles per token:
- **Query (Q)**: "What information am I looking for?"
- **Key (K)**: "What information do I have?"
- **Value (V)**: "What do I actually share if selected?"

The attention weight from token `i` to token `j` is how well `Q_i` matches `K_j`:
```
score(i,j) = softmax( Q_i · K_j / sqrt(d_k) )
output_i   = Σ_j score(i,j) × V_j
```

In recipe terms: when generating `INSTRUCTIONS:`, the model can query back to `TITLE:` to know what dish it's making before predicting the next word.

### Causal Masking

Self-attention without masking would let each token see the future — making training trivial (just copy from the next token) but inference impossible.

The causal mask zeroes out future attention weights. Token at position `t` can only attend to positions `0..t`. This is what makes the model "decoder-only" — it predicts the future given only the past.

### Residual Connection

```python
x = x + sublayer(x)   # NOT just: x = sublayer(x)
```

Without residuals: in a 6-layer network, gradients must pass through all 6 layers during backprop. Each layer's Jacobian is typically <1 → gradients vanish exponentially.

With residuals: the gradient of the loss has a direct path to every layer via the skip connections. Each layer only needs to learn the "correction" (δx), not recompute everything from scratch.

### LayerNorm

Normalizes each token's representation to zero mean, unit variance, then scales and shifts with learned parameters.

```python
# For each token's 384-dim vector:
x_normalized = (x - mean(x)) / std(x)
out = gamma * x_normalized + beta   # gamma, beta are learned
```

**Why?** Keeps activations in a stable range regardless of depth. Without it, activations in a 6-layer network can explode or collapse.

**Pre-LN** (this model): applied *before* each sublayer. More training-stable than the original paper's Post-LN.

### Feed-Forward Network (FFN)

Position-wise: applied to each token independently, no cross-token interaction.

```
Linear(384 → 1536) → GELU → Linear(1536 → 384)
```

Role: after attention (tokens *communicate*), FFN lets each token *process* what it gathered. The 4× expansion gives a wider "thinking space."

### Multi-Head Attention

Running `n_head=6` independent attention heads in parallel, then concatenating.

Each head can specialize in a different type of relationship. One might track local character patterns (spelling), another tracks structural recipe markers (`TITLE:` → `INGREDIENTS:` → `INSTRUCTIONS:`), another tracks ingredient types.

---

## Training

### Gradient Descent

Training loop:
1. Forward pass: compute loss
2. Backward pass: compute how each weight contributed to the loss (gradients)
3. Update: nudge weights in the direction that reduces loss

`loss.backward()` computes gradients. `optimizer.step()` applies the update.

### AdamW

An optimizer — the algorithm that updates weights given gradients.

Adam tracks two things per weight:
- **First moment** (m): exponential moving average of gradients (momentum)
- **Second moment** (v): exponential moving average of squared gradients (adaptive LR)

Update: `w = w - lr * m / sqrt(v + ε)`

The "adaptive" part means weights with high variance gradients get smaller updates (more careful), weights with consistent gradients get larger updates (more confident).

**AdamW vs Adam**: AdamW applies weight decay *after* the Adam update (decoupled). Standard Adam + L2 regularization incorrectly couples decay with the adaptive learning rate.

### Weight Decay

Adds `lambda * ||w||²` to the loss — penalizes large weights.

Effect: regularization. Prevents any single weight from dominating → forces the model to distribute information across many weights → better generalization.

**Only applied to weight matrices (2D tensors)**, not biases or LayerNorm parameters. Decaying a bias toward zero is actively harmful.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the L2 norm of all gradients exceeds 1.0, scale all gradients down proportionally.

Prevents "gradient explosions" — rare events where a particularly bad batch produces destructively large weight updates.

### Cosine Learning Rate Schedule

```
Phase 1: Linear warmup  (0 → lr over warmup_iters steps)
Phase 2: Cosine decay   (lr → min_lr over lr_decay_iters steps)
Phase 3: Floor          (stays at min_lr)
```

**Why warmup?** At random initialization, gradient noise is high. A small LR at the start prevents large destructive updates before the model has any useful structure.

**Why cosine?** Smooth decay avoids abrupt LR drops. The cosine curve decreases slowly at first, faster in the middle, slowly at the end — spending more time near the good region.

### Dropout

During training, randomly zeroes `p` fraction of activations at each step.

Forces the network to not rely on any single activation path → reduces co-adaptation of neurons → better generalization.

Disabled during inference (`model.eval()` handles this automatically).

---

## Tokenization

### Character-Level (this project)

Each unique character → one integer. Vocabulary ~90-110 chars.

**Pros**: simple, no library needed, easy to understand, interesting to watch learning happen character by character  
**Cons**: large vocabulary relative to information content; model must learn to spell every word; higher loss values

### Byte-Pair Encoding (BPE)

Learns to merge common character pairs into subwords iteratively.

`h + e + l + l + o → he + ll + o → hell + o → hello` (if common enough)

GPT-2 BPE (tiktoken): vocab size 50,257. Common words (`"the"`, `" ingredients"`) are single tokens.

**Pros**: much lower loss, word-level patterns visible immediately, faster training for same quality  
**Cons**: black-box tokenizer, harder to inspect, less dramatic "learning from scratch" story

---

## Inference

### Temperature Sampling

```
logits = logits / temperature
probs  = softmax(logits)
next   = sample(probs)
```

- `T < 1.0`: divides logits by a number <1 → makes them larger → softmax more peaked → more deterministic
- `T = 1.0`: no change — raw model probabilities
- `T > 1.0`: divides by a number >1 → smaller logits → softmax flatter → more random

### Top-K Sampling

Set all logits below the k-th largest to `-inf` before softmax.

Prevents the model from ever sampling very improbable characters (garbage characters, rare unicode). Common values: 40–100.

### Greedy Decoding

`top_k=1` — always pick the most probable next character.

Generates the "safest" output but is fully deterministic and tends to produce repetitive, generic recipes.

---

## Implementation Patterns

### `register_buffer` vs parameter

```python
self.register_buffer('tril', torch.tril(torch.ones(T, T)))
```

`register_buffer`: saved with `state_dict()`, moves to GPU with `.to(device)`, but **not updated by the optimizer**.

Used for the causal mask — it's a fixed matrix, not something to learn.

### `@torch.no_grad()`

Disables gradient tracking for everything inside the decorated function.

Used for `estimate_loss()` and `generate()`. Faster (no activation storage for backprop) and uses less memory.

### `set_to_none=True` in zero_grad

```python
optimizer.zero_grad(set_to_none=True)
```

Deallocates gradient tensors entirely instead of writing zeros. Slightly faster because it avoids a GPU kernel launch.

### `model.eval()` / `model.train()`

Switches dropout (and BatchNorm, if used) between training and inference mode.

Always call `model.eval()` before evaluating loss or generating — otherwise dropout randomly zeros activations and gives noisy results.

### `np.memmap` for data loading

```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

Memory-maps the file — reads from disk on demand rather than loading everything into RAM. Essential for large datasets (>1GB). Less important for our 32MB `train.bin` but correct practice.
