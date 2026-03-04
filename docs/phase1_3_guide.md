# Phase 1.3 — Transformer Forward Pass: Technical Guide

This document explains the forward pass implementation — how token IDs become logits.

**Source files:**

- [weights.rs](../src/transformer/weights.rs) — `TransformerWeights`, `LayerWeights`
- [forward.rs](../src/transformer/forward.rs) — `forward()`, `attention()`, `feed_forward()`, `generate_greedy()`

---

## High-Level Flow

```
token_ids → embedding lookup → hidden_state [embed_dim]
for each layer (0..block_count):
    attn_input  = rms_norm(hidden_state)
    attn_output = multi_head_attention(attn_input)
    hidden_state += attn_output                      ← residual connection
    ffn_input   = rms_norm(hidden_state)
    ffn_output  = swiglu_ffn(ffn_input)
    hidden_state += ffn_output                       ← residual connection
logits = output_weight × rms_norm(hidden_state)      ← [vocab_size] vector
```

See [forward()](../src/transformer/forward.rs#L12-L60) for the implementation.

---

## Weight Loading

[TransformerWeights::load](../src/transformer/weights.rs#L29-L65) uses `load_tensor_f32` (from Phase 1.2) to dequantize every tensor from the GGUF file into f32. It maps GGUF tensor names to struct fields:

| GGUF Name                 | Struct Field         | Shape                    |
| ------------------------- | -------------------- | ------------------------ |
| `token_embd.weight`       | `token_embedding`    | `[vocab, embed_dim]`     |
| `blk.{i}.attn_q.weight`   | `layers[i].attn_q`   | `[embed_dim, embed_dim]` |
| `blk.{i}.attn_k.weight`   | `layers[i].attn_k`   | `[kv_dim, embed_dim]`    |
| `blk.{i}.ffn_gate.weight` | `layers[i].ffn_gate` | `[ffn_dim, embed_dim]`   |
| `output.weight`           | `output`             | `[vocab, embed_dim]`     |

**Weight tying:** some models reuse `token_embd.weight` as `output.weight`. We detect this and clone.

---

## Multi-Head Attention

[attention()](../src/transformer/forward.rs#L63-L129) implements the full attention mechanism for one position:

### Step 1: Q/K/V Projections

```
Q = W_q × x    → [n_heads * head_dim]
K = W_k × x    → [n_kv_heads * head_dim]
V = W_v × x    → [n_kv_heads * head_dim]
```

### Step 2: RoPE

Q and K are rotated by position-dependent angles (from Phase 1.2's `rope()`). This encodes position without additive embeddings.

### Step 3: Scaled Dot-Product Attention (per head)

```
score = dot(Q_head, K_head) / sqrt(head_dim)
weights = softmax(scores)              ← over all positions 0..=pos
output_head = sum(weights[p] * V_head[p])
```

**Causal masking** is implicit: we only compute attention over positions `0..=pos`, so the model can't look ahead.

### Step 4: GQA (Grouped Query Attention)

When `n_kv_heads < n_heads`, multiple Q heads share one K/V head. The mapping: `kv_head = q_head / (n_heads / n_kv_heads)`. This saves memory proportional to the ratio.

### Step 5: Output Projection

```
attn_output = W_o × concatenated_head_outputs
```

---

## SwiGLU Feed-Forward Network

[feed_forward()](../src/transformer/forward.rs#L132-L139) implements the gated FFN used in LLaMA:

```
gate = W_gate × x       → [ffn_dim]
up   = W_up × x         → [ffn_dim]
hidden = silu(gate) * up → [ffn_dim]    ← element-wise gate
output = W_down × hidden → [embed_dim]
```

The SiLU-gated element-wise multiply (SwiGLU) outperforms standard ReLU FFN in practice.

---

## Residual Connections

After each attention block and FFN block, the original input is added to the output:

```
hidden = hidden + attn_output
hidden = hidden + ffn_output
```

This creates "shortcut" paths for gradients during training and information during inference. Without residuals, deep transformers (30+ layers) would suffer from vanishing gradients during training and degenerate representations during inference.

---

## Greedy Decoding

[generate_greedy()](../src/transformer/forward.rs#L142-L170) iteratively generates tokens:

1. Run `forward()` on the full token sequence → logits
2. Pick `argmax(logits)` as the next token
3. Append it to the sequence
4. Repeat until `max_tokens` or EOS

**No KV-cache yet** — each step recomputes attention for all positions. This is O(n²) per step (n = sequence length). Phase 2 adds KV-cache to make it O(n) per step.

---

## Performance Notes (Naive)

The current implementation is **correct but slow** by design:

- **No KV-cache:** O(n²) recomputation per step
- **All dequantized to f32:** entire model lives in RAM as f32
- **No SIMD:** scalar matmul loops
- **No threading:** single-threaded

For SmolLM-135M (30 layers, embed_dim=576), expect ~seconds per token on CPU. Optimization comes in Phase 2.
