# Phase 1.2 — Tensor Primitives: Technical Guide

This document explains the math operations implemented in `src/tensor/`.

**Source files:**

- [tensor.rs](../src/tensor/tensor.rs) — `Tensor` struct
- [ops.rs](../src/tensor/ops.rs) — all math operations
- [dequantize.rs](../src/tensor/dequantize.rs) — quantized format → f32 conversion

---

## Tensor Layout

The [Tensor](../src/tensor/tensor.rs#L9-L14) struct uses **row-major (C-order) contiguous storage**: a flat `Vec<f32>` plus shape and strides.

For shape `[3, 4]`:

- **Strides** = `[4, 1]` — moving one step in dim 0 jumps 4 elements, dim 1 jumps 1
- Element `[i, j]` at flat index `i * 4 + j`

Strides are computed by [compute_strides](../src/tensor/tensor.rs#L120-L126). Row-major means accessing along the last dimension is sequential in memory (cache-friendly).

Key methods: [from_vec](../src/tensor/tensor.rs#L30-L42) for construction, [row](../src/tensor/tensor.rs#L97-L103) for extracting rows from weight matrices, [reshape](../src/tensor/tensor.rs#L83-L95) for changing shape without copying data.

---

## Matrix Multiplication

**[matmul](../src/tensor/ops.rs#L7-L27)** — `[M, K] × [K, N] → [M, N]`, the core bottleneck of neural networks:

```
for each output row i (0..M):
    for each output col j (0..N):
        out[i,j] = sum(a[i,p] * b[p,j] for p in 0..K)
```

O(M×K×N) — roughly cubic. The naive triple loop is correct but slow; Phase 2 will add SIMD and threading.

**[matvec](../src/tensor/ops.rs#L30-L46)** — `[M, K] × [K] → [M]`, specialized for the common matrix × vector case (avoids the inner loop over output columns). During inference with batch size 1, almost all of our matmuls are actually matvecs.

---

## RMSNorm

**[rms_norm](../src/tensor/ops.rs#L59-L74)** — used in LLaMA instead of LayerNorm. Simpler — no mean subtraction, no learned bias:

```
rms = sqrt(mean(x²) + eps)
output = (x / rms) * weight
```

`eps` (typically 1e-5) prevents division by zero. **Why RMSNorm over LayerNorm?** ~15% faster (no mean computation, no bias) with comparable training dynamics. LLaMA, Mistral, and most modern models use it.

---

## SiLU (Swish) Activation

**[silu](../src/tensor/ops.rs#L78-L84)** — `x * sigmoid(x) = x / (1 + exp(-x))`

Properties: smooth, non-monotonic (unlike ReLU), slight negative values for small negatives.

LLaMA uses a **SwiGLU** FFN variant: `silu(gate_proj(x)) * up_proj(x)`, which is why this activation is needed.

---

## Softmax

**[softmax](../src/tensor/ops.rs#L88-L99)** — converts raw logits into a probability distribution:

```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

The `max(x)` subtraction is a **numerical stability trick** — prevents `exp()` from overflowing to infinity for large values. Mathematically equivalent (cancels out in the division).

Used in two places: **attention scores** (Q·K^T → weights) and **final output** (logits → token probabilities).

---

## RoPE (Rotary Positional Embeddings)

**[rope](../src/tensor/ops.rs#L108-L130)** — encodes token position by rotating pairs of dimensions in Q and K vectors:

```
For dimension pair (2i, 2i+1):
    freq = 1 / (base ^ (2i / dim))      ← high-freq for early dims, low-freq for later
    angle = position * freq
    x_2i'   = x_2i * cos(angle) - x_(2i+1) * sin(angle)
    x_(2i+1)' = x_2i * sin(angle) + x_(2i+1) * cos(angle)
```

This is a 2D rotation matrix applied to each pair. Key property: **the dot product between two rotated vectors depends only on their relative position**, giving the model translation-invariant positional awareness.

The `freq_base` (typically 10000) controls the frequency spectrum. Higher base = longer effective context.

---

## Embedding Lookup

**[embedding](../src/tensor/ops.rs#L134-L148)** — index into a `[vocab_size, embed_dim]` weight matrix by token ID. Returns a `[n_tokens, embed_dim]` tensor. Simply copies the corresponding rows.

---

## Dequantization

**[dequantize](../src/tensor/dequantize.rs#L12-L21)** — dispatcher that converts quantized bytes from GGUF back to f32 based on `GgmlType`.

### Q8_0 — [dequantize_q8_0](../src/tensor/dequantize.rs#L62-L82)

Each block of 32 values (34 bytes): `[f16 scale (2 bytes)] [32 × int8 (32 bytes)]`

Dequantize: `float = int8_value * scale`

### Q4_0 — [dequantize_q4_0](../src/tensor/dequantize.rs#L90-L115)

Each block of 32 values (18 bytes): `[f16 scale (2 bytes)] [16 × packed bytes (16 bytes)]`

Each byte packs two 4-bit unsigned ints (low nibble first), centered by subtracting 8 → signed range [-8, 7].

Dequantize: `float = (nibble - 8) * scale`

### BF16 — [dequantize_bf16](../src/tensor/dequantize.rs#L46-L55)

Same exponent range as f32, truncated mantissa. Convert by padding the 16-bit value with 16 zero bits.

---

## load_tensor_f32 Bridge

**[load_tensor_f32](../src/tensor/dequantize.rs#L118-L130)** connects the GGUF parser (Phase 1.1) to the tensor system:

1. Look up tensor info by name in `GgufModel`
2. Get raw byte slice from the mmap'd file
3. Dequantize based on `ggml_type`
4. Wrap into a `Tensor` with correct shape

This is used by the transformer forward pass in Phase 1.3 to load model weights.
