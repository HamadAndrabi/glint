# Tensors & Ops

Glint uses two tensor representations: `Tensor` for f32 activations and intermediate results, and `QuantizedTensor` for compressed weight storage. This page covers `Tensor` and the activation-path ops.

Source: `src/tensor/tensor.rs`, `src/tensor/ops.rs`, `src/tensor/flash.rs`

---

## The `Tensor` Type

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

Tensors are heap-allocated, contiguous, row-major f32 arrays. Strides are implicit (each dimension is fully packed). Shape is stored as a `Vec<usize>`.

Key methods:
- `Tensor::zeros(shape)` — allocate zeroed
- `Tensor::from_vec(data, shape)` — wrap existing data
- `t.data()` — `&[f32]` slice
- `t.data_mut()` — `&mut [f32]` slice
- `t.shape()` — `&[usize]`
- `t.numel()` — total element count

Activations are always f32. Only weight matrices are stored as `QuantizedTensor`.

---

## Matrix-Vector Multiply (matvec)

The hot path. `W × x → out` where `W` is `[out_dim, in_dim]`.

```
for o in 0..out_dim:
    out[o] = Σ W[o, k] * x[k]   for k in 0..in_dim
```

For quantized weights this dispatches to the appropriate kernel (scalar, AVX2, or GPU). For f32 weights it uses a direct dot product loop. See [SIMD](./simd.md) for the AVX2 path.

---

## RMSNorm

Used in place of LayerNorm throughout the LLaMA family. Simpler (no mean subtraction) and slightly faster.

**Formula:**

```
y[i] = x[i] / rms(x) * weight[i]

where rms(x) = sqrt( mean(x²) + ε )
            = sqrt( (1/n) Σ x[i]² + ε )
```

Implementation (`src/tensor/ops.rs`):
```rust
let ss: f32 = x.data().iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-5;
let inv_rms = 1.0 / ss.sqrt();
for i in 0..n {
    out[i] = weight[i] * x[i] * inv_rms;
}
```

Applied before both the attention block and the FFN block in every transformer layer.

---

## Rotary Positional Embeddings (RoPE)

RoPE encodes token position by rotating Q and K vectors in 2D planes. Unlike additive positional embeddings, rotation is length-preserving and composes cleanly with dot-product attention.

**For each pair of dimensions `(2i, 2i+1)` at position `pos`:**

```
θᵢ = pos / (rope_freq_base ^ (2i / head_dim))

[q[2i],   q[2i+1] ] → [ q[2i]*cos(θᵢ) - q[2i+1]*sin(θᵢ),
                         q[2i]*sin(θᵢ) + q[2i+1]*cos(θᵢ) ]
```

Same transformation is applied to K vectors. Q and K are rotated with the same angle for the same position, so their dot product encodes relative position.

`rope_freq_base` (typically 10000 for LLaMA, 500000 for LLaMA-3) is read from GGUF metadata.

---

## SiLU Activation

Used in the SwiGLU feed-forward network: `silu(x) = x * sigmoid(x) = x / (1 + e^(-x))`.

Properties:
- Smooth everywhere (differentiable)
- Approximately linear for `x >> 0`
- Approaches zero for `x << 0`

---

## Softmax

Numerically stable implementation using the max-subtraction trick:

```rust
let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
let sum: f32 = exps.iter().sum();
exps.iter().map(|&v| v / sum).collect()
```

Subtracting `max_val` before `exp()` prevents overflow without changing the result (the normalization constant cancels out).

---

## Flash Attention

Glint implements a CPU variant of Flash Attention for single-query decode in `src/tensor/flash.rs`.

**The problem with standard attention:**

For a sequence of length N, standard attention materializes an `[N, N]` score matrix. At N=4096 with f32, this is 64 MB per layer per head — and grows quadratically.

**Flash attention solution:**

Compute `softmax(QKᵀ/√d) × V` in a single pass without storing the full score matrix. For single-query decode (the common case during generation), Q is a single vector `[1, head_dim]` and the attention becomes:

```
scores[pos] = Q · K[pos] / sqrt(head_dim)   for pos in 0..seq_len
output = Σ softmax(scores)[pos] * V[pos]
```

Glint's implementation uses the **online softmax** algorithm: accumulate a running maximum and a running sum, then rescale as new tokens are processed. Memory usage is O(head_dim), not O(seq_len).

This is critical for long-context models (8K+ token context windows).

---

## Embedding Lookup

The simplest op: index into the embedding table.

```rust
let embed = &weights.token_embd[token_id * embed_dim .. (token_id + 1) * embed_dim];
```

The embedding matrix is `[vocab_size, embed_dim]` stored as f32 (not quantized) to avoid dequantization overhead on every input token.
