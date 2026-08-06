# Forward Pass

The transformer forward pass is the core computation: given a sequence of token IDs, produce a probability distribution over the vocabulary for the next token.

Source: `src/transformer/forward.rs`, `src/transformer/weights.rs`

---

## High-Level Flow

```
token_ids: [t₀, t₁, ..., tₙ]
    │
    ▼
token_embd[tᵢ]                         ← embedding lookup
    │
    ▼  × n_layers
┌─────────────────────────────────┐
│  attn_norm(x)                   │  ← RMSNorm
│  Q = W_q × x_norm               │  ← query projection
│  K = W_k × x_norm               │  ← key projection
│  V = W_v × x_norm               │  ← value projection
│  Q, K = rope(Q, K, pos)         │  ← rotary position encoding
│  write K, V to KV-cache         │
│  attn = flash_attn(Q, K, V)     │  ← scaled dot-product attention
│  x += W_o × attn                │  ← output projection + residual
│                                 │
│  ffn_norm(x)                    │  ← RMSNorm
│  gate = silu(W_gate × x_norm)   │  ← SwiGLU gate
│  up   = W_up × x_norm           │
│  x += W_down × (gate * up)      │  ← FFN down projection + residual
└─────────────────────────────────┘
    │
    ▼
output_norm(x)                         ← final RMSNorm
    │
    ▼
logits = W_lm_head × x_norm            ← vocabulary projection [vocab_size]
    │
    ▼
sampler.sample(logits) → next_token_id
```

---

## Two Modes: Prefill vs. Decode

### Prefill (`forward_prefill_all`)

Processes all prompt tokens at once. For each token at position `i`:
- Computes Q/K/V
- Writes K and V into the cache at position `i`
- Returns the logits for all positions

Prefill uses `rayon` to parallelize across layers when the `rayon` feature is active.

### Decode (`forward_one`)

Processes a single new token. Much cheaper than prefill:
- Computes Q/K/V for the current position only
- Reads all cached K/V from positions `[0, current_pos)`
- Returns logits for the current position

The loop that calls `forward_one` repeatedly is `generate_cached`:

```rust
pub fn generate_cached(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    // Prefill
    forward_prefill_all(weights, config, prompt_tokens, &mut cache, 0, gpu);
    let mut tokens = prompt_tokens.to_vec();

    // Decode loop
    for _ in 0..max_new_tokens {
        let pos = tokens.len() - 1;
        let logits = forward_one(weights, config, *tokens.last().unwrap(), pos, &mut cache, gpu);
        let next = sampler.sample(logits.data(), &tokens);
        if next == eos_token { break; }
        tokens.push(next);
    }
    tokens
}
```

---

## Batched Decode

`forward_batch` (and `forward_batch_lora`) runs one decode step for several
sequences in a single pass: per layer, each weight matrix is traversed once and
applied to every sequence's activation vector (`matvec_batch` in
`src/tensor/quantized.rs`), while attention stays per-sequence — each sequence
reads its own KV cache at its own position. Because decode is
memory-bandwidth-bound, streaming the weights once per step instead of once per
sequence is what makes concurrent serving scale; the HTTP engine uses this for
[continuous batching](./server-api.md#continuous-batching).

Per-sequence outputs are bit-identical to `forward_one` — the batched kernels
keep each dot product's accumulation order unchanged — so batching is
observationally invisible. A batch of one takes the single-sequence path.

---

## Multi-Head Attention

Attention is the most complex part of the forward pass.

### Grouped Query Attention (GQA)

Modern models (LLaMA-3, Mistral, etc.) use Grouped Query Attention: there are `n_heads` query heads but only `n_kv_heads` key/value heads. Each KV head is shared by `n_heads / n_kv_heads` query heads. This reduces KV cache memory without much quality loss.

```
n_heads = 32, n_kv_heads = 8  (LLaMA-3 8B)
→ each of the 8 KV heads is shared by 4 query heads
```

In code:

```rust
let kv_head = q_head / (n_heads / n_kv_heads);
cache.read_k_head(layer, pos, kv_head, head_dim, &mut k_buf);
```

### Attention Score Computation

For each query head at position `pos`, attention over all prior positions:

```
score[p] = Q · K[p] / sqrt(head_dim)    for p in 0..=pos
output   = Σ softmax(scores)[p] * V[p]
```

Glint uses the flash attention implementation from `src/tensor/flash.rs` which avoids materializing the full `[seq_len, head_dim]` score buffer.

---

## SwiGLU Feed-Forward Network

Every transformer layer has a feed-forward network after attention. LLaMA uses the SwiGLU variant:

```
ffn(x) = W_down × (silu(W_gate × x) ⊙ W_up × x)
```

Where ⊙ is element-wise multiplication. This is a "gated" linear unit: the gate controls how much of the `up` projection flows into `down`.

The intermediate dimension is typically `8/3 × embed_dim` (rounded to a multiple of 256).

---

## Residual Connections

Every sub-layer (attention and FFN) uses a residual connection:

```
x = x + attention_output(x)    # attention block
x = x + ffn_output(x)          # FFN block
```

Residuals are added to the input of each sub-layer, not the normalized input. This is the Pre-Norm architecture used by LLaMA (as opposed to Post-Norm in the original transformer).

---

## Weight Loading

Weights are loaded from GGUF in `src/transformer/weights.rs`. The key concern is name mapping: GGUF tensor names must be correctly mapped to struct fields.

```rust
pub struct LayerWeights {
    pub attn_norm: Tensor,       // blk.{i}.attn_norm.weight (f32)
    pub attn_q:    QuantizedTensor,  // blk.{i}.attn_q.weight
    pub attn_k:    QuantizedTensor,  // blk.{i}.attn_k.weight
    pub attn_v:    QuantizedTensor,  // blk.{i}.attn_v.weight
    pub attn_output: QuantizedTensor, // blk.{i}.attn_output.weight
    pub ffn_norm:  Tensor,
    pub ffn_gate:  QuantizedTensor,  // blk.{i}.ffn_gate.weight
    pub ffn_up:    QuantizedTensor,  // blk.{i}.ffn_up.weight
    pub ffn_down:  QuantizedTensor,  // blk.{i}.ffn_down.weight
    pub lora: Option<LoraLayerAdapters>,
}
```

Norm weights are always loaded as f32 (they're small and precision matters for normalization). Projection weights use the quantization type stored in the GGUF file.

---

## GPU Dispatch

The forward pass conditionally dispatches to the GPU backend:

```rust
fn matvec_dispatch(qt: &QuantizedTensor, x: &[f32], out: &mut [f32], gpu: &mut Option<&mut GpuBackend>) {
    if let Some(gpu) = gpu {
        if let Some(handle) = &qt.gpu_handle {
            return gpu.matvec(handle, x, out);
        }
    }
    // CPU fallback
    qt.matvec(x, out);
}
```

GPU handles are populated during `weights.upload_all_to_gpu()`.
