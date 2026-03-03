# Phase 1.1 — GGUF Parser: Technical Guide

This document explains the concepts behind the GGUF parser implementation. Read this alongside the code in `src/model/`.

---

## GGUF File Format

GGUF (GPT-Generated Unified Format) is a binary format from the llama.cpp project for storing LLM weights. It replaced older formats (GGML, GGMF, GGJT) with a more extensible design.

### File Layout

```
┌──────────────────────────────┐
│ Header                       │  magic "GGUF", version, tensor_count, kv_count
├──────────────────────────────┤
│ Metadata KV pairs            │  model config: architecture, dims, vocab, etc.
├──────────────────────────────┤
│ Tensor Info array            │  "directory" — name, shape, type, offset per tensor
├──────────────────────────────┤
│ Padding to alignment         │  zeroed bytes to reach the next 32-byte boundary
├──────────────────────────────┤
│ Tensor Data                  │  actual weight bytes (may be quantized)
└──────────────────────────────┘
```

### Magic Number

The first 4 bytes are `0x47 0x47 0x55 0x46` (ASCII "GGUF"). Read as a little-endian `u32`, this is `0x46554747`. If this doesn't match, the file isn't GGUF.

### Strings

GGUF strings are **length-prefixed**, NOT null-terminated:

```
[u64 length] [length bytes of UTF-8]
```

This is different from C strings and means we always know exactly how many bytes to read.

### Metadata KV Store

Each entry has the layout:

```
[string key] [u32 value_type] [value]
```

Arrays have an additional header: `[u32 element_type] [u64 count] [elements...]`

Keys follow a hierarchical convention like `general.architecture`, `llama.context_length`, `tokenizer.ggml.tokens`.

### Tensor Info

Each tensor descriptor:

```
[string name] [u32 n_dims] [u64 dim0] [u64 dim1] ... [u32 ggml_type] [u64 offset]
```

The offset is **relative to the tensor data section**, not the file start. We compute the absolute position as `tensor_data_offset + offset`.

### Alignment

Tensor data is padded to `general.alignment` boundaries (default 32 bytes). This ensures SIMD-friendly memory addresses. The `align_offset` function rounds up: `offset + (alignment - offset % alignment) % alignment`.

---

## Quantization

Neural network weights are originally f32 (4 bytes each). A 7B parameter model in f32 = **28 GB**. Quantization reduces precision to save space:

| Format | Per 32 Elements | Compression | How It Works                                  |
| ------ | --------------- | ----------- | --------------------------------------------- |
| F32    | 128 bytes       | 1x          | Full precision, 4 bytes each                  |
| F16    | 64 bytes        | 2x          | Half precision, 2 bytes each                  |
| Q8_0   | 34 bytes        | 3.8x        | 32 int8 values + 1 f16 scale factor           |
| Q4_0   | 18 bytes        | 7.1x        | 32 × 4-bit ints (packed 2/byte) + 1 f16 scale |

### Block Quantization

Rather than using a single scale for the entire tensor (which would destroy dynamic range), values are quantized in **blocks of 32** (or 256 for K-quants). Each block has its own scale factor, preserving local dynamic range while still achieving good compression.

For Q8_0: each block stores 32 int8 values (32 bytes) + one f16 scale factor (2 bytes) = **34 bytes per block**.

To dequantize: `float_value = int8_value * scale`

### Data Size Calculation

```
n_blocks = ceil(n_elements / block_size)
data_bytes = n_blocks * type_size
```

---

## Memory-Mapped I/O

Instead of `fs::read()` (which copies the entire file into heap memory), we use **mmap**. This tells the OS: "map this file into my virtual address space." The OS loads pages on demand as we access them.

**Benefits:**

- **Fast startup** — mmap is nearly instant regardless of file size
- **Memory efficient** — only accessed pages are loaded into RAM
- **OS cache sharing** — multiple processes can share the same pages
- **No extra copies** — we read directly from the OS page cache

**Why `unsafe`?** — The OS could theoretically modify the file while we read it, violating Rust's aliasing guarantees. In practice, model files don't change during inference.

---

## Model Configuration

Transformer models have fixed hyperparameters baked in at training time. In GGUF, these live in metadata under the architecture prefix:

| Key                             | Meaning                               |
| ------------------------------- | ------------------------------------- |
| `llama.context_length`          | Maximum sequence length               |
| `llama.embedding_length`        | Hidden state dimensionality (d_model) |
| `llama.block_count`             | Number of transformer layers          |
| `llama.attention.head_count`    | Query attention heads                 |
| `llama.attention.head_count_kv` | Key/Value attention heads (for GQA)   |

### Grouped Query Attention (GQA)

Modern models share Key/Value heads across groups of Query heads to reduce KV-cache memory:

- **Multi-Head Attention (MHA):** `head_count_kv == head_count` — original transformer
- **Multi-Query Attention (MQA):** `head_count_kv == 1` — all heads share one KV pair
- **Grouped Query Attention (GQA):** `1 < head_count_kv < head_count` — groups share KV

`head_dim = embedding_length / head_count` — each head processes a slice of the hidden state.

---

## Cursor-Based Parsing

We maintain a `pos` index into the mmap'd byte slice and read values sequentially. Each `read_*` method:

1. Checks there are enough remaining bytes
2. Reads the value in little-endian byte order (GGUF's default)
3. Advances `pos` by the number of bytes consumed

This is the standard approach for parsing binary formats — simple, efficient, and easy to debug.

---

## Tensor Naming Convention

GGUF tensor names follow a pattern:

- `token_embd.weight` — token embedding matrix
- `blk.{N}.attn_q.weight` — query projection for layer N
- `blk.{N}.attn_k.weight` — key projection for layer N
- `blk.{N}.attn_v.weight` — value projection for layer N
- `blk.{N}.attn_output.weight` — attention output projection
- `blk.{N}.ffn_gate.weight` — FFN gate projection (for SwiGLU)
- `blk.{N}.ffn_up.weight` — FFN up projection
- `blk.{N}.ffn_down.weight` — FFN down projection
- `blk.{N}.attn_norm.weight` — pre-attention RMSNorm
- `blk.{N}.ffn_norm.weight` — pre-FFN RMSNorm
- `output_norm.weight` — final layer norm
- `output.weight` — language model head (hidden → vocab logits)

Understanding these names is essential for Phase 1.3 when we wire up the forward pass.
