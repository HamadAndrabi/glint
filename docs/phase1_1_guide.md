# Phase 1.1 — GGUF Parser: Technical Guide

This document explains the concepts behind the GGUF parser implementation.

**Source files:**

- [gguf.rs](../src/model/gguf.rs) — parser, types, `GgufModel`
- [config.rs](../src/model/config.rs) — `ModelConfig` extracted from metadata

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

Parsing happens sequentially in [GgufModel::load](../src/model/gguf.rs#L432-L480): magic → version → metadata KVs → tensor infos → compute aligned data offset.

### Magic Number

The first 4 bytes are `0x47 0x47 0x55 0x46` (ASCII "GGUF"). Read as a little-endian `u32`, this is `0x46554747`. Defined as [`GGUF_MAGIC`](../src/model/gguf.rs#L424).

### Strings

GGUF strings are **length-prefixed**, NOT null-terminated: `[u64 length][UTF-8 bytes]`. See [Cursor::read_string](../src/model/gguf.rs#L365-L371).

### Metadata KV Store

Each entry: `[string key] [u32 value_type] [value]`. Arrays have an additional header: `[u32 element_type] [u64 count] [elements...]`.

Parsed by [Cursor::read_metadata_kv](../src/model/gguf.rs#L393-L398) which dispatches to [read_metadata_value](../src/model/gguf.rs#L378-L392) based on the type tag.

Keys follow a hierarchical convention: `general.architecture`, `llama.context_length`, `tokenizer.ggml.tokens`.

### Tensor Info

Each descriptor: `[string name] [u32 n_dims] [u64 dim0] ... [u32 ggml_type] [u64 offset]`. Parsed by [Cursor::read_tensor_info](../src/model/gguf.rs#L400-L414). The offset is **relative to the tensor data section**, not the file start.

Result stored in the [TensorInfo](../src/model/gguf.rs#L262-L269) struct.

### Alignment

Tensor data is padded to `general.alignment` boundaries (default 32 bytes). Computed by the [align_offset](../src/model/gguf.rs#L511-L513) function: `offset + (alignment - offset % alignment) % alignment`.

---

## Quantization

Neural network weights are originally f32 (4 bytes each). A 7B parameter model in f32 = **28 GB**. Quantization reduces precision to save space.

| Format | Per 32 Elements | Compression | How It Works                                  |
| ------ | --------------- | ----------- | --------------------------------------------- |
| F32    | 128 bytes       | 1x          | Full precision, 4 bytes each                  |
| F16    | 64 bytes        | 2x          | Half precision, 2 bytes each                  |
| Q8_0   | 34 bytes        | 3.8x        | 32 int8 values + 1 f16 scale factor           |
| Q4_0   | 18 bytes        | 7.1x        | 32 × 4-bit ints (packed 2/byte) + 1 f16 scale |

Types, block sizes, and byte sizes are defined in the [GgmlType](../src/model/gguf.rs#L55-L87) enum and its [block_size](../src/model/gguf.rs#L120-L133) / [type_size](../src/model/gguf.rs#L136-L169) methods.

### Block Quantization

Rather than using a single scale for the entire tensor (destroying dynamic range), values are quantized in **blocks of 32** (or 256 for K-quants). Each block has its own scale factor. For Q8_0: 32 int8 values (32 bytes) + one f16 scale (2 bytes) = **34 bytes per block**.

---

## Memory-Mapped I/O

Instead of `fs::read()`, we use **mmap** via the `memmap2` crate. This tells the OS to map the file into virtual address space, loading pages on demand.

**Benefits:** near-instant startup, only accessed pages loaded into RAM, OS cache sharing, zero-copy access.

The mmap is created in [GgufModel::load](../src/model/gguf.rs#L435-L436) and tensor data is accessed via [GgufModel::tensor_data](../src/model/gguf.rs#L490-L505) which returns a slice into the mmap'd file.

**Why `unsafe`?** The OS could theoretically modify the file while we read it, but model files don't change during inference.

---

## Model Configuration

Transformer models have fixed hyperparameters baked in at training time. [ModelConfig::from_metadata](../src/model/config.rs#L13-L50) extracts these from metadata under the architecture prefix:

| Key                              | Meaning                               |
| -------------------------------- | ------------------------------------- |
| `{arch}.context_length`          | Maximum sequence length               |
| `{arch}.embedding_length`        | Hidden state dimensionality (d_model) |
| `{arch}.block_count`             | Number of transformer layers          |
| `{arch}.attention.head_count`    | Query attention heads                 |
| `{arch}.attention.head_count_kv` | Key/Value attention heads (for GQA)   |

### Grouped Query Attention (GQA)

Modern models share Key/Value heads across groups of Query heads to reduce KV-cache memory:

- **MHA:** `head_count_kv == head_count` — original transformer
- **MQA:** `head_count_kv == 1` — all heads share one KV pair
- **GQA:** `1 < head_count_kv < head_count` — groups share KV

`head_dim = embedding_length / head_count` — computed by [ModelConfig::head_dim](../src/model/config.rs#L53-L55).

---

## Tensor Naming Convention

GGUF tensor names follow a pattern used in Phase 1.3 to wire up the forward pass:

| Name                         | Purpose                         |
| ---------------------------- | ------------------------------- |
| `token_embd.weight`          | Token embedding matrix          |
| `blk.{N}.attn_q.weight`      | Query projection, layer N       |
| `blk.{N}.attn_k.weight`      | Key projection, layer N         |
| `blk.{N}.attn_v.weight`      | Value projection, layer N       |
| `blk.{N}.attn_output.weight` | Attention output projection     |
| `blk.{N}.ffn_gate.weight`    | FFN gate projection (SwiGLU)    |
| `blk.{N}.ffn_up.weight`      | FFN up projection               |
| `blk.{N}.ffn_down.weight`    | FFN down projection             |
| `blk.{N}.attn_norm.weight`   | Pre-attention RMSNorm           |
| `blk.{N}.ffn_norm.weight`    | Pre-FFN RMSNorm                 |
| `output_norm.weight`         | Final layer norm                |
| `output.weight`              | LM head (hidden → vocab logits) |
