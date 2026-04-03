# GGUF Format

GGUF (GGML Universal Format) is a binary file format for storing quantized transformer model weights along with all associated metadata. Glint parses GGUF natively without any external parser library.

Source: `src/model/gguf.rs`

---

## File Layout

```
┌─────────────────────┐
│  Magic: "GGUF"      │  4 bytes: 0x47 0x47 0x55 0x46
├─────────────────────┤
│  Version            │  u32 (versions 2 and 3 supported)
├─────────────────────┤
│  Tensor count       │  u64
├─────────────────────┤
│  Metadata KV count  │  u64
├─────────────────────┤
│  Metadata KV pairs  │  variable length
│    key: string      │
│    type: u32        │
│    value: varies    │
├─────────────────────┤
│  Tensor descriptors │  variable length
│    name: string     │
│    dimensions: [u64]│
│    ggml_type: u32   │
│    offset: u64      │
├─────────────────────┤
│  Alignment padding  │  pad to 32-byte boundary
├─────────────────────┤
│  Tensor data        │  raw bytes for all tensors
└─────────────────────┘
```

Everything is little-endian. Strings are encoded as `u64 length` followed by UTF-8 bytes (no null terminator).

---

## Memory-Mapped Loading

Glint uses `memmap2` to memory-map the file rather than reading it into heap memory:

```rust
let mmap = unsafe { Mmap::map(&file)? };
```

Benefits:
- **Zero-copy access.** Tensor data is read directly from the OS page cache without a heap allocation. For a 4 GB Q4_K model, this avoids allocating and copying 4 GB.
- **Lazy page faulting.** The OS loads pages on demand; cold regions are never touched.
- **Shared across processes.** Multiple processes can mmap the same file and share physical pages.

The `QuantizedTensor` type stores a reference (offset + length) into the mmap rather than copying bytes.

---

## Metadata Key-Value Pairs

Metadata encodes model hyperparameters, tokenizer vocabulary, and training information. Key names follow the pattern `{namespace}.{key}`:

| Namespace | Example keys |
|-----------|-------------|
| `general` | `general.architecture`, `general.name` |
| `{arch}` (e.g. `llama`) | `llama.context_length`, `llama.embedding_length`, `llama.block_count`, `llama.attention.head_count`, `llama.attention.head_count_kv`, `llama.rope.freq_base` |
| `tokenizer` | `tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id` |

The complete list of hyperparameters extracted into `ModelConfig`:

```rust
pub struct ModelConfig {
    pub architecture: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub vocab_size: u32,
    pub rope_freq_base: f32,
    pub sliding_window: Option<u32>,
    pub chat_template: Option<String>,
}
```

---

## Tensor Descriptors

Each tensor descriptor contains:
- **name** — string identifier (e.g. `blk.0.attn_q.weight`)
- **dimensions** — shape in GGUF column-major order (Glint reverses this to row-major)
- **ggml_type** — quantization format enum
- **offset** — byte offset of tensor data from the start of the data section

### Shape Convention

GGUF stores shapes in column-major order (last dimension first). Glint reverses this to row-major on load:

```rust
// GGUF: [out_dim, in_dim] (column-major)
// After reversal: [in_dim, out_dim] (row-major, Glint convention)
let shape: Vec<usize> = info.dimensions.iter().rev().map(|&d| d as usize).collect();
```

This reversal happens in `src/transformer/weights.rs` and is a critical invariant — getting it wrong silently produces garbage outputs.

### Tensor Naming Convention

Weight names follow the LLaMA naming scheme used by GGUF:

| Name | Description |
|------|-------------|
| `token_embd.weight` | Token embedding matrix `[vocab_size, embed_dim]` |
| `output_norm.weight` | Final RMSNorm weights `[embed_dim]` |
| `output.weight` | LM head projection `[vocab_size, embed_dim]` |
| `blk.{i}.attn_norm.weight` | Pre-attention RMSNorm for layer i |
| `blk.{i}.attn_q.weight` | Query projection `[n_heads * head_dim, embed_dim]` |
| `blk.{i}.attn_k.weight` | Key projection `[n_kv_heads * head_dim, embed_dim]` |
| `blk.{i}.attn_v.weight` | Value projection `[n_kv_heads * head_dim, embed_dim]` |
| `blk.{i}.attn_output.weight` | Attention output projection `[embed_dim, embed_dim]` |
| `blk.{i}.ffn_norm.weight` | Pre-FFN RMSNorm for layer i |
| `blk.{i}.ffn_gate.weight` | FFN gate projection (SwiGLU) |
| `blk.{i}.ffn_up.weight` | FFN up projection |
| `blk.{i}.ffn_down.weight` | FFN down projection |

---

## Supported Quantization Types

| GGML type | Glint name | Bits/weight | Block size |
|-----------|-----------|-------------|------------|
| 0 | F32 | 32 | — |
| 1 | F16 | 16 | — |
| 8 | Q8_0 | 8 | 32 elements |
| 2 | Q4_0 | 4 | 32 elements |
| 12 | Q4_K | ~4.5 | 256 elements (super-block) |
| 13 | Q5_K | ~5.5 | 256 elements |
| 14 | Q6_K | 6.5 | 256 elements |
| 10 | Q2_K | 2.6 | 256 elements |
| 11 | Q3_K | 3.4 | 256 elements |
| 29 | IQ4_NL | 4 | 32 elements (non-linear) |

See [Quantization](./quantization.md) for the byte layout of each format.
