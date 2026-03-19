# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Glint** — a CPU-based LLM inference engine built from scratch in Rust. It loads GGUF-format models, runs transformer forward passes with quantized weights, and exposes an OpenAI-compatible HTTP API.

Binary name: `glint` (package name in Cargo.toml is `glint`)

## Commands

```bash
# Build
cargo build --release

# Run all library tests
cargo test --lib

# Run a single test by name
cargo test --lib <test_name>

# Benchmarks (matvec throughput across quantization formats)
cargo bench --bench matvec

# Lint
cargo clippy

# Format
cargo fmt
```

## Architecture

The codebase is layered:

```
CLI (main.rs)  →  HTTP server (axum/tokio)
                       ↓
              Inference: forward.rs
                       ↓
         Tensor ops (ops.rs) + quantized matvec
                       ↓
    SIMD kernels (AVX2+FMA) or scalar fallback
                       ↓
        GGUF model loaded via memmap2
```

**Key types:**
- `Tensor` — f32 heap-allocated array; used for activations and small weights (norm scales)
- `QuantizedTensor` — raw quantized bytes kept compressed; dequantized block-by-block during `matvec()`
- `TransformerWeights` / `LayerWeights` — own the loaded model weights; large projections are `QuantizedTensor`, norms are `Tensor`
- `GlintError` — project-wide error type via `thiserror`; no panics in hot paths

**Quantization formats supported:** Q8_0, Q4_0, Q4_K, Q5_K, Q6_K
Block sizes: Q8_0/Q4_0 use 32-element blocks; K-quants use 256-element super-blocks.

**SIMD dispatch pattern** (in `quantized.rs` and `simd.rs`):
```rust
#[cfg(target_arch = "x86_64")]
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    return unsafe { simd::matvec_q8_0_avx2(...) };
}
matvec_q8_0_scalar(...)
```
Always keep a scalar fallback alongside every SIMD kernel.

**Parallelism:** `rayon` parallelizes matmul row-wise across all quantization kernels.

**GGUF loading:** zero-copy via `memmap2`. GGUF stores tensors in column-major order; internally everything is row-major — the shape reversal happens in `weights.rs`.

**Weight naming convention** follows LLaMA GGUF: `blk.{i}.attn_q.weight`, `blk.{i}.ffn_gate.weight`, etc.

## Module Map

| Path | Responsibility |
|------|---------------|
| `src/error.rs` | `GlintError` enum |
| `src/model/gguf.rs` | GGUF binary parser |
| `src/model/config.rs` | Hyperparameters from GGUF metadata |
| `src/model/tokenizer.rs` | BPE tokenizer (encode/decode) |
| `src/model/chat_template.rs` | Chat template rendering (ChatML, Llama 3, Mistral, …) |
| `src/tensor/tensor.rs` | f32 `Tensor` struct |
| `src/tensor/ops.rs` | RMSNorm, RoPE, matmul, softmax, SiLU, etc. |
| `src/tensor/quantized.rs` | `QuantizedTensor`, dispatch logic, scalar kernels |
| `src/tensor/dequantize.rs` | Block dequantization for all formats |
| `src/tensor/simd.rs` | AVX2+FMA kernels (unsafe) |
| `src/transformer/weights.rs` | Weight loading from GGUF into structs |
| `src/transformer/forward.rs` | Forward pass, KV-cache integration, generation loop |
| `src/cache/` | Pre-allocated KV-cache (key/value tensors per layer) |
| `src/sampling/sampler.rs` | Temperature, top-k, top-p, min-p, repetition penalty |
| `src/server/mod.rs` | axum router setup, CORS |
| `src/server/routes.rs` | `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings` handlers |
| `src/server/types.rs` | OpenAI-compatible request/response structs |
| `src/server/state.rs` | `AppState` (Arc-wrapped model + config) |

## Testing Notes

Tests live alongside the source in `#[cfg(test)]` blocks. The `dequantize.rs` and `quantized.rs` files have the heaviest coverage — reference values were computed offline with NumPy and hardcoded. When adding new quantization paths, add equivalence tests comparing SIMD output to scalar output.

`tempfile` crate is used in tests that need real files.
