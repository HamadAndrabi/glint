# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project

**Glint** is a CPU-first LLM inference engine built from scratch in Rust. It loads GGUF-format models, runs transformer forward passes with quantized weights, and exposes an OpenAI-compatible HTTP API.

The default surface is native CPU inference, but the repo also contains optional feature-gated surfaces for:

- `python` via PyO3
- `wasm` via `wasm-bindgen`
- `vulkan` via `wgpu`

Binary name: `glint` (package name in `Cargo.toml` is also `glint`)

## Commands

```bash
# Build the default native surface
cargo build --release

# Core validation
cargo test --lib
cargo test --lib <test_name>
cargo clippy
cargo fmt
cargo fmt --check

# Hot-path performance
cargo bench --bench matvec

# Feature-surface compile checks
cargo build --features python
cargo build --features wasm
cargo build --features vulkan
```

## Architecture

The codebase is layered roughly like this:

```text
GGUF file / bytes
    |
    +--> metadata + tokenizer + tensor descriptors
    |
TransformerWeights / ModelConfig
    |
Transformer forward pass + KV cache
    |
Tensor ops + quantized matvec
    |
Scalar CPU fallback or AVX2/FMA SIMD or optional GPU backend
    |
CLI / HTTP server / Python bindings / WASM bindings
```

Key types:

- `Tensor`: heap-allocated `f32` tensor for activations and small weights
- `QuantizedTensor`: compressed weight storage used directly by matvec kernels
- `TransformerWeights` / `LayerWeights`: loaded model weights
- `KvCache` / `KvCacheQ8`: f32 and Q8_0-quantized KV cache implementations
- `KvStore`: trait abstracting over cache formats (used by flash attention and forward pass)
- `GlintError`: project-wide error enum via `thiserror`

Quantization formats supported: `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, `IQ4_NL`

- `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, and `Q5_1` use 32-element blocks
- K-quants use 256-element super-blocks
- 4-bit and 5-bit simple quants pack nibbles split-plane like ggml (byte `j`
  holds element `j` in its low nibble and element `j+16` in its high nibble)

SIMD dispatch pattern:

```rust
#[cfg(all(target_arch = "x86_64", feature = "rayon"))]
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    return unsafe { simd::matvec_q8_0_avx2(...) };
}
matvec_q8_0_scalar(...)
```

Note: `simd.rs` is only compiled when both `x86_64` architecture AND `rayon` feature are active.

Keep a scalar fallback alongside every optimized path.

GGUF loading is zero-copy via `memmap2` on native builds. GGUF stores tensors in column-major order; internally Glint uses row-major tensors, and the shape reversal happens in `src/transformer/weights.rs`.

SafeTensors (HuggingFace) loading is the second, optional entry point — a directory holding `config.json` + `tokenizer.json` + `*.safetensors`, or a `.safetensors` file inside one. GGUF stays the default; the loader is picked by path in `Model::load` and in the CLI's `load_model`. HF stores linear weights row-major as `[out_features, in_features]`, which is already Glint's convention, so **no transpose is applied** — but Q/K projections *are* row-permuted, because HF's RoPE pairs dimension `j` with `j + head_dim/2` while Glint's `ops::rope` rotates adjacent pairs (see `permute_qk_rows`).

## Module Map

| Path | Responsibility |
|------|---------------|
| `src/error.rs` | `GlintError` enum |
| `src/model/gguf.rs` | GGUF parser and tensor metadata |
| `src/model/safetensors.rs` | SafeTensors parser and HuggingFace model-directory loader |
| `src/model/config.rs` | Hyperparameters from GGUF metadata or HF `config.json` |
| `src/model/tokenizer.rs` | BPE tokenizer (GGUF metadata or HF `tokenizer.json`) |
| `src/model/chat_template.rs` | Chat template detection and rendering |
| `src/model/lora.rs` | LoRA adapter loading and application (`ΔW = scale × B @ A`) |
| `src/model/pull.rs` | Hugging Face model search and download |
| `src/tensor/tensor.rs` | f32 tensor type |
| `src/tensor/ops.rs` | RMSNorm, RoPE, softmax, SiLU, matmul helpers |
| `src/tensor/flash.rs` | Flash attention for single-query decode (online softmax, O(N) memory) |
| `src/tensor/quantized.rs` | Quantized tensor storage, dispatch, scalar kernels |
| `src/tensor/dequantize.rs` | Reference dequantization for all formats |
| `src/tensor/simd.rs` | AVX2 + FMA kernels and unsafe SIMD helpers |
| `src/cache/` | KV-cache implementations |
| `src/cache/paged.rs` | PagedAttention-style paged KV cache (`PagePool`, `PagedKvCache`, CoW `fork_from`) |
| `src/cache/prefix.rs` | Prefix-cache registry reusing KV pages across requests sharing a prompt prefix |
| `src/transformer/weights.rs` | Weight loading from GGUF into Glint structs |
| `src/transformer/forward.rs` | Forward pass and generation loops |
| `src/transformer/speculative.rs` | Speculative decoding — draft generates k tokens, target verifies |
| `src/server/mod.rs` | Router setup and CORS |
| `src/server/engine.rs` | Background inference engine and request queue |
| `src/server/routes.rs` | `/health`, `/v1/models`, `/v1/metrics`, completions, chat, embeddings |
| `src/server/types.rs` | OpenAI-compatible request and response shapes |
| `src/backend/` | Optional GPU backend and pipeline setup |
| `src/python.rs` | Optional PyO3 bindings |
| `src/wasm.rs` | Optional browser bindings |

## Technical References

These guides are the best repo-specific references when making non-trivial changes:

- `docs/phase1_1_guide.md`: GGUF parsing, quantization layout, naming conventions
- `docs/phase1_2_guide.md`: tensor primitives and dequantization
- `docs/phase1_3_guide.md`: forward pass, attention, cache usage
- `docs/phase1_4_guide.md`: tokenizer details

## Review Hotspots

Prioritize these invariants during implementation and review:

1. Quantization layout consistency
   Keep GGUF parsing, dequantization, quantized kernels, SIMD code, and GPU shaders aligned on block sizes, byte layout, and row/column interpretation.

2. Scalar fallback and parity
   Treat scalar or dequantize-then-dot behavior as the correctness reference. Optimized paths should preserve behavior and keep a fallback.

3. Unsafe code discipline
   New or edited `unsafe` blocks should document their invariants with a `// SAFETY:` comment.

4. Performance-sensitive paths
   Avoid regressions in hot loops. Do not accidentally dequantize whole large tensors into `f32` or introduce avoidable allocations in kernel-adjacent code.

5. API compatibility
   Preserve the current server contract, especially SSE streaming behavior:
   - non-streaming remains the default unless `stream: true`
   - a stream that completed ends with a final chunk carrying `finish_reason`
     (`"stop"` for EOS, `"length"` for a token-budget or context cutoff),
     followed by `data: [DONE]`
   - a stream truncated by slow-client eviction ends with an SSE `error` event
     and deliberately **no** `[DONE]`, so a partial response cannot be mistaken
     for a complete one; non-streaming endpoints return 500 in that case
   - object names in `src/server/types.rs` stay aligned with route behavior

   The engine reports why a sequence ended via `Finish`/`FinishSignal`
   (`src/server/engine.rs`), written before the token sender is dropped. Routes
   must consult it rather than assuming a closed channel means success.

6. Feature-gated surfaces
   If a shared API changes, audit `python`, `wasm`, and `vulkan` paths as needed. Do not assume the default native CLI/server surface is the only consumer.

## Validation Guidance

Use the smallest relevant checks for the touched surface, then broaden if needed:

- Tensor, SIMD, cache, backend, or forward-pass changes:
  - run targeted tests first
  - run `cargo test --lib`
  - run `cargo clippy`
  - run `cargo bench --bench matvec` for hot-path changes

- Server or API-shape changes:
  - run `cargo test --lib`
  - run `cargo clippy`
  - inspect both streaming and non-streaming paths in `src/server/routes.rs`
  - keep `src/server/types.rs` synchronized with route behavior

- Feature-gated changes:
  - run the relevant feature build: `python`, `wasm`, and/or `vulkan`

## Testing Notes

Tests live alongside the source in `#[cfg(test)]` blocks. The heaviest quantization coverage is in `src/tensor/dequantize.rs` and `src/tensor/quantized.rs`, where reference values are hardcoded from offline calculations.

When adding or changing quantized paths:

- add equivalence tests between scalar and optimized implementations when possible
- prefer local, targeted tests near the edited module
- keep validation close to the exact quant format being changed

`tempfile` is used for tests that need real files.
