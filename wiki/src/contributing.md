# Contributing

---

## Development Commands

```bash
# Build (native, default features)
cargo build --release

# Run all unit tests
cargo test --lib

# Run a single test by name
cargo test --lib kv_cache_q8_roundtrip

# Linting
cargo clippy

# Format (check only)
cargo fmt --check

# Format (apply)
cargo fmt

# Hot-path benchmarks
cargo bench --bench matvec

# Feature-surface compile checks
cargo build --features python
cargo build --features cffi
cargo build --no-default-features --features wasm
cargo build --features vulkan
```

---

## Validation Guidance

Use the smallest relevant checks for the touched surface, then broaden if needed.

### Tensor, SIMD, cache, backend, or forward-pass changes

```bash
cargo test --lib <specific_test>   # targeted first
cargo test --lib                   # then full suite
cargo clippy
cargo bench --bench matvec         # for hot-path changes
```

### Server or API-shape changes

```bash
cargo test --lib
cargo clippy
# Manually inspect both streaming and non-streaming paths in src/server/routes.rs
# Ensure src/server/types.rs stays synchronized with route behavior
```

### Feature-gated changes

```bash
cargo test --lib --features cffi  # if C ABI surface affected
cargo build --features python    # if Python surface affected
cargo build --features cffi      # if C ABI surface affected
cargo build --no-default-features --features wasm  # if WASM surface affected
cargo build --features vulkan    # if GPU backend affected
```

---

## Code Review Hotspots

When writing or reviewing code, pay special attention to these six invariant categories:

### 1. Quantization Layout Consistency

GGUF parsing, dequantization, quantized kernels, SIMD code, and GPU shaders must all agree on:
- Block sizes (32 for simple quants, 256 for K-quants)
- Byte layout within each block (scale position, nibble packing order)
- Row/column interpretation (Glint is row-major; GGUF stores column-major — reversal in `weights.rs`)

If these diverge, the model will produce garbage without panicking.

### 2. Scalar Fallback and Parity

Every optimized path (SIMD, GPU) must produce results within floating-point tolerance of the scalar reference:
- The scalar implementation in `dequantize.rs` is the correctness oracle
- Add equivalence tests when adding new quant formats
- If you change a kernel, run the existing tests to verify parity

### 3. Unsafe Code Discipline

Every `unsafe` block must have a `// SAFETY:` comment:

```rust
// SAFETY: avx2 + fma detected at dispatch site; all pointer accesses
// are within the bounds of QuantizedTensor::data (block_count × 34 bytes for Q8_0).
unsafe fn matvec_q8_0_avx2(...) { ... }
```

Document: CPU feature requirements, pointer alignment guarantees, and buffer bounds reasoning.

### 4. Performance-Sensitive Paths

The forward pass hot loop runs millions of times per generation. Avoid:
- Heap allocations inside `forward_one()` (pre-allocate in outer scope)
- Dequantizing entire large tensors to f32 (operate on blocks)
- Unnecessary cloning of large slices
- Synchronous I/O or system calls during generation

Profile before and after any change that touches `forward.rs`, `quantized.rs`, or `simd.rs`.

### 5. API Compatibility

Preserve the server contract, especially SSE streaming:
- Non-streaming remains the default (`stream: false` if omitted)
- Final SSE chunk carries `finish_reason: "stop"`
- Streams terminate with `data: [DONE]`
- Object names in `types.rs` stay aligned with route behavior

Breaking API changes break every client that uses the server.

### 6. Feature-Gated Surfaces

If a shared type or function changes, audit all feature-gated surfaces:
- `src/python.rs` — if `TransformerWeights`, `Tokenizer`, or `Sampler` API changes
- `src/wasm.rs` — same
- `src/ffi/mod.rs` and `include/glint.h` — if public runtime/session APIs change
- `src/backend/` — if tensor types or forward pass signature changes

---

## Testing Philosophy

Tests live in `#[cfg(test)]` blocks alongside the source code. Key principles:

- **Local and targeted.** A test for `KvCacheQ8` lives in `cache/mod.rs`, not in a separate test file.
- **Reference values.** Quantization tests use hardcoded expected values computed offline (Python/NumPy). Don't trust floating-point equality; use tolerances like `< 0.12` for Q8_0 round-trips.
- **Edge cases.** Test boundary conditions: empty inputs, all-negative logits, zero seed, maximum sequence length.
- **Equivalence tests.** When adding SIMD kernels, add a test that compares scalar and SIMD output on the same input.

```bash
# Example: run just the quantization tests
cargo test --lib dequantize
cargo test --lib quantized
```

The heaviest quantization coverage is in `src/tensor/dequantize.rs` and `src/tensor/quantized.rs`.

---

## Correctness Anchors

Glint's numerics are validated against two *external* references, so the kernels
are checked against the wider ecosystem rather than only against themselves:

### ggml reference vectors

`scripts/gen_ggml_vectors.py` transcribes the `dequantize_row_*` algorithms from
llama.cpp's `ggml/src/ggml-quants.c` (plus the block byte layouts from
`ggml-common.h`) and emits hardcoded golden vectors for every supported format.
These land in the `ggml_reference_tests` module in `src/tensor/dequantize.rs`, so
each dequantizer is pinned to the reference output rather than to a Glint-internal
round-trip. Regenerate them when adding or changing a format:

```bash
python3 scripts/gen_ggml_vectors.py   # prints Rust source to paste into the test module
```

### Golden-output parity vs llama.cpp

`scripts/golden_parity.sh MODEL.gguf [N_TOKENS] [PROMPT]` greedy-decodes the same
prompt through both Glint and llama.cpp at temperature 0 and requires
byte-identical output. Because greedy decode is deterministic, any divergence
means the two stacks disagree somewhere — tokenizer, BOS handling, quantized
kernels, RoPE/GQA attention, or sampling.

```bash
cargo build --release
scripts/golden_parity.sh models/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

A CI workflow (`.github/workflows/parity.yml`) runs this weekly across
`Q8_0 / Q4_K_M / Q2_K / Q3_K_M` builds of SmolLM2-135M.

---

## Fuzzing and Miri

The two untrusted-byte surfaces — the GGUF parser and the KV-snapshot importer —
are fuzzed under `fuzz/`. The contract each target asserts is that parsing any
input returns `Ok`/`Err` and never panics, aborts, or attempts an unbounded
allocation:

```bash
cargo fuzz run gguf_parse
cargo fuzz run snapshot_import
```

CI smoke-runs both targets on every push (`fuzz-smoke` job) and runs the
unsafe-heavy `tensor::` and `cache::` modules under Miri to catch undefined
behaviour the tests cannot see on real hardware:

```bash
cargo miri test --no-default-features --lib -- tensor:: cache::
```

---

## Continuous Integration

`.github/workflows/rust.yml` gates every push and PR with:

- **Build + test** on an x86_64 runner (exercises the AVX2/FMA SIMD kernels) *and*
  an ARM64 runner (exercises the scalar fallback), each in both the default and
  `--features cffi` configurations.
- **Format + lint** — `cargo fmt --all --check` plus `cargo clippy --lib -D warnings`
  for the default and `cffi` feature sets (the library surface is warning-clean).
- **fuzz-smoke** — a short run of each fuzz target so they cannot rot.
- **Miri** — the `tensor::`/`cache::` UB check described above.
- **Feature checks** — `vulkan`, `python`, and `wasm` compile surfaces.

---

## Adding a New Quantization Format

When adding support for a new GGUF quant type:

1. Add the type to the `GgmlType` enum in `src/model/gguf.rs`
2. Implement `dequantize_block_*` in `src/tensor/dequantize.rs` — this is the scalar reference
3. Add dispatch in `src/tensor/quantized.rs`
4. Write tests with hardcoded reference values (computed in Python against `ggml` library)
5. Optionally, add an AVX2 kernel in `src/tensor/simd.rs` with an equivalence test
6. Update `src/backend/pipeline.rs` if a GPU shader is needed

---

## Adding a New Sampling Strategy

1. Implement the function in `src/sampling/sampler.rs` alongside existing pipeline stages
2. Add to `SamplerConfig` with a disabled default
3. Insert at the appropriate position in `Sampler::sample()`
4. Add unit tests (the existing tests for `apply_top_k`, `apply_min_p`, etc. are good templates)
5. Expose as a CLI flag in `src/main.rs` and as an API parameter in `src/server/types.rs`
