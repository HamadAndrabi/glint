# Glint

**Glint** is a CPU-first LLM inference engine built from scratch in Rust. It loads GGUF-format models, runs full transformer forward passes with quantized weights, and exposes an OpenAI-compatible HTTP API.

```
glint serve -f llama-3-8b-q4_k.gguf -p 8080
```

Point any OpenAI-compatible client at `http://localhost:8080` and it just works.

---

## Feature Matrix

| Feature | Status |
|---------|--------|
| GGUF model loading (all versions) | ✅ |
| Quantized inference: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL | ✅ |
| KV-cache (f32 and Q8_0-compressed) | ✅ |
| Flash attention (O(N) memory, single-query decode) | ✅ |
| OpenAI-compatible HTTP API | ✅ |
| Structured output (`json_object`) | ✅ |
| SSE streaming responses | ✅ |
| Multi-turn chat with context compression | ✅ |
| Speculative decoding (draft + target model) | ✅ |
| LoRA adapter loading and application | ✅ |
| Session API + KV snapshots | ✅ |
| AVX2 + FMA SIMD kernels | ✅ |
| Vulkan GPU backend (via wgpu) | ✅ |
| Python bindings (PyO3 / maturin) | ✅ |
| Browser WASM bindings (wasm-bindgen) | ✅ |
| C FFI (`include/glint.h`) | ✅ |
| Hugging Face Hub model pull | ✅ |
| Embeddings endpoint | ✅ |
| `glint bench` benchmarking CLI | ✅ |
| 156 passing tests (`cargo test --lib --features cffi`) | ✅ |

---

## Design Philosophy

Glint makes three bets:

1. **CPU-first, but not CPU-only.** Modern CPUs with AVX2/FMA run 4–8B quantized models at usable speeds without a discrete GPU. The CPU path is always the correctness reference; SIMD and GPU paths must produce identical results.

2. **Zero external ML framework.** No PyTorch, Candle, or ONNX Runtime. Every operation — matmul, attention, normalization — is implemented directly in Rust. This makes the code auditable and the binary portable.

3. **Real serving semantics.** The HTTP server is not a demo wrapper. It runs a background inference engine with a proper request queue, interleaves decode work across active requests, and exposes the OpenAI wire format production systems expect.

---

## Architecture at a Glance

```
GGUF file (mmap)
    │
    ├─ metadata + tokenizer vocab
    └─ tensor descriptors → TransformerWeights
                                │
                      Transformer forward pass
                         + KV cache
                                │
                  Quantized tensor ops + matvec dispatch
                                │
              ┌─────────────────┼──────────────────┐
         Scalar CPU          AVX2/FMA           Vulkan GPU
              │
    ┌─────────┴──────────────────┐
  CLI / HTTP server / Python / WASM
```

See [Architecture](./architecture.md) for a deeper walkthrough.

---

## What's in This Wiki

| Section | Contents |
|---------|----------|
| [Getting Started](./getting-started.md) | Build, install, first run |
| [Architecture](./architecture.md) | Layer map, module responsibilities |
| [GGUF Format](./gguf-format.md) | File layout, metadata, tensor encoding |
| [Quantization](./quantization.md) | All 8 formats, block layouts, compression |
| [Tensors & Ops](./tensors.md) | f32 Tensor type, matmul, RMSNorm, RoPE |
| [Forward Pass](./forward-pass.md) | End-to-end inference walkthrough |
| [Tokenization](./tokenization.md) | BPE, GPT-2 mapping, special tokens |
| [KV Cache](./kv-cache.md) | f32 and Q8_0 cache, KvStore trait |
| [Sampling](./sampling.md) | Temperature, top-k/p, min-p, repetition penalty |
| [Session API & Snapshots](./session-api.md) | `Model`/`Session`, snapshot export/import, deterministic resume |
| [SIMD](./simd.md) | AVX2/FMA kernels, dispatch, unsafe discipline |
| [Speculative Decoding](./speculative-decoding.md) | Draft/target protocol, speedup analysis |
| [LoRA Adapters](./lora.md) | Adapter loading, ΔW = scale × B @ A |
| [CLI Reference](./cli.md) | All subcommands with examples |
| [HTTP Server API](./server-api.md) | Endpoints, request/response shapes, SSE |
| [GPU Backend](./gpu-backend.md) | Vulkan via wgpu, WGSL shaders |
| [C FFI](./c-ffi.md) | Opaque handles, generation, snapshots, error handling |
| [Python Bindings](./python-bindings.md) | PyO3 class, maturin build |
| [Browser WASM](./wasm.md) | wasm-bindgen API, Web Worker demo |
| [Benchmarks](./benchmarks.md) | Matvec throughput, profiling tips |
| [Contributing](./contributing.md) | Development workflow, invariants, testing |
| [Roadmap](./roadmap.md) | What's built, what's next |
