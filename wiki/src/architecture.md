# System Overview

## Layered Architecture

Glint is organized as a stack of layers with clean separation between loading, computation, and serving.

```
┌─────────────────────────────────────────────────────┐
│                   HTTP API Layer                     │
│        (axum + tokio, SSE streaming)                 │
│        src/server/: mod.rs, routes.rs, engine.rs     │
├─────────────────────────────────────────────────────┤
│              Scheduling & Batching                   │
│         (request queue, continuous batching)         │
│         src/server/engine.rs                         │
├─────────────────────────────────────────────────────┤
│                Inference Runtime                     │
│    (transformer forward pass, KV-cache, sampling)    │
│    src/transformer/, src/cache/, src/sampling/       │
├─────────────────────────────────────────────────────┤
│                 Tensor Operations                    │
│   (matmul, softmax, RMSNorm, RoPE, SiLU)            │
│   src/tensor/                                        │
├─────────────────────────────────────────────────────┤
│               Compute Backends                       │
│   CPU scalar │ AVX2/FMA SIMD │ Vulkan GPU            │
│   src/tensor/quantized.rs │ simd.rs │ src/backend/   │
├─────────────────────────────────────────────────────┤
│                  Model Loading                       │
│    (GGUF parser, mmap, quantization formats)         │
│    src/model/                                        │
└─────────────────────────────────────────────────────┘
```

## Data Flow

A single inference request moves through the stack as follows:

```
.gguf file on disk
     │ memmap2 (zero-copy)
     ▼
GgufModel { metadata, tensor_infos, mmap }
     │
     ├─▶ ModelConfig::from_metadata()   (hyperparameters)
     ├─▶ Tokenizer::from_gguf()         (BPE vocab + merges)
     └─▶ TransformerWeights::load()     (QuantizedTensor refs)
                  │
         [prompt text]
                  │ tokenizer.encode()
                  ▼
         [token_ids: Vec<u32>]
                  │
         forward_prefill_all()   ← fills KV-cache for prompt
                  │
         loop {
           forward_one()         ← single-token decode step
           sampler.sample()      ← pick next token
           if eos → break
         }
                  │
         [output token_ids]
                  │ tokenizer.decode()
                  ▼
         "generated text"
```

## Module Map

| Path | Responsibility |
|------|----------------|
| `src/error.rs` | `GlintError` enum (via `thiserror`) |
| `src/model/gguf.rs` | GGUF parser: header, metadata KV pairs, tensor descriptors, mmap |
| `src/model/config.rs` | `ModelConfig`: context length, embedding dim, layer count, GQA ratio |
| `src/model/tokenizer.rs` | BPE tokenizer: encode, decode, special tokens |
| `src/model/chat_template.rs` | Chat template detection and message rendering |
| `src/model/lora.rs` | LoRA adapter loading; `ΔW = scale × B @ A` |
| `src/model/pull.rs` | Hugging Face Hub search and download |
| `src/tensor/tensor.rs` | `Tensor`: heap-allocated f32 tensor for activations |
| `src/tensor/ops.rs` | RMSNorm, RoPE, softmax, SiLU, matmul helpers |
| `src/tensor/flash.rs` | Flash attention for single-query decode (online softmax, O(N) memory) |
| `src/tensor/quantized.rs` | `QuantizedTensor`: storage + dispatch to scalar or SIMD kernels |
| `src/tensor/dequantize.rs` | Reference dequantization for all 8 formats |
| `src/tensor/simd.rs` | AVX2 + FMA matvec kernels (x86_64 + rayon only) |
| `src/cache/mod.rs` | `KvCache` (f32), `KvCacheQ8` (Q8_0), `KvStore` trait |
| `src/transformer/weights.rs` | Weight loading: GGUF tensors → `TransformerWeights` / `LayerWeights` |
| `src/transformer/forward.rs` | Forward pass, prefill, generation loops |
| `src/transformer/speculative.rs` | Speculative decoding: draft generates k tokens, target verifies |
| `src/sampling/sampler.rs` | `Sampler`: temperature, top-k, top-p, min-p, repetition penalty, PRNG |
| `src/server/mod.rs` | Router setup and CORS |
| `src/server/engine.rs` | Background inference engine and request queue |
| `src/server/routes.rs` | `/health`, `/v1/models`, `/v1/metrics`, completions, chat, embeddings |
| `src/server/types.rs` | OpenAI-compatible request/response shapes |
| `src/backend/gpu.rs` | `GpuBackend`: wgpu device, pipelines, buffer management |
| `src/backend/pipeline.rs` | WGSL compute pipeline setup and dispatch |
| `src/python.rs` | Optional PyO3 bindings |
| `src/wasm.rs` | Optional wasm-bindgen browser bindings |
| `src/main.rs` | CLI entry point (inspect, run, chat, serve, generate, pull) |

## Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `Tensor` | `tensor/tensor.rs` | Heap-allocated f32 tensor for activations and small weights |
| `QuantizedTensor` | `tensor/quantized.rs` | Compressed weight storage used directly by matvec kernels |
| `TransformerWeights` | `transformer/weights.rs` | All model weights loaded from GGUF |
| `LayerWeights` | `transformer/weights.rs` | Per-layer attention and FFN weights |
| `KvCache` | `cache/mod.rs` | f32 key-value cache |
| `KvCacheQ8` | `cache/mod.rs` | Q8_0-compressed KV cache (~3.8× smaller than f32) |
| `KvStore` | `cache/mod.rs` | Trait abstracting over cache formats |
| `Sampler` | `sampling/sampler.rs` | Owns config and PRNG state for token selection |
| `GpuBackend` | `backend/gpu.rs` | wgpu device + queue + compiled pipelines |
| `GlintError` | `error.rs` | Project-wide error enum |

## Feature Flags

| Flag | Default | Enables |
|------|---------|---------|
| `rayon` | ✅ | Parallel matmul across CPU cores; required for SIMD kernels |
| `server` | ✅ | HTTP API (axum/tokio) + HF Hub download (reqwest) |
| `python` | — | PyO3 extension module |
| `wasm` | — | wasm-bindgen JS bindings |
| `vulkan` | — | wgpu GPU compute backend |

SIMD kernels in `src/tensor/simd.rs` are only compiled when **both** `x86_64` architecture and the `rayon` feature are active.

## Concurrency Model

The HTTP server runs in an `async` tokio runtime. Because transformer inference is CPU-bound, it runs on a dedicated OS thread (spawned by `InferenceEngine::start`). Route handlers submit requests via an `mpsc` channel and receive generated tokens through a per-request `mpsc::Receiver`. The SSE streaming response wraps this receiver in a `ReceiverStream`.

```
tokio async threads          OS thread (blocking)
─────────────────          ─────────────────────
route handler              InferenceEngine::run_loop()
  │                            │
  │ engine.submit(tokens) ──▶  │ forward_one() per token
  │                            │
  │ ◀── mpsc::Receiver ──────  │ sender.send(token_id)
  │
  └─▶ SSE stream to client
```
