# ferrite- A Rust LLM Inference Engine — Project Roadmap

## Project Vision

Build a production-grade LLM inference engine from scratch in Rust, starting with CPU-only support and progressively adding optimizations, GPU backends, and a serving layer. The engine will be capable of loading quantized models (GGUF format), running efficient transformer inference, and serving multiple concurrent requests via an HTTP API.

**Target Hardware (Development):** AMD Ryzen 7 with integrated Radeon graphics, no discrete GPU.
**Primary Model Targets:** SmolLM 135M (dev/test), TinyLlama 1.1B (benchmark), Llama 3 8B quantized (stretch).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   HTTP API Layer                     │
│              (axum + tokio, SSE streaming)           │
├─────────────────────────────────────────────────────┤
│                 Scheduling & Batching                │
│         (request queue, continuous batching)         │
├─────────────────────────────────────────────────────┤
│                 Inference Runtime                    │
│     (transformer forward pass, KV-cache, sampling)  │
├─────────────────────────────────────────────────────┤
│                  Tensor Operations                  │
│      (matmul, softmax, RMSNorm, RoPE, GELU/SiLU)   │
├─────────────────────────────────────────────────────┤
│                Compute Backends                     │
│        CPU (AVX2/AVX-512)  │  Vulkan  │  WASM      │
├─────────────────────────────────────────────────────┤
│                  Model Loading                      │
│        (GGUF parser, mmap, quantization formats)    │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundations — Model Loading & Naive Inference

**Goal:** Load a GGUF model file and generate one token with a correct transformer forward pass. No optimization — just correctness.

**Duration:** 2–3 weeks

### 1.1 Project Setup & GGUF Parser

**What to build:**

- Initialize the Rust project with a clean module structure.
- Implement a GGUF file format parser that reads the header, metadata key-value pairs, and tensor descriptors (name, shape, quantization type, offset).
- Use `memmap2` to memory-map the file rather than reading it all into heap memory.

**Key concepts to learn:**

- GGUF file format specification (magic number, version, tensor info layout).
- Memory-mapped I/O in Rust (`memmap2` crate).
- Rust byte-level parsing (reading little-endian integers, floats from raw bytes).

**Deliverables:**

- A CLI tool that takes a `.gguf` file path and prints all metadata and tensor names/shapes/types.
- A `GGUFModel` struct that provides access to any tensor by name as a raw byte slice.

**Resources:**

- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- `memmap2` crate docs.
- Reference: study how `llama.cpp` reads GGUF files (`llama.cpp/gguf.c`).

**Test model:** Download `SmolLM-135M-Instruct` in GGUF format (Q8_0 or F16).

### 1.2 Tensor Primitives (f32, Naive)

**What to build:**

- A simple `Tensor` struct: contiguous f32 data + shape + strides.
- Basic operations, all in naive f32 (no SIMD, no quantization yet):
  - Matrix multiplication (2D): `[M, K] × [K, N] → [M, N]`
  - Element-wise addition
  - RMSNorm: `x * rsqrt(mean(x²) + eps) * weight`
  - SiLU activation: `x * sigmoid(x)`
  - Softmax (for attention scores)
  - RoPE (Rotary Positional Embeddings)
  - Embedding lookup (index into weight matrix)

**Key concepts:**

- Row-major memory layout and strided access.
- Why matmul is the bottleneck (O(n³) compute, memory-bandwidth bound).
- How RoPE encodes position without additive embeddings.

**Deliverables:**

- A `tensor` module with all the above ops, each with unit tests comparing against known outputs.
- A standalone benchmark for matmul at different sizes (128×128, 512×512, 2048×2048).

**Testing strategy:**

- Compute expected outputs using Python/NumPy for small cases.
- Compare your Rust results within floating-point tolerance (1e-5).

### 1.3 Transformer Forward Pass

**What to build:**

- Wire up the full decoder-only transformer architecture:
  1. Token embedding lookup
  2. For each layer:
     - RMSNorm (pre-attention)
     - Multi-head self-attention (Q, K, V projections → RoPE → scaled dot-product attention → output projection)
     - Residual connection
     - RMSNorm (pre-FFN)
     - Feed-forward network (gate_proj + up_proj → SiLU → down_proj)
     - Residual connection
  3. Final RMSNorm
  4. LM head projection → logits
- Implement greedy decoding: pick argmax of logits, feed back as next token.

**Key concepts:**

- The transformer architecture at the implementation level.
- How attention masks work for causal (autoregressive) decoding.
- Residual connections and why they matter for gradient flow / stability.
- Weight naming conventions in GGUF (mapping layer names to transformer components).

**Deliverables:**

- Given a prompt tokenized as `[1, 15043, 29892, 590]` (or similar), the engine produces a next token.
- Full forward pass produces the same logits as a reference implementation (compare against `llama.cpp` or HuggingFace Transformers output for the same model and input).

**Validation:**

- Run the same prompt through HuggingFace Transformers in Python.
- Compare logits at each layer. They should match within tolerance.
- If logits diverge, debug layer by layer until you find the discrepancy.

### 1.4 Tokenizer

**What to build:**

- Integrate a tokenizer. Two options:
  - **Option A (recommended to start):** Use the `tokenizers` crate (HuggingFace's Rust tokenizer library). This handles BPE/SentencePiece natively.
  - **Option B (for learning):** Implement BPE tokenization from scratch by reading the tokenizer model file.
- Handle special tokens (BOS, EOS, padding).

**Deliverables:**

- Encode a string → token IDs → feed to model → get output token IDs → decode back to string.
- End-to-end: type a prompt, get generated text out.

---

## Phase 2: CPU Performance Optimization

**Goal:** Make inference fast enough to be usable on your Ryzen 7. Target: 10+ tokens/sec for SmolLM-135M, 2+ tokens/sec for TinyLlama-1.1B (quantized).

**Duration:** 3–4 weeks

### 2.1 KV-Cache

**What to build:**

- Instead of recomputing attention over all previous tokens at each step, cache the K and V tensors from previous positions.
- On each new token, only compute Q/K/V for the new position, append K and V to the cache, and compute attention against the full cached K/V.

**Why it matters:**

- Without KV-cache, generation is O(n²) per token (where n = sequence length). With it, each new token is O(n) — a massive speedup for longer sequences.

**Key concepts:**

- Pre-allocated cache buffers sized to max sequence length.
- Cache layout: `[layer, max_seq_len, n_heads, head_dim]`.
- Position tracking for where to write the next K/V entry.

**Deliverables:**

- Before/after benchmarks showing speedup on 100-token generation.
- Verify that outputs remain identical with and without KV-cache.

### 2.2 Quantized Inference (Q8_0 and Q4_0)

**What to build:**

- Implement dequantization routines for GGUF quantization formats:
  - **Q8_0:** 32 values stored as 8-bit integers + one f16 scale factor per block.
  - **Q4_0:** 32 values stored as 4-bit integers (packed two per byte) + one f16 scale factor.
- Implement quantized matrix multiplication: instead of dequantizing entire tensors to f32, operate directly on quantized blocks.
  - For Q8_0: dot product of two int8 vectors, multiply by scales.
  - For Q4_0: unpack nibbles, compute dot product, multiply by scale.

**Why it matters:**

- A 7B parameter model in f32 = 28 GB. In Q4_0 = ~3.5 GB. This is the difference between "doesn't fit in RAM" and "runs on a laptop."
- Quantized matmul is also faster because you're moving less data through the memory bus (inference is memory-bandwidth-bound, not compute-bound, for batch size 1).

**Key concepts:**

- Block quantization: why quantize in blocks of 32 rather than per-tensor.
- The tradeoff between quantization precision and speed/memory.
- How f16 scale factors preserve dynamic range.

**Deliverables:**

- Load Q4_0 and Q8_0 GGUF models and run inference without first dequantizing to f32.
- Benchmark: tokens/sec and memory usage for f32 vs Q8_0 vs Q4_0.
- Perplexity comparison (optional): verify that quantized output quality is comparable.

### 2.3 SIMD Optimization (AVX2)

**What to build:**

- Rewrite the hot-path operations using AVX2 SIMD intrinsics (your Ryzen 7 supports this):
  - f32 matmul with `_mm256_fmadd_ps` (fused multiply-add on 8 floats).
  - Quantized dot products: `_mm256_maddubs_epi16` for int8, manual nibble unpacking for int4.
  - RMSNorm and SiLU with vectorized math.
- Use Rust's `std::arch::x86_64` module for intrinsics (requires `unsafe` blocks).

**Key concepts:**

- SIMD (Single Instruction Multiple Data): process 8 f32s or 32 int8s in one instruction.
- Data alignment requirements (256-bit = 32-byte alignment).
- Loop tiling and cache-friendly access patterns.
- Benchmarking with `criterion` to measure actual speedup.

**Deliverables:**

- SIMD-optimized matmul that is 4–8x faster than naive implementation.
- Feature-gated code: `#[cfg(target_feature = "avx2")]` so it falls back gracefully.
- A micro-benchmark suite comparing naive vs. SIMD for each operation.

**Stretch (if your Ryzen supports it):**

- AVX-512 versions for even wider vectors (16 f32s per instruction). Check with `cat /proc/cpuinfo | grep avx512`.

### 2.4 Memory & Threading Optimization

**What to build:**

- Thread-parallel matmul using `rayon`: split the output rows across threads.
- Ensure tensor data is properly aligned for SIMD (use aligned allocators).
- Profile memory allocation: minimize heap allocations during inference (pre-allocate all scratch buffers).
- Use `mmap` with `MAP_POPULATE` to pre-fault pages and avoid page faults during inference.

**Key concepts:**

- Why memory bandwidth is the bottleneck for LLM inference at batch size 1.
- Arithmetic intensity and the roofline model.
- Thread overhead vs. parallelism gains for different matrix sizes.

**Deliverables:**

- Multi-threaded inference that scales with core count.
- A profiling report (using `perf` or `flamegraph`) showing where time is spent.
- Zero heap allocations during the token generation loop (all pre-allocated).

---

## Phase 3: Serving Layer

**Goal:** Expose the inference engine as an HTTP API compatible with the OpenAI API format, supporting multiple concurrent users with streaming.

**Duration:** 2–3 weeks

### 3.1 HTTP API with Streaming

**What to build:**

- An `axum` + `tokio` HTTP server with the following endpoints:
  - `POST /v1/completions` — text completion.
  - `POST /v1/chat/completions` — chat-format completion.
  - `GET /v1/models` — list loaded models.
- Server-Sent Events (SSE) streaming: send each token as it's generated rather than waiting for the full response.
- Request/response format compatible with the OpenAI API spec (so existing clients and tools work out of the box).

**Key concepts:**

- Async Rust with tokio: the inference itself is CPU-bound, so run it on `tokio::task::spawn_blocking` to avoid blocking the async runtime.
- SSE streaming pattern in axum.
- Request validation, error handling, and graceful shutdown.

**Deliverables:**

- A running server that you can `curl` or hit with any OpenAI-compatible client.
- Streaming responses that appear token-by-token.
- Proper error responses (model not loaded, invalid parameters, etc.).

### 3.2 Sampling Strategies

**What to build:**

- Go beyond greedy decoding. Implement:
  - **Temperature scaling:** divide logits by temperature before softmax.
  - **Top-k sampling:** zero out all logits except the top k.
  - **Top-p (nucleus) sampling:** keep the smallest set of tokens whose cumulative probability exceeds p.
  - **Repetition penalty:** reduce logits for tokens that have already appeared.
  - **Min-p sampling:** filter tokens below a minimum probability threshold.
- Expose these as API parameters.

**Deliverables:**

- Configurable sampling via API parameters (`temperature`, `top_k`, `top_p`, `repeat_penalty`).
- Demonstrate qualitative difference: greedy vs. creative sampling outputs.

### 3.3 Continuous Batching

**What to build:**

- When multiple requests arrive, batch them together for more efficient execution:
  - Maintain a pool of active sequences, each with its own KV-cache.
  - At each step, run a batched forward pass for all active sequences.
  - When a sequence finishes (hits EOS or max length), remove it and admit a new one from the queue.
- This is the key technique that makes vLLM, TGI, etc. efficient at high concurrency.

**Key concepts:**

- Why batching improves GPU/memory utilization (you're amortizing the weight-loading cost).
- The scheduling problem: balancing latency for individual requests vs. throughput.
- Memory management: pre-allocate a KV-cache pool and assign slots to requests.

**Deliverables:**

- Benchmark: throughput (tokens/sec total) with 1, 2, 4, 8 concurrent requests.
- Demonstrate that batching improves throughput without proportionally hurting latency.

---

## Phase 4: Advanced Optimizations

**Goal:** Push performance further and add features that differentiate this from a toy project.

**Duration:** 3–4 weeks (can be ongoing)

### 4.1 Speculative Decoding

**What to build:**

- Use a small "draft" model (e.g., SmolLM-135M) to speculatively generate k tokens, then verify them in a single forward pass of the larger "target" model (e.g., TinyLlama-1.1B).
- Accepted tokens are free; rejected tokens are re-sampled from the target model's distribution.
- This can give 2–3x speedup when the draft model's acceptance rate is high.

**Key concepts:**

- Why speculative decoding works (the target model verifies in parallel what the draft model guessed sequentially).
- Acceptance/rejection sampling to maintain the exact output distribution of the target model.
- How to choose a good draft model (similar tokenizer, much smaller, decent agreement rate).

### 4.2 Flash Attention (CPU Variant)

**What to build:**

- Implement a fused attention kernel that computes `softmax(QK^T / sqrt(d)) * V` without materializing the full `[seq_len, seq_len]` attention matrix.
- On CPU, this means tiling the computation to fit in L1/L2 cache.

**Why it matters:**

- Standard attention materializes an `[N, N]` matrix which is O(N²) memory. Flash attention is O(N) memory.
- For longer sequences (2K+ tokens), this is the difference between feasible and out-of-memory.

### 4.3 Model Format Support

**What to build:**

- Add support for loading SafeTensors format (used by most HuggingFace models).
- Add support for more GGUF quantization types: Q5_0, Q5_1, Q4_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (the "K-quants" from llama.cpp).

### 4.4 PagedAttention (Optional, Advanced)

**What to build:**

- Implement PagedAttention (the core innovation of vLLM): manage KV-cache as virtual memory pages rather than contiguous blocks.
- This allows sequences to share physical memory pages (e.g., shared system prompt) and eliminates memory fragmentation.

---

## Phase 5: GPU & Cross-Platform Backends

**Goal:** Run inference on GPU hardware using Vulkan (works on your AMD iGPU and essentially all GPUs).

**Duration:** 4–6 weeks

### 5.1 Vulkan Compute Backend

**What to build:**

- Use the `vulkano` or `ash` crate to write compute shaders for the core operations (matmul, attention, element-wise ops).
- Implement a backend trait so the inference engine can dispatch to either CPU or Vulkan.
- Handle memory transfers between host and device.

**Why Vulkan over CUDA:**

- Works on your AMD integrated GPU (no NVIDIA required).
- Cross-platform: Windows, Linux, macOS (via MoltenVK), Android.
- Good learning experience for GPU compute programming.

**Key concepts:**

- Compute shaders in GLSL/SPIR-V.
- GPU memory management (buffers, memory types, transfers).
- Work group sizing and occupancy.
- The `Backend` trait pattern: abstract over CPU and GPU dispatch.

### 5.2 WASM Backend (Optional, Differentiator)

**What to build:**

- Compile the core inference engine to WebAssembly.
- Run small models (SmolLM-135M) directly in the browser.
- Use WASM SIMD for vectorized operations.

**Why this is interesting:**

- Very few projects do this well. It's a genuine differentiator.
- Demonstrates that the architecture is clean and portable.
- Cool demo: a fully client-side LLM chatbot with no server.

---

## Suggested Crate Dependencies

| Crate                  | Purpose                                    | Phase |
| ---------------------- | ------------------------------------------ | ----- |
| `memmap2`              | Memory-mapped file I/O for model loading   | 1     |
| `tokenizers`           | HuggingFace BPE/SentencePiece tokenizer    | 1     |
| `half`                 | f16 type support (for quantization scales) | 1     |
| `byteorder`            | Reading little-endian values from bytes    | 1     |
| `criterion`            | Micro-benchmarking                         | 2     |
| `rayon`                | Data parallelism for matmul                | 2     |
| `axum`                 | HTTP server framework                      | 3     |
| `tokio`                | Async runtime                              | 3     |
| `serde` / `serde_json` | JSON serialization for API                 | 3     |
| `tower-http`           | CORS, logging middleware                   | 3     |
| `tracing`              | Structured logging                         | 3     |
| `vulkano` or `ash`     | Vulkan compute backend                     | 5     |
| `clap`                 | CLI argument parsing                       | 1     |

---

## Project Structure

```
inference-engine/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs                  # CLI entry point
│   ├── lib.rs                   # Library root
│   ├── model/
│   │   ├── mod.rs
│   │   ├── gguf.rs              # GGUF file parser
│   │   ├── safetensors.rs       # SafeTensors loader (Phase 4)
│   │   └── config.rs            # Model hyperparameters
│   ├── tensor/
│   │   ├── mod.rs
│   │   ├── tensor.rs            # Core Tensor struct
│   │   ├── ops.rs               # Naive f32 operations
│   │   ├── quantize.rs          # Quantization/dequantization
│   │   └── simd.rs              # AVX2/AVX-512 kernels
│   ├── transformer/
│   │   ├── mod.rs
│   │   ├── attention.rs         # Multi-head attention
│   │   ├── ffn.rs               # Feed-forward network
│   │   ├── norm.rs              # RMSNorm
│   │   ├── rope.rs              # Rotary positional embeddings
│   │   └── forward.rs           # Full forward pass orchestration
│   ├── cache/
│   │   ├── mod.rs
│   │   └── kv_cache.rs          # KV-cache management
│   ├── sampling/
│   │   ├── mod.rs
│   │   └── sampler.rs           # Temperature, top-k, top-p, etc.
│   ├── server/
│   │   ├── mod.rs
│   │   ├── api.rs               # Route handlers
│   │   ├── types.rs             # Request/response types
│   │   └── scheduler.rs         # Request batching & scheduling
│   └── backend/
│       ├── mod.rs
│       ├── cpu.rs               # CPU compute backend
│       └── vulkan.rs            # Vulkan compute backend (Phase 5)
├── benches/
│   ├── matmul.rs                # Matmul micro-benchmarks
│   └── inference.rs             # End-to-end inference benchmarks
└── tests/
    ├── gguf_parser.rs           # Parser correctness tests
    ├── tensor_ops.rs            # Op correctness vs. NumPy
    └── forward_pass.rs          # Full model output comparison
```

---

## Milestone Checkpoints

| Milestone                | What You Can Demo                                       | Target  |
| ------------------------ | ------------------------------------------------------- | ------- |
| M1: Parser works         | CLI prints model metadata and tensor shapes             | Week 2  |
| M2: First token          | Engine outputs a single correct next token for a prompt | Week 4  |
| M3: Text generation      | Type a prompt, get coherent multi-token output          | Week 5  |
| M4: Fast inference       | 10+ tok/s SmolLM-135M, 2+ tok/s TinyLlama-1.1B (Q4)     | Week 9  |
| M5: Server running       | `curl` the API and get streaming responses              | Week 11 |
| M6: Multi-user           | Handle concurrent requests with continuous batching     | Week 13 |
| M7: Speculative decoding | Demonstrable speedup with draft model                   | Week 16 |
| M8: Vulkan backend       | Inference running on AMD iGPU                           | Week 22 |

---

## Key Reference Materials

**Codebases to study:**

- `llama.cpp` — The gold standard for CPU inference. Study its GGUF handling, quantization kernels, and KV-cache implementation.
- `candle` (Hugging Face) — Rust-native ML framework. Study its tensor abstraction and backend trait pattern.
- `mistral.rs` — Smaller Rust inference project, good for seeing a clean implementation.
- `vLLM` — For understanding PagedAttention and continuous batching (Python, but the concepts transfer).

**Papers:**

- "Attention Is All You Need" (Vaswani et al.) — The transformer architecture.
- "FlashAttention" (Dao et al.) — Memory-efficient attention.
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.).
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al.) — vLLM paper.

**Specs:**

- GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- OpenAI API spec: https://platform.openai.com/docs/api-reference

---

## Tips

- **Test against a reference constantly.** After every new op or layer, compare your output against HuggingFace Transformers or llama.cpp. Bugs in matrix indexing or normalization are silent — the model just produces garbage instead of crashing.
- **Profile before optimizing.** Use `perf`, `flamegraph`, or `criterion` to find the actual bottleneck before reaching for SIMD. The bottleneck is almost always matmul, but verify.
- **Start with the smallest model possible.** SmolLM-135M is fast to iterate with. Only move to larger models once correctness is confirmed.
- **Git tag each milestone.** You'll want to go back and compare performance or behavior at different stages.
- **Write a devlog.** Documenting what you learn as you build this is incredibly valuable for your career narrative and for the community. A blog series on "Building an LLM Inference Engine in Rust" would be very well-received.
