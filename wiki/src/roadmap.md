# Roadmap & Future Exploration

> **Note:** The "What's Built" list below reflects an earlier milestone and lags
> the current code (which has since added K-quant SIMD, runtime sessions and
> snapshots, C FFI, LoRA, speculative decoding, and constrained decoding). Treat
> [README.md](https://github.com/HamadAndrabi/glint/blob/master/README.md) as the
> source of truth for shipped features.

This page documents what's already built, what's planned next, and the longer-term exploration paths worth investigating.

---

## What's Built (Current State)

All five original phases have shipped to varying degrees:

| Phase | Goal | Status |
|-------|------|--------|
| 1 — Foundations | GGUF parsing, f32 tensors, forward pass, tokenizer | ✅ Complete |
| 2 — CPU Optimization | KV-cache, quantized inference, AVX2 SIMD, rayon | ✅ Complete |
| 3 — Serving Layer | HTTP API, sampling strategies, queued concurrent serving | ✅ Complete |
| 4 — Advanced Opts | Speculative decoding, flash attention, K-quants, LoRA | ✅ Complete |
| 5 — GPU Backends | Vulkan via wgpu, WASM browser inference | ✅ Complete |

**Milestones achieved:**
- `glint inspect` prints model metadata ✅
- First correct token generated ✅
- Multi-token coherent text generation ✅
- 10+ tok/s on SmolLM-135M (Q8_0, AVX2) ✅
- OpenAI-compatible server with streaming ✅
- Multi-user concurrent requests ✅
- Speculative decoding working ✅
- Vulkan GPU backend ✅

---

## Near-Term: Planned Work

### PagedAttention — shipped ✅

The core innovation from [vLLM (Kwon et al.)](https://arxiv.org/abs/2309.05657): instead of pre-allocating a contiguous KV cache for each request, use virtual memory-style paging. `PagedKvCache` and `PagePool` (`src/cache/paged.rs`) implement it:

- KV divided into fixed-size **pages** of 16 tokens, one page per layer
- Pages allocated on demand from a shared pool and returned when sequences finish
- Pages are reference-counted, so sequences can share a common prefix
  (`fork_from`), with copy-on-write the moment either side writes a shared page
- Pool exhaustion is a recoverable error, not a panic: the engine ends that one
  sequence and keeps serving

Opt in with `glint serve --kv-cache paged`, or `EngineLimits::kv_pool_pages` /
`SessionOptions::page_pool` in library code. The default is still the
pre-allocated `KvCache`. See [KV Cache](./kv-cache.md) for the design.

**Still open:** paged storage for the Q8 cache format.

### Prefix Caching — ✅ shipped

Requests that share a prompt prefix — a thousand chats behind the same
2 000-token system prompt — used to re-run that prefill every time and compute
bit-identical K/V every time. `PrefixCache` (`src/cache/prefix.rs`) retains the
pages instead:

- The registry keeps a page-sharing `fork_from` of each completed prefill and
  finds the longest cached prefix of a new prompt ✅
- Sharing is whole-page only, keyed by a chained per-page hash and then
  verified against the tokens, so a hash collision costs a missed reuse and
  never a wrong answer ✅
- Admission forks the prefix and prefills only the suffix, at a non-zero
  position offset — output is bit-identical to a cold prefill ✅
- Bounded by entries and pages, LRU-evicted; a request that runs out of pages
  reclaims prefixes and retries, so a cached prefix never starves live work ✅
- Hit/miss/eviction counters and pool occupancy on `/v1/metrics` ✅

Opt in with `glint serve --kv-cache paged --prefix-cache`, or
`EngineLimits::prefix_cache` in library code. Off by default. See
[KV Cache → Prefix Caching](./kv-cache.md#prefix-caching).

Still open here: sharing prefixes across *restarts* (the registry is in-memory
and per-process), and a page-level trie so lookup is a descent rather than a
scan over entries — worth it only at a much larger entry budget.

### SafeTensors Format Support — shipped

Loading of the [SafeTensors](https://github.com/huggingface/safetensors) format
used by most HuggingFace models. Point any command at a model directory
(`config.json` + `tokenizer.json` + `*.safetensors`, sharded or not) and it
loads without a GGUF conversion step:

```bash
glint run --file ./SmolLM2-135M-Instruct --prompt "Hello"
```

Scope: LLaMA-family architectures (`llama`, `mistral`, and relatives), F32 /
F16 / BF16 weights, and byte-level BPE tokenizers. Anything Glint's forward
pass cannot express — fused QKV projections, attention biases, per-head
QK norms, non-linear RoPE scaling, SentencePiece tokenizers — is rejected with
a specific error rather than loaded into a silently wrong run.

### Continuous Batching — ✅ shipped

The `InferenceEngine` used to serve multiple requests by interleaving one decode
step per active request, so every sequence paid the full cost of streaming the
model's weights from memory. It now decodes all active sequences in a single
forward pass per step:

- Multiple active sequences share one forward pass ✅
- Each weight matrix is traversed once per step, not once per sequence, so the
  cost of a step is roughly flat in batch size ✅
- New requests join mid-generation as slots free up; finished sequences leave
  without stalling the rest ✅

Batching is bit-identical to decoding each sequence alone, so a response never
depends on how busy the server was. See
[Inference Engine → Continuous batching](./server-api.md#continuous-batching).

Still open here: batching the *prefill* of newly admitted requests (today each
admitted prompt is prefilled on its own), and chunked prefill so a long prompt
cannot delay a step for the sequences already decoding.

### Quantization: More Formats

- **Q4_1 / Q5_0 / Q5_1** — simple quant variants not yet implemented
- **GPTQ** — post-training quantization scheme common in HuggingFace ecosystem
- **fp8** — 8-bit float (E4M3/E5M2), increasingly common in new models
- **AWQ** — activation-aware weight quantization, better quality at low bits

---

## Medium-Term: Exploration Paths

### Multi-Model Serving

Currently one model per server process. Multi-model support would allow:
- Single server loading multiple models
- Dynamic model swapping (LRU cache, load on demand)
- Routing requests to different model variants based on task

Architecture question: shared memory-mapped weight files, or separate process-per-model with a multiplexing proxy?

### Quantized Training / QLoRA

Fine-tune directly on quantized models. QLoRA (Dettmers et al.) uses:
- 4-bit NormalFloat (NF4) quantization for base weights
- Double quantization (quantize the scale constants too)
- LoRA adapters trained in bf16

This would make Glint a training-capable engine, not just inference-only.

### Model Architecture Extensions

Currently Glint targets the LLaMA/Mistral family. Extensions to explore:
- **Mixture-of-Experts (MoE)** — Mixtral 8×7B routes tokens to expert sub-networks; requires modified forward pass with expert routing
- **Sliding window attention** — Mistral uses a sliding attention window; partially handled via `config.sliding_window` but not fully exploited
- **Multi-modal** — image token projection (LLaVA-style), audio tokens
- **State space models** — Mamba/Jamba architecture as alternative to attention

### Structured Output / Grammar-Guided Decoding

Constrain generation to follow a schema (JSON, regex, context-free grammar). Implementation approaches:
- Token masking: at each step, mask logits for tokens that would violate the grammar
- Requires a parser that can determine valid next tokens at each position

Libraries to study: `outlines`, `guidance`, `llama.cpp`'s grammar sampler.

---

## Long-Term: Ambitious Directions

### CUDA/ROCm Backend

`wgpu` covers Vulkan/Metal/DX12 but not CUDA. A native CUDA backend would unlock:
- NCCL for multi-GPU tensor parallelism
- cuBLAS/cuBLASLt for highly optimized matmul
- Flash Attention 2/3 CUDA kernels (Tri Dao's implementation)

Alternative: expose an interface that lets external CUDA kernels plug in, keeping glint's core GPU-agnostic.

### Tensor Parallelism

Split model weights across multiple GPUs (or machines). Attention heads can be split across GPUs trivially; FFN layers require an allreduce after each layer. This is how production systems serve 70B+ models.

### Speculative Decoding: Medusa / Multi-Draft

Beyond standard speculative decoding:
- **Medusa** — add extra prediction heads to the target model itself, avoiding a separate draft model
- **EAGLE** — draft model that observes target model feature vectors (higher acceptance rate)
- **Lookahead decoding** — use n-grams from past generation as draft candidates

### AVX-512

AVX-512 doubles the SIMD register width (512 bits = 16 f32 or 64 int8). Most modern Intel/AMD CPUs support it. The dispatch infrastructure already exists; adding AVX-512 kernels is a matter of implementing them in `simd.rs` with proper `#[cfg]` guards.

### Inference Profiling Dashboard

A built-in web UI at `/dashboard` that shows:
- Per-layer latency breakdown
- Memory usage over time
- KV-cache and prefix-cache hit rate over time (the counters exist on
  `/v1/metrics`; what is missing is history and a UI)
- Token throughput per request

---

## Contributing Ideas

If you're looking for a well-scoped contribution:

| Difficulty | Task |
|-----------|------|
| Beginner | Add `--format json` output flag to `inspect` subcommand |
| Intermediate | Add Python streaming callback (generator-based) |
| Intermediate | Paged storage for the Q8 KV-cache format |
| Advanced | Batched prefill (admit several queued prompts in one pass) |
| Advanced | MoE routing in `forward.rs` |

---

## Reference Materials

**Papers:**
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the transformer architecture
- ["FlashAttention"](https://arxiv.org/abs/2205.14135) — memory-efficient attention
- ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)
- ["Efficient Memory Management for LLM Serving with PagedAttention"](https://arxiv.org/abs/2309.05657)
- ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)
- ["EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077)

**Codebases:**
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) — gold standard for CPU inference; study GGUF handling and SIMD kernels
- [`candle`](https://github.com/huggingface/candle) — Rust-native ML framework; study tensor abstraction and backend traits
- [`vLLM`](https://github.com/vllm-project/vllm) — PagedAttention and continuous batching reference
- [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs) — another Rust inference engine; good for architecture comparison
