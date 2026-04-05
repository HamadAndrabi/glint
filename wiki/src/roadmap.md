# Roadmap & Future Exploration

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

### PagedAttention

The core innovation from [vLLM (Kwon et al.)](https://arxiv.org/abs/2309.05657). Instead of pre-allocating a contiguous KV cache for each request, use virtual memory-style paging:

- KV cache divided into fixed-size **pages** (e.g., 16 tokens per page)
- Pages allocated on demand and freed when sequences complete
- Sequences can share pages for common prefixes (e.g., system prompts)
- Eliminates memory fragmentation from variable-length sequences

**Why it matters:** Current `KvCache` pre-allocates `max_seq_len` slots per request. With 100 concurrent requests and 4096 context, that's 100 × full KV-cache allocations even if most requests are short. PagedAttention reduces peak memory usage significantly.

### SafeTensors Format Support

Add loading of the [SafeTensors](https://github.com/huggingface/safetensors) format used by most HuggingFace models. Currently Glint requires GGUF; with SafeTensors, models can be loaded directly without conversion.

### Continuous Batching Improvements

The current `InferenceEngine` already serves multiple requests concurrently, but it does so by interleaving one decode step per active request. True continuous batching would go further:
- Multiple active sequences share a single forward pass
- Logit computation amortizes weight loading across batch
- New requests join mid-generation when slots free up

This is the key throughput multiplier for high-concurrency serving.

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

### Prefix Caching

Reuse KV-cache across requests that share a common prefix (e.g., a long system prompt). If 1000 users all send a 2048-token system prompt, the KV-cache for those tokens could be computed once and shared.

Requires the PagedAttention infrastructure (shared pages) to implement efficiently.

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
- KV-cache hit rate (for prefix caching)
- Token throughput per request

---

## Contributing Ideas

If you're looking for a well-scoped contribution:

| Difficulty | Task |
|-----------|------|
| Beginner | Add Q4_1 / Q5_0 dequantization (follow Q4_0 pattern) |
| Beginner | Add `--format json` output flag to `inspect` subcommand |
| Intermediate | Add SafeTensors loading |
| Intermediate | Implement prefix caching for the HTTP server |
| Intermediate | Add Python streaming callback (generator-based) |
| Advanced | PagedAttention KV cache manager |
| Advanced | Continuous batching (multiple sequences per forward pass) |
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
