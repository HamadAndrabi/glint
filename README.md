# <div align="center">Glint</div>

<div align="center">
  <img src="assets/logo.svg" alt="Glint logo" width="280" />

  <p><strong>A high-performance LLM inference engine built from scratch in Rust.</strong></p>

  <p>
    Glint loads GGUF and HuggingFace SafeTensors models, runs the full transformer forward pass,
    and serves an OpenAI-compatible HTTP API without depending on PyTorch, ONNX, or any other ML framework.
  </p>

  <p>
    <img alt="Rust" src="https://img.shields.io/badge/Rust-2021-black?logo=rust" />
    <img alt="License" src="https://img.shields.io/badge/license-MIT-blue" />
  </p>
</div>

## Why Glint?

Glint is a focused inference engine for GGUF-based LLaMA-family models. It is designed to be small, understandable, and fast on CPU while still covering the pieces that make a local inference stack practical:

- zero-copy GGUF and SafeTensors loading with memory-mapped I/O
- quantized execution with SIMD kernels and Rayon parallelism
- KV-cache backed autoregressive generation
- built-in tokenizer and chat template detection
- OpenAI-compatible HTTP endpoints with SSE streaming

## Feature Overview

| Area | What you get |
| --- | --- |
| Inference | Full LLaMA-family transformer: RMSNorm, RoPE, SwiGLU, grouped-query attention, flash attention, batched prefill |
| Generation | Greedy, sampling (temperature/top-k/top-p/min-p/repetition-penalty/seed), streaming, speculative decoding, JSON object mode |
| Quantization | Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL — weights stay compressed in memory |
| Performance | AVX2+FMA kernels, scalar fallback, Rayon row-parallel matvec, optional GPU backend (wgpu/Vulkan/Metal/DX12) |
| KV Cache | f32, Q8_0-quantized, and PagedAttention-style paged variants (`--kv-cache paged`, refcounted page sharing); prefix caching across requests (`--prefix-cache`); `KvStore` trait abstraction |
| Sessions | First-class `Session` API, deterministic RNG state, snapshot export/import, cache-format-aware resume |
| LoRA | Load and apply LoRA adapters at inference time via `--lora` |
| Server | OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `GET /v1/metrics`, SSE streaming, queued concurrent serving |
| CLI | `run`, `chat`, `serve`, `inspect`, `generate`, `pull`, `bench` |
| Bindings | Python (`pyo3`), browser WASM (`wasm-bindgen`), C FFI (`include/glint.h`), native CLI |

## Build Profiles

CI validates these build surfaces on every push and pull request:

| Surface | Command |
| --- | --- |
| Core CLI + server | `cargo build --release` |
| GPU backend | `cargo check --features vulkan` |
| C FFI | `cargo check --features cffi` |
| Browser / WASM | `cargo check --no-default-features --features wasm` + `wasm-pack build --target web --no-default-features --features wasm` |
| Python bindings | `cargo check --features python` |

## Highlights

### Inference

- Full LLaMA-family transformer implementation in Rust
- KV-cache for efficient per-token generation without recomputing prior context
- Greedy decoding and configurable sampling with temperature, top-k, top-p, repetition penalty, and seeding

### Quantization

- Eight GGUF quantization formats: `Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, `IQ4_NL`
- AVX2+FMA SIMD kernels for the five hottest formats (`Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`); the rest use the scalar path
- Compressed weights stay compressed in memory
- Example footprint: a 135M-parameter `Q8_0` model is about 140 MB in memory versus about 540 MB dequantized

### Server

- OpenAI-style completions and chat completions APIs
- `response_format: { "type": "json_object" }` for constrained JSON object output
- Streaming responses via Server-Sent Events
- Request-queue concurrency with interleaved decode steps
- CORS enabled for browser clients
- `/health` endpoint for readiness checks and simple orchestration

### Model Loading

- Zero-copy GGUF parser using `memmap2`
- Tokenizer loaded directly from GGUF vocabulary
- HuggingFace SafeTensors models load without conversion: point any command at a
  directory holding `config.json` + `tokenizer.json` + `*.safetensors` (sharded or
  not), or at one of its `.safetensors` files. F32/F16/BF16 weights and byte-level
  BPE tokenizers, LLaMA-family architectures; GGUF remains the default path
- Chat template detection for ChatML, Llama 3, Mistral, Zephyr, Gemma, and a generic fallback

### LoRA Adapters

- Load low-rank adapters from a `.gguf` file at inference time
- Applied to attention and FFN projections with `alpha/rank` scaling
- `glint run --lora adapter.gguf -f base.gguf -p "Hello" -m 100`

### Speculative Decoding

- Draft model generates k tokens; target verifies in a single batched pass
- Transparent to callers — enabled via `--draft-model` and `--lookahead` flags
- Speeds up generation with minimal accuracy loss

### GPU Acceleration (optional, `--features vulkan`)

- wgpu compute backend: Vulkan on Linux/Windows, Metal on macOS, DX12 on Windows
- WGSL shaders for matvec (Q4_0, Q8_0, f32), RMSNorm, RoPE, softmax, attention, SiLU/mul, add
- Select at runtime with `--gpu`

### Runtime API

- First-class `Model` + `Session` API for embedding Glint in-process
- Snapshot export/import for KV cache, token history, prompt length, and RNG state
- Deterministic resume across f32 and Q8 KV-cache formats

### Browser / WASM

- `wasm-pack build --target web --no-default-features --features wasm`
- `GlintModel` JS class: `new GlintModel(bytes)`, `.generate()`, `.generate_streaming(callback)`
- Demo in `demo/index.html` — drag-drop a `.gguf` and generate in-browser, zero server

### Python Bindings (optional, `--features python`)

- `import glint; m = glint.GlintLLM("model.gguf"); m.generate("Hello", 64)`
- PyO3 integration with `maturin` for seamless packaging

### C FFI (optional, `--features cffi`)

- Opaque-handle ABI in `include/glint.h`
- Load models, create sessions, stream tokens, and export/import snapshots from C-compatible hosts

### Model Pull

- `glint pull Qwen/Qwen2-0.5B-Instruct-GGUF Qwen2-0.5B-Instruct-Q4_K_M.gguf`
- Downloads from HuggingFace Hub with a progress bar
- Saves to `~/.cache/glint/models/` for future use

## Quick Start

```bash
git clone https://github.com/HamadAndrabi/glint
cd glint
cargo build --release
```

## CLI Usage

### Generate text

```bash
glint run -f model.Q4_K_M.gguf -p "The future of AI is" -m 100
```

From a HuggingFace SafeTensors directory (no conversion step):

```bash
glint run -f ./SmolLM2-135M-Instruct -p "The future of AI is" -m 100
```

Sampling example:

```bash
glint run -f model.gguf -p "Once upon a time" -m 200 \
  --temperature 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1
```

With LoRA adapter:

```bash
glint run -f base.gguf --lora adapter.gguf -p "Hello" -m 100
```

With speculative decoding:

```bash
glint run -f target.gguf --draft-model draft.gguf --lookahead 4 -p "Once" -m 200
```

Reproducible sampling (fixed seed):

```bash
glint run -f model.gguf -p "Hello" --seed 42 -m 100
```

### Interactive chat

```bash
glint chat -f model.gguf --system "You are a helpful assistant"
```

With custom sampling:

```bash
glint chat -f model.gguf --temperature 0.8 --top-k 40 --top-p 0.95 -m 512
```

### Run the server

```bash
glint serve -f model.Q4_K_M.gguf -p 8080
```

You can also bind to another host:

```bash
glint serve -f model.gguf --host 0.0.0.0 -p 8080
```

### Inspect a model

```bash
glint inspect -f model.gguf --show-metadata --show-tensors
```

## OpenAI-Compatible API

### Chat completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Streaming completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "prompt": "The meaning of life is",
    "max_tokens": 100,
    "stream": true
  }'
```

### Embeddings (mean-pooled hidden states)

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "model", "input": "Hello world"}'
```

### Metrics (request count, token count, latency)

```bash
curl http://localhost:8080/v1/metrics
```

### Health check

```bash
curl http://localhost:8080/health
```

## Benchmarks

Matvec throughput at `4096 x 4096` (Llama 3 8B scale), AVX2+FMA, Rayon.
Reproduce with `cargo bench --bench matvec` on your own hardware — absolute
numbers are machine-dependent, so treat these as relative ordering, not a
cross-machine baseline:

| Format | Throughput | Time |
| --- | --- | --- |
| `Q4_0` | 24.4 Gelem/s | 687 us |
| `Q8_0` | 22.7 Gelem/s | 739 us |
| `Q4_K` | 20.8 Gelem/s | 808 us |
| `Q6_K` | 18.7 Gelem/s | 899 us |
| `Q5_K` | 17.0 Gelem/s | 984 us |

Run the end-to-end benchmark CLI with:

```bash
glint bench -f model.gguf --mode all
```

For micro-benchmarks of the hot matvec kernels:

```bash
cargo bench --bench matvec
```

## Architecture

```mermaid
flowchart TD
    GGUF["GGUF file / bytes<br/>(mmap or in-memory)"]

    GGUF --> Config["ModelConfig"]
    GGUF --> Tok["BPE Tokenizer"]
    GGUF --> Weights["TransformerWeights<br/>+ LoRA adapters"]
    GGUF --> QT["QuantizedTensor<br/>(weights stay compressed)"]

    QT --> Dispatch["matvec dispatch"]
    Dispatch --> CPU["CPU kernels<br/>(AVX2+FMA or scalar)"]
    Dispatch --> GPU["GPU kernels<br/>(wgpu · Vulkan/Metal/DX12)<br/>feature: vulkan"]

    Weights --> FWD["forward_one /<br/>forward_prefill"]
    Config --> FWD
    FWD <--> KV["KvCache<br/>(f32 or Q8_0)"]
    FWD --> Flash["flash attention<br/>(O(N) memory)"]
    CPU --> FWD
    GPU --> FWD

    FWD --> Spec["speculative decode<br/>(draft + target)"]
    FWD --> Sampler["Sampler<br/>temp · top-k/p · min-p<br/>rep-penalty · seed"]

    Spec --> Sampler

    Sampler --> CLI["CLI output<br/>glint run / chat"]
    Sampler --> Engine["InferenceEngine<br/>queued concurrent serving"]
    Engine --> SSE["HTTP SSE<br/>/v1/completions<br/>/v1/chat/completions<br/>/v1/embeddings"]
    Sampler --> Py["Python<br/>feature: python"]
    Sampler --> WASM["Browser / WASM<br/>feature: wasm"]
    Sampler --> CFFI["C FFI<br/>feature: cffi"]
```

## Project Structure

```text
src/
  main.rs              CLI entry point (inspect, run, chat, serve, generate, pull, bench)
  lib.rs               Crate root; feature-gates server/python/wasm/cffi modules
  error.rs             GlintError
  api/                 Library-facing Model/Session generation API
  ffi/                 C-compatible ABI (feature: cffi)
  python.rs            PyO3 bindings (feature: python)
  wasm.rs              wasm-bindgen bindings (feature: wasm)
  model/
    gguf.rs            GGUF binary parser (mmap + in-memory)
    safetensors.rs     SafeTensors parser + HuggingFace directory loader
    config.rs          Model hyperparameters from metadata or config.json
    tokenizer.rs       BPE tokenizer (GGUF vocab or tokenizer.json)
    chat_template.rs   Chat template detection and rendering
    lora.rs            LoRA adapter loading and application
    pull.rs            HuggingFace Hub download (feature: server)
  tensor/
    tensor.rs          Contiguous f32 tensor
    ops.rs             RMSNorm, RoPE, matmul, softmax, SiLU
    flash.rs           Flash attention (O(N) memory, single-query)
    quantized.rs       Quantized tensor storage and block-wise matvec
    simd.rs            AVX2+FMA kernels (x86_64 + rayon)
    dequantize.rs      Scalar dequantization helpers
  transformer/
    weights.rs         Weight loading from GGUF or SafeTensors
    forward.rs         Forward pass and generation loops
    speculative.rs     Speculative decoding (draft/target verification)
  cache/               KvCache (f32), KvCacheQ8, KvStore trait
  constrained/         Structured-output constraints (JSON object, enums)
  sampling/            Temperature, top-k, top-p, min-p, rep-penalty, seeded RNG
  session/             Session state and KV snapshot serialization
  server/              (feature: server)
    mod.rs             Axum router setup and CORS
    engine.rs          Queued concurrent inference engine
    routes.rs          /health, /v1/models, /v1/metrics, completions, chat, embeddings
    types.rs           OpenAI-compatible request/response shapes
    state.rs           AppState (Arc-wrapped model + config + metrics)
  backend/             (feature: vulkan)
    mod.rs             GpuBackend trait / dispatch stub
    gpu.rs             wgpu Vulkan/Metal/DX12 backend
    pipeline.rs        GPU pipeline management
    shaders/           WGSL compute shaders (12 kernels)
demo/
  index.html           Browser demo UI (drag-drop model, streaming output)
  worker.js            Web Worker running WASM inference off main thread
```

## Testing

The library test suite currently covers tokenizer behavior, GGUF and SafeTensors
parsing (including adversarial/malformed headers), quantization paths, tensor ops,
sampling, KV-cache handling, transformer forward logic, snapshot resume, and C FFI
edge cases.

```bash
cargo test --lib                   # default surface
cargo test --lib --features cffi   # + C FFI edge cases
```

It also fuzzes the untrusted-byte surfaces — the GGUF parser, the SafeTensors
parser, and the KV-snapshot importer — under `fuzz/` (`cargo fuzz run gguf_parse` /
`safetensors_parse` / `snapshot_import`), and runs the unsafe-heavy
`tensor::`/`cache::` modules under Miri.

Two external anchors keep the numerics honest:

- **ggml reference vectors** — hardcoded dequantization outputs for every
  supported format, derived independently from llama.cpp's `ggml-quants.c`
  (`ggml_reference_tests` in `src/tensor/dequantize.rs`), so the kernels are
  validated against the ecosystem's reference rather than against Glint itself.
- **Golden-output parity** — `scripts/golden_parity.sh model.gguf` greedy-decodes
  the same prompt through Glint and llama.cpp and requires byte-identical output.
  A CI workflow runs this weekly across Q8_0 / Q4_K_M / Q2_K / Q3_K_M builds of
  SmolLM2-135M (mixed-format files also cover Q5_0, Q5_1, Q6_K).

## License

MIT. See [LICENSE](LICENSE).
