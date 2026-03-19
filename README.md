# <div align="center">Glint</div>

<div align="center">
  <img src="assets/logo.svg" alt="Glint logo" width="280" />

  <p><strong>A high-performance LLM inference engine built from scratch in Rust.</strong></p>

  <p>
    Glint loads GGUF models, runs the full transformer forward pass, and serves an
    OpenAI-compatible HTTP API without depending on PyTorch, ONNX, or any other ML framework.
  </p>

  <p>
    <img alt="Rust" src="https://img.shields.io/badge/Rust-2021-black?logo=rust" />
    <img alt="Tests" src="https://img.shields.io/badge/tests-84%20passing-success" />
    <img alt="License" src="https://img.shields.io/badge/license-MIT-blue" />
  </p>
</div>

## Why Glint?

Glint is a focused inference engine for GGUF-based LLaMA-family models. It is designed to be small, understandable, and fast on CPU while still covering the pieces that make a local inference stack practical:

- zero-copy GGUF loading with memory-mapped I/O
- quantized execution with SIMD kernels and Rayon parallelism
- KV-cache backed autoregressive generation
- built-in tokenizer and chat template detection
- OpenAI-compatible HTTP endpoints with SSE streaming

## Feature Overview

| Area | What you get |
| --- | --- |
| Inference | Full transformer forward pass with RMSNorm, RoPE, SwiGLU, and grouped-query attention |
| Quantization | Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K support with compressed weights kept in memory |
| Performance | AVX2+FMA kernels where available, scalar fallback elsewhere, Rayon row-parallel matvec |
| Serving | `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, and token streaming over SSE |
| Compatibility | GGUF loading, built-in BPE tokenizer, and chat template auto-detection from model metadata |

## Highlights

### Inference

- Full LLaMA-family transformer implementation in Rust
- KV-cache for efficient per-token generation without recomputing prior context
- Greedy decoding and configurable sampling with temperature, top-k, top-p, repetition penalty, and seeding

### Quantization

- Five commonly used GGUF quantization formats: `Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`
- SIMD kernels for all supported formats on AVX2+FMA systems
- Compressed weights stay compressed in memory
- Example footprint: a 135M-parameter `Q8_0` model is about 140 MB in memory versus about 540 MB dequantized

### Server

- OpenAI-style completions and chat completions APIs
- Streaming responses via Server-Sent Events
- CORS enabled for browser clients
- `/health` endpoint for readiness checks and simple orchestration

### Model Loading

- Zero-copy GGUF parser using `memmap2`
- Tokenizer loaded directly from GGUF vocabulary
- Chat template detection for ChatML, Llama 3, Mistral, Zephyr, Gemma, and a generic fallback

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

Sampling example:

```bash
glint run -f model.gguf -p "Once upon a time" -m 200 \
  --temperature 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1
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

### Health check

```bash
curl http://localhost:8080/health
```

## Benchmarks

Matvec throughput at `4096 x 4096` (Llama 3 8B scale), AVX2+FMA, Rayon:

| Format | Throughput | Time |
| --- | --- | --- |
| `Q4_0` | 24.4 Gelem/s | 687 us |
| `Q8_0` | 22.7 Gelem/s | 739 us |
| `Q4_K` | 20.8 Gelem/s | 808 us |
| `Q6_K` | 18.7 Gelem/s | 899 us |
| `Q5_K` | 17.0 Gelem/s | 984 us |

Run the benchmark suite with:

```bash
cargo bench --bench matvec
```

## Architecture

```text
                    GGUF file (mmap)
                         |
              +----------+----------+
              |          |          |
          Metadata    Tokenizer   Tensor data
              |          |          |
         ModelConfig   BPE      QuantizedTensor
              |        encode/     (raw bytes)
              |        decode        |
              |          |      +----+----+
              |          |      | matvec  |
              |          |      | Q8/Q4/K |
              |          |      | AVX2    |
              |          |      +---------+
              |          |          |
              +-----+----+----+----+
                    |              |
              TransformerWeights   |
                    |              |
               forward(token) -----+
                    |
                  logits
                    |
                  Sampler
                    |
                next token
                    |
          +---------+---------+
          |                   |
     CLI output        HTTP SSE stream
```

## Project Structure

```text
src/
  model/
    gguf.rs            GGUF binary format parser
    config.rs          Model hyperparameters from metadata
    tokenizer.rs       BPE tokenizer (encode/decode)
    chat_template.rs   Chat template detection and rendering
  tensor/
    tensor.rs          Contiguous f32 tensor type
    ops.rs             Primitives: matmul, RMSNorm, RoPE, softmax, SiLU
    quantized.rs       Quantized tensor storage and block-wise matvec kernels
    simd.rs            AVX2+FMA SIMD kernels
    dequantize.rs      Scalar dequantization helpers
  transformer/
    weights.rs         Weight loading from GGUF
    forward.rs         Transformer forward pass and generation loops
  cache/               KV-cache for autoregressive generation
  sampling/            Temperature, top-k, top-p, repetition penalty
  server/              Axum server, OpenAI-compatible routes, SSE streaming
```

## Testing

The library test suite currently covers tokenizer behavior, GGUF parsing, quantization paths, tensor ops, sampling, KV-cache handling, and transformer forward logic.

```bash
cargo test --lib
```

Current status: `84` tests passing.

## License

MIT. See [LICENSE](LICENSE).
