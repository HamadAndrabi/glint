# Ferrite

A high-performance LLM inference engine built from scratch in Rust. Ferrite loads GGUF models, runs the full transformer forward pass, and serves an OpenAI-compatible HTTP API — all without PyTorch, ONNX, or any ML framework dependency.

## Features

**Inference**
- Full LLaMA-family transformer: RMSNorm, RoPE, SwiGLU, grouped-query attention
- KV-cache for O(n) per-token generation (no re-computing past tokens)
- Greedy and configurable sampling (temperature, top-k, top-p, repetition penalty, seeds)

**Quantization**
- 5 formats: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K — covers the vast majority of published GGUF models
- AVX2+FMA SIMD kernels for all formats (runtime detection, scalar fallback on non-x86)
- Rayon row-parallel matvec across all CPU cores
- Weights stay compressed in memory — ~140 MB for a 135M-parameter Q8_0 model vs ~540 MB dequantized

**Server**
- OpenAI-compatible HTTP API: `/v1/completions`, `/v1/chat/completions`, `/v1/models`
- Token-by-token SSE streaming
- Chat template auto-detection from GGUF metadata (ChatML, Llama 3, Mistral, Zephyr, Gemma)
- CORS enabled for browser clients, `/health` endpoint for orchestrators

**Model loading**
- Zero-copy GGUF parser with memory-mapped I/O
- Built-in BPE tokenizer loaded from GGUF vocabulary
- Supports any LLaMA-architecture model in GGUF format

## Getting Started

```bash
git clone https://github.com/HamadAndrabi/ferrite
cd ferrite
cargo build --release
```

### Generate text

```bash
ferrite run -f model.Q4_K_M.gguf -p "The future of AI is" -m 100
```

With sampling:

```bash
ferrite run -f model.gguf -p "Once upon a time" -m 200 \
  --temperature 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1
```

### Start the server

```bash
ferrite serve -f model.Q4_K_M.gguf -p 8080
```

Then use any OpenAI-compatible client:

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Streaming
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "prompt": "The meaning of life is",
    "max_tokens": 100,
    "stream": true
  }'

# Health check
curl http://localhost:8080/health
```

### Inspect a model

```bash
ferrite inspect -f model.gguf --show-metadata --show-tensors
```

## Architecture

```
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
               forward(token) ----+
                    |
              logits [vocab]
                    |
               Sampler
            (temp/top-k/top-p)
                    |
               next token
                    |
          +----+----+----+
          |              |
     CLI output    HTTP SSE stream
                   (OpenAI API)
```

## Benchmarks

Matvec throughput at 4096x4096 (Llama-3-8B scale), AVX2+FMA, rayon:

| Format | Throughput | Time |
|--------|-----------|------|
| Q4_0 | 24.4 Gelem/s | 687 us |
| Q8_0 | 22.7 Gelem/s | 739 us |
| Q4_K | 20.8 Gelem/s | 808 us |
| Q6_K | 18.7 Gelem/s | 899 us |
| Q5_K | 17.0 Gelem/s | 984 us |

Run benchmarks:

```bash
cargo bench --bench matvec
```

## Project Structure

```
src/
  model/
    gguf.rs            GGUF binary format parser
    config.rs          Model hyperparameters from metadata
    tokenizer.rs       BPE tokenizer (encode/decode)
    chat_template.rs   Chat template detection and rendering
  tensor/
    tensor.rs          Contiguous f32 tensor type
    ops.rs             Primitives: matmul, RMSNorm, RoPE, softmax, SiLU
    quantized.rs       QuantizedTensor with block-wise matvec kernels
    simd.rs            AVX2+FMA SIMD kernels (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K)
    dequantize.rs      Scalar dequantization for all formats
  transformer/
    weights.rs         Weight loading from GGUF into QuantizedTensors
    forward.rs         Full transformer forward pass + generation loops
  cache/               KV-cache for autoregressive generation
  sampling/            Temperature, top-k, top-p, repetition penalty
  server/              Axum HTTP server, OpenAI-compatible routes, SSE streaming
```

## Tests

```bash
cargo test --lib     # 84 tests covering all modules
```

## License

MIT. See [LICENSE](LICENSE).
