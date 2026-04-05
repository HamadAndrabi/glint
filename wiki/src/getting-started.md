# Installation & Quick Start

## Prerequisites

- Rust 1.75+ (`rustup update stable`)
- A GGUF model file (see [Downloading Models](#downloading-models))

---

## Build

```bash
# Default: CPU inference + HTTP server (rayon + server features)
cargo build --release

# GPU support (requires Vulkan-capable GPU)
cargo build --release --features vulkan

# Python bindings (requires maturin)
pip install maturin
maturin develop --features python

# Browser WASM (requires wasm-pack)
wasm-pack build --target web --no-default-features --features wasm
```

The `default` feature set includes `rayon` (parallel matmul) and `server` (HTTP API + HF Hub download). Omit `--release` for faster compilation during development.

---

## Downloading Models

Glint can pull models from the Hugging Face Hub:

```bash
# Pull SmolLM-135M (fast, good for testing)
./target/release/glint pull bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# Pull TinyLlama-1.1B (good for benchmarking)
./target/release/glint pull TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Pull Llama-3-8B (requires ~4–5 GB RAM for Q4_K_M)
./target/release/glint pull bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

Models are cached in `~/.cache/ferrite/models/` by default.

You can also point Glint directly at any `.gguf` file on disk.

---

## First Run

### Single-turn generation

```bash
./target/release/glint run \
  -f ~/.cache/ferrite/models/SmolLM2-135M-Instruct-Q8_0.gguf \
  -p "The first step to learning Rust is" \
  -m 100
```

### Interactive chat

```bash
./target/release/glint chat \
  -f ~/.cache/ferrite/models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --system "You are a helpful assistant."
```

### Start the HTTP server

```bash
./target/release/glint serve \
  -f ~/.cache/ferrite/models/SmolLM2-135M-Instruct-Q8_0.gguf \
  -p 8080
```

Then query it with curl:

```bash
# Text completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct-Q8_0",
    "prompt": "Rust is great because",
    "max_tokens": 100
  }'

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct-Q8_0",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'

# Streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct-Q8_0",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

---

## Inspect a Model

```bash
./target/release/glint inspect \
  -f model.gguf \
  --show-metadata \
  --show-tensors
```

Prints the GGUF header, model config (context length, layers, heads, vocab size), all metadata key-value pairs, and every tensor with its shape and quantization type.

---

## GPU Acceleration

```bash
# Build with Vulkan support
cargo build --release --features vulkan

# Run with GPU
./target/release/glint run \
  -f model.gguf \
  -p "Hello" \
  --gpu
```

The GPU backend uses `wgpu` and works on Vulkan, Metal, and DX12. Falls back to CPU with a warning if no compatible adapter is found.

---

## Run Tests

```bash
cargo test --lib         # all 108 unit tests
cargo clippy             # lints
cargo fmt --check        # format check
cargo bench --bench matvec  # performance micro-benchmarks
```

See [Contributing](./contributing.md) for guidance on what to run after different types of changes.
