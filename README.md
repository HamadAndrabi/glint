# Ferrite

Ferrite is a high-performance, production-grade LLM inference engine built from scratch in Rust. It is designed to be a lightweight, dependency-minimal alternative for running large language models with a focus on memory efficiency and architectural clarity.

## Overview

Ferrite implements the full transformer inference stack, from raw GGUF parsing to token generation. By leveraging Rust's memory safety and zero-cost abstractions, Ferrite provides a robust foundation for local LLM deployment without the overhead of heavy machine learning frameworks.

## Key Features

- **Zero-Copy Loading**: Utilizes memory-mapped file I/O (`mmap`) to map GGUF models directly into virtual memory, enabling near-instant startup and efficient memory usage.
- **Block Quantization**: Native support for quantized formats including Q8_0 and Q4_0, allowing large models to run on consumer hardware with minimal precision loss.
- **Optimized Tensor Core**: A custom tensor implementation featuring row-major contiguous storage and optimized primitives for matrix-vector operations.
- **LLaMA Architecture**: Full support for modern transformer architectures, including RMSNorm, RoPE (Rotary Positional Embeddings), and SwiGLU activation.
- **Integrated Tokenizer**: Built-in GGUF-based tokenizer support for seamless end-to-end text generation.
- **Type-Safe Inference**: Leverages Rust's type system to ensure architectural integrity and prevent common runtime errors in the inference pipeline.

## Technical Architecture

### GGUF Parser
A robust parser for the GGUF (GPT-Generated Unified Format) binary format. It handles hierarchical metadata, tensor descriptors, and aligned data sections with strict validation.

### Tensor Operations
The engine includes a specialized suite of linear algebra primitives:
- **MatVec**: Optimized matrix-vector multiplication for single-batch inference.
- **RMSNorm**: Root Mean Square Layer Normalization for improved stability.
- **RoPE**: Frequency-based rotary positional embeddings for long-context support.
- **Softmax**: Numerically stable softmax implementation for probability distribution.

### KV Caching
Implements an efficient Key-Value (KV) cache to store past activations, significantly accelerating autoregressive generation by avoiding redundant computations.

## Getting Started

### Installation

Ensure you have the Rust toolchain installed. Clone the repository and build in release mode for maximum performance:

```bash
git clone https://github.com/your-username/ferrite
cd ferrite
cargo build --release
```

The binary will be available at `./target/release/ferrite`.

### Usage

#### Inspect a Model
View metadata, architecture details, and tensor information of a GGUF file:

```bash
./target/release/ferrite inspect --file path/to/model.gguf
```

#### Run Inference
Generate text from a prompt using a GGUF model:

```bash
./target/release/ferrite run --file path/to/model.gguf --prompt "The future of AI is" --max-tokens 100
```

## Performance

Ferrite is engineered for efficiency:
- **Memory Efficiency**: Only required model weights are paged into RAM by the OS.
- **CPU Optimization**: Core loops are designed for cache-friendliness and future SIMD acceleration.
- **Minimal Dependencies**: Keeps the binary small and the security surface narrow.

## Roadmap

- [ ] SIMD Acceleration (AVX2/NEON) for tensor primitives.
- [ ] Multithreaded matrix operations.
- [ ] Support for additional quantization formats (Q4_K, Q5_K).
- [ ] GPU Backend support (WGPU/CUDA).
- [ ] Python bindings for easier integration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
