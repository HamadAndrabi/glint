# CLI Reference

The `glint` binary provides six subcommands for model inspection, text generation, interactive chat, serving, and model management.

Source: `src/main.rs`

---

## Global Usage

```
glint <SUBCOMMAND> [OPTIONS]
```

If a model file path is provided and the file does not exist, Glint will attempt to search Hugging Face Hub and offer to download a matching model.

---

## `inspect`

Print model metadata, architecture config, and tensor inventory.

```bash
glint inspect -f <MODEL> [--show-metadata] [--show-tensors]
```

| Flag | Description |
|------|-------------|
| `-f, --file <PATH>` | Path to `.gguf` model file |
| `--show-metadata` | Print all GGUF metadata key-value pairs |
| `--show-tensors` | Print every tensor name, shape, and quantization type |

**Output sections:**
- Header (GGUF version, tensor count, metadata entries, model name, architecture)
- Model configuration (context length, embedding dim, layers, heads, vocab size)
- Metadata (if `--show-metadata`)
- Tensor inventory (if `--show-tensors`)
- Summary (total parameters, total tensor data bytes, breakdown by quantization type)

**Example:**
```bash
glint inspect -f smollm.gguf --show-metadata
# GGUF version: 3
# Tensor count: 218
# ...
# Total parameters: 134,515,200 (134.5M)
# Total tensor data: 128.8 MB
```

---

## `run`

Generate text from a text prompt.

```bash
glint run -f <MODEL> -p <PROMPT> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --file <PATH>` | required | Model file |
| `-p, --prompt <TEXT>` | required | Input prompt |
| `-m, --max-tokens <N>` | 100 | Maximum tokens to generate |
| `--temperature <F>` | 0.0 | Sampling temperature (0 = greedy) |
| `--top-k <N>` | 0 | Top-k filtering (0 = disabled) |
| `--top-p <F>` | 1.0 | Top-p nucleus sampling |
| `--repeat-penalty <F>` | 1.0 | Repetition penalty |
| `--seed <N>` | random | RNG seed for reproducibility |
| `--draft-model <PATH>` | — | Enable speculative decoding with this draft model |
| `--lookahead <N>` | 4 | Draft steps per verification round |
| `--lora <PATH>` | — | LoRA adapter file |
| `--gpu` | false | Use GPU backend (requires `vulkan` feature) |

**Example:**
```bash
glint run -f llama-3-8b.gguf \
  -p "Write a haiku about Rust" \
  -m 50 \
  --temperature 0.7 \
  --top-p 0.9
```

---

## `chat`

Interactive multi-turn conversation. Streams tokens to stdout as they are generated.

```bash
glint chat -f <MODEL> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --file <PATH>` | required | Model file |
| `--system <TEXT>` | — | System prompt |
| `-m, --max-tokens <N>` | 256 | Max tokens per response |
| `--temperature <F>` | 0.8 | Sampling temperature |
| `--top-k <N>` | 0 | Top-k filtering |
| `--top-p <F>` | 1.0 | Top-p filtering |
| `--repeat-penalty <F>` | 1.0 | Repetition penalty |
| `--seed <N>` | random | RNG seed |
| `--lora <PATH>` | — | LoRA adapter |
| `--gpu` | false | GPU backend |

Type your message and press Enter. Press Ctrl+D (EOF) to exit.

When the context window fills up, the chat mode automatically summarizes old messages to free space.

**Example:**
```bash
glint chat -f mistral-7b.gguf \
  --system "You are an expert Rust programmer." \
  --temperature 0.5
```

---

## `serve`

Start an OpenAI-compatible HTTP inference server.

```bash
glint serve -f <MODEL> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --file <PATH>` | required | Model file |
| `-p, --port <N>` | 8080 | TCP port |
| `--host <ADDR>` | `127.0.0.1` | Bind address |
| `--gpu` | false | GPU backend |
| `--kv-cache <FMT>` | `f32` | KV storage: `f32`, `q8` (~3.8× smaller), or `paged` (f32 in on-demand 16-token pages shared by all requests) |
| `--prefix-cache` | false | Reuse KV pages of a shared prompt prefix instead of re-prefilling it per request; requires `--kv-cache paged` |

The model name is derived from the file stem (e.g. `smollm-135m-instruct.Q8_0` → `smollm-135m-instruct.Q8_0`). Use this name in API requests.

Runs a background inference engine with a request queue and continuous batching: every active request advances in a single shared forward pass per step, so each weight matrix streams from memory once per step instead of once per sequence, and new requests join mid-generation as slots free up. Batching is bit-identical to decoding each request alone — see [Continuous batching](./server-api.md#continuous-batching).

**Example:**
```bash
glint serve -f llama-3-8b-q4_k.gguf -p 8080 --host 0.0.0.0
```

See [HTTP Server API](./server-api.md) for endpoint documentation.

---

## `bench`

Run Glint's end-to-end inference benchmarks.

```bash
glint bench -f <MODEL> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --file <PATH>` | required | Model file |
| `--mode <NAME>` | `all` | `all`, `prefill`, `decode`, `concurrency`, or `cache-format` |
| `--prompt-len <N>` | `512` | Prompt tokens used for all benchmarks |
| `--decode-tokens <N>` | `128` | New tokens decoded in decode/concurrency benchmarks |
| `--max-concurrent <N>` | `8` | Maximum concurrent sessions in the concurrency benchmark |
| `--warmup <N>` | `3` | Warm-up rounds |
| `--iters <N>` | `10` | Timed measurement rounds |
| `--output <PATH>` | — | Write JSON results to a file |

Use this when you want end-to-end numbers for prompt prefill, decode throughput, concurrency, or f32-vs-Q8 KV cache tradeoffs. For matvec micro-benchmarks, use `cargo bench --bench matvec`.

---

## `generate`

Low-level generation from raw token IDs (mainly for debugging).

```bash
glint generate -f <MODEL> --tokens <ID,ID,...> [-m <N>]
```

| Flag | Description |
|------|-------------|
| `-f, --file <PATH>` | Model file |
| `--tokens <IDs>` | Comma-separated token IDs (e.g. `1,15043,29892`) |
| `-m, --max-tokens <N>` | Max new tokens (default 20) |

Uses greedy decoding. Prints all output token IDs. Useful for comparing raw model output against other implementations.

---

## `pull`

Download a model from Hugging Face Hub.

```bash
glint pull <REPO> <FILE> [--dir <PATH>]
```

| Argument | Description |
|----------|-------------|
| `<REPO>` | HuggingFace repo in `owner/name` format |
| `<FILE>` | Filename to download (e.g. `model-Q4_K_M.gguf`) |
| `--dir <PATH>` | Download directory (default: `~/.cache/ferrite/models/`) |

Shows a progress bar during download. Skips download if file already exists.

**Example:**
```bash
glint pull bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

---

## Auto-Download

When a model file is not found on disk, Glint automatically searches Hugging Face for matching models:

```bash
glint run -f SmolLM2-135M-Instruct-Q8_0.gguf -p "Hello"
# File not found: SmolLM2-135M-Instruct-Q8_0.gguf
# Searching HuggingFace for "SmolLM2-135M-Instruct"... found 3 match(es):
#   [1] bartowski/SmolLM2-135M-Instruct-GGUF
#   [2] ...
# Download which? [1-3/N]: 1
```
