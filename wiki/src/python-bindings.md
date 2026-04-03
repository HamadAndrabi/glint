# Python Bindings

Glint exposes a `GlintLLM` Python class via PyO3, built as a native extension module with maturin.

Source: `src/python.rs`

Build with: `cargo build --features python` or `maturin develop --features python`

---

## Installation

```bash
# Install maturin
pip install maturin

# Build and install into current Python environment
maturin develop --features python

# Or build a wheel
maturin build --release --features python
pip install target/wheels/glint-*.whl
```

---

## API

### `GlintLLM`

```python
import glint

llm = glint.GlintLLM("path/to/model.gguf")
```

Loads the GGUF model synchronously on construction. Weights are kept in RAM. The model object is not thread-safe (call from one thread at a time).

#### `generate(prompt, **kwargs) -> str`

Generate text continuing `prompt`. Returns only the newly generated tokens (not the prompt itself).

```python
text = llm.generate(
    "The capital of France is",
    max_tokens=100,       # default 256
    temperature=0.7,      # default 0.0 (greedy)
    top_k=40,             # default 0 (disabled)
    top_p=0.9,            # default 1.0 (disabled)
    repeat_penalty=1.1,   # default 1.0 (disabled)
    seed=42,              # default None (random)
)
print(text)
# " Paris, the City of Light."
```

#### `model_info() -> dict`

Returns a dictionary of model hyperparameters:

```python
info = llm.model_info()
# {
#   "architecture": "llama",
#   "context_length": 4096,
#   "embedding_length": 2048,
#   "block_count": 22,
#   "head_count": 32,
#   "head_count_kv": 4,
#   "vocab_size": 32000,
#   "head_dim": 64,
# }
```

---

## Full Example

```python
import glint

# Load model
llm = glint.GlintLLM("smollm-135m.gguf")

# Print architecture info
info = llm.model_info()
print(f"Model: {info['architecture']}, {info['block_count']} layers")
print(f"Context: {info['context_length']} tokens, Vocab: {info['vocab_size']}")

# Greedy generation
print(llm.generate("Once upon a time", max_tokens=50))

# Creative generation with sampling
print(llm.generate(
    "Write a short poem about",
    max_tokens=100,
    temperature=0.8,
    top_p=0.9,
))

# Reproducible generation
output1 = llm.generate("The answer is", seed=123)
output2 = llm.generate("The answer is", seed=123)
assert output1 == output2  # same seed → same output
```

---

## Error Handling

Errors during model loading or generation raise `ValueError`:

```python
try:
    llm = glint.GlintLLM("nonexistent.gguf")
except ValueError as e:
    print(f"Failed to load model: {e}")
```

---

## Limitations

- Synchronous only (blocks the calling thread during generation)
- No streaming callback (all tokens returned at once)
- One model per `GlintLLM` instance; creating multiple instances loads multiple copies into RAM
- GPU backend not exposed through Python bindings (CPU inference only)

For streaming or async usage, consider using the [HTTP Server API](./server-api.md) instead.

---

## Building for Distribution

```bash
# Build a manylinux wheel (for PyPI upload)
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build \
  --release --features python -i python3.11

# Build macOS universal wheel
maturin build --release --features python --target universal2-apple-darwin
```

The resulting `.whl` file includes the Rust inference engine statically linked — no separate Rust installation needed by end users.
