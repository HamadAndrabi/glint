# SafeTensors & HF Models

Glint loads HuggingFace-format models directly — a directory holding
`config.json` + `tokenizer.json` + `*.safetensors` (sharded or not) — with no
GGUF conversion step. GGUF remains the default format; the loader is picked by
path in `Model::load` and the CLI.

Source: `src/model/safetensors.rs`, `src/model/config.rs`,
`src/model/tokenizer.rs`, `src/transformer/weights.rs`

```bash
glint run -f ./SmolLM2-135M-Instruct -p "Hello" -m 100
glint serve -f ./SmolLM2-135M-Instruct -p 8080
```

Pointing at one `.safetensors` file inside a model directory works too — the
sibling `config.json`/`tokenizer.json` (and any other shards) are found
automatically.

---

## Format

A `.safetensors` file is an 8-byte little-endian header length, a JSON header
mapping tensor names to `{dtype, shape, data_offsets}`, then the raw tensor
data. Glint parses it by hand (no `safetensors` crate) and memory-maps the data
region exactly like the GGUF path.

The parser treats the file as untrusted input, with the same discipline as the
GGUF parser: the header length is bounded against the file size before any
allocation, offsets are validated in-bounds and non-overlapping, and shapes
must match their dtype's byte count. It has its own fuzz target
(`fuzz/fuzz_targets/safetensors_parse.rs`), smoke-run in CI alongside the GGUF
and snapshot fuzzers.

Multi-shard checkpoints (`model-00001-of-000NN.safetensors` +
`model.safetensors.index.json`) are supported; passing any shard opens the
whole set.

---

## What Maps, and How

| HF | Glint |
|----|-------|
| `config.json` | `ModelConfig` (hidden size, heads, KV heads, layers, RMS eps, RoPE theta, …) |
| `tokenizer.json` | The same BPE structures the GGUF vocabulary builds — see [Tokenization](./tokenization.md) |
| `model.layers.N.*` weights | `TransformerWeights` / `LayerWeights` |
| `lm_head` absent (`tie_word_embeddings`) | Embedding matrix reused as output head |

Two layout facts do the heavy lifting:

- **No transpose.** HF stores linear weights row-major as
  `[out_features, in_features]`, which is already Glint's convention. (GGUF
  stores column-major, which is why the GGUF path reverses dims — see
  `src/transformer/weights.rs`.)
- **Q/K rows are permuted.** HF's RoPE pairs dimension `j` with
  `j + head_dim/2` (`rotate_half`), while Glint's `ops::rope` rotates adjacent
  pairs. `permute_qk_rows` applies llama.cpp's `permute()` to the Q/K
  projection rows at load, verified both analytically and against a GGUF load
  of identical weights.

F32, F16, and BF16 weights are supported; F16/BF16 stay in their on-disk width
in memory and are widened per row inside the matvec fallback.

---

## Scope and Limits

Anything the loader cannot faithfully express is **rejected with a specific
error** rather than loaded into a silently wrong model:

- Architectures: `llama`, `mistral`, `phi3`, `qwen2` by name — and in practice
  the llama/mistral family, since fused `qkv_proj`/`gate_up_proj` (Phi-3),
  attention biases (Qwen2), and `q_norm`/`k_norm` are refused explicitly.
- Tokenizers: byte-level BPE only (LLaMA-3 / SmolLM / Qwen / GPT-2 style).
  SentencePiece-derived `tokenizer.json` (LLaMA-2, Mistral-v0.1) is rejected.
- RoPE scaling: `linear` only; `llama3`/`yarn`/`dynamic` error out.
- LoRA adapters remain GGUF-only.
