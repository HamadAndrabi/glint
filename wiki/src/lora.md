# LoRA Adapters

LoRA (Low-Rank Adaptation) allows fine-tuned behavior to be injected into a base model without modifying its weights. Glint supports loading LoRA adapters stored as GGUF files.

Source: `src/model/lora.rs`

Reference: [Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)

---

## How LoRA Works

A LoRA adapter replaces the weight update `ΔW` (which would be the same size as the original weight) with a low-rank decomposition:

```
ΔW = scale × B @ A

where:
  A: [rank, in_dim]   — the "down" projection
  B: [out_dim, rank]  — the "up" projection
  scale = alpha / rank
```

During inference, the adapter modifies the output of each projection:

```
output = W @ x + scale × B @ (A @ x)
       = (W + ΔW) @ x
```

The original weights `W` are unchanged; only the adapter matrices are loaded.

**Why low rank?** For a typical weight matrix of shape `[4096, 4096]`, full fine-tuning updates 16M parameters. A LoRA adapter with rank 16 uses only `4096×16 + 16×4096 = 131K` parameters — ~120× fewer.

---

## Adapter File Format

Adapter weights follow the standard GGUF tensor naming convention, with `.lora_a` and `.lora_b` suffixes:

```
blk.{i}.attn_q.weight.lora_a   shape: [rank, in_dim]
blk.{i}.attn_q.weight.lora_b   shape: [out_dim, rank]
blk.{i}.attn_k.weight.lora_a
blk.{i}.attn_k.weight.lora_b
...
```

The scaling factor is derived from GGUF metadata:
```
adapter.lora.alpha  →  scale = alpha / rank
```
If absent, `scale = 1.0` (equivalent to `alpha == rank`).

---

## Supported Projection Targets

Adapters can target any combination of these projections per layer:

| Projection | GGUF name suffix |
|-----------|-----------------|
| Query | `attn_q` |
| Key | `attn_k` |
| Value | `attn_v` |
| Attention output | `attn_output` |
| FFN gate | `ffn_gate` |
| FFN up | `ffn_up` |
| FFN down | `ffn_down` |

If an adapter file doesn't include a particular projection, that projection runs as normal (no adapter applied).

---

## Loading a LoRA Adapter

### CLI

```bash
glint run \
  -f base-model.gguf \
  --lora adapter.gguf \
  -p "Your prompt here"

glint chat \
  -f base-model.gguf \
  --lora adapter.gguf
```

### Code

```rust
let base_model = GgufModel::load("base.gguf")?;
let mut weights = TransformerWeights::load(&base_model, &config)?;
let adapter_model = GgufModel::load("adapter.gguf")?;
let weights = weights.with_lora(&adapter_model);  // returns new weights with adapter attached
```

`with_lora` calls `LoraWeights::load` internally. If the adapter file contains no `lora_a`/`lora_b` tensors, `with_lora` is a no-op.

---

## Apply Path

During the forward pass, after each projection matvec, the adapter is applied in-place:

```rust
// Normal projection
qt.matvec(x, out);

// Apply LoRA if present
if let Some(adapter) = &lora_layer.attn_q {
    adapter.apply(x, out);
}
```

The `LoraAdapter::apply` method:
```rust
pub fn apply(&self, x: &[f32], out: &mut [f32]) {
    // tmp = A @ x  →  [rank]
    let mut tmp = vec![0.0f32; rank];
    for r in 0..rank {
        tmp[r] = dot(&self.a.data()[r*in_dim..], x);
    }
    // out += scale * B @ tmp
    for o in 0..out_dim {
        out[o] += self.scale * dot(&self.b.data()[o*rank..], &tmp);
    }
}
```

---

## Memory and Performance

LoRA adapters add minimal overhead:
- **Memory:** adapter matrices are loaded as f32 tensors. For rank=16 on all 7 projections × 32 layers of a 7B model: `~130K × 32 × 7 × 4 bytes ≈ 116 MB`
- **Compute:** two small matmuls per projection per layer. At rank=16, each adds `in_dim × 16 + 16 × out_dim` multiplications — roughly 1% of the base matvec cost.

---

## Where to Find LoRA Adapters

Hugging Face Hub hosts thousands of LoRA adapters. To use them with Glint:
1. Find an adapter in GGUF format (or convert using `llama.cpp`'s `convert_lora_to_gguf.py`)
2. Load it with `--lora adapter.gguf`

Common use cases: instruct fine-tuning, coding specialization, language-specific adaptation, domain expertise (legal, medical, etc.).
