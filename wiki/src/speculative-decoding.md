# Speculative Decoding

Speculative decoding accelerates generation by using a small, fast "draft" model to propose tokens in advance, then verifying them in a single forward pass of the large "target" model.

Source: `src/transformer/speculative.rs`

Reference: [Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)

---

## Core Idea

Autoregressive generation is inherently sequential: each token requires one full forward pass. Speculative decoding breaks this bottleneck:

1. **Draft phase:** run a small model (e.g., SmolLM-135M) to generate `k` tokens speculatively — fast because the model is small.
2. **Verify phase:** run the large target model (e.g., LLaMA-3-8B) on all `k` draft tokens in one batched forward call. This is roughly the cost of a single sequential target step.
3. **Accept/reject:** keep as many draft tokens as the target model agrees with; resample from the target distribution for any rejected token.

If the draft model's token choices agree with the target ~80% of the time and `k = 4`, you get roughly 3–4 tokens per target model call instead of 1 — a 3–4× speedup.

---

## Acceptance Criterion

For each draft token `t` at position `i`:

```
p(t) = target probability at token t
q(t) = draft probability at token t

Accept with probability min(1, p(t) / q(t))
```

If rejected, sample from the **residual distribution**:
```
residual(x) = max(0, p(x) - q(x)) / Z
```
where `Z` normalizes to a valid probability distribution.

This procedure guarantees that the **output distribution is identical to the target model's** — speculative decoding changes speed, not quality.

If all `k` draft tokens are accepted, one bonus token is sampled from the target model's logits at the last position.

---

## API

```rust
pub fn speculative_decode(
    draft_weights: &TransformerWeights,
    draft_config: &ModelConfig,
    target_weights: &TransformerWeights,
    target_config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    lookahead: usize,        // k: draft tokens per round (typically 4–6)
    temperature: f32,
    eos_token: u32,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32>
```

Both models must share the same tokenizer and vocabulary. The draft model always runs on CPU; the target model uses GPU if available.

---

## CLI Usage

```bash
glint run \
  -f llama-3-8b-q4_k.gguf \
  --draft-model smollm-135m.gguf \
  --lookahead 4 \
  -p "The history of Rust programming language is" \
  -m 200
```

---

## Implementation Details

### Separate KV Caches

Each model has its own `KvCache`. During prefill, both caches are populated with the prompt. During generation, the caches are updated independently.

### Cache Rollback

After generating `k` draft tokens, the draft cache is advanced `k` positions ahead of the target. After verification, only the accepted tokens are committed. The caches of both models are rolled back to the current accepted length via `truncate()`, then re-prefilled with the accepted tokens:

```rust
draft_cache.truncate(current_len);
target_cache.truncate(current_len);
forward_prefill_all(draft_weights, draft_config, accepted_tokens, &mut draft_cache, ...);
forward_prefill_all(target_weights, target_config, accepted_tokens, &mut target_cache, ...);
```

### Batched Verification

The verify step calls `forward_prefill_all` on the `k` draft tokens, which returns a logit tensor for each position. This is what makes the verification a single call rather than `k` calls — the target model processes all draft tokens in one batched prefill.

---

## Choosing a Draft Model

Good draft model properties:
- **Same tokenizer.** Draft and target must use the same vocabulary and BPE merges.
- **Much smaller.** The draft model's speed advantage is the source of the speedup. A 2× faster draft model with 50% acceptance rate barely breaks even.
- **High agreement rate.** The acceptance rate `E[min(1, p/q)]` determines the average accepted tokens per round. Aim for >70%.

Common pairings:
| Draft | Target | Expected speedup |
|-------|--------|-----------------|
| SmolLM-135M | SmolLM-1.7B | 2–3× |
| SmolLM-135M | LLaMA-3-8B | 1.5–2× (lower agreement rate) |
| TinyLlama-1.1B | LLaMA-3-8B | 2–3× |

---

## Limitations

- Requires two models loaded in memory simultaneously
- Draft model must share vocabulary with target
- Speedup is context-dependent (varies with topic, temperature, and acceptance rate)
- Not currently integrated with the HTTP server (CLI-only)
