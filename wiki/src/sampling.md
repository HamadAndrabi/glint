# Sampling

The sampling pipeline converts raw logits (one f32 per vocabulary token) into a single token choice. Glint implements a full pipeline of composable strategies that can be combined freely.

Source: `src/sampling/sampler.rs`

---

## Pipeline Overview

```
logits [vocab_size]
    │
    ├─ 1. Repetition penalty   (penalize already-seen tokens)
    ├─ 2. Temperature scaling  (control sharpness of distribution)
    ├─ 3. Top-k filtering      (keep only k best candidates)
    ├─ 4. Min-p filtering      (drop tokens below min_p × max_prob)
    ├─ 5. Softmax              (convert to probabilities)
    ├─ 6. Top-p (nucleus)      (trim tail of distribution)
    └─ 7. Weighted random sample → token_id
```

Each stage is a standalone function — independently testable and composable.

---

## Configuration

```rust
pub struct SamplerConfig {
    pub temperature:    f32,   // default 1.0 (no change). 0.0 = greedy.
    pub top_k:          usize, // default 0 (disabled)
    pub top_p:          f32,   // default 1.0 (disabled)
    pub min_p:          f32,   // default 0.0 (disabled)
    pub repeat_penalty: f32,   // default 1.0 (disabled)
    pub seed:           Option<u64>, // None = system time
}
```

---

## Greedy Decoding

When `temperature == 0.0`, the sampler short-circuits to argmax:

```rust
if self.config.temperature == 0.0 {
    return argmax(logits);
}
```

Greedy is deterministic and maximally confident — it always picks the single most likely token. Useful for debugging and embedding generation. Downside: can produce repetitive or degenerate text.

---

## Temperature

Divides all logits by the temperature before softmax:

```
logit'[i] = logit[i] / T
```

- **T = 1.0** — no change (default)
- **T < 1.0** — sharpens the distribution (more confident, more repetitive)
- **T > 1.0** — flattens the distribution (more random, more creative)
- **T → 0** — approaches greedy (argmax)

Temperature affects the relative differences between logits. A logit gap of 2.0 at T=0.5 becomes a gap of 4.0, which after softmax gives a much higher probability to the top token.

---

## Top-k

Retains only the k highest-logit tokens; sets the rest to `-inf` before softmax.

```rust
// Find k-th largest using partial select
indices.select_nth_unstable_by(k, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
for &idx in &indices[k..] {
    logits[idx] = f32::NEG_INFINITY;
}
```

**k = 0** disables top-k (keep all tokens). Common values: k = 40–50.

Top-k eliminates the "long tail" of very unlikely tokens that might be sampled by chance.

---

## Min-p

Filters tokens whose probability would fall below `min_p × max_probability`.

Implemented in log-space to avoid a full softmax pass:

```
prob(i) < min_p × max_prob
⟺ logit(i) < max_logit + ln(min_p)    [softmax is monotone]
```

```rust
let threshold = max_logit + min_p.ln();
for logit in logits.iter_mut() {
    if *logit < threshold { *logit = f32::NEG_INFINITY; }
}
```

Min-p is typically more adaptive than top-k: when the model is very confident (one token dominates), it trims more aggressively; when uncertain (many near-equal tokens), it keeps more candidates.

---

## Top-p (Nucleus Sampling)

Keeps the smallest set of tokens whose cumulative probability is ≥ p, and re-normalizes.

```
Sort tokens by probability (descending)
Accumulate until Σ probs ≥ p
Zero out everything below the cutoff
Re-normalize to sum to 1
```

Unlike top-k, the number of retained tokens adapts to the model's confidence. If the top token has probability 0.95, only one token might survive at p=0.9. If the distribution is flat, many tokens survive.

Top-p operates on **probabilities** (after softmax), not logits. In the pipeline, it runs after softmax (stage 6), while min-p runs on logits (stage 4).

---

## Repetition Penalty

Penalizes tokens that have already appeared in the generated sequence.

For each token in `past_tokens`:
- If its logit is **positive**: divide by `penalty` (reduces the logit)
- If its logit is **negative**: multiply by `penalty` (makes it more negative)

```rust
if logits[idx] > 0.0 {
    logits[idx] /= penalty;
} else {
    logits[idx] *= penalty;
}
```

This is the HuggingFace convention for repetition penalty. Values near 1.0 have little effect; values around 1.1–1.3 noticeably reduce repetition.

---

## PRNG

Glint uses a self-contained xorshift64 PRNG — no external `rand` crate:

```rust
struct Xorshift64 { state: u64 }

fn next_u64(&mut self) -> u64 {
    let mut s = self.state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    self.state = s;
    s
}

fn next_f32(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
}
```

Period of 2^64 - 1. When a `seed` is provided, generation is fully deterministic — the same seed always produces the same sequence of tokens.

---

## Recommended Combinations

| Use case | Settings |
|----------|---------|
| Greedy / deterministic | `temperature=0.0` |
| Creative writing | `temperature=0.8, top_p=0.9` |
| Factual Q&A | `temperature=0.3, top_k=40` |
| Code generation | `temperature=0.2, top_k=50, repeat_penalty=1.1` |
| Max diversity | `temperature=1.2, top_p=0.95, min_p=0.05` |
