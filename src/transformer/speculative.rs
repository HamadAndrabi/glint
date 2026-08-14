//! Speculative decoding (Leviathan et al. 2023).
//!
//! A small "draft" model generates `lookahead` tokens speculatively;
//! the large "target" model verifies all of them in one batched forward call.
//! If the target accepts k ≤ lookahead tokens, we get k+1 tokens for roughly
//! the cost of one target-model call — a throughput gain of up to `lookahead+1×`
//! when the draft model's distribution is close to the target's.
//!
//! ## Acceptance criterion (per token i)
//!
//! Let `p(x)` = target probability, `q(x)` = draft probability, `t` = draft token.
//! Accept token `t` with probability `min(1, p(t) / q(t))`.
//! If rejected, sample from the residual distribution `max(0, p(x) - q(x)) / Z`.
//! If all `k` accepted, sample one extra token from target logits.

use super::{forward_one, forward_prefill_all, TransformerWeights};
use crate::backend::GpuBackend;
use crate::cache::{KvCache, KvStore};
use crate::model::config::ModelConfig;
use crate::sampling::Xorshift64;
use crate::tensor::{softmax, Tensor};

/// Run speculative decoding with a draft model and a target model.
///
/// Both models must share the same tokeniser and vocabulary.
///
/// # Arguments
/// * `draft_weights` / `draft_config`   — small, fast draft model
/// * `target_weights` / `target_config` — large, accurate target model
/// * `prompt_tokens`                    — initial context (already tokenised)
/// * `max_new_tokens`                   — upper bound on new tokens generated
/// * `lookahead`                        — draft steps per verification round (default 4–6)
/// * `temperature`                      — sampling temperature (0 = greedy)
/// * `eos_token`                        — stop token ID
/// * `seed`                             — PRNG seed for reproducibility; `None` = random
///
/// # Returns
/// All tokens including the prompt.
pub fn speculative_decode(
    draft_weights: &TransformerWeights,
    draft_config: &ModelConfig,
    target_weights: &TransformerWeights,
    target_config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    lookahead: usize,
    temperature: f32,
    eos_token: u32,
    seed: Option<u64>,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let lookahead = lookahead.max(1);

    // Allocate separate KV caches for draft and target
    let mut draft_cache = KvCache::new(
        draft_config.block_count as usize,
        draft_config.context_length as usize,
        draft_config.head_count_kv as usize,
        draft_config.head_dim() as usize,
    );
    let mut target_cache = KvCache::new(
        target_config.block_count as usize,
        target_config.context_length as usize,
        target_config.head_count_kv as usize,
        target_config.head_dim() as usize,
    );

    // Prefill both caches with the prompt
    // Draft model always runs on CPU; target model uses GPU if available.
    forward_prefill_all(
        draft_weights,
        draft_config,
        prompt_tokens,
        &mut draft_cache,
        0,
        &mut None,
    );
    forward_prefill_all(
        target_weights,
        target_config,
        prompt_tokens,
        &mut target_cache,
        0,
        gpu,
    );

    let mut tokens = prompt_tokens.to_vec();
    let rng_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });
    let mut rng = Xorshift64::new(rng_seed);

    let mut generated = 0usize;
    while generated < max_new_tokens {
        let current_len = tokens.len();
        if current_len >= draft_config.context_length as usize
            || current_len >= target_config.context_length as usize
        {
            break;
        }

        // 1. Draft: generate `lookahead` tokens speculatively
        let k = lookahead.min(max_new_tokens - generated);
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_probs = Vec::with_capacity(k); // full prob vector for each draft step

        for step in 0..k {
            let draft_pos = current_len + step;
            let logits = forward_one(
                draft_weights,
                draft_config,
                *tokens.last().unwrap_or(&0),
                draft_pos - 1,
                &mut draft_cache,
                &mut None,
            );
            // Draft uses temperature sampling to get probabilities
            let probs = softmax_with_temp(logits.data(), temperature);
            let sampled = sample_from_probs(&probs, &mut rng);
            draft_tokens.push(sampled);
            draft_probs.push(probs); // store full distribution for residual correction
                                     // Tentatively add to token list so the next draft step sees context
            tokens.push(sampled);
        }

        // Roll back tokens to current_len — we verify before committing
        tokens.truncate(current_len);

        // Roll back draft cache to current_len (the draft already wrote the speculated tokens)
        draft_cache.truncate(current_len);

        // 2. Target: verify all k draft tokens in one batched call
        //    forward_prefill_all returns logits for each of the k positions
        target_cache.truncate(current_len);
        let target_logits = forward_prefill_all(
            target_weights,
            target_config,
            &draft_tokens,
            &mut target_cache,
            current_len,
            gpu,
        );

        // 3. Acceptance-rejection sampling
        let mut n_accepted = 0usize;
        let mut bonus_token: Option<u32> = None;

        for i in 0..k {
            let target_probs = softmax_with_temp(target_logits[i].data(), temperature);
            let q = draft_probs[i][draft_tokens[i] as usize]; // draft prob at sampled token
            let p = target_probs[draft_tokens[i] as usize]; // target prob at draft choice

            let accept_prob = (p / (q + 1e-10)).min(1.0);
            if rng.next_f32() < accept_prob {
                // Accept this draft token
                tokens.push(draft_tokens[i]);
                generated += 1;
                n_accepted += 1;
                if draft_tokens[i] == eos_token || generated >= max_new_tokens {
                    break;
                }
            } else {
                // Reject: sample from true residual distribution max(0, p(x) - q(x)) / Z
                let residual = residual_distribution(&target_probs, &draft_probs[i]);
                let correction = sample_from_probs(&residual, &mut rng);
                tokens.push(correction);
                generated += 1;
                break;
            }
        }

        // If all k tokens accepted, sample one bonus token from target
        if n_accepted == k && generated < max_new_tokens {
            let last_target = &target_logits[k - 1];
            let probs = softmax_with_temp(last_target.data(), temperature);
            bonus_token = Some(sample_from_probs(&probs, &mut rng));
        }

        if let Some(bonus) = bonus_token {
            tokens.push(bonus);
            generated += 1;
            if bonus == eos_token {
                break;
            }
        }

        // Sync both caches to accepted token count
        let new_len = tokens.len();
        draft_cache.truncate(current_len);
        target_cache.truncate(current_len);
        // Re-prefill both caches with the newly accepted tokens
        let new_tokens = &tokens[current_len..new_len];
        if !new_tokens.is_empty() {
            forward_prefill_all(
                draft_weights,
                draft_config,
                new_tokens,
                &mut draft_cache,
                current_len,
                &mut None,
            );
            forward_prefill_all(
                target_weights,
                target_config,
                new_tokens,
                &mut target_cache,
                current_len,
                gpu,
            );
        }

        if tokens.last() == Some(&eos_token) {
            break;
        }
    }

    tokens
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 {
        // Greedy: one-hot on argmax
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut out = vec![0.0f32; logits.len()];
        out[max_idx] = 1.0;
        return out;
    }
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    let t = Tensor::from_vec(scaled, &[logits.len()]);
    softmax(&t).data().to_vec()
}

/// Compute the normalised residual `max(0, p - q) / Z`.
fn residual_distribution(p: &[f32], q: &[f32]) -> Vec<f32> {
    let mut residual: Vec<f32> = p
        .iter()
        .zip(q)
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();
    let z: f32 = residual.iter().sum();
    if z > 0.0 {
        for r in &mut residual {
            *r /= z;
        }
    } else {
        // Fall back to uniform if residual is empty
        let n = residual.len() as f32;
        for r in &mut residual {
            *r = 1.0 / n;
        }
    }
    residual
}

fn sample_from_probs(probs: &[f32], rng: &mut Xorshift64) -> u32 {
    let u = rng.next_f32();
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if u < cdf {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_distribution_sums_to_one() {
        let p = vec![0.6, 0.3, 0.1];
        let q = vec![0.4, 0.4, 0.2];
        let r = residual_distribution(&p, &q);
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        // p[0]-q[0]=0.2 > 0; p[1]-q[1]=-0.1 → clipped to 0; p[2]-q[2]=-0.1 → 0
        assert!(r[0] > 0.0);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 0.0);
    }

    #[test]
    fn test_residual_distribution_fallback_uniform() {
        // When p == q exactly, residual is all-zero → should fall back to uniform
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let r = residual_distribution(&p, &q);
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_with_temp_greedy() {
        let logits = vec![1.0f32, 3.0, 2.0];
        let probs = softmax_with_temp(&logits, 0.0);
        assert_eq!(probs[1], 1.0); // argmax at index 1
        assert_eq!(probs[0] + probs[2], 0.0);
    }

    #[test]
    fn test_softmax_with_temp_sums_to_one() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32 * 0.5 - 2.0).collect();
        let probs = softmax_with_temp(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn test_sample_from_probs_deterministic() {
        // With a one-hot distribution, always picks the hot index
        let mut probs = vec![0.0f32; 5];
        probs[3] = 1.0;
        let mut rng = Xorshift64::new(99);
        for _ in 0..10 {
            assert_eq!(sample_from_probs(&probs, &mut rng), 3);
        }
    }

    #[test]
    fn test_xorshift64_f32_range() {
        // Verify Xorshift64::next_f32 produces values in [0, 1) — the old
        // random_f32 was broken and produced ~1e-19. This confirms the fix.
        let mut rng = Xorshift64::new(12345);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "v={v} out of [0,1)");
        }
    }

    #[test]
    fn test_softmax_with_temp_seeded_determinism() {
        // sample_from_probs with the same Xorshift64 seed yields the same sequence
        let probs = vec![0.1f32, 0.5, 0.2, 0.2];
        let mut rng_a = Xorshift64::new(42);
        let mut rng_b = Xorshift64::new(42);
        for _ in 0..20 {
            assert_eq!(
                sample_from_probs(&probs, &mut rng_a),
                sample_from_probs(&probs, &mut rng_b),
            );
        }
    }

    #[test]
    fn test_residual_uses_draft_probs_not_target() {
        // When q >> p for token 0, after rejection the correction token
        // must be drawn from max(0, p - q) / Z, not from target alone.
        // Specifically, if p[0]=0.1, q[0]=0.9 and p[1]=0.9, q[1]=0.1,
        // then the residual has all mass on token 1 (since p[0]-q[0] < 0).
        let p = vec![0.1f32, 0.9];
        let q = vec![0.9f32, 0.1];
        let r = residual_distribution(&p, &q);
        // residual[0] = max(0, 0.1-0.9) = 0; residual[1] = 0.9-0.1 = 0.8 → normalised to 1.0
        assert!(r[0] < 1e-6, "r[0]={}", r[0]);
        assert!((r[1] - 1.0).abs() < 1e-5, "r[1]={}", r[1]);
        // sample_from_probs must always yield token 1 from this residual
        let mut rng = Xorshift64::new(7777);
        for _ in 0..50 {
            assert_eq!(sample_from_probs(&r, &mut rng), 1);
        }
    }
}
