//! Sampling strategies for token selection from logits.
//!
//! The sampling pipeline transforms raw logits into a token choice:
//!
//! ```text
//! logits [vocab_size]
//!   → repetition penalty (reduce already-seen tokens)
//!   → temperature scaling (control randomness)
//!   → top-k filtering (keep only k best candidates)
//!   → top-p / nucleus filtering (keep cumulative prob ≥ p)
//!   → min-p filtering (drop tokens below min_p × max_prob)
//!   → softmax → probabilities
//!   → weighted random sample → token_id
//! ```
//!
//! Each stage is a standalone function so it can be tested in isolation.
//! The `Sampler` struct orchestrates the full pipeline.

// ── Xorshift64 PRNG ─────────────────────────────────────────────────────────
//
// A minimal pseudo-random number generator. We avoid pulling in the `rand`
// crate for this single use case. Xorshift64 has a period of 2^64 - 1 and
// passes basic statistical tests — more than adequate for token sampling.
//
// The algorithm:
//   state ^= state << 13
//   state ^= state >> 7
//   state ^= state << 17
//
// To get a uniform f32 in [0, 1): take the upper 32 bits and divide by 2^32.

/// A simple 64-bit xorshift PRNG.
///
/// Produces a full period of 2^64 - 1 values (state must never be 0).
pub struct Xorshift64 {
    pub state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed. If seed is 0, uses 1 instead
    /// (the xorshift algorithm requires a non-zero state).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Restore a PRNG from a previously saved state (e.g. from a snapshot).
    pub fn restore(state: u64) -> Self {
        Self { state: if state == 0 { 1 } else { state } }
    }

    /// Generate the next random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Generate a uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use upper 24 bits (f32 has 23-bit mantissa + 1 implicit bit)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ── Sampler configuration ────────────────────────────────────────────────────

/// Configuration for the sampling pipeline.
///
/// All fields have defaults that disable their effect, so you can enable
/// only the strategies you want.
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// Temperature for logit scaling. 1.0 = no change, >1 = more random,
    /// <1 = more deterministic, 0.0 = greedy (argmax).
    pub temperature: f32,

    /// Top-k filtering. Keep only the k tokens with highest logits.
    /// 0 = disabled (keep all).
    pub top_k: usize,

    /// Top-p (nucleus) filtering. Keep the smallest set of tokens whose
    /// cumulative probability is ≥ p. 1.0 = disabled.
    pub top_p: f32,

    /// Min-p filtering. Drop any token whose probability is less than
    /// min_p × max_token_probability. 0.0 = disabled.
    pub min_p: f32,

    /// Repetition penalty. Tokens that appear in the context have their
    /// logits divided (if positive) or multiplied (if negative) by this
    /// value. 1.0 = disabled.
    pub repeat_penalty: f32,

    /// Optional seed for the PRNG. None = seed from system time.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: None,
        }
    }
}

// ── Sampler ──────────────────────────────────────────────────────────────────

/// Token sampler that owns its configuration and RNG state.
///
/// Usage:
/// ```ignore
/// let mut sampler = Sampler::new(SamplerConfig { temperature: 0.8, top_p: 0.9, ..Default::default() });
/// let token = sampler.sample(&logits, &past_tokens);
/// ```
pub struct Sampler {
    config: SamplerConfig,
    pub rng: Xorshift64,
}

impl Sampler {
    /// Create a sampler with the given configuration.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| {
            // Seed from system time nanos — good enough for non-cryptographic use
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        });
        Self {
            rng: Xorshift64::new(seed),
            config,
        }
    }

    /// Convenience constructor for greedy (argmax) decoding.
    pub fn greedy() -> Self {
        Self::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        })
    }

    /// Run the full sampling pipeline on a logit vector and return a token ID.
    ///
    /// `logits` is the raw output from the LM head — one f32 per vocab token.
    /// `past_tokens` is the sequence generated so far (for repetition penalty).
    /// `constraint` optionally masks disallowed tokens before sampling.
    pub fn sample(
        &mut self,
        logits:     &[f32],
        past_tokens: &[u32],
    ) -> u32 {
        self.sample_inner(logits, past_tokens, None)
    }

    /// Like [`sample`] but applies a token constraint mask before sampling.
    ///
    /// Tokens where `mask[i] == false` are set to `f32::NEG_INFINITY` so they
    /// are never sampled.  This happens after repetition penalty but before
    /// temperature scaling, so the constraint dominates.
    pub fn sample_constrained(
        &mut self,
        logits:      &[f32],
        past_tokens: &[u32],
        mask:        &[bool],
    ) -> u32 {
        self.sample_inner(logits, past_tokens, Some(mask))
    }

    fn sample_inner(
        &mut self,
        logits:      &[f32],
        past_tokens: &[u32],
        mask:        Option<&[bool]>,
    ) -> u32 {
        // Greedy fast path when no constraint is active.
        if self.config.temperature == 0.0 && mask.is_none() {
            return argmax(logits);
        }

        // Work on a mutable copy of the logits.
        let mut logits = logits.to_vec();

        // 1. Repetition penalty.
        if self.config.repeat_penalty != 1.0 {
            apply_repetition_penalty(&mut logits, past_tokens, self.config.repeat_penalty);
        }

        // 2. Constraint mask — applied before temperature so disallowed tokens
        //    have −∞ logit and are excluded from all downstream filtering.
        if let Some(m) = mask {
            for (l, &allowed) in logits.iter_mut().zip(m.iter()) {
                if !allowed { *l = f32::NEG_INFINITY; }
            }
        }

        // Greedy path: argmax after masking.
        if self.config.temperature == 0.0 {
            return argmax(&logits);
        }

        // 3. Temperature scaling.
        if self.config.temperature != 1.0 {
            apply_temperature(&mut logits, self.config.temperature);
        }

        // 4. Top-k filtering.
        if self.config.top_k > 0 {
            apply_top_k(&mut logits, self.config.top_k);
        }

        // 5. Min-p filtering (before top-p, since min-p works on raw logits).
        if self.config.min_p > 0.0 {
            apply_min_p(&mut logits, self.config.min_p);
        }

        // 6. Softmax → probabilities.
        let probs = softmax(&logits);

        // 7. Top-p (nucleus) filtering — operates on probabilities.
        let probs = if self.config.top_p < 1.0 {
            apply_top_p(&probs, self.config.top_p)
        } else {
            probs
        };

        // 8. Weighted random sample.
        sample_from_probs(&probs, &mut self.rng)
    }
}

// ── Pipeline stages ──────────────────────────────────────────────────────────

/// Argmax: return the index of the largest element.
///
/// Uses `total_cmp` for a NaN-safe total order — NaN values sort last,
/// so they are never selected as the maximum when finite values exist.
/// Returns 0 for an empty slice.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Repetition penalty: penalize tokens that have already appeared.
///
/// For each token in `past_tokens`:
/// - If its logit is positive, divide by `penalty`
/// - If its logit is negative, multiply by `penalty`
///
/// This makes repeated tokens less attractive regardless of whether they
/// have positive or negative logits. (HuggingFace convention.)
pub(crate) fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
    for &token in past_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Temperature scaling: divide all logits by the temperature.
///
/// - temp > 1.0 → flatter distribution (more random)
/// - temp < 1.0 → sharper distribution (more deterministic)
/// - temp = 1.0 → no change (caller should skip this)
pub(crate) fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}

/// Top-k filtering: keep only the `k` highest logits, set the rest to -inf.
///
/// k=0 means disabled (caller should skip this).
pub(crate) fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }

    // Find the k-th largest value using a partial sort.
    // We collect (index, value) pairs, sort by value descending, and mark
    // everything below rank k as -inf.
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap()
    });

    // Everything from index k onward (in the partitioned order) is below top-k
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Min-p filtering: drop tokens whose probability would be below `min_p × max_prob`.
///
/// We work in log-space to avoid computing a full softmax:
///   prob(i) < min_p × max_prob
///   ↔ logit(i) < max_logit + ln(min_p)     [since softmax is monotonic]
pub(crate) fn apply_min_p(logits: &mut [f32], min_p: f32) {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_logit + min_p.ln();

    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Softmax: convert logits to probabilities.
///
/// Uses the standard numerically stable formula: exp(x - max) / Σexp(x - max).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        // Edge case: all logits were -inf. Return uniform over all tokens.
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&v| v / sum).collect()
    }
}

/// Top-p (nucleus) filtering: keep the smallest set of tokens whose
/// cumulative probability is ≥ p, zero out the rest, re-normalize.
///
/// Operates on probabilities (after softmax), not logits.
pub(crate) fn apply_top_p(probs: &[f32], p: f32) -> Vec<f32> {
    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    // Accumulate until we pass the threshold
    let mut cumulative = 0.0f32;
    let mut cutoff = indices.len(); // keep all by default
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= p {
            cutoff = rank + 1; // keep this token and everything above
            break;
        }
    }

    // Build new probability vector: zero out everything below cutoff
    let mut filtered = vec![0.0f32; probs.len()];
    for &idx in &indices[..cutoff] {
        filtered[idx] = probs[idx];
    }

    // Re-normalize
    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        for v in filtered.iter_mut() {
            *v /= sum;
        }
    }
    filtered
}

/// Sample a token index from a probability distribution using the given RNG.
///
/// Walks through cumulative probabilities until the random value is exceeded.
pub(crate) fn sample_from_probs(probs: &[f32], rng: &mut Xorshift64) -> u32 {
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    // Floating point rounding — return last token
    (probs.len() - 1) as u32
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: check that f32 slices are approximately equal.
    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    // ── Xorshift64 tests ────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_different_seeds_differ() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(123);
        // Very unlikely that the first 10 values match
        let v1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut rng = Xorshift64::new(12345);
        for _ in 0..10000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "f32 out of range: {v}");
        }
    }

    #[test]
    fn test_xorshift_zero_seed_handled() {
        // Seed 0 should not cause the RNG to get stuck at 0
        let mut rng = Xorshift64::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    // ── Argmax tests ────────────────────────────────────────────────────────

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[-1.0, -3.0, -0.5]), 2);
    }

    // ── Temperature tests ───────────────────────────────────────────────────

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        // Dividing by 2: [1.0, 2.0, 3.0]
        approx_eq(&logits, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.1);
        // Dividing by 0.1 → [10.0, 20.0, 30.0]
        approx_eq(&logits, &[10.0, 20.0, 30.0], 1e-5);
    }

    // ── Top-k tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        apply_top_k(&mut logits, 2);
        // Top 2 are indices 1 (5.0) and 3 (4.0)
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_top_k(&mut logits, 10);
        // Nothing should be filtered
        approx_eq(&logits, &[1.0, 2.0, 3.0], 1e-6);
    }

    // ── Top-p tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_top_p_keeps_top_tokens() {
        // Probabilities: [0.1, 0.6, 0.3] — top-p=0.8 should keep indices 1 and 2
        let probs = vec![0.1, 0.6, 0.3];
        let filtered = apply_top_p(&probs, 0.8);
        // Index 1 (0.6) alone is < 0.8, so we need index 2 (0.3) too: 0.6+0.3 = 0.9 ≥ 0.8
        assert!(filtered[0] == 0.0, "index 0 should be filtered out");
        assert!(filtered[1] > 0.0, "index 1 should survive");
        assert!(filtered[2] > 0.0, "index 2 should survive");
        // Should be re-normalized
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_top_p_one_token_sufficient() {
        // If the top token alone exceeds p, only that token survives
        let probs = vec![0.1, 0.85, 0.05];
        let filtered = apply_top_p(&probs, 0.8);
        assert!(filtered[0] == 0.0);
        assert!((filtered[1] - 1.0).abs() < 1e-6); // re-normalized to 1.0
        assert!(filtered[2] == 0.0);
    }

    // ── Min-p tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_min_p_filters_low_logits() {
        // logits: [10.0, 5.0, 0.0, -5.0]
        // max = 10.0, threshold = 10.0 + ln(0.1) ≈ 10.0 - 2.302 = 7.698
        // Only index 0 (10.0) survives
        let mut logits = vec![10.0, 5.0, 0.0, -5.0];
        apply_min_p(&mut logits, 0.1);
        assert_eq!(logits[0], 10.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_zero_disabled() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        // min_p = 0.0 → ln(0) = -inf → threshold = -inf → nothing filtered
        // But we skip this call entirely in the sampler when min_p == 0
        // Still, the function should handle it gracefully
        apply_min_p(&mut logits, 0.0);
        // ln(0.0) = -inf, so threshold = max + (-inf) = -inf → nothing filtered
        approx_eq(&logits, &original, 1e-6);
    }

    // ── Repetition penalty tests ────────────────────────────────────────────

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -3.0, 1.0, 0.5];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        // Index 0: positive → divide by 2 → 1.0
        assert!((logits[0] - 1.0).abs() < 1e-6);
        // Index 1: negative → multiply by 2 → -6.0
        assert!((logits[1] - (-6.0)).abs() < 1e-6);
        // Index 2: not penalized → unchanged
        assert!((logits[2] - 1.0).abs() < 1e-6);
        // Index 3: not penalized → unchanged
        assert!((logits[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_disabled() {
        let mut logits = vec![2.0, -3.0, 1.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        approx_eq(&logits, &original, 1e-6);
    }

    // ── Sampler integration tests ───────────────────────────────────────────

    #[test]
    fn test_sampler_greedy() {
        let mut sampler = Sampler::greedy();
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let token = sampler.sample(&logits, &[]);
        assert_eq!(token, 1); // argmax
    }

    #[test]
    fn test_sampler_seeded_reproducibility() {
        let logits = vec![1.0, 2.0, 3.0, 2.5, 1.5];

        let mut sampler1 = Sampler::new(SamplerConfig {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        });
        let mut sampler2 = Sampler::new(SamplerConfig {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        });

        // Same seed → same sequence of samples
        for _ in 0..20 {
            let t1 = sampler1.sample(&logits, &[]);
            let t2 = sampler2.sample(&logits, &[]);
            assert_eq!(t1, t2);
        }
    }

    #[test]
    fn test_sampler_temperature_affects_distribution() {
        // With very low temperature, sampling should almost always pick the max
        let logits = vec![1.0, 10.0, 2.0];
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.01,
            seed: Some(42),
            ..Default::default()
        });

        let mut count_max = 0;
        for _ in 0..100 {
            if sampler.sample(&logits, &[]) == 1 {
                count_max += 1;
            }
        }
        // With temp=0.01, the max logit token should be picked nearly every time
        assert!(count_max >= 95, "expected ~100 picks of argmax, got {count_max}");
    }

    #[test]
    fn test_softmax_all_neginf() {
        // Edge case: all logits are -inf (e.g., everything was filtered)
        let logits = vec![f32::NEG_INFINITY; 4];
        let probs = softmax(&logits);
        // Should return uniform distribution
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }
}
