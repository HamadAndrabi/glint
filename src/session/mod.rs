//! Session abstraction — first-class owner of per-sequence generation state.
//!
//! A [`Session`] bundles the KV cache, sampler, position counter, token
//! history, and generation budget for one active sequence.  This replaces the
//! ad-hoc `ActiveSequence` in the server engine and the scattered `KvCache`
//! ownership in `generate_cached` / Python / WASM paths.
//!
//! # Design
//!
//! `Session` is a pure library type: it has no knowledge of HTTP, channels, or
//! the inference engine.  Higher-level layers (`InferenceEngine`, `Model`,
//! Python / WASM bindings) own `Session`s and drive them via the forward-pass
//! functions in `crate::transformer`.

pub mod snapshot;

use std::sync::Arc;

use crate::cache::{KvCache, KvCacheQ8, KvStore};
use crate::constrained::{TokenConstraint, VocabIndex};
use crate::model::lora::LoraWeights;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tensor::Tensor;

// ── CacheFormat ───────────────────────────────────────────────────────────────

/// Which KV-cache storage format to allocate for a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CacheFormat {
    /// Full f32 precision — highest accuracy, most memory.
    #[default]
    F32,
    /// Q8_0 quantised — ~3.8× smaller, minimal accuracy loss.
    Q8,
}

// ── SessionOptions ────────────────────────────────────────────────────────────

/// Parameters used to create a new [`Session`].
pub struct SessionOptions {
    pub max_new_tokens: usize,
    pub sampler_cfg:    SamplerConfig,
    pub eos_token:      u32,
    pub cache_format:   CacheFormat,
    // KV cache dimensions — normally taken from ModelConfig.
    pub context_length: usize,
    pub n_layers:       usize,
    pub n_kv_heads:     usize,
    pub head_dim:       usize,
    /// Optional LoRA adapter to apply for this session.  Overrides the base
    /// model's built-in adapter (if any) when `Some`.
    pub lora_adapter:   Option<Arc<LoraWeights>>,
}

// ── Session ───────────────────────────────────────────────────────────────────

/// All mutable state for one in-flight generation sequence.
///
/// The forward-pass functions in `crate::transformer` borrow `cache` and
/// `sampler` directly; position tracking is the caller's responsibility (use
/// `session.pos` after each `advance` call).
pub struct Session {
    /// All tokens seen so far (prompt + generated).
    pub tokens:        Vec<u32>,
    /// Number of prompt/prefill tokens at the start of generation.
    ///
    /// This lets restored sessions rebuild structured-output constraints by
    /// replaying only the generated suffix.
    pub prefill_len:   usize,
    /// The KV cache.  Boxed trait object — `F32` and `Q8` are interchangeable.
    pub cache:         Box<dyn KvStore>,
    /// Which storage format the cache uses.
    pub cache_format:  CacheFormat,
    /// Sampler owns its RNG; seeded at session creation.
    pub sampler:       Sampler,
    /// Current decode position (`tokens.len() - 1` after prefill).
    pub pos:           usize,
    /// Logits from the most recent forward pass, ready to sample from.
    pub last_logits:   Tensor,
    /// EOS token id for this model.
    pub eos_token:     u32,
    /// Remaining token budget.
    pub max_remaining: usize,
    /// Optional token constraint (e.g. JSON mode).  Applied during sampling.
    pub constraint:    Option<Box<dyn TokenConstraint>>,
    /// Vocabulary index used by the constraint for mask lookups.
    /// Set alongside `constraint`; `None` when no constraint is active.
    pub vocab_index:   Option<Arc<VocabIndex>>,
    /// Per-session LoRA adapter.  When `Some`, overrides the base model's
    /// global adapter during forward passes.
    pub lora_adapter:  Option<Arc<LoraWeights>>,
}

impl Session {
    /// Allocate a new, empty session (no prefill yet).
    pub fn new(opts: SessionOptions) -> Self {
        let cache: Box<dyn KvStore> = match opts.cache_format {
            CacheFormat::F32 => Box::new(KvCache::new(
                opts.n_layers,
                opts.context_length,
                opts.n_kv_heads,
                opts.head_dim,
            )),
            CacheFormat::Q8 => Box::new(KvCacheQ8::new(
                opts.n_layers,
                opts.context_length,
                opts.n_kv_heads,
                opts.head_dim,
            )),
        };
        let sampler = Sampler::new(opts.sampler_cfg);
        Self {
            tokens:        Vec::new(),
            prefill_len:   0,
            cache,
            cache_format:  opts.cache_format,
            sampler,
            pos:           0,
            last_logits:   Tensor::zeros(&[1]), // placeholder until prefill
            eos_token:     opts.eos_token,
            max_remaining: opts.max_new_tokens,
            constraint:    None,
            vocab_index:   None,
            lora_adapter:  opts.lora_adapter,
        }
    }

    /// True if this session has finished (EOS hit or budget exhausted).
    pub fn is_finished(&self) -> bool {
        self.max_remaining == 0
            || self.tokens.last() == Some(&self.eos_token)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(fmt: CacheFormat) -> SessionOptions {
        SessionOptions {
            max_new_tokens: 10,
            sampler_cfg:    SamplerConfig { seed: Some(1), ..Default::default() },
            eos_token:      2,
            cache_format:   fmt,
            context_length: 64,
            n_layers:       2,
            n_kv_heads:     2,
            head_dim:       8,
            lora_adapter:   None,
        }
    }

    #[test]
    fn test_session_new_f32() {
        let s = Session::new(opts(CacheFormat::F32));
        assert!(s.tokens.is_empty());
        assert_eq!(s.eos_token, 2);
        assert_eq!(s.max_remaining, 10);
        assert!(!s.is_finished());
    }

    #[test]
    fn test_session_new_q8() {
        let s = Session::new(opts(CacheFormat::Q8));
        assert!(s.tokens.is_empty());
        assert!(!s.is_finished());
    }

    #[test]
    fn test_session_finished_budget() {
        let mut s = Session::new(opts(CacheFormat::F32));
        s.max_remaining = 0;
        assert!(s.is_finished());
    }

    #[test]
    fn test_session_finished_eos() {
        let mut s = Session::new(opts(CacheFormat::F32));
        s.tokens.push(2); // eos token
        assert!(s.is_finished());
    }
}
