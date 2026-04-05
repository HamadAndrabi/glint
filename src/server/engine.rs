//! Round-robin inference engine.
//!
//! `InferenceEngine` runs a single dedicated OS thread that owns the model
//! weights and (optionally) the GPU backend. Incoming requests are queued and
//! processed in a tight loop:
//!
//! 1. **Prefill** — for each newly arrived request, run the full prompt
//!    through the transformer and write K/V into a fresh per-sequence cache.
//! 2. **Round-robin decode** — advance every active sequence by one token
//!    (each with its own `forward_one` call), sample, and push the token ID
//!    down the per-request channel. This is fair interleaving, not true batched
//!    inference — each sequence gets a separate forward pass.
//! 3. **Eviction** — remove sequences that have hit EOS, exhausted their
//!    token budget, or whose client disconnected.
//!
//! The engine blocks (parks the thread) only when there are no active
//! sequences and no pending requests — i.e. when the server is completely idle.
//!
//! # Concurrency model
//!
//! Routes send `InferenceRequest`s through a `tokio::sync::mpsc::UnboundedSender`.
//! The engine thread calls `blocking_recv()` (safe from a non-async thread).
//! Each request carries its own `tokio::sync::mpsc::Sender<u32>` for token
//! delivery; the engine thread uses `blocking_send()` on it.
//!
//! Compared with the old `spawn_blocking`-per-request approach:
//! * Request 2 no longer waits for all 200 tokens of request 1 to finish.
//! * TTFT (time-to-first-token) for queued requests drops from O(full_request)
//!   to O(prefill + one_decode_step).
//! * The GPU is never idle between requests; it moves to the next sequence
//!   the moment the previous one finishes its decode step.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::backend::GpuBackend;
use crate::constrained::{build_constraint, ConstraintSpec, VocabIndex};
use crate::model::config::ModelConfig;
use crate::model::lora_registry::AdapterRegistry;
use crate::sampling::SamplerConfig;
use crate::session::{CacheFormat, Session, SessionOptions};
use crate::transformer::{forward_one_lora, forward_prefill_lora, TransformerWeights};

// ── Types ─────────────────────────────────────────────────────────────────────

/// A request submitted to the engine by a route handler.
pub struct InferenceRequest {
    pub prompt_tokens:  Vec<u32>,
    pub max_new_tokens: usize,
    pub sampler_cfg:    SamplerConfig,
    pub eos_token:      u32,
    /// Optional structured-output constraint (e.g. JSON object mode).
    pub constraint:     Option<ConstraintSpec>,
    /// Optional LoRA adapter name to apply for this request.
    /// The engine resolves the name against its `AdapterRegistry` at prefill time.
    pub lora_name:      Option<String>,
    /// Tokens are delivered here; dropping the receiver signals client disconnect.
    pub tx: mpsc::Sender<u32>,
}

/// An in-flight sequence being decoded by the engine.
struct ActiveSequence {
    session: Session,
    tx:      mpsc::Sender<u32>,
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

/// Handle to the background inference thread.
///
/// Clone-able; all clones share the same underlying channel.
pub struct InferenceEngine {
    tx: mpsc::UnboundedSender<InferenceRequest>,
}

impl InferenceEngine {
    /// Spawn the inference thread and return a handle.
    pub fn start(
        weights:      Arc<TransformerWeights>,
        config:       Arc<ModelConfig>,
        gpu:          Option<GpuBackend>,
        cache_format: CacheFormat,
        vocab_index:  Arc<VocabIndex>,
        registry:     Arc<AdapterRegistry>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("glint-inference".into())
            .spawn(move || engine_loop(rx, weights, config, gpu, cache_format, vocab_index, registry))
            .expect("failed to spawn inference thread");
        Self { tx }
    }

    /// Enqueue a generation request.
    ///
    /// Returns the token receiver immediately; the caller can await tokens
    /// as they arrive.  Returns `None` if the engine thread has shut down.
    pub fn submit(
        &self,
        prompt_tokens:  Vec<u32>,
        max_new_tokens: usize,
        sampler_cfg:    SamplerConfig,
        eos_token:      u32,
        constraint:     Option<ConstraintSpec>,
        lora_name:      Option<String>,
    ) -> Option<mpsc::Receiver<u32>> {
        let (token_tx, token_rx) = mpsc::channel(64);
        self.tx.send(InferenceRequest {
            prompt_tokens,
            max_new_tokens,
            sampler_cfg,
            eos_token,
            constraint,
            lora_name,
            tx: token_tx,
        }).ok()?;
        Some(token_rx)
    }
}

// ── Engine loop ───────────────────────────────────────────────────────────────

fn engine_loop(
    mut rx:       mpsc::UnboundedReceiver<InferenceRequest>,
    weights:      Arc<TransformerWeights>,
    config:       Arc<ModelConfig>,
    mut gpu:      Option<GpuBackend>,
    cache_format: CacheFormat,
    vocab_index:  Arc<VocabIndex>,
    registry:     Arc<AdapterRegistry>,
) {
    let mut active: Vec<ActiveSequence> = Vec::new();

    loop {
        // ── Drain any pending requests (prefill each one) ─────────────────
        loop {
            match rx.try_recv() {
                Ok(req) => prefill_and_add(&mut active, req, &weights, &config, &mut gpu, cache_format, &vocab_index, &registry),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        if active.is_empty() {
            // Nothing to decode — block until a request arrives.
            match rx.blocking_recv() {
                Some(req) => prefill_and_add(&mut active, req, &weights, &config, &mut gpu, cache_format, &vocab_index, &registry),
                None => return, // all senders dropped; shut down
            }
            // Loop back to drain any additional requests that arrived while
            // we were blocked (so we prefill them before the first decode step).
            continue;
        }

        // ── Sample from last_logits, then batch-decode all active sequences ────

        // 1. Sample next token for every active sequence.
        let mut finished: Vec<usize> = Vec::new();
        for (i, seq) in active.iter_mut().enumerate() {
            let s = &mut seq.session;
            if s.max_remaining == 0 {
                finished.push(i);
                continue;
            }
            s.max_remaining -= 1;
            let next = if let Some(constraint) = s.constraint.as_mut() {
                let vi   = s.vocab_index.as_ref().unwrap();
                let mask = constraint.allowed_tokens(&s.tokens, vi);
                let tok  = s.sampler.sample_constrained(s.last_logits.data(), &s.tokens, &mask);
                constraint.advance(tok);
                tok
            } else {
                s.sampler.sample(s.last_logits.data(), &s.tokens)
            };
            s.tokens.push(next);

            let eos_hit      = next == s.eos_token;
            let disconnected = seq.tx.blocking_send(next).is_err();
            if eos_hit || disconnected { finished.push(i); }
        }

        // Remove finished sequences (reverse order so indices stay valid).
        for i in finished.into_iter().rev() {
            active.swap_remove(i);
        }

        if active.is_empty() { continue; }

        // 2. Advance all active sequences by one decode step.
        //
        // GPU path: sequential (only one GPU context).
        // CPU path: rayon par_iter_mut — N sequences in parallel, each calling
        //           forward_one against its own independent KV cache.
        if gpu.is_some() {
            for seq in active.iter_mut() {
                let s = &mut seq.session;
                let tok = *s.tokens.last().unwrap();
                let pos = s.tokens.len() - 1;
                let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
                let lora = s.lora_adapter.as_deref();
                s.last_logits = forward_one_lora(&weights, &config, tok, pos, s.cache.as_mut(), &mut gpu_ref, lora);
            }
        } else {
            decode_batch_cpu(&mut active, &weights, &config);
        }
    }
}

/// Run one prefill pass for `req` and add it to `active`.
fn prefill_and_add(
    active:       &mut Vec<ActiveSequence>,
    req:          InferenceRequest,
    weights:      &Arc<TransformerWeights>,
    config:       &Arc<ModelConfig>,
    gpu:          &mut Option<GpuBackend>,
    cache_format: CacheFormat,
    vocab_index:  &Arc<VocabIndex>,
    registry:     &Arc<AdapterRegistry>,
) {
    // Resolve LoRA adapter by name (if requested).
    let lora_adapter = req.lora_name
        .as_deref()
        .and_then(|name| registry.get(name));

    let opts = SessionOptions {
        max_new_tokens: req.max_new_tokens,
        sampler_cfg:    req.sampler_cfg,
        eos_token:      req.eos_token,
        cache_format,
        context_length: config.context_length as usize,
        n_layers:       config.block_count as usize,
        n_kv_heads:     config.head_count_kv as usize,
        head_dim:       config.head_dim() as usize,
        lora_adapter:   lora_adapter.clone(),
    };
    let mut session = Session::new(opts);
    // Attach constraint if requested.
    if let Some(spec) = req.constraint {
        session.constraint  = Some(build_constraint(&spec, Arc::clone(vocab_index)));
        session.vocab_index = Some(Arc::clone(vocab_index));
    }
    session.tokens = req.prompt_tokens.clone();
    session.prefill_len = req.prompt_tokens.len();
    let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
    let lora_ref = lora_adapter.as_deref();
    session.last_logits = forward_prefill_lora(
        weights, config, &req.prompt_tokens, session.cache.as_mut(), 0, &mut gpu_ref, lora_ref,
    );
    session.pos = req.prompt_tokens.len().saturating_sub(1);
    active.push(ActiveSequence { session, tx: req.tx });
}

/// Advance every active sequence by one decode step in parallel (CPU path).
///
/// Each sequence has its own independent KV cache, so rayon can process all
/// sequences concurrently — no shared mutable state between iterations.
fn decode_batch_cpu(
    active:  &mut Vec<ActiveSequence>,
    weights: &Arc<TransformerWeights>,
    config:  &Arc<ModelConfig>,
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        active.par_iter_mut().for_each(|seq| {
            let s = &mut seq.session;
            let tok = *s.tokens.last().unwrap();
            let pos = s.tokens.len() - 1;
            let lora = s.lora_adapter.as_deref();
            s.last_logits = forward_one_lora(weights, config, tok, pos, s.cache.as_mut(), &mut None, lora);
        });
    }
    #[cfg(not(feature = "rayon"))]
    for seq in active.iter_mut() {
        let s = &mut seq.session;
        let tok = *s.tokens.last().unwrap();
        let pos = s.tokens.len() - 1;
        let lora = s.lora_adapter.as_deref();
        s.last_logits = forward_one_lora(weights, config, tok, pos, s.cache.as_mut(), &mut None, lora);
    }
}
