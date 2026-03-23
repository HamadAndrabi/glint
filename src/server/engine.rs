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
use crate::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tensor::Tensor;
use crate::transformer::{forward_one, forward_prefill, TransformerWeights};

// ── Types ─────────────────────────────────────────────────────────────────────

/// A request submitted to the engine by a route handler.
pub struct InferenceRequest {
    pub prompt_tokens:  Vec<u32>,
    pub max_new_tokens: usize,
    pub sampler_cfg:    SamplerConfig,
    pub eos_token:      u32,
    /// Tokens are delivered here; dropping the receiver signals client disconnect.
    pub tx: mpsc::Sender<u32>,
}

/// An in-flight sequence being decoded by the engine.
struct ActiveSequence {
    tokens:        Vec<u32>,
    cache:         KvCache,
    sampler:       Sampler,
    eos_token:     u32,
    max_remaining: usize,
    tx:            mpsc::Sender<u32>,
    last_logits:   Tensor,
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
        weights: Arc<TransformerWeights>,
        config:  Arc<ModelConfig>,
        gpu:     Option<GpuBackend>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("glint-inference".into())
            .spawn(move || engine_loop(rx, weights, config, gpu))
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
    ) -> Option<mpsc::Receiver<u32>> {
        let (token_tx, token_rx) = mpsc::channel(64);
        self.tx.send(InferenceRequest {
            prompt_tokens,
            max_new_tokens,
            sampler_cfg,
            eos_token,
            tx: token_tx,
        }).ok()?;
        Some(token_rx)
    }
}

// ── Engine loop ───────────────────────────────────────────────────────────────

fn engine_loop(
    mut rx:  mpsc::UnboundedReceiver<InferenceRequest>,
    weights: Arc<TransformerWeights>,
    config:  Arc<ModelConfig>,
    mut gpu: Option<GpuBackend>,
) {
    let mut active: Vec<ActiveSequence> = Vec::new();

    loop {
        // ── Drain any pending requests (prefill each one) ─────────────────
        loop {
            match rx.try_recv() {
                Ok(req) => prefill_and_add(&mut active, req, &weights, &config, &mut gpu),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        if active.is_empty() {
            // Nothing to decode — block until a request arrives.
            match rx.blocking_recv() {
                Some(req) => prefill_and_add(&mut active, req, &weights, &config, &mut gpu),
                None => return, // all senders dropped; shut down
            }
            // Loop back to drain any additional requests that arrived while
            // we were blocked (so we prefill them before the first decode step).
            continue;
        }

        // ── One decode step for every active sequence ─────────────────────
        let mut finished: Vec<usize> = Vec::new();

        for (i, seq) in active.iter_mut().enumerate() {
            // Check budget *before* sampling to avoid off-by-one overrun.
            if seq.max_remaining == 0 {
                finished.push(i);
                continue;
            }
            seq.max_remaining -= 1;

            let next = seq.sampler.sample(seq.last_logits.data(), &seq.tokens);
            seq.tokens.push(next);

            let eos_hit     = next == seq.eos_token;
            let disconnected = seq.tx.blocking_send(next).is_err();

            if eos_hit || disconnected {
                finished.push(i);
            } else {
                let pos = seq.tokens.len() - 1;
                let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
                seq.last_logits = forward_one(
                    &weights, &config, next, pos, &mut seq.cache, &mut gpu_ref,
                );
            }
        }

        // Remove finished sequences (reverse order so indices stay valid).
        for i in finished.into_iter().rev() {
            active.swap_remove(i);
        }
    }
}

/// Run one prefill pass for `req` and add it to `active`.
fn prefill_and_add(
    active:  &mut Vec<ActiveSequence>,
    req:     InferenceRequest,
    weights: &Arc<TransformerWeights>,
    config:  &Arc<ModelConfig>,
    gpu:     &mut Option<GpuBackend>,
) {
    let mut cache = KvCache::new(
        config.block_count as usize,
        config.context_length as usize,
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
    let last_logits = forward_prefill(
        weights, config, &req.prompt_tokens, &mut cache, 0, &mut gpu_ref,
    );
    active.push(ActiveSequence {
        tokens:        req.prompt_tokens,
        cache,
        sampler:       Sampler::new(req.sampler_cfg),
        eos_token:     req.eos_token,
        max_remaining: req.max_new_tokens,
        tx:            req.tx,
        last_logits,
    });
}
