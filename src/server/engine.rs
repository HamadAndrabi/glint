//! Round-robin inference engine.
//!
//! `InferenceEngine` runs a single dedicated OS thread that owns the model
//! weights and (optionally) the GPU backend. Incoming requests are queued and
//! processed in a tight loop:
//!
//! 1. **Admission** — new requests are accepted from the queue only while the
//!    number of active sequences is below [`EngineLimits::max_active`]; each
//!    admitted request has its prompt prefilled into a fresh per-sequence cache.
//! 2. **Round-robin decode** — advance every live sequence by one token
//!    (each with its own `forward_one` call), sample, and push the token ID
//!    down the per-request channel. This is fair interleaving, not true batched
//!    inference — each sequence gets a separate forward pass.
//! 3. **Draining & eviction** — a sequence that has produced its last token
//!    (EOS, token budget, or context limit) is kept in a *draining* state until
//!    every queued token has been delivered to the client, then removed. A
//!    disconnected client or one that stays too far behind is evicted.
//!
//! The engine blocks (parks the thread) only when there are no active
//! sequences and no pending requests — i.e. when the server is completely idle.
//!
//! # Concurrency model
//!
//! Routes send `InferenceRequest`s through a **bounded** `tokio::sync::mpsc`
//! channel; when the queue is full [`InferenceEngine::submit`] fails fast with
//! [`SubmitError::Busy`] so the route can return HTTP 503 instead of letting
//! memory grow without bound (each admitted sequence preallocates a
//! full-context KV cache). The engine thread receives with `blocking_recv()`
//! (safe from a non-async thread).
//!
//! Each request carries its own `tokio::sync::mpsc::Sender<u32>` for token
//! delivery; the engine thread delivers with a **non-blocking** `try_send`
//! (see [`try_deliver`]) so one slow or stalled client cannot freeze decoding
//! for every other in-flight sequence. Undelivered tokens queue per sequence
//! and are retried each loop. Crucially, a *finished* sequence is not removed
//! until its queue has fully drained (or the client disconnects / exceeds
//! [`DRAIN_TIMEOUT`]) — removal on finish alone would silently drop the tail
//! of a completion whose reader is momentarily behind.
//!
//! Eviction is still possible, though, and a closed token channel looks the
//! same whether the completion finished or the engine gave up on it. Every
//! sequence therefore carries a [`FinishSignal`]: the engine records a
//! [`Finish`] there *before* dropping the sender, so the HTTP layer can tell a
//! finished completion ([`Finish::Stop`] / [`Finish::Length`]) from a truncated
//! one ([`Finish::Truncated`]) and terminate the stream honestly.
//!
//! # Fault containment
//!
//! The engine thread runs its loop under `catch_unwind`. If a decode step
//! panics, the panic is logged, every in-flight sequence is dropped (their
//! clients see their streams end), and the loop is **respawned** with fresh
//! state — the server keeps serving subsequent requests rather than turning
//! into a zombie that accepts work it can never perform.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::backend::GpuBackend;
use crate::constrained::{build_constraint, ConstraintSpec, VocabIndex};
use crate::model::config::ModelConfig;
use crate::model::lora_registry::AdapterRegistry;
use crate::sampling::SamplerConfig;
use crate::session::{CacheFormat, Session, SessionOptions};
use crate::transformer::{forward_one_lora, forward_prefill_lora, TransformerWeights};

// ── Types ─────────────────────────────────────────────────────────────────────

/// Why a sequence stopped producing tokens.
///
/// Token *data* flows over an mpsc channel, but that channel closing cannot
/// distinguish "the completion finished" from "the engine gave up on this
/// client and dropped the rest of it". Without that distinction the HTTP layer
/// has no honest way to terminate a stream, so the outcome travels out of band
/// alongside the tokens (see [`FinishSignal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Finish {
    /// No outcome was ever recorded — the decode loop panicked and the
    /// sequence was dropped, or the prompt was rejected during prefill.
    /// Treated exactly like [`Finish::Truncated`]: callers must not report
    /// success for a sequence that never reported one.
    Incomplete = 0,
    /// The EOS token was sampled — the model chose to stop.
    Stop = 1,
    /// The request's token budget or the model's context window ran out.
    /// Generation was cut short but every produced token was delivered.
    Length = 2,
    /// Evicted before its queued tail could be delivered (the client fell too
    /// far behind, or disconnected). What the client received is a strict
    /// prefix of what was generated.
    Truncated = 3,
}

/// Shared, write-once-per-outcome cell carrying a sequence's [`Finish`].
///
/// The engine writes the outcome **before** dropping the token sender, so a
/// reader that observes the channel closing is guaranteed to observe the
/// outcome too (`Release` store paired with the `Acquire` load in [`get`]).
///
/// [`get`]: FinishSignal::get
#[derive(Clone, Debug)]
pub struct FinishSignal(Arc<AtomicU8>);

impl FinishSignal {
    fn new() -> Self {
        Self(Arc::new(AtomicU8::new(Finish::Incomplete as u8)))
    }

    /// Record the outcome. Later calls overwrite earlier ones — a sequence
    /// that finished cleanly and *then* got evicted mid-drain is truncated.
    fn set(&self, finish: Finish) {
        self.0.store(finish as u8, Ordering::Release);
    }

    /// Read the recorded outcome. Only meaningful once the token channel has
    /// closed; before that the sequence is still running and reads
    /// [`Finish::Incomplete`].
    pub fn get(&self) -> Finish {
        match self.0.load(Ordering::Acquire) {
            1 => Finish::Stop,
            2 => Finish::Length,
            3 => Finish::Truncated,
            _ => Finish::Incomplete,
        }
    }
}

impl Default for FinishSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`InferenceEngine::submit`] hands back: the token stream plus the
/// out-of-band outcome for the sequence feeding it.
pub struct Submitted {
    /// Generated token IDs. Closes when the sequence leaves the engine.
    pub rx: mpsc::Receiver<u32>,
    /// Read this *after* `rx` closes to learn why the stream ended.
    pub finish: FinishSignal,
}

/// A request submitted to the engine by a route handler.
pub struct InferenceRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampler_cfg: SamplerConfig,
    pub eos_token: u32,
    /// Optional structured-output constraint (e.g. JSON object mode).
    pub constraint: Option<ConstraintSpec>,
    /// Optional LoRA adapter name to apply for this request.
    /// The engine resolves the name against its `AdapterRegistry` at prefill time.
    pub lora_name: Option<String>,
    /// Tokens are delivered here; dropping the receiver signals client disconnect.
    pub tx: mpsc::Sender<u32>,
    /// Where the engine records why this sequence ended, before dropping `tx`.
    pub finish: FinishSignal,
}

/// Admission limits for the engine.
///
/// Every admitted sequence preallocates a full-context KV cache, so
/// `max_active` bounds the engine's memory footprint and `queue_capacity`
/// bounds how much work can pile up behind it. A submit against a full queue
/// fails fast ([`SubmitError::Busy`]) so the HTTP layer can shed load with a
/// 503 instead of the process growing until the OOM killer sheds it instead.
#[derive(Clone, Copy, Debug)]
pub struct EngineLimits {
    /// Maximum sequences decoded concurrently; further requests wait in the queue.
    pub max_active: usize,
    /// Request-queue capacity; submits beyond this are rejected with `Busy`.
    pub queue_capacity: usize,
}

impl Default for EngineLimits {
    fn default() -> Self {
        Self {
            max_active: 8,
            queue_capacity: 32,
        }
    }
}

/// Why a submit was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// The request queue is full — the server is at capacity right now.
    Busy,
    /// The engine thread has shut down and cannot accept work.
    Shutdown,
}

/// An in-flight sequence being decoded by the engine.
struct ActiveSequence {
    session: Session,
    tx: mpsc::Sender<u32>,
    /// Tokens produced but not yet accepted by the client's channel.
    ///
    /// The engine never blocks on delivery (see [`try_deliver`]); when a
    /// client's channel is momentarily full, freshly sampled tokens queue here
    /// and are retried on the next loop iteration. This is what keeps one slow
    /// reader from stalling every other sequence sharing the engine thread.
    pending: std::collections::VecDeque<u32>,
    /// `Some(t)` once the sequence has produced its final token (EOS, budget,
    /// or context limit) at time `t`. A draining sequence is no longer decoded
    /// but stays alive until `pending` is empty, the client disconnects, or
    /// [`DRAIN_TIMEOUT`] elapses — whichever comes first.
    draining_since: Option<Instant>,
    /// Outcome cell shared with the requesting route. Written when decoding
    /// stops ([`Finish::Stop`] / [`Finish::Length`]) and overwritten with
    /// [`Finish::Truncated`] if the sequence is later evicted mid-drain.
    finish: FinishSignal,
}

/// Per-sequence outbound backlog past which we treat the client as unable to
/// keep up and evict its sequence, reclaiming the engine's time for others.
/// A well-behaved reader keeps `pending` at or near zero, so this is only ever
/// hit by a stalled or disconnected-but-not-yet-closed consumer.
const MAX_PENDING_TOKENS: usize = 4096;

/// How long a finished sequence may sit in the draining state waiting for a
/// slow client to accept its remaining tokens. Past this, the sequence is
/// evicted and the undelivered tail is dropped — the client had many seconds
/// to read a completed response and did not.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Outcome of attempting to deliver a sequence's queued tokens.
#[derive(PartialEq, Eq, Debug)]
enum Delivery {
    /// Channel still open and backlog within bounds — keep the sequence.
    Ok,
    /// Receiver dropped (client disconnected) — stop this sequence.
    Disconnected,
    /// Backlog exceeded `MAX_PENDING_TOKENS` — client too slow, evict.
    TooSlow,
}

/// Flush as many queued tokens as the client will accept **without blocking**.
///
/// Returns whether the sequence should continue, disconnected, or be evicted
/// for being too slow. Never parks the engine thread.
fn try_deliver(tx: &mpsc::Sender<u32>, pending: &mut std::collections::VecDeque<u32>) -> Delivery {
    while let Some(&tok) = pending.front() {
        match tx.try_send(tok) {
            Ok(()) => {
                pending.pop_front();
            }
            // Channel full: leave the rest queued and move on — the client is
            // reading, just not as fast as we produce this instant.
            Err(mpsc::error::TrySendError::Full(_)) => break,
            Err(mpsc::error::TrySendError::Closed(_)) => return Delivery::Disconnected,
        }
    }
    if pending.len() > MAX_PENDING_TOKENS {
        Delivery::TooSlow
    } else {
        Delivery::Ok
    }
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

/// Handle to the background inference thread.
///
/// Clone-able; all clones share the same underlying channel.
pub struct InferenceEngine {
    tx: mpsc::Sender<InferenceRequest>,
}

impl InferenceEngine {
    /// Spawn the inference thread and return a handle.
    pub fn start(
        weights: Arc<TransformerWeights>,
        config: Arc<ModelConfig>,
        gpu: Option<GpuBackend>,
        cache_format: CacheFormat,
        vocab_index: Arc<VocabIndex>,
        registry: Arc<AdapterRegistry>,
        limits: EngineLimits,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel(limits.queue_capacity.max(1));
        std::thread::Builder::new()
            .name("glint-inference".into())
            .spawn(move || {
                let mut gpu = gpu;
                // Supervise the decode loop: a panic drops the in-flight
                // sequences (their clients see their streams end) but the loop
                // is restarted with fresh state, so the server keeps serving
                // instead of becoming a zombie holding a dead channel.
                loop {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        engine_loop(
                            &mut rx,
                            &weights,
                            &config,
                            &mut gpu,
                            cache_format,
                            &vocab_index,
                            &registry,
                            limits,
                        )
                    }));
                    match result {
                        // Clean exit: all request senders dropped (server shutdown).
                        Ok(()) => break,
                        Err(_) => {
                            eprintln!(
                                "ERROR: glint inference engine panicked; in-flight requests \
                                 were dropped. Respawning the engine loop."
                            );
                        }
                    }
                }
            })
            .expect("failed to spawn inference thread");
        Self { tx }
    }

    /// Enqueue a generation request.
    ///
    /// Returns the token receiver immediately; the caller can await tokens as
    /// they arrive. Fails fast with [`SubmitError::Busy`] when the request
    /// queue is full (shed load — e.g. HTTP 503) and [`SubmitError::Shutdown`]
    /// if the engine thread is gone.
    ///
    /// The returned [`Submitted::finish`] carries why the stream ended and must
    /// be consulted once the receiver closes — a closed channel alone does not
    /// distinguish a finished completion from an evicted one.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        sampler_cfg: SamplerConfig,
        eos_token: u32,
        constraint: Option<ConstraintSpec>,
        lora_name: Option<String>,
    ) -> Result<Submitted, SubmitError> {
        let (token_tx, token_rx) = mpsc::channel(64);
        let finish = FinishSignal::new();
        let req = InferenceRequest {
            prompt_tokens,
            max_new_tokens,
            sampler_cfg,
            eos_token,
            constraint,
            lora_name,
            tx: token_tx,
            finish: finish.clone(),
        };
        match self.tx.try_send(req) {
            Ok(()) => Ok(Submitted {
                rx: token_rx,
                finish,
            }),
            Err(mpsc::error::TrySendError::Full(_)) => Err(SubmitError::Busy),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(SubmitError::Shutdown),
        }
    }
}

// ── Engine loop ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn engine_loop(
    rx: &mut mpsc::Receiver<InferenceRequest>,
    weights: &Arc<TransformerWeights>,
    config: &Arc<ModelConfig>,
    gpu: &mut Option<GpuBackend>,
    cache_format: CacheFormat,
    vocab_index: &Arc<VocabIndex>,
    registry: &Arc<AdapterRegistry>,
    limits: EngineLimits,
) {
    let mut active: Vec<ActiveSequence> = Vec::new();
    // Decoding one token at position `pos` requires `pos < context_length`
    // in the KV cache, i.e. `tokens.len() <= context_length` before the step.
    let context_length = config.context_length as usize;

    loop {
        // ── Admit pending requests while below the concurrency cap ────────
        while active.len() < limits.max_active {
            match rx.try_recv() {
                Ok(req) => prefill_and_add(
                    &mut active,
                    req,
                    weights,
                    config,
                    gpu,
                    cache_format,
                    vocab_index,
                    registry,
                ),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        if active.is_empty() {
            // Nothing to decode — block until a request arrives.
            match rx.blocking_recv() {
                Some(req) => prefill_and_add(
                    &mut active,
                    req,
                    weights,
                    config,
                    gpu,
                    cache_format,
                    vocab_index,
                    registry,
                ),
                None => return, // all senders dropped; shut down
            }
            // Loop back to admit any additional requests that arrived while
            // we were blocked (so we prefill them before the first decode step).
            continue;
        }

        // ── Sample from last_logits, deliver, and collect finished ────────
        let now = Instant::now();
        let mut finished: Vec<usize> = Vec::new();
        for (i, seq) in active.iter_mut().enumerate() {
            if seq.draining_since.is_none() {
                let s = &mut seq.session;
                if s.max_remaining == 0 {
                    // Budget exhausted before this step — nothing more to emit.
                    seq.finish.set(Finish::Length);
                    seq.draining_since = Some(now);
                } else {
                    s.max_remaining -= 1;
                    let next = if let Some(constraint) = s.constraint.as_mut() {
                        let vi = s.vocab_index.as_ref().unwrap();
                        let mask = constraint.allowed_tokens(&s.tokens, vi);
                        let tok =
                            s.sampler
                                .sample_constrained(s.last_logits.data(), &s.tokens, &mask);
                        constraint.advance(tok);
                        tok
                    } else {
                        s.sampler.sample(s.last_logits.data(), &s.tokens)
                    };
                    s.tokens.push(next);
                    seq.pending.push_back(next);

                    // Stop producing at EOS or when the next decode step would
                    // overflow the KV cache (which asserts, killing the loop).
                    // These are different outcomes to a client: EOS is the model
                    // choosing to stop, a full context is generation cut short.
                    if next == s.eos_token {
                        seq.finish.set(Finish::Stop);
                        seq.draining_since = Some(now);
                    } else if s.tokens.len() >= context_length {
                        seq.finish.set(Finish::Length);
                        seq.draining_since = Some(now);
                    }
                }
            }

            // Deliver queued tokens without blocking — for every sequence,
            // draining or not. A finished sequence leaves only once its queue
            // is empty (all tokens delivered) or its client gives up on us.
            match try_deliver(&seq.tx, &mut seq.pending) {
                // Evicted with tokens still queued: whatever the client got is
                // a strict prefix of the completion. Overwrites any Stop/Length
                // already recorded — the outcome the client can observe is that
                // its response is incomplete.
                Delivery::Disconnected | Delivery::TooSlow => {
                    seq.finish.set(Finish::Truncated);
                    finished.push(i);
                }
                Delivery::Ok => match seq.draining_since {
                    // Fully drained — keep the Stop/Length recorded at finish.
                    Some(_) if seq.pending.is_empty() => finished.push(i),
                    Some(since) if now.duration_since(since) > DRAIN_TIMEOUT => {
                        seq.finish.set(Finish::Truncated);
                        finished.push(i);
                    }
                    _ => {}
                },
            }
        }

        // Remove finished sequences (reverse order so indices stay valid).
        for i in finished.into_iter().rev() {
            active.swap_remove(i);
        }

        if active.is_empty() {
            continue;
        }

        // If every remaining sequence is draining (done decoding, waiting for
        // a slow client to read), there is no forward pass to run — yield
        // briefly instead of spinning on try_deliver.
        if active.iter().all(|s| s.draining_since.is_some()) {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        // ── Advance all live (non-draining) sequences by one decode step ──
        //
        // GPU path: sequential (only one GPU context).
        // CPU path: rayon par_iter_mut — N sequences in parallel, each calling
        //           forward_one against its own independent KV cache.
        if gpu.is_some() {
            for seq in active.iter_mut().filter(|s| s.draining_since.is_none()) {
                let s = &mut seq.session;
                let tok = *s.tokens.last().unwrap();
                let pos = s.tokens.len() - 1;
                let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
                let lora = s.lora_adapter.as_deref();
                s.last_logits = forward_one_lora(
                    weights,
                    config,
                    tok,
                    pos,
                    s.cache.as_mut(),
                    &mut gpu_ref,
                    lora,
                );
            }
        } else {
            decode_batch_cpu(&mut active, weights, config);
        }
    }
}

/// Run one prefill pass for `req` and add it to `active`.
///
/// A prompt that does not fit the model's context window is rejected outright:
/// prefilling it would overflow the KV cache (a hard assert). The request's
/// token channel is dropped without an outcome ever being recorded, so its
/// [`FinishSignal`] stays [`Finish::Incomplete`] and the client sees an error
/// rather than a successful empty completion.
/// (The HTTP layer clamps `max_tokens` against the context before submitting,
/// so this guard only fires for non-HTTP callers or future clamping bugs.)
#[allow(clippy::too_many_arguments)]
fn prefill_and_add(
    active: &mut Vec<ActiveSequence>,
    req: InferenceRequest,
    weights: &Arc<TransformerWeights>,
    config: &Arc<ModelConfig>,
    gpu: &mut Option<GpuBackend>,
    cache_format: CacheFormat,
    vocab_index: &Arc<VocabIndex>,
    registry: &Arc<AdapterRegistry>,
) {
    if req.prompt_tokens.is_empty() || req.prompt_tokens.len() > config.context_length as usize {
        return; // drops req.tx — client sees its stream end immediately
    }

    // Resolve LoRA adapter by name (if requested).
    let lora_adapter = req.lora_name.as_deref().and_then(|name| registry.get(name));

    let opts = SessionOptions {
        max_new_tokens: req.max_new_tokens,
        sampler_cfg: req.sampler_cfg,
        eos_token: req.eos_token,
        cache_format,
        context_length: config.context_length as usize,
        n_layers: config.block_count as usize,
        n_kv_heads: config.head_count_kv as usize,
        head_dim: config.head_dim() as usize,
        lora_adapter: lora_adapter.clone(),
    };
    let mut session = Session::new(opts);
    // Attach constraint if requested.
    if let Some(spec) = req.constraint {
        session.constraint = Some(build_constraint(&spec, Arc::clone(vocab_index)));
        session.vocab_index = Some(Arc::clone(vocab_index));
    }
    session.tokens = req.prompt_tokens.clone();
    session.prefill_len = req.prompt_tokens.len();
    let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
    let lora_ref = lora_adapter.as_deref();
    session.last_logits = forward_prefill_lora(
        weights,
        config,
        &req.prompt_tokens,
        session.cache.as_mut(),
        0,
        &mut gpu_ref,
        lora_ref,
    );
    session.pos = req.prompt_tokens.len().saturating_sub(1);
    active.push(ActiveSequence {
        session,
        tx: req.tx,
        pending: std::collections::VecDeque::new(),
        draining_since: None,
        finish: req.finish,
    });
}

/// Advance every live (non-draining) sequence by one decode step in parallel
/// (CPU path).
///
/// Each sequence has its own independent KV cache, so rayon can process all
/// sequences concurrently — no shared mutable state between iterations.
fn decode_batch_cpu(
    active: &mut [ActiveSequence],
    weights: &Arc<TransformerWeights>,
    config: &Arc<ModelConfig>,
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        active
            .par_iter_mut()
            .filter(|s| s.draining_since.is_none())
            .for_each(|seq| {
                let s = &mut seq.session;
                let tok = *s.tokens.last().unwrap();
                let pos = s.tokens.len() - 1;
                let lora = s.lora_adapter.as_deref();
                s.last_logits =
                    forward_one_lora(weights, config, tok, pos, s.cache.as_mut(), &mut None, lora);
            });
    }
    #[cfg(not(feature = "rayon"))]
    for seq in active.iter_mut().filter(|s| s.draining_since.is_none()) {
        let s = &mut seq.session;
        let tok = *s.tokens.last().unwrap();
        let pos = s.tokens.len() - 1;
        let lora = s.lora_adapter.as_deref();
        s.last_logits =
            forward_one_lora(weights, config, tok, pos, s.cache.as_mut(), &mut None, lora);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    // ── try_deliver unit tests ────────────────────────────────────────────

    // A well-behaved client that drains promptly: every queued token is
    // delivered and the sequence stays healthy.
    #[test]
    fn deliver_drains_when_receiver_reads() {
        let (tx, mut rx) = mpsc::channel::<u32>(64);
        let mut pending = VecDeque::from([1, 2, 3]);
        assert_eq!(try_deliver(&tx, &mut pending), Delivery::Ok);
        assert!(pending.is_empty(), "all tokens should have been sent");
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
    }

    // A client whose channel is full does NOT block the engine: undelivered
    // tokens stay queued for a later retry, and the sequence keeps going.
    #[test]
    fn deliver_backpressures_without_blocking() {
        let (tx, _rx) = mpsc::channel::<u32>(2); // capacity 2, never drained
        let mut pending = VecDeque::from([10, 20, 30, 40]);
        // Non-blocking: fills the 2 slots, leaves the rest queued, returns Ok.
        assert_eq!(try_deliver(&tx, &mut pending), Delivery::Ok);
        assert_eq!(pending.len(), 2, "two tokens remain queued for retry");
    }

    // A disconnected client (receiver dropped) is reported so the engine can
    // evict the sequence.
    #[test]
    fn deliver_detects_disconnect() {
        let (tx, rx) = mpsc::channel::<u32>(64);
        drop(rx);
        let mut pending = VecDeque::from([1]);
        assert_eq!(try_deliver(&tx, &mut pending), Delivery::Disconnected);
    }

    // A client that never reads eventually trips the too-slow guard instead of
    // letting the backlog grow without bound.
    #[test]
    fn deliver_evicts_when_backlog_exceeds_cap() {
        let (tx, _rx) = mpsc::channel::<u32>(1); // capacity 1, never drained
        let mut pending: VecDeque<u32> = (0..(MAX_PENDING_TOKENS as u32 + 10)).collect();
        assert_eq!(try_deliver(&tx, &mut pending), Delivery::TooSlow);
    }

    // ── End-to-end engine tests (tiny in-memory model, no HTTP) ──────────

    use crate::transformer::make_tiny_weights;

    /// Greedy sampling so token streams are deterministic.
    fn greedy_cfg() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(7),
            min_p: 0.0,
        }
    }

    /// Spin up an engine over the tiny test model.
    ///
    /// `context_length` is overridden so tests can generate more tokens than
    /// the per-request token channel holds (64) to exercise the drain path.
    fn start_tiny_engine(limits: EngineLimits, context_length: u32) -> InferenceEngine {
        let (weights, mut config) = make_tiny_weights();
        config.context_length = context_length;
        let vocab: Vec<String> = (0..config.vocab_size).map(|i| format!("t{i}")).collect();
        InferenceEngine::start(
            Arc::new(weights),
            Arc::new(config),
            None,
            CacheFormat::F32,
            VocabIndex::from_vocab(&vocab),
            Arc::new(AdapterRegistry::new()),
            limits,
        )
    }

    /// EOS id outside the tiny model's vocab, so generation always runs to
    /// its token budget (deterministic lengths for the assertions below).
    const NO_EOS: u32 = 9999;

    fn drain_all(sub: Submitted) -> Vec<u32> {
        drain_with_finish(sub).0
    }

    /// Drain the stream and read the outcome the engine recorded for it.
    ///
    /// Reading `finish` only after the channel closes is the contract callers
    /// must follow: the engine writes the outcome before dropping the sender.
    fn drain_with_finish(sub: Submitted) -> (Vec<u32>, Finish) {
        let Submitted { mut rx, finish } = sub;
        let mut out = Vec::new();
        while let Some(tok) = rx.blocking_recv() {
            out.push(tok);
        }
        (out, finish.get())
    }

    // Regression test for the token-loss bug: a completion longer than the
    // token channel's capacity (64), read only AFTER generation has already
    // finished, must still arrive in full. Before the draining state existed,
    // the engine dropped whatever had not fit in the channel at finish time.
    #[test]
    fn slow_reader_still_receives_every_token() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let budget = 100; // > channel capacity of 64
        let rx = engine
            .submit(vec![1, 2, 3], budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");

        // Give the engine ample time to finish generating before we read a
        // single token — the first 64 sit in the channel, the rest must be
        // held in the sequence's pending queue rather than dropped.
        std::thread::sleep(Duration::from_millis(500));

        let tokens = drain_all(rx);
        assert_eq!(
            tokens.len(),
            budget,
            "finished sequence must drain all tokens to a late reader"
        );
    }

    // Several concurrent requests each get their full, independent stream.
    #[test]
    fn concurrent_requests_all_complete() {
        let engine = start_tiny_engine(EngineLimits::default(), 64);
        let budget = 20;
        let receivers: Vec<_> = (0..3)
            .map(|i| {
                engine
                    .submit(vec![1 + i, 2], budget, greedy_cfg(), NO_EOS, None, None)
                    .expect("queue has room")
            })
            .collect();
        for rx in receivers {
            assert_eq!(drain_all(rx).len(), budget);
        }
    }

    // A client that disconnects mid-stream is evicted and does not wedge the
    // engine: a subsequent request still completes in full.
    #[test]
    fn disconnect_does_not_wedge_engine() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let mut sub_a = engine
            .submit(vec![1, 2], 200, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        // Read one token to be sure the sequence is live, then walk away.
        assert!(sub_a.rx.blocking_recv().is_some());
        drop(sub_a);

        let budget = 15;
        let rx_b = engine
            .submit(vec![3, 4], budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        assert_eq!(drain_all(rx_b).len(), budget);
    }

    // With max_active=1 and queue_capacity=1, a third request is rejected
    // with Busy instead of queueing without bound.
    #[test]
    fn saturated_queue_rejects_with_busy() {
        let limits = EngineLimits {
            max_active: 1,
            queue_capacity: 1,
        };
        let engine = start_tiny_engine(limits, 256);

        // Request A: long-running; wait for its first token so we know the
        // engine has admitted it (active=1 == max_active, admission paused).
        let mut sub_a = engine
            .submit(vec![1, 2], 200, greedy_cfg(), NO_EOS, None, None)
            .expect("first submit");
        assert!(sub_a.rx.blocking_recv().is_some());

        // Request B parks in the queue (capacity 1 → now full).
        let _sub_b = engine
            .submit(vec![3, 4], 5, greedy_cfg(), NO_EOS, None, None)
            .expect("second submit fills the queue");

        // Request C: queue full → shed load.
        let err = match engine.submit(vec![5, 6], 5, greedy_cfg(), NO_EOS, None, None) {
            Ok(_) => panic!("third submit must be rejected"),
            Err(err) => err,
        };
        assert_eq!(err, SubmitError::Busy);
    }

    // A prompt longer than the context window must not panic the engine (the
    // KV cache hard-asserts on overflow); the client sees an empty stream and
    // the engine keeps serving.
    #[test]
    fn oversized_prompt_is_rejected_not_panicking() {
        let engine = start_tiny_engine(EngineLimits::default(), 32);
        let huge_prompt: Vec<u32> = (0..40).map(|i| i % 8).collect(); // 40 > 32
        let rx = engine
            .submit(huge_prompt, 10, greedy_cfg(), NO_EOS, None, None)
            .expect("submit itself succeeds");
        assert!(
            drain_all(rx).is_empty(),
            "oversized prompt yields an empty stream"
        );

        // Engine is still alive and serving.
        let budget = 5;
        let rx = engine
            .submit(vec![1, 2], budget, greedy_cfg(), NO_EOS, None, None)
            .expect("engine still accepts work");
        assert_eq!(drain_all(rx).len(), budget);
    }

    // ── Finish-outcome tests ─────────────────────────────────────────────
    //
    // A closed token channel looks identical whether the completion finished
    // or the engine abandoned it, so every sequence also records a `Finish`.
    // These pin each outcome to the condition that produces it — the HTTP
    // layer turns them directly into finish_reason / stream termination.

    // Running out of token budget is "length", not "stop". Reporting this as
    // "stop" tells a client the model chose to end there, when in fact the
    // reply was cut off and could be continued.
    #[test]
    fn budget_exhaustion_reports_length() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let budget = 12;
        let sub = engine
            .submit(vec![1, 2], budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        let (tokens, finish) = drain_with_finish(sub);
        assert_eq!(tokens.len(), budget, "budget is spent in full");
        assert_eq!(finish, Finish::Length);
    }

    // Filling the context window is also "length" — generation stopped short
    // for a reason outside the model's control.
    #[test]
    fn context_limit_reports_length() {
        let context = 24u32;
        let engine = start_tiny_engine(EngineLimits::default(), context);
        let prompt = vec![1, 2, 3];
        // Budget far exceeds the remaining context, so the context bound is
        // what actually stops generation.
        let sub = engine
            .submit(prompt.clone(), 500, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        let (tokens, finish) = drain_with_finish(sub);
        assert_eq!(finish, Finish::Length);
        assert_eq!(
            prompt.len() + tokens.len(),
            context as usize,
            "generation runs exactly up to the context window"
        );
    }

    // Sampling EOS is the one genuine "stop". Greedy decoding is
    // deterministic, so learn which token this model emits first, then re-run
    // declaring that token to be EOS.
    #[test]
    fn eos_reports_stop() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let prompt = vec![1, 2];
        let probe = engine
            .submit(prompt.clone(), 1, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        let first = drain_all(probe)[0];

        let sub = engine
            .submit(prompt, 50, greedy_cfg(), first, None, None)
            .expect("queue has room");
        let (tokens, finish) = drain_with_finish(sub);
        assert_eq!(finish, Finish::Stop);
        assert_eq!(
            tokens,
            vec![first],
            "the EOS token is delivered, then generation stops"
        );
    }

    // A client that walks away is evicted, and the tail it never read is gone
    // — that is truncation, not completion. Holding the signal after dropping
    // the receiver is exactly how a route observes this.
    #[test]
    fn abandoned_stream_reports_truncated() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let Submitted { mut rx, finish } = engine
            .submit(vec![1, 2], 500, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        assert!(rx.blocking_recv().is_some(), "sequence is live");
        drop(rx);

        // The engine notices on its next delivery attempt.
        let deadline = Instant::now() + Duration::from_secs(5);
        while finish.get() == Finish::Incomplete && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(finish.get(), Finish::Truncated);
    }

    // A prompt the engine refuses never reaches the decode loop, so no outcome
    // is ever recorded. It must read Incomplete rather than defaulting to
    // something success-shaped: routes turn this into an error, not into an
    // empty-but-successful completion.
    #[test]
    fn rejected_prompt_reports_incomplete() {
        let engine = start_tiny_engine(EngineLimits::default(), 32);
        let huge_prompt: Vec<u32> = (0..40).map(|i| i % 8).collect(); // 40 > 32
        let sub = engine
            .submit(huge_prompt, 10, greedy_cfg(), NO_EOS, None, None)
            .expect("submit itself succeeds");
        let (tokens, finish) = drain_with_finish(sub);
        assert!(tokens.is_empty());
        assert_eq!(finish, Finish::Incomplete);
    }
}
