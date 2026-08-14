//! Continuously-batched inference engine.
//!
//! `InferenceEngine` runs a single dedicated OS thread that owns the model
//! weights and (optionally) the GPU backend. Incoming requests are queued and
//! processed in a tight loop:
//!
//! 1. **Admission** — new requests are accepted from the queue whenever the
//!    number of active sequences is below [`EngineLimits::max_active`]; each
//!    admitted request has its prompt prefilled into a fresh per-sequence cache
//!    (full-context by default, or pages from a shared pool when
//!    [`EngineLimits::kv_pool_pages`] is set). Admission is re-checked every
//!    iteration, so a request joins the running batch as soon as a slot frees
//!    rather than waiting for the batch to empty.
//! 2. **Batched decode** — every live sequence advances by one token in a
//!    *single* forward pass ([`decode_batch_cpu`] → `forward_batch_lora`).
//!    Decoding is memory-bound, so the win is that each weight matrix is
//!    streamed from RAM once per step instead of once per sequence: the cost of
//!    a step is roughly flat in the number of sequences sharing it, which is
//!    what turns concurrency into throughput.
//! 3. **Sampling & delivery** — each sequence samples from its own logits with
//!    its own sampler and constraint, and the token is pushed down that
//!    request's channel.
//! 4. **Draining & eviction** — a sequence that has produced its last token
//!    (EOS, token budget, or context limit) is kept in a *draining* state until
//!    every queued token has been delivered to the client, then removed. A
//!    disconnected client or one that stays too far behind is evicted. Either
//!    way it drops out of the next batch without stalling the sequences that
//!    are still decoding.
//!
//! Batching is invisible to a request: `forward_batch` is bit-identical to
//! `forward_one` per sequence, and everything else — sampler state, KV cache,
//! token budget, LoRA adapter, [`Finish`] outcome — stays strictly
//! per-sequence. A completion must not depend on how busy the server was, so
//! sharing a step with other requests cannot change the tokens it receives.
//! A batch of one degrades to exactly the previous single-sequence behaviour.
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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::backend::GpuBackend;
use crate::cache::{
    KvStore, PagePool, PoolStats, PrefixCache, PrefixCacheConfig, PrefixCacheStats,
};
use crate::constrained::{build_constraint, ConstraintSpec, VocabIndex};
use crate::error::GlintError;
use crate::model::config::ModelConfig;
use crate::model::lora::LoraWeights;
use crate::model::lora_registry::AdapterRegistry;
use crate::sampling::SamplerConfig;
use crate::session::{CacheFormat, Session, SessionOptions};
use crate::tensor::Tensor;
use crate::transformer::{
    forward_batch_lora, forward_one_lora, forward_prefill_lora, TransformerWeights,
};

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
    /// Size, in [`PAGE_SIZE`]-token pages, of a shared paged KV pool.
    ///
    /// `None` (the default) keeps the per-sequence full-context [`KvCache`].
    /// `Some(n)` gives every admitted f32 sequence a [`PagedKvCache`] drawing
    /// from one pool of `n` pages, so KV memory follows the tokens actually
    /// generated instead of `max_active` worst-case contexts. A sequence that
    /// cannot get a page ends with [`Finish::Length`] rather than taking the
    /// engine down.
    ///
    /// [`PAGE_SIZE`]: crate::cache::PAGE_SIZE
    /// [`KvCache`]: crate::cache::KvCache
    /// [`PagedKvCache`]: crate::cache::PagedKvCache
    pub kv_pool_pages: Option<usize>,
    /// Bounds for a [`PrefixCache`] over the paged pool, or `None` (the
    /// default) to prefill every prompt from scratch.
    ///
    /// When set — and only when `kv_pool_pages` is also set, since prefix reuse
    /// is page sharing — the engine keeps the KV pages of completed prefills
    /// and admits a request that shares a prompt prefix by forking them,
    /// prefilling only the suffix. Reuse is exact: a request served from a fork
    /// produces the same tokens, bit for bit, as one prefilled cold.
    pub prefix_cache: Option<PrefixCacheConfig>,
}

impl Default for EngineLimits {
    fn default() -> Self {
        Self {
            max_active: 8,
            queue_capacity: 32,
            kv_pool_pages: None,
            prefix_cache: None,
        }
    }
}

/// KV-memory bookkeeping, sampled by the engine for `/v1/metrics`.
///
/// Both halves are `None` unless the corresponding feature is configured. The
/// snapshot is refreshed at admission and retirement boundaries — not per
/// token — so `pool.live` is exact as of the last sequence that joined or left
/// rather than continuously.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineKvStats {
    /// Occupancy of the shared page pool ([`EngineLimits::kv_pool_pages`]).
    pub pool: Option<PoolStats>,
    /// Prefix-cache counters ([`EngineLimits::prefix_cache`]).
    pub prefix: Option<PrefixCacheStats>,
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
    /// Latest KV-memory snapshot published by the loop thread. Written at
    /// admission and retirement boundaries, read by `/v1/metrics`.
    kv_stats: Arc<Mutex<EngineKvStats>>,
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
        let kv_stats = Arc::new(Mutex::new(EngineKvStats::default()));
        let loop_stats = Arc::clone(&kv_stats);
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
                            &loop_stats,
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
        Self { tx, kv_stats }
    }

    /// Latest KV-memory snapshot: page-pool occupancy and prefix-cache
    /// counters, each present only when that feature is configured.
    ///
    /// Sampled by the engine thread when sequences are admitted or retired, so
    /// reading it costs one uncontended lock and never interferes with decoding.
    pub fn kv_stats(&self) -> EngineKvStats {
        *self
            .kv_stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    kv_stats: &Mutex<EngineKvStats>,
) {
    let mut active: Vec<ActiveSequence> = Vec::new();
    // Decoding one token at position `pos` requires `pos < context_length`
    // in the KV cache, i.e. `tokens.len() <= context_length` before the step.
    let context_length = config.context_length as usize;
    // One pool shared by every sequence this loop admits (see
    // `EngineLimits::kv_pool_pages`). Rebuilt if the loop is respawned after a
    // panic — the sequences holding its pages are gone by then anyway.
    let page_pool = limits.kv_pool_pages.map(|pages| {
        PagePool::new(
            pages,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        )
    });
    // Prefix reuse is page sharing, so it exists only where sessions actually
    // get paged caches: an f32 session drawing from a pool. (`Session::new`
    // gives a `Q8` session the contiguous quantised cache and ignores the pool,
    // so a registry there could never be filled — or safely read from.) Owned
    // here — the loop is the only thread that touches it, and it is touched
    // only when a sequence is admitted or retired, never per token.
    let mut prefix_cache = match cache_format {
        CacheFormat::F32 => page_pool
            .as_ref()
            .and(limits.prefix_cache)
            .map(PrefixCache::new),
        CacheFormat::Q8 => None,
    };

    loop {
        // ── Admit pending requests while below the concurrency cap ────────
        let mut admitted = false;
        while active.len() < limits.max_active {
            match rx.try_recv() {
                Ok(req) => {
                    prefill_and_add(
                        &mut active,
                        req,
                        weights,
                        config,
                        gpu,
                        cache_format,
                        page_pool.as_ref(),
                        prefix_cache.as_mut(),
                        vocab_index,
                        registry,
                    );
                    admitted = true;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        if active.is_empty() {
            if admitted {
                publish_kv_stats(kv_stats, page_pool.as_ref(), prefix_cache.as_ref());
            }
            // Nothing to decode — block until a request arrives.
            match rx.blocking_recv() {
                Some(req) => prefill_and_add(
                    &mut active,
                    req,
                    weights,
                    config,
                    gpu,
                    cache_format,
                    page_pool.as_ref(),
                    prefix_cache.as_mut(),
                    vocab_index,
                    registry,
                ),
                None => return, // all senders dropped; shut down
            }
            publish_kv_stats(kv_stats, page_pool.as_ref(), prefix_cache.as_ref());
            // Loop back to admit any additional requests that arrived while
            // we were blocked (so we prefill them before the first decode step).
            continue;
        }
        if admitted {
            publish_kv_stats(kv_stats, page_pool.as_ref(), prefix_cache.as_ref());
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
        let retired = !finished.is_empty();
        for i in finished.into_iter().rev() {
            active.swap_remove(i);
        }
        if retired {
            // Their pages have just gone back to the pool — resample.
            publish_kv_stats(kv_stats, page_pool.as_ref(), prefix_cache.as_ref());
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

        // A paged cache may need a new page for the position about to be
        // written. Claim it here, before the forward pass: a pool with nothing
        // left ends that one sequence (every token it produced is still
        // delivered — that is `Length`, not a truncation) instead of panicking
        // inside the decode and taking every other sequence down with it.
        //
        // Retained prefixes are given up first: a cached prefix is an
        // optimisation for future requests and must never cost a running one
        // its next token.
        for seq in active.iter_mut().filter(|s| s.draining_since.is_none()) {
            let need = seq.session.tokens.len();
            if reserve_or_evict(seq.session.cache.as_mut(), need, prefix_cache.as_mut()).is_err() {
                seq.finish.set(Finish::Length);
                seq.draining_since = Some(now);
            }
        }

        // ── Advance all live (non-draining) sequences by one decode step ──
        //
        // GPU path: sequential (a single device context — nothing to share).
        // CPU path: one batched forward pass over every live sequence, so the
        //           weights are traversed once for the whole step.
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

/// Reserve `total_positions` in `cache`, giving up retained prefixes first.
///
/// A cached prefix holds pages a live request could be using, so a reservation
/// that fails against a full pool evicts prefix entries — least-recently-used
/// first — and retries until either the reservation succeeds or there is
/// nothing left to give back. Without a prefix cache this is exactly
/// `cache.reserve`.
fn reserve_or_evict(
    cache: &mut dyn KvStore,
    total_positions: usize,
    prefix_cache: Option<&mut PrefixCache>,
) -> Result<(), GlintError> {
    let mut last = match cache.reserve(total_positions) {
        Ok(()) => return Ok(()),
        Err(err) => err,
    };
    let Some(prefix_cache) = prefix_cache else {
        return Err(last);
    };
    while prefix_cache.evict_lru() {
        match cache.reserve(total_positions) {
            Ok(()) => return Ok(()),
            Err(err) => last = err,
        }
    }
    Err(last)
}

/// Publish a KV-memory snapshot for `/v1/metrics` to read.
///
/// Returns immediately when neither the pool nor the prefix cache is
/// configured, so the default engine never takes the lock at all.
fn publish_kv_stats(
    slot: &Mutex<EngineKvStats>,
    pool: Option<&PagePool>,
    prefix: Option<&PrefixCache>,
) {
    if pool.is_none() && prefix.is_none() {
        return;
    }
    let snapshot = EngineKvStats {
        pool: pool.map(PagePool::stats),
        prefix: prefix.map(PrefixCache::stats),
    };
    *slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = snapshot;
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
///
/// # Prefix reuse
///
/// With a `prefix_cache`, the prompt's longest cached page-aligned prefix is
/// forked into the new session's cache and only the remaining suffix is
/// prefilled, at a position offset of the prefix length. RoPE positions are
/// absolute and attention reads the whole cache, so the suffix sees exactly the
/// K/V a cold prefill would have written for those positions — the completion
/// is bit-identical either way. Afterwards the prompt's complete pages are
/// handed to the registry for the next request that shares them.
#[allow(clippy::too_many_arguments)]
fn prefill_and_add(
    active: &mut Vec<ActiveSequence>,
    req: InferenceRequest,
    weights: &Arc<TransformerWeights>,
    config: &Arc<ModelConfig>,
    gpu: &mut Option<GpuBackend>,
    cache_format: CacheFormat,
    page_pool: Option<&PagePool>,
    mut prefix_cache: Option<&mut PrefixCache>,
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
        page_pool: page_pool.cloned(),
    };
    let mut session = Session::new(opts);

    // Warm start: adopt a fork of the longest cached prefix of this prompt.
    // The session's own cache is empty and holds no pages at this point, so
    // replacing it costs nothing. The registry is keyed by adapter *name*
    // because the engine's adapter registry is fixed at startup, so the same
    // name always resolves to the same weights.
    let lora_key = req.lora_name.as_deref();
    let mut start = 0usize;
    if let Some(prefix_cache) = prefix_cache.as_deref_mut() {
        if let Some(fork) = prefix_cache.lookup(&req.prompt_tokens, lora_key) {
            start = fork.len();
            session.cache = Box::new(fork);
        }
    }

    // A paged cache takes its pages here rather than during the prefill, so a
    // pool with no room left rejects the request (the client sees its stream
    // end, `Finish::Incomplete`) instead of panicking mid-forward-pass.
    // Pre-allocated caches implement `reserve` as a no-op.
    if reserve_or_evict(
        session.cache.as_mut(),
        req.prompt_tokens.len(),
        prefix_cache.as_deref_mut(),
    )
    .is_err()
    {
        return;
    }
    // Attach constraint if requested.
    if let Some(spec) = req.constraint {
        match build_constraint(&spec, Arc::clone(vocab_index)) {
            Ok(c) => {
                session.constraint = Some(c);
                session.vocab_index = Some(Arc::clone(vocab_index));
            }
            Err(e) => {
                eprintln!("Error building constraint: {e}");
                req.finish.set(Finish::Truncated);
                return;
            }
        }
    }
    session.tokens = req.prompt_tokens.clone();
    session.prefill_len = req.prompt_tokens.len();
    let mut gpu_ref: Option<&mut GpuBackend> = gpu.as_mut();
    let lora_ref = lora_adapter.as_deref();
    session.last_logits = forward_prefill_lora(
        weights,
        config,
        &req.prompt_tokens[start..],
        session.cache.as_mut(),
        start,
        &mut gpu_ref,
        lora_ref,
    );
    session.pos = req.prompt_tokens.len().saturating_sub(1);

    // Retain this prompt's complete pages for the next request that shares
    // them. Only the prompt is offered — a generated continuation belongs to
    // this request alone — and only whole pages, so the sequence's own decoding
    // never writes a page the registry holds.
    if let (Some(prefix_cache), Some(paged)) = (prefix_cache, session.cache.as_paged()) {
        prefix_cache.insert(&req.prompt_tokens, paged, lora_key);
    }

    active.push(ActiveSequence {
        session,
        tx: req.tx,
        pending: std::collections::VecDeque::new(),
        draining_since: None,
        finish: req.finish,
    });
}

/// Advance every live (non-draining) sequence by one decode step — the batched
/// step at the heart of continuous batching (CPU path).
///
/// All live sequences go through a **single** [`forward_batch_lora`] call, so
/// each weight matrix is streamed from memory once for the whole batch instead
/// of once per sequence. Decoding is memory-bound, so this is where the
/// throughput of concurrent serving comes from.
///
/// Which sequences take part is decided fresh every step, which is what makes
/// the batching *continuous*: a sequence admitted one step ago joins the next
/// one automatically, and a sequence that finished simply stops appearing —
/// neither waits for the others.
///
/// Sequences are gathered in `active` order, and the logits come back in the
/// same order; every sequence keeps its own cache, position, sampler and LoRA
/// adapter, so sharing a step changes nothing it observes (see the parity
/// tests on `forward_batch`).
fn decode_batch_cpu(
    active: &mut [ActiveSequence],
    weights: &Arc<TransformerWeights>,
    config: &Arc<ModelConfig>,
) {
    let live = active.iter().filter(|s| s.draining_since.is_none()).count();
    if live == 0 {
        return;
    }

    // Single live sequence: call the single-sequence path directly rather than
    // building batch vectors for a batch of one. Nothing is amortised at B=1,
    // and a lone request should not pay an allocation per token for machinery
    // it does not use.
    if live == 1 {
        let seq = active
            .iter_mut()
            .find(|s| s.draining_since.is_none())
            .expect("live count says there is one");
        let s = &mut seq.session;
        let tok = *s.tokens.last().unwrap();
        let pos = s.tokens.len() - 1;
        let lora = s.lora_adapter.as_deref();
        s.last_logits =
            forward_one_lora(weights, config, tok, pos, s.cache.as_mut(), &mut None, lora);
        return;
    }

    let mut tokens: Vec<u32> = Vec::with_capacity(live);
    let mut positions: Vec<usize> = Vec::with_capacity(live);
    let mut caches: Vec<&mut dyn KvStore> = Vec::with_capacity(live);
    let mut loras: Vec<Option<&LoraWeights>> = Vec::with_capacity(live);
    let mut logit_slots: Vec<&mut Tensor> = Vec::with_capacity(live);

    for seq in active.iter_mut().filter(|s| s.draining_since.is_none()) {
        let s = &mut seq.session;
        tokens.push(*s.tokens.last().unwrap());
        positions.push(s.tokens.len() - 1);
        loras.push(s.lora_adapter.as_deref());
        caches.push(s.cache.as_mut());
        logit_slots.push(&mut s.last_logits);
    }

    let logits = forward_batch_lora(
        weights,
        config,
        &tokens,
        &positions,
        &mut caches,
        &mut None,
        &loras,
    );

    for (slot, produced) in logit_slots.into_iter().zip(logits) {
        *slot = produced;
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

    // ── Paged KV pool ─────────────────────────────────────────────────────

    // Switching the engine to a shared paged pool is a memory-layout change,
    // not a numerical one: the same prompt must yield the same tokens.
    #[test]
    fn paged_pool_produces_the_same_tokens_as_the_default_cache() {
        let plain = start_tiny_engine(EngineLimits::default(), 256);
        let paged = start_tiny_engine(
            EngineLimits {
                kv_pool_pages: Some(64),
                ..Default::default()
            },
            256,
        );
        let budget = 40; // spans several 16-token pages
        let from_plain = drain_all(
            plain
                .submit(vec![1, 2, 3], budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let from_paged = drain_all(
            paged
                .submit(vec![1, 2, 3], budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(from_plain.len(), budget);
        assert_eq!(from_plain, from_paged);
    }

    // A sequence that outgrows the pool stops cleanly — every token it did
    // produce is delivered and the outcome is Length — and the pages it hands
    // back on the way out serve the next request. Running out of KV memory
    // must never take the engine down.
    #[test]
    fn paged_pool_exhaustion_ends_the_sequence_not_the_engine() {
        let limits = EngineLimits {
            kv_pool_pages: Some(2), // 2 pages × 16 tokens, single-layer model
            ..Default::default()
        };
        let engine = start_tiny_engine(limits, 256);

        let (tokens, finish) = drain_with_finish(
            engine
                .submit(vec![1, 2, 3], 200, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert!(
            !tokens.is_empty() && tokens.len() < 200,
            "expected a short but non-empty completion, got {}",
            tokens.len()
        );
        assert_eq!(finish, Finish::Length);

        // The engine is still alive and the pool is usable again.
        let (again, finish_again) = drain_with_finish(
            engine
                .submit(vec![4, 5], 6, greedy_cfg(), NO_EOS, None, None)
                .expect("engine still accepts work"),
        );
        assert_eq!(again.len(), 6);
        assert_eq!(finish_again, Finish::Length);
    }

    // With max_active=1 and queue_capacity=1, a third request is rejected
    // with Busy instead of queueing without bound.
    #[test]
    fn saturated_queue_rejects_with_busy() {
        let limits = EngineLimits {
            max_active: 1,
            queue_capacity: 1,
            ..Default::default()
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

    // ── Continuous batching ──────────────────────────────────────────────
    //
    // Sequences now share one forward pass per step, and which sequences take
    // part changes as requests are admitted and retired. The contract that
    // makes that safe is that a request cannot tell: these tests pin the
    // outputs of concurrent requests against the same requests run alone, and
    // check that sequences joining and leaving a running batch disturb
    // neither the others' tokens nor their own outcomes.

    /// Build one live sequence: prompt prefilled, `next` staged as the token
    /// this step will decode (what the engine's sampler would have pushed).
    ///
    /// The receiver is returned alongside so the caller can keep it alive —
    /// dropping it would make the sequence look disconnected.
    fn live_sequence(
        weights: &Arc<TransformerWeights>,
        config: &Arc<ModelConfig>,
        prompt: &[u32],
        next: u32,
    ) -> (ActiveSequence, mpsc::Receiver<u32>) {
        let mut session = Session::new(SessionOptions {
            max_new_tokens: 16,
            sampler_cfg: greedy_cfg(),
            eos_token: NO_EOS,
            cache_format: CacheFormat::F32,
            context_length: config.context_length as usize,
            n_layers: config.block_count as usize,
            n_kv_heads: config.head_count_kv as usize,
            head_dim: config.head_dim() as usize,
            lora_adapter: None,
            page_pool: None,
        });
        session.tokens = prompt.to_vec();
        session.prefill_len = prompt.len();
        session.last_logits = forward_prefill_lora(
            weights,
            config,
            prompt,
            session.cache.as_mut(),
            0,
            &mut None,
            None,
        );
        session.pos = prompt.len().saturating_sub(1);
        session.tokens.push(next);
        let (tx, rx) = mpsc::channel(64);
        (
            ActiveSequence {
                session,
                tx,
                pending: VecDeque::new(),
                draining_since: None,
                finish: FinishSignal::new(),
            },
            rx,
        )
    }

    // Timing-independent check of the step the scheduler runs: three sequences
    // with different prompts, positions and histories advanced in one batched
    // pass must land on exactly the logits they would have got stepping alone.
    // (The end-to-end tests below can only observe tokens, and cannot force
    // the requests to overlap; this one pins the batched step directly.)
    #[test]
    fn batched_step_matches_single_sequence_steps() {
        let (weights, mut config) = make_tiny_weights();
        config.context_length = 64;
        let weights = Arc::new(weights);
        let config = Arc::new(config);
        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[4], &[5, 0, 6]];
        let next: [u32; 3] = [6, 2, 7];

        // Reference: each sequence advanced by itself.
        let expected: Vec<Tensor> = (0..3)
            .map(|s| {
                let (mut seq, _rx) = live_sequence(&weights, &config, prompts[s], next[s]);
                let session = &mut seq.session;
                let pos = session.tokens.len() - 1;
                forward_one_lora(
                    &weights,
                    &config,
                    next[s],
                    pos,
                    session.cache.as_mut(),
                    &mut None,
                    None,
                )
            })
            .collect();

        // Same sequences, advanced together in one batched step.
        let mut keepalive = Vec::new();
        let mut active: Vec<ActiveSequence> = (0..3)
            .map(|s| {
                let (seq, rx) = live_sequence(&weights, &config, prompts[s], next[s]);
                keepalive.push(rx);
                seq
            })
            .collect();
        decode_batch_cpu(&mut active, &weights, &config);

        for (s, seq) in active.iter().enumerate() {
            let got = seq.session.last_logits.data();
            assert_eq!(got.len(), expected[s].data().len(), "seq {s} logits len");
            for (i, (&a, &b)) in expected[s].data().iter().zip(got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "seq {s} logit {i}: alone={a}, batched={b}"
                );
            }
        }
    }

    // A batch of one must behave exactly like the previous single-sequence
    // engine — same result, and no batch machinery involved.
    #[test]
    fn batch_of_one_matches_the_single_sequence_path() {
        let (weights, mut config) = make_tiny_weights();
        config.context_length = 64;
        let weights = Arc::new(weights);
        let config = Arc::new(config);

        let (mut solo, _rx_a) = live_sequence(&weights, &config, &[1, 2, 3], 5);
        let pos = solo.session.tokens.len() - 1;
        let expected = forward_one_lora(
            &weights,
            &config,
            5,
            pos,
            solo.session.cache.as_mut(),
            &mut None,
            None,
        );

        let (seq, _rx_b) = live_sequence(&weights, &config, &[1, 2, 3], 5);
        let mut active = vec![seq];
        decode_batch_cpu(&mut active, &weights, &config);

        for (i, (&a, &b)) in expected
            .data()
            .iter()
            .zip(active[0].session.last_logits.data())
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "logit {i}");
        }
    }

    // Draining sequences (finished, still delivering their tail) must be left
    // out of the batch entirely — decoding them again would run past their
    // last token and corrupt their cache.
    #[test]
    fn draining_sequences_are_excluded_from_the_batch() {
        let (weights, mut config) = make_tiny_weights();
        config.context_length = 64;
        let weights = Arc::new(weights);
        let config = Arc::new(config);

        let mut keepalive = Vec::new();
        let mut active: Vec<ActiveSequence> = [(&[1u32, 2, 3][..], 6u32), (&[4u32][..], 2)]
            .iter()
            .map(|(prompt, next)| {
                let (seq, rx) = live_sequence(&weights, &config, prompt, *next);
                keepalive.push(rx);
                seq
            })
            .collect();
        active[0].draining_since = Some(Instant::now());
        let frozen_logits = active[0].session.last_logits.data().to_vec();
        let frozen_len = active[0].session.cache.len();

        decode_batch_cpu(&mut active, &weights, &config);

        assert_eq!(
            active[0].session.last_logits.data(),
            frozen_logits.as_slice(),
            "a draining sequence must not be decoded"
        );
        assert_eq!(active[0].session.cache.len(), frozen_len);
    }

    // The load-bearing test. A completion must not depend on how busy the
    // server was when it arrived, so requests decoded together must produce
    // exactly what they produce one at a time.
    #[test]
    fn batched_decode_matches_serial_decode() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let budget = 24;
        let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 0, 7, 1]];

        // Serial: each request has the engine to itself (a batch of one).
        let serial: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                drain_all(
                    engine
                        .submit(p.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                        .expect("queue has room"),
                )
            })
            .collect();
        for stream in &serial {
            assert_eq!(stream.len(), budget, "serial run should spend its budget");
        }

        // Concurrent: all three in flight at once, sharing decode steps.
        let submitted: Vec<Submitted> = prompts
            .iter()
            .map(|p| {
                engine
                    .submit(p.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                    .expect("queue has room")
            })
            .collect();
        let concurrent: Vec<Vec<u32>> = submitted.into_iter().map(drain_all).collect();

        assert_eq!(
            concurrent, serial,
            "sharing a batch changed the tokens a request received"
        );
    }

    // Sequences enter and leave a running batch, so the engine must keep each
    // one's logits, cache and sampler matched to the right request as the
    // batch's membership shifts under it.
    #[test]
    fn sequences_leaving_the_batch_do_not_disturb_the_rest() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let budget = 40;
        let prompt = vec![1, 2, 3];

        let alone = drain_all(
            engine
                .submit(prompt.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        // Same request, but now short-lived requests keep joining and retiring
        // around it mid-generation.
        let long = engine
            .submit(prompt, budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        let shorts: Vec<Submitted> = (0..3u32)
            .map(|i| {
                engine
                    .submit(vec![4 + i, 5], 5, greedy_cfg(), NO_EOS, None, None)
                    .expect("queue has room")
            })
            .collect();
        for short in shorts {
            assert_eq!(drain_all(short).len(), 5);
        }
        assert_eq!(
            drain_all(long),
            alone,
            "a neighbour retiring changed this sequence's tokens"
        );
    }

    // A request that arrives while another is mid-generation joins the batch
    // straight away — it must not wait for the running sequence to finish, and
    // both must complete in full.
    #[test]
    fn request_admitted_mid_generation_completes() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let long_budget = 60;
        let mut first = engine
            .submit(vec![1, 2], long_budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");

        // Let the first request get well into its generation before the
        // second one shows up.
        let mut seen = 0;
        for _ in 0..5 {
            assert!(first.rx.blocking_recv().is_some(), "first is live");
            seen += 1;
        }

        let late_budget = 12;
        let second = engine
            .submit(vec![3, 4], late_budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        assert_eq!(
            drain_all(second).len(),
            late_budget,
            "a request admitted mid-batch must complete"
        );

        while first.rx.blocking_recv().is_some() {
            seen += 1;
        }
        assert_eq!(seen, long_budget, "the running request also completes");
        assert_eq!(first.finish.get(), Finish::Length);
    }

    // With every slot busy, queued requests must be admitted as sequences
    // retire — not held until the whole batch drains.
    #[test]
    fn finished_sequences_free_slots_for_queued_requests() {
        let limits = EngineLimits {
            max_active: 2,
            queue_capacity: 16,
            ..Default::default()
        };
        let engine = start_tiny_engine(limits, 256);
        let budget = 8;
        let submitted: Vec<Submitted> = (0..6u32)
            .map(|i| {
                engine
                    .submit(vec![1 + i % 4, 2], budget, greedy_cfg(), NO_EOS, None, None)
                    .expect("queue has room")
            })
            .collect();
        for sub in submitted {
            let (tokens, finish) = drain_with_finish(sub);
            assert_eq!(tokens.len(), budget, "every queued request completes");
            assert_eq!(finish, Finish::Length);
        }
    }

    // Outcomes are per-sequence, not per-batch: one sequence hitting EOS in a
    // shared step must not end the others, and each client must learn why its
    // own stream stopped.
    #[test]
    fn finish_outcomes_stay_per_sequence_within_a_batch() {
        let engine = start_tiny_engine(EngineLimits::default(), 256);
        let prompt = vec![1, 2];
        let probe = engine
            .submit(prompt.clone(), 1, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        let first = drain_all(probe)[0];

        // Declaring the model's first token as EOS makes this sequence stop
        // after one token, while its batch-mate keeps decoding.
        let stopper = engine
            .submit(prompt, 50, greedy_cfg(), first, None, None)
            .expect("queue has room");
        let budget = 20;
        let runner = engine
            .submit(vec![3, 4, 5], budget, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");

        let (stop_tokens, stop_finish) = drain_with_finish(stopper);
        let (run_tokens, run_finish) = drain_with_finish(runner);
        assert_eq!(stop_finish, Finish::Stop);
        assert_eq!(stop_tokens, vec![first]);
        assert_eq!(run_finish, Finish::Length);
        assert_eq!(run_tokens.len(), budget);
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

    // ── Prefix caching ───────────────────────────────────────────────────
    //
    // A request whose prompt opens with a prefix another request already
    // prefilled starts from that prefix's KV pages instead of recomputing
    // them. The bar is that this is invisible: skipping work must not change a
    // single token, or the server's answers would depend on what it happened
    // to have served before.

    /// Paged pool, no prefix reuse — the reference configuration.
    fn paged_limits(pool_pages: usize) -> EngineLimits {
        EngineLimits {
            kv_pool_pages: Some(pool_pages),
            ..Default::default()
        }
    }

    /// Paged pool with prefix reuse enabled.
    fn prefix_limits(pool_pages: usize) -> EngineLimits {
        EngineLimits {
            kv_pool_pages: Some(pool_pages),
            prefix_cache: Some(PrefixCacheConfig::default()),
            ..Default::default()
        }
    }

    /// A 40-token shared opening (two full pages plus a partial one) followed
    /// by a per-request tail — the shape of a chat request behind a long
    /// system prompt.
    fn shared_prompt(tail: &[u32]) -> Vec<u32> {
        let mut prompt: Vec<u32> = (0..40u32).map(|i| i % 7).collect();
        prompt.extend_from_slice(tail);
        prompt
    }

    fn prefix_stats(engine: &InferenceEngine) -> PrefixCacheStats {
        engine
            .kv_stats()
            .prefix
            .expect("prefix cache is enabled for this engine")
    }

    /// Poll `ready` until it holds or five seconds pass.
    ///
    /// The KV snapshot is published by the engine thread just *after* a
    /// sequence is removed, which is the same moment its client's channel
    /// closes — so a test that drains a stream and reads the counters
    /// immediately can beat the publish. Waiting removes the race without
    /// weakening the assertion.
    fn wait_for(engine: &InferenceEngine, ready: impl Fn(PrefixCacheStats) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready(prefix_stats(engine)) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Admit `prompt` through the real admission path, returning the sequence
    /// and its receiver (dropping the receiver would look like a disconnect).
    ///
    /// Calling [`prefill_and_add`] directly is what lets the tests below
    /// compare *logits and K/V rows* rather than sampled tokens: the tiny test
    /// model's argmax is the same for almost any input, so a token stream is a
    /// coarse probe for "did the cache change".
    fn admit(
        weights: &Arc<TransformerWeights>,
        config: &Arc<ModelConfig>,
        pool: &PagePool,
        prefix_cache: Option<&mut PrefixCache>,
        prompt: Vec<u32>,
    ) -> (ActiveSequence, mpsc::Receiver<u32>) {
        let (tx, rx) = mpsc::channel(64);
        let req = InferenceRequest {
            prompt_tokens: prompt,
            max_new_tokens: 8,
            sampler_cfg: greedy_cfg(),
            eos_token: NO_EOS,
            constraint: None,
            lora_name: None,
            tx,
            finish: FinishSignal::new(),
        };
        let vocab: Vec<String> = (0..config.vocab_size).map(|i| format!("t{i}")).collect();
        let mut active = Vec::new();
        prefill_and_add(
            &mut active,
            req,
            weights,
            config,
            &mut None,
            CacheFormat::F32,
            Some(pool),
            prefix_cache,
            &VocabIndex::from_vocab(&vocab),
            &Arc::new(AdapterRegistry::new()),
        );
        (active.pop().expect("the prompt was admitted"), rx)
    }

    /// Admit `prompts` twice — once with no registry, once with one — and
    /// assert the two runs are indistinguishable: same logits, and the same
    /// K/V in every cache row of every layer.
    ///
    /// This is the correctness bar for the whole feature. Skipping the prefill
    /// of a shared prefix is only sound if what the borrower ends up holding is
    /// bit-for-bit what it would have computed itself.
    fn assert_reuse_is_bit_identical(prompts: &[Vec<u32>], expect_reused: u64) {
        let (raw_weights, mut config) = make_tiny_weights();
        config.context_length = 256;
        let weights = Arc::new(raw_weights);
        let config = Arc::new(config);
        let n_kv_heads = config.head_count_kv as usize;
        let head_dim = config.head_dim() as usize;
        let n_layers = config.block_count as usize;

        let cold_pool = PagePool::new(256, n_kv_heads, head_dim);
        let cold: Vec<_> = prompts
            .iter()
            .map(|p| admit(&weights, &config, &cold_pool, None, p.clone()))
            .collect();

        let warm_pool = PagePool::new(256, n_kv_heads, head_dim);
        let mut registry = PrefixCache::new(PrefixCacheConfig::default());
        let warm: Vec<_> = prompts
            .iter()
            .map(|p| {
                admit(
                    &weights,
                    &config,
                    &warm_pool,
                    Some(&mut registry),
                    p.clone(),
                )
            })
            .collect();
        assert_eq!(
            registry.stats().tokens_reused,
            expect_reused,
            "unexpected amount of prefix reuse"
        );

        for (seq, ((cold_seq, _), (warm_seq, _))) in cold.iter().zip(&warm).enumerate() {
            let cold_logits = cold_seq.session.last_logits.data();
            let warm_logits = warm_seq.session.last_logits.data();
            assert_eq!(cold_logits.len(), warm_logits.len(), "seq {seq} logits len");
            for (i, (&a, &b)) in cold_logits.iter().zip(warm_logits).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "seq {seq} logit {i}: cold={a}, reused={b}"
                );
            }

            let cold_cache = cold_seq.session.cache.as_paged().expect("paged");
            let warm_cache = warm_seq.session.cache.as_paged().expect("paged");
            assert_eq!(cold_cache.len(), warm_cache.len(), "seq {seq} cache len");
            for layer in 0..n_layers {
                for pos in 0..cold_cache.len() {
                    assert_eq!(
                        cold_cache.k_at(layer, pos),
                        warm_cache.k_at(layer, pos),
                        "seq {seq} layer {layer} K row {pos}"
                    );
                    assert_eq!(
                        cold_cache.v_at(layer, pos),
                        warm_cache.v_at(layer, pos),
                        "seq {seq} layer {layer} V row {pos}"
                    );
                }
            }
        }
    }

    // Two prompts sharing a 40-token opening: the second is admitted on the
    // first's pages and prefills only its suffix, at a non-zero position
    // offset. Every cache row and every logit must match the cold run.
    #[test]
    fn a_forked_admission_is_bit_identical_to_a_cold_one() {
        assert_reuse_is_bit_identical(&[shared_prompt(&[1, 2, 3]), shared_prompt(&[4, 5, 6])], 32);
    }

    // Same, where the shared opening is 20 tokens — one whole page plus a
    // partial one. Exactly the complete page is reused; the four positions of
    // the partial page are recomputed per request, so the page in which the
    // prompts diverge is never shared.
    #[test]
    fn a_partial_page_prefix_is_bit_identical_to_a_cold_one() {
        let base: Vec<u32> = (0..20u32).map(|i| i % 5).collect();
        let mut a = base.clone();
        let mut b = base;
        a.extend([1, 1]);
        b.extend([6, 6]);
        assert_reuse_is_bit_identical(&[a, b], 16);
    }

    // A chain of three requests, each extending the last: reuse must stay
    // exact as the registry's entries are superseded by longer prefixes.
    #[test]
    fn a_growing_chain_of_prompts_stays_bit_identical() {
        let long = shared_prompt(&[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]);
        let prompts = vec![long[..24].to_vec(), long[..40].to_vec(), long.clone()];
        // 24 → miss; 40 → one page of the 16-token entry; 52 → two pages of
        // the 32-token entry.
        assert_reuse_is_bit_identical(&prompts, 16 + 32);
    }

    // Prompts that share nothing reuse nothing, and a registry sitting in the
    // path leaves their caches exactly as a cold admission would.
    #[test]
    fn unrelated_prompts_reuse_nothing_and_stay_bit_identical() {
        let a: Vec<u32> = (0..40u32).map(|i| i % 7).collect();
        let b: Vec<u32> = (0..40u32).map(|i| (i * 3 + 1) % 7).collect();
        assert_reuse_is_bit_identical(&[a, b], 0);
    }

    // THE load-bearing test. A completion served from a forked prefix must be
    // bit-identical to the same completion prefilled from scratch — the reuse
    // is a memory optimisation, never a numerical one.
    #[test]
    fn prefix_hit_produces_the_same_tokens_as_a_cold_prefill() {
        let budget = 24;
        let prompt_a = shared_prompt(&[1, 2, 3]);
        let prompt_b = shared_prompt(&[4, 5, 6]);

        // Reference: both prompts prefilled cold (pool, but no prefix reuse).
        let cold = start_tiny_engine(paged_limits(256), 256);
        let expect_a = drain_all(
            cold.submit(prompt_a.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let expect_b = drain_all(
            cold.submit(prompt_b.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(expect_a.len(), budget);

        // Warm: A fills the registry, B is admitted on top of A's pages.
        let warm = start_tiny_engine(prefix_limits(256), 256);
        let got_a = drain_all(
            warm.submit(prompt_a, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let got_b = drain_all(
            warm.submit(prompt_b, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        assert_eq!(got_a, expect_a, "the cold-path request changed");
        assert_eq!(
            got_b, expect_b,
            "reuse changed the tokens a client received"
        );

        let stats = prefix_stats(&warm);
        assert_eq!(stats.hits, 1, "B should have been served from A's pages");
        assert_eq!(stats.misses, 1, "A found an empty registry");
        assert_eq!(
            stats.tokens_reused, 32,
            "43 prompt positions share two whole pages"
        );
    }

    // The prefix's producer does not have to be finished: a request can fork
    // the pages of a sequence that is still decoding. Copy-on-write is what
    // keeps the two apart, and both must still match their cold outputs.
    #[test]
    fn prefix_is_reusable_while_the_first_request_is_still_decoding() {
        let prompt_a = shared_prompt(&[1, 2, 3]);
        let prompt_b = shared_prompt(&[4, 5, 6]);
        let budget_b = 20;

        let cold = start_tiny_engine(paged_limits(256), 256);
        let expect_b = drain_all(
            cold.submit(prompt_b.clone(), budget_b, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        let warm = start_tiny_engine(prefix_limits(256), 256);
        // A is long-running; wait for a token so we know it has been admitted
        // (and therefore that its prompt is in the registry).
        let mut sub_a = warm
            .submit(prompt_a, 200, greedy_cfg(), NO_EOS, None, None)
            .expect("queue has room");
        assert!(sub_a.rx.blocking_recv().is_some(), "A is live");

        let got_b = drain_all(
            warm.submit(prompt_b, budget_b, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(got_b, expect_b, "forking a live sequence's pages changed B");
        assert_eq!(prefix_stats(&warm).hits, 1);

        // A keeps decoding correctly with B alongside it.
        let mut seen = 1;
        while sub_a.rx.blocking_recv().is_some() {
            seen += 1;
        }
        assert_eq!(seen, 200, "A completed its own budget");
    }

    // Sharing is whole pages only. Prompts with 20 tokens in common share the
    // one complete page; the four tokens of the partial page are prefilled per
    // request, so divergence inside that page cannot corrupt either sequence.
    // (The bitwise version of this is
    // `a_partial_page_prefix_is_bit_identical_to_a_cold_one`; this one pins the
    // page accounting end to end.)
    #[test]
    fn a_partial_page_prefix_shares_exactly_one_page() {
        let budget = 16;
        let base: Vec<u32> = (0..20u32).map(|i| i % 5).collect();
        let mut prompt_a = base.clone();
        let mut prompt_b = base;
        prompt_a.extend([1, 1]); // diverges at position 20 — inside page 1
        prompt_b.extend([6, 6]);

        let cold = start_tiny_engine(paged_limits(256), 256);
        let expect_a = drain_all(
            cold.submit(prompt_a.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let expect_b = drain_all(
            cold.submit(prompt_b.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        let warm = start_tiny_engine(prefix_limits(256), 256);
        let got_a = drain_all(
            warm.submit(prompt_a, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let got_b = drain_all(
            warm.submit(prompt_b, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        assert_eq!(got_a, expect_a);
        assert_eq!(
            got_b, expect_b,
            "the diverging tail page leaked between them"
        );
        let stats = prefix_stats(&warm);
        assert_eq!(stats.hits, 1);
        assert_eq!(
            stats.tokens_reused, 16,
            "only the one complete page is shared"
        );
    }

    // A prompt with nothing in common with anything cached is prefilled cold
    // and produces exactly what it would have without a registry.
    #[test]
    fn an_unrelated_prompt_misses_and_is_unaffected() {
        let budget = 12;
        let cold = start_tiny_engine(paged_limits(256), 256);
        let expect = drain_all(
            cold.submit(vec![7, 0, 7], budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        let warm = start_tiny_engine(prefix_limits(256), 256);
        let _ = drain_all(
            warm.submit(shared_prompt(&[1]), 4, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let got = drain_all(
            warm.submit(vec![7, 0, 7], budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(got, expect);
        let stats = prefix_stats(&warm);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.tokens_reused, 0);
    }

    // Retained prefixes must never starve a live request. A sequence that runs
    // out of pages gives up cached prefixes and keeps decoding — it must reach
    // exactly as far as it would have with the whole pool to itself.
    #[test]
    fn a_running_sequence_evicts_cached_prefixes_rather_than_stopping_short() {
        let pool_pages = 6; // single-layer test model: 6 × 16 = 96 positions
        let prompt = vec![1, 2, 3];

        // Reference: the pool holds nothing back.
        let cold = start_tiny_engine(paged_limits(pool_pages), 512);
        let (expect, cold_finish) = drain_with_finish(
            cold.submit(prompt.clone(), 500, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(cold_finish, Finish::Length, "the pool is the limit here");
        assert!(!expect.is_empty());

        // Warm: a cached prefix is holding pages when the long request starts.
        let warm = start_tiny_engine(prefix_limits(pool_pages), 512);
        let _ = drain_all(
            warm.submit(shared_prompt(&[]), 4, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert!(prefix_stats(&warm).entries > 0, "registry is holding pages");

        let (got, warm_finish) = drain_with_finish(
            warm.submit(prompt, 500, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        assert_eq!(warm_finish, Finish::Length);
        assert_eq!(
            got.len(),
            expect.len(),
            "a cached prefix cost a live request its tokens"
        );
        assert_eq!(got, expect);
        wait_for(&warm, |s| s.evictions >= 1);
        assert!(
            prefix_stats(&warm).evictions >= 1,
            "the prefix should have been given back to the pool"
        );
    }

    // A prompt too large to admit even after every prefix is evicted still
    // fails cleanly, and the engine keeps serving.
    #[test]
    fn admission_that_cannot_fit_even_after_eviction_fails_cleanly() {
        let warm = start_tiny_engine(prefix_limits(4), 512); // 4 × 16 = 64 positions
        let _ = drain_all(
            warm.submit(shared_prompt(&[]), 2, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let huge: Vec<u32> = (0..100u32).map(|i| i % 8).collect(); // needs 7 pages
        let (tokens, finish) = drain_with_finish(
            warm.submit(huge, 4, greedy_cfg(), NO_EOS, None, None)
                .expect("submit itself succeeds"),
        );
        assert!(tokens.is_empty());
        assert_eq!(finish, Finish::Incomplete);

        // Still alive, and the pool is usable again.
        let budget = 5;
        let (again, _) = drain_with_finish(
            warm.submit(vec![1, 2], budget, greedy_cfg(), NO_EOS, None, None)
                .expect("engine still accepts work"),
        );
        assert_eq!(again.len(), budget);
    }

    // Prefix reuse is opt-in. The default engine — and a paged engine without
    // it — must behave exactly as before and report no registry at all.
    #[test]
    fn prefix_cache_is_off_by_default() {
        let budget = 20;
        let prompt_a = shared_prompt(&[1, 2, 3]);
        let prompt_b = shared_prompt(&[4, 5, 6]);

        let default_engine = start_tiny_engine(EngineLimits::default(), 256);
        assert_eq!(
            default_engine.kv_stats(),
            EngineKvStats::default(),
            "the default engine reports neither a pool nor a registry"
        );
        let expect_a = drain_all(
            default_engine
                .submit(prompt_a.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let expect_b = drain_all(
            default_engine
                .submit(prompt_b.clone(), budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );

        // A paged engine reports its pool but no registry, and its tokens are
        // unchanged — enabling paging alone must not start reusing prefixes.
        let paged = start_tiny_engine(paged_limits(256), 256);
        let got_a = drain_all(
            paged
                .submit(prompt_a, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let got_b = drain_all(
            paged
                .submit(prompt_b, budget, greedy_cfg(), NO_EOS, None, None)
                .expect("queue has room"),
        );
        let stats = paged.kv_stats();
        assert!(stats.pool.is_some(), "the pool is reported");
        assert!(stats.prefix.is_none(), "no registry without opting in");
        assert_eq!(got_a, expect_a);
        assert_eq!(got_b, expect_b);
    }

    // The Q8 cache format has no pages to share, so a registry must not be
    // built for it even when one is configured — a `Q8` session keeps the
    // contiguous quantised cache, and its tokens are unaffected.
    #[test]
    fn q8_sessions_get_no_prefix_cache() {
        let (weights, mut config) = make_tiny_weights();
        config.context_length = 256;
        let vocab: Vec<String> = (0..config.vocab_size).map(|i| format!("t{i}")).collect();
        let engine = InferenceEngine::start(
            Arc::new(weights),
            Arc::new(config),
            None,
            CacheFormat::Q8,
            VocabIndex::from_vocab(&vocab),
            Arc::new(AdapterRegistry::new()),
            prefix_limits(256),
        );
        let budget = 12;
        let tokens = drain_all(
            engine
                .submit(
                    shared_prompt(&[1]),
                    budget,
                    greedy_cfg(),
                    NO_EOS,
                    None,
                    None,
                )
                .expect("queue has room"),
        );
        assert_eq!(tokens.len(), budget);
        assert!(
            engine.kv_stats().prefix.is_none(),
            "Q8 sessions never draw from the pool, so nothing could be cached"
        );
    }

    // ── reserve_or_evict ─────────────────────────────────────────────────

    // Without a registry, the helper is exactly `KvStore::reserve`.
    #[test]
    fn reserve_or_evict_without_a_registry_is_a_plain_reserve() {
        use crate::cache::{PagedKvCache, PAGE_SIZE};
        let pool = PagePool::new(1, 1, 4);
        let mut cache = PagedKvCache::new(&pool, 1);
        assert!(reserve_or_evict(&mut cache, PAGE_SIZE, None).is_ok());
        assert!(reserve_or_evict(&mut cache, PAGE_SIZE + 1, None).is_err());
    }

    // With one, an exhausted pool costs the least-recently-used prefix rather
    // than the reservation.
    #[test]
    fn reserve_or_evict_frees_prefixes_until_the_reservation_fits() {
        use crate::cache::{PagedKvCache, PrefixCache, PAGE_SIZE};
        let pool = PagePool::new(3, 1, 4);
        let mut registry = PrefixCache::new(PrefixCacheConfig::default());

        // Two cached prefixes hold one page each.
        for tag in 0..2u32 {
            let mut producer = PagedKvCache::new(&pool, 1);
            let tokens: Vec<u32> = (0..PAGE_SIZE as u32).map(|i| i * 3 + tag).collect();
            producer.reserve(tokens.len()).expect("pool has room");
            for pos in 0..tokens.len() {
                producer.write(0, pos, &[1.0; 4], &[2.0; 4]);
                producer.advance();
            }
            registry.insert(&tokens, &producer, None);
        }
        assert_eq!(registry.len(), 2);
        assert_eq!(pool.available_pages(), 1);

        // A sequence wanting all three pages gets them back.
        let mut live = PagedKvCache::new(&pool, 1);
        reserve_or_evict(&mut live, 3 * PAGE_SIZE, Some(&mut registry)).expect("prefixes freed");
        assert!(registry.is_empty(), "both prefixes were given back");
        assert_eq!(registry.stats().evictions, 2);
    }

    // Nothing left to evict is still an error — the caller must handle it.
    #[test]
    fn reserve_or_evict_reports_failure_when_the_registry_is_empty() {
        use crate::cache::{PagedKvCache, PrefixCache, PAGE_SIZE};
        let pool = PagePool::new(1, 1, 4);
        let mut registry = PrefixCache::new(PrefixCacheConfig::default());
        let mut cache = PagedKvCache::new(&pool, 1);
        let err = reserve_or_evict(&mut cache, 4 * PAGE_SIZE, Some(&mut registry))
            .expect_err("nothing can free these pages");
        assert!(matches!(err, GlintError::KvPagePoolExhausted { .. }));
    }
}
