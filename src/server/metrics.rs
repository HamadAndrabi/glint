//! Server-wide runtime counters.
//!
//! All fields are `Atomic*` — no mutex needed.  The struct is placed behind an
//! `Arc<AppState>`, so every handler increments counters with a cheap atomic
//! operation and no contention.
//!
//! ## Counter semantics
//!
//! | Counter | When incremented | When decremented |
//! |---|---|---|
//! | `requests_total` | on completion | — |
//! | `requests_failed` | on any error response | — |
//! | `tokens_generated` | on completion | — |
//! | `total_latency_ms` | on completion | — |
//! | `ttft_total_us` | on first token | — |
//! | `decode_total_us` | on completion | — |
//! | `active_sessions` | on submit | on completion / disconnect |
//! | `queue_depth` | on submit | on first token (prefill done) |

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

pub struct Metrics {
    // ── Throughput ────────────────────────────────────────────────────────────
    /// Total completed inference requests.
    pub requests_total: AtomicU64,
    /// Total tokens generated across all requests.
    pub tokens_generated: AtomicU64,

    // ── Latency ───────────────────────────────────────────────────────────────
    /// Sum of end-to-end request durations in milliseconds.
    pub total_latency_ms: AtomicU64,
    /// Sum of time-to-first-token durations in microseconds.
    pub ttft_total_us: AtomicU64,
    /// Sum of decode-phase durations in microseconds.
    pub decode_total_us: AtomicU64,

    // ── Errors ────────────────────────────────────────────────────────────────
    /// Requests that returned an HTTP error response.
    pub requests_failed: AtomicU64,

    // ── Session concurrency ───────────────────────────────────────────────────
    /// Requests currently in flight (submitted but not yet completed).
    pub active_sessions: AtomicI64,
    /// Requests submitted to the engine but not yet past the prefill stage
    /// (i.e. first token not yet delivered to the client).
    pub queue_depth: AtomicI64,

    // ── Server uptime ─────────────────────────────────────────────────────────
    pub started_at: Instant,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            ttft_total_us: AtomicU64::new(0),
            decode_total_us: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            active_sessions: AtomicI64::new(0),
            queue_depth: AtomicI64::new(0),
            started_at: Instant::now(),
        }
    }

    // ── Write helpers ─────────────────────────────────────────────────────────

    /// Record a successfully completed request.
    pub fn record(&self, tokens: u64, latency_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
        self.decode_total_us
            .fetch_add(latency_ms.saturating_mul(1_000), Ordering::Relaxed);
    }

    /// Record the time-to-first-token for one request (in microseconds).
    pub fn record_ttft_us(&self, us: u64) {
        self.ttft_total_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record one failed request.
    pub fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Called immediately after `engine.submit()` succeeds.
    ///
    /// Increments both `active_sessions` (request is in-flight) and
    /// `queue_depth` (prefill not yet done).
    pub fn on_submit(&self) {
        self.active_sessions.fetch_add(1, Ordering::Relaxed);
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
    }

    /// Called when the first token is received from the engine.
    ///
    /// The prefill phase is done; the request transitions from queued → decoding.
    pub fn on_first_token(&self) {
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
    }

    /// Called when the request is fully complete (stream done or error).
    pub fn on_complete(&self) {
        self.active_sessions.fetch_sub(1, Ordering::Relaxed);
    }

    // ── Read helpers ──────────────────────────────────────────────────────────

    /// Average TTFT in milliseconds, or `0.0` if no completed requests.
    pub fn avg_ttft_ms(&self) -> f64 {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        self.ttft_total_us.load(Ordering::Relaxed) as f64 / reqs as f64 / 1_000.0
    }

    /// Average end-to-end latency in milliseconds, or `0.0`.
    pub fn avg_latency_ms(&self) -> f64 {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        self.total_latency_ms.load(Ordering::Relaxed) as f64 / reqs as f64
    }

    /// Average decode-phase duration in milliseconds, or `0.0`.
    pub fn avg_decode_ms(&self) -> f64 {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        self.decode_total_us.load(Ordering::Relaxed) as f64 / reqs as f64 / 1_000.0
    }
}
