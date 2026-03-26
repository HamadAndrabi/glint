//! Shared server state — model weights, tokenizer, and config.
//!
//! Wrapped in `Arc` so every request handler gets a cheap clone that points
//! to the same memory. The inference weights are read-only after loading, so
//! there's no need for a mutex.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::engine::InferenceEngine;
use crate::model::chat_template::ChatTemplate;
use crate::model::config::ModelConfig;
use crate::model::tokenizer::Tokenizer;
use crate::transformer::TransformerWeights;

/// Server-wide runtime counters. All fields are atomically updated — no locks needed.
///
/// Wrapped in `AppState` which is itself behind an `Arc`, so every handler
/// can increment counters without taking a mutex.
pub struct Metrics {
    /// Total completed inference requests (completions + chat completions).
    pub requests_total: AtomicU64,
    /// Total tokens generated across all requests.
    pub tokens_generated: AtomicU64,
    /// Sum of server-side inference latencies in milliseconds (non-streaming only).
    pub total_latency_ms: AtomicU64,
    /// When the server started — used to compute uptime.
    pub started_at: Instant,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    pub fn record(&self, tokens: u64, latency_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
    }
}

/// All the data needed to serve inference requests.
///
/// Created once at startup and shared across all request handlers via `Arc`.
pub struct AppState {
    pub weights: Arc<TransformerWeights>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Arc<ModelConfig>,
    /// The model identifier returned in API responses (e.g. "smollm-135m-q8_0").
    pub model_name: String,
    /// Detected chat template format for formatting chat messages into prompts.
    pub chat_template: ChatTemplate,
    /// Runtime metrics — request counts, token throughput, latency.
    pub metrics: Metrics,
    /// Concurrent round-robin inference engine — owns the model weights and GPU.
    pub engine: Arc<InferenceEngine>,
}
