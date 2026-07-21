//! Shared server state — model weights, tokenizer, and config.
//!
//! Wrapped in `Arc` so every request handler gets a cheap clone that points
//! to the same memory. The inference weights are read-only after loading, so
//! there's no need for a mutex.

use std::sync::Arc;
use std::sync::RwLock;

use super::engine::InferenceEngine;
use super::metrics::Metrics;
use crate::constrained::VocabIndex;
use crate::model::chat_template::ChatTemplate;
use crate::model::config::ModelConfig;
use crate::model::lora_registry::AdapterRegistry;
use crate::model::tokenizer::Tokenizer;
use crate::transformer::TransformerWeights;

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
    /// Vocabulary index for constraint-based generation (JSON mode, etc.).
    pub vocab_index: Arc<VocabIndex>,
    /// Registry of named LoRA adapters (pre-loaded at startup).
    ///
    /// Behind `RwLock` to allow future dynamic registration without restarting;
    /// currently written only once at startup.
    pub adapter_registry: Arc<RwLock<AdapterRegistry>>,
}
