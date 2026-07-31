//! Transformer architecture — forward pass, weights, and generation.

pub mod forward;
pub mod speculative;
pub mod weights;

// Re-exported for the server engine's end-to-end tests; forward.rs's own
// tests reach the fixture directly.
#[cfg(all(test, feature = "server"))]
pub(crate) use forward::make_tiny_weights;
pub use forward::{
    embed, embed_batch, forward, forward_batch, forward_batch_lora, forward_one, forward_one_lora,
    forward_prefill, forward_prefill_all, forward_prefill_lora, generate_cached,
    generate_cached_q8, generate_greedy, generate_greedy_cached, generate_streaming,
};
pub use speculative::speculative_decode;
pub use weights::TransformerWeights;
