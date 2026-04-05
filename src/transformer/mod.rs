//! Transformer architecture — forward pass, weights, and generation.

pub mod forward;
pub mod weights;
pub mod speculative;

pub use forward::{embed, embed_batch, forward, forward_batch, forward_one, forward_one_lora, forward_prefill, forward_prefill_lora, forward_prefill_all, generate_cached, generate_cached_q8, generate_greedy, generate_greedy_cached, generate_streaming};
pub use weights::TransformerWeights;
pub use speculative::speculative_decode;
