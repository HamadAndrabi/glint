//! Transformer architecture — forward pass, weights, and generation.

pub mod forward;
pub mod weights;

pub use forward::{embed, forward, forward_one, generate_cached, generate_greedy, generate_greedy_cached, generate_streaming};
pub use weights::TransformerWeights;
