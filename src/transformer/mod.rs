//! Transformer architecture — forward pass, weights, and generation.

pub mod forward;
pub mod weights;

pub use forward::{forward, forward_one, generate_greedy, generate_greedy_cached};
pub use weights::TransformerWeights;
