//! Transformer architecture — forward pass, weights, and generation.

pub mod forward;
pub mod weights;

pub use forward::{forward, generate_greedy};
pub use weights::TransformerWeights;
