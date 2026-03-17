//! Sampling strategies for token selection.
//!
//! Provides configurable sampling pipelines (temperature, top-k, top-p,
//! min-p, repetition penalty) to replace greedy argmax decoding.

mod sampler;

pub use sampler::{Sampler, SamplerConfig};
