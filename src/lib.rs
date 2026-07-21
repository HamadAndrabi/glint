//! Glint — LLM Inference Engine

// Lint policy — these three fire pervasively in code where the idiomatic Rust
// alternative is *worse* than the flagged form, so they are allowed crate-wide
// with rationale rather than suppressed case by case:
//
// * `too_many_arguments` — the transformer forward/prefill entry points and the
//   HTTP route handlers thread many independent parameters (weights, config,
//   caches, GPU handle, LoRA, sampler options, …). Bundling them into structs
//   purely to satisfy the lint would add indirection without improving clarity.
// * `needless_range_loop` — the quantization kernels and flash-attention loops
//   index by block/lane position, and the index is the meaningful quantity
//   (it maps to a byte-layout offset). `for i in 0..n` reads clearer than a
//   zipped iterator chain here and keeps the math aligned with the ggml layout.
// * `result_large_err` — the server handlers return `Result<_, axum::Response>`;
//   `Response` is intentionally large and boxing it at every `?` would pessimise
//   the hot success path for no real benefit.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::result_large_err)]

pub mod api;
pub mod bench;
pub mod constrained;
pub mod error;
pub mod model;

pub mod backend;
pub mod cache;
pub mod sampling;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod tensor;
pub mod transformer;

#[cfg(feature = "cffi")]
pub mod ffi;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;
