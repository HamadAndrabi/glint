//! Compute backends — CPU (default), GPU via wgpu (Vulkan/Metal/DX12).
//!
//! Enable the `vulkan` feature to compile the GPU backend:
//!
//! ```bash
//! cargo build --release --features vulkan
//! ```

#[cfg(feature = "vulkan")]
pub mod gpu;

#[cfg(feature = "vulkan")]
pub mod pipeline;

#[cfg(feature = "vulkan")]
pub use gpu::GpuBackend;

/// Stub type when `vulkan` feature is disabled.
///
/// Allows `Option<&mut GpuBackend>` to appear in function signatures
/// unconditionally — it simply can never be `Some`.
#[cfg(not(feature = "vulkan"))]
pub struct GpuBackend;

