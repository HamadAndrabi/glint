//! Error types for Glint.
//!
//! `GlintError` covers failures that can occur at model load time —
//! missing tensors, unsupported formats, or bad metadata. These are
//! returned as `Result` so callers can print a clean message instead of
//! crashing with a Rust panic backtrace.

use std::fmt;

/// Errors that can occur while loading or running a Glint model.
#[derive(Debug)]
pub enum GlintError {
    /// A required weight tensor was not found in the GGUF file.
    TensorNotFound(String),
    /// A tensor had an unexpected number of dimensions (we support 1-D and 2-D).
    InvalidTensorShape { name: String, ndim: usize },
    /// Reading raw tensor bytes from the GGUF memory map failed.
    TensorReadError { name: String, detail: String },
    /// The model uses a quantization format Glint does not yet support.
    UnsupportedQuantization(String),
    /// The GGUF file is missing the `tokenizer.ggml.tokens` vocabulary.
    MissingVocabulary,
    /// The GGUF metadata does not contain a recognised model architecture.
    MissingModelConfig,
    /// No compatible GPU adapter was found (Vulkan/Metal/DX12).
    #[cfg(feature = "vulkan")]
    GpuAdapterNotFound,
    /// Failed to obtain a GPU device or queue.
    #[cfg(feature = "vulkan")]
    GpuDeviceError(String),
    /// A GPU buffer operation (upload, download, map) failed.
    #[cfg(feature = "vulkan")]
    GpuBufferError(String),
    /// A GPU compute shader failed to compile or execute.
    #[cfg(feature = "vulkan")]
    GpuShaderError(String),
}

impl fmt::Display for GlintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorNotFound(name) => {
                write!(f, "tensor '{name}' not found in model")
            }
            Self::InvalidTensorShape { name, ndim } => {
                write!(f, "tensor '{name}' has unexpected {ndim}-D shape (expected 1-D or 2-D)")
            }
            Self::TensorReadError { name, detail } => {
                write!(f, "failed to read tensor '{name}': {detail}")
            }
            Self::UnsupportedQuantization(t) => {
                write!(
                    f,
                    "quantization format '{t}' is not yet supported — \
                     load a Q4_K_M, Q5_K_M, Q6_K, Q8_0, or Q4_0 model"
                )
            }
            Self::MissingVocabulary => {
                write!(f, "model metadata is missing 'tokenizer.ggml.tokens'")
            }
            Self::MissingModelConfig => {
                write!(f, "could not extract model configuration from GGUF metadata")
            }
            #[cfg(feature = "vulkan")]
            Self::GpuAdapterNotFound => {
                write!(f, "no compatible GPU adapter found (need Vulkan, Metal, or DX12)")
            }
            #[cfg(feature = "vulkan")]
            Self::GpuDeviceError(msg) => write!(f, "GPU device error: {msg}"),
            #[cfg(feature = "vulkan")]
            Self::GpuBufferError(msg) => write!(f, "GPU buffer error: {msg}"),
            #[cfg(feature = "vulkan")]
            Self::GpuShaderError(msg) => write!(f, "GPU shader error: {msg}"),
        }
    }
}

impl std::error::Error for GlintError {}
