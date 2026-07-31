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
    // ── SafeTensors / HuggingFace loading ─────────────────────────────────────
    /// A `.safetensors` file is malformed: bad header length, unparseable JSON
    /// header, or tensor offsets that do not describe the data region.
    SafeTensorsMalformed(String),
    /// A `.safetensors` tensor uses a dtype Glint cannot load.
    SafeTensorsUnsupportedDtype { name: String, dtype: String },
    /// A HuggingFace model directory is missing a file Glint needs.
    HfMissingFile { dir: String, file: String },
    /// A HuggingFace JSON file (`config.json`, `tokenizer.json`, …) is invalid.
    HfInvalidJson { file: String, detail: String },
    /// A HuggingFace `config.json` is missing a field Glint requires.
    HfMissingConfigField(&'static str),
    /// A HuggingFace model uses a feature Glint's forward pass cannot express.
    HfUnsupported(String),
    /// Reading a model file from disk failed.
    Io { path: String, detail: String },
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
    // ── Snapshot errors ───────────────────────────────────────────────────────
    /// The snapshot file does not begin with the expected magic bytes.
    SnapshotBadMagic,
    /// The snapshot format version is not supported by this build.
    SnapshotVersionUnsupported { found: u32, current: u32 },
    /// The snapshot's model hash does not match the loaded model.
    SnapshotModelMismatch { expected: u64, found: u64 },
    /// A metadata field in the snapshot does not match the loaded model.
    SnapshotMetaMismatch {
        field: &'static str,
        expected: u64,
        found: u64,
    },
    /// The snapshot data is truncated or otherwise malformed.
    SnapshotTruncated,
    /// The snapshot cache data cannot be imported into a cache with different dimensions.
    SnapshotCacheSizeMismatch {
        layer: usize,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for GlintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorNotFound(name) => {
                write!(f, "tensor '{name}' not found in model")
            }
            Self::InvalidTensorShape { name, ndim } => {
                write!(
                    f,
                    "tensor '{name}' has unexpected {ndim}-D shape (expected 1-D or 2-D)"
                )
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
                write!(
                    f,
                    "could not extract model configuration from GGUF metadata"
                )
            }
            Self::SafeTensorsMalformed(detail) => {
                write!(f, "malformed safetensors file: {detail}")
            }
            Self::SafeTensorsUnsupportedDtype { name, dtype } => {
                write!(
                    f,
                    "tensor '{name}' has dtype '{dtype}' — Glint can load F32, F16, and BF16"
                )
            }
            Self::HfMissingFile { dir, file } => {
                write!(f, "HuggingFace model directory '{dir}' has no '{file}'")
            }
            Self::HfInvalidJson { file, detail } => {
                write!(f, "could not parse '{file}': {detail}")
            }
            Self::HfMissingConfigField(field) => {
                write!(f, "config.json is missing the required field '{field}'")
            }
            Self::HfUnsupported(detail) => {
                write!(f, "unsupported HuggingFace model: {detail}")
            }
            Self::Io { path, detail } => {
                write!(f, "could not read '{path}': {detail}")
            }
            #[cfg(feature = "vulkan")]
            Self::GpuAdapterNotFound => {
                write!(
                    f,
                    "no compatible GPU adapter found (need Vulkan, Metal, or DX12)"
                )
            }
            #[cfg(feature = "vulkan")]
            Self::GpuDeviceError(msg) => write!(f, "GPU device error: {msg}"),
            #[cfg(feature = "vulkan")]
            Self::GpuBufferError(msg) => write!(f, "GPU buffer error: {msg}"),
            #[cfg(feature = "vulkan")]
            Self::GpuShaderError(msg) => write!(f, "GPU shader error: {msg}"),
            Self::SnapshotBadMagic => {
                write!(f, "snapshot: invalid magic bytes (not a Glint snapshot)")
            }
            Self::SnapshotVersionUnsupported { found, current } => {
                write!(
                    f,
                    "snapshot version {found} is not supported (current: {current})"
                )
            }
            Self::SnapshotModelMismatch { expected, found } => {
                write!(
                    f,
                    "snapshot model hash mismatch: expected {expected:#018x}, found {found:#018x}"
                )
            }
            Self::SnapshotMetaMismatch {
                field,
                expected,
                found,
            } => {
                write!(
                    f,
                    "snapshot metadata mismatch in '{field}': expected {expected}, found {found}"
                )
            }
            Self::SnapshotTruncated => write!(f, "snapshot data is truncated or malformed"),
            Self::SnapshotCacheSizeMismatch {
                layer,
                expected,
                found,
            } => {
                write!(
                    f,
                    "snapshot cache layer {layer}: expected {expected} bytes, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for GlintError {}
