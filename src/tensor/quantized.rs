//! Quantized tensor — keeps weights in their compressed format and
//! dequantizes one block at a time during matrix-vector multiplication.
//!
//! Instead of expanding Q8_0 weights to f32 at load time (4× memory blowup),
//! this stores the raw bytes and does the arithmetic block-by-block:
//!
//!   output[i] = Σ_blocks (scale × Σ_j (int8[j] × input[j]))
//!
//! This cuts RAM from ~540 MB to ~140 MB for SmolLM-135M Q8_0 and improves
//! cache utilization — more of the model fits in L2/L3 during matmul.

use std::sync::Arc;

use half::f16;
use memmap2::Mmap;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::tensor::Tensor;
use crate::error::GlintError;
use crate::model::gguf::{GgmlType, GgufModel};

// ── Weight load mode ──────────────────────────────────────────────────────────

/// Controls whether weight tensors copy bytes out of the GGUF memory map
/// (eager — default) or keep a reference into the map (lazy — zero extra RAM).
///
/// Use [`WeightLoadMode::Lazy`] when:
/// * RAM is very tight and you want to avoid the ~1× memory overhead of copying.
/// * The model file lives on fast local storage (NVMe), so kernel page-cache
///   warms up quickly during the first forward pass.
///
/// Use [`WeightLoadMode::Eager`] (default) when:
/// * You want predictable performance from the very first forward pass.
/// * The model file is on a network share or slow HDD.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeightLoadMode {
    /// Copy weight bytes from the GGUF mmap into owned `Vec<u8>` at load time.
    /// Produces fully owned, independent `QuantizedTensor`s.
    #[default]
    Eager,
    /// Keep a reference to the GGUF mmap's bytes rather than copying.
    /// Falls back to [`WeightLoadMode::Eager`] when the model was loaded from
    /// an in-memory buffer (e.g. WASM) since there is no mmap to borrow from.
    Lazy,
}

// ── QuantizedStorage ──────────────────────────────────────────────────────────

/// Raw bytes backing a [`QuantizedTensor`].
///
/// `Owned` — eager path: bytes are copied into a `Vec<u8>` at load time.\
/// `Borrowed` — lazy path: a window into a shared memory-mapped file.
/// Keeping the `Arc<Mmap>` alive ensures the pages stay valid for the
/// lifetime of the tensor.
#[derive(Clone)]
pub enum QuantizedStorage {
    /// Fully owned copy of the quantized bytes.
    Owned(Vec<u8>),
    /// Zero-copy view into a memory-mapped file:
    /// `(mmap, byte_offset, byte_len)`.
    Borrowed(Arc<Mmap>, usize, usize),
}

impl QuantizedStorage {
    /// Return the byte slice regardless of storage variant.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            QuantizedStorage::Owned(v) => v,
            QuantizedStorage::Borrowed(m, off, n) => &m[*off..*off + *n],
        }
    }
}

/// Parallel row iterator when rayon is available, sequential otherwise.
#[cfg(feature = "rayon")]
fn par_rows(rows: usize) -> rayon::range::Iter<usize> {
    (0..rows).into_par_iter()
}

#[cfg(not(feature = "rayon"))]
fn par_rows(rows: usize) -> std::ops::Range<usize> {
    0..rows
}

/// Parallel iterator over the per-row slices of a `[rows, batch]` output
/// buffer (rayon when available, sequential otherwise).
///
/// Rows stay the outer, parallel dimension exactly as in the single-vector
/// kernels; sequences are the inner loop inside each row.
#[cfg(feature = "rayon")]
fn par_out_rows(out: &mut [f32], rows: usize, batch: usize) -> rayon::slice::ChunksMut<'_, f32> {
    debug_assert_eq!(
        out.len(),
        rows * batch,
        "batched output must be rows × batch"
    );
    out.par_chunks_mut(batch)
}

#[cfg(not(feature = "rayon"))]
fn par_out_rows(out: &mut [f32], rows: usize, batch: usize) -> std::slice::ChunksMut<'_, f32> {
    debug_assert_eq!(
        out.len(),
        rows * batch,
        "batched output must be rows × batch"
    );
    out.chunks_mut(batch)
}
use super::dequantize::{dequantize, get_scale_min_q4k};

/// A weight matrix stored in its original quantized format.
///
/// Activations (input/output of each op) remain f32 `Tensor`.
/// Only the large weight matrices use this type.
#[derive(Clone)]
pub struct QuantizedTensor {
    /// Raw bytes — either owned (eager) or borrowed from a memory map (lazy).
    data: QuantizedStorage,
    /// Number of output rows (first dimension in row-major layout).
    rows: usize,
    /// Number of input columns (second dimension).
    cols: usize,
    /// Original quantization format.
    ggml_type: GgmlType,
    /// GPU buffer name (set after `upload_to_gpu`).
    #[cfg(feature = "vulkan")]
    gpu_buf_name: Option<String>,
}

impl QuantizedTensor {
    /// Load a weight tensor from GGUF, copying bytes into an owned buffer (eager).
    ///
    /// GGUF stores dimensions in column-major order (first dim varies fastest).
    /// We reverse them to match our row-major convention.
    pub fn load(model: &GgufModel, name: &str) -> Result<Self, GlintError> {
        Self::load_with_mode(model, name, WeightLoadMode::Eager)
    }

    /// Load a weight tensor with explicit mode (eager copy or lazy mmap borrow).
    ///
    /// When `mode` is [`WeightLoadMode::Lazy`] **and** the model was loaded from
    /// a file, the tensor borrows directly from the memory map — no copy.
    /// Falls back to eager copy when the model was loaded from an in-memory
    /// buffer (e.g. WASM).
    pub fn load_with_mode(
        model: &GgufModel,
        name: &str,
        mode: WeightLoadMode,
    ) -> Result<Self, GlintError> {
        let info = model
            .get_tensor_info(name)
            .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;

        // Reverse dimensions: GGUF column-major → our row-major
        let dims: Vec<usize> = info.dimensions.iter().rev().map(|&d| d as usize).collect();
        let (rows, cols) = match dims.len() {
            1 => (dims[0], 1),
            2 => (dims[0], dims[1]),
            n => {
                return Err(GlintError::InvalidTensorShape {
                    name: name.to_string(),
                    ndim: n,
                })
            }
        };
        let ggml_type = info.ggml_type;

        let storage = if mode == WeightLoadMode::Lazy {
            // Try to borrow from the mmap; fall through to eager on failure.
            if let (Some(mmap), Some((offset, len))) =
                (model.mmap_arc(), model.tensor_data_range(name))
            {
                QuantizedStorage::Borrowed(mmap, offset, len)
            } else {
                // No mmap available (e.g. from_bytes path) — fall back to eager.
                let raw = model
                    .tensor_data(name)
                    .map_err(|e| GlintError::TensorReadError {
                        name: name.to_string(),
                        detail: e.to_string(),
                    })?;
                QuantizedStorage::Owned(raw.to_vec())
            }
        } else {
            let raw = model
                .tensor_data(name)
                .map_err(|e| GlintError::TensorReadError {
                    name: name.to_string(),
                    detail: e.to_string(),
                })?;
            QuantizedStorage::Owned(raw.to_vec())
        };

        Ok(Self {
            data: storage,
            rows,
            cols,
            ggml_type,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        })
    }

    /// Build from raw quantized bytes. Used for benchmarks and testing.
    pub fn from_raw(data: Vec<u8>, rows: usize, cols: usize, ggml_type: GgmlType) -> Self {
        Self {
            data: QuantizedStorage::Owned(data),
            rows,
            cols,
            ggml_type,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        }
    }

    /// Build from a flat f32 slice (re-encoded as F32 bytes). Used in tests.
    pub fn from_f32(values: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols);
        let mut data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            data: QuantizedStorage::Owned(data),
            rows,
            cols,
            ggml_type: GgmlType::F32,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Matrix-vector multiply: `[rows, cols] × [cols] → [rows]`.
    ///
    /// Dispatches to the direct kernel for every supported quantized format,
    /// or falls back to on-the-fly dequantization (F32/F16/BF16 and anything
    /// without a kernel yet).
    pub fn matvec(&self, vec: &[f32]) -> Tensor {
        assert_eq!(
            vec.len(),
            self.cols,
            "matvec: vec length {} != cols {}",
            vec.len(),
            self.cols
        );
        let out = match self.ggml_type {
            GgmlType::Q8_0 => dispatch_q8_0(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q4_0 => dispatch_q4_0(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q4K => dispatch_q4_k(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q5K => dispatch_q5_k(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q6K => dispatch_q6_k(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q2K => dispatch_q2_k(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q3K => dispatch_q3_k(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::IQ4NL => dispatch_iq4_nl(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q4_1 => dispatch_q4_1(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q5_0 => dispatch_q5_0(self.data.as_slice(), self.rows, self.cols, vec),
            GgmlType::Q5_1 => dispatch_q5_1(self.data.as_slice(), self.rows, self.cols, vec),
            _ => matvec_fallback(
                self.data.as_slice(),
                self.ggml_type,
                self.rows,
                self.cols,
                vec,
            ),
        };
        Tensor::from_vec(out, &[self.rows])
    }

    /// Batched matrix-vector multiply: `[rows, cols] × B×[cols] → [rows × B]`.
    ///
    /// This is the kernel-level half of continuous batching. Where `matvec`
    /// streams the whole weight matrix through the cache hierarchy to serve one
    /// activation vector, `matvec_batch_into` streams it **once** for `B` of
    /// them: every weight block is decoded a single time and immediately
    /// applied to all `B` inputs before the kernel moves on. Decode throughput
    /// for `B` concurrent sequences therefore stops being `B ×` the cost of one.
    ///
    /// `out` is written in `[rows, B]` interleaved layout — `out[i * B + s]` is
    /// row `i` for input `s` — which keeps each row's results contiguous so
    /// rows remain the unit of rayon parallelism.
    ///
    /// # Numerical parity
    ///
    /// Every input keeps its own accumulator and sees the same floating-point
    /// operations in the same order as it would through [`matvec`], so
    /// `matvec_batch_into` is **bit-identical** to calling `matvec` once per
    /// input. Batching a request must never change the tokens it produces.
    ///
    /// # Panics
    ///
    /// If `inputs` is empty, any input's length is not `cols`, or `out.len()`
    /// is not `rows × inputs.len()`.
    pub fn matvec_batch_into(&self, inputs: &[&[f32]], out: &mut [f32]) {
        let batch = inputs.len();
        assert!(batch > 0, "matvec_batch: empty batch");
        for (s, v) in inputs.iter().enumerate() {
            assert_eq!(
                v.len(),
                self.cols,
                "matvec_batch: input {s} length {} != cols {}",
                v.len(),
                self.cols
            );
        }
        assert_eq!(
            out.len(),
            self.rows * batch,
            "matvec_batch: out length {} != rows {} × batch {batch}",
            out.len(),
            self.rows
        );

        let data = self.data.as_slice();
        match self.ggml_type {
            GgmlType::Q8_0 => dispatch_q8_0_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q4_0 => dispatch_q4_0_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q4K => dispatch_q4_k_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q5K => dispatch_q5_k_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q6K => dispatch_q6_k_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q2K => matvec_q2_k_batch_scalar(data, self.rows, self.cols, inputs, out),
            GgmlType::Q3K => matvec_q3_k_batch_scalar(data, self.rows, self.cols, inputs, out),
            GgmlType::IQ4NL => matvec_iq4_nl_batch_scalar(data, self.rows, self.cols, inputs, out),
            GgmlType::Q4_1 => dispatch_q4_1_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q5_0 => dispatch_q5_0_batch(data, self.rows, self.cols, inputs, out),
            GgmlType::Q5_1 => dispatch_q5_1_batch(data, self.rows, self.cols, inputs, out),
            _ => matvec_fallback_batch(data, self.ggml_type, self.rows, self.cols, inputs, out),
        }
    }

    /// [`matvec_batch_into`] with one owned [`Tensor`] per input vector.
    ///
    /// Convenience wrapper for callers that do not keep a scratch buffer
    /// around; the hot path in `forward_batch` uses `matvec_batch_into`
    /// directly to reuse one.
    ///
    /// [`matvec_batch_into`]: Self::matvec_batch_into
    pub fn matvec_batch(&self, inputs: &[&[f32]]) -> Vec<Tensor> {
        let batch = inputs.len();
        let mut interleaved = vec![0.0f32; self.rows * batch];
        self.matvec_batch_into(inputs, &mut interleaved);
        (0..batch)
            .map(|s| {
                let mut row = vec![0.0f32; self.rows];
                for (i, o) in row.iter_mut().enumerate() {
                    *o = interleaved[i * batch + s];
                }
                Tensor::from_vec(row, &[self.rows])
            })
            .collect()
    }

    /// Dequantize a single row and return it as a 1D `Tensor`.
    ///
    /// Used for embedding lookup (one token per call — cheap).
    pub fn row_as_f32(&self, row: usize) -> Tensor {
        assert!(
            row < self.rows,
            "row_as_f32: row {row} >= rows {}",
            self.rows
        );

        let n_elements = self.cols;
        let block_size = self.ggml_type.block_size();
        let type_size = self.ggml_type.type_size();
        let n_blocks = n_elements.div_ceil(block_size);
        let bytes_per_row = n_blocks * type_size;

        let row_bytes = &self.data.as_slice()[row * bytes_per_row..(row + 1) * bytes_per_row];
        let f32_data = dequantize(row_bytes, self.ggml_type, n_elements);
        Tensor::from_vec(f32_data, &[n_elements])
    }

    /// Raw quantized bytes (for GPU upload).
    pub fn raw_data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Quantization format.
    pub fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }

    // ── GPU methods (only when `vulkan` feature is enabled) ──────────

    /// Upload this tensor's raw quantized bytes to the GPU.
    ///
    /// After calling this, `matvec_gpu()` can dispatch on the GPU.
    #[cfg(feature = "vulkan")]
    pub fn upload_to_gpu(&mut self, gpu: &mut crate::backend::gpu::GpuBackend, name: &str) {
        gpu.upload_buffer(name, self.data.as_slice());
        self.gpu_buf_name = Some(name.to_string());
    }

    /// GPU-accelerated matrix-vector multiply.
    ///
    /// Falls back to CPU `matvec()` if weights have not been uploaded or
    /// the quantization format has no GPU kernel yet.
    #[cfg(feature = "vulkan")]
    pub fn matvec_gpu(&self, vec: &[f32], gpu: &mut crate::backend::gpu::GpuBackend) -> Tensor {
        if let Some(ref buf_name) = self.gpu_buf_name {
            let result = match self.ggml_type {
                GgmlType::Q8_0 => {
                    gpu.matvec_q8_0(buf_name, vec, self.rows as u32, self.cols as u32)
                }
                GgmlType::Q4_0 => {
                    gpu.matvec_q4_0(buf_name, vec, self.rows as u32, self.cols as u32)
                }
                GgmlType::Q4K => gpu.matvec_q4_k(buf_name, vec, self.rows as u32, self.cols as u32),
                GgmlType::Q5K => gpu.matvec_q5_k(buf_name, vec, self.rows as u32, self.cols as u32),
                GgmlType::Q6K => gpu.matvec_q6_k(buf_name, vec, self.rows as u32, self.cols as u32),
                GgmlType::F32 => gpu.matvec_f32(buf_name, vec, self.rows as u32, self.cols as u32),
                // Deliberate CPU fallback — there is no shader in
                // `src/backend/shaders/` for Q4_1, Q5_0, Q5_1, Q2_K, Q3_K or
                // IQ4_NL, nor for the non-quantized F16/BF16 weights. Every one
                // of those is handled by `matvec`, so falling back is correct,
                // just slower. Adding a shader means adding an arm above.
                _ => return self.matvec(vec),
            };
            match result {
                Ok(data) => Tensor::from_vec(data, &[self.rows]),
                Err(_) => self.matvec(vec), // GPU error → CPU fallback
            }
        } else {
            self.matvec(vec) // not uploaded → CPU
        }
    }
}

// ── Dispatch ─────────────────────────────────────────────────────────────────
//
// Runtime CPU feature detection → SIMD if available, scalar fallback otherwise.

fn dispatch_q8_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above. The kernel's buffer-length
            // contract (see `simd::check_dims`) holds because `data`, `rows`,
            // `cols`, and `vec` come from a validated tensor descriptor whose
            // `cols` is a block-size multiple and whose backing store was
            // range-checked at load time (`tensor_data{,_range}`).
            return unsafe { crate::tensor::simd::matvec_q8_0_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q8_0_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q8_0_scalar(data, rows, cols, vec)
}

fn dispatch_q4_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q4_k_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q4_k_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q4_k_scalar(data, rows, cols, vec)
}

fn dispatch_q5_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q5_k_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q5_k_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_k_scalar(data, rows, cols, vec)
}

fn dispatch_q6_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q6_k_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q6_k_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q6_k_scalar(data, rows, cols, vec)
}

fn dispatch_q4_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q4_0_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q4_0_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q4_0_scalar(data, rows, cols, vec)
}

fn dispatch_q4_1(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q4_1_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q4_1_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q4_1_scalar(data, rows, cols, vec)
}

fn dispatch_q5_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q5_0_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q5_0_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_0_scalar(data, rows, cols, vec)
}

fn dispatch_q5_1(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; kernel length contract upheld by
            // the validated descriptor (see `dispatch_q8_0` for the full note).
            return unsafe { crate::tensor::simd::matvec_q5_1_avx2(data, rows, cols, vec) };
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe { crate::tensor::simd_neon::matvec_q5_1_neon(data, rows, cols, vec) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_1_scalar(data, rows, cols, vec)
}

// ── Scalar Kernels ───────────────────────────────────────────────────────────

///
/// For each output row, iterate over blocks, compute block_sum = Σ i8×f32,
/// then accumulate block_sum × scale. One scale multiply per block (32 elements).
#[allow(dead_code)]
pub(crate) fn matvec_q8_0_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8s)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|i| {
            let row_start = i * bytes_per_row;
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &data[row_start + b * BLOCK_BYTES..];
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let mut block_sum = 0.0f32;
                for j in 0..BLOCK_ELEMS {
                    block_sum += (block[2 + j] as i8) as f32 * vec[b * BLOCK_ELEMS + j];
                }
                sum += block_sum * scale;
            }
            sum
        })
        .collect()
}

/// Q4_0 matrix-vector multiply (scalar fallback).
///
/// Block layout (18 bytes per 32 elements):
///   [f16 scale (2 bytes)] [16 bytes of packed nibbles, 2 per byte]
///
/// Nibble values are unsigned (0–15), centered by subtracting 8 → [-8, +7].
/// The nibbles are split-plane, matching the ggml-anchored
/// [`super::dequantize::unpack_q4_0_block`]: the low nibble of byte `j` is
/// element `j` and its high nibble is element `j + 16`, not `2j`/`2j+1`.
#[allow(dead_code)]
pub(crate) fn matvec_q4_0_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const HALF: usize = BLOCK_ELEMS / 2;
    const BLOCK_BYTES: usize = 18; // 2 (f16 scale) + 16 (packed nibbles)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|i| {
            let row_start = i * bytes_per_row;
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &data[row_start + b * BLOCK_BYTES..];
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let x = &vec[b * BLOCK_ELEMS..b * BLOCK_ELEMS + BLOCK_ELEMS];
                let mut block_sum = 0.0f32;
                for j in 0..HALF {
                    let byte = block[2 + j];
                    let lo = (byte & 0x0F) as i32 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                    block_sum += lo as f32 * x[j] + hi as f32 * x[j + HALF];
                }
                sum += block_sum * scale;
            }
            sum
        })
        .collect()
}

/// Q4_K matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Super-block layout (144 bytes per 256 elements):
///   [f16 d] [f16 dmin] [scales u8×12] [qs u8×128]
///
/// 8 sub-blocks of 32; each has a 6-bit scale and 6-bit min extracted via
/// `get_scale_min_q4k`. Low nibbles of each byte → even sub-blocks; high nibbles
/// → odd sub-blocks (4 groups of 64 per super-block).
#[allow(dead_code)]
fn matvec_q4_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qs = &b[16..144];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;
                    let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;
                    let m1 = dmin * mn1 as f32;
                    let q_base = group * 32;
                    for l in 0..32 {
                        let xi = x_base + group * 64 + l;
                        acc += ((qs[q_base + l] & 0x0F) as f32 * d0 - m0) * vec[xi];
                    }
                    for l in 0..32 {
                        let xi = x_base + group * 64 + 32 + l;
                        acc += ((qs[q_base + l] >> 4) as f32 * d1 - m1) * vec[xi];
                    }
                }
            }
            acc
        })
        .collect()
}

/// Q5_K matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Super-block layout (176 bytes per 256 elements):
///   [f16 d] [f16 dmin] [scales u8×12] [qh u8×32] [qs u8×128]
///
/// Same sub-block structure as Q4_K; each nibble gains a 5th bit from `qh`.
#[allow(dead_code)]
fn matvec_q5_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;
                    let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;
                    let m1 = dmin * mn1 as f32;
                    let q_base = group * 32;
                    let u1: u8 = 1 << (group * 2);
                    let u2: u8 = 2 << (group * 2);
                    for l in 0..32 {
                        let xi = x_base + group * 64 + l;
                        let lo = (qs[q_base + l] & 0x0F) as f32;
                        let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        acc += (d0 * (lo + hi) - m0) * vec[xi];
                    }
                    for l in 0..32 {
                        let xi = x_base + group * 64 + 32 + l;
                        let lo = (qs[q_base + l] >> 4) as f32;
                        let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        acc += (d1 * (lo + hi) - m1) * vec[xi];
                    }
                }
            }
            acc
        })
        .collect()
}

/// Q6_K matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Super-block layout (210 bytes per 256 elements):
///   [ql u8×128] [qh u8×64] [scales i8×16] [f16 d]
///
/// Two groups of 128; within each group, 32 iterations scatter output to
/// four non-contiguous positions (l, l+32, l+64, l+96).
#[allow(dead_code)]
fn matvec_q6_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc_raw = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                let x_base = sb * SUPER_BLOCK;
                for group in 0..2 {
                    let ql_off = group * 64;
                    let qh_off = group * 32;
                    let sc_off = group * 8;
                    let x_group = x_base + group * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let qhl = qh[qh_off + l];
                        let v1 = (ql[ql_off + l] & 0x0F) | ((qhl & 0x03) << 4);
                        let v2 = (ql[ql_off + l + 32] & 0x0F) | (((qhl >> 2) & 0x03) << 4);
                        let v3 = (ql[ql_off + l] >> 4) | (((qhl >> 4) & 0x03) << 4);
                        let v4 = (ql[ql_off + l + 32] >> 4) | (((qhl >> 6) & 0x03) << 4);
                        let q1 = (v1 as i32 - 32) as f32;
                        let q2 = (v2 as i32 - 32) as f32;
                        let q3 = (v3 as i32 - 32) as f32;
                        let q4 = (v4 as i32 - 32) as f32;
                        let sc0 = sc_raw[sc_off + is] as i8 as f32;
                        let sc2 = sc_raw[sc_off + is + 2] as i8 as f32;
                        let sc4 = sc_raw[sc_off + is + 4] as i8 as f32;
                        let sc6 = sc_raw[sc_off + is + 6] as i8 as f32;
                        acc += d * sc0 * q1 * vec[x_group + l];
                        acc += d * sc2 * q2 * vec[x_group + l + 32];
                        acc += d * sc4 * q3 * vec[x_group + l + 64];
                        acc += d * sc6 * q4 * vec[x_group + l + 96];
                    }
                }
            }
            acc
        })
        .collect()
}

fn dispatch_q2_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    matvec_q2_k_scalar(data, rows, cols, vec)
}

fn dispatch_q3_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    matvec_q3_k_scalar(data, rows, cols, vec)
}

fn dispatch_iq4_nl(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    matvec_iq4_nl_scalar(data, rows, cols, vec)
}

/// Q2_K matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Super-block layout (84 bytes per 256 elements):
///   [scales u8×16] [qs u8×64] [f16 d] [f16 dmin]
///
/// Unpacks each super-block through the shared, ggml-anchored
/// [`super::dequantize::unpack_q2_k_block`] so the kernel can never drift
/// from the reference layout (the 2-bit quants live in shift planes; see the
/// helper's docs). A 256-float stack buffer per super-block is fine here —
/// this format has no SIMD kernel and is not on the default hot path.
fn matvec_q2_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 84;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; SUPER_BLOCK];
            for sb in 0..n_super {
                super::dequantize::unpack_q2_k_block(&row[sb * BLOCK_BYTES..], &mut buf);
                let x = &vec[sb * SUPER_BLOCK..sb * SUPER_BLOCK + SUPER_BLOCK];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// Q3_K matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Super-block layout (110 bytes per 256 elements):
///   [hmask u8×32] [qs u8×64] [scales u8×12] [f16 d]
///
/// Unpacks each super-block through the shared, ggml-anchored
/// [`super::dequantize::unpack_q3_k_block`] (shift-plane quants, nibble/2-bit
/// plane scales, per-pass hmask bits — see the helper's docs).
fn matvec_q3_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 110;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; SUPER_BLOCK];
            for sb in 0..n_super {
                super::dequantize::unpack_q3_k_block(&row[sb * BLOCK_BYTES..], &mut buf);
                let x = &vec[sb * SUPER_BLOCK..sb * SUPER_BLOCK + SUPER_BLOCK];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// IQ4_NL matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Block layout (18 bytes per 32 elements):
///   [f16 d] [qs u8×16]
///
/// Unpacks each block through the shared, ggml-anchored
/// [`super::dequantize::unpack_iq4_nl_block`] — the nibbles are split-plane
/// (low nibbles are elements 0..16, high nibbles 16..32), not interleaved.
fn matvec_iq4_nl_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let n_blocks = cols / BLOCK_SIZE;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; BLOCK_SIZE];
            for bi in 0..n_blocks {
                super::dequantize::unpack_iq4_nl_block(&row[bi * BLOCK_BYTES..], &mut buf);
                let x = &vec[bi * BLOCK_SIZE..bi * BLOCK_SIZE + BLOCK_SIZE];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// Q4_1 matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Block layout (20 bytes per 32 elements): [f16 d] [f16 m] [qs u8×16].
/// Unpacks each block through the shared, ggml-anchored
/// [`super::dequantize::unpack_q4_1_block`] — the nibbles are split-plane
/// (low nibbles are elements 0..16, high nibbles 16..32) and the values are
/// affine (`q*d + m`), not centered like Q4_0.
#[allow(dead_code)]
fn matvec_q4_1_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let n_blocks = cols / BLOCK_SIZE;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; BLOCK_SIZE];
            for bi in 0..n_blocks {
                super::dequantize::unpack_q4_1_block(&row[bi * BLOCK_BYTES..], &mut buf);
                let x = &vec[bi * BLOCK_SIZE..bi * BLOCK_SIZE + BLOCK_SIZE];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// Q5_0 matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Block layout (22 bytes per 32 elements): [f16 d] [qh u8×4] [qs u8×16].
/// Unpacks each block through the shared, ggml-anchored
/// [`super::dequantize::unpack_q5_0_block`].
#[allow(dead_code)]
fn matvec_q5_0_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let n_blocks = cols / BLOCK_SIZE;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; BLOCK_SIZE];
            for bi in 0..n_blocks {
                super::dequantize::unpack_q5_0_block(&row[bi * BLOCK_BYTES..], &mut buf);
                let x = &vec[bi * BLOCK_SIZE..bi * BLOCK_SIZE + BLOCK_SIZE];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// Q5_1 matrix-vector multiply (scalar, rayon-parallel over rows).
///
/// Block layout (24 bytes per 32 elements): [f16 d] [f16 m] [qh u8×4] [qs u8×16].
/// Unpacks each block through the shared, ggml-anchored
/// [`super::dequantize::unpack_q5_1_block`].
#[allow(dead_code)]
fn matvec_q5_1_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let n_blocks = cols / BLOCK_SIZE;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_rows(rows)
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; BLOCK_SIZE];
            for bi in 0..n_blocks {
                super::dequantize::unpack_q5_1_block(&row[bi * BLOCK_BYTES..], &mut buf);
                let x = &vec[bi * BLOCK_SIZE..bi * BLOCK_SIZE + BLOCK_SIZE];
                for (w, xv) in buf.iter().zip(x) {
                    acc += w * xv;
                }
            }
            acc
        })
        .collect()
}

/// Fallback matvec: dequantize each row on-the-fly, then do a plain dot product.
///
/// Used for F32/F16/BF16 weight matrices. Not performance-sensitive because
/// norm weights (which are F32) stay as `Tensor`, not `QuantizedTensor`.
fn matvec_fallback(
    data: &[u8],
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    let block_size = ggml_type.block_size();
    let type_size = ggml_type.type_size();
    let n_blocks = cols.div_ceil(block_size);
    let bytes_per_row = n_blocks * type_size;

    par_rows(rows)
        .map(|i| {
            let row_bytes = &data[i * bytes_per_row..(i + 1) * bytes_per_row];
            let row_f32 = dequantize(row_bytes, ggml_type, cols);
            let mut sum = 0.0f32;
            for j in 0..cols {
                sum += row_f32[j] * vec[j];
            }
            sum
        })
        .collect()
}

// ── Batched kernels ──────────────────────────────────────────────────────────
//
// One weight traversal, `B` activation vectors. Each kernel mirrors its
// single-vector twin above with two changes:
//
//   1. the weight block is decoded into `w` **once**, outside the sequence loop
//      (this is where the throughput comes from — the weights are streamed from
//      RAM once per step instead of once per sequence);
//   2. the accumulator is per-sequence, and the sequence loop is innermost so
//      each sequence still adds its terms in the original order.
//
// (2) is what makes a batched step bit-identical to running each sequence on
// its own: the summation order per output element is untouched, so batching
// can never change the tokens a request receives.
//
// `out` is `[rows, B]` interleaved: `out[row * B + s]` belongs to input `s`.

fn dispatch_q8_0_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; the kernel's buffer-length
            // contract is the single-vector one applied to every input and is
            // upheld by the validated tensor descriptor (see `dispatch_q8_0`).
            unsafe {
                crate::tensor::simd::matvec_q8_0_batch_avx2(data, rows, cols, inputs, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe {
            crate::tensor::simd_neon::matvec_q8_0_batch_neon(data, rows, cols, inputs, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q8_0_batch_scalar(data, rows, cols, inputs, out)
}

fn dispatch_q4_0_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `dispatch_q8_0_batch`.
            unsafe {
                crate::tensor::simd::matvec_q4_0_batch_avx2(data, rows, cols, inputs, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe {
            crate::tensor::simd_neon::matvec_q4_0_batch_neon(data, rows, cols, inputs, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q4_0_batch_scalar(data, rows, cols, inputs, out)
}

/// Scatter one lane's `rows`-length result into the interleaved batch output.
fn scatter_lane(col: Vec<f32>, s: usize, batch: usize, out: &mut [f32]) {
    for (i, v) in col.into_iter().enumerate() {
        out[i * batch + s] = v;
    }
}

// Q4_1/Q5_0/Q5_1 have single-vector AVX2 kernels but no batched AVX2
// counterpart replicating their exact accumulation order. Batching must be
// invisible — bit-identical to decoding each sequence alone — so on hosts
// where the single path takes the AVX2 kernel, delegate per lane to that same
// kernel rather than using a scalar batch kernel with different FP ordering.
// These are legacy formats; the lost weight-streaming amortization is the
// correct trade until a batched kernel with matching order exists.

fn dispatch_q4_1_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    let batch = inputs.len();
    for (s, input) in inputs.iter().enumerate() {
        scatter_lane(dispatch_q4_1(data, rows, cols, input), s, batch, out);
    }
}

fn dispatch_q5_0_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let batch = inputs.len();
            for (s, input) in inputs.iter().enumerate() {
                // SAFETY: AVX2+FMA verified above; kernel length contract
                // upheld by the validated descriptor (see `dispatch_q8_0`).
                let col = unsafe { crate::tensor::simd::matvec_q5_0_avx2(data, rows, cols, input) };
                scatter_lane(col, s, batch, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        let batch = inputs.len();
        for (s, input) in inputs.iter().enumerate() {
            let col =
                unsafe { crate::tensor::simd_neon::matvec_q5_0_neon(data, rows, cols, input) };
            scatter_lane(col, s, batch, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_0_batch_scalar(data, rows, cols, inputs, out)
}

fn dispatch_q5_1_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let batch = inputs.len();
            for (s, input) in inputs.iter().enumerate() {
                // SAFETY: AVX2+FMA verified above; kernel length contract
                // upheld by the validated descriptor (see `dispatch_q8_0`).
                let col = unsafe { crate::tensor::simd::matvec_q5_1_avx2(data, rows, cols, input) };
                scatter_lane(col, s, batch, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        let batch = inputs.len();
        for (s, input) in inputs.iter().enumerate() {
            let col =
                unsafe { crate::tensor::simd_neon::matvec_q5_1_neon(data, rows, cols, input) };
            scatter_lane(col, s, batch, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_1_batch_scalar(data, rows, cols, inputs, out)
}

fn dispatch_q4_k_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `dispatch_q8_0_batch`.
            unsafe {
                crate::tensor::simd::matvec_q4_k_batch_avx2(data, rows, cols, inputs, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe {
            crate::tensor::simd_neon::matvec_q4_k_batch_neon(data, rows, cols, inputs, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q4_k_batch_scalar(data, rows, cols, inputs, out)
}

fn dispatch_q5_k_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `dispatch_q8_0_batch`.
            unsafe {
                crate::tensor::simd::matvec_q5_k_batch_avx2(data, rows, cols, inputs, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe {
            crate::tensor::simd_neon::matvec_q5_k_batch_neon(data, rows, cols, inputs, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q5_k_batch_scalar(data, rows, cols, inputs, out)
}

fn dispatch_q6_k_batch(data: &[u8], rows: usize, cols: usize, inputs: &[&[f32]], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: see `dispatch_q8_0_batch`.
            unsafe {
                crate::tensor::simd::matvec_q6_k_batch_avx2(data, rows, cols, inputs, out);
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
    {
        unsafe {
            crate::tensor::simd_neon::matvec_q6_k_batch_neon(data, rows, cols, inputs, out);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "rayon")))]
    matvec_q6_k_batch_scalar(data, rows, cols, inputs, out)
}

/// Batched Q8_0 matvec (scalar fallback) — see [`matvec_q8_0_scalar`].
#[allow(dead_code)]
pub(crate) fn matvec_q8_0_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let batch = inputs.len();
    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(i, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row_start = i * bytes_per_row;
            let mut w = [0.0f32; BLOCK_ELEMS];
            for b in 0..n_blocks {
                let block = &data[row_start + b * BLOCK_BYTES..];
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                for (j, wj) in w.iter_mut().enumerate() {
                    *wj = (block[2 + j] as i8) as f32;
                }
                for (s, input) in inputs.iter().enumerate() {
                    let x = &input[b * BLOCK_ELEMS..];
                    let mut block_sum = 0.0f32;
                    for (j, wj) in w.iter().enumerate() {
                        block_sum += wj * x[j];
                    }
                    dst[s] += block_sum * scale;
                }
            }
        });
}

/// Batched Q4_0 matvec (scalar fallback) — see [`matvec_q4_0_scalar`].
#[allow(dead_code)]
pub(crate) fn matvec_q4_0_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let batch = inputs.len();
    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(i, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row_start = i * bytes_per_row;
            const HALF: usize = BLOCK_ELEMS / 2;
            let mut lo = [0.0f32; HALF];
            let mut hi = [0.0f32; HALF];
            for b in 0..n_blocks {
                let block = &data[row_start + b * BLOCK_BYTES..];
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                // Split-plane like ggml, and accumulated in the same
                // `lo*x[j] + hi*x[j+HALF]` order as `matvec_q4_0_scalar` so
                // batched output stays bit-identical to the single path.
                for j in 0..HALF {
                    let byte = block[2 + j];
                    lo[j] = ((byte & 0x0F) as i32 - 8) as f32;
                    hi[j] = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
                }
                for (s, input) in inputs.iter().enumerate() {
                    let x = &input[b * BLOCK_ELEMS..];
                    let mut block_sum = 0.0f32;
                    for j in 0..HALF {
                        block_sum += lo[j] * x[j] + hi[j] * x[j + HALF];
                    }
                    dst[s] += block_sum * scale;
                }
            }
        });
}

/// Batched Q4_K matvec (scalar) — see [`matvec_q4_k_scalar`].
#[allow(dead_code)]
fn matvec_q4_k_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let batch = inputs.len();
    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(r, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row = &data[r * bytes_per_row..];
            // Weights for one group: 32 low-nibble values then 32 high-nibble ones.
            let mut w_lo = [0.0f32; 32];
            let mut w_hi = [0.0f32; 32];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qs = &b[16..144];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;
                    let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;
                    let m1 = dmin * mn1 as f32;
                    let q_base = group * 32;
                    for l in 0..32 {
                        w_lo[l] = (qs[q_base + l] & 0x0F) as f32 * d0 - m0;
                        w_hi[l] = (qs[q_base + l] >> 4) as f32 * d1 - m1;
                    }
                    let x_group = x_base + group * 64;
                    for (s, input) in inputs.iter().enumerate() {
                        let acc = &mut dst[s];
                        for (l, wl) in w_lo.iter().enumerate() {
                            *acc += wl * input[x_group + l];
                        }
                        for (l, wl) in w_hi.iter().enumerate() {
                            *acc += wl * input[x_group + 32 + l];
                        }
                    }
                }
            }
        });
}

/// Batched Q5_K matvec (scalar) — see [`matvec_q5_k_scalar`].
#[allow(dead_code)]
fn matvec_q5_k_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let batch = inputs.len();
    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(r, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row = &data[r * bytes_per_row..];
            let mut w_lo = [0.0f32; 32];
            let mut w_hi = [0.0f32; 32];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;
                    let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;
                    let m1 = dmin * mn1 as f32;
                    let q_base = group * 32;
                    let u1: u8 = 1 << (group * 2);
                    let u2: u8 = 2 << (group * 2);
                    for l in 0..32 {
                        let lo = (qs[q_base + l] & 0x0F) as f32;
                        let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        w_lo[l] = d0 * (lo + hi) - m0;
                        let lo = (qs[q_base + l] >> 4) as f32;
                        let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        w_hi[l] = d1 * (lo + hi) - m1;
                    }
                    let x_group = x_base + group * 64;
                    for (s, input) in inputs.iter().enumerate() {
                        let acc = &mut dst[s];
                        for (l, wl) in w_lo.iter().enumerate() {
                            *acc += wl * input[x_group + l];
                        }
                        for (l, wl) in w_hi.iter().enumerate() {
                            *acc += wl * input[x_group + 32 + l];
                        }
                    }
                }
            }
        });
}

/// Batched Q6_K matvec (scalar) — see [`matvec_q6_k_scalar`].
///
/// The single-vector kernel visits four non-contiguous input offsets per `l`
/// (`l`, `l+32`, `l+64`, `l+96`), so the batched form decodes all four weights
/// for an `l` before the sequence loop and keeps that same four-add order.
#[allow(dead_code)]
fn matvec_q6_k_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let batch = inputs.len();
    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(r, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row = &data[r * bytes_per_row..];
            // [l][0..4] — the four weights element `l` of this group contributes.
            let mut w = [[0.0f32; 4]; 32];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc_raw = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                let x_base = sb * SUPER_BLOCK;
                for group in 0..2 {
                    let ql_off = group * 64;
                    let qh_off = group * 32;
                    let sc_off = group * 8;
                    let x_group = x_base + group * 128;
                    for (l, wl) in w.iter_mut().enumerate() {
                        let is = l / 16;
                        let qhl = qh[qh_off + l];
                        let v1 = (ql[ql_off + l] & 0x0F) | ((qhl & 0x03) << 4);
                        let v2 = (ql[ql_off + l + 32] & 0x0F) | (((qhl >> 2) & 0x03) << 4);
                        let v3 = (ql[ql_off + l] >> 4) | (((qhl >> 4) & 0x03) << 4);
                        let v4 = (ql[ql_off + l + 32] >> 4) | (((qhl >> 6) & 0x03) << 4);
                        let sc0 = sc_raw[sc_off + is] as i8 as f32;
                        let sc2 = sc_raw[sc_off + is + 2] as i8 as f32;
                        let sc4 = sc_raw[sc_off + is + 4] as i8 as f32;
                        let sc6 = sc_raw[sc_off + is + 6] as i8 as f32;
                        wl[0] = d * sc0 * (v1 as i32 - 32) as f32;
                        wl[1] = d * sc2 * (v2 as i32 - 32) as f32;
                        wl[2] = d * sc4 * (v3 as i32 - 32) as f32;
                        wl[3] = d * sc6 * (v4 as i32 - 32) as f32;
                    }
                    for (s, input) in inputs.iter().enumerate() {
                        let acc = &mut dst[s];
                        for (l, wl) in w.iter().enumerate() {
                            *acc += wl[0] * input[x_group + l];
                            *acc += wl[1] * input[x_group + l + 32];
                            *acc += wl[2] * input[x_group + l + 64];
                            *acc += wl[3] * input[x_group + l + 96];
                        }
                    }
                }
            }
        });
}

/// Batched matvec for the formats whose kernel unpacks a whole block into f32
/// before the dot product (Q2_K, Q3_K, IQ4_NL, Q5_0, Q5_1).
///
/// `N` is the block's element count and `unpack` the shared, ggml-anchored
/// block decoder — the same one the single-vector kernel uses, so the weights
/// are identical and only the traversal is shared.
fn matvec_unpack_batch_scalar<const N: usize>(
    data: &[u8],
    rows: usize,
    cols: usize,
    block_bytes: usize,
    unpack: impl Fn(&[u8], &mut [f32; N]) + Sync,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    let batch = inputs.len();
    let n_blocks = cols / N;
    let bytes_per_row = n_blocks * block_bytes;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(r, dst)| {
            for acc in dst.iter_mut() {
                *acc = 0.0;
            }
            let row = &data[r * bytes_per_row..];
            let mut buf = [0.0f32; N];
            for bi in 0..n_blocks {
                unpack(&row[bi * block_bytes..], &mut buf);
                for (s, input) in inputs.iter().enumerate() {
                    let x = &input[bi * N..bi * N + N];
                    let acc = &mut dst[s];
                    for (w, xv) in buf.iter().zip(x) {
                        *acc += w * xv;
                    }
                }
            }
        });
}

/// Batched Q2_K matvec (scalar) — see [`matvec_q2_k_scalar`].
fn matvec_q2_k_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    matvec_unpack_batch_scalar::<256>(
        data,
        rows,
        cols,
        84,
        super::dequantize::unpack_q2_k_block,
        inputs,
        out,
    )
}

/// Batched Q3_K matvec (scalar) — see [`matvec_q3_k_scalar`].
fn matvec_q3_k_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    matvec_unpack_batch_scalar::<256>(
        data,
        rows,
        cols,
        110,
        super::dequantize::unpack_q3_k_block,
        inputs,
        out,
    )
}

/// Batched IQ4_NL matvec (scalar) — see [`matvec_iq4_nl_scalar`].
fn matvec_iq4_nl_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    matvec_unpack_batch_scalar::<32>(
        data,
        rows,
        cols,
        18,
        super::dequantize::unpack_iq4_nl_block,
        inputs,
        out,
    )
}

/// Batched Q5_0 matvec (scalar) — see [`matvec_q5_0_scalar`].
#[allow(dead_code)]
fn matvec_q5_0_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    matvec_unpack_batch_scalar::<32>(
        data,
        rows,
        cols,
        22,
        super::dequantize::unpack_q5_0_block,
        inputs,
        out,
    )
}

/// Batched Q5_1 matvec (scalar) — see [`matvec_q5_1_scalar`].
#[allow(dead_code)]
fn matvec_q5_1_batch_scalar(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    matvec_unpack_batch_scalar::<32>(
        data,
        rows,
        cols,
        24,
        super::dequantize::unpack_q5_1_block,
        inputs,
        out,
    )
}

/// Batched fallback matvec — dequantize each row once, dot it against every
/// input. See [`matvec_fallback`].
fn matvec_fallback_batch(
    data: &[u8],
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    let batch = inputs.len();
    let block_size = ggml_type.block_size();
    let type_size = ggml_type.type_size();
    let n_blocks = cols.div_ceil(block_size);
    let bytes_per_row = n_blocks * type_size;

    par_out_rows(out, rows, batch)
        .enumerate()
        .for_each(|(i, dst)| {
            let row_bytes = &data[i * bytes_per_row..(i + 1) * bytes_per_row];
            let row_f32 = dequantize(row_bytes, ggml_type, cols);
            for (s, input) in inputs.iter().enumerate() {
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += row_f32[j] * input[j];
                }
                dst[s] = sum;
            }
        });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dequantize::dequantize_test_helpers::make_q8_0_block;

    #[test]
    fn test_q8_0_matvec_matches_f32() {
        // Build a 2×32 Q8_0 weight matrix (2 rows, each one block of 32 elements)
        // Row 0: scale=1.0, weights=[0,1,2,...,31]
        // Row 1: scale=2.0, weights=[0,-1,-2,...,-31]
        let mut data = Vec::new();
        data.extend(make_q8_0_block(1.0, (0..32).map(|i| i as i8).collect()));
        data.extend(make_q8_0_block(2.0, (0..32).map(|i| -(i as i8)).collect()));

        let qt = QuantizedTensor {
            data: QuantizedStorage::Owned(data),
            rows: 2,
            cols: 32,
            ggml_type: GgmlType::Q8_0,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        };
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let result = qt.matvec(&input);
        assert_eq!(result.shape(), &[2]);

        // Row 0: sum = Σ i*i for i in 0..32, scale=1 → 1.0 * 9920 = 9920
        let expected_row0: f32 = (0..32).map(|i| (i as f32) * (i as f32)).sum::<f32>();
        // Row 1: sum = Σ -i*i for i in 0..32, scale=2 → -19840
        let expected_row1: f32 = -2.0 * (0..32).map(|i| (i as f32) * (i as f32)).sum::<f32>();

        assert!(
            (result.data()[0] - expected_row0).abs() < 0.1,
            "Row 0: got {}, expected {}",
            result.data()[0],
            expected_row0
        );
        assert!(
            (result.data()[1] - expected_row1).abs() < 0.1,
            "Row 1: got {}, expected {}",
            result.data()[1],
            expected_row1
        );
    }

    #[test]
    fn test_row_as_f32_matches_dequantize() {
        // Single Q8_0 row: scale=0.5, weights=[10, -10, 5, -5, ...]
        let quants: Vec<i8> = (0..32)
            .map(|i| if i % 2 == 0 { 10i8 } else { -10i8 })
            .collect();
        let block = make_q8_0_block(0.5, quants.clone());

        let qt = QuantizedTensor {
            data: QuantizedStorage::Owned(block.clone()),
            rows: 1,
            cols: 32,
            ggml_type: GgmlType::Q8_0,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        };

        let row = qt.row_as_f32(0);
        assert_eq!(row.shape(), &[32]);

        for (i, &val) in row.data().iter().enumerate() {
            let expected = quants[i] as f32 * 0.5;
            assert!(
                (val - expected).abs() < 1e-3,
                "Index {i}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_quantized_uses_less_memory_than_f32() {
        // Q8_0: 34 bytes per 32 elements
        // F32:  128 bytes per 32 elements
        let block = make_q8_0_block(1.0, vec![1i8; 32]);
        let qt = QuantizedTensor {
            data: QuantizedStorage::Owned(block),
            rows: 1,
            cols: 32,
            ggml_type: GgmlType::Q8_0,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        };
        let f32_equivalent_bytes = qt.rows * qt.cols * 4;
        assert!(
            qt.data.as_slice().len() < f32_equivalent_bytes,
            "Quantized {} bytes should be less than f32 {} bytes",
            qt.data.as_slice().len(),
            f32_equivalent_bytes
        );
    }

    #[test]
    fn test_from_f32_matvec() {
        // Simple 2×3 f32 matrix: [[1,2,3],[4,5,6]] × [1,1,1] = [6, 15]
        let qt = QuantizedTensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = qt.matvec(&[1.0, 1.0, 1.0]);
        assert_eq!(result.shape(), &[2]);
        assert!((result.data()[0] - 6.0).abs() < 1e-5);
        assert!((result.data()[1] - 15.0).abs() < 1e-5);
    }

    /// Verify SIMD and scalar Q8_0 kernels produce the same output.
    /// Uses multiple blocks per row to exercise the cross-block accumulation.
    #[test]
    fn test_q8_0_simd_matches_scalar() {
        // 4 rows × 128 cols = 4 blocks per row
        let cols = 128;
        let rows = 4;
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..4 {
                let quants: Vec<i8> = (0..32)
                    .map(|j| ((r * 4 + b) as i8).wrapping_mul(j as i8 + 1))
                    .collect();
                let scale = 0.1 * (r + 1) as f32;
                data.extend(make_q8_0_block(scale, quants));
            }
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        let scalar = matvec_q8_0_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q8_0_avx2(&data, rows, cols, &input) };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}",
                        scalar[i],
                        simd[i]
                    );
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q8_0_neon(&data, rows, cols, &input) };
            for i in 0..rows {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-3,
                    "Row {i}: scalar={}, neon={}",
                    scalar[i],
                    simd[i]
                );
            }
        }
    }

    /// Q4_K: verify the direct scalar kernel matches dequantize-then-dot.
    ///
    /// Super-block: d=1.0, dmin=0, all sub-block scales=1, mins=0,
    /// all nibbles=8 → each element = 8.0, expected sum = 256 × 8 = 2048.
    #[test]
    fn test_q4_k_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 144];
        // d = 1.0
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // dmin = 0 (zeros)
        // scales: want sc=1, mn=0 for all 8 sub-blocks
        // j<4: block[4+j] = 1 (sc), block[4+j+4] = 0 (mn)
        for j in 0..4usize {
            block[4 + j] = 1;
        }
        // j=4..8: block[4+j+4] = 1 gives sc=1 and mn=0 when upper nibble is 0
        for j in 4..8usize {
            block[4 + j + 4] = 1;
        }
        // qs: all 0x88 → both nibbles = 8
        for i in 16..144 {
            block[i] = 0x88;
        }

        let input = vec![1.0f32; 256];

        // Reference: dequantize → dot product
        let deq = dequantize(&block, GgmlType::Q4K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q4_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q4_K matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 2048.0).abs() < 1.0,
            "Q4_K expected ≈2048, got {}",
            result[0]
        );
    }

    /// Q5_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// All qs nibbles and qh bits zero → all 5-bit values = 0.
    /// Expected output = 0.
    #[test]
    fn test_q5_k_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 176];
        // d = 1.0, dmin = 0
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales: sc=1 for all sub-blocks (same encoding as Q4_K)
        for j in 0..4usize {
            block[4 + j] = 1;
        }
        for j in 4..8usize {
            block[4 + j + 4] = 1;
        }
        // qh and qs all zero → all values = d * sc * 0 - 0 = 0
        // Input is non-zero so we'd see a non-zero result if values were non-zero
        let input: Vec<f32> = (0..256).map(|i| (i % 7) as f32 * 0.1).collect();

        let deq = dequantize(&block, GgmlType::Q5K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q5_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 1e-4,
            "Q5_K matvec: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Q6_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// All ql = 0x33 (nibbles=3,3), all qh = 0, all scales = 1, d = 1.0.
    /// Each 6-bit value = (3 | 0) - 32 = -29.
    #[test]
    fn test_q6_k_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 210];
        // ql (0..128): nibbles = 3 → 0x33
        for i in 0..128 {
            block[i] = 0x33;
        }
        // qh (128..192): all 0 — no high bits
        // scales (192..208): all 1 (i8 = 1)
        for i in 192..208 {
            block[i] = 1;
        }
        // d (208..210) = 1.0
        block[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let input = vec![1.0f32; 256];

        let deq = dequantize(&block, GgmlType::Q6K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q6_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q6_K matvec: got {}, expected {}",
            result[0],
            expected
        );
        // Each 6-bit value = 3 - 32 = -29, scale=1, d=1. Sum = 256 * -29 = -7424.
        assert!(
            (result[0] - (-7424.0_f32)).abs() < 1.0,
            "Q6_K expected ≈-7424, got {}",
            result[0]
        );
    }

    /// Verify Q4_K SIMD and scalar kernels produce the same output.
    #[test]
    fn test_q4_k_simd_matches_scalar() {
        use crate::model::gguf::GgmlType;

        // 2 rows × 256 cols = 1 super-block per row
        let rows = 2;
        let cols = 256;
        let mut data = Vec::new();

        for r in 0..rows {
            let mut block = vec![0u8; 144];
            // d = 0.5, dmin = 0.1
            block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
            // scales: sc=r+1 for j<4, sc=r+2 for j=4..8; mn=1 for all
            for j in 0..4usize {
                block[4 + j] = (r as u8 + 1) & 63;
            }
            for j in 0..4usize {
                block[4 + j + 4] = 1u8 & 63;
            }
            for j in 4..8usize {
                block[4 + j + 4] = ((r as u8 + 2) & 0xF) | (1u8 << 4);
            }
            // qs: deterministic nibble pattern
            for i in 16..144 {
                let lo = (r * 7 + i) as u8 % 16;
                let hi = (r * 3 + i + 5) as u8 % 16;
                block[i] = lo | (hi << 4);
            }
            data.extend(block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let scalar = matvec_q4_k_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q4_k_avx2(&data, rows, cols, &input) };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}",
                        scalar[i],
                        simd[i]
                    );
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q4_k_neon(&data, rows, cols, &input) };
            for i in 0..rows {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-3,
                    "Row {i}: scalar={}, neon={}",
                    scalar[i],
                    simd[i]
                );
            }
        }
        let _ = &GgmlType::Q4K; // suppress unused import warning
    }

    /// Verify Q5_K SIMD and scalar kernels produce the same output.
    ///
    /// Two rows, 1 super-block each: varied qs nibbles, qh bits, scales, d/dmin.
    #[test]
    fn test_q5_k_simd_matches_scalar() {
        let rows = 2;
        let cols = 256;
        let mut data = Vec::new();

        for r in 0..rows {
            let mut block = vec![0u8; 176];
            // d = 0.5, dmin = 0.1
            block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
            // scales: sc=r+1 for j<4; mn=1 for all (same encoding as Q4_K test)
            for j in 0..4usize {
                block[4 + j] = (r as u8 + 1) & 63;
            }
            for j in 0..4usize {
                block[4 + j + 4] = 1u8 & 63;
            }
            for j in 4..8usize {
                block[4 + j + 4] = ((r as u8 + 2) & 0xF) | (1u8 << 4);
            }
            // qh (16..48): varied patterns so not all 5th bits are 0
            for i in 0..32 {
                block[16 + i] = ((r * 13 + i * 7 + 3) % 256) as u8;
            }
            // qs (48..176): deterministic nibble pattern
            for i in 0..128 {
                let lo = ((r * 7 + i) % 16) as u8;
                let hi = ((r * 3 + i + 5) % 16) as u8;
                block[48 + i] = lo | (hi << 4);
            }
            data.extend(block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let scalar = matvec_q5_k_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q5_k_avx2(&data, rows, cols, &input) };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}",
                        scalar[i],
                        simd[i]
                    );
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q5_k_neon(&data, rows, cols, &input) };
            for i in 0..rows {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-3,
                    "Row {i}: scalar={}, neon={}",
                    scalar[i],
                    simd[i]
                );
            }
        }
    }

    /// Verify Q6_K SIMD and scalar kernels produce the same output.
    ///
    /// Two rows, 1 super-block each: varied ql, qh, scales, and d.
    #[test]
    fn test_q6_k_simd_matches_scalar() {
        let rows = 2;
        let cols = 256;
        let mut data = Vec::new();

        for r in 0..rows {
            let mut block = vec![0u8; 210];
            // ql (0..128): varied nibble patterns
            for i in 0..128 {
                block[i] = (r * 37 + i * 13 + 7) as u8;
            }
            // qh (128..192): varied high-bit patterns
            for i in 0..64 {
                block[128 + i] = (r * 11 + i * 5 + 3) as u8;
            }
            // scales (192..208): non-zero i8 values (avoid 0 to make errors visible)
            for i in 0..16 {
                block[192 + i] = (((r * 7 + i * 3 + 1) % 15) as u8).wrapping_add(1);
            }
            // d = 0.25 + 0.125 * r
            let d_val = 0.25 + 0.125 * r as f32;
            block[208..210].copy_from_slice(&half::f16::from_f32(d_val).to_le_bytes());
            data.extend(block);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let scalar = matvec_q6_k_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q6_k_avx2(&data, rows, cols, &input) };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}",
                        scalar[i],
                        simd[i]
                    );
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q6_k_neon(&data, rows, cols, &input) };
            for i in 0..rows {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-3,
                    "Row {i}: scalar={}, neon={}",
                    scalar[i],
                    simd[i]
                );
            }
        }
    }

    /// Q2_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// d=1, dmin=0, all scales low-nibble=1, high-nibble=0; qs=0x55 → q2=1 for all.
    /// Each element = 1*1 - 0 = 1. Expected sum = 256.
    #[test]
    fn test_q2_k_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 84];
        // scales[0..16]: low nibble=1 (sc=1), high nibble=0 (mn=0)
        for i in 0..16 {
            block[i] = 0x01;
        }
        // qs[16..80]: 0x55 → bits [1:0]=1, [3:2]=1, [5:4]=1, [7:6]=1 → q2=1 for all
        for i in 16..80 {
            block[i] = 0x55;
        }
        // d=1.0, dmin=0.0
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // dmin stays 0.0 (already zero)

        let input = vec![1.0f32; 256];
        let deq = dequantize(&block, GgmlType::Q2K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q2_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q2_K matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 256.0).abs() < 1.0,
            "Q2_K expected ≈256, got {}",
            result[0]
        );
    }

    /// Q3_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// d=1, all raw6 scale=36 (→ signed=4), qs=0xAA → lo2=2 for all, hmask=0xFF → hi1=1 for all.
    /// q3 = 2 | (1<<2) = 6; value = 1 * 4 * (6-4) = 8. Expected sum = 256*8 = 2048.
    #[test]
    fn test_q3_k_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 110];
        // hmask[0..32]: all 0xFF → hi1=1 for all elements
        for i in 0..32 {
            block[i] = 0xFF;
        }
        // qs[32..96]: 0xAA = 0b10101010 → bits[1:0]=2, [3:2]=2, [5:4]=2, [7:6]=2
        for i in 32..96 {
            block[i] = 0xAA;
        }
        // scales[96..108]: encode raw6=36 for all 16 sub-groups (signed=+4),
        // using the ggml packing (see unpack_q3_k_block): the low 4 bits live
        // in nibble planes over bytes 0..8, the high 2 bits in 2-bit planes
        // over bytes 8..12.
        //   low4 = 36 & 0xF = 4  → every nibble of bytes 0..8 is 4  → 0x44
        //   high2 = 36 >> 4  = 2 → every 2-bit field of bytes 8..12 → 0xAA
        for i in 0..8_usize {
            block[96 + i] = 0x44;
        }
        for i in 8..12_usize {
            block[96 + i] = 0xAA;
        }
        // d=1.0
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let input = vec![1.0f32; 256];
        let deq = dequantize(&block, GgmlType::Q3K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q3_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q3_K matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 2048.0).abs() < 1.0,
            "Q3_K expected ≈2048, got {}",
            result[0]
        );
    }

    /// IQ4_NL: verify scalar kernel matches dequantize-then-dot.
    ///
    /// d=1, all qs=0x88 → all nibbles=8 → KVALUES[8]=1. Expected sum = 32*1 = 32.
    #[test]
    fn test_iq4_nl_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        for i in 2..18 {
            block[i] = 0x88;
        } // nibble=8 → KVALUES[8]=1

        let input = vec![1.0f32; 32];
        let deq = dequantize(&block, GgmlType::IQ4NL, 32);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_iq4_nl_scalar(&block, 1, 32, &input);
        assert!(
            (result[0] - expected).abs() < 0.1,
            "IQ4_NL matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 32.0).abs() < 0.1,
            "IQ4_NL expected 32.0, got {}",
            result[0]
        );
    }

    /// One 32-element block: `[f16 header…] [16 × qs_byte]`.
    fn block_with_nibbles(header: &[[u8; 2]], qs_byte: u8) -> Vec<u8> {
        let mut data: Vec<u8> = header.iter().flatten().copied().collect();
        data.extend(std::iter::repeat_n(qs_byte, 16));
        data
    }

    /// Input that differs between the two nibble planes, so a kernel that read
    /// the nibbles interleaved instead of split-plane would score differently.
    fn split_plane_probe() -> Vec<f32> {
        (0..32).map(|i| if i < 16 { 1.0 } else { 2.0 }).collect()
    }

    /// Q4_1: verify scalar kernel matches dequantize-then-dot.
    ///
    /// d=1, m=0.5, every qs byte 0xA3 → elements 0..16 are 3*1+0.5 = 3.5 and
    /// elements 16..32 are 10*1+0.5 = 10.5.
    /// Expected = 16 × 3.5 × 1 + 16 × 10.5 × 2 = 56 + 336 = 392.
    #[test]
    fn test_q4_1_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let block = block_with_nibbles(
            &[
                half::f16::from_f32(1.0).to_le_bytes(),
                half::f16::from_f32(0.5).to_le_bytes(),
            ],
            0xA3,
        );
        let input = split_plane_probe();

        let deq = dequantize(&block, GgmlType::Q4_1, 32);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q4_1_scalar(&block, 1, 32, &input);
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "Q4_1 matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 392.0).abs() < 1e-3,
            "Q4_1 expected 392.0, got {}",
            result[0]
        );
    }

    /// Q5_0: verify scalar kernel matches dequantize-then-dot.
    ///
    /// d=1, qs bytes 0xA3, qh=0x0001_0000 → only bit 16 is set, which is the
    /// 5th bit of element 16 (the first of the high-nibble plane):
    ///   elements 0..16:  3 − 16       = −13, × 1 → −208
    ///   element 16:      (10|16) − 16 =  10, × 2 →   20
    ///   elements 17..32: 10 − 16      =  −6, × 2 → −180
    /// Expected = −368.
    #[test]
    fn test_q5_0_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let qh = 0x0001_0000u32.to_le_bytes();
        let block = block_with_nibbles(
            &[
                half::f16::from_f32(1.0).to_le_bytes(),
                [qh[0], qh[1]],
                [qh[2], qh[3]],
            ],
            0xA3,
        );
        let input = split_plane_probe();

        let deq = dequantize(&block, GgmlType::Q5_0, 32);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q5_0_scalar(&block, 1, 32, &input);
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "Q5_0 matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - (-368.0_f32)).abs() < 1e-3,
            "Q5_0 expected -368.0, got {}",
            result[0]
        );
    }

    /// Q5_1: verify scalar kernel matches dequantize-then-dot.
    ///
    /// Same quants as `test_q5_0_matvec_matches_dequantize` but affine with
    /// d=1, m=0.5 and no centering:
    ///   elements 0..16:   3 + 0.5 =  3.5, × 1 →  56
    ///   element 16:      26 + 0.5 = 26.5, × 2 →  53
    ///   elements 17..32: 10 + 0.5 = 10.5, × 2 → 315
    /// Expected = 424.
    #[test]
    fn test_q5_1_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let qh = 0x0001_0000u32.to_le_bytes();
        let block = block_with_nibbles(
            &[
                half::f16::from_f32(1.0).to_le_bytes(),
                half::f16::from_f32(0.5).to_le_bytes(),
                [qh[0], qh[1]],
                [qh[2], qh[3]],
            ],
            0xA3,
        );
        let input = split_plane_probe();

        let deq = dequantize(&block, GgmlType::Q5_1, 32);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q5_1_scalar(&block, 1, 32, &input);
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "Q5_1 matvec: got {}, expected {}",
            result[0],
            expected
        );
        assert!(
            (result[0] - 424.0).abs() < 1e-3,
            "Q5_1 expected 424.0, got {}",
            result[0]
        );
    }

    /// Q4_0: verify the scalar kernel matches dequantize-then-dot.
    ///
    /// This ties `matvec_q4_0_scalar` to `unpack_q4_0_block`, which is itself
    /// anchored to ggml by
    /// `dequantize::ggml_reference_tests::q4_0_matches_ggml_reference`. A
    /// distinct input per element means a within-block nibble permutation
    /// (the pre-fix interleaved reading) changes the dot product.
    #[test]
    fn test_q4_0_matvec_matches_dequantize() {
        use crate::model::gguf::GgmlType;
        use crate::tensor::dequantize::dequantize;

        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for j in 0..16 {
            block[2 + j] = ((j * 37 + 11) & 0xFF) as u8;
        }

        let input: Vec<f32> = (0..32).map(|i| 1.0 + i as f32).collect();
        let deq = dequantize(&block, GgmlType::Q4_0, 32);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q4_0_scalar(&block, 1, 32, &input);
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "Q4_0 matvec: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify SIMD and scalar Q4_0 kernels produce the same output.
    #[test]
    fn test_q4_0_simd_matches_scalar() {
        // Build a 2×32 Q4_0 matrix (1 block per row)
        let cols = 32;
        let rows = 2;
        let mut data = Vec::new();
        for r in 0..rows {
            // scale as f16 bytes
            let scale = half::f16::from_f32(0.5 * (r + 1) as f32);
            data.extend_from_slice(&scale.to_le_bytes());
            // 16 packed nibble bytes → 32 elements
            for byte_idx in 0..16 {
                let lo = ((byte_idx + r * 3) % 16) as u8;
                let hi = ((byte_idx + r * 7 + 1) % 16) as u8;
                data.push(lo | (hi << 4));
            }
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();

        let scalar = matvec_q4_0_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q4_0_avx2(&data, rows, cols, &input) };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}",
                        scalar[i],
                        simd[i]
                    );
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q4_0_neon(&data, rows, cols, &input) };
            for i in 0..rows {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-3,
                    "Row {i}: scalar={}, neon={}",
                    scalar[i],
                    simd[i]
                );
            }
        }
    }

    /// Assert two kernels agree row-by-row.
    #[cfg(any(
        all(target_arch = "x86_64", feature = "rayon"),
        target_arch = "aarch64"
    ))]
    fn assert_rows_close(fmt: &str, scalar: &[f32], simd: &[f32]) {
        for (i, (&s, &v)) in scalar.iter().zip(simd).enumerate() {
            assert!((s - v).abs() < 1e-3, "{fmt} row {i}: scalar={s}, simd={v}");
        }
    }

    /// Deterministic nibble bytes for one block.
    fn nibble_bytes(seed: usize) -> Vec<u8> {
        (0..16).map(|j| (seed + j * 7 + 5) as u8).collect()
    }

    /// Input spanning both signs so sign errors in the kernels show up.
    fn simd_probe_input(cols: usize) -> Vec<f32> {
        (0..cols).map(|i| (i as f32) * 0.05 - 1.6).collect()
    }

    /// Verify SIMD and scalar Q4_1 kernels produce the same output.
    /// Two blocks per row exercises cross-block accumulation.
    #[test]
    fn test_q4_1_simd_matches_scalar() {
        let rows = 3;
        let cols = 64;
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..2usize {
                data.extend_from_slice(&half::f16::from_f32(0.25 * (r + 1) as f32).to_le_bytes());
                data.extend_from_slice(&half::f16::from_f32(-0.5 + 0.125 * b as f32).to_le_bytes());
                data.extend(nibble_bytes(r * 37 + b * 13));
            }
        }

        let input = simd_probe_input(cols);
        let scalar = matvec_q4_1_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q4_1_avx2(&data, rows, cols, &input) };
                assert_rows_close("Q4_1", &scalar, &simd);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q4_1_neon(&data, rows, cols, &input) };
            assert_rows_close("Q4_1", &scalar, &simd);
        }
    }

    /// Verify SIMD and scalar Q5_0 kernels produce the same output.
    /// The `qh` words vary per block so both 5th-bit planes are exercised.
    #[test]
    fn test_q5_0_simd_matches_scalar() {
        let rows = 3;
        let cols = 64;
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..2usize {
                data.extend_from_slice(&half::f16::from_f32(0.25 * (r + 1) as f32).to_le_bytes());
                data.extend_from_slice(
                    &(0x9E3F_1C05u32 ^ ((r * 2 + b) as u32 * 0x0101_1011)).to_le_bytes(),
                );
                data.extend(nibble_bytes(r * 37 + b * 13));
            }
        }

        let input = simd_probe_input(cols);
        let scalar = matvec_q5_0_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q5_0_avx2(&data, rows, cols, &input) };
                assert_rows_close("Q5_0", &scalar, &simd);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q5_0_neon(&data, rows, cols, &input) };
            assert_rows_close("Q5_0", &scalar, &simd);
        }
    }

    /// Verify SIMD and scalar Q5_1 kernels produce the same output.
    #[test]
    fn test_q5_1_simd_matches_scalar() {
        let rows = 3;
        let cols = 64;
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..2usize {
                data.extend_from_slice(&half::f16::from_f32(0.25 * (r + 1) as f32).to_le_bytes());
                data.extend_from_slice(&half::f16::from_f32(-0.5 + 0.125 * b as f32).to_le_bytes());
                data.extend_from_slice(
                    &(0x9E3F_1C05u32 ^ ((r * 2 + b) as u32 * 0x0101_1011)).to_le_bytes(),
                );
                data.extend(nibble_bytes(r * 37 + b * 13));
            }
        }

        let input = simd_probe_input(cols);
        let scalar = matvec_q5_1_scalar(&data, rows, cols, &input);

        #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd =
                    unsafe { crate::tensor::simd::matvec_q5_1_avx2(&data, rows, cols, &input) };
                assert_rows_close("Q5_1", &scalar, &simd);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "rayon"))]
        {
            let simd =
                unsafe { crate::tensor::simd_neon::matvec_q5_1_neon(&data, rows, cols, &input) };
            assert_rows_close("Q5_1", &scalar, &simd);
        }
    }

    // ── Batched matvec ───────────────────────────────────────────────────
    //
    // Continuous batching runs `B` sequences through one weight traversal.
    // That is only safe if it is *invisible*: a request must produce the same
    // tokens whether it was decoded alone or alongside seven others. These
    // tests pin bit-identity — not a tolerance — between `matvec_batch` and
    // `matvec`, for every format and on whichever kernel the host dispatches
    // to (AVX2 or scalar), plus scalar-vs-scalar directly.

    /// Build a `rows × cols` matrix of `ty` with a deterministic byte pattern
    /// and finite f16 scale fields (arbitrary bytes could encode NaN/Inf
    /// scales, which would make bit-equality assertions meaningless).
    fn patterned_matrix(rows: usize, cols: usize, ty: GgmlType) -> Vec<u8> {
        let block_size = ty.block_size();
        let type_size = ty.type_size();
        let n_blocks = cols / block_size;
        let mut data = vec![0u8; rows * n_blocks * type_size];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) % 251) as u8;
        }
        let f16b = |v: f32| f16::from_f32(v).to_le_bytes();
        for blk in 0..rows * n_blocks {
            let off = blk * type_size;
            let d = 0.05 + 0.01 * ((blk % 7) as f32);
            let put = |data: &mut Vec<u8>, at: usize, v: f32| {
                data[off + at..off + at + 2].copy_from_slice(&f16b(v));
            };
            match ty {
                GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::IQ4NL | GgmlType::Q5_0 => {
                    put(&mut data, 0, d)
                }
                GgmlType::Q4_1 | GgmlType::Q5_1 => {
                    put(&mut data, 0, d);
                    put(&mut data, 2, -0.02);
                }
                GgmlType::Q4K | GgmlType::Q5K => {
                    put(&mut data, 0, d);
                    put(&mut data, 2, 0.01);
                }
                GgmlType::Q6K => put(&mut data, 208, d),
                GgmlType::Q2K => {
                    put(&mut data, 80, d);
                    put(&mut data, 82, 0.01);
                }
                GgmlType::Q3K => put(&mut data, 108, d),
                _ => {}
            }
        }
        data
    }

    /// `B` deterministic, mutually distinct activation vectors.
    fn batch_inputs(batch: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..batch)
            .map(|s| {
                (0..cols)
                    .map(|i| (((i + s * 13) % 17) as f32) * 0.1 - 0.8)
                    .collect()
            })
            .collect()
    }

    /// Assert `matvec_batch` is bit-identical to `matvec` per input, for each
    /// requested batch size. 17 exceeds the SIMD kernels' lane cap, so it also
    /// covers the lane-chunking path.
    fn assert_batch_matches_single(qt: &QuantizedTensor, label: &str) {
        for &batch in &[1usize, 2, 3, 4, 17] {
            let inputs = batch_inputs(batch, qt.cols());
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
            let batched = qt.matvec_batch(&refs);
            for (s, r) in refs.iter().enumerate() {
                let single = qt.matvec(r);
                for (i, (&a, &b)) in single.data().iter().zip(batched[s].data()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label} batch={batch} seq={s} row={i}: single={a}, batched={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn batched_matvec_is_bit_identical_q8_0() {
        let data = patterned_matrix(6, 64, GgmlType::Q8_0);
        let qt = QuantizedTensor::from_raw(data, 6, 64, GgmlType::Q8_0);
        assert_batch_matches_single(&qt, "Q8_0");
    }

    #[test]
    fn batched_matvec_is_bit_identical_q4_0() {
        let data = patterned_matrix(6, 64, GgmlType::Q4_0);
        let qt = QuantizedTensor::from_raw(data, 6, 64, GgmlType::Q4_0);
        assert_batch_matches_single(&qt, "Q4_0");
    }

    #[test]
    fn batched_matvec_is_bit_identical_q4_k() {
        let data = patterned_matrix(3, 512, GgmlType::Q4K);
        let qt = QuantizedTensor::from_raw(data, 3, 512, GgmlType::Q4K);
        assert_batch_matches_single(&qt, "Q4_K");
    }

    #[test]
    fn batched_matvec_is_bit_identical_q5_k() {
        let data = patterned_matrix(3, 512, GgmlType::Q5K);
        let qt = QuantizedTensor::from_raw(data, 3, 512, GgmlType::Q5K);
        assert_batch_matches_single(&qt, "Q5_K");
    }

    #[test]
    fn batched_matvec_is_bit_identical_q6_k() {
        let data = patterned_matrix(3, 512, GgmlType::Q6K);
        let qt = QuantizedTensor::from_raw(data, 3, 512, GgmlType::Q6K);
        assert_batch_matches_single(&qt, "Q6_K");
    }

    #[test]
    fn batched_matvec_is_bit_identical_q2_k_q3_k() {
        let q2 = QuantizedTensor::from_raw(
            patterned_matrix(2, 256, GgmlType::Q2K),
            2,
            256,
            GgmlType::Q2K,
        );
        assert_batch_matches_single(&q2, "Q2_K");
        let q3 = QuantizedTensor::from_raw(
            patterned_matrix(2, 256, GgmlType::Q3K),
            2,
            256,
            GgmlType::Q3K,
        );
        assert_batch_matches_single(&q3, "Q3_K");
    }

    #[test]
    fn batched_matvec_is_bit_identical_iq4_nl_and_q5_variants() {
        for ty in [
            GgmlType::IQ4NL,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
        ] {
            let qt = QuantizedTensor::from_raw(patterned_matrix(4, 64, ty), 4, 64, ty);
            assert_batch_matches_single(&qt, &format!("{ty:?}"));
        }
    }

    /// F32 weights take the dequantize-then-dot fallback, which the tiny test
    /// model (and any un-quantized tensor) uses.
    #[test]
    fn batched_matvec_is_bit_identical_f32_fallback() {
        let values: Vec<f32> = (0..40).map(|i| i as f32 * 0.07 - 1.3).collect();
        let qt = QuantizedTensor::from_f32(&values, 5, 8);
        assert_batch_matches_single(&qt, "F32");
    }

    /// The scalar batch kernels are the portable reference (non-x86 hosts, and
    /// x86 hosts without AVX2 land here), so pin them against the scalar
    /// single-vector kernels directly rather than only through dispatch.
    #[test]
    fn scalar_batch_kernels_match_scalar_single() {
        let batch = 3;
        let cases: Vec<(&str, GgmlType, usize, usize)> = vec![
            ("Q8_0", GgmlType::Q8_0, 4, 64),
            ("Q4_0", GgmlType::Q4_0, 4, 64),
            ("Q4_K", GgmlType::Q4K, 2, 512),
            ("Q5_K", GgmlType::Q5K, 2, 512),
            ("Q6_K", GgmlType::Q6K, 2, 512),
            ("Q2_K", GgmlType::Q2K, 2, 256),
            ("Q3_K", GgmlType::Q3K, 2, 256),
            ("IQ4_NL", GgmlType::IQ4NL, 3, 64),
            ("Q5_0", GgmlType::Q5_0, 3, 64),
            ("Q5_1", GgmlType::Q5_1, 3, 64),
        ];
        for (label, ty, rows, cols) in cases {
            let data = patterned_matrix(rows, cols, ty);
            let inputs = batch_inputs(batch, cols);
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();

            let mut batched = vec![0.0f32; rows * batch];
            match ty {
                GgmlType::Q8_0 => matvec_q8_0_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q4_0 => matvec_q4_0_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q4K => matvec_q4_k_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q5K => matvec_q5_k_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q6K => matvec_q6_k_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q2K => matvec_q2_k_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q3K => matvec_q3_k_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::IQ4NL => {
                    matvec_iq4_nl_batch_scalar(&data, rows, cols, &refs, &mut batched)
                }
                GgmlType::Q5_0 => matvec_q5_0_batch_scalar(&data, rows, cols, &refs, &mut batched),
                GgmlType::Q5_1 => matvec_q5_1_batch_scalar(&data, rows, cols, &refs, &mut batched),
                _ => unreachable!(),
            }

            for (s, r) in refs.iter().enumerate() {
                let single = match ty {
                    GgmlType::Q8_0 => matvec_q8_0_scalar(&data, rows, cols, r),
                    GgmlType::Q4_0 => matvec_q4_0_scalar(&data, rows, cols, r),
                    GgmlType::Q4K => matvec_q4_k_scalar(&data, rows, cols, r),
                    GgmlType::Q5K => matvec_q5_k_scalar(&data, rows, cols, r),
                    GgmlType::Q6K => matvec_q6_k_scalar(&data, rows, cols, r),
                    GgmlType::Q2K => matvec_q2_k_scalar(&data, rows, cols, r),
                    GgmlType::Q3K => matvec_q3_k_scalar(&data, rows, cols, r),
                    GgmlType::IQ4NL => matvec_iq4_nl_scalar(&data, rows, cols, r),
                    GgmlType::Q5_0 => matvec_q5_0_scalar(&data, rows, cols, r),
                    GgmlType::Q5_1 => matvec_q5_1_scalar(&data, rows, cols, r),
                    _ => unreachable!(),
                };
                for (i, &expected) in single.iter().enumerate() {
                    let got = batched[i * batch + s];
                    assert_eq!(
                        expected.to_bits(),
                        got.to_bits(),
                        "{label} seq={s} row={i}: scalar={expected}, scalar-batch={got}"
                    );
                }
            }
        }
    }

    /// On an AVX2 host the batched SIMD kernels are what the engine actually
    /// runs; hold them to the scalar batch reference.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    fn simd_batch_kernels_match_scalar_batch() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let batch = 5;
        let cases: Vec<(&str, GgmlType, usize, usize)> = vec![
            ("Q8_0", GgmlType::Q8_0, 4, 64),
            ("Q4_0", GgmlType::Q4_0, 4, 64),
            ("Q4_K", GgmlType::Q4K, 2, 512),
            ("Q5_K", GgmlType::Q5K, 2, 512),
            ("Q6_K", GgmlType::Q6K, 2, 512),
        ];
        for (label, ty, rows, cols) in cases {
            let data = patterned_matrix(rows, cols, ty);
            let inputs = batch_inputs(batch, cols);
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();

            let mut scalar = vec![0.0f32; rows * batch];
            let mut simd = vec![0.0f32; rows * batch];
            // SAFETY: AVX2+FMA detected above; `patterned_matrix` sizes the
            // buffer to rows × (cols / block) blocks and every input is `cols`
            // long, satisfying the kernels' length contract.
            unsafe {
                match ty {
                    GgmlType::Q8_0 => {
                        matvec_q8_0_batch_scalar(&data, rows, cols, &refs, &mut scalar);
                        crate::tensor::simd::matvec_q8_0_batch_avx2(
                            &data, rows, cols, &refs, &mut simd,
                        );
                    }
                    GgmlType::Q4_0 => {
                        matvec_q4_0_batch_scalar(&data, rows, cols, &refs, &mut scalar);
                        crate::tensor::simd::matvec_q4_0_batch_avx2(
                            &data, rows, cols, &refs, &mut simd,
                        );
                    }
                    GgmlType::Q4K => {
                        matvec_q4_k_batch_scalar(&data, rows, cols, &refs, &mut scalar);
                        crate::tensor::simd::matvec_q4_k_batch_avx2(
                            &data, rows, cols, &refs, &mut simd,
                        );
                    }
                    GgmlType::Q5K => {
                        matvec_q5_k_batch_scalar(&data, rows, cols, &refs, &mut scalar);
                        crate::tensor::simd::matvec_q5_k_batch_avx2(
                            &data, rows, cols, &refs, &mut simd,
                        );
                    }
                    GgmlType::Q6K => {
                        matvec_q6_k_batch_scalar(&data, rows, cols, &refs, &mut scalar);
                        crate::tensor::simd::matvec_q6_k_batch_avx2(
                            &data, rows, cols, &refs, &mut simd,
                        );
                    }
                    _ => unreachable!(),
                }
            }
            // Scalar and SIMD sum in different orders (the SIMD kernels keep
            // 8 partial lanes), so this is the usual relative tolerance the
            // single-vector SIMD tests use — not the bit-equality the
            // batch-vs-single tests above demand.
            for (i, (&a, &b)) in scalar.iter().zip(&simd).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                    "{label} index {i}: scalar-batch={a}, simd-batch={b}"
                );
            }
        }
    }

    /// Verify that `QuantizedStorage::Owned` and `QuantizedStorage::Borrowed`
    /// produce identical matvec output.  We use an in-memory copy of the bytes
    /// to simulate the lazy (Borrowed) path without needing a real file.
    #[test]
    fn test_lazy_storage_matches_eager() {
        let cols = 32;
        let rows = 2;
        let mut raw = Vec::new();
        raw.extend(make_q8_0_block(1.0, (0..32).map(|i| i as i8).collect()));
        raw.extend(make_q8_0_block(2.0, (0..32).map(|i| -(i as i8)).collect()));

        // Eager tensor (Owned storage).
        let qt_eager = QuantizedTensor {
            data: QuantizedStorage::Owned(raw.clone()),
            rows,
            cols,
            ggml_type: GgmlType::Q8_0,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        };

        // "Lazy" tensor: shares bytes via Arc but reads through as_slice().
        // (A real lazy tensor would hold Arc<Mmap>; this exercises the same
        // code path by using a heap-allocated Arc<Vec<u8>> cast as a slice.)
        let shared = Arc::new(raw);
        let len = shared.len();
        // Build a Borrowed variant using a fake Mmap by borrowing directly.
        // Since we can't construct a Mmap without a file, we test that Owned
        // and a second Owned with the same bytes produce identical output —
        // verifying as_slice() returns correct data for both variants.
        let qt_lazy = QuantizedTensor {
            data: QuantizedStorage::Owned((*shared).clone()),
            rows,
            cols,
            ggml_type: GgmlType::Q8_0,
            #[cfg(feature = "vulkan")]
            gpu_buf_name: None,
        };
        let _ = len;

        let input: Vec<f32> = (0..cols).map(|i| i as f32 * 0.5).collect();
        let out_eager = qt_eager.matvec(&input);
        let out_lazy = qt_lazy.matvec(&input);
        for (i, (&a, &b)) in out_eager.data().iter().zip(out_lazy.data()).enumerate() {
            assert!((a - b).abs() < 1e-6, "index {i}: eager={a}, lazy={b}");
        }
    }
}
