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

use half::f16;
use rayon::prelude::*;

use crate::model::gguf::{GgmlType, GgufModel};
use super::tensor::Tensor;
use super::dequantize::{dequantize, get_scale_min_q4k};

/// A weight matrix stored in its original quantized format.
///
/// Activations (input/output of each op) remain f32 `Tensor`.
/// Only the large weight matrices use this type.
#[derive(Clone)]
pub struct QuantizedTensor {
    /// Raw bytes copied from the GGUF memory map.
    data: Vec<u8>,
    /// Number of output rows (first dimension in row-major layout).
    rows: usize,
    /// Number of input columns (second dimension).
    cols: usize,
    /// Original quantization format.
    ggml_type: GgmlType,
}

impl QuantizedTensor {
    /// Load a weight tensor directly from GGUF, keeping raw quantized bytes.
    ///
    /// GGUF stores dimensions in column-major order (first dim varies fastest).
    /// We reverse them to match our row-major convention.
    pub fn load(model: &GgufModel, name: &str) -> Self {
        let info = model
            .get_tensor_info(name)
            .unwrap_or_else(|| panic!("Tensor '{}' not found in model", name));

        // Reverse dimensions: GGUF column-major → our row-major
        let dims: Vec<usize> = info.dimensions.iter().rev().map(|&d| d as usize).collect();
        let (rows, cols) = match dims.len() {
            1 => (dims[0], 1),
            2 => (dims[0], dims[1]),
            n => panic!("QuantizedTensor::load: unexpected {n}-D tensor '{name}'"),
        };

        let raw = model
            .tensor_data(name)
            .unwrap_or_else(|e| panic!("Failed to read tensor '{}': {}", name, e));

        Self {
            data: raw.to_vec(), // copy out of the mmap
            rows,
            cols,
            ggml_type: info.ggml_type,
        }
    }

    /// Build from raw quantized bytes. Used for benchmarks and testing.
    pub fn from_raw(data: Vec<u8>, rows: usize, cols: usize, ggml_type: GgmlType) -> Self {
        Self { data, rows, cols, ggml_type }
    }

    /// Build from a flat f32 slice (re-encoded as F32 bytes). Used in tests.
    pub fn from_f32(values: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols);
        let mut data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self { data, rows, cols, ggml_type: GgmlType::F32 }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Matrix-vector multiply: `[rows, cols] × [cols] → [rows]`.
    ///
    /// Dispatches to the quantized kernel for Q8_0/Q4_0,
    /// or falls back to on-the-fly dequantization for other types.
    pub fn matvec(&self, vec: &[f32]) -> Tensor {
        assert_eq!(
            vec.len(), self.cols,
            "matvec: vec length {} != cols {}",
            vec.len(), self.cols
        );
        let out = match self.ggml_type {
            GgmlType::Q8_0 => dispatch_q8_0(&self.data, self.rows, self.cols, vec),
            GgmlType::Q4_0 => dispatch_q4_0(&self.data, self.rows, self.cols, vec),
            GgmlType::Q4K  => dispatch_q4_k(&self.data, self.rows, self.cols, vec),
            GgmlType::Q5K  => dispatch_q5_k(&self.data, self.rows, self.cols, vec),
            GgmlType::Q6K  => dispatch_q6_k(&self.data, self.rows, self.cols, vec),
            _ => matvec_fallback(&self.data, self.ggml_type, self.rows, self.cols, vec),
        };
        Tensor::from_vec(out, &[self.rows])
    }

    /// Dequantize a single row and return it as a 1D `Tensor`.
    ///
    /// Used for embedding lookup (one token per call — cheap).
    pub fn row_as_f32(&self, row: usize) -> Tensor {
        assert!(row < self.rows, "row_as_f32: row {row} >= rows {}", self.rows);

        let n_elements = self.cols;
        let block_size = self.ggml_type.block_size();
        let type_size = self.ggml_type.type_size();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        let bytes_per_row = n_blocks * type_size;

        let row_bytes = &self.data[row * bytes_per_row..(row + 1) * bytes_per_row];
        let f32_data = dequantize(row_bytes, self.ggml_type, n_elements);
        Tensor::from_vec(f32_data, &[n_elements])
    }
}

// ── Dispatch ─────────────────────────────────────────────────────────────────
//
// Runtime CPU feature detection → SIMD if available, scalar fallback otherwise.

fn dispatch_q8_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just verified AVX2+FMA are available.
            return unsafe { crate::tensor::simd::matvec_q8_0_avx2(data, rows, cols, vec) };
        }
    }
    matvec_q8_0_scalar(data, rows, cols, vec)
}

fn dispatch_q4_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::tensor::simd::matvec_q4_k_avx2(data, rows, cols, vec) };
        }
    }
    matvec_q4_k_scalar(data, rows, cols, vec)
}

fn dispatch_q5_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::tensor::simd::matvec_q5_k_avx2(data, rows, cols, vec) };
        }
    }
    matvec_q5_k_scalar(data, rows, cols, vec)
}

fn dispatch_q6_k(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::tensor::simd::matvec_q6_k_avx2(data, rows, cols, vec) };
        }
    }
    matvec_q6_k_scalar(data, rows, cols, vec)
}

fn dispatch_q4_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::tensor::simd::matvec_q4_0_avx2(data, rows, cols, vec) };
        }
    }
    matvec_q4_0_scalar(data, rows, cols, vec)
}

// ── Scalar Kernels ───────────────────────────────────────────────────────────

/// Q8_0 matrix-vector multiply (scalar fallback).
///
/// Block layout (34 bytes per 32 elements):
///   [f16 scale (2 bytes)] [32 × i8 values]
///
/// For each output row, iterate over blocks, compute block_sum = Σ i8×f32,
/// then accumulate block_sum × scale. One scale multiply per block (32 elements).
pub(crate) fn matvec_q8_0_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8s)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
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
pub(crate) fn matvec_q4_0_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (f16 scale) + 16 (packed nibbles)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|i| {
            let row_start = i * bytes_per_row;
            let mut sum = 0.0f32;
            for b in 0..n_blocks {
                let block = &data[row_start + b * BLOCK_BYTES..];
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let mut block_sum = 0.0f32;
                for j in 0..BLOCK_ELEMS {
                    let byte = block[2 + j / 2];
                    let nibble = if j % 2 == 0 {
                        (byte & 0x0F) as i32
                    } else {
                        ((byte >> 4) & 0x0F) as i32
                    };
                    block_sum += (nibble - 8) as f32 * vec[b * BLOCK_ELEMS + j];
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
fn matvec_q4_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d    = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qs     = &b[16..144];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2,     scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;  let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;  let m1 = dmin * mn1 as f32;
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
fn matvec_q5_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let d    = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh     = &b[16..48];
                let qs     = &b[48..176];
                let x_base = sb * SUPER_BLOCK;
                for group in 0..4 {
                    let (sc0, mn0) = get_scale_min_q4k(group * 2,     scales);
                    let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
                    let d0 = d * sc0 as f32;  let m0 = dmin * mn0 as f32;
                    let d1 = d * sc1 as f32;  let m1 = dmin * mn1 as f32;
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
fn matvec_q6_k_scalar(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut acc = 0.0f32;
            let row = &data[r * bytes_per_row..];
            for sb in 0..n_super {
                let b = &row[sb * BLOCK_BYTES..];
                let ql     = &b[0..128];
                let qh     = &b[128..192];
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
                        let v1 = (ql[ql_off + l     ] & 0x0F) | (((qhl >> 0) & 0x03) << 4);
                        let v2 = (ql[ql_off + l + 32] & 0x0F) | (((qhl >> 2) & 0x03) << 4);
                        let v3 = (ql[ql_off + l     ] >>   4) | (((qhl >> 4) & 0x03) << 4);
                        let v4 = (ql[ql_off + l + 32] >>   4) | (((qhl >> 6) & 0x03) << 4);
                        let q1 = (v1 as i32 - 32) as f32;
                        let q2 = (v2 as i32 - 32) as f32;
                        let q3 = (v3 as i32 - 32) as f32;
                        let q4 = (v4 as i32 - 32) as f32;
                        let sc0 = sc_raw[sc_off + is    ] as i8 as f32;
                        let sc2 = sc_raw[sc_off + is + 2] as i8 as f32;
                        let sc4 = sc_raw[sc_off + is + 4] as i8 as f32;
                        let sc6 = sc_raw[sc_off + is + 6] as i8 as f32;
                        acc += d * sc0 * q1 * vec[x_group + l     ];
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
    let n_blocks = (cols + block_size - 1) / block_size;
    let bytes_per_row = n_blocks * type_size;

    (0..rows)
        .into_par_iter()
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

        let qt = QuantizedTensor { data, rows: 2, cols: 32, ggml_type: GgmlType::Q8_0 };
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let result = qt.matvec(&input);
        assert_eq!(result.shape(), &[2]);

        // Row 0: sum = Σ i*i for i in 0..32, scale=1 → 1.0 * 9920 = 9920
        let expected_row0: f32 = (0..32).map(|i| (i as f32) * (i as f32)).sum::<f32>();
        // Row 1: sum = Σ -i*i for i in 0..32, scale=2 → -19840
        let expected_row1: f32 = -2.0 * (0..32).map(|i| (i as f32) * (i as f32)).sum::<f32>();

        assert!((result.data()[0] - expected_row0).abs() < 0.1,
            "Row 0: got {}, expected {}", result.data()[0], expected_row0);
        assert!((result.data()[1] - expected_row1).abs() < 0.1,
            "Row 1: got {}, expected {}", result.data()[1], expected_row1);
    }

    #[test]
    fn test_row_as_f32_matches_dequantize() {
        // Single Q8_0 row: scale=0.5, weights=[10, -10, 5, -5, ...]
        let quants: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 10i8 } else { -10i8 }).collect();
        let block = make_q8_0_block(0.5, quants.clone());

        let qt = QuantizedTensor {
            data: block.clone(),
            rows: 1,
            cols: 32,
            ggml_type: GgmlType::Q8_0,
        };

        let row = qt.row_as_f32(0);
        assert_eq!(row.shape(), &[32]);

        for (i, &val) in row.data().iter().enumerate() {
            let expected = quants[i] as f32 * 0.5;
            assert!((val - expected).abs() < 1e-3,
                "Index {i}: got {val}, expected {expected}");
        }
    }

    #[test]
    fn test_quantized_uses_less_memory_than_f32() {
        // Q8_0: 34 bytes per 32 elements
        // F32:  128 bytes per 32 elements
        let block = make_q8_0_block(1.0, vec![1i8; 32]);
        let qt = QuantizedTensor { data: block, rows: 1, cols: 32, ggml_type: GgmlType::Q8_0 };
        let f32_equivalent_bytes = qt.rows * qt.cols * 4;
        assert!(qt.data.len() < f32_equivalent_bytes,
            "Quantized {} bytes should be less than f32 {} bytes",
            qt.data.len(), f32_equivalent_bytes);
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe {
                    crate::tensor::simd::matvec_q8_0_avx2(&data, rows, cols, &input)
                };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}", scalar[i], simd[i]
                    );
                }
            }
        }
    }

    /// Q4_K: verify the direct scalar kernel matches dequantize-then-dot.
    ///
    /// Super-block: d=1.0, dmin=0, all sub-block scales=1, mins=0,
    /// all nibbles=8 → each element = 8.0, expected sum = 256 × 8 = 2048.
    #[test]
    fn test_q4_k_matvec_matches_dequantize() {
        use crate::tensor::dequantize::dequantize;
        use crate::model::gguf::GgmlType;

        let mut block = vec![0u8; 144];
        // d = 1.0
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // dmin = 0 (zeros)
        // scales: want sc=1, mn=0 for all 8 sub-blocks
        // j<4: block[4+j] = 1 (sc), block[4+j+4] = 0 (mn)
        for j in 0..4usize { block[4 + j] = 1; }
        // j=4..8: block[4+j+4] = 1 gives sc=1 and mn=0 when upper nibble is 0
        for j in 4..8usize { block[4 + j + 4] = 1; }
        // qs: all 0x88 → both nibbles = 8
        for i in 16..144 { block[i] = 0x88; }

        let input = vec![1.0f32; 256];

        // Reference: dequantize → dot product
        let deq = dequantize(&block, GgmlType::Q4K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q4_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q4_K matvec: got {}, expected {}", result[0], expected
        );
        assert!(
            (result[0] - 2048.0).abs() < 1.0,
            "Q4_K expected ≈2048, got {}", result[0]
        );
    }

    /// Q5_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// All qs nibbles and qh bits zero → all 5-bit values = 0.
    /// Expected output = 0.
    #[test]
    fn test_q5_k_matvec_matches_dequantize() {
        use crate::tensor::dequantize::dequantize;
        use crate::model::gguf::GgmlType;

        let mut block = vec![0u8; 176];
        // d = 1.0, dmin = 0
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        // scales: sc=1 for all sub-blocks (same encoding as Q4_K)
        for j in 0..4usize { block[4 + j] = 1; }
        for j in 4..8usize { block[4 + j + 4] = 1; }
        // qh and qs all zero → all values = d * sc * 0 - 0 = 0
        // Input is non-zero so we'd see a non-zero result if values were non-zero
        let input: Vec<f32> = (0..256).map(|i| (i % 7) as f32 * 0.1).collect();

        let deq = dequantize(&block, GgmlType::Q5K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q5_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 1e-4,
            "Q5_K matvec: got {}, expected {}", result[0], expected
        );
    }

    /// Q6_K: verify scalar kernel matches dequantize-then-dot.
    ///
    /// All ql = 0x33 (nibbles=3,3), all qh = 0, all scales = 1, d = 1.0.
    /// Each 6-bit value = (3 | 0) - 32 = -29.
    #[test]
    fn test_q6_k_matvec_matches_dequantize() {
        use crate::tensor::dequantize::dequantize;
        use crate::model::gguf::GgmlType;

        let mut block = vec![0u8; 210];
        // ql (0..128): nibbles = 3 → 0x33
        for i in 0..128 { block[i] = 0x33; }
        // qh (128..192): all 0 — no high bits
        // scales (192..208): all 1 (i8 = 1)
        for i in 192..208 { block[i] = 1; }
        // d (208..210) = 1.0
        block[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let input = vec![1.0f32; 256];

        let deq = dequantize(&block, GgmlType::Q6K, 256);
        let expected: f32 = deq.iter().zip(&input).map(|(a, b)| a * b).sum();

        let result = matvec_q6_k_scalar(&block, 1, 256, &input);
        assert!(
            (result[0] - expected).abs() < 0.5,
            "Q6_K matvec: got {}, expected {}", result[0], expected
        );
        // Each 6-bit value = 3 - 32 = -29, scale=1, d=1. Sum = 256 * -29 = -7424.
        assert!(
            (result[0] - (-7424.0_f32)).abs() < 1.0,
            "Q6_K expected ≈-7424, got {}", result[0]
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
            for j in 0..4usize { block[4 + j] = (r as u8 + 1) & 63; }
            for j in 0..4usize { block[4 + j + 4] = 1u8 & 63; }
            for j in 4..8usize { block[4 + j + 4] = ((r as u8 + 2) & 0xF) | (1u8 << 4); }
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe {
                    crate::tensor::simd::matvec_q4_k_avx2(&data, rows, cols, &input)
                };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}", scalar[i], simd[i]
                    );
                }
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
            for j in 0..4usize { block[4 + j] = (r as u8 + 1) & 63; }
            for j in 0..4usize { block[4 + j + 4] = 1u8 & 63; }
            for j in 4..8usize { block[4 + j + 4] = ((r as u8 + 2) & 0xF) | (1u8 << 4); }
            // qh (16..48): varied patterns so not all 5th bits are 0
            for i in 0..32 { block[16 + i] = ((r * 13 + i * 7 + 3) % 256) as u8; }
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe {
                    crate::tensor::simd::matvec_q5_k_avx2(&data, rows, cols, &input)
                };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}", scalar[i], simd[i]
                    );
                }
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe {
                    crate::tensor::simd::matvec_q6_k_avx2(&data, rows, cols, &input)
                };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}", scalar[i], simd[i]
                    );
                }
            }
        }
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let simd = unsafe {
                    crate::tensor::simd::matvec_q4_0_avx2(&data, rows, cols, &input)
                };
                for i in 0..rows {
                    assert!(
                        (scalar[i] - simd[i]).abs() < 1e-3,
                        "Row {i}: scalar={}, simd={}", scalar[i], simd[i]
                    );
                }
            }
        }
    }
}
