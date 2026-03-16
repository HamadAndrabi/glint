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

use crate::model::gguf::{GgmlType, GgufModel};
use super::tensor::Tensor;
use super::dequantize::dequantize;

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
            GgmlType::Q8_0 => matvec_q8_0(&self.data, self.rows, self.cols, vec),
            GgmlType::Q4_0 => matvec_q4_0(&self.data, self.rows, self.cols, vec),
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

// ── Quantized Kernels ────────────────────────────────────────────────────────

/// Q8_0 matrix-vector multiply.
///
/// Block layout (34 bytes per 32 elements):
///   [f16 scale (2 bytes)] [32 × i8 values]
///
/// For each output row, iterate over blocks, compute block_sum = Σ i8×f32,
/// then accumulate block_sum × scale. One scale multiply per block (32 elements).
fn matvec_q8_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8s)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    let mut out = vec![0.0f32; rows];

    for i in 0..rows {
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
        out[i] = sum;
    }
    out
}

/// Q4_0 matrix-vector multiply.
///
/// Block layout (18 bytes per 32 elements):
///   [f16 scale (2 bytes)] [16 bytes of packed nibbles, 2 per byte]
///
/// Nibble values are unsigned (0–15), centered by subtracting 8 → [-8, +7].
fn matvec_q4_0(data: &[u8], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (f16 scale) + 16 (packed nibbles)

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    let mut out = vec![0.0f32; rows];

    for i in 0..rows {
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
        out[i] = sum;
    }
    out
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

    let mut out = vec![0.0f32; rows];
    for i in 0..rows {
        let row_bytes = &data[i * bytes_per_row..(i + 1) * bytes_per_row];
        let row_f32 = dequantize(row_bytes, ggml_type, cols);
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += row_f32[j] * vec[j];
        }
        out[i] = sum;
    }
    out
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
}
