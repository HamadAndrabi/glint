//! Dequantization — convert quantized GGUF tensor data to f32.
//!
//! GGUF stores weights in quantized formats to save memory. Before we can
//! do math, we need to convert them to f32. Phase 2 will add direct
//! quantized math; for now we dequantize everything upfront.

use half::f16;

use crate::model::gguf::{GgmlType, GgufModel};
use super::tensor::Tensor;

/// Dequantize raw bytes into f32 values based on the ggml type.
pub fn dequantize(data: &[u8], ggml_type: GgmlType, n_elements: usize) -> Vec<f32> {
    match ggml_type {
        GgmlType::F32 => dequantize_f32(data, n_elements),
        GgmlType::F16 => dequantize_f16(data, n_elements),
        GgmlType::BF16 => dequantize_bf16(data, n_elements),
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        _ => unimplemented!("Dequantization not yet implemented for {}", ggml_type),
    }
}

/// F32 — no conversion needed, just reinterpret bytes.
fn dequantize_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(data.len() >= n_elements * 4);
    let mut out = vec![0.0f32; n_elements];
    for i in 0..n_elements {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        out[i] = f32::from_le_bytes(bytes);
    }
    out
}

/// F16 → f32 conversion.
fn dequantize_f16(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(data.len() >= n_elements * 2);
    let mut out = vec![0.0f32; n_elements];
    for i in 0..n_elements {
        let bytes = [data[i * 2], data[i * 2 + 1]];
        out[i] = f16::from_le_bytes(bytes).to_f32();
    }
    out
}

/// BF16 → f32 conversion (bfloat16 — same exponent range as f32, truncated mantissa).
fn dequantize_bf16(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(data.len() >= n_elements * 2);
    let mut out = vec![0.0f32; n_elements];
    for i in 0..n_elements {
        // BF16 is the upper 16 bits of f32, so we pad with zeros
        let bytes = [0u8, 0u8, data[i * 2], data[i * 2 + 1]];
        out[i] = f32::from_le_bytes(bytes);
    }
    out
}

/// Q8_0 dequantization.
///
/// Block layout (34 bytes per block of 32 elements):
///   [f16 scale] [32 × int8 values]
///
/// Dequantized value = int8_value * scale
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (scale) + 32 (int8s)

    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let block_data = &data[block * BLOCK_BYTES..];

        // First 2 bytes: f16 scale factor
        let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();

        // Next 32 bytes: int8 quantized values
        let quants = &block_data[2..2 + BLOCK_SIZE];
        let remaining = (n_elements - block * BLOCK_SIZE).min(BLOCK_SIZE);

        for i in 0..remaining {
            out.push(quants[i] as i8 as f32 * scale);
        }
    }
    out
}

/// Q4_0 dequantization.
///
/// Block layout (18 bytes per block of 32 elements):
///   [f16 scale] [16 × packed bytes, each containing two 4-bit values]
///
/// The 4-bit values are unsigned [0..15], centered by subtracting 8
/// to get signed range [-8..7]. Then multiplied by scale.
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (packed nibbles)

    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let block_data = &data[block * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
        let quants = &block_data[2..2 + 16];
        let remaining = (n_elements - block * BLOCK_SIZE).min(BLOCK_SIZE);

        for i in 0..remaining {
            let byte = quants[i / 2];
            // Low nibble for even indices, high nibble for odd
            let nibble = if i % 2 == 0 {
                (byte & 0x0F) as i32
            } else {
                ((byte >> 4) & 0x0F) as i32
            };
            // Center: subtract 8 to get signed range [-8, 7]
            out.push((nibble - 8) as f32 * scale);
        }
    }
    out
}

/// Load a tensor from a GGUF model as f32, dequantizing if needed.
///
/// GGUF dimensions are column-major (first dim = fastest-varying),
/// so we reverse them to match our row-major Tensor layout.
pub fn load_tensor_f32(model: &GgufModel, name: &str) -> Tensor {
    let info = model
        .get_tensor_info(name)
        .unwrap_or_else(|| panic!("Tensor '{}' not found in model", name));

    let n_elements = info.n_elements() as usize;
    // Reverse dimensions: GGUF is column-major, we are row-major
    let shape: Vec<usize> = info.dimensions.iter().rev().map(|&d| d as usize).collect();

    let raw_data = model
        .tensor_data(name)
        .unwrap_or_else(|e| panic!("Failed to read tensor '{}': {}", name, e));

    let f32_data = dequantize(raw_data, info.ggml_type, n_elements);
    Tensor::from_vec(f32_data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_f32() {
        let values: Vec<f32> = vec![1.0, -2.5, 3.14];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = dequantize_f32(&data, 3);
        assert_eq!(result, values);
    }

    #[test]
    fn test_dequantize_f16() {
        let values: Vec<f32> = vec![1.0, -2.0, 0.5];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        let result = dequantize_f16(&data, 3);
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{} vs {}", a, b);
        }
    }

    #[test]
    fn test_dequantize_q8_0() {
        // Construct one block: scale=2.0, values=[1, -1, 2, -2, ...]
        let mut data = Vec::new();
        let scale = f16::from_f32(2.0);
        data.extend_from_slice(&scale.to_le_bytes());

        let mut quants = [0i8; 32];
        for i in 0..32 {
            quants[i] = if i % 2 == 0 { (i / 2) as i8 } else { -((i / 2) as i8) };
        }
        for &q in &quants {
            data.push(q as u8);
        }

        let result = dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            let expected = quants[i] as f32 * 2.0;
            assert!(
                (result[i] - expected).abs() < 1e-3,
                "index {}: {} vs {}",
                i, result[i], expected
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0() {
        // Construct one block: scale=1.0, all nibbles = 8 (which centers to 0)
        let mut data = Vec::new();
        let scale = f16::from_f32(1.0);
        data.extend_from_slice(&scale.to_le_bytes());

        // 16 bytes, each 0x88 → low nibble=8, high nibble=8 → both center to 0
        for _ in 0..16 {
            data.push(0x88);
        }

        let result = dequantize_q4_0(&data, 32);
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 0.0).abs() < 1e-3, "index {}: expected 0.0, got {}", i, v);
        }
    }
}
