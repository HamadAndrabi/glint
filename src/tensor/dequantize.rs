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
        GgmlType::Q4K  => dequantize_q4_k(data, n_elements),
        GgmlType::Q5K  => dequantize_q5_k(data, n_elements),
        GgmlType::Q6K  => dequantize_q6_k(data, n_elements),
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

// ── K-Quant helpers ──────────────────────────────────────────────────────────

/// Extract one (scale, min) pair from the 12-byte k-quant scale buffer.
///
/// The 12 bytes encode 8 scale values and 8 min values, each 6 bits wide.
/// This mirrors `get_scale_min_k4` from llama.cpp ggml-quants.c.
///
/// Returns `(scale, min)` as raw u8 values (multiply by the super-block d/dmin
/// to get the real floating-point scale and min).
pub(super) fn get_scale_min_q4k(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >>   4) | ((scales[j    ] >> 6) << 4),
        )
    }
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

/// Q4_K dequantization.
///
/// Super-block layout (144 bytes per 256 elements):
///   [f16 d (2B)] [f16 dmin (2B)] [scales u8 × 12] [qs u8 × 128]
///
/// 8 sub-blocks of 32 elements. Sub-block scales are packed 6 bits each into
/// the 12 `scales` bytes via `get_scale_min_q4k`. The 128 qs bytes hold 256
/// nibbles; low nibbles map to sub-blocks 0,2,4,6 and high nibbles to 1,3,5,7.
fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let n_blocks = (n_elements + SUPER_BLOCK - 1) / SUPER_BLOCK;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let d    = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];   // 12 bytes
        let qs     = &b[16..144]; // 128 bytes

        let remaining = (n_elements - block * SUPER_BLOCK).min(SUPER_BLOCK);
        let mut emitted = 0;

        for group in 0..4 {
            let (sc0, mn0) = get_scale_min_q4k(group * 2,     scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;  let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;  let m1 = dmin * mn1 as f32;
            let q_base = group * 32;

            // Sub-block 2*group: low nibbles of qs[q_base..q_base+32]
            for l in 0..32 {
                if emitted >= remaining { break; }
                out.push(d0 * (qs[q_base + l] & 0x0F) as f32 - m0);
                emitted += 1;
            }
            // Sub-block 2*group+1: high nibbles of qs[q_base..q_base+32]
            for l in 0..32 {
                if emitted >= remaining { break; }
                out.push(d1 * (qs[q_base + l] >> 4) as f32 - m1);
                emitted += 1;
            }
        }
    }

    out
}

/// Q5_K dequantization.
///
/// Super-block layout (176 bytes per 256 elements):
///   [f16 d (2B)] [f16 dmin (2B)] [scales u8 × 12] [qh u8 × 32] [qs u8 × 128]
///
/// Same sub-block structure as Q4_K, but each nibble gains a 5th bit from
/// the packed high-bit array `qh`.
fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let n_blocks = (n_elements + SUPER_BLOCK - 1) / SUPER_BLOCK;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let d    = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];   // 12 bytes
        let qh     = &b[16..48];  // 32 bytes  — high bits
        let qs     = &b[48..176]; // 128 bytes — low nibbles

        let remaining = (n_elements - block * SUPER_BLOCK).min(SUPER_BLOCK);
        let mut emitted = 0;

        for group in 0..4 {
            let (sc0, mn0) = get_scale_min_q4k(group * 2,     scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;  let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;  let m1 = dmin * mn1 as f32;
            let q_base = group * 32;
            // Bit masks into qh: u1 selects the high bit for low nibbles,
            // u2 for high nibbles. They shift left by 2 each group.
            let u1: u8 = 1 << (group * 2);
            let u2: u8 = 2 << (group * 2);

            for l in 0..32 {
                if emitted >= remaining { break; }
                let lo = (qs[q_base + l] & 0x0F) as f32;
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d0 * (lo + hi) - m0);
                emitted += 1;
            }
            for l in 0..32 {
                if emitted >= remaining { break; }
                let lo = (qs[q_base + l] >> 4) as f32;
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * (lo + hi) - m1);
                emitted += 1;
            }
        }
    }

    out
}

/// Q6_K dequantization.
///
/// Super-block layout (210 bytes per 256 elements):
///   [ql u8 × 128] [qh u8 × 64] [scales i8 × 16] [f16 d (2B)]
///
/// 16 sub-blocks of 16 elements. Each element is a 6-bit signed integer in
/// the range [-32, +31] assembled from 4 bits of `ql` and 2 bits of `qh`.
fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let n_blocks = (n_elements + SUPER_BLOCK - 1) / SUPER_BLOCK;
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    // Pre-allocate with zeros so we can scatter-write within each super-block
    let mut out = vec![0.0f32; n_blocks * SUPER_BLOCK];

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let ql     = &b[0..128];    // low 4 bits
        let qh     = &b[128..192];  // high 2 bits
        let sc_raw = &b[192..208];  // i8 sub-block scales
        let d = f16::from_le_bytes([b[208], b[209]]).to_f32();

        let out_base = block * SUPER_BLOCK;

        // Two groups of 128 elements each
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let y_base = out_base + group * 128;

            for l in 0..32 {
                let is = l / 16;
                let qhl = qh[qh_off + l];

                // Assemble four 6-bit values from this l-offset
                let v1 = (ql[ql_off + l     ] & 0x0F) | (((qhl >> 0) & 0x03) << 4);
                let v2 = (ql[ql_off + l + 32] & 0x0F) | (((qhl >> 2) & 0x03) << 4);
                let v3 = (ql[ql_off + l     ] >>   4) | (((qhl >> 4) & 0x03) << 4);
                let v4 = (ql[ql_off + l + 32] >>   4) | (((qhl >> 6) & 0x03) << 4);

                // Center: subtract 32 to get range [-32, +31]
                let q1 = v1 as i32 - 32;
                let q2 = v2 as i32 - 32;
                let q3 = v3 as i32 - 32;
                let q4 = v4 as i32 - 32;

                let sc0 = sc_raw[sc_off + is    ] as i8 as f32;
                let sc2 = sc_raw[sc_off + is + 2] as i8 as f32;
                let sc4 = sc_raw[sc_off + is + 4] as i8 as f32;
                let sc6 = sc_raw[sc_off + is + 6] as i8 as f32;

                // Scatter to non-contiguous output positions (mirrors llama.cpp layout)
                out[y_base + l     ] = d * sc0 * q1 as f32;
                out[y_base + l + 32] = d * sc2 * q2 as f32;
                out[y_base + l + 64] = d * sc4 * q3 as f32;
                out[y_base + l + 96] = d * sc6 * q4 as f32;
            }
        }
    }

    out.truncate(n_elements);
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

/// Test helpers for building quantized blocks in unit tests.
/// Exported so `quantized.rs` tests can reuse them.
#[cfg(test)]
pub mod dequantize_test_helpers {
    use half::f16;

    /// Build one Q8_0 block: [f16 scale (2 bytes)] + [32 × i8 values].
    pub fn make_q8_0_block(scale: f32, quants: Vec<i8>) -> Vec<u8> {
        assert_eq!(quants.len(), 32);
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for q in quants {
            block.push(q as u8);
        }
        block
    }
}
