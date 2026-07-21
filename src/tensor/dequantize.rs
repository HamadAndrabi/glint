//! Dequantization — convert quantized GGUF tensor data to f32.
//!
//! GGUF stores weights in quantized formats to save memory. Before we can
//! do math, we need to convert them to f32. Phase 2 will add direct
//! quantized math; for now we dequantize everything upfront.

use half::f16;

use super::tensor::Tensor;
use crate::error::GlintError;
use crate::model::gguf::{GgmlType, GgufModel};

/// Dequantize raw bytes into f32 values based on the ggml type.
pub fn dequantize(data: &[u8], ggml_type: GgmlType, n_elements: usize) -> Vec<f32> {
    match ggml_type {
        GgmlType::F32 => dequantize_f32(data, n_elements),
        GgmlType::F16 => dequantize_f16(data, n_elements),
        GgmlType::BF16 => dequantize_bf16(data, n_elements),
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlType::Q5_0 => dequantize_q5_0(data, n_elements),
        GgmlType::Q5_1 => dequantize_q5_1(data, n_elements),
        GgmlType::Q4K => dequantize_q4_k(data, n_elements),
        GgmlType::Q5K => dequantize_q5_k(data, n_elements),
        GgmlType::Q6K => dequantize_q6_k(data, n_elements),
        GgmlType::Q2K => dequantize_q2_k(data, n_elements),
        GgmlType::Q3K => dequantize_q3_k(data, n_elements),
        GgmlType::IQ4NL => dequantize_iq4_nl(data, n_elements),
        _ => panic!(
            "{}",
            GlintError::UnsupportedQuantization(ggml_type.to_string())
        ),
    }
}

/// F32 — no conversion needed, just reinterpret bytes.
fn dequantize_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(data.len() >= n_elements * 4);
    let mut out = vec![0.0f32; n_elements];
    for i in 0..n_elements {
        let bytes = [
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ];
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

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let block_data = &data[block * BLOCK_BYTES..];

        // First 2 bytes: f16 scale factor
        let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();

        // Next 32 bytes: int8 quantized values
        let quants = &block_data[2..2 + BLOCK_SIZE];
        let remaining = (n_elements - block * BLOCK_SIZE).min(BLOCK_SIZE);

        for q in quants.iter().take(remaining) {
            out.push(*q as i8 as f32 * scale);
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
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
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

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
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

    let n_blocks = n_elements.div_ceil(SUPER_BLOCK);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16]; // 12 bytes
        let qs = &b[16..144]; // 128 bytes

        let remaining = (n_elements - block * SUPER_BLOCK).min(SUPER_BLOCK);
        let mut emitted = 0;

        for group in 0..4 {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let q_base = group * 32;

            // Sub-block 2*group: low nibbles of qs[q_base..q_base+32]
            for l in 0..32 {
                if emitted >= remaining {
                    break;
                }
                out.push(d0 * (qs[q_base + l] & 0x0F) as f32 - m0);
                emitted += 1;
            }
            // Sub-block 2*group+1: high nibbles of qs[q_base..q_base+32]
            for l in 0..32 {
                if emitted >= remaining {
                    break;
                }
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

    let n_blocks = n_elements.div_ceil(SUPER_BLOCK);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_elements);

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16]; // 12 bytes
        let qh = &b[16..48]; // 32 bytes  — high bits
        let qs = &b[48..176]; // 128 bytes — low nibbles

        let remaining = (n_elements - block * SUPER_BLOCK).min(SUPER_BLOCK);
        let mut emitted = 0;

        for group in 0..4 {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let q_base = group * 32;
            // Bit masks into qh: u1 selects the high bit for low nibbles,
            // u2 for high nibbles. They shift left by 2 each group.
            let u1: u8 = 1 << (group * 2);
            let u2: u8 = 2 << (group * 2);

            for l in 0..32 {
                if emitted >= remaining {
                    break;
                }
                let lo = (qs[q_base + l] & 0x0F) as f32;
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d0 * (lo + hi) - m0);
                emitted += 1;
            }
            for l in 0..32 {
                if emitted >= remaining {
                    break;
                }
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

    let n_blocks = n_elements.div_ceil(SUPER_BLOCK);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    // Pre-allocate with zeros so we can scatter-write within each super-block
    let mut out = vec![0.0f32; n_blocks * SUPER_BLOCK];

    for block in 0..n_blocks {
        let b = &data[block * BLOCK_BYTES..];
        let ql = &b[0..128]; // low 4 bits
        let qh = &b[128..192]; // high 2 bits
        let sc_raw = &b[192..208]; // i8 sub-block scales
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
                let v1 = (ql[ql_off + l] & 0x0F) | ((qhl & 0x03) << 4);
                let v2 = (ql[ql_off + l + 32] & 0x0F) | (((qhl >> 2) & 0x03) << 4);
                let v3 = (ql[ql_off + l] >> 4) | (((qhl >> 4) & 0x03) << 4);
                let v4 = (ql[ql_off + l + 32] >> 4) | (((qhl >> 6) & 0x03) << 4);

                // Center: subtract 32 to get range [-32, +31]
                let q1 = v1 as i32 - 32;
                let q2 = v2 as i32 - 32;
                let q3 = v3 as i32 - 32;
                let q4 = v4 as i32 - 32;

                let sc0 = sc_raw[sc_off + is] as i8 as f32;
                let sc2 = sc_raw[sc_off + is + 2] as i8 as f32;
                let sc4 = sc_raw[sc_off + is + 4] as i8 as f32;
                let sc6 = sc_raw[sc_off + is + 6] as i8 as f32;

                // Scatter to non-contiguous output positions (mirrors llama.cpp layout)
                out[y_base + l] = d * sc0 * q1 as f32;
                out[y_base + l + 32] = d * sc2 * q2 as f32;
                out[y_base + l + 64] = d * sc4 * q3 as f32;
                out[y_base + l + 96] = d * sc6 * q4 as f32;
            }
        }
    }

    out.truncate(n_elements);
    out
}

/// Unpack one Q2_K super-block (84 bytes) into 256 f32 values.
///
/// Mirrors `dequantize_row_q2_K` in ggml-quants.c exactly. The 2-bit quants
/// are stored in *shift planes*, not byte-major order: within each 128-element
/// half, four passes over the same 32 `qs` bytes extract bits (0,1), (2,3),
/// (4,5), (6,7), and each pass covers two 16-element strips (`q[l]`, then
/// `q[l+16]`) with their own 4-bit scale/min pair.
///
/// Anchored externally by `ggml_reference_tests::q2_k_matches_ggml_reference`.
pub(crate) fn unpack_q2_k_block(b: &[u8], out: &mut [f32; 256]) {
    let scales = &b[0..16];
    let qs = &b[16..80];
    let d = f16::from_le_bytes([b[80], b[81]]).to_f32();
    let dmin = f16::from_le_bytes([b[82], b[83]]).to_f32();

    let mut y = 0;
    let mut is = 0;
    for half in 0..2 {
        let q = &qs[32 * half..32 * (half + 1)];
        for j in 0..4 {
            let shift = 2 * j;
            for strip in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let qv = ((q[strip * 16 + l] >> shift) & 3) as f32;
                    out[y] = dl * qv - ml;
                    y += 1;
                }
            }
        }
    }
}

/// Q2_K dequantization.
///
/// Super-block layout (84 bytes per 256 elements):
///   [scales u8×16] [qs u8×64] [f16 d] [f16 dmin]
///
/// See [`unpack_q2_k_block`] for the element layout.
fn dequantize_q2_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 84;

    let n_blocks = n_elements.div_ceil(SUPER_BLOCK);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_blocks * SUPER_BLOCK);
    let mut buf = [0.0f32; SUPER_BLOCK];
    for block in 0..n_blocks {
        unpack_q2_k_block(&data[block * BLOCK_BYTES..], &mut buf);
        out.extend_from_slice(&buf);
    }
    out.truncate(n_elements);
    out
}

/// Unpack one Q3_K super-block (110 bytes) into 256 f32 values.
///
/// Mirrors `dequantize_row_q3_K` in ggml-quants.c exactly:
///
/// * **Scales** — 16 signed 6-bit values: the low 4 bits live in nibble
///   planes over bytes 0..8 (`sc_raw[j] & 0xF` for j<8, `sc_raw[j-8] >> 4`
///   for j≥8), the high 2 bits in 2-bit planes over bytes 8..12
///   (`sc_raw[8 + j%4] >> (2*(j/4))`). Signed value = raw − 32.
/// * **Quants** — 2-bit shift planes, same shape as Q2_K: per 128-element
///   half, four passes over 32 `qs` bytes at shifts 0/2/4/6, two 16-element
///   strips per pass.
/// * **High bits** — `hmask` bit `b` of byte `strip*16 + l`, where `b`
///   advances once per (half, shift) pass — bits 0..3 in the first half,
///   4..7 in the second. A **set** bit means "no −4 offset".
///
/// Anchored externally by `ggml_reference_tests::q3_k_matches_ggml_reference`.
pub(crate) fn unpack_q3_k_block(b: &[u8], out: &mut [f32; 256]) {
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let sc_raw = &b[96..108];
    let d_all = f16::from_le_bytes([b[108], b[109]]).to_f32();

    let mut scales = [0i32; 16];
    for (j, s) in scales.iter_mut().enumerate() {
        let low4 = if j < 8 {
            sc_raw[j] & 0xF
        } else {
            sc_raw[j - 8] >> 4
        };
        let high2 = (sc_raw[8 + (j % 4)] >> (2 * (j / 4))) & 3;
        *s = ((low4 | (high2 << 4)) as i32) - 32;
    }

    let mut y = 0;
    let mut is = 0;
    let mut mbit = 0u32; // advances once per (half, shift) pass: bits 0..7
    for half in 0..2 {
        let q = &qs[32 * half..32 * (half + 1)];
        for j in 0..4 {
            let shift = 2 * j;
            for strip in 0..2 {
                let dl = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let idx = strip * 16 + l;
                    let lo = ((q[idx] >> shift) & 3) as i32;
                    let qv = if hmask[idx] & (1 << mbit) != 0 {
                        lo
                    } else {
                        lo - 4
                    };
                    out[y] = dl * qv as f32;
                    y += 1;
                }
            }
            mbit += 1;
        }
    }
}

/// Q3_K dequantization.
///
/// Super-block layout (110 bytes per 256 elements):
///   [hmask u8×32] [qs u8×64] [scales u8×12] [f16 d]
///
/// See [`unpack_q3_k_block`] for the element layout.
fn dequantize_q3_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 110;

    let n_blocks = n_elements.div_ceil(SUPER_BLOCK);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_blocks * SUPER_BLOCK);
    let mut buf = [0.0f32; SUPER_BLOCK];
    for block in 0..n_blocks {
        unpack_q3_k_block(&data[block * BLOCK_BYTES..], &mut buf);
        out.extend_from_slice(&buf);
    }
    out.truncate(n_elements);
    out
}

/// Unpack one Q5_0 block (22 bytes) into 32 f32 values.
///
/// Mirrors `dequantize_row_q5_0` in ggml-quants.c exactly. Block layout:
/// `[f16 d] [qh u8×4] [qs u8×16]`. The 5th bits live in the little-endian
/// u32 `qh`: bit `j` completes the low nibble of byte `j` (element `j`),
/// bit `j+16` completes the high nibble (element `j+16`) — the ggml code
/// reaches the high-half bits via `(qh >> (j + 12)) & 0x10`. Split-plane
/// nibbles like Q4_0/IQ4_NL; values are centered by −16.
///
/// Anchored externally by `ggml_reference_tests::q5_0_matches_ggml_reference`.
pub(crate) fn unpack_q5_0_block(b: &[u8], out: &mut [f32; 32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((qs[j] & 0xF) as u32 | xh_0) as i32 - 16;
        let x1 = ((qs[j] >> 4) as u32 | xh_1) as i32 - 16;
        out[j] = x0 as f32 * d;
        out[j + 16] = x1 as f32 * d;
    }
}

/// Q5_0 dequantization. See [`unpack_q5_0_block`] for the layout.
fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_blocks * BLOCK_SIZE);
    let mut buf = [0.0f32; BLOCK_SIZE];
    for block in 0..n_blocks {
        unpack_q5_0_block(&data[block * BLOCK_BYTES..], &mut buf);
        out.extend_from_slice(&buf);
    }
    out.truncate(n_elements);
    out
}

/// Unpack one Q5_1 block (24 bytes) into 32 f32 values.
///
/// Mirrors `dequantize_row_q5_1` in ggml-quants.c exactly. Block layout:
/// `[f16 d] [f16 m] [qh u8×4] [qs u8×16]` — same 5th-bit scheme as Q5_0 but
/// affine (`x*d + m`) instead of centered.
///
/// Anchored externally by `ggml_reference_tests::q5_1_matches_ggml_reference`.
pub(crate) fn unpack_q5_1_block(b: &[u8], out: &mut [f32; 32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((qs[j] & 0xF) as u32 | xh_0) as f32;
        let x1 = ((qs[j] >> 4) as u32 | xh_1) as f32;
        out[j] = x0 * d + m;
        out[j + 16] = x1 * d + m;
    }
}

/// Q5_1 dequantization. See [`unpack_q5_1_block`] for the layout.
fn dequantize_q5_1(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_blocks * BLOCK_SIZE);
    let mut buf = [0.0f32; BLOCK_SIZE];
    for block in 0..n_blocks {
        unpack_q5_1_block(&data[block * BLOCK_BYTES..], &mut buf);
        out.extend_from_slice(&buf);
    }
    out.truncate(n_elements);
    out
}

/// The IQ4_NL non-linear codebook (kvalues_iq4nl in ggml-common.h).
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Unpack one IQ4_NL block (18 bytes) into 32 f32 values.
///
/// Mirrors `dequantize_row_iq4_nl` in ggml-quants.c exactly: the nibbles are
/// stored in *split planes* — the low nibbles of bytes 0..16 are elements
/// 0..16 and the high nibbles are elements 16..32 (NOT interleaved per byte
/// like a naive reading would suggest).
///
/// Anchored externally by `ggml_reference_tests::iq4_nl_matches_ggml_reference`.
pub(crate) fn unpack_iq4_nl_block(b: &[u8], out: &mut [f32; 32]) {
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let qs = &b[2..18];
    for j in 0..16 {
        out[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

/// IQ4_NL dequantization.
///
/// Block layout (18 bytes per 32 elements — same size as Q4_0):
///   [f16 d] [qs u8×16]
///
/// See [`unpack_iq4_nl_block`] for the element layout.
fn dequantize_iq4_nl(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    assert!(data.len() >= n_blocks * BLOCK_BYTES);

    let mut out = Vec::with_capacity(n_blocks * BLOCK_SIZE);
    let mut buf = [0.0f32; BLOCK_SIZE];
    for block in 0..n_blocks {
        unpack_iq4_nl_block(&data[block * BLOCK_BYTES..], &mut buf);
        out.extend_from_slice(&buf);
    }
    out.truncate(n_elements);
    out
}

/// Load a tensor from a GGUF model as f32, dequantizing if needed.
///
/// GGUF dimensions are column-major (first dim = fastest-varying),
/// so we reverse them to match our row-major Tensor layout.
pub fn load_tensor_f32(model: &GgufModel, name: &str) -> Result<Tensor, GlintError> {
    let info = model
        .get_tensor_info(name)
        .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;

    let n_elements = info.n_elements() as usize;
    // Reverse dimensions: GGUF is column-major, we are row-major
    let shape: Vec<usize> = info.dimensions.iter().rev().map(|&d| d as usize).collect();

    let raw_data = model
        .tensor_data(name)
        .map_err(|e| GlintError::TensorReadError {
            name: name.to_string(),
            detail: e.to_string(),
        })?;

    let f32_data = dequantize(raw_data, info.ggml_type, n_elements);
    Ok(Tensor::from_vec(f32_data, &shape))
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
            quants[i] = if i % 2 == 0 {
                (i / 2) as i8
            } else {
                -((i / 2) as i8)
            };
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
                i,
                result[i],
                expected
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
            assert!(
                (v - 0.0).abs() < 1e-3,
                "index {}: expected 0.0, got {}",
                i,
                v
            );
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

/// Golden vectors derived from the ggml reference implementation.
///
/// These tests exist to break a circularity: the K-quant and IQ4_NL matvec
/// kernels are validated against `dequantize`, so `dequantize` itself must be
/// anchored to something *external*. The expected values below were computed
/// by an independent transcription of `dequantize_row_q{2,3,4,5,6}_K` /
/// `dequantize_row_iq4_nl` from llama.cpp `ggml/src/ggml-quants.c` (fetched
/// 2026-07-04) using the block layouts in `ggml-common.h`. Generator script:
/// `scripts/gen_ggml_vectors.py` regenerates them.
///
/// Inputs are procedural (`patterned`) so the vectors are reproducible;
/// super-block scales are d = 0.5, dmin = 0.25, which makes every expected
/// value a small dyadic rational — exactly representable in f32, so the
/// comparisons below are exact, not tolerance-based.
#[cfg(test)]
mod ggml_reference_tests {
    use super::*;
    use crate::model::gguf::GgmlType;

    /// Deterministic byte pattern shared with the generator script.
    fn patterned(len: usize, start: usize, a: usize, c: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (((start + i) * a + c) & 0xFF) as u8)
            .collect()
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    const D: f32 = 0.5;
    const DMIN: f32 = 0.25;
    /// Two super-blocks worth of elements for the K-quants.
    const N: usize = 512;

    fn check(vals: &[f32], expected: &[(usize, f32)], sum: f64, fmt: &str) {
        for &(idx, want) in expected {
            assert!(
                (vals[idx] - want).abs() < 1e-6,
                "{fmt}[{idx}]: glint={} ggml={want}",
                vals[idx]
            );
        }
        let total: f64 = vals.iter().map(|&v| v as f64).sum();
        assert!(
            (total - sum).abs() < 1e-6,
            "{fmt} sum: glint={total} ggml={sum}"
        );
    }

    // block_q4_K: d(f16) dmin(f16) scales[12] qs[128]
    #[test]
    fn q4_k_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&f16_bytes(DMIN));
            data.extend_from_slice(&patterned(12, k * 12, 83, 29));
            data.extend_from_slice(&patterned(128, k * 128, 37, 11));
        }
        let vals = dequantize(&data, GgmlType::Q4K, N);
        check(&vals, Q4K_EXPECTED, Q4K_SUM, "q4_k");
    }

    const Q4K_EXPECTED: &[(usize, f32)] = &[
        (0, 149.25),
        (17, -10.25),
        (34, 105.0),
        (51, 273.0),
        (68, 18.75),
        (85, 2.25),
        (102, 79.5),
        (119, 156.5),
        (136, 0.75),
        (153, 13.25),
        (170, 124.0),
        (187, 28.0),
        (204, 205.25),
        (221, 352.75),
        (238, 98.5),
        (255, 35.5),
        (256, 2.25),
        (273, -3.25),
        (290, 122.0),
        (307, 32.0),
        (324, 279.75),
        (341, 65.25),
        (358, -1.5),
        (375, 201.5),
        (392, 7.25),
        (409, 29.75),
        (426, 30.5),
        (443, 142.5),
        (460, 151.75),
        (477, 269.25),
        (494, 168.75),
        (511, 343.75),
    ];
    const Q4K_SUM: f64 = 50432.0;

    // block_q5_K: d(f16) dmin(f16) scales[12] qh[32] qs[128]
    #[test]
    fn q5_k_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&f16_bytes(DMIN));
            data.extend_from_slice(&patterned(12, k * 12, 83, 29));
            data.extend_from_slice(&patterned(32, k * 32, 59, 17));
            data.extend_from_slice(&patterned(128, k * 128, 37, 11));
        }
        let vals = dequantize(&data, GgmlType::Q5K, N);
        check(&vals, Q5K_EXPECTED, Q5K_SUM, "q5_k");
    }

    const Q5K_EXPECTED: &[(usize, f32)] = &[
        (0, 381.25),
        (17, -10.25),
        (34, 489.0),
        (51, 657.0),
        (68, 42.75),
        (85, 2.25),
        (102, 79.5),
        (119, 332.5),
        (136, 0.75),
        (153, 53.25),
        (170, 124.0),
        (187, 28.0),
        (204, 677.25),
        (221, 824.75),
        (238, 98.5),
        (255, 35.5),
        (256, 10.25),
        (273, -3.25),
        (290, 282.0),
        (307, 192.0),
        (324, 591.75),
        (341, 65.25),
        (358, -1.5),
        (375, 665.5),
        (392, 7.25),
        (409, 101.75),
        (426, 254.5),
        (443, 366.5),
        (460, 151.75),
        (477, 269.25),
        (494, 568.75),
        (511, 743.75),
    ];
    const Q5K_SUM: f64 = 109472.0;

    // block_q6_K: ql[128] qh[64] scales[16:int8] d(f16)
    #[test]
    fn q6_k_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&patterned(128, k * 128, 37, 11));
            data.extend_from_slice(&patterned(64, k * 64, 59, 17));
            data.extend_from_slice(&patterned(16, k * 16, 83, 29));
            data.extend_from_slice(&f16_bytes(D));
        }
        let vals = dequantize(&data, GgmlType::Q6K, N);
        check(&vals, Q6K_EXPECTED, Q6K_SUM, "q6_k");
    }

    const Q6K_EXPECTED: &[(usize, f32)] = &[
        (0, -72.5),
        (17, -1792.0),
        (34, 335.5),
        (51, -242.0),
        (68, 1312.5),
        (85, -34.0),
        (102, -60.0),
        (119, -49.0),
        (136, 487.5),
        (153, -96.0),
        (170, 1319.5),
        (187, -82.0),
        (204, 8.0),
        (221, 294.0),
        (238, -667.5),
        (255, -18.0),
        (256, -192.5),
        (273, 1536.0),
        (290, 71.5),
        (307, -770.0),
        (324, -875.5),
        (341, -90.0),
        (358, -1008.0),
        (375, 1375.0),
        (392, 175.5),
        (409, -672.0),
        (426, -1696.5),
        (443, -34.0),
        (460, 588.0),
        (477, -930.0),
        (494, 184.5),
        (511, -42.0),
    ];
    const Q6K_SUM: f64 = -788.0;

    // block_q2_K: scales[16] qs[64] d(f16) dmin(f16)
    #[test]
    fn q2_k_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&patterned(16, k * 16, 83, 29));
            data.extend_from_slice(&patterned(64, k * 64, 37, 11));
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&f16_bytes(DMIN));
        }
        let vals = dequantize(&data, GgmlType::Q2K, N);
        check(&vals, Q2K_EXPECTED, Q2K_SUM, "q2_k");
    }

    const Q2K_EXPECTED: &[(usize, f32)] = &[
        (0, 19.25),
        (17, -1.75),
        (34, -1.5),
        (51, 5.75),
        (68, 3.0),
        (85, 3.25),
        (102, 22.5),
        (119, -0.5),
        (136, 4.75),
        (153, 0.0),
        (170, 15.25),
        (187, -2.5),
        (204, 1.0),
        (221, 0.75),
        (238, 4.5),
        (255, -3.75),
        (256, 18.5),
        (273, -2.5),
        (290, -2.25),
        (307, 5.0),
        (324, 2.25),
        (341, 2.5),
        (358, -0.75),
        (375, -0.25),
        (392, 4.0),
        (409, -0.75),
        (426, 14.5),
        (443, -3.25),
        (460, 0.25),
        (477, 0.0),
        (494, 7.25),
        (511, 4.5),
    ];
    const Q2K_SUM: f64 = 1886.0;

    // block_q3_K: hmask[32] qs[64] scales[12] d(f16)
    #[test]
    fn q3_k_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&patterned(32, k * 32, 59, 17));
            data.extend_from_slice(&patterned(64, k * 64, 37, 11));
            data.extend_from_slice(&patterned(12, k * 12, 83, 29));
            data.extend_from_slice(&f16_bytes(D));
        }
        let vals = dequantize(&data, GgmlType::Q3K, N);
        check(&vals, Q3K_EXPECTED, Q3K_SUM, "q3_k");
    }

    const Q3K_EXPECTED: &[(usize, f32)] = &[
        (0, -4.5),
        (17, 64.0),
        (34, 9.5),
        (51, 6.0),
        (68, -3.5),
        (85, -18.0),
        (102, -7.5),
        (119, 9.0),
        (136, -8.5),
        (153, -0.0),
        (170, 2.0),
        (187, -2.0),
        (204, 6.0),
        (221, -10.5),
        (238, 16.0),
        (255, -12.0),
        (256, -22.5),
        (273, 56.0),
        (290, 11.5),
        (307, 10.0),
        (324, 6.5),
        (341, -24.0),
        (358, -38.0),
        (375, -26.0),
        (392, 8.0),
        (409, 0.0),
        (426, 39.0),
        (443, -0.0),
        (460, -4.0),
        (477, -39.0),
        (494, -25.5),
        (511, 2.0),
    ];
    const Q3K_SUM: f64 = -271.5;

    // block_q5_0: d(f16) qh[4] qs[16] — 32-element blocks, 5th bit in qh
    #[test]
    fn q5_0_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&patterned(4, k * 4, 59, 17));
            data.extend_from_slice(&patterned(16, k * 16, 37, 11));
        }
        let vals = dequantize(&data, GgmlType::Q5_0, 64);
        check(&vals, Q5_0_EXPECTED, Q5_0_SUM, "q5_0");
    }

    const Q5_0_EXPECTED: &[(usize, f32)] = &[
        (0, 5.5),
        (1, -8.0),
        (2, -5.5),
        (3, -3.0),
        (4, 7.5),
        (5, -6.0),
        (6, -3.5),
        (7, -1.0),
        (8, -6.5),
        (9, -4.0),
        (10, 6.5),
        (11, 1.0),
        (12, -4.5),
        (13, -2.0),
        (14, 0.5),
        (15, -5.0),
        (16, 0.0),
        (17, 1.5),
        (18, 2.5),
        (19, -4.5),
        (20, -3.5),
        (21, -2.0),
        (22, -1.0),
        (23, 0.0),
        (24, -6.5),
        (25, 2.5),
        (26, -4.5),
        (27, -3.0),
        (28, -2.0),
        (29, -1.0),
        (30, 0.5),
        (31, 1.5),
        (32, 5.5),
        (33, -8.0),
        (34, 2.5),
        (35, 5.0),
        (36, 7.5),
        (37, 2.0),
        (38, 4.5),
        (39, 7.0),
        (40, -6.5),
        (41, -4.0),
        (42, -1.5),
        (43, 1.0),
        (44, 3.5),
        (45, 6.0),
        (46, -7.5),
        (47, -5.0),
        (48, 2.5),
        (49, 4.0),
        (50, -3.0),
        (51, -2.0),
        (52, 7.0),
        (53, 0.5),
        (54, 1.5),
        (55, -5.5),
        (56, -4.0),
        (57, 5.0),
        (58, 6.0),
        (59, 7.5),
        (60, -7.5),
        (61, 1.5),
        (62, -5.0),
        (63, 4.0),
    ];
    const Q5_0_SUM: f64 = -23.0;

    // block_q5_1: d(f16) m(f16) qh[4] qs[16] — affine variant of Q5_0
    #[test]
    fn q5_1_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&f16_bytes(DMIN));
            data.extend_from_slice(&patterned(4, k * 4, 59, 17));
            data.extend_from_slice(&patterned(16, k * 16, 37, 11));
        }
        let vals = dequantize(&data, GgmlType::Q5_1, 64);
        check(&vals, Q5_1_EXPECTED, Q5_1_SUM, "q5_1");
    }

    const Q5_1_EXPECTED: &[(usize, f32)] = &[
        (0, 13.75),
        (1, 0.25),
        (2, 2.75),
        (3, 5.25),
        (4, 15.75),
        (5, 2.25),
        (6, 4.75),
        (7, 7.25),
        (8, 1.75),
        (9, 4.25),
        (10, 14.75),
        (11, 9.25),
        (12, 3.75),
        (13, 6.25),
        (14, 8.75),
        (15, 3.25),
        (16, 8.25),
        (17, 9.75),
        (18, 10.75),
        (19, 3.75),
        (20, 4.75),
        (21, 6.25),
        (22, 7.25),
        (23, 8.25),
        (24, 1.75),
        (25, 10.75),
        (26, 3.75),
        (27, 5.25),
        (28, 6.25),
        (29, 7.25),
        (30, 8.75),
        (31, 9.75),
        (32, 13.75),
        (33, 0.25),
        (34, 10.75),
        (35, 13.25),
        (36, 15.75),
        (37, 10.25),
        (38, 12.75),
        (39, 15.25),
        (40, 1.75),
        (41, 4.25),
        (42, 6.75),
        (43, 9.25),
        (44, 11.75),
        (45, 14.25),
        (46, 0.75),
        (47, 3.25),
        (48, 10.75),
        (49, 12.25),
        (50, 5.25),
        (51, 6.25),
        (52, 15.25),
        (53, 8.75),
        (54, 9.75),
        (55, 2.75),
        (56, 4.25),
        (57, 13.25),
        (58, 14.25),
        (59, 15.75),
        (60, 0.75),
        (61, 9.75),
        (62, 3.25),
        (63, 12.25),
    ];
    const Q5_1_SUM: f64 = 505.0;

    // block_iq4_nl: d(f16) qs[16] — 32-element blocks, codebook lookup
    #[test]
    fn iq4_nl_matches_ggml_reference() {
        let mut data = Vec::new();
        for k in 0..2 {
            data.extend_from_slice(&f16_bytes(D));
            data.extend_from_slice(&patterned(16, k * 16, 37, 11));
        }
        let vals = dequantize(&data, GgmlType::IQ4NL, 64);
        check(&vals, IQ4NL_EXPECTED, IQ4NL_SUM, "iq4_nl");
    }

    const IQ4NL_EXPECTED: &[(usize, f32)] = &[
        (0, 19.0),
        (1, -63.5),
        (2, -17.5),
        (3, 12.5),
        (4, 56.5),
        (5, -24.5),
        (6, 6.5),
        (7, 44.5),
        (8, -32.5),
        (9, 0.5),
        (10, 34.5),
        (11, -41.5),
        (12, -5.0),
        (13, 26.5),
        (14, -52.0),
        (15, -11.0),
        (16, -63.5),
        (17, -32.5),
        (18, -17.5),
        (19, -5.0),
        (20, 6.5),
        (21, 26.5),
        (22, 44.5),
        (23, -63.5),
        (24, -32.5),
        (25, -17.5),
        (26, -5.0),
        (27, 12.5),
        (28, 26.5),
        (29, 44.5),
        (30, -52.0),
        (31, -32.5),
        (32, 19.0),
        (33, -63.5),
        (34, -17.5),
        (35, 12.5),
        (36, 56.5),
        (37, -24.5),
        (38, 6.5),
        (39, 44.5),
        (40, -32.5),
        (41, 0.5),
        (42, 34.5),
        (43, -41.5),
        (44, -5.0),
        (45, 26.5),
        (46, -52.0),
        (47, -11.0),
        (48, -17.5),
        (49, 0.5),
        (50, 12.5),
        (51, 26.5),
        (52, 44.5),
        (53, -52.0),
        (54, -32.5),
        (55, -17.5),
        (56, 0.5),
        (57, 12.5),
        (58, 26.5),
        (59, 56.5),
        (60, -52.0),
        (61, -32.5),
        (62, -11.0),
        (63, 0.5),
    ];
    const IQ4NL_SUM: f64 = -289.0;
}
