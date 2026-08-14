//! ARM NEON SIMD kernels for quantized matrix-vector multiplication.
//!
//! Vectorized execution on `aarch64` architectures (Apple Silicon M-series,
//! AWS Graviton, ARM64 Linux servers).
//!
//! On `aarch64`, NEON is part of the baseline ISA, so runtime feature detection
//! is not required. Row-level parallelism is driven via Rayon across CPU cores.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::f16;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::dequantize::get_scale_min_q4k;

const MAX_BATCH_LANES: usize = 4;

// ── Precondition check ────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn check_dims(
    data_len: usize,
    rows: usize,
    cols: usize,
    vec_len: usize,
    block_elems: usize,
    block_bytes: usize,
) {
    debug_assert!(
        cols.is_multiple_of(block_elems),
        "cols ({cols}) must be a multiple of block size ({block_elems})"
    );
    debug_assert!(vec_len >= cols, "vec.len() ({vec_len}) < cols ({cols})");
    let bytes_per_row = (cols / block_elems) * block_bytes;
    debug_assert!(
        data_len >= rows * bytes_per_row,
        "data.len() ({data_len}) < rows*bytes_per_row ({})",
        rows * bytes_per_row
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn check_dims_batch(
    data_len: usize,
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out_len: usize,
    block_elems: usize,
    block_bytes: usize,
) {
    let batch = inputs.len();
    debug_assert!(batch > 0, "batch must be non-empty");
    debug_assert_eq!(
        out_len,
        rows * batch,
        "out.len() ({out_len}) != rows {rows} * batch {batch}"
    );
    debug_assert!(
        cols.is_multiple_of(block_elems),
        "cols ({cols}) must be a multiple of block size ({block_elems})"
    );
    for (s, inp) in inputs.iter().enumerate() {
        debug_assert!(
            inp.len() >= cols,
            "inputs[{s}].len() ({}) < cols ({cols})",
            inp.len()
        );
    }
    let bytes_per_row = (cols / block_elems) * block_bytes;
    debug_assert!(
        data_len >= rows * bytes_per_row,
        "data.len() ({data_len}) < rows*bytes_per_row ({})",
        rows * bytes_per_row
    );
}

// ── Unpacking Helpers ────────────────────────────────────────────────────────

/// Convert 16 signed i8 values into 4×`float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_i8x16(v: int8x16_t) -> [float32x4_t; 4] {
    let lo_i8 = vget_low_s8(v);
    let hi_i8 = vget_high_s8(v);
    let lo_i16 = vmovl_s8(lo_i8);
    let hi_i16 = vmovl_s8(hi_i8);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16))),
    ]
}

/// Convert 16 unsigned u8 values into 4×`float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_u8x16(v: uint8x16_t) -> [float32x4_t; 4] {
    let lo_u8 = vget_low_u8(v);
    let hi_u8 = vget_high_u8(v);
    let lo_u16 = vmovl_u8(lo_u8);
    let hi_u16 = vmovl_u8(hi_u8);
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_u16))),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_u16))),
    ]
}

/// Convert 32 signed i8 values into 8×`float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_i8x32(ptr: *const i8) -> [float32x4_t; 8] {
    let chunk0 = vld1q_s8(ptr);
    let chunk1 = vld1q_s8(ptr.add(16));
    let [w0, w1, w2, w3] = unpack_i8x16(chunk0);
    let [w4, w5, w6, w7] = unpack_i8x16(chunk1);
    [w0, w1, w2, w3, w4, w5, w6, w7]
}

/// FMA-accumulate an already-decoded 32-element weight block against one input vector:
/// `acc += scale × dot32(w, input)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn accum_block_w(
    w: &[float32x4_t; 8],
    input_ptr: *const f32,
    scale: f32,
    acc: float32x4_t,
) -> float32x4_t {
    let v0 = vld1q_f32(input_ptr);
    let v1 = vld1q_f32(input_ptr.add(4));
    let v2 = vld1q_f32(input_ptr.add(8));
    let v3 = vld1q_f32(input_ptr.add(12));
    let v4 = vld1q_f32(input_ptr.add(16));
    let v5 = vld1q_f32(input_ptr.add(20));
    let v6 = vld1q_f32(input_ptr.add(24));
    let v7 = vld1q_f32(input_ptr.add(28));

    let mut p0 = vmulq_f32(w[0], v0);
    p0 = vfmaq_f32(p0, w[1], v1);
    p0 = vfmaq_f32(p0, w[2], v2);
    p0 = vfmaq_f32(p0, w[3], v3);

    let mut p1 = vmulq_f32(w[4], v4);
    p1 = vfmaq_f32(p1, w[5], v5);
    p1 = vfmaq_f32(p1, w[6], v6);
    p1 = vfmaq_f32(p1, w[7], v7);

    let block_sum = vaddq_f32(p0, p1);
    let scale_v = vdupq_n_f32(scale);
    vfmaq_f32(acc, scale_v, block_sum)
}

// ── Q8_0 Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q8_0(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let w = unpack_i8x32(block[2..].as_ptr() as *const i8);
        acc = accum_block_w(&w, vec.as_ptr().add(b * BLOCK_ELEMS), scale, acc);
    }
    vaddvq_f32(acc)
}

/// Q8_0 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q8_0_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q8_0(data, r * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q4_0 Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q4_0(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let low_mask = vdupq_n_u8(0x0F);
    let offset_8 = vdupq_n_s8(8);

    let mut acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let packed = vld1q_u8(block[2..].as_ptr());
        let lo_nib = vandq_u8(packed, low_mask);
        let hi_nib = vshrq_n_u8::<4>(packed);

        let lo_i8 = vsubq_s8(vreinterpretq_s8_u8(lo_nib), offset_8);
        let hi_i8 = vsubq_s8(vreinterpretq_s8_u8(hi_nib), offset_8);

        let [w0, w1, w2, w3] = unpack_i8x16(lo_i8);
        let [w4, w5, w6, w7] = unpack_i8x16(hi_i8);
        let w = [w0, w1, w2, w3, w4, w5, w6, w7];

        acc = accum_block_w(&w, vec.as_ptr().add(b * BLOCK_ELEMS), scale, acc);
    }
    vaddvq_f32(acc)
}

/// Q4_0 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q4_0_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q4_0(data, r * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q4_1 Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q4_1(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let low_mask = vdupq_n_u8(0x0F);

    let mut acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let packed = vld1q_u8(block[4..].as_ptr());
        let lo_nib = vandq_u8(packed, low_mask);
        let hi_nib = vshrq_n_u8::<4>(packed);

        let [q0, q1, q2, q3] = unpack_u8x16(lo_nib);
        let [q4, q5, q6, q7] = unpack_u8x16(hi_nib);
        let qs = [q0, q1, q2, q3, q4, q5, q6, q7];

        let dv = vdupq_n_f32(d);
        let mv = vdupq_n_f32(m);
        let input_ptr = vec.as_ptr().add(b * BLOCK_ELEMS);

        for (k, &q_val) in qs.iter().enumerate() {
            let w = vfmaq_f32(mv, dv, q_val);
            let x = vld1q_f32(input_ptr.add(k * 4));
            acc = vfmaq_f32(acc, w, x);
        }
    }
    vaddvq_f32(acc)
}

/// Q4_1 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q4_1_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 20;
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q4_1(data, r * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q5_0 Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q5_0(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let low_mask = vdupq_n_u8(0x0F);
    let offset_16 = vdupq_n_s8(16);

    let mut acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

        let packed = vld1q_u8(block[6..].as_ptr());
        let lo_nib = vandq_u8(packed, low_mask);
        let hi_nib = vshrq_n_u8::<4>(packed);

        let mut lo_bytes = [0u8; 16];
        let mut hi_bytes = [0u8; 16];
        vst1q_u8(lo_bytes.as_mut_ptr(), lo_nib);
        vst1q_u8(hi_bytes.as_mut_ptr(), hi_nib);

        for j in 0..16 {
            let h0 = if (qh & (1 << j)) != 0 { 16 } else { 0 };
            let h1 = if (qh & (1 << (j + 16))) != 0 { 16 } else { 0 };
            lo_bytes[j] |= h0;
            hi_bytes[j] |= h1;
        }

        let lo_i8 = vsubq_s8(vreinterpretq_s8_u8(vld1q_u8(lo_bytes.as_ptr())), offset_16);
        let hi_i8 = vsubq_s8(vreinterpretq_s8_u8(vld1q_u8(hi_bytes.as_ptr())), offset_16);

        let [w0, w1, w2, w3] = unpack_i8x16(lo_i8);
        let [w4, w5, w6, w7] = unpack_i8x16(hi_i8);
        let w = [w0, w1, w2, w3, w4, w5, w6, w7];

        acc = accum_block_w(&w, vec.as_ptr().add(b * BLOCK_ELEMS), d, acc);
    }
    vaddvq_f32(acc)
}

/// Q5_0 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q5_0_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 22;
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q5_0(data, r * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q5_1 Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q5_1(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let low_mask = vdupq_n_u8(0x0F);

    let mut acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);

        let packed = vld1q_u8(block[8..].as_ptr());
        let lo_nib = vandq_u8(packed, low_mask);
        let hi_nib = vshrq_n_u8::<4>(packed);

        let mut lo_bytes = [0u8; 16];
        let mut hi_bytes = [0u8; 16];
        vst1q_u8(lo_bytes.as_mut_ptr(), lo_nib);
        vst1q_u8(hi_bytes.as_mut_ptr(), hi_nib);

        for j in 0..16 {
            let h0 = if (qh & (1 << j)) != 0 { 16 } else { 0 };
            let h1 = if (qh & (1 << (j + 16))) != 0 { 16 } else { 0 };
            lo_bytes[j] |= h0;
            hi_bytes[j] |= h1;
        }

        let [q0, q1, q2, q3] = unpack_u8x16(vld1q_u8(lo_bytes.as_ptr()));
        let [q4, q5, q6, q7] = unpack_u8x16(vld1q_u8(hi_bytes.as_ptr()));
        let qs = [q0, q1, q2, q3, q4, q5, q6, q7];

        let dv = vdupq_n_f32(d);
        let mv = vdupq_n_f32(m);
        let input_ptr = vec.as_ptr().add(b * BLOCK_ELEMS);

        for (k, &q_val) in qs.iter().enumerate() {
            let w = vfmaq_f32(mv, dv, q_val);
            let x = vld1q_f32(input_ptr.add(k * 4));
            acc = vfmaq_f32(acc, w, x);
        }
    }
    vaddvq_f32(acc)
}

/// Q5_1 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q5_1_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 24;
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q5_1(data, r * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q4_K Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q4_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let low_mask = vdupq_n_u8(0x0F);
    let mut acc = vdupq_n_f32(0.0);

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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

            let q_ptr = qs.as_ptr().add(group * 32);
            let packed0 = vld1q_u8(q_ptr);
            let packed1 = vld1q_u8(q_ptr.add(16));

            // Low nibbles -> subblock 0 (elements 0..32 of group)
            let lo0 = vandq_u8(packed0, low_mask);
            let lo1 = vandq_u8(packed1, low_mask);
            let [q0, q1, q2, q3] = unpack_u8x16(lo0);
            let [q4, q5, q6, q7] = unpack_u8x16(lo1);
            let qs0 = [q0, q1, q2, q3, q4, q5, q6, q7];

            let d0_v = vdupq_n_f32(d0);
            let neg_m0_v = vdupq_n_f32(-m0);
            let in_ptr0 = vec.as_ptr().add(x_base + group * 64);
            for (k, &qv) in qs0.iter().enumerate() {
                let w = vfmaq_f32(neg_m0_v, d0_v, qv);
                let x = vld1q_f32(in_ptr0.add(k * 4));
                acc = vfmaq_f32(acc, w, x);
            }

            // High nibbles -> subblock 1 (elements 32..64 of group)
            let hi0 = vshrq_n_u8::<4>(packed0);
            let hi1 = vshrq_n_u8::<4>(packed1);
            let [q8, q9, q10, q11] = unpack_u8x16(hi0);
            let [q12, q13, q14, q15] = unpack_u8x16(hi1);
            let qs1 = [q8, q9, q10, q11, q12, q13, q14, q15];

            let d1_v = vdupq_n_f32(d1);
            let neg_m1_v = vdupq_n_f32(-m1);
            let in_ptr1 = vec.as_ptr().add(x_base + group * 64 + 32);
            for (k, &qv) in qs1.iter().enumerate() {
                let w = vfmaq_f32(neg_m1_v, d1_v, qv);
                let x = vld1q_f32(in_ptr1.add(k * 4));
                acc = vfmaq_f32(acc, w, x);
            }
        }
    }
    vaddvq_f32(acc)
}

/// Q4_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q4_k_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    check_dims(data.len(), rows, cols, vec.len(), SUPER_BLOCK, BLOCK_BYTES);

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q4_k(data, r * bytes_per_row, n_super, vec) })
        .collect()
}

// ── Q5_K Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q5_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let mut acc = vdupq_n_f32(0.0);
    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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

            let in_ptr0 = vec.as_ptr().add(x_base + group * 64);
            let in_ptr1 = vec.as_ptr().add(x_base + group * 64 + 32);

            for l in (0..32).step_by(4) {
                let lo_arr = [
                    ((qs[q_base + l] & 0x0F) | (if qh[l] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 1] & 0x0F) | (if qh[l + 1] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 2] & 0x0F) | (if qh[l + 2] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 3] & 0x0F) | (if qh[l + 3] & u1 != 0 { 16 } else { 0 })) as f32,
                ];
                let hi_arr = [
                    ((qs[q_base + l] >> 4) | (if qh[l] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 1] >> 4) | (if qh[l + 1] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 2] >> 4) | (if qh[l + 2] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 3] >> 4) | (if qh[l + 3] & u2 != 0 { 16 } else { 0 })) as f32,
                ];

                let q0 = vld1q_f32(lo_arr.as_ptr());
                let w0 = vfmaq_f32(vdupq_n_f32(-m0), vdupq_n_f32(d0), q0);
                let x0 = vld1q_f32(in_ptr0.add(l));
                acc = vfmaq_f32(acc, w0, x0);

                let q1 = vld1q_f32(hi_arr.as_ptr());
                let w1 = vfmaq_f32(vdupq_n_f32(-m1), vdupq_n_f32(d1), q1);
                let x1 = vld1q_f32(in_ptr1.add(l));
                acc = vfmaq_f32(acc, w1, x1);
            }
        }
    }
    vaddvq_f32(acc)
}

/// Q5_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q5_k_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    check_dims(data.len(), rows, cols, vec.len(), SUPER_BLOCK, BLOCK_BYTES);

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q5_k(data, r * bytes_per_row, n_super, vec) })
        .collect()
}

// ── Q6_K Kernel ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q6_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let mut acc = 0.0f32;
    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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
}

/// Q6_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q6_k_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    check_dims(data.len(), rows, cols, vec.len(), SUPER_BLOCK, BLOCK_BYTES);

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q6_k(data, r * bytes_per_row, n_super, vec) })
        .collect()
}

// ── Batched Kernels ──────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q8_0_batch(
    data: &[u8],
    row_start: usize,
    n_blocks: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    debug_assert!(inputs.len() <= MAX_BATCH_LANES);

    let mut acc = [vdupq_n_f32(0.0); MAX_BATCH_LANES];
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let w = unpack_i8x32(block[2..].as_ptr() as *const i8);

        for (s, input) in inputs.iter().enumerate() {
            acc[s] = accum_block_w(&w, input.as_ptr().add(b * BLOCK_ELEMS), scale, acc[s]);
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = vaddvq_f32(acc[s]);
    }
}

/// Batched Q8_0 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q8_0_batch_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    check_dims_batch(
        data.len(),
        rows,
        cols,
        inputs,
        out.len(),
        BLOCK_ELEMS,
        BLOCK_BYTES,
    );

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;
    let batch = inputs.len();

    out.par_chunks_mut(batch).enumerate().for_each(|(r, dst)| {
        for start in (0..batch).step_by(MAX_BATCH_LANES) {
            let end = (start + MAX_BATCH_LANES).min(batch);
            unsafe {
                dot_row_q8_0_batch(
                    data,
                    r * bytes_per_row,
                    n_blocks,
                    &inputs[start..end],
                    &mut dst[start..end],
                )
            };
        }
    });
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q4_0_batch(
    data: &[u8],
    row_start: usize,
    n_blocks: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    debug_assert!(inputs.len() <= MAX_BATCH_LANES);

    let low_mask = vdupq_n_u8(0x0F);
    let offset_8 = vdupq_n_s8(8);

    let mut acc = [vdupq_n_f32(0.0); MAX_BATCH_LANES];
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let packed = vld1q_u8(block[2..].as_ptr());
        let lo_nib = vandq_u8(packed, low_mask);
        let hi_nib = vshrq_n_u8::<4>(packed);

        let lo_i8 = vsubq_s8(vreinterpretq_s8_u8(lo_nib), offset_8);
        let hi_i8 = vsubq_s8(vreinterpretq_s8_u8(hi_nib), offset_8);

        let [w0, w1, w2, w3] = unpack_i8x16(lo_i8);
        let [w4, w5, w6, w7] = unpack_i8x16(hi_i8);
        let w = [w0, w1, w2, w3, w4, w5, w6, w7];

        for (s, input) in inputs.iter().enumerate() {
            acc[s] = accum_block_w(&w, input.as_ptr().add(b * BLOCK_ELEMS), scale, acc[s]);
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = vaddvq_f32(acc[s]);
    }
}

/// Batched Q4_0 matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q4_0_batch_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    check_dims_batch(
        data.len(),
        rows,
        cols,
        inputs,
        out.len(),
        BLOCK_ELEMS,
        BLOCK_BYTES,
    );

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;
    let batch = inputs.len();

    out.par_chunks_mut(batch).enumerate().for_each(|(r, dst)| {
        for start in (0..batch).step_by(MAX_BATCH_LANES) {
            let end = (start + MAX_BATCH_LANES).min(batch);
            unsafe {
                dot_row_q4_0_batch(
                    data,
                    r * bytes_per_row,
                    n_blocks,
                    &inputs[start..end],
                    &mut dst[start..end],
                )
            };
        }
    });
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q4_k_batch(
    data: &[u8],
    row_start: usize,
    n_super: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    debug_assert!(inputs.len() <= MAX_BATCH_LANES);

    let low_mask = vdupq_n_u8(0x0F);
    let mut acc = [vdupq_n_f32(0.0); MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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

            let q_ptr = qs.as_ptr().add(group * 32);
            let packed0 = vld1q_u8(q_ptr);
            let packed1 = vld1q_u8(q_ptr.add(16));

            let lo0 = vandq_u8(packed0, low_mask);
            let lo1 = vandq_u8(packed1, low_mask);
            let [q0, q1, q2, q3] = unpack_u8x16(lo0);
            let [q4, q5, q6, q7] = unpack_u8x16(lo1);
            let qs0 = [q0, q1, q2, q3, q4, q5, q6, q7];

            let d0_v = vdupq_n_f32(d0);
            let neg_m0_v = vdupq_n_f32(-m0);

            for (s, input) in inputs.iter().enumerate() {
                let in_ptr0 = input.as_ptr().add(x_base + group * 64);
                for (k, &qv) in qs0.iter().enumerate() {
                    let w = vfmaq_f32(neg_m0_v, d0_v, qv);
                    let x = vld1q_f32(in_ptr0.add(k * 4));
                    acc[s] = vfmaq_f32(acc[s], w, x);
                }
            }

            let hi0 = vshrq_n_u8::<4>(packed0);
            let hi1 = vshrq_n_u8::<4>(packed1);
            let [q8, q9, q10, q11] = unpack_u8x16(hi0);
            let [q12, q13, q14, q15] = unpack_u8x16(hi1);
            let qs1 = [q8, q9, q10, q11, q12, q13, q14, q15];

            let d1_v = vdupq_n_f32(d1);
            let neg_m1_v = vdupq_n_f32(-m1);

            for (s, input) in inputs.iter().enumerate() {
                let in_ptr1 = input.as_ptr().add(x_base + group * 64 + 32);
                for (k, &qv) in qs1.iter().enumerate() {
                    let w = vfmaq_f32(neg_m1_v, d1_v, qv);
                    let x = vld1q_f32(in_ptr1.add(k * 4));
                    acc[s] = vfmaq_f32(acc[s], w, x);
                }
            }
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = vaddvq_f32(acc[s]);
    }
}

/// Batched Q4_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q4_k_batch_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    check_dims_batch(
        data.len(),
        rows,
        cols,
        inputs,
        out.len(),
        SUPER_BLOCK,
        BLOCK_BYTES,
    );

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;
    let batch = inputs.len();

    out.par_chunks_mut(batch).enumerate().for_each(|(r, dst)| {
        for start in (0..batch).step_by(MAX_BATCH_LANES) {
            let end = (start + MAX_BATCH_LANES).min(batch);
            unsafe {
                dot_row_q4_k_batch(
                    data,
                    r * bytes_per_row,
                    n_super,
                    &inputs[start..end],
                    &mut dst[start..end],
                )
            };
        }
    });
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q5_k_batch(
    data: &[u8],
    row_start: usize,
    n_super: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    debug_assert!(inputs.len() <= MAX_BATCH_LANES);

    let mut acc = [vdupq_n_f32(0.0); MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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

            let d0_v = vdupq_n_f32(d0);
            let neg_m0_v = vdupq_n_f32(-m0);
            let d1_v = vdupq_n_f32(d1);
            let neg_m1_v = vdupq_n_f32(-m1);

            for l in (0..32).step_by(4) {
                let lo_arr = [
                    ((qs[q_base + l] & 0x0F) | (if qh[l] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 1] & 0x0F) | (if qh[l + 1] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 2] & 0x0F) | (if qh[l + 2] & u1 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 3] & 0x0F) | (if qh[l + 3] & u1 != 0 { 16 } else { 0 })) as f32,
                ];
                let hi_arr = [
                    ((qs[q_base + l] >> 4) | (if qh[l] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 1] >> 4) | (if qh[l + 1] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 2] >> 4) | (if qh[l + 2] & u2 != 0 { 16 } else { 0 })) as f32,
                    ((qs[q_base + l + 3] >> 4) | (if qh[l + 3] & u2 != 0 { 16 } else { 0 })) as f32,
                ];

                let q0 = vld1q_f32(lo_arr.as_ptr());
                let w0 = vfmaq_f32(neg_m0_v, d0_v, q0);
                let q1 = vld1q_f32(hi_arr.as_ptr());
                let w1 = vfmaq_f32(neg_m1_v, d1_v, q1);

                for (s, input) in inputs.iter().enumerate() {
                    let in_ptr0 = input.as_ptr().add(x_base + group * 64 + l);
                    let in_ptr1 = input.as_ptr().add(x_base + group * 64 + 32 + l);
                    let x0 = vld1q_f32(in_ptr0);
                    let x1 = vld1q_f32(in_ptr1);
                    acc[s] = vfmaq_f32(acc[s], w0, x0);
                    acc[s] = vfmaq_f32(acc[s], w1, x1);
                }
            }
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = vaddvq_f32(acc[s]);
    }
}

/// Batched Q5_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q5_k_batch_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    check_dims_batch(
        data.len(),
        rows,
        cols,
        inputs,
        out.len(),
        SUPER_BLOCK,
        BLOCK_BYTES,
    );

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;
    let batch = inputs.len();

    out.par_chunks_mut(batch).enumerate().for_each(|(r, dst)| {
        for start in (0..batch).step_by(MAX_BATCH_LANES) {
            let end = (start + MAX_BATCH_LANES).min(batch);
            unsafe {
                dot_row_q5_k_batch(
                    data,
                    r * bytes_per_row,
                    n_super,
                    &inputs[start..end],
                    &mut dst[start..end],
                )
            };
        }
    });
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_row_q6_k_batch(
    data: &[u8],
    row_start: usize,
    n_super: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const BLOCK_BYTES: usize = 210;
    const SUPER_BLOCK: usize = 256;
    debug_assert!(inputs.len() <= MAX_BATCH_LANES);

    let mut total = [0.0f32; MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
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
                let sc0 = d * sc_raw[sc_off + is] as i8 as f32;
                let sc2 = d * sc_raw[sc_off + is + 2] as i8 as f32;
                let sc4 = d * sc_raw[sc_off + is + 4] as i8 as f32;
                let sc6 = d * sc_raw[sc_off + is + 6] as i8 as f32;

                let term0 = sc0 * q1;
                let term1 = sc2 * q2;
                let term2 = sc4 * q3;
                let term3 = sc6 * q4;

                for (s, input) in inputs.iter().enumerate() {
                    let vec = *input;
                    total[s] += term0 * vec[x_group + l];
                    total[s] += term1 * vec[x_group + l + 32];
                    total[s] += term2 * vec[x_group + l + 64];
                    total[s] += term3 * vec[x_group + l + 96];
                }
            }
        }
    }
    out.copy_from_slice(&total[..out.len()]);
}

/// Batched Q6_K matrix-vector multiply on ARM NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn matvec_q6_k_batch_neon(
    data: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[&[f32]],
    out: &mut [f32],
) {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    check_dims_batch(
        data.len(),
        rows,
        cols,
        inputs,
        out.len(),
        SUPER_BLOCK,
        BLOCK_BYTES,
    );

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;
    let batch = inputs.len();

    out.par_chunks_mut(batch).enumerate().for_each(|(r, dst)| {
        for start in (0..batch).step_by(MAX_BATCH_LANES) {
            let end = (start + MAX_BATCH_LANES).min(batch);
            unsafe {
                dot_row_q6_k_batch(
                    data,
                    r * bytes_per_row,
                    n_super,
                    &inputs[start..end],
                    &mut dst[start..end],
                )
            };
        }
    });
}
