//! AVX2 + FMA SIMD kernels for quantized matrix-vector multiplication.
//!
//! These process 32 int8 weights per block using 256-bit SIMD registers,
//! converting i8→i16→i32→f32 and using fused multiply-add (FMA) to accumulate.
//!
//! Each block of 32 elements requires only 4 FMA instructions (8 floats each),
//! versus 32 scalar multiply-adds in the fallback. The per-block scale factor
//! is broadcast and accumulated into a single __m256, with a single horizontal
//! sum at the end of each output row.
//!
//! Row-level parallelism via rayon: each output row's dot product is independent,
//! so we distribute rows across all CPU cores.
//!
//! Only compiled on x86_64. Caller must verify `is_x86_feature_detected!("avx2")`
//! and `is_x86_feature_detected!("fma")` before calling these functions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;
use rayon::prelude::*;

use super::dequantize::get_scale_min_q4k;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Horizontal sum of 8 f32 lanes in a __m256 register.
///
/// Reduces [a,b,c,d,e,f,g,h] → a+b+c+d+e+f+g+h via pairwise adds:
///   1. Fold upper 128 into lower 128: [a+e, b+f, c+g, d+h]
///   2. Duplicate odd lanes:            [b+f, b+f, d+h, d+h]
///   3. Add:                            [a+b+e+f, -, c+d+g+h, -]
///   4. Move high 64 to low:            [c+d+g+h, -, -, -]
///   5. Final add:                      [a+b+c+d+e+f+g+h, -, -, -]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi128, lo128);
    let shuf = _mm_movehdup_ps(sum128);       // [b,b,d,d]
    let sums = _mm_add_ps(sum128, shuf);       // [a+b, -, c+d, -]
    let hi64 = _mm_movehl_ps(sums, sums);      // [c+d, -, -, -]
    let total = _mm_add_ss(sums, hi64);
    _mm_cvtss_f32(total)
}

/// Convert 32 signed i8 values (in a __m256i) into 4×__m256 of f32,
/// and FMA-accumulate them against 4 input f32 vectors into `acc`.
///
/// The i8→f32 conversion pipeline:
///   __m256i (32×i8) → split into 2×__m128i (16×i8 each)
///   → cvtepi8_epi16 → 2×__m256i (16×i16 each)
///   → split each into 2×__m128i (8×i16 each)
///   → cvtepi16_epi32 → 4×__m256i (8×i32 each)
///   → cvtepi32_ps → 4×__m256 (8×f32 each)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_block_q8(
    weights_i8: __m256i,
    input_ptr: *const f32,
    scale: f32,
    acc: __m256,
) -> __m256 {
    // Split 32 i8 values into lower and upper 16
    let lo_i8 = _mm256_castsi256_si128(weights_i8);
    let hi_i8 = _mm256_extracti128_si256::<1>(weights_i8);

    // Sign-extend i8 → i16 (16 values each, 256-bit output)
    let lo_i16 = _mm256_cvtepi8_epi16(lo_i8);
    let hi_i16 = _mm256_cvtepi8_epi16(hi_i8);

    // Convert i16 → i32 → f32, 8 values at a time (4 groups total)
    let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_i16)));
    let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_i16)));
    let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_i16)));
    let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_i16)));

    // Load 32 f32 input values (4×8)
    let v0 = _mm256_loadu_ps(input_ptr);
    let v1 = _mm256_loadu_ps(input_ptr.add(8));
    let v2 = _mm256_loadu_ps(input_ptr.add(16));
    let v3 = _mm256_loadu_ps(input_ptr.add(24));

    // Partial dot products: p = w0*v0 + w1*v1 + w2*v2 + w3*v3
    let p01 = _mm256_fmadd_ps(w1, v1, _mm256_mul_ps(w0, v0));
    let p23 = _mm256_fmadd_ps(w3, v3, _mm256_mul_ps(w2, v2));
    let block_partial = _mm256_add_ps(p01, p23);

    // Accumulate: acc += scale * block_partial
    let scale_vec = _mm256_set1_ps(scale);
    _mm256_fmadd_ps(scale_vec, block_partial, acc)
}

// ── Per-row functions ────────────────────────────────────────────────────────
//
// Closures (including rayon's) don't inherit #[target_feature], so we extract
// the per-row SIMD logic into standalone functions that closures can call.

/// Compute the dot product of one Q8_0 row against the input vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_q8_0(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut acc = _mm256_setzero_ps();
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let weights = _mm256_loadu_si256(block[2..].as_ptr() as *const __m256i);
        acc = accum_block_q8(weights, vec.as_ptr().add(b * BLOCK_ELEMS), scale, acc);
    }
    hsum_avx2(acc)
}

/// Compute the dot product of one Q4_0 row against the input vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_q4_0(data: &[u8], row_start: usize, n_blocks: usize, vec: &[f32]) -> f32 {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let low_mask = _mm_set1_epi8(0x0F);
    let offset_8 = _mm256_set1_epi8(8);

    let mut acc = _mm256_setzero_ps();
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let packed = _mm_loadu_si128(block[2..].as_ptr() as *const __m128i);
        let lo_nib = _mm_and_si128(packed, low_mask);
        let hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let interleaved_lo = _mm_unpacklo_epi8(lo_nib, hi_nib);
        let interleaved_hi = _mm_unpackhi_epi8(lo_nib, hi_nib);
        let values_u8 = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(interleaved_lo),
            interleaved_hi,
        );
        let centered = _mm256_sub_epi8(values_u8, offset_8);

        acc = accum_block_q8(centered, vec.as_ptr().add(b * BLOCK_ELEMS), scale, acc);
    }
    hsum_avx2(acc)
}

// ── Public kernels ───────────────────────────────────────────────────────────

/// Q8_0 matrix-vector multiply using AVX2 + FMA + rayon.
///
/// Rows are distributed across threads via rayon's work-stealing pool.
/// Each thread computes its rows using SIMD, then results are collected.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q8_0_avx2(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|i| unsafe { dot_row_q8_0(data, i * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Q4_K helpers ─────────────────────────────────────────────────────────────

/// Compute dot(nibbles_f32, input) and sum(input) in one pass.
///
/// `nibbles`: 32 u8 nibble values (0..15) packed in a __m256i.
/// `input_ptr`: pointer to 32 contiguous f32 values.
///
/// Returns `(dot_sum, input_sum)` — caller multiplies by the sub-block
/// scale and min to get the final contribution:
///   `contribution = d_scale * dot_sum - d_min * input_sum`
///
/// The two scalars are returned separately so the caller can share the
/// `input_sum` across multiple sub-block passes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_sum_q4k(nibbles: __m256i, input_ptr: *const f32) -> (f32, f32) {
    // Convert 32 nibbles (u8, 0..15) → 4×__m256 of f32.
    // Sign-extend == zero-extend for values 0..15, so cvtepi8_epi16 is correct.
    let lo_i8 = _mm256_castsi256_si128(nibbles);
    let hi_i8 = _mm256_extracti128_si256::<1>(nibbles);
    let lo_i16 = _mm256_cvtepi8_epi16(lo_i8);
    let hi_i16 = _mm256_cvtepi8_epi16(hi_i8);
    let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_i16)));
    let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_i16)));
    let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_i16)));
    let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_i16)));

    // Load 32 f32 input values.
    let v0 = _mm256_loadu_ps(input_ptr);
    let v1 = _mm256_loadu_ps(input_ptr.add(8));
    let v2 = _mm256_loadu_ps(input_ptr.add(16));
    let v3 = _mm256_loadu_ps(input_ptr.add(24));

    // Dot product: sum(w × v)
    let d01 = _mm256_fmadd_ps(w1, v1, _mm256_mul_ps(w0, v0));
    let d23 = _mm256_fmadd_ps(w3, v3, _mm256_mul_ps(w2, v2));
    let dot_sum = hsum_avx2(_mm256_add_ps(d01, d23));

    // Input sum: sum(v)
    let s01 = _mm256_add_ps(v0, v1);
    let s23 = _mm256_add_ps(v2, v3);
    let inp_sum = hsum_avx2(_mm256_add_ps(s01, s23));

    (dot_sum, inp_sum)
}

/// Compute the dot product of one Q4_K row against the input vector.
///
/// Each super-block (144 bytes, 256 elements) is split into 4 groups of 64.
/// Each group uses two (scale, min) pairs for its 32 low-nibble and 32
/// high-nibble values.
///
/// `contribution = d_scale × dot(nibbles, input) − d_min × sum(input)`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_q4_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let lo_mask = _mm256_set1_epi8(0x0F_u8 as i8);
    let mut total = 0.0f32;

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let d    = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];
        let x_base = sb * SUPER_BLOCK;

        for group in 0..4_usize {
            let (sc0, mn0) = get_scale_min_q4k(group * 2,     scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;  let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;  let m1 = dmin * mn1 as f32;

            // Load 32 packed nibble bytes: low nibbles → first 32 elements,
            // high nibbles → next 32 elements of this group.
            let q_base = group * 32;
            let packed = _mm256_loadu_si256(qs[q_base..].as_ptr() as *const __m256i);

            // Extract nibbles. srli_epi16 + mask works correctly byte-wise:
            // for each 16-bit lane [B1,B0], shifted >> 4 gives [B1>>4, B0>>4 | B1<<4],
            // and masking with 0x0F extracts hi_B0 into byte position 0, hi_B1 into byte 1.
            let lo_nibbles = _mm256_and_si256(packed, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);

            let x_lo = vec.as_ptr().add(x_base + group * 64);
            let x_hi = x_lo.add(32);

            let (dot_lo, sum_lo) = dot_and_sum_q4k(lo_nibbles, x_lo);
            let (dot_hi, sum_hi) = dot_and_sum_q4k(hi_nibbles, x_hi);

            total += d0 * dot_lo - m0 * sum_lo;
            total += d1 * dot_hi - m1 * sum_hi;
        }
    }

    total
}

/// Q4_K matrix-vector multiply using AVX2 + FMA + rayon.
///
/// Rows distributed across threads; each row uses `dot_row_q4_k` which
/// processes 4 groups of 64 elements per super-block with SIMD nibble
/// extraction and dot-product accumulation.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q4_k_avx2(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q4_k(data, r * bytes_per_row, n_super, vec) })
        .collect()
}

/// Q4_0 matrix-vector multiply using AVX2 + FMA + rayon.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q4_0_avx2(
    data: &[u8],
    rows: usize,
    cols: usize,
    vec: &[f32],
) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|i| unsafe { dot_row_q4_0(data, i * bytes_per_row, n_blocks, vec) })
        .collect()
}
