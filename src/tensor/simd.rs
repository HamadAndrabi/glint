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
