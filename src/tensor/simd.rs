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
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::dequantize::get_scale_min_q4k;

// ── Safety contract for the public kernels ───────────────────────────────────
//
// Every `matvec_*_avx2` entry point below reads `data` and `vec` through raw
// unaligned SIMD loads that perform NO bounds checking. For the loads to stay
// in bounds the caller MUST uphold, for a format with `block_elems` elements
// and `block_bytes` bytes per block:
//
//   * `cols % block_elems == 0`         (no partial trailing block)
//   * `vec.len()  >= cols`              (input vector fully covers a row)
//   * `data.len() >= rows * (cols / block_elems) * block_bytes`
//                                       (weight buffer covers every row)
//
// These hold for weights loaded from a validated GGUF descriptor. `check_dims`
// asserts them in debug builds so a violated invariant trips a panic in tests
// instead of silently reading out of bounds; it compiles to nothing in release
// so the hot path is unaffected.

/// Debug-only precondition check for a quantized matvec kernel.
#[cfg(target_arch = "x86_64")]
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
    let shuf = _mm_movehdup_ps(sum128); // [b,b,d,d]
    let sums = _mm_add_ps(sum128, shuf); // [a+b, -, c+d, -]
    let hi64 = _mm_movehl_ps(sums, sums); // [c+d, -, -, -]
    let total = _mm_add_ss(sums, hi64);
    _mm_cvtss_f32(total)
}

/// Convert 32 signed i8 values (in a __m256i) into 4×__m256 of f32.
///
/// The i8→f32 conversion pipeline:
///   __m256i (32×i8) → split into 2×__m128i (16×i8 each)
///   → cvtepi8_epi16 → 2×__m256i (16×i16 each)
///   → split each into 2×__m128i (8×i16 each)
///   → cvtepi16_epi32 → 4×__m256i (8×i32 each)
///   → cvtepi32_ps → 4×__m256 (8×f32 each)
///
/// Split out from [`accum_block_q8`] so a batched kernel can decode a weight
/// block **once** and then apply it to every sequence's activation vector —
/// the decode is pure integer work, so single and batched kernels feed
/// bit-identical `w` registers to the FMA chain below.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_i8x32(weights_i8: __m256i) -> [__m256; 4] {
    // Split 32 i8 values into lower and upper 16
    let lo_i8 = _mm256_castsi256_si128(weights_i8);
    let hi_i8 = _mm256_extracti128_si256::<1>(weights_i8);

    // Sign-extend i8 → i16 (16 values each, 256-bit output)
    let lo_i16 = _mm256_cvtepi8_epi16(lo_i8);
    let hi_i16 = _mm256_cvtepi8_epi16(hi_i8);

    // Convert i16 → i32 → f32, 8 values at a time (4 groups total)
    [
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_i16))),
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_i16))),
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_i16))),
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_i16))),
    ]
}

/// FMA-accumulate an already-decoded 32-element weight block against one
/// input vector: `acc += scale × dot32(w, input)`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_block_w(w: &[__m256; 4], input_ptr: *const f32, scale: f32, acc: __m256) -> __m256 {
    // Load 32 f32 input values (4×8)
    let v0 = _mm256_loadu_ps(input_ptr);
    let v1 = _mm256_loadu_ps(input_ptr.add(8));
    let v2 = _mm256_loadu_ps(input_ptr.add(16));
    let v3 = _mm256_loadu_ps(input_ptr.add(24));

    // Partial dot products: p = w0*v0 + w1*v1 + w2*v2 + w3*v3
    let p01 = _mm256_fmadd_ps(w[1], v1, _mm256_mul_ps(w[0], v0));
    let p23 = _mm256_fmadd_ps(w[3], v3, _mm256_mul_ps(w[2], v2));
    let block_partial = _mm256_add_ps(p01, p23);

    // Accumulate: acc += scale * block_partial
    let scale_vec = _mm256_set1_ps(scale);
    _mm256_fmadd_ps(scale_vec, block_partial, acc)
}

/// Convert 32 signed i8 values into f32 and FMA-accumulate them against one
/// input vector into `acc`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_block_q8(
    weights_i8: __m256i,
    input_ptr: *const f32,
    scale: f32,
    acc: __m256,
) -> __m256 {
    accum_block_w(&unpack_i8x32(weights_i8), input_ptr, scale, acc)
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
        let values_u8 =
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(interleaved_lo), interleaved_hi);
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
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

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
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_sum_q4k(nibbles: __m256i, input_ptr: *const f32) -> (f32, f32) {
    // Convert 32 nibbles (u8, 0..15) → 4×__m256 of f32.
    // Sign-extend == zero-extend for values 0..15, so cvtepi8_epi16 is correct.
    dot_and_sum_w(&unpack_i8x32(nibbles), input_ptr)
}

/// [`dot_and_sum_q4k`] against an already-decoded weight block, so a batched
/// kernel can decode the nibbles once and reuse them across sequences.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_sum_w(w: &[__m256; 4], input_ptr: *const f32) -> (f32, f32) {
    // Load 32 f32 input values.
    let v0 = _mm256_loadu_ps(input_ptr);
    let v1 = _mm256_loadu_ps(input_ptr.add(8));
    let v2 = _mm256_loadu_ps(input_ptr.add(16));
    let v3 = _mm256_loadu_ps(input_ptr.add(24));

    // Dot product: sum(w × v)
    let d01 = _mm256_fmadd_ps(w[1], v1, _mm256_mul_ps(w[0], v0));
    let d23 = _mm256_fmadd_ps(w[3], v3, _mm256_mul_ps(w[2], v2));
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
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];
        let x_base = sb * SUPER_BLOCK;

        for group in 0..4_usize {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;

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
    check_dims(data.len(), rows, cols, vec.len(), SUPER_BLOCK, BLOCK_BYTES);

    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|r| unsafe { dot_row_q4_k(data, r * bytes_per_row, n_super, vec) })
        .collect()
}

// ── Q5_K kernel ──────────────────────────────────────────────────────────────

/// Compute the dot product of one Q5_K row against the input vector.
///
/// Super-block layout (176 bytes / 256 elements):
///   [f16 d (2B)] [f16 dmin (2B)] [scales u8×12] [qh u8×32] [qs u8×128]
///
/// Q5_K shares the same scale/min and group structure as Q4_K (4 groups of 64,
/// 2 sub-blocks each via `get_scale_min_q4k`). The difference: each 4-bit
/// nibble gains a 5th bit from the packed `qh` array, giving 5-bit values
/// in [0, 31] instead of [0, 15].
///
/// Bit extraction from qh:
///   low  nibbles of group g: bit `2*g`   of qh[l]  (mask = 1 << (2*g))
///   high nibbles of group g: bit `2*g+1` of qh[l]  (mask = 2 << (2*g))
///
/// The 5th bit is extracted with `_mm256_cmpeq_epi8` vs zero + `_mm256_andnot_si256`
/// to produce 0 or 16 per byte, then OR'd with the low nibbles.
/// The resulting 5-bit values (0..31) go straight into `dot_and_sum_q4k`,
/// which sign-extends u8→f32 (values ≤ 31 are safely non-negative in i8). ✓
///
/// Contribution per sub-block:  `d_scale × dot(nibble5, input) − d_min × sum(input)`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_q5_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const SUPER_BLOCK: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let lo_mask = _mm256_set1_epi8(0x0F_u8 as i8);
    let zero = _mm256_setzero_si256();
    let sixteen = _mm256_set1_epi8(16_u8 as i8);

    let mut total = 0.0f32;

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];
        let qh = &b[16..48]; // 32 bytes — one bit per element per sub-block
        let qs = &b[48..176]; // 128 bytes — low 4 bits (same layout as Q4_K)
        let x_base = sb * SUPER_BLOCK;

        // Load all 32 qh bytes once; each group uses two specific bits per byte.
        let qh_v = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);

        for group in 0..4_usize {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;

            // 32 packed nibble bytes (same position as Q4_K)
            let q_base = group * 32;
            let packed = _mm256_loadu_si256(qs[q_base..].as_ptr() as *const __m256i);

            // Extract 4-bit nibbles
            let lo4 = _mm256_and_si256(packed, lo_mask);
            let hi4 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);

            // Bit masks for the 5th bit of each sub-block in this group
            //   u1 = 1 << (2*group)   → bit for low nibbles
            //   u2 = 1 << (2*group+1) → bit for high nibbles
            let u1: u8 = 1 << (group * 2);
            let u2: u8 = 1 << (group * 2 + 1);

            // For each of the 32 qh bytes: test bit u1 → 0 or 16 per byte.
            //   cmpeq_epi8(and, zero) → 0xFF where bit was 0, 0x00 where bit was 1
            //   andnot(is_zero, 16)   → 16 where bit was 1, 0 where bit was 0  ✓
            let hi_bit_lo = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(qh_v, _mm256_set1_epi8(u1 as i8)), zero),
                sixteen,
            );
            let hi_bit_hi = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(qh_v, _mm256_set1_epi8(u2 as i8)), zero),
                sixteen,
            );

            // 5-bit values in [0, 31]: low nibble | 5th bit (no overlap, OR = ADD)
            let nibbles5_lo = _mm256_or_si256(lo4, hi_bit_lo);
            let nibbles5_hi = _mm256_or_si256(hi4, hi_bit_hi);

            let x_lo = vec.as_ptr().add(x_base + group * 64);
            let x_hi = x_lo.add(32);

            // Reuse Q4_K dot-and-sum: values 0..31 sign-extend to correct f32 ✓
            let (dot_lo, sum_lo) = dot_and_sum_q4k(nibbles5_lo, x_lo);
            let (dot_hi, sum_hi) = dot_and_sum_q4k(nibbles5_hi, x_hi);

            total += d0 * dot_lo - m0 * sum_lo;
            total += d1 * dot_hi - m1 * sum_hi;
        }
    }

    total
}

/// Q5_K matrix-vector multiply using AVX2 + FMA + rayon.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q5_k_avx2(
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

// ── Q6_K helpers ─────────────────────────────────────────────────────────────

/// Dot product of 16 signed i8 weights × 16 f32 inputs → f32.
///
/// i8 values are sign-extended to i16 → i32 → f32, then multiplied and summed.
/// Handles 16 elements using two AVX2 registers of 8 f32 each.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot16_i8_f32(weights: __m128i, input_ptr: *const f32) -> f32 {
    dot16_w(&unpack_i8x16(weights), input_ptr)
}

/// Sign-extend 16 × i8 into 2×__m256 of f32.
///
/// Split out of [`dot16_i8_f32`] so the batched Q6_K kernel decodes each
/// 16-element group once and dots it against every sequence.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_i8x16(weights: __m128i) -> [__m256; 2] {
    // Sign-extend 16 × i8 → 16 × i16 (256-bit register)
    let w_i16 = _mm256_cvtepi8_epi16(weights);
    // Convert two groups of 8 i16 → i32 → f32
    [
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w_i16))),
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(w_i16))),
    ]
}

/// [`dot16_i8_f32`] against an already-decoded 16-element weight group.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot16_w(w: &[__m256; 2], input_ptr: *const f32) -> f32 {
    // Load 16 f32 input values
    let v0 = _mm256_loadu_ps(input_ptr);
    let v1 = _mm256_loadu_ps(input_ptr.add(8));
    // Dot: sum(w0*v0 + w1*v1)
    hsum_avx2(_mm256_fmadd_ps(w[1], v1, _mm256_mul_ps(w[0], v0)))
}

/// Compute the dot product of one Q6_K row against the input vector.
///
/// Super-block layout (210 bytes / 256 elements):
///   [ql u8×128] [qh u8×64] [scales i8×16] [f16 d]
///
/// Each super-block splits into 2 groups of 128. Within each group,
/// 32 × l-index assembles 4 non-contiguous output positions (l, l+32, l+64,
/// l+96) from `ql` and `qh`. The 16 scales change at l=16 (is=0→1), so
/// we process each group in two halves of 16 elements.
///
/// 6-bit assembly per element:
///   q1 = (ql[l]    & 0x0F) | ((qh[l] & 0x03)      << 4) − 32
///   q2 = (ql[l+32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4) − 32
///   q3 = (ql[l]    >>   4) | (((qh[l] >> 4) & 0x03) << 4) − 32
///   q4 = (ql[l+32] >>   4) | (((qh[l] >> 6) & 0x03) << 4) − 32
///
/// The `<< 4` inside each byte uses `_mm_mullo_epi16(val_0..3, 16)`:
/// for a 16-bit lane [B1, B0] with B0, B1 ∈ 0..3, this produces
/// [B1×16, B0×16] — correct byte-level ×16 without AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_q6_k(data: &[u8], row_start: usize, n_super: usize, vec: &[f32]) -> f32 {
    const BLOCK_BYTES: usize = 210;
    const SUPER_BLOCK: usize = 256;

    let lo4 = _mm_set1_epi8(0x0F_u8 as i8); // mask low nibble
    let mask03 = _mm_set1_epi8(0x03_u8 as i8); // mask 2 bits
    let mul16 = _mm_set1_epi16(16); // ×16 per byte (values 0..3 only)
    let sub32 = _mm_set1_epi8(32_u8 as i8); // centering offset

    let mut total = 0.0f32;

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208]; // 16 × i8 sub-block scales
        let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
        let x_base = sb * SUPER_BLOCK;

        for group in 0..2_usize {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let x_group = x_base + group * 128;

            // Each group has 32 l-values; the scale index `is = l / 16` changes
            // at l=16, so we process in two halves.
            for half in 0..2_usize {
                let l = half * 16;

                // Sub-block scales for this (group, half): indices 0,2,4,6 + half
                let sc0 = d * sc[sc_off + half] as i8 as f32;
                let sc2 = d * sc[sc_off + half + 2] as i8 as f32;
                let sc4 = d * sc[sc_off + half + 4] as i8 as f32;
                let sc6 = d * sc[sc_off + half + 6] as i8 as f32;

                // Load 16 raw bytes from each needed region
                let ql_a = _mm_loadu_si128(ql[ql_off + l..].as_ptr() as *const __m128i);
                let ql_b = _mm_loadu_si128(ql[ql_off + 32 + l..].as_ptr() as *const __m128i);
                let qhv = _mm_loadu_si128(qh[qh_off + l..].as_ptr() as *const __m128i);

                // ── Assemble 6-bit values (0..63) then subtract 32 → i8 (−32..+31) ──
                //
                // Byte-level ×16 via mullo_epi16: for a 16-bit lane [B1,B0] with
                // each byte ∈ 0..3, multiplying by 16 gives [B1×16, B0×16].
                // The ql high-nibble extraction uses srli_epi16+lo4 which, for
                // each byte B in a 16-bit lane: gives B>>4 in the low nibble. ✓

                // q1 = (ql_a & 0x0F) | ((qh & 0x03) << 4) − 32
                let q1 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql_a, lo4),
                        _mm_mullo_epi16(_mm_and_si128(qhv, mask03), mul16),
                    ),
                    sub32,
                );

                // q2 = (ql_b & 0x0F) | (((qh >> 2) & 0x03) << 4) − 32
                let q2 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql_b, lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 2), mask03), mul16),
                    ),
                    sub32,
                );

                // q3 = (ql_a >> 4) | (((qh >> 4) & 0x03) << 4) − 32
                let q3 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql_a, 4), lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 4), mask03), mul16),
                    ),
                    sub32,
                );

                // q4 = (ql_b >> 4) | (((qh >> 6) & 0x03) << 4) − 32
                let q4 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql_b, 4), lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 6), mask03), mul16),
                    ),
                    sub32,
                );

                // Dot each 16-element region against its input slice
                total += sc0 * dot16_i8_f32(q1, vec.as_ptr().add(x_group + l));
                total += sc2 * dot16_i8_f32(q2, vec.as_ptr().add(x_group + 32 + l));
                total += sc4 * dot16_i8_f32(q3, vec.as_ptr().add(x_group + 64 + l));
                total += sc6 * dot16_i8_f32(q4, vec.as_ptr().add(x_group + 96 + l));
            }
        }
    }

    total
}

/// Q6_K matrix-vector multiply using AVX2 + FMA + rayon.
///
/// Rows distributed across threads; each row uses `dot_row_q6_k` which
/// processes 16-element chunks with AVX2 6-bit assembly and FMA dot products.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q6_k_avx2(
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
    check_dims(data.len(), rows, cols, vec.len(), BLOCK_ELEMS, BLOCK_BYTES);

    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    (0..rows)
        .into_par_iter()
        .map(|i| unsafe { dot_row_q4_0(data, i * bytes_per_row, n_blocks, vec) })
        .collect()
}

// ── Batched kernels (continuous batching) ────────────────────────────────────
//
// A batched kernel walks the weight matrix **once** and applies every decoded
// block to all `B` activation vectors before moving on, so the cost of
// streaming the weights through the cache hierarchy is amortised across the
// batch instead of paid once per sequence.
//
// ## Numerical parity
//
// Each sequence keeps its own accumulator and sees the exact same sequence of
// floating-point operations, in the same order, as the single-vector kernels
// above: the shared `unpack_*` step is pure integer work, and the per-sequence
// `accum_block_w` / `dot_and_sum_w` / `dot16_w` calls are the very functions
// the single-vector path uses. Batched output is therefore bit-identical to
// running each sequence on its own — pinned by the `matvec_batch` tests in
// `quantized.rs`.
//
// `out` is `[rows, B]` interleaved: `out[row * B + s]` belongs to sequence `s`.
// Rows stay the unit of rayon parallelism (outer); sequences are the inner
// loop, exactly as in the single-vector kernels.

/// Maximum sequences a batched kernel accumulates in one weight traversal.
///
/// Bounds the register/stack space a kernel needs for its per-sequence
/// accumulators. A larger batch is split into chunks of this size, each chunk
/// re-walking the row — still `ceil(B / MAX_BATCH_LANES)` traversals instead of
/// `B`, and far above the batch sizes CPU serving actually runs at.
const MAX_BATCH_LANES: usize = 16;

/// Debug-only precondition check for a batched kernel: every input vector must
/// satisfy the single-vector contract, and `out` must hold `rows × B` results.
#[cfg(target_arch = "x86_64")]
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
    debug_assert!(
        !inputs.is_empty(),
        "batched matvec needs at least one input"
    );
    for v in inputs {
        check_dims(data_len, rows, cols, v.len(), block_elems, block_bytes);
    }
    debug_assert_eq!(
        out_len,
        rows * inputs.len(),
        "batched matvec output must be rows × batch"
    );
}

/// Dot one Q8_0 row against up to [`MAX_BATCH_LANES`] input vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
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

    let mut acc = [_mm256_setzero_ps(); MAX_BATCH_LANES];
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let weights = _mm256_loadu_si256(block[2..].as_ptr() as *const __m256i);
        let w = unpack_i8x32(weights);
        for (s, input) in inputs.iter().enumerate() {
            acc[s] = accum_block_w(&w, input.as_ptr().add(b * BLOCK_ELEMS), scale, acc[s]);
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = hsum_avx2(acc[s]);
    }
}

/// Batched Q8_0 matrix-vector multiply (AVX2 + FMA + rayon over rows).
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q8_0_batch_avx2(
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
            // SAFETY: the caller verified AVX2+FMA and `check_dims_batch`
            // pinned the buffer lengths, so every load below stays in bounds.
            // `dst` is this row's `batch`-wide slice of the output.
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

/// Dot one Q4_0 row against up to [`MAX_BATCH_LANES`] input vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
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

    let low_mask = _mm_set1_epi8(0x0F);
    let offset_8 = _mm256_set1_epi8(8);

    let mut acc = [_mm256_setzero_ps(); MAX_BATCH_LANES];
    for b in 0..n_blocks {
        let block = &data[row_start + b * BLOCK_BYTES..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let packed = _mm_loadu_si128(block[2..].as_ptr() as *const __m128i);
        let lo_nib = _mm_and_si128(packed, low_mask);
        let hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let interleaved_lo = _mm_unpacklo_epi8(lo_nib, hi_nib);
        let interleaved_hi = _mm_unpackhi_epi8(lo_nib, hi_nib);
        let values_u8 =
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(interleaved_lo), interleaved_hi);
        let centered = _mm256_sub_epi8(values_u8, offset_8);
        let w = unpack_i8x32(centered);

        for (s, input) in inputs.iter().enumerate() {
            acc[s] = accum_block_w(&w, input.as_ptr().add(b * BLOCK_ELEMS), scale, acc[s]);
        }
    }
    for (s, o) in out.iter_mut().enumerate() {
        *o = hsum_avx2(acc[s]);
    }
}

/// Batched Q4_0 matrix-vector multiply (AVX2 + FMA + rayon over rows).
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q4_0_batch_avx2(
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
            // SAFETY: see `matvec_q8_0_batch_avx2`.
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

/// Dot one Q4_K row against up to [`MAX_BATCH_LANES`] input vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
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

    let lo_mask = _mm256_set1_epi8(0x0F_u8 as i8);
    let mut total = [0.0f32; MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];
        let qs = &b[16..144];
        let x_base = sb * SUPER_BLOCK;

        for group in 0..4_usize {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;

            let q_base = group * 32;
            let packed = _mm256_loadu_si256(qs[q_base..].as_ptr() as *const __m256i);
            let lo_nibbles = _mm256_and_si256(packed, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);
            let w_lo = unpack_i8x32(lo_nibbles);
            let w_hi = unpack_i8x32(hi_nibbles);

            for (s, input) in inputs.iter().enumerate() {
                let x_lo = input.as_ptr().add(x_base + group * 64);
                let x_hi = x_lo.add(32);
                let (dot_lo, sum_lo) = dot_and_sum_w(&w_lo, x_lo);
                let (dot_hi, sum_hi) = dot_and_sum_w(&w_hi, x_hi);
                total[s] += d0 * dot_lo - m0 * sum_lo;
                total[s] += d1 * dot_hi - m1 * sum_hi;
            }
        }
    }
    out.copy_from_slice(&total[..out.len()]);
}

/// Batched Q4_K matrix-vector multiply (AVX2 + FMA + rayon over rows).
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q4_k_batch_avx2(
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
            // SAFETY: see `matvec_q8_0_batch_avx2`.
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

/// Dot one Q5_K row against up to [`MAX_BATCH_LANES`] input vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
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

    let lo_mask = _mm256_set1_epi8(0x0F_u8 as i8);
    let zero = _mm256_setzero_si256();
    let sixteen = _mm256_set1_epi8(16_u8 as i8);
    let mut total = [0.0f32; MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        let x_base = sb * SUPER_BLOCK;

        let qh_v = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);

        for group in 0..4_usize {
            let (sc0, mn0) = get_scale_min_q4k(group * 2, scales);
            let (sc1, mn1) = get_scale_min_q4k(group * 2 + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;

            let q_base = group * 32;
            let packed = _mm256_loadu_si256(qs[q_base..].as_ptr() as *const __m256i);
            let lo4 = _mm256_and_si256(packed, lo_mask);
            let hi4 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);

            let u1: u8 = 1 << (group * 2);
            let u2: u8 = 1 << (group * 2 + 1);
            let hi_bit_lo = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(qh_v, _mm256_set1_epi8(u1 as i8)), zero),
                sixteen,
            );
            let hi_bit_hi = _mm256_andnot_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(qh_v, _mm256_set1_epi8(u2 as i8)), zero),
                sixteen,
            );
            let w_lo = unpack_i8x32(_mm256_or_si256(lo4, hi_bit_lo));
            let w_hi = unpack_i8x32(_mm256_or_si256(hi4, hi_bit_hi));

            for (s, input) in inputs.iter().enumerate() {
                let x_lo = input.as_ptr().add(x_base + group * 64);
                let x_hi = x_lo.add(32);
                let (dot_lo, sum_lo) = dot_and_sum_w(&w_lo, x_lo);
                let (dot_hi, sum_hi) = dot_and_sum_w(&w_hi, x_hi);
                total[s] += d0 * dot_lo - m0 * sum_lo;
                total[s] += d1 * dot_hi - m1 * sum_hi;
            }
        }
    }
    out.copy_from_slice(&total[..out.len()]);
}

/// Batched Q5_K matrix-vector multiply (AVX2 + FMA + rayon over rows).
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q5_k_batch_avx2(
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
            // SAFETY: see `matvec_q8_0_batch_avx2`.
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

/// Dot one Q6_K row against up to [`MAX_BATCH_LANES`] input vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
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

    let lo4 = _mm_set1_epi8(0x0F_u8 as i8);
    let mask03 = _mm_set1_epi8(0x03_u8 as i8);
    let mul16 = _mm_set1_epi16(16);
    let sub32 = _mm_set1_epi8(32_u8 as i8);

    let mut total = [0.0f32; MAX_BATCH_LANES];

    for sb in 0..n_super {
        let b = &data[row_start + sb * BLOCK_BYTES..];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
        let x_base = sb * SUPER_BLOCK;

        for group in 0..2_usize {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let x_group = x_base + group * 128;

            for half in 0..2_usize {
                let l = half * 16;

                let sc0 = d * sc[sc_off + half] as i8 as f32;
                let sc2 = d * sc[sc_off + half + 2] as i8 as f32;
                let sc4 = d * sc[sc_off + half + 4] as i8 as f32;
                let sc6 = d * sc[sc_off + half + 6] as i8 as f32;

                let ql_a = _mm_loadu_si128(ql[ql_off + l..].as_ptr() as *const __m128i);
                let ql_b = _mm_loadu_si128(ql[ql_off + 32 + l..].as_ptr() as *const __m128i);
                let qhv = _mm_loadu_si128(qh[qh_off + l..].as_ptr() as *const __m128i);

                let q1 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql_a, lo4),
                        _mm_mullo_epi16(_mm_and_si128(qhv, mask03), mul16),
                    ),
                    sub32,
                );
                let q2 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql_b, lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 2), mask03), mul16),
                    ),
                    sub32,
                );
                let q3 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql_a, 4), lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 4), mask03), mul16),
                    ),
                    sub32,
                );
                let q4 = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql_b, 4), lo4),
                        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 6), mask03), mul16),
                    ),
                    sub32,
                );

                let w1 = unpack_i8x16(q1);
                let w2 = unpack_i8x16(q2);
                let w3 = unpack_i8x16(q3);
                let w4 = unpack_i8x16(q4);

                for (s, input) in inputs.iter().enumerate() {
                    let x = input.as_ptr();
                    total[s] += sc0 * dot16_w(&w1, x.add(x_group + l));
                    total[s] += sc2 * dot16_w(&w2, x.add(x_group + 32 + l));
                    total[s] += sc4 * dot16_w(&w3, x.add(x_group + 64 + l));
                    total[s] += sc6 * dot16_w(&w4, x.add(x_group + 96 + l));
                }
            }
        }
    }
    out.copy_from_slice(&total[..out.len()]);
}

/// Batched Q6_K matrix-vector multiply (AVX2 + FMA + rayon over rows).
///
/// # Safety
/// Caller must verify AVX2 and FMA are available via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_q6_k_batch_avx2(
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
            // SAFETY: see `matvec_q8_0_batch_avx2`.
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
