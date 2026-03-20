//! Flash Attention for the single-query (decode) case.
//!
//! Implements the online softmax trick from Dao et al. (2022) to compute
//! attention without ever materialising the full seq_len score vector.
//! Working memory per head is O(BLOCK_SIZE + head_dim) — all on the stack.

use crate::cache::KvCache;

/// Number of key/value positions processed per inner loop iteration.
/// 32 positions × 4 bytes = 128 bytes for the score buffer, comfortably in L1.
/// Combined with a head_dim=128 accumulator (512 bytes), well under 1 KB per head.
const BLOCK_SIZE: usize = 32;

/// Compute single-query attention against all cached positions using the
/// online softmax (Flash Attention) algorithm.
///
/// Fuses the three classical phases — score, softmax, V-accumulate — into a
/// single tiled pass with no heap allocations.
///
/// # Arguments
/// * `q`        — query vector for this head, length `head_dim`
/// * `cache`    — KV cache for all layers
/// * `layer`    — transformer layer index
/// * `kv_h`     — KV head index (for GQA/MQA, each Q head maps to one KV head)
/// * `seq_len`  — number of positions to attend to (pos + 1)
/// * `head_dim` — per-head dimension
/// * `scale`    — pre-computed `1 / sqrt(head_dim)`
/// * `out`      — output slice of length `head_dim`; **must be pre-zeroed**
pub fn flash_attn_1d(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    kv_h: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), head_dim);
    debug_assert_eq!(out.len(), head_dim);

    let mut m = f32::NEG_INFINITY; // running global max of scores seen so far
    let mut l = 0.0f32;            // running normaliser (sum of exp weights)
    // `out` starts zeroed (caller's responsibility) and serves as the running
    // unnormalised output accumulator throughout the loop.

    let mut start = 0;
    while start < seq_len {
        let end = (start + BLOCK_SIZE).min(seq_len);
        let block_len = end - start;

        // Stack buffer for this block's raw attention scores
        let mut scores = [0.0f32; BLOCK_SIZE];

        // Phase 1 — Q · K scores for each position in the block
        for i in 0..block_len {
            let k_row = cache.k_at(layer, start + i);
            let k_head = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_head[d];
            }
            scores[i] = dot * scale;
        }

        // Phase 2 — update the running maximum
        let mut m_block = f32::NEG_INFINITY;
        for i in 0..block_len {
            if scores[i] > m_block {
                m_block = scores[i];
            }
        }
        let m_new = m.max(m_block);

        // Rescale the old accumulator by exp(m_old - m_new).
        // On the very first block m == NEG_INFINITY, so rescale == 0 and the
        // still-zero `out` / `l` are left unchanged — correct initialisation.
        let rescale = (m - m_new).exp();
        for d in 0..head_dim {
            out[d] *= rescale;
        }
        l *= rescale;

        // Phase 3 — incorporate this block's V contributions
        for i in 0..block_len {
            let e = (scores[i] - m_new).exp(); // always ≤ 1.0, no overflow
            let v_row = cache.v_at(layer, start + i);
            let v_head = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            for d in 0..head_dim {
                out[d] += e * v_head[d];
            }
            l += e;
        }

        m = m_new;
        start = end;
    }

    // Final normalisation — divide unnormalised accumulator by the sum of weights
    if l > 0.0 {
        let inv_l = 1.0 / l;
        for d in 0..head_dim {
            out[d] *= inv_l;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;

    /// Build a tiny KvCache pre-filled with deterministic data.
    fn build_cache(
        seq_len: usize,
        head_dim: usize,
        n_kv_heads: usize,
        k_data: &[Vec<f32>],
        v_data: &[Vec<f32>],
    ) -> KvCache {
        let mut cache = KvCache::new(1, seq_len + 1, n_kv_heads, head_dim);
        for pos in 0..seq_len {
            cache.write(0, pos, &k_data[pos], &v_data[pos]);
            cache.advance();
        }
        cache
    }

    /// Reference: standard three-phase attention (allocates freely — oracle only).
    fn standard_attn(
        q: &[f32],
        cache: &KvCache,
        layer: usize,
        kv_h: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        // Scores
        let mut scores = vec![0.0f32; seq_len];
        for s in 0..seq_len {
            let k_row = cache.k_at(layer, s);
            let k_head = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_head[d];
            }
            scores[s] = dot * scale;
        }
        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        // V accumulation
        let mut out = vec![0.0f32; head_dim];
        for s in 0..seq_len {
            let v_row = cache.v_at(layer, s);
            let v_head = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            for d in 0..head_dim {
                out[d] += weights[s] * v_head[d];
            }
        }
        out
    }

    fn check(seq_len: usize, head_dim: usize, n_kv_heads: usize) {
        let kv_h = 0;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_dim = n_kv_heads * head_dim;

        let k_data: Vec<Vec<f32>> = (0..seq_len)
            .map(|s| (0..kv_dim).map(|d| (s * kv_dim + d) as f32 * 0.1 - 0.5).collect())
            .collect();
        let v_data: Vec<Vec<f32>> = (0..seq_len)
            .map(|s| (0..kv_dim).map(|d| (s * kv_dim + d) as f32 * 0.07 + 0.1).collect())
            .collect();
        let q: Vec<f32> = (0..head_dim).map(|d| d as f32 * 0.05 - 0.1).collect();

        let cache = build_cache(seq_len, head_dim, n_kv_heads, &k_data, &v_data);
        let expected = standard_attn(&q, &cache, 0, kv_h, seq_len, head_dim, scale);

        let mut got = vec![0.0f32; head_dim];
        flash_attn_1d(&q, &cache, 0, kv_h, seq_len, head_dim, scale, &mut got);

        for d in 0..head_dim {
            let diff = (expected[d] - got[d]).abs();
            assert!(
                diff < 1e-4,
                "seq_len={seq_len} head_dim={head_dim}: dim {d}: expected={:.6}, got={:.6}, diff={diff:.2e}",
                expected[d], got[d]
            );
        }
    }

    #[test]
    fn test_flash_vs_standard_attn_aligned() {
        // seq_len=64 is exactly 2 × BLOCK_SIZE — tests the aligned path
        check(64, 4, 2);
    }

    #[test]
    fn test_flash_vs_standard_attn_unaligned() {
        // seq_len=50: two full blocks (0..32, 32..50), last block has 18 elements
        check(50, 4, 2);
    }

    #[test]
    fn test_flash_vs_standard_attn_seq_len_1() {
        // Edge case: single cached position
        check(1, 4, 2);
    }

    #[test]
    fn test_flash_vs_standard_large_head_dim() {
        // Larger head_dim to catch any index arithmetic mistake
        check(35, 128, 1);
    }
}
