//! KV-cache — stores computed K and V vectors to avoid recomputation.
//!
//! Without a cache, generating token N requires recomputing K and V for
//! all N positions at every layer. With a cache, we compute K/V once per
//! position and reuse them, reducing per-token work from O(N) matvecs to O(1).

use half::f16;

// ── KvStore trait ────────────────────────────────────────────────────────────

/// Abstraction over KV-cache storage formats (f32 or quantised INT8).
///
/// Both `KvCache` (f32) and `KvCacheQ8` (Q8_0 compressed) implement this
/// trait so that `flash_attn_1d` and the forward pass are format-agnostic.
pub trait KvStore: Send + Sync {
    /// Fill `buf` (length `head_dim`) with the K vector for `kv_h`-th head.
    fn read_k_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]);

    /// Fill `buf` (length `head_dim`) with the V vector for `kv_h`-th head.
    fn read_v_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]);

    /// Write K and V for one position (all KV heads concatenated) in one layer.
    fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]);

    /// Increment the cached length by one. Call after writing all layers for a position.
    fn advance(&mut self);

    /// Number of valid positions currently in the cache.
    fn len(&self) -> usize;

    /// True if no positions have been written.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Truncate to `new_len` valid positions (used to roll back speculative tokens).
    fn truncate(&mut self, new_len: usize);

    /// Reset to zero positions for a new sequence.
    fn clear(&mut self) { self.truncate(0); }
}

// ── F32 KV-cache ─────────────────────────────────────────────────────────────

/// Pre-allocated KV cache with f32 storage.
///
/// Row `pos` in layer `l` starts at index `pos * kv_dim` in `k[l]` / `v[l]`,
/// where `kv_dim = n_kv_heads * head_dim`.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    max_seq_len: usize,
    len: usize,
}

impl KvCache {
    /// Allocate a new zeroed f32 KV cache.
    pub fn new(n_layers: usize, max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        let layer_size = max_seq_len * kv_dim;
        Self {
            k: (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect(),
            v: (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect(),
            kv_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Direct slice access — returns the full kv_dim-length K row for `pos`.
    /// Used by tests and flash.rs's reference `standard_attn` implementation.
    pub fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.k[layer][offset..offset + self.kv_dim]
    }

    /// Direct slice access — returns the full kv_dim-length V row for `pos`.
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.v[layer][offset..offset + self.kv_dim]
    }
}

impl KvStore for KvCache {
    fn read_k_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = pos * self.kv_dim + kv_h * head_dim;
        buf.copy_from_slice(&self.k[layer][start..start + head_dim]);
    }

    fn read_v_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = pos * self.kv_dim + kv_h * head_dim;
        buf.copy_from_slice(&self.v[layer][start..start + head_dim]);
    }

    fn write(&mut self, layer: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        assert!(pos < self.max_seq_len, "KV-cache overflow: pos {pos} >= {}", self.max_seq_len);
        debug_assert_eq!(k_vec.len(), self.kv_dim);
        debug_assert_eq!(v_vec.len(), self.kv_dim);
        let offset = pos * self.kv_dim;
        self.k[layer][offset..offset + self.kv_dim].copy_from_slice(k_vec);
        self.v[layer][offset..offset + self.kv_dim].copy_from_slice(v_vec);
    }

    fn advance(&mut self) { self.len += 1; }
    fn len(&self) -> usize { self.len }
    fn truncate(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "truncate: {new_len} > len {}", self.len);
        self.len = new_len;
    }
}

// ── Q8_0 KV-cache ─────────────────────────────────────────────────────────────

/// KV cache with Q8_0-compressed storage (~3.8× less RAM than f32).
///
/// Each 32-element block is encoded as `[f16 d][32 × i8]` = 34 bytes, versus
/// 128 bytes for f32 — a 3.76× compression ratio.
///
/// Write quantises f32 → Q8_0; reads dequantise on the fly into a caller-
/// supplied scratch buffer, keeping the hot path allocation-free.
pub struct KvCacheQ8 {
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    kv_dim: usize,
    max_seq_len: usize,
    len: usize,
    bytes_per_pos: usize, // = kv_dim.div_ceil(32) * 34
}

impl KvCacheQ8 {
    /// Allocate a new Q8_0-compressed KV cache.
    pub fn new(n_layers: usize, max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        let n_blocks = kv_dim.div_ceil(32);
        let bytes_per_pos = n_blocks * 34;
        let layer_size = max_seq_len * bytes_per_pos;
        Self {
            k: (0..n_layers).map(|_| vec![0u8; layer_size]).collect(),
            v: (0..n_layers).map(|_| vec![0u8; layer_size]).collect(),
            kv_dim,
            max_seq_len,
            len: 0,
            bytes_per_pos,
        }
    }

    fn quantize_into(src: &[f32], dst: &mut [u8]) {
        let n_blocks = src.len().div_ceil(32);
        for b in 0..n_blocks {
            let start = b * 32;
            let end = (start + 32).min(src.len());
            let block = &src[start..end];
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_d = 1.0 / d;
            let out = &mut dst[b * 34..];
            out[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            for (j, &v) in block.iter().enumerate() {
                out[2 + j] = (v * inv_d).round().clamp(-127.0, 127.0) as i8 as u8;
            }
            // Zero-pad any partial last block
            for j in block.len()..32 {
                out[2 + j] = 0;
            }
        }
    }

    /// Dequantise elements `[start_elem, end_elem)` from a Q8_0-encoded row into `buf`.
    fn dequant_range(row: &[u8], start_elem: usize, end_elem: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), end_elem - start_elem);
        let first_block = start_elem / 32;
        let last_block = (end_elem - 1) / 32;
        let mut out_idx = 0;
        for b in first_block..=last_block {
            let block = &row[b * 34..];
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for j in 0..32 {
                let elem = b * 32 + j;
                if elem >= start_elem && elem < end_elem {
                    buf[out_idx] = block[2 + j] as i8 as f32 * d;
                    out_idx += 1;
                }
            }
        }
    }
}

impl KvStore for KvCacheQ8 {
    fn read_k_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = pos * self.bytes_per_pos;
        Self::dequant_range(&self.k[layer][start..start + self.bytes_per_pos],
            kv_h * head_dim, (kv_h + 1) * head_dim, buf);
    }

    fn read_v_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = pos * self.bytes_per_pos;
        Self::dequant_range(&self.v[layer][start..start + self.bytes_per_pos],
            kv_h * head_dim, (kv_h + 1) * head_dim, buf);
    }

    fn write(&mut self, layer: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        assert!(pos < self.max_seq_len, "KV-cache overflow: pos {pos} >= {}", self.max_seq_len);
        debug_assert_eq!(k_vec.len(), self.kv_dim);
        debug_assert_eq!(v_vec.len(), self.kv_dim);
        let start = pos * self.bytes_per_pos;
        Self::quantize_into(k_vec, &mut self.k[layer][start..start + self.bytes_per_pos]);
        Self::quantize_into(v_vec, &mut self.v[layer][start..start + self.bytes_per_pos]);
    }

    fn advance(&mut self) { self.len += 1; }
    fn len(&self) -> usize { self.len }
    fn truncate(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "truncate: {new_len} > len {}", self.len);
        self.len = new_len;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_write_read() {
        let mut cache = KvCache::new(2, 8, 1, 4);
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        cache.write(0, 0, &k, &v);
        cache.advance();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.k_at(0, 0), &k);
        assert_eq!(cache.v_at(0, 0), &v);
    }

    #[test]
    fn test_kv_cache_multiple_positions() {
        let mut cache = KvCache::new(1, 8, 2, 3);
        let k0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v0 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        cache.write(0, 0, &k0, &v0);
        cache.advance();
        let k1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let v1 = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0];
        cache.write(0, 1, &k1, &v1);
        cache.advance();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.k_at(0, 0), &k0);
        assert_eq!(cache.k_at(0, 1), &k1);
        assert_eq!(cache.v_at(0, 0), &v0);
        assert_eq!(cache.v_at(0, 1), &v1);
    }

    #[test]
    fn test_kv_cache_multiple_layers() {
        let mut cache = KvCache::new(3, 8, 1, 2);
        cache.write(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.write(1, 0, &[5.0, 6.0], &[7.0, 8.0]);
        cache.write(2, 0, &[9.0, 10.0], &[11.0, 12.0]);
        cache.advance();
        assert_eq!(cache.k_at(0, 0), &[1.0, 2.0]);
        assert_eq!(cache.k_at(1, 0), &[5.0, 6.0]);
        assert_eq!(cache.k_at(2, 0), &[9.0, 10.0]);
        assert_eq!(cache.v_at(2, 0), &[11.0, 12.0]);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(1, 8, 1, 4);
        cache.write(0, 0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_truncate() {
        let mut cache = KvCache::new(1, 8, 1, 4);
        for i in 0..4usize {
            cache.write(0, i, &[i as f32; 4], &[i as f32; 4]);
            cache.advance();
        }
        assert_eq!(cache.len(), 4);
        cache.truncate(2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_kv_cache_q8_roundtrip() {
        // 1 KV head, head_dim=32 → kv_dim=32 (exactly 1 Q8_0 block)
        let mut cache = KvCacheQ8::new(1, 4, 1, 32);
        let k: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 8.0).collect();
        let v: Vec<f32> = (0..32).map(|i| i as f32 * 0.3 + 1.0).collect();
        cache.write(0, 0, &k, &v);
        cache.advance();

        let mut k_out = vec![0.0f32; 32];
        let mut v_out = vec![0.0f32; 32];
        cache.read_k_head(0, 0, 0, 32, &mut k_out);
        cache.read_v_head(0, 0, 0, 32, &mut v_out);

        for (orig, dec) in k.iter().zip(&k_out) {
            assert!((orig - dec).abs() < 0.12, "K: {orig:.3} vs {dec:.3}");
        }
        for (orig, dec) in v.iter().zip(&v_out) {
            assert!((orig - dec).abs() < 0.12, "V: {orig:.3} vs {dec:.3}");
        }
    }

    #[test]
    fn test_kv_cache_q8_memory_vs_f32() {
        // Q8 should use ~3.76× less memory than f32
        let (n_layers, max_seq, n_kv_heads, head_dim) = (1, 1024, 4, 32);
        let q8 = KvCacheQ8::new(n_layers, max_seq, n_kv_heads, head_dim);
        let q8_bytes: usize = q8.k.iter().chain(q8.v.iter()).map(|l| l.len()).sum();
        let f32_bytes = n_layers * max_seq * n_kv_heads * head_dim * 4 * 2;
        let ratio = f32_bytes as f64 / q8_bytes as f64;
        assert!(ratio > 3.5 && ratio < 4.0, "Compression ratio {ratio:.2} not in [3.5, 4.0]");
    }

    #[test]
    fn test_kv_cache_q8_multi_head() {
        // 2 KV heads, head_dim=32 → kv_dim=64 (2 blocks)
        let mut cache = KvCacheQ8::new(1, 4, 2, 32);
        let k: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        cache.write(0, 0, &k, &vec![0.0f32; 64]);
        cache.advance();
        let mut head1_k = vec![0.0f32; 32];
        cache.read_k_head(0, 0, 1, 32, &mut head1_k);
        for (i, (&orig, &dec)) in k[32..].iter().zip(&head1_k).enumerate() {
            assert!((orig - dec).abs() < 0.12, "head1 K[{i}]: {orig:.3} vs {dec:.3}");
        }
    }
}
