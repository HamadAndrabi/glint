//! KV-cache — stores computed K and V vectors to avoid recomputation.
//!
//! Without a cache, generating token N requires recomputing K and V for
//! all N positions at every layer. With a cache, we compute K/V once per
//! position and reuse them, reducing per-token work from O(N) matvecs to O(1).

/// Pre-allocated KV cache for all transformer layers.
///
/// For each layer, stores K and V vectors as flat `Vec<f32>` buffers.
/// Row `pos` in layer `l` starts at index `pos * kv_dim` in `k[l]` / `v[l]`,
/// where `kv_dim = n_kv_heads * head_dim`.
pub struct KvCache {
    /// k[layer_idx] — flat buffer holding K vectors for all cached positions.
    k: Vec<Vec<f32>>,
    /// v[layer_idx] — flat buffer holding V vectors for all cached positions.
    v: Vec<Vec<f32>>,
    /// Width of each row: `n_kv_heads * head_dim`.
    kv_dim: usize,
    /// Maximum sequence length this cache can hold.
    max_seq_len: usize,
    /// Number of positions written so far (same across all layers).
    len: usize,
}

impl KvCache {
    /// Allocate a new cache with zeroed buffers.
    ///
    /// # Arguments
    /// * `n_layers` — number of transformer layers (e.g., 30)
    /// * `max_seq_len` — maximum context length (e.g., 2048)
    /// * `n_kv_heads` — number of key/value attention heads (e.g., 3)
    /// * `head_dim` — dimension per head (e.g., 64)
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

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if no positions have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write K and V vectors for one position in one layer.
    ///
    /// Called once per layer per position. After all layers have written
    /// the same position, call `advance()` to bump the length counter.
    pub fn write(&mut self, layer: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        assert!(
            pos < self.max_seq_len,
            "KV-cache overflow: position {pos} >= max_seq_len {}",
            self.max_seq_len
        );
        assert_eq!(k_vec.len(), self.kv_dim, "K vector length mismatch");
        assert_eq!(v_vec.len(), self.kv_dim, "V vector length mismatch");

        let offset = pos * self.kv_dim;
        self.k[layer][offset..offset + self.kv_dim].copy_from_slice(k_vec);
        self.v[layer][offset..offset + self.kv_dim].copy_from_slice(v_vec);
    }

    /// Advance the cached length by one position.
    ///
    /// Call this once after all layers have written position `self.len`.
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Read the K vector for a given layer and position.
    ///
    /// Returns a slice of length `kv_dim`.
    pub fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.k[layer][offset..offset + self.kv_dim]
    }

    /// Read the V vector for a given layer and position.
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.v[layer][offset..offset + self.kv_dim]
    }

    /// Reset the cache for a new sequence.
    ///
    /// No need to zero the buffers — we only ever read up to `self.len`.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_write_read() {
        let mut cache = KvCache::new(2, 8, 1, 4); // 2 layers, max 8 pos, 1 head, dim 4
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
        let mut cache = KvCache::new(1, 8, 2, 3); // 1 layer, kv_dim = 2*3 = 6

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
        let mut cache = KvCache::new(3, 8, 1, 2); // 3 layers, kv_dim = 2

        // Write position 0 across all layers
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
}
