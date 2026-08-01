//! Paged KV-cache — PagedAttention-style block storage (Kwon et al., 2023).
//!
//! [`KvCache`](super::KvCache) reserves `max_seq_len` rows per sequence up
//! front: with 100 concurrent requests and a 4096-token context that is 100
//! full-context allocations even when most requests stop after 50 tokens. A
//! paged cache instead hands each sequence fixed-size **pages** of
//! [`PAGE_SIZE`] positions, drawn on demand from a shared [`PagePool`], so a
//! sequence's footprint tracks the tokens it actually produced.
//!
//! # Page layout
//!
//! One page holds `PAGE_SIZE` positions of K **and** V for a *single layer*.
//! That mirrors `KvCache`'s per-layer storage exactly — row `slot` of a page
//! starts at `slot * kv_dim`, just as row `pos` of a `KvCache` layer starts at
//! `pos * kv_dim` — so the read path is the same indexing arithmetic with a
//! page indirection in front of it. A sequence owns one page table per layer
//! (`table[layer][page]`); pages carry no layer identity, so the pool can hand
//! any free page to any layer of any sequence.
//!
//! # Sharing and copy-on-write
//!
//! Pages are reference-counted. [`PagedKvCache::fork_from`] clones the page
//! *pointers* of a common prefix rather than its data — the primitive prefix
//! caching needs. A shared page is immutable: the first write into one
//! (typically the partially-filled tail page a fork inherited) copies it into a
//! fresh page owned by the writer, so neither sequence can observe the other's
//! tokens.
//!
//! # Allocation discipline
//!
//! The pool lock is taken only when pages are acquired or returned — once per
//! `PAGE_SIZE` tokens per layer, plus once per copy-on-write. Reads and
//! in-page writes go through the sequence's own page table with no locking and
//! no allocation, so the per-token hot path is unchanged from `KvCache` apart
//! from one extra index and one refcount load.
//!
//! # Running out of pages
//!
//! The pool is capacity-bounded, so allocation can fail. Failure is reported
//! as [`GlintError::KvPagePoolExhausted`] from [`KvStore::reserve`], which
//! callers run *before* a forward pass (the engine ends the sequence cleanly
//! instead of dying inside a decode step). `write` still grows the cache
//! lazily for callers that skip `reserve`, and panics if the pool is empty at
//! that point — the same contract as `KvCache::write` past `max_seq_len`.

use std::sync::{Arc, Mutex, MutexGuard};

use super::KvStore;
use crate::error::GlintError;

/// Positions stored in one page.
///
/// 16 is the vLLM default: small enough that a short sequence wastes little,
/// large enough that page bookkeeping stays off the per-token path.
pub const PAGE_SIZE: usize = 16;

// ── Page storage ─────────────────────────────────────────────────────────────

/// `PAGE_SIZE` positions of K and V for one layer.
///
/// Row `slot` occupies `k[slot * kv_dim .. (slot + 1) * kv_dim]` (and likewise
/// in `v`) — the same row layout `KvCache` uses.
#[derive(Clone)]
struct PageBuf {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl PageBuf {
    fn zeroed(kv_dim: usize) -> Self {
        Self {
            k: vec![0.0f32; PAGE_SIZE * kv_dim],
            v: vec![0.0f32; PAGE_SIZE * kv_dim],
        }
    }

    /// Wipe a recycled page so one sequence can never read another's tokens
    /// out of a slot it has not written yet.
    fn reset(&mut self) {
        self.k.fill(0.0);
        self.v.fill(0.0);
    }
}

// ── PagePool ─────────────────────────────────────────────────────────────────

/// Occupancy snapshot for a [`PagePool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolStats {
    /// Maximum pages the pool will ever hand out at once.
    pub capacity: usize,
    /// Pages currently checked out by sequences.
    pub live: usize,
    /// High-water mark of `live` since the pool was created.
    pub peak_live: usize,
    /// Freed page buffers kept for reuse (allocated, not checked out).
    pub pooled: usize,
}

struct PoolInner {
    kv_dim: usize,
    capacity: usize,
    live: usize,
    peak_live: usize,
    /// Buffers returned by finished sequences, ready to be handed out again.
    free: Vec<PageBuf>,
}

impl PoolInner {
    /// Fail unless `n` more pages can be handed out.
    ///
    /// Callers check once and then acquire, so a request that cannot be
    /// satisfied in full leaves no half-grown page tables behind.
    fn check_available(&self, n: usize) -> Result<(), GlintError> {
        if self.live + n > self.capacity {
            return Err(GlintError::KvPagePoolExhausted {
                needed: n,
                available: self.capacity - self.live,
                capacity: self.capacity,
            });
        }
        Ok(())
    }

    /// Hand out one page. Only call after a successful [`Self::check_available`].
    fn acquire(&mut self) -> PageBuf {
        debug_assert!(self.live < self.capacity, "acquire past pool capacity");
        let page = match self.free.pop() {
            Some(mut recycled) => {
                recycled.reset();
                recycled
            }
            // Buffers are created on first use rather than up front, so the
            // process's RSS follows actual KV usage; `capacity` is the cap.
            None => PageBuf::zeroed(self.kv_dim),
        };
        self.live += 1;
        self.peak_live = self.peak_live.max(self.live);
        page
    }

    /// Drop one reference to `page`, recycling the buffer if it was the last.
    fn release(&mut self, page: Arc<PageBuf>) {
        // `try_unwrap` succeeds for exactly one holder, so a page shared by
        // several sequences is counted down once — never twice, never leaked.
        if let Ok(buf) = Arc::try_unwrap(page) {
            self.live -= 1;
            self.free.push(buf);
        }
    }
}

/// Shared, capacity-bounded pool of KV pages.
///
/// Cheap to clone (it is a handle): every clone refers to the same pool, which
/// is what lets sequences share pages and hand freed pages back to each other.
#[derive(Clone)]
pub struct PagePool {
    inner: Arc<Mutex<PoolInner>>,
}

impl PagePool {
    /// Create a pool that will hand out at most `capacity_pages` pages.
    ///
    /// Page buffers are allocated lazily on first use and recycled thereafter,
    /// so memory grows with real usage and never past `capacity_pages` pages.
    pub fn new(capacity_pages: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                kv_dim: n_kv_heads * head_dim,
                capacity: capacity_pages,
                live: 0,
                peak_live: 0,
                free: Vec::new(),
            })),
        }
    }

    /// Like [`Self::new`], but allocates every page buffer immediately.
    ///
    /// Use when steady-state latency matters more than start-up cost or peak
    /// RSS — no sequence then pays for a page allocation mid-decode.
    pub fn preallocated(capacity_pages: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let pool = Self::new(capacity_pages, n_kv_heads, head_dim);
        {
            let mut inner = pool.lock();
            let kv_dim = inner.kv_dim;
            inner.free = (0..capacity_pages)
                .map(|_| PageBuf::zeroed(kv_dim))
                .collect();
        }
        pool
    }

    /// Pages one sequence of `positions` tokens needs across `n_layers` layers.
    pub const fn pages_for(positions: usize, n_layers: usize) -> usize {
        positions.div_ceil(PAGE_SIZE) * n_layers
    }

    /// KV row width (`n_kv_heads * head_dim`) every page in this pool stores.
    pub fn kv_dim(&self) -> usize {
        self.lock().kv_dim
    }

    /// Maximum pages this pool hands out at once.
    pub fn capacity(&self) -> usize {
        self.lock().capacity
    }

    /// Pages currently checked out by sequences.
    pub fn live_pages(&self) -> usize {
        self.lock().live
    }

    /// Pages that can still be handed out.
    pub fn available_pages(&self) -> usize {
        let inner = self.lock();
        inner.capacity - inner.live
    }

    /// Full occupancy snapshot (useful for `/v1/metrics`-style reporting).
    pub fn stats(&self) -> PoolStats {
        let inner = self.lock();
        PoolStats {
            capacity: inner.capacity,
            live: inner.live,
            peak_live: inner.peak_live,
            pooled: inner.free.len(),
        }
    }

    /// Lock the pool, recovering from poisoning.
    ///
    /// The guarded state is a free list and three counters; if a thread
    /// panicked while holding the lock the worst case is a slightly stale
    /// count, which is not worth failing every other live sequence over.
    fn lock(&self) -> MutexGuard<'_, PoolInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

// ── PagedKvCache ─────────────────────────────────────────────────────────────

/// Per-sequence KV cache backed by pages from a shared [`PagePool`].
///
/// Implements [`KvStore`] with f32 storage identical to `KvCache`'s, so flash
/// attention and the forward pass produce bit-identical results with either.
///
/// Pages are returned to the pool when the cache is dropped (or via
/// [`release`](Self::release)); pages shared with a fork survive until the last
/// holder goes away.
pub struct PagedKvCache {
    pool: PagePool,
    /// `table[layer][page]` — page tables, one per layer.
    table: Vec<Vec<Arc<PageBuf>>>,
    kv_dim: usize,
    len: usize,
}

impl PagedKvCache {
    /// Create an empty cache for `n_layers` layers. No pages are allocated
    /// until the first write or [`KvStore::reserve`].
    pub fn new(pool: &PagePool, n_layers: usize) -> Self {
        Self {
            pool: pool.clone(),
            table: (0..n_layers).map(|_| Vec::new()).collect(),
            kv_dim: pool.kv_dim(),
            len: 0,
        }
    }

    /// Fork a new sequence that shares this one's pages for positions
    /// `0..upto_position`.
    ///
    /// No data is copied and no page is allocated: the child's page table
    /// points at the same pages and their reference counts go up. The child
    /// starts with `len() == upto_position`, so its next append continues from
    /// there.
    ///
    /// # Copy-on-write invariant
    ///
    /// A page reachable from more than one cache is never written in place.
    /// When either sequence writes into a shared page — which happens as soon
    /// as the child appends into the partially-filled tail page it inherited —
    /// the writer copies that page into a fresh one first. Whole pages of the
    /// prefix, which neither side writes, stay shared for the rest of their
    /// lives.
    ///
    /// Panics if `upto_position` exceeds this cache's length (the prefix must
    /// actually be present), mirroring `KvStore::truncate`'s bounds check.
    pub fn fork_from(&self, upto_position: usize) -> PagedKvCache {
        assert!(
            upto_position <= self.len,
            "fork_from: {upto_position} > len {}",
            self.len
        );
        let pages = upto_position.div_ceil(PAGE_SIZE);
        PagedKvCache {
            pool: self.pool.clone(),
            table: self
                .table
                .iter()
                .map(|layer| layer[..pages].to_vec())
                .collect(),
            kv_dim: self.kv_dim,
            len: upto_position,
        }
    }

    /// Return every page to the pool and reset the cache to zero positions.
    ///
    /// Called automatically on drop; exposed so a serving loop can reclaim
    /// pages at a point of its choosing.
    pub fn release(&mut self) {
        let mut pool = self.pool.lock();
        for layer in &mut self.table {
            for page in layer.drain(..) {
                pool.release(page);
            }
        }
        self.len = 0;
    }

    /// The pool this cache draws from.
    pub fn pool(&self) -> &PagePool {
        &self.pool
    }

    /// Pages held per layer — `ceil(reserved_positions / PAGE_SIZE)`.
    pub fn pages_per_layer(&self) -> usize {
        self.table.first().map_or(0, Vec::len)
    }

    /// Pages held across all layers.
    pub fn allocated_pages(&self) -> usize {
        self.table.iter().map(Vec::len).sum()
    }

    /// How many sequences share one page (1 = exclusively owned).
    pub fn page_refcount(&self, layer: usize, page: usize) -> usize {
        Arc::strong_count(&self.table[layer][page])
    }

    /// True if another cache also holds `page` of `layer`.
    pub fn is_page_shared(&self, layer: usize, page: usize) -> bool {
        self.page_refcount(layer, page) > 1
    }

    /// Full `kv_dim`-length K row for `pos` — the paged counterpart of
    /// `KvCache::k_at`, used by tests and reference implementations.
    pub fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = (pos % PAGE_SIZE) * self.kv_dim;
        &self.table[layer][pos / PAGE_SIZE].k[offset..offset + self.kv_dim]
    }

    /// Full `kv_dim`-length V row for `pos`.
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let offset = (pos % PAGE_SIZE) * self.kv_dim;
        &self.table[layer][pos / PAGE_SIZE].v[offset..offset + self.kv_dim]
    }

    /// Grow every layer to cover `total_positions` and un-share the pages the
    /// upcoming appends will write into.
    ///
    /// Counts first and allocates second, so an exhausted pool is reported
    /// before any page changes hands.
    fn ensure_capacity(&mut self, total_positions: usize) -> Result<(), GlintError> {
        let want = total_positions.div_ceil(PAGE_SIZE);
        // Positions `>= len` are the ones still to be written, so every page
        // from `len / PAGE_SIZE` on must end up exclusively owned.
        let first_writable = self.len / PAGE_SIZE;
        let kv_dim = self.kv_dim;

        let mut pool = self.pool.lock();
        let mut needed = 0;
        for layer in &self.table {
            needed += want.saturating_sub(layer.len());
            needed += layer
                .iter()
                .take(want)
                .skip(first_writable)
                .filter(|page| Arc::strong_count(page) > 1)
                .count();
        }
        if needed == 0 {
            return Ok(());
        }
        pool.check_available(needed)?;

        for layer in &mut self.table {
            while layer.len() < want {
                layer.push(Arc::new(pool.acquire()));
            }
            for idx in first_writable..want {
                if Arc::get_mut(&mut layer[idx]).is_none() {
                    copy_on_write(&mut pool, &mut layer[idx], kv_dim);
                }
            }
        }
        Ok(())
    }

    /// Make `page` of `layer` exclusively owned and present, allocating or
    /// copying if needed. The cold half of [`KvStore::write`].
    fn make_writable(&mut self, layer: usize, page: usize) -> Result<(), GlintError> {
        let kv_dim = self.kv_dim;
        let table = &mut self.table[layer];

        let mut pool = self.pool.lock();
        let mut needed = (page + 1).saturating_sub(table.len());
        if page < table.len() && Arc::strong_count(&table[page]) > 1 {
            needed += 1;
        }
        pool.check_available(needed)?;

        while table.len() <= page {
            table.push(Arc::new(pool.acquire()));
        }
        if Arc::get_mut(&mut table[page]).is_none() {
            copy_on_write(&mut pool, &mut table[page], kv_dim);
        }
        Ok(())
    }
}

/// Replace a shared page with a private copy of its contents.
///
/// Upholds the copy-on-write invariant: after this returns, `slot` holds a page
/// no other cache can see, and the previous page's refcount has been decremented
/// (it stays alive for whoever else still points at it).
fn copy_on_write(pool: &mut PoolInner, slot: &mut Arc<PageBuf>, kv_dim: usize) {
    debug_assert!(
        Arc::strong_count(slot) > 1,
        "copy-on-write of a private page"
    );
    let mut fresh = pool.acquire();
    debug_assert_eq!(fresh.k.len(), PAGE_SIZE * kv_dim);
    fresh.k.copy_from_slice(&slot.k);
    fresh.v.copy_from_slice(&slot.v);
    let previous = std::mem::replace(slot, Arc::new(fresh));
    pool.release(previous);
}

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        self.release();
    }
}

impl KvStore for PagedKvCache {
    fn read_k_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = (pos % PAGE_SIZE) * self.kv_dim + kv_h * head_dim;
        let page = &self.table[layer][pos / PAGE_SIZE];
        buf.copy_from_slice(&page.k[start..start + head_dim]);
    }

    fn read_v_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), head_dim);
        let start = (pos % PAGE_SIZE) * self.kv_dim + kv_h * head_dim;
        let page = &self.table[layer][pos / PAGE_SIZE];
        buf.copy_from_slice(&page.v[start..start + head_dim]);
    }

    fn write(&mut self, layer: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        debug_assert_eq!(k_vec.len(), self.kv_dim);
        debug_assert_eq!(v_vec.len(), self.kv_dim);
        let page_idx = pos / PAGE_SIZE;
        let kv_dim = self.kv_dim;

        // Hot path: the page exists and nothing else shares it — no lock, no
        // allocation, just the two row copies below. The cold branch runs once
        // per PAGE_SIZE tokens (a boundary crossing) or once per shared page.
        let private = self.table[layer]
            .get(page_idx)
            .is_some_and(|page| Arc::strong_count(page) == 1);
        if !private {
            self.make_writable(layer, page_idx).unwrap_or_else(|e| {
                panic!("paged KV-cache write at pos {pos}: {e} — call KvStore::reserve() before the forward pass to handle exhaustion")
            });
        }

        let offset = (pos % PAGE_SIZE) * kv_dim;
        let page = Arc::get_mut(&mut self.table[layer][page_idx])
            .expect("page is exclusively owned after make_writable");
        page.k[offset..offset + kv_dim].copy_from_slice(k_vec);
        page.v[offset..offset + kv_dim].copy_from_slice(v_vec);
    }

    fn reserve(&mut self, total_positions: usize) -> Result<(), GlintError> {
        self.ensure_capacity(total_positions)
    }

    fn as_paged(&self) -> Option<&PagedKvCache> {
        Some(self)
    }

    fn advance(&mut self) {
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Drop back to `new_len` valid positions.
    ///
    /// Pages are deliberately kept: the position at `new_len` is usually
    /// rewritten immediately (speculative rollback, snapshot restore), and
    /// re-acquiring a page there could fail against a full pool. Use
    /// [`PagedKvCache::release`] or drop the cache to return pages.
    fn truncate(&mut self, new_len: usize) {
        assert!(
            new_len <= self.len,
            "truncate: {new_len} > len {}",
            self.len
        );
        self.len = new_len;
    }

    fn export_raw(&self) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let row_bytes = self.kv_dim * std::mem::size_of::<f32>();
        let mut k_out = Vec::with_capacity(self.table.len());
        let mut v_out = Vec::with_capacity(self.table.len());
        for layer in 0..self.table.len() {
            let mut k_bytes = Vec::with_capacity(self.len * row_bytes);
            let mut v_bytes = Vec::with_capacity(self.len * row_bytes);
            for pos in 0..self.len {
                // Native-endian f32 bytes — the format `KvCache` exports.
                for &x in self.k_at(layer, pos) {
                    k_bytes.extend_from_slice(&x.to_ne_bytes());
                }
                for &x in self.v_at(layer, pos) {
                    v_bytes.extend_from_slice(&x.to_ne_bytes());
                }
            }
            k_out.push(k_bytes);
            v_out.push(v_bytes);
        }
        (k_out, v_out)
    }

    fn import_raw(
        &mut self,
        k_layers: &[Vec<u8>],
        v_layers: &[Vec<u8>],
        token_count: usize,
    ) -> Result<(), GlintError> {
        let expected_bytes = token_count * self.kv_dim * std::mem::size_of::<f32>();
        for (l, (k_src, v_src)) in k_layers.iter().zip(v_layers.iter()).enumerate() {
            if k_src.len() != expected_bytes {
                return Err(GlintError::SnapshotCacheSizeMismatch {
                    layer: l,
                    expected: expected_bytes,
                    found: k_src.len(),
                });
            }
            if v_src.len() != expected_bytes {
                return Err(GlintError::SnapshotCacheSizeMismatch {
                    layer: l,
                    expected: expected_bytes,
                    found: v_src.len(),
                });
            }
        }
        // Pages for the whole snapshot, up front: a pool that cannot hold it
        // fails here rather than part-way through the scatter below.
        self.len = 0;
        self.ensure_capacity(token_count)?;

        let kv_dim = self.kv_dim;
        let mut row = vec![0.0f32; kv_dim];
        for (l, (k_src, v_src)) in k_layers.iter().zip(v_layers.iter()).enumerate() {
            for pos in 0..token_count {
                let start = pos * kv_dim * std::mem::size_of::<f32>();
                decode_row(&k_src[start..], &mut row);
                let offset = (pos % PAGE_SIZE) * kv_dim;
                let page = Arc::get_mut(&mut self.table[l][pos / PAGE_SIZE])
                    .expect("import target pages are exclusively owned");
                page.k[offset..offset + kv_dim].copy_from_slice(&row);
                decode_row(&v_src[start..], &mut row);
                page.v[offset..offset + kv_dim].copy_from_slice(&row);
            }
        }
        self.len = token_count;
        Ok(())
    }
}

/// Decode `row.len()` native-endian f32s from the front of `src`.
fn decode_row(src: &[u8], row: &mut [f32]) {
    for (i, out) in row.iter_mut().enumerate() {
        let b = &src[i * 4..i * 4 + 4];
        *out = f32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::tensor::flash_attn_1d;

    const N_LAYERS: usize = 2;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

    fn pool(capacity_pages: usize) -> PagePool {
        PagePool::new(capacity_pages, N_KV_HEADS, HEAD_DIM)
    }

    /// Deterministic, distinct K/V rows so a mixed-up page shows up immediately.
    fn row(tag: f32, pos: usize) -> Vec<f32> {
        (0..KV_DIM)
            .map(|d| tag + pos as f32 * 0.25 + d as f32 * 0.0625)
            .collect()
    }

    fn append(cache: &mut dyn KvStore, positions: std::ops::Range<usize>) {
        for pos in positions {
            for layer in 0..N_LAYERS {
                let k = row(layer as f32, pos);
                let v = row(100.0 + layer as f32, pos);
                cache.write(layer, pos, &k, &v);
            }
            cache.advance();
        }
    }

    // ── Equivalence with the contiguous f32 cache ────────────────────────

    /// Bitwise equality on real hardware. Under Miri, transcendental
    /// functions (the `exp` in flash attention's online softmax) are
    /// deliberately non-deterministic between calls, so two runs of the same
    /// computation drift by a few ULP there — same as the RoPE sin/cos case
    /// in `ops::tests::test_rope_scaling_factor`. A tight tolerance still
    /// fails loudly for a real paging bug (a wrong row or page moves the
    /// output by whole units, not 1e-5).
    fn assert_flash_eq(want: &[f32], got: &[f32], ctx: &str) {
        for (d, (w, g)) in want.iter().zip(got).enumerate() {
            if cfg!(miri) {
                assert!(
                    (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                    "{ctx} dim {d}: {w} vs {g}"
                );
            } else {
                assert_eq!(w.to_bits(), g.to_bits(), "{ctx} dim {d}: {w} vs {g}");
            }
        }
    }

    /// The paged cache must be indistinguishable from `KvCache` through the
    /// flash-attention path — same f32 values, same order, so bit-identical.
    #[test]
    fn test_paged_flash_output_matches_kv_cache_bitwise() {
        let seq_len = 37; // spans 3 pages, last one partial
        let mut plain = KvCache::new(N_LAYERS, 64, N_KV_HEADS, HEAD_DIM);
        let mut paged = PagedKvCache::new(&pool(64), N_LAYERS);
        append(&mut plain, 0..seq_len);
        append(&mut paged, 0..seq_len);
        assert_eq!(plain.len(), paged.len());

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let q: Vec<f32> = (0..HEAD_DIM).map(|d| d as f32 * 0.3 - 0.4).collect();
        for layer in 0..N_LAYERS {
            for kv_h in 0..N_KV_HEADS {
                let mut want = vec![0.0f32; HEAD_DIM];
                let mut got = vec![0.0f32; HEAD_DIM];
                flash_attn_1d(
                    &q, &plain, layer, kv_h, 0, seq_len, HEAD_DIM, scale, &mut want,
                );
                flash_attn_1d(
                    &q, &paged, layer, kv_h, 0, seq_len, HEAD_DIM, scale, &mut got,
                );
                assert_flash_eq(&want, &got, &format!("layer {layer} head {kv_h}"));
            }
        }
    }

    /// A windowed read (sliding-window attention) crosses page boundaries at
    /// an offset — the indexing must still line up with the contiguous cache.
    #[test]
    fn test_paged_flash_output_matches_with_window_offset() {
        let seq_len = 40;
        let mut plain = KvCache::new(1, 64, N_KV_HEADS, HEAD_DIM);
        let mut paged = PagedKvCache::new(&pool(64), 1);
        for pos in 0..seq_len {
            let (k, v) = (row(0.0, pos), row(9.0, pos));
            plain.write(0, pos, &k, &v);
            plain.advance();
            paged.write(0, pos, &k, &v);
            paged.advance();
        }
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let q: Vec<f32> = (0..HEAD_DIM).map(|d| 0.1 * d as f32).collect();
        let (start, span) = (13, 21); // starts mid-page, ends mid-page
        let mut want = vec![0.0f32; HEAD_DIM];
        let mut got = vec![0.0f32; HEAD_DIM];
        flash_attn_1d(&q, &plain, 0, 1, start, span, HEAD_DIM, scale, &mut want);
        flash_attn_1d(&q, &paged, 0, 1, start, span, HEAD_DIM, scale, &mut got);
        assert_flash_eq(&want, &got, "windowed read");
    }

    #[test]
    fn test_paged_rows_match_kv_cache_rows() {
        let mut plain = KvCache::new(N_LAYERS, 64, N_KV_HEADS, HEAD_DIM);
        let mut paged = PagedKvCache::new(&pool(64), N_LAYERS);
        append(&mut plain, 0..35);
        append(&mut paged, 0..35);
        for layer in 0..N_LAYERS {
            for pos in 0..35 {
                assert_eq!(plain.k_at(layer, pos), paged.k_at(layer, pos), "K {pos}");
                assert_eq!(plain.v_at(layer, pos), paged.v_at(layer, pos), "V {pos}");
            }
        }
    }

    #[test]
    fn test_export_matches_kv_cache_bytes() {
        let mut plain = KvCache::new(N_LAYERS, 64, N_KV_HEADS, HEAD_DIM);
        let mut paged = PagedKvCache::new(&pool(64), N_LAYERS);
        append(&mut plain, 0..20);
        append(&mut paged, 0..20);
        assert_eq!(plain.export_raw(), paged.export_raw());
    }

    #[test]
    fn test_export_import_roundtrip() {
        let shared = pool(64);
        let mut src = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut src, 0..20);
        let (k, v) = src.export_raw();

        let mut dst = PagedKvCache::new(&shared, N_LAYERS);
        dst.import_raw(&k, &v, 20).expect("pool has room");
        assert_eq!(dst.len(), 20);
        for layer in 0..N_LAYERS {
            for pos in 0..20 {
                assert_eq!(src.k_at(layer, pos), dst.k_at(layer, pos));
                assert_eq!(src.v_at(layer, pos), dst.v_at(layer, pos));
            }
        }
    }

    // ── Page accounting ──────────────────────────────────────────────────

    /// Pages appear only when a write crosses a page boundary — never
    /// per token.
    #[test]
    fn test_pages_allocated_only_on_boundary_crossing() {
        let shared = pool(64);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        assert_eq!(shared.live_pages(), 0, "empty cache holds no pages");

        for pos in 0..(2 * PAGE_SIZE + 1) {
            let before = shared.live_pages();
            append(&mut cache, pos..pos + 1);
            let after = shared.live_pages();
            let expected_new = if pos % PAGE_SIZE == 0 { N_LAYERS } else { 0 };
            assert_eq!(
                after - before,
                expected_new,
                "pos {pos}: pages went {before} -> {after}"
            );
        }
        assert_eq!(cache.pages_per_layer(), 3);
        assert_eq!(shared.live_pages(), 3 * N_LAYERS);
    }

    #[test]
    fn test_reserve_allocates_up_front_and_write_is_free() {
        let shared = pool(64);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        cache.reserve(3 * PAGE_SIZE).expect("pool has room");
        assert_eq!(shared.live_pages(), 3 * N_LAYERS);

        let before = shared.live_pages();
        append(&mut cache, 0..3 * PAGE_SIZE);
        assert_eq!(
            shared.live_pages(),
            before,
            "writes into reserved pages must not allocate"
        );
    }

    #[test]
    fn test_release_returns_pages_to_pool() {
        let shared = pool(64);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut cache, 0..33);
        let held = shared.live_pages();
        assert_eq!(held, 3 * N_LAYERS);

        cache.release();
        assert_eq!(shared.live_pages(), 0, "release returns every page");
        assert_eq!(shared.stats().pooled, held, "buffers are kept for reuse");
        assert_eq!(cache.len(), 0);

        // Recycled buffers are reused rather than re-allocated.
        append(&mut cache, 0..PAGE_SIZE);
        assert_eq!(shared.stats().pooled, held - N_LAYERS);
    }

    #[test]
    fn test_drop_returns_pages_to_pool() {
        let shared = pool(64);
        {
            let mut cache = PagedKvCache::new(&shared, N_LAYERS);
            append(&mut cache, 0..20);
            assert_eq!(shared.live_pages(), 2 * N_LAYERS);
        }
        assert_eq!(
            shared.live_pages(),
            0,
            "dropping a sequence frees its pages"
        );
    }

    #[test]
    fn test_recycled_page_is_zeroed() {
        let shared = pool(N_LAYERS); // one page per layer, forces reuse
        {
            let mut first = PagedKvCache::new(&shared, N_LAYERS);
            append(&mut first, 0..PAGE_SIZE);
        }
        let mut second = PagedKvCache::new(&shared, N_LAYERS);
        second.reserve(PAGE_SIZE).expect("page was returned");
        // Only position 0 is written; the rest of the recycled page must not
        // still hold the previous sequence's tokens.
        append(&mut second, 0..1);
        assert!(
            second.k_at(0, PAGE_SIZE - 1).iter().all(|&x| x == 0.0),
            "recycled page leaked data"
        );
    }

    #[test]
    fn test_truncate_keeps_pages_and_allows_rewrite() {
        let shared = pool(64);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut cache, 0..33);
        let held = shared.live_pages();

        cache.truncate(16);
        assert_eq!(cache.len(), 16);
        assert_eq!(shared.live_pages(), held, "truncate keeps pages allocated");

        // Rewriting from the truncation point needs no new page.
        append(&mut cache, 16..17);
        assert_eq!(shared.live_pages(), held);
        assert_eq!(cache.len(), 17);
    }

    // ── Forking, refcounts and copy-on-write ─────────────────────────────

    #[test]
    fn test_fork_shares_pages_without_allocating() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..32);
        let live_before = shared.live_pages();

        let child = parent.fork_from(32);
        assert_eq!(child.len(), 32);
        assert_eq!(
            shared.live_pages(),
            live_before,
            "fork shares pages instead of allocating"
        );
        for layer in 0..N_LAYERS {
            for page in 0..2 {
                assert_eq!(parent.page_refcount(layer, page), 2);
                assert!(child.is_page_shared(layer, page));
            }
        }
        // The child sees the prefix the parent wrote.
        for pos in 0..32 {
            assert_eq!(parent.k_at(0, pos), child.k_at(0, pos));
        }
    }

    #[test]
    fn test_fork_drop_restores_refcount_and_frees_nothing() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..32);
        let live = shared.live_pages();
        {
            let child = parent.fork_from(32);
            assert_eq!(child.page_refcount(0, 0), 2);
        }
        assert_eq!(parent.page_refcount(0, 0), 1, "child's reference is gone");
        assert_eq!(
            shared.live_pages(),
            live,
            "the parent still owns every page"
        );
    }

    /// Writing into the shared tail page copies it: the child's new tokens are
    /// invisible to the parent, and the parent's rows are untouched.
    #[test]
    fn test_fork_cow_child_write_does_not_corrupt_parent() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..20); // 2 pages, second one partially filled
        let parent_rows: Vec<Vec<f32>> = (0..20).map(|p| parent.k_at(0, p).to_vec()).collect();

        let mut child = parent.fork_from(20);
        let live_before = shared.live_pages();
        assert!(child.is_page_shared(0, 1), "tail page starts out shared");

        // Child appends into the shared tail page → copy-on-write.
        for layer in 0..N_LAYERS {
            child.write(layer, 20, &row(7.0, 20), &row(8.0, 20));
        }
        child.advance();

        assert_eq!(
            shared.live_pages(),
            live_before + N_LAYERS,
            "one copied page per layer"
        );
        assert!(
            !child.is_page_shared(0, 1),
            "child's tail page is now private"
        );
        assert!(
            !parent.is_page_shared(0, 1),
            "parent's tail page is private again"
        );
        assert_eq!(
            parent.page_refcount(0, 0),
            2,
            "full prefix page stays shared"
        );

        // Parent is unchanged and still cannot see position 20.
        assert_eq!(parent.len(), 20);
        for (pos, want) in parent_rows.iter().enumerate() {
            assert_eq!(parent.k_at(0, pos), want.as_slice(), "parent row {pos}");
        }
        // Child kept the prefix it forked and has its own tail token.
        for (pos, want) in parent_rows.iter().enumerate() {
            assert_eq!(child.k_at(0, pos), want.as_slice(), "child prefix {pos}");
        }
        assert_eq!(child.k_at(0, 20), row(7.0, 20).as_slice());
    }

    /// The same protection in the other direction: the parent continuing its
    /// own generation must not overwrite what the child inherited.
    #[test]
    fn test_fork_cow_parent_write_does_not_corrupt_child() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..20);
        let child = parent.fork_from(20);
        let child_rows: Vec<Vec<f32>> = (0..20).map(|p| child.k_at(0, p).to_vec()).collect();

        append(&mut parent, 20..24); // parent writes into the shared tail page

        assert!(!parent.is_page_shared(0, 1));
        for (pos, want) in child_rows.iter().enumerate() {
            assert_eq!(child.k_at(0, pos), want.as_slice(), "child row {pos}");
        }
        assert_eq!(child.len(), 20, "child length is unaffected");
    }

    /// A fork that stops on a page boundary shares only whole pages, and the
    /// child's first append lands in a brand-new page — no copy needed.
    #[test]
    fn test_fork_on_page_boundary_needs_no_copy() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..PAGE_SIZE);
        let mut child = parent.fork_from(PAGE_SIZE);
        assert_eq!(child.pages_per_layer(), 1);

        let live_before = shared.live_pages();
        append(&mut child, PAGE_SIZE..PAGE_SIZE + 1);
        assert_eq!(
            shared.live_pages(),
            live_before + N_LAYERS,
            "one fresh page per layer, no copy-on-write"
        );
        assert!(
            child.is_page_shared(0, 0),
            "the whole prefix page stays shared"
        );
    }

    #[test]
    fn test_fork_partial_prefix() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..40);
        let child = parent.fork_from(18);
        assert_eq!(child.len(), 18);
        assert_eq!(child.pages_per_layer(), 2, "18 positions span two pages");
        for pos in 0..18 {
            assert_eq!(child.k_at(0, pos), parent.k_at(0, pos));
        }
        assert_eq!(parent.page_refcount(0, 2), 1, "later pages are not shared");
    }

    #[test]
    #[should_panic(expected = "fork_from")]
    fn test_fork_beyond_len_panics() {
        let shared = pool(64);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..4);
        let _ = parent.fork_from(5);
    }

    // ── Pool exhaustion ──────────────────────────────────────────────────

    /// Exhaustion is a recoverable error: the sequence that hit it is
    /// unharmed, other sequences keep their data, and freeing pages makes the
    /// pool usable again.
    #[test]
    fn test_pool_exhaustion_is_recoverable() {
        let shared = pool(2 * N_LAYERS); // exactly two pages per layer
        let mut first = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut first, 0..PAGE_SIZE);

        let mut second = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut second, 0..PAGE_SIZE);
        assert_eq!(shared.available_pages(), 0);

        // No pages left for a third page-worth of tokens.
        let err = second
            .reserve(PAGE_SIZE + 1)
            .expect_err("pool is exhausted");
        assert!(
            matches!(err, GlintError::KvPagePoolExhausted { .. }),
            "unexpected error: {err}"
        );

        // Both sequences are still intact and readable.
        assert_eq!(first.len(), PAGE_SIZE);
        assert_eq!(second.len(), PAGE_SIZE);
        for pos in 0..PAGE_SIZE {
            assert_eq!(first.k_at(0, pos), row(0.0, pos).as_slice());
            assert_eq!(second.k_at(1, pos), row(1.0, pos).as_slice());
        }
        // Nothing was handed out by the failed reserve.
        assert_eq!(shared.live_pages(), 2 * N_LAYERS);

        // Freeing the first sequence lets the second grow.
        drop(first);
        second.reserve(PAGE_SIZE + 1).expect("pages were returned");
        append(&mut second, PAGE_SIZE..PAGE_SIZE + 1);
        assert_eq!(second.len(), PAGE_SIZE + 1);
    }

    #[test]
    fn test_exhausted_reserve_reports_the_shortfall() {
        let shared = pool(N_LAYERS);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        cache.reserve(PAGE_SIZE).expect("first page fits");
        match cache.reserve(4 * PAGE_SIZE) {
            Err(GlintError::KvPagePoolExhausted {
                needed,
                available,
                capacity,
            }) => {
                assert_eq!(needed, 3 * N_LAYERS);
                assert_eq!(available, 0);
                assert_eq!(capacity, N_LAYERS);
            }
            other => panic!("expected exhaustion, got {other:?}"),
        }
    }

    /// Copy-on-write needs a page too — when the pool cannot provide one the
    /// forked sequence gets an error instead of silently sharing storage.
    #[test]
    fn test_cow_reports_exhaustion_instead_of_sharing() {
        let shared = pool(N_LAYERS);
        let mut parent = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut parent, 0..4);
        let mut child = parent.fork_from(4);
        assert_eq!(shared.available_pages(), 0);

        let err = child.reserve(5).expect_err("no page left to copy into");
        assert!(matches!(err, GlintError::KvPagePoolExhausted { .. }));
        // The shared page is untouched, so the parent's data is still correct.
        assert!(child.is_page_shared(0, 0));
        assert_eq!(parent.k_at(0, 3), row(0.0, 3).as_slice());
    }

    // ── Pool bookkeeping ─────────────────────────────────────────────────

    #[test]
    fn test_pages_for_and_stats() {
        assert_eq!(PagePool::pages_for(0, 4), 0);
        assert_eq!(PagePool::pages_for(1, 4), 4);
        assert_eq!(PagePool::pages_for(PAGE_SIZE, 4), 4);
        assert_eq!(PagePool::pages_for(PAGE_SIZE + 1, 4), 8);

        let shared = pool(10);
        assert_eq!(shared.kv_dim(), KV_DIM);
        assert_eq!(shared.capacity(), 10);
        {
            let mut cache = PagedKvCache::new(&shared, N_LAYERS);
            append(&mut cache, 0..PAGE_SIZE + 1);
            assert_eq!(cache.allocated_pages(), 2 * N_LAYERS);
        }
        let stats = shared.stats();
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.live, 0);
        assert_eq!(stats.peak_live, 2 * N_LAYERS);
        assert_eq!(stats.pooled, 2 * N_LAYERS);
    }

    #[test]
    fn test_preallocated_pool_hands_out_ready_buffers() {
        let shared = PagePool::preallocated(4, N_KV_HEADS, HEAD_DIM);
        assert_eq!(shared.stats().pooled, 4);
        let mut cache = PagedKvCache::new(&shared, N_LAYERS);
        append(&mut cache, 0..1);
        assert_eq!(shared.stats().pooled, 4 - N_LAYERS);
        assert_eq!(shared.live_pages(), N_LAYERS);
    }

    /// `reserve` is the pre-flight check; the fixed caches implement it as a
    /// no-op, so callers can run it unconditionally.
    #[test]
    fn test_fixed_caches_reserve_is_a_noop() {
        let mut plain = KvCache::new(1, 8, 1, 4);
        assert!(plain.reserve(1_000_000).is_ok());
    }
}
