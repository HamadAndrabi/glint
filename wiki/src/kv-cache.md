# KV Cache

The KV cache stores previously computed Key and Value vectors so they don't need to be recomputed at each generation step. It's one of the most important optimizations in LLM inference.

Source: `src/cache/mod.rs`

---

## Why a KV Cache?

In autoregressive generation, at each step `t` we need attention over all positions `[0..t]`. Without a cache:

```
Step 1: compute K, V for tokens [0]         → 1 matvec
Step 2: compute K, V for tokens [0, 1]      → 2 matvecs (re-computing position 0!)
Step 3: compute K, V for tokens [0, 1, 2]   → 3 matvecs
...
Step N: compute K, V for tokens [0..N]      → N matvecs

Total: N(N+1)/2 matvecs  →  O(N²)
```

With a KV cache, each step computes K, V only for the new token:

```
Step 1: compute K, V for token [0]    → write to cache
Step 2: compute K, V for token [1]    → write to cache; read [0] from cache
...
Step N: compute K, V for token [N]    → write to cache; read [0..N-1] from cache

Total: N matvecs  →  O(N)
```

For a 100-token generation, the cache reduces attention matvecs from ~5000 to 100.

---

## The `KvStore` Trait

All cache implementations share a common interface:

```rust
pub trait KvStore: Send + Sync {
    fn read_k_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]);
    fn read_v_head(&self, layer: usize, pos: usize, kv_h: usize, head_dim: usize, buf: &mut [f32]);
    fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]);
    fn reserve(&mut self, total_positions: usize) -> Result<(), GlintError>;  // capacity, up front
    fn advance(&mut self);          // called once per token, after writing all layers
    fn len(&self) -> usize;         // number of valid positions
    fn truncate(&mut self, new_len: usize);  // roll back (used by speculative decoding)
    fn clear(&mut self);            // reset for new sequence
}
```

The flash attention implementation and the forward pass are both written against `KvStore`, making them format-agnostic.

`reserve` defaults to a no-op: `KvCache` and `KvCacheQ8` own every slot from the moment they are constructed. Only `PagedKvCache` does real work there, which is how a caller finds out it is out of KV memory *before* a forward pass rather than during one.

---

## `KvCache` (f32)

Pre-allocated f32 storage. Simple and exact.

```rust
pub struct KvCache {
    k: Vec<Vec<f32>>,   // k[layer][pos * kv_dim .. (pos+1) * kv_dim]
    v: Vec<Vec<f32>>,
    kv_dim: usize,      // n_kv_heads * head_dim
    max_seq_len: usize,
    len: usize,
}
```

Memory usage:
```
2 (K+V) × n_layers × max_seq_len × n_kv_heads × head_dim × 4 bytes

Example: LLaMA-3 8B, 8192 context
  2 × 32 × 8192 × 8 × 128 × 4 = 1.7 GB
```

---

## `KvCacheQ8` (Q8_0 compressed)

Stores KV vectors in Q8_0 format, achieving ~3.76× compression versus f32.

```rust
pub struct KvCacheQ8 {
    k: Vec<Vec<u8>>,   // k[layer][raw Q8_0 bytes]
    v: Vec<Vec<u8>>,
    kv_dim: usize,
    bytes_per_pos: usize,  // = ceil(kv_dim / 32) * 34
    ...
}
```

### Q8_0 Block Layout (in cache)

Each block of 32 f32 values is compressed to 34 bytes:
- 2 bytes: f16 scale `d`
- 32 bytes: `i8` quantized values

**Write path (quantize):**
```
max_abs = max(|x[i]|)
d = max_abs / 127
q[i] = round(x[i] / d)
```

**Read path (dequantize):**
```
x[i] = q[i] * d
```

Dequantization happens per-head during the attention loop — only the specific head dimension needed at each attention step is dequantized, keeping the hot path allocation-free.

### Memory usage comparison

```
LLaMA-3 8B, 8192 context:
  f32:   1.7 GB
  Q8_0:  0.46 GB  (3.76× smaller)
```

### When to use Q8_0 cache

Use `KvCacheQ8` when:
- Running large models (7B+) with long contexts
- Memory is the primary constraint
- You can tolerate a tiny quality difference (typically < 0.1% perplexity increase)

The rounding error introduced per token is small (max ~0.1% relative error per element), and errors don't accumulate because each token's K/V is written and read independently.

---

## `PagedKvCache` (paged f32)

Source: `src/cache/paged.rs`

The same f32 rows as `KvCache`, but handed out in fixed-size **pages** of `PAGE_SIZE` (16) positions drawn from a shared, capacity-bounded `PagePool` — the PagedAttention idea from [vLLM](https://arxiv.org/abs/2309.05657).

```rust
pub struct PagedKvCache {
    pool: PagePool,               // shared handle: free list + capacity + refcounts
    table: Vec<Vec<Arc<PageBuf>>>, // table[layer][page] — the page table
    kv_dim: usize,
    len: usize,
}
```

One page holds `PAGE_SIZE` positions of K and V for a **single layer**, laid out exactly like a `KvCache` layer (`row = slot * kv_dim`), so reads are the same arithmetic behind one page indirection and attention output is bit-identical to `KvCache`.

### Why it matters

`KvCache` reserves `context_length` rows per sequence at admission. With 100 concurrent requests against a 4096-token context that is 100 full-context allocations, even if most requests stop after 50 tokens. A paged sequence only holds `ceil(len / 16)` pages per layer, and its pages return to the pool when it finishes.

### Allocation and hot path

- A page is allocated only when a write crosses a page boundary — never per token.
- Reads and in-page writes touch the sequence's own page table: no locking, no allocation.
- The pool lock is taken once per boundary crossing, per copy-on-write, and on release.

### Sharing and copy-on-write

Pages are reference-counted, which is the primitive prefix caching needs:

```rust
let child = parent.fork_from(prompt_len);  // shares pages, copies nothing
```

The fork's page table points at the parent's pages. A page reachable from more than one cache is immutable: the first write into one (typically the partially-filled tail page the fork inherited) copies it into a fresh page owned by the writer, so neither sequence can observe the other's tokens. Whole prefix pages, which neither side writes, stay shared for their lifetime.

### Running out of pages

The pool is bounded, so allocation can fail. `reserve` reports `GlintError::KvPagePoolExhausted` and the caller decides what to do; the inference engine ends that one sequence with `Finish::Length` and keeps serving everything else.

### Using it

```bash
glint serve model.gguf --kv-cache paged
```

or, in library code, `SessionOptions::page_pool` / `EngineLimits::kv_pool_pages` / `generate_cached_paged`.

Paged storage is f32-only today — a `CacheFormat::Q8` session still gets the contiguous `KvCacheQ8`.

---

## Speculative Decoding and Cache Rollback

The `truncate(new_len)` method enables speculative decoding to roll back the cache when draft tokens are rejected:

```rust
// After generating k draft tokens speculatively
draft_cache.truncate(current_len);   // discard speculated positions
target_cache.truncate(current_len);  // same for target cache
// Re-prefill with the actually accepted tokens
```

This is why the cache supports truncation: it's not just for clearing at conversation boundaries — it's used as a rollback mechanism mid-generation.

---

## Context Window Management

`KvCache` and `KvCacheQ8` are pre-allocated for `max_seq_len` positions (from `config.context_length`). Attempts to write beyond this raise a panic:

```rust
assert!(pos < self.max_seq_len, "KV-cache overflow: pos {pos} >= {}", self.max_seq_len);
```

The chat mode in `main.rs` handles this by summarizing old messages when the context window fills up, then truncating the history.
