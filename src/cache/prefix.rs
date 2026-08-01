//! Prefix cache — reuse the KV pages of a prompt prefix across requests.
//!
//! A serving workload repeats itself: a thousand chat requests carrying the
//! same 2 000-token system prompt each re-run the same 2 000 prefill positions
//! and get bit-identical K/V out of every one of them. [`PagedKvCache`] already
//! has the primitive that fixes this — [`fork_from`] hands a new sequence the
//! *pages* of a prefix another sequence computed, with no copying — so what is
//! left is the bookkeeping: recognising the shared prefix, and deciding what to
//! keep.
//!
//! [`PrefixCache`] is that bookkeeping. It retains a page-sharing fork of
//! completed prefills and, for each new prompt, finds the longest cached prefix
//! it can start from.
//!
//! # Keying: chained per-page hashes, verified against the tokens
//!
//! Sharing is at **whole-page granularity**. An entry always ends on a
//! [`PAGE_SIZE`] boundary, which is what makes reuse free: a fork that stops
//! mid-page inherits a partially-filled page, and the borrower's first append
//! copies it (copy-on-write). Whole pages are never written by either side, so
//! they stay shared for their lifetime.
//!
//! Each entry stores one hash per full page, chained: `h[p]` mixes `h[p-1]`
//! with the 16 token ids of page `p`, so equal hashes at page `p` mean the
//! whole prefix up to `p` agreed, and a lookup is a prefix-compare of two short
//! `u64` vectors instead of a token-by-token walk. The hash is only an index:
//! before a fork is handed out the candidate's tokens are compared with the
//! prompt for real, so a collision costs a missed reuse and never a wrong
//! answer.
//!
//! A trie keyed on page hashes would turn the linear scan over entries into a
//! descent; with an entry budget in the tens and a compare that is a handful of
//! `u64`s per entry, the scan is far below the cost of the prefill it saves,
//! and this way the exact-token check has somewhere obvious to live.
//!
//! # Bounds and eviction
//!
//! Retained pages are pages the pool cannot hand to a live request, so the
//! registry is bounded twice — by entry count and by pages held — and evicts
//! the least-recently-used entry when either bound is crossed. Dropping an
//! entry releases its pages back to the pool (those a live sequence or another
//! entry still shares stay alive until the last holder goes away).
//!
//! Callers additionally evict on demand: the inference engine, when a
//! reservation fails with [`KvPagePoolExhausted`], evicts prefix entries and
//! retries before giving up on the request. A cached prefix is an optimisation
//! and must never starve a request that is actually running.
//!
//! [`fork_from`]: PagedKvCache::fork_from
//! [`KvPagePoolExhausted`]: crate::error::GlintError::KvPagePoolExhausted

use super::{KvStore, PagedKvCache, PAGE_SIZE};

/// Smallest prefix worth sharing, in whole pages.
///
/// Below one page there is nothing to share — a fork that carries no complete
/// page saves no prefill positions and still costs a lookup.
pub const MIN_SHARED_PAGES: usize = 1;

/// Hit/miss accounting for a [`PrefixCache`], plus its current occupancy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Lookups that found a usable prefix and returned a fork.
    pub hits: u64,
    /// Lookups that found nothing to reuse.
    pub misses: u64,
    /// Entries dropped to stay within budget (or on a caller's demand).
    pub evictions: u64,
    /// Prompt positions served from cached pages instead of being prefilled.
    pub tokens_reused: u64,
    /// Entries currently retained.
    pub entries: usize,
    /// Pages currently retained (upper bound — see [`PrefixCache::pages`]).
    pub pages: usize,
}

/// Bounds on how much KV memory the registry may hold back from the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    /// Maximum retained entries; the least-recently-used one goes first.
    pub max_entries: usize,
    /// Maximum retained pages, counted as in [`PrefixCache::pages`].
    pub max_pages: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 16,
            max_pages: 512,
        }
    }
}

/// One retained prefix: the tokens it covers and a page-sharing fork of the
/// cache that computed them.
struct PrefixEntry {
    /// Exactly the cached prefix. Length is a multiple of [`PAGE_SIZE`].
    tokens: Vec<u32>,
    /// Chained hash per full page; `page_hashes.len() == tokens.len() / PAGE_SIZE`.
    page_hashes: Vec<u64>,
    /// Adapter this prefix was computed under. A LoRA changes K/V, so entries
    /// are never shared across adapters (`None` = base weights).
    lora: Option<String>,
    /// Fork of the producing sequence's cache, `len() == tokens.len()`. Holding
    /// it is what keeps the pages alive; dropping it returns them to the pool.
    cache: PagedKvCache,
    /// Pages the fork's page table spans, across all layers.
    pages: usize,
    /// LRU stamp — the registry's clock at the last hit or insert.
    last_used: u64,
}

/// Registry of reusable prompt prefixes.
///
/// Owned by whoever admits requests (the inference engine owns one on its loop
/// thread), so no locking is involved: lookups and inserts happen at admission
/// and completion boundaries, never on the per-token path.
pub struct PrefixCache {
    entries: Vec<PrefixEntry>,
    config: PrefixCacheConfig,
    /// Monotonic counter stamped onto entries for LRU ordering.
    clock: u64,
    stats: PrefixCacheStats,
}

impl PrefixCache {
    /// Create an empty registry with the given bounds.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            clock: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Longest cached prefix of `prompt`, as a fork ready to be prefilled from.
    ///
    /// The returned cache has `len()` equal to the number of prompt positions
    /// already present — the caller prefills `prompt[len()..]` with a position
    /// offset of `len()` and gets exactly what a cold prefill of the whole
    /// prompt would have produced.
    ///
    /// At least one token is always left to prefill: the forward pass needs a
    /// real suffix to produce logits from, so a prompt that matches an entry
    /// completely still re-runs its final page.
    pub fn lookup(&mut self, prompt: &[u32], lora: Option<&str>) -> Option<PagedKvCache> {
        let usable_pages = prompt.len().saturating_sub(1) / PAGE_SIZE;
        if usable_pages < MIN_SHARED_PAGES || self.entries.is_empty() {
            self.stats.misses += 1;
            return None;
        }
        let chain = page_chain(&prompt[..usable_pages * PAGE_SIZE]);

        let mut best: Option<(usize, usize)> = None; // (entry, shared pages)
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.lora.as_deref() != lora {
                continue;
            }
            let shared = entry
                .page_hashes
                .iter()
                .zip(&chain)
                .take_while(|(a, b)| a == b)
                .count();
            if shared >= MIN_SHARED_PAGES && best.is_none_or(|(_, best_pages)| shared > best_pages)
            {
                best = Some((idx, shared));
            }
        }

        let Some((idx, pages)) = best else {
            self.stats.misses += 1;
            return None;
        };
        let upto = pages * PAGE_SIZE;
        // The hashes are an index, not the answer: reusing another prompt's K/V
        // would be silently wrong output, so the tokens are compared for real
        // and a collision degrades to a miss.
        if self.entries[idx].tokens[..upto] != prompt[..upto] {
            self.stats.misses += 1;
            return None;
        }

        self.clock += 1;
        self.entries[idx].last_used = self.clock;
        self.stats.hits += 1;
        self.stats.tokens_reused += upto as u64;
        Some(self.entries[idx].cache.fork_from(upto))
    }

    /// Retain the whole pages of `prompt` that `cache` has just computed.
    ///
    /// `cache` must hold at least `prompt.len()` positions — i.e. call this
    /// once the prompt's prefill has finished. Only the prompt's *complete*
    /// pages are retained: generated tokens are never cached (a continuation is
    /// this request's alone), and stopping on a page boundary means the entry
    /// and the live sequence never write the same page, so neither pays for a
    /// copy-on-write.
    ///
    /// Inserting a prefix that extends an entry already held replaces it: the
    /// shorter entry's pages are the very pages the new fork holds, so anything
    /// that matched the old entry matches the new one just as far.
    pub fn insert(&mut self, prompt: &[u32], cache: &PagedKvCache, lora: Option<&str>) {
        let pages = prompt.len() / PAGE_SIZE;
        let upto = pages * PAGE_SIZE;
        if pages < MIN_SHARED_PAGES || upto > cache.len() {
            return;
        }
        let tokens = &prompt[..upto];

        // Already held: keep the entry (its pages are these pages) and just
        // refresh its LRU stamp.
        self.clock += 1;
        let stamp = self.clock;
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.lora.as_deref() == lora && e.tokens == tokens)
        {
            entry.last_used = stamp;
            return;
        }

        // Superseded entries: shorter prefixes of this one, holding pages this
        // fork also holds. Dropping them frees nothing yet but keeps the
        // registry from filling with one entry per page of the same prompt.
        let mut idx = 0;
        while idx < self.entries.len() {
            let entry = &self.entries[idx];
            let superseded = entry.lora.as_deref() == lora
                && entry.tokens.len() < tokens.len()
                && tokens.starts_with(&entry.tokens);
            if superseded {
                self.entries.swap_remove(idx);
            } else {
                idx += 1;
            }
        }

        let fork = cache.fork_from(upto);
        self.entries.push(PrefixEntry {
            tokens: tokens.to_vec(),
            page_hashes: page_chain(tokens),
            lora: lora.map(str::to_owned),
            pages: fork.allocated_pages(),
            cache: fork,
            last_used: stamp,
        });
        self.enforce_budget();
    }

    /// Drop the least-recently-used entry, returning its pages to the pool.
    ///
    /// Returns `false` when there was nothing left to evict — callers retrying
    /// an allocation use that to stop looping.
    pub fn evict_lru(&mut self) -> bool {
        let Some(idx) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(idx, _)| idx)
        else {
            return false;
        };
        // Dropping the entry drops its `PagedKvCache`, which releases every
        // page no live sequence (or other entry) still points at.
        self.entries.swap_remove(idx);
        self.stats.evictions += 1;
        true
    }

    /// Drop every entry, releasing all pages the registry was holding.
    pub fn clear(&mut self) {
        self.stats.evictions += self.entries.len() as u64;
        self.entries.clear();
    }

    /// Entries currently retained.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pages currently retained, summed per entry.
    ///
    /// This is an **upper bound** on the pages the registry keeps out of the
    /// pool: two entries sharing a common ancestor prefix hold the same page
    /// twice here, but the pool sees one. Over-counting is the safe direction
    /// for a budget — it evicts slightly sooner than strictly necessary rather
    /// than holding more memory than configured.
    pub fn pages(&self) -> usize {
        self.entries.iter().map(|e| e.pages).sum()
    }

    /// Counters plus current occupancy.
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.entries.len(),
            pages: self.pages(),
            ..self.stats
        }
    }

    /// Evict until both configured bounds hold.
    fn enforce_budget(&mut self) {
        while self.entries.len() > self.config.max_entries || self.pages() > self.config.max_pages {
            if !self.evict_lru() {
                return;
            }
        }
    }
}

// ── Hashing ──────────────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the previous chain value followed by one page of token ids.
///
/// Chaining is what makes a per-page hash usable as a *prefix* key: `h[p]`
/// depends on every token up to the end of page `p`, so two entries agreeing at
/// page `p` agree everywhere before it.
fn page_chain_hash(prev: u64, page_tokens: &[u32]) -> u64 {
    let mut h = FNV_OFFSET;
    for byte in prev.to_le_bytes() {
        h = (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    for &tok in page_tokens {
        for byte in tok.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Chained hash for every complete [`PAGE_SIZE`] page of `tokens`.
fn page_chain(tokens: &[u32]) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len() / PAGE_SIZE);
    let mut prev = 0u64;
    for page in tokens.chunks_exact(PAGE_SIZE) {
        prev = page_chain_hash(prev, page);
        out.push(prev);
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvStore, PagePool};

    const N_LAYERS: usize = 2;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

    fn pool(capacity_pages: usize) -> PagePool {
        PagePool::new(capacity_pages, N_KV_HEADS, HEAD_DIM)
    }

    /// A cache holding `tokens.len()` positions, with rows derived from the
    /// token ids so a mixed-up page is visible in the data.
    fn filled(pool: &PagePool, tokens: &[u32]) -> PagedKvCache {
        let mut cache = PagedKvCache::new(pool, N_LAYERS);
        cache.reserve(tokens.len()).expect("pool has room");
        for (pos, &tok) in tokens.iter().enumerate() {
            for layer in 0..N_LAYERS {
                let k: Vec<f32> = (0..KV_DIM)
                    .map(|d| tok as f32 + layer as f32 * 0.5 + d as f32 * 0.125)
                    .collect();
                let v: Vec<f32> = k.iter().map(|x| x + 100.0).collect();
                cache.write(layer, pos, &k, &v);
            }
            cache.advance();
        }
        cache
    }

    fn prompt(len: usize, tag: u32) -> Vec<u32> {
        (0..len as u32).map(|i| i * 7 + tag).collect()
    }

    fn registry() -> PrefixCache {
        PrefixCache::new(PrefixCacheConfig {
            max_entries: 4,
            max_pages: 64,
        })
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    /// The common prefix is 40 tokens, so two whole pages (32 positions) are
    /// shared and the partial third page is left to the borrower.
    #[test]
    fn test_lookup_returns_longest_page_aligned_prefix() {
        let shared = pool(64);
        let mut pc = registry();
        let mut a = prompt(40, 0);
        let mut b = a.clone();
        a.extend([1, 1, 1]);
        b.extend([2, 2, 2]);

        pc.insert(&a, &filled(&shared, &a), None);
        let fork = pc.lookup(&b, None).expect("prefix is cached");
        assert_eq!(fork.len(), 32, "sharing stops on a page boundary");
        assert_eq!(pc.stats().hits, 1);
        assert_eq!(pc.stats().tokens_reused, 32);
    }

    /// Prompts sharing 20 tokens (one full page plus four) share exactly the
    /// one complete page — the partial page is the borrower's to compute.
    #[test]
    fn test_lookup_shares_whole_pages_only() {
        let shared = pool(64);
        let mut pc = registry();
        let base = prompt(20, 0);
        let mut a = base.clone();
        let mut b = base.clone();
        a.extend([3, 3]);
        b.extend([4, 4]);

        pc.insert(&a, &filled(&shared, &a), None);
        let fork = pc.lookup(&b, None).expect("one page is cached");
        assert_eq!(fork.len(), PAGE_SIZE);
        assert_eq!(fork.pages_per_layer(), 1);
    }

    /// A cached prefix that covers the whole prompt still leaves the last page
    /// to prefill — a forward pass over zero tokens produces no logits.
    #[test]
    fn test_lookup_always_leaves_a_suffix_to_prefill() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), None);

        let fork = pc.lookup(&tokens, None).expect("prefix is cached");
        assert_eq!(fork.len(), 16, "the final page is re-prefilled");
        assert!(fork.len() < tokens.len());
    }

    #[test]
    fn test_lookup_below_one_page_is_a_miss() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), None);

        assert!(pc.lookup(&tokens[..16], None).is_none(), "16 - 1 < a page");
        assert!(pc.lookup(&[1, 2, 3], None).is_none());
        assert_eq!(pc.stats().misses, 2);
    }

    #[test]
    fn test_lookup_on_an_empty_registry_is_a_miss() {
        let mut pc = registry();
        assert!(pc.lookup(&prompt(64, 0), None).is_none());
        assert_eq!(
            pc.stats(),
            PrefixCacheStats {
                misses: 1,
                ..Default::default()
            }
        );
    }

    /// Prompts that differ inside the first page share nothing, even though
    /// they are the same length.
    #[test]
    fn test_lookup_rejects_a_different_prompt() {
        let shared = pool(64);
        let mut pc = registry();
        let a = prompt(48, 0);
        let b = prompt(48, 1);
        pc.insert(&a, &filled(&shared, &a), None);
        assert!(pc.lookup(&b, None).is_none());
    }

    /// A LoRA adapter changes every K/V it touches, so entries are keyed by
    /// adapter and a request under a different one must not reuse them.
    #[test]
    fn test_entries_are_keyed_by_lora_adapter() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(48, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), Some("adapter-a"));

        assert!(pc.lookup(&tokens, None).is_none(), "base weights differ");
        assert!(pc.lookup(&tokens, Some("adapter-b")).is_none());
        assert!(pc.lookup(&tokens, Some("adapter-a")).is_some());
    }

    /// The longest match wins when several entries share a prompt's opening.
    #[test]
    fn test_lookup_picks_the_longest_match() {
        let shared = pool(128);
        let mut pc = registry();
        let long = prompt(80, 0);
        let short_divergent = {
            let mut t = long[..32].to_vec();
            t.extend(prompt(32, 5));
            t
        };
        pc.insert(&short_divergent, &filled(&shared, &short_divergent), None);
        pc.insert(&long, &filled(&shared, &long), None);

        let fork = pc.lookup(&long, None).expect("both entries match");
        assert_eq!(fork.len(), 64, "80 - 1 rounds down to four pages");
    }

    // ── Insert ───────────────────────────────────────────────────────────

    /// Entries stop at the prompt's last complete page: a generated
    /// continuation belongs to one request and must never be handed to another.
    #[test]
    fn test_insert_never_caches_past_the_prompt() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(20, 0);
        let mut cache = filled(&shared, &tokens);
        // Sequence keeps generating past its prompt.
        let generated = prompt(30, 9);
        cache.reserve(50).expect("pool has room");
        for (i, _) in generated.iter().enumerate().take(30) {
            for layer in 0..N_LAYERS {
                cache.write(layer, 20 + i, &[0.5; KV_DIM], &[0.25; KV_DIM]);
            }
            cache.advance();
        }
        pc.insert(&tokens, &cache, None);
        assert_eq!(pc.len(), 1);

        let mut probe = tokens.clone();
        probe.extend([1, 2]);
        let fork = pc.lookup(&probe, None).expect("the prompt page is cached");
        assert_eq!(fork.len(), PAGE_SIZE, "only the prompt's full page");
    }

    #[test]
    fn test_insert_below_one_page_is_ignored() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(15, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), None);
        assert!(pc.is_empty());
    }

    /// Re-inserting the same prefix keeps the entry already held instead of
    /// piling up duplicates that hold the very same pages.
    #[test]
    fn test_insert_of_a_known_prefix_is_a_touch() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), None);
        let pages_after_first = pc.pages();
        pc.insert(&tokens, &filled(&shared, &tokens), None);
        assert_eq!(pc.len(), 1);
        assert_eq!(pc.pages(), pages_after_first);
    }

    /// A longer prefix replaces the shorter one it extends — the shorter
    /// entry's pages are the pages the new fork holds.
    #[test]
    fn test_insert_supersedes_a_shorter_prefix() {
        let shared = pool(128);
        let mut pc = registry();
        let long = prompt(64, 0);
        pc.insert(&long[..32], &filled(&shared, &long[..32]), None);
        assert_eq!(pc.len(), 1);
        pc.insert(&long, &filled(&shared, &long), None);
        assert_eq!(pc.len(), 1, "the shorter prefix was superseded");

        let fork = pc.lookup(&long, None).expect("still matches");
        assert_eq!(fork.len(), 48, "64 - 1 rounds down to three pages");
    }

    /// Prompts that diverge keep separate entries — neither supersedes the
    /// other.
    #[test]
    fn test_diverging_prompts_keep_separate_entries() {
        let shared = pool(128);
        let mut pc = registry();
        let a = prompt(48, 0);
        let b = prompt(48, 1);
        pc.insert(&a, &filled(&shared, &a), None);
        pc.insert(&b, &filled(&shared, &b), None);
        assert_eq!(pc.len(), 2);
    }

    // ── Eviction and budgets ─────────────────────────────────────────────

    #[test]
    fn test_evict_lru_drops_the_least_recently_used_entry() {
        let shared = pool(128);
        let mut pc = registry();
        let a = prompt(32, 0);
        let b = prompt(32, 1);
        pc.insert(&a, &filled(&shared, &a), None);
        pc.insert(&b, &filled(&shared, &b), None);
        // Touch `a` so `b` becomes the least recently used.
        assert!(pc.lookup(&a, None).is_some());

        assert!(pc.evict_lru());
        assert_eq!(pc.len(), 1);
        assert!(pc.lookup(&a, None).is_some(), "the touched entry survived");
        assert!(pc.lookup(&b, None).is_none());
        assert_eq!(pc.stats().evictions, 1);
    }

    #[test]
    fn test_evict_lru_on_an_empty_registry_reports_nothing_to_do() {
        let mut pc = registry();
        assert!(!pc.evict_lru());
    }

    /// Evicting an entry returns its pages to the pool, so the memory a cached
    /// prefix holds is memory a live request can take back.
    #[test]
    fn test_eviction_returns_pages_to_the_pool() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        {
            let cache = filled(&shared, &tokens);
            pc.insert(&tokens, &cache, None);
            // The producing sequence is gone; only the entry holds the pages.
        }
        let held = shared.live_pages();
        assert_eq!(held, 2 * N_LAYERS, "two full pages per layer");

        assert!(pc.evict_lru());
        assert_eq!(shared.live_pages(), 0, "eviction frees the pages");
    }

    /// Pages shared with a live sequence survive their entry's eviction.
    #[test]
    fn test_eviction_keeps_pages_a_live_sequence_still_shares() {
        let shared = pool(64);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        let live = filled(&shared, &tokens);
        pc.insert(&tokens, &live, None);
        let held = shared.live_pages();

        assert!(pc.evict_lru());
        assert_eq!(
            shared.live_pages(),
            held,
            "the live sequence still owns every page"
        );
        drop(live);
        assert_eq!(shared.live_pages(), 0);
    }

    #[test]
    fn test_entry_budget_evicts_the_oldest() {
        let shared = pool(256);
        let mut pc = PrefixCache::new(PrefixCacheConfig {
            max_entries: 2,
            max_pages: 1024,
        });
        let prompts: Vec<Vec<u32>> = (0..3).map(|i| prompt(32, i)).collect();
        for p in &prompts {
            pc.insert(p, &filled(&shared, p), None);
        }
        assert_eq!(pc.len(), 2);
        assert!(pc.lookup(&prompts[0], None).is_none(), "oldest evicted");
        assert!(pc.lookup(&prompts[2], None).is_some());
        assert_eq!(pc.stats().evictions, 1);
    }

    #[test]
    fn test_page_budget_evicts_until_it_fits() {
        let shared = pool(256);
        // Two pages per layer per entry → one entry fits, a second does not.
        let mut pc = PrefixCache::new(PrefixCacheConfig {
            max_entries: 16,
            max_pages: 2 * N_LAYERS,
        });
        let a = prompt(32, 0);
        let b = prompt(32, 1);
        pc.insert(&a, &filled(&shared, &a), None);
        assert_eq!(pc.len(), 1);
        pc.insert(&b, &filled(&shared, &b), None);
        assert_eq!(pc.len(), 1, "page budget forced an eviction");
        assert!(pc.pages() <= 2 * N_LAYERS);
    }

    #[test]
    fn test_clear_releases_everything() {
        let shared = pool(128);
        let mut pc = registry();
        let tokens = prompt(32, 0);
        pc.insert(&tokens, &filled(&shared, &tokens), None);
        pc.clear();
        assert!(pc.is_empty());
        assert_eq!(shared.live_pages(), 0);
        assert_eq!(pc.stats().evictions, 1);
    }

    // ── Data correctness ─────────────────────────────────────────────────

    /// The fork must expose the producing sequence's rows verbatim — that is
    /// the whole point of reusing it instead of recomputing.
    #[test]
    fn test_fork_carries_the_cached_rows() {
        let shared = pool(128);
        let mut pc = registry();
        let tokens = prompt(48, 0);
        let source = filled(&shared, &tokens);
        pc.insert(&tokens, &source, None);

        let fork = pc.lookup(&tokens, None).expect("prefix is cached");
        for layer in 0..N_LAYERS {
            for pos in 0..fork.len() {
                assert_eq!(fork.k_at(layer, pos), source.k_at(layer, pos), "K {pos}");
                assert_eq!(fork.v_at(layer, pos), source.v_at(layer, pos), "V {pos}");
            }
        }
    }

    /// Two borrowers of the same entry write their own suffixes without
    /// disturbing each other or the entry.
    #[test]
    fn test_two_forks_diverge_without_corrupting_the_entry() {
        let shared = pool(128);
        let mut pc = registry();
        let base = prompt(32, 0);
        let mut a = base.clone();
        let mut b = base.clone();
        a.extend([1, 1]);
        b.extend([2, 2]);
        pc.insert(&base, &filled(&shared, &base), None);

        let mut fork_a = pc.lookup(&a, None).expect("hit");
        let mut fork_b = pc.lookup(&b, None).expect("hit");
        let start = fork_a.len();
        for (fork, tag) in [(&mut fork_a, 11.0f32), (&mut fork_b, 22.0f32)] {
            fork.reserve(start + 2).expect("pool has room");
            for pos in start..start + 2 {
                for layer in 0..N_LAYERS {
                    fork.write(layer, pos, &[tag; KV_DIM], &[tag; KV_DIM]);
                }
                fork.advance();
            }
        }

        assert_eq!(fork_a.k_at(0, start), [11.0; KV_DIM]);
        assert_eq!(fork_b.k_at(0, start), [22.0; KV_DIM]);
        // The shared prefix is untouched in both.
        for pos in 0..start {
            assert_eq!(fork_a.k_at(0, pos), fork_b.k_at(0, pos), "prefix {pos}");
        }
    }

    // ── Hashing ──────────────────────────────────────────────────────────

    #[test]
    fn test_page_chain_is_a_prefix_key() {
        let a = prompt(64, 0);
        let mut b = a[..32].to_vec();
        b.extend(prompt(32, 3));
        let (ha, hb) = (page_chain(&a), page_chain(&b));
        assert_eq!(ha[..2], hb[..2], "the shared pages hash the same");
        assert_ne!(ha[2], hb[2], "divergence changes every later page");
        assert_eq!(ha.len(), 4, "only complete pages are hashed");
    }

    #[test]
    fn test_page_chain_ignores_a_partial_tail() {
        let tokens = prompt(20, 0);
        assert_eq!(page_chain(&tokens).len(), 1);
        assert_eq!(page_chain(&tokens[..16]), page_chain(&tokens));
    }
}
