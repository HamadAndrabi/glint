//! KV snapshot — serialise and restore a [`Session`]'s full cache state.
//!
//! ## Binary format (version 2)
//!
//! ```text
//! [u8; 8]             magic = b"GLNTSNAP"
//! u32 LE              version = 2
//! u64 LE              model_hash   (FNV-64 of model file path + file size)
//! u32 LE              context_len
//! u32 LE              n_layers
//! u32 LE              n_kv_heads
//! u32 LE              head_dim
//! u8                  cache_format (0 = F32, 1 = Q8)
//! u32 LE              token_count
//! [u32 LE; token_count]  tokens
//! u32 LE              prefill_len
//! u32 LE              pos
//! u64 LE              rng_state    (Xorshift64 state for deterministic resume)
//! For each layer l in 0..n_layers:
//!   u64 LE            k_bytes_len
//!   [u8; k_bytes_len] K data
//!   u64 LE            v_bytes_len
//!   [u8; v_bytes_len] V data
//! ```
//!
//! Metadata is verified before any cache data is read — the first mismatch
//! returns a named error immediately.

use super::{CacheFormat, Session, SessionOptions};
use crate::error::GlintError;
use crate::sampling::Xorshift64;

// ── Magic and version ────────────────────────────────────────────────────────

const MAGIC: &[u8; 8] = b"GLNTSNAP";
const VERSION: u32 = 2;

// ── Types ────────────────────────────────────────────────────────────────────

/// Snapshot metadata — verified before any cache data is loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMetadata {
    /// FNV-64 hash of (model file path as bytes) + (file size as 8 LE bytes).
    pub model_hash:   u64,
    pub context_len:  u32,
    pub n_layers:     u32,
    pub n_kv_heads:   u32,
    pub head_dim:     u32,
    pub cache_format: CacheFormat,
}

/// A fully deserialised snapshot — ready to be restored into a [`Session`].
pub struct KvSnapshot {
    pub meta:      SnapshotMetadata,
    pub tokens:    Vec<u32>,
    pub prefill_len: u32,
    pub pos:       u32,
    pub rng_state: u64,
    /// One `Vec<u8>` per layer — raw K storage (f32 bytes or Q8 bytes).
    pub k_layers:  Vec<Vec<u8>>,
    /// One `Vec<u8>` per layer — raw V storage.
    pub v_layers:  Vec<Vec<u8>>,
}

// ── Serialisation helpers ────────────────────────────────────────────────────

fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    write_u64(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

struct Reader<'a> {
    data: &'a [u8],
    pos:  usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], GlintError> {
        if self.pos + n > self.data.len() {
            return Err(GlintError::SnapshotTruncated);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GlintError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, GlintError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GlintError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_byte_vec(&mut self) -> Result<Vec<u8>, GlintError> {
        let len = self.read_u64()? as usize;
        Ok(self.read_exact(len)?.to_vec())
    }
}

// ── FNV-64 model hash ────────────────────────────────────────────────────────

/// Compute a cheap identity hash for a model file: FNV-64 of the UTF-8 path
/// bytes followed by the 8-byte little-endian file size.
///
/// This is not a cryptographic hash — it detects the most common cases
/// (different file, truncated file) without reading any model content.
pub fn model_hash(path: &str, file_size: u64) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME:  u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for &b in path.as_bytes().iter().chain(file_size.to_le_bytes().iter()) {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ── Export ───────────────────────────────────────────────────────────────────

/// Serialise a [`Session`] to bytes.
///
/// `meta` must reflect the loaded model's dimensions so that import can
/// fail-fast if the snapshot is loaded against a different model.
pub fn export_snapshot_with_meta(
    session: &Session,
    meta: &SnapshotMetadata,
) -> Result<Vec<u8>, GlintError> {
    let (k_layers, v_layers) = session.cache.export_raw();

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);

    // ── Header ────────────────────────────────────────────────────────────
    buf.extend_from_slice(MAGIC);
    write_u32(&mut buf, VERSION);
    write_u64(&mut buf, meta.model_hash);
    write_u32(&mut buf, meta.context_len);
    write_u32(&mut buf, meta.n_layers);
    write_u32(&mut buf, meta.n_kv_heads);
    write_u32(&mut buf, meta.head_dim);
    write_u8(&mut buf, cache_format_tag(meta.cache_format));

    // ── Session state ────────────────────────────────────────────────────
    write_u32(&mut buf, session.tokens.len() as u32);
    for &tok in &session.tokens {
        write_u32(&mut buf, tok);
    }
    write_u32(&mut buf, session.prefill_len as u32);
    write_u32(&mut buf, session.pos as u32);
    write_u64(&mut buf, session.sampler.rng.state);

    // ── KV cache data ────────────────────────────────────────────────────
    for (k, v) in k_layers.iter().zip(v_layers.iter()) {
        write_bytes(&mut buf, k);
        write_bytes(&mut buf, v);
    }

    Ok(buf)
}

fn cache_format_tag(fmt: CacheFormat) -> u8 {
    match fmt { CacheFormat::F32 => 0, CacheFormat::Q8 => 1 }
}

fn cache_format_from_tag(tag: u8) -> Option<CacheFormat> {
    match tag { 0 => Some(CacheFormat::F32), 1 => Some(CacheFormat::Q8), _ => None }
}

/// Read just the cache-format tag from a snapshot header.
///
/// This is useful for callers that need the expected metadata before they can
/// run full validation.
pub fn peek_snapshot_cache_format(bytes: &[u8]) -> Result<CacheFormat, GlintError> {
    let mut r = Reader::new(bytes);
    let magic = r.read_exact(8)?;
    if magic != MAGIC {
        return Err(GlintError::SnapshotBadMagic);
    }
    let version = r.read_u32()?;
    if version != VERSION {
        return Err(GlintError::SnapshotVersionUnsupported { found: version, current: VERSION });
    }
    r.read_u64()?; // model_hash
    r.read_u32()?; // context_len
    r.read_u32()?; // n_layers
    r.read_u32()?; // n_kv_heads
    r.read_u32()?; // head_dim
    let fmt_tag = r.read_u8()?;
    cache_format_from_tag(fmt_tag).ok_or(GlintError::SnapshotTruncated)
}

// ── Import ───────────────────────────────────────────────────────────────────

/// Deserialise and verify a snapshot blob.
///
/// `expected` must match the loaded model's dimensions; any mismatch returns
/// a named error before any cache data is read.
pub fn import_snapshot(
    bytes: &[u8],
    expected: &SnapshotMetadata,
) -> Result<KvSnapshot, GlintError> {
    let mut r = Reader::new(bytes);

    // ── Magic ──────────────────────────────────────────────────────────────
    let magic = r.read_exact(8)?;
    if magic != MAGIC {
        return Err(GlintError::SnapshotBadMagic);
    }

    // ── Version ────────────────────────────────────────────────────────────
    let version = r.read_u32()?;
    if version != VERSION {
        return Err(GlintError::SnapshotVersionUnsupported { found: version, current: VERSION });
    }

    // ── Metadata — fail-fast on every mismatch ────────────────────────────
    let model_hash = r.read_u64()?;
    if model_hash != expected.model_hash {
        return Err(GlintError::SnapshotModelMismatch {
            expected: expected.model_hash,
            found: model_hash,
        });
    }

    let context_len = r.read_u32()?;
    if context_len != expected.context_len {
        return Err(GlintError::SnapshotMetaMismatch {
            field: "context_len",
            expected: expected.context_len as u64,
            found: context_len as u64,
        });
    }

    let n_layers = r.read_u32()?;
    if n_layers != expected.n_layers {
        return Err(GlintError::SnapshotMetaMismatch {
            field: "n_layers",
            expected: expected.n_layers as u64,
            found: n_layers as u64,
        });
    }

    let n_kv_heads = r.read_u32()?;
    if n_kv_heads != expected.n_kv_heads {
        return Err(GlintError::SnapshotMetaMismatch {
            field: "n_kv_heads",
            expected: expected.n_kv_heads as u64,
            found: n_kv_heads as u64,
        });
    }

    let head_dim = r.read_u32()?;
    if head_dim != expected.head_dim {
        return Err(GlintError::SnapshotMetaMismatch {
            field: "head_dim",
            expected: expected.head_dim as u64,
            found: head_dim as u64,
        });
    }

    let fmt_tag = r.read_u8()?;
    let cache_format = cache_format_from_tag(fmt_tag)
        .ok_or(GlintError::SnapshotTruncated)?;
    if cache_format != expected.cache_format {
        return Err(GlintError::SnapshotMetaMismatch {
            field: "cache_format",
            expected: cache_format_tag(expected.cache_format) as u64,
            found: fmt_tag as u64,
        });
    }

    // ── Session state ─────────────────────────────────────────────────────
    let token_count = r.read_u32()? as usize;
    let mut tokens = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        tokens.push(r.read_u32()?);
    }
    let prefill_len = r.read_u32()?;
    let pos = r.read_u32()?;
    let rng_state = r.read_u64()?;

    // ── KV cache data ─────────────────────────────────────────────────────
    let mut k_layers = Vec::with_capacity(n_layers as usize);
    let mut v_layers = Vec::with_capacity(n_layers as usize);
    for _ in 0..n_layers {
        k_layers.push(r.read_byte_vec()?);
        v_layers.push(r.read_byte_vec()?);
    }

    Ok(KvSnapshot {
        meta: SnapshotMetadata {
            model_hash,
            context_len,
            n_layers,
            n_kv_heads,
            head_dim,
            cache_format,
        },
        tokens,
        prefill_len,
        pos,
        rng_state,
        k_layers,
        v_layers,
    })
}

// ── Restore ───────────────────────────────────────────────────────────────────

/// Restore a [`Session`] from a verified [`KvSnapshot`].
///
/// The session's sampler RNG is seeded from `snap.rng_state` so that
/// generation resumes deterministically when a fixed seed was used.
pub fn restore_session(
    snap: KvSnapshot,
    opts: SessionOptions,
) -> Result<Session, GlintError> {
    let mut session = Session::new(opts);
    session.tokens = snap.tokens;
    session.prefill_len = snap.prefill_len as usize;
    session.pos = snap.pos as usize;

    // Restore RNG state for deterministic resume.
    session.sampler.rng = Xorshift64::restore(snap.rng_state);

    // Import KV data into the freshly allocated cache.
    let token_count = session.tokens.len();
    session.cache.import_raw(&snap.k_layers, &snap.v_layers, token_count)?;

    Ok(session)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplerConfig;
    use crate::session::CacheFormat;

    fn make_meta() -> SnapshotMetadata {
        SnapshotMetadata {
            model_hash:   0xdeadbeef_cafebabe,
            context_len:  64,
            n_layers:     2,
            n_kv_heads:   2,
            head_dim:     8,
            cache_format: CacheFormat::F32,
        }
    }

    fn make_opts(meta: &SnapshotMetadata) -> SessionOptions {
        SessionOptions {
            max_new_tokens: 16,
            sampler_cfg:    SamplerConfig { seed: Some(42), ..Default::default() },
            eos_token:      2,
            cache_format:   meta.cache_format,
            context_length: meta.context_len as usize,
            n_layers:       meta.n_layers as usize,
            n_kv_heads:     meta.n_kv_heads as usize,
            head_dim:       meta.head_dim as usize,
            lora_adapter:   None,
        }
    }

    fn write_dummy_cache(session: &mut Session) {
        // Write 2 positions of known K/V data (2 layers × 2 kv_heads × 8 head_dim = kv_dim 16)
        let kv_dim = 2 * 8; // n_kv_heads * head_dim
        let k0: Vec<f32> = (0..kv_dim as u32).map(|i| i as f32 * 0.1).collect();
        let v0: Vec<f32> = (0..kv_dim as u32).map(|i| i as f32 * 0.2 + 1.0).collect();
        let k1: Vec<f32> = (0..kv_dim as u32).map(|i| i as f32 * 0.3 + 10.0).collect();
        let v1: Vec<f32> = (0..kv_dim as u32).map(|i| i as f32 * 0.4 + 20.0).collect();
        for layer in 0..2 {
            session.cache.write(layer, 0, &k0, &v0);
            session.cache.write(layer, 1, &k1, &v1);
        }
        session.cache.advance();
        session.cache.advance();
        session.tokens = vec![10, 20];
        session.prefill_len = 1;
        session.pos = 1;
    }

    #[test]
    fn test_roundtrip_f32() {
        let meta = make_meta();
        let opts = make_opts(&meta);

        let mut original = Session::new(opts);
        write_dummy_cache(&mut original);

        // Export
        let bytes = export_snapshot_with_meta(&original, &meta).unwrap();
        assert!(bytes.starts_with(b"GLNTSNAP"), "magic mismatch");

        // Import
        let snap = import_snapshot(&bytes, &meta).unwrap();
        assert_eq!(snap.tokens, vec![10, 20]);
        assert_eq!(snap.prefill_len, 1);
        assert_eq!(snap.pos, 1);

        // Restore
        let opts2 = make_opts(&meta);
        let restored = restore_session(snap, opts2).unwrap();

        // Verify cache content matches original
        let mut buf_orig = vec![0.0f32; 8];
        let mut buf_rest = vec![0.0f32; 8];
        for layer in 0..2 {
            for pos in 0..2 {
                original.cache.read_k_head(layer, pos, 0, 8, &mut buf_orig);
                restored.cache.read_k_head(layer, pos, 0, 8, &mut buf_rest);
                assert_eq!(buf_orig, buf_rest, "K mismatch layer={layer} pos={pos}");
                original.cache.read_v_head(layer, pos, 0, 8, &mut buf_orig);
                restored.cache.read_v_head(layer, pos, 0, 8, &mut buf_rest);
                assert_eq!(buf_orig, buf_rest, "V mismatch layer={layer} pos={pos}");
            }
        }
        assert_eq!(restored.prefill_len, 1);
    }

    #[test]
    fn test_bad_magic() {
        let meta = make_meta();
        let mut bytes = vec![0u8; 16];
        bytes[0..8].copy_from_slice(b"NOTMAGIC");
        assert!(matches!(import_snapshot(&bytes, &meta), Err(GlintError::SnapshotBadMagic)));
    }

    #[test]
    fn test_model_hash_mismatch() {
        let meta = make_meta();
        let opts = make_opts(&meta);
        let mut session = Session::new(opts);
        write_dummy_cache(&mut session);

        let bytes = export_snapshot_with_meta(&session, &meta).unwrap();

        let mut wrong_meta = meta.clone();
        wrong_meta.model_hash ^= 1;
        assert!(matches!(
            import_snapshot(&bytes, &wrong_meta),
            Err(GlintError::SnapshotModelMismatch { .. })
        ));
    }

    #[test]
    fn test_n_layers_mismatch() {
        let meta = make_meta();
        let opts = make_opts(&meta);
        let mut session = Session::new(opts);
        write_dummy_cache(&mut session);

        let bytes = export_snapshot_with_meta(&session, &meta).unwrap();

        let mut wrong_meta = meta.clone();
        wrong_meta.n_layers = 4;
        assert!(matches!(
            import_snapshot(&bytes, &wrong_meta),
            Err(GlintError::SnapshotMetaMismatch { field: "n_layers", .. })
        ));
    }

    #[test]
    fn test_rng_state_restored() {
        let meta = make_meta();
        let opts = make_opts(&meta);
        let mut session = Session::new(opts);
        // Advance RNG a few steps
        for _ in 0..10 { session.sampler.rng.next_f32(); }
        let state_before = session.sampler.rng.state;
        write_dummy_cache(&mut session);

        let bytes = export_snapshot_with_meta(&session, &meta).unwrap();
        let snap = import_snapshot(&bytes, &meta).unwrap();
        let opts2 = make_opts(&meta);
        let restored = restore_session(snap, opts2).unwrap();
        assert_eq!(restored.sampler.rng.state, state_before);
    }

    #[test]
    fn test_model_hash_fn() {
        let h1 = model_hash("path/to/model.gguf", 1234567890);
        let h2 = model_hash("path/to/model.gguf", 1234567890);
        assert_eq!(h1, h2, "hash must be deterministic");

        let h3 = model_hash("path/to/other.gguf", 1234567890);
        assert_ne!(h1, h3, "different paths must produce different hashes");

        let h4 = model_hash("path/to/model.gguf", 9999999999);
        assert_ne!(h1, h4, "different sizes must produce different hashes");
    }
}
