//! C-compatible FFI — opaque handle wrappers and `#[no_mangle]` functions.
//!
//! Enable with `cargo build --features cffi`.
//!
//! # Safety model
//!
//! * All pointer parameters are null-checked; null returns `NULL` or -1 and
//!   sets the thread-local error string (readable via `glint_last_error()`).
//! * Ownership follows a simple rule: `*_new` / `*_load` / `*_import` hand
//!   out heap-allocated objects; the corresponding `*_free` function must be
//!   called exactly once.  After `free`, the pointer is dangling — never use it.
//! * `const GlintModel*` / `const GlintSession*` parameters are borrowed for
//!   the duration of the call only; the caller retains ownership.
//! * Every entry point runs inside `catch_unwind`: an internal Rust panic is
//!   caught at the boundary (unwinding across FFI is undefined behaviour),
//!   turned into the error string, and reported as `NULL` / -1. A panic never
//!   propagates into the C caller and never aborts the host process.
//!
//! # Thread-safety
//!
//! * `GlintModel*` is read-only after load and may be shared across threads:
//!   any number of threads may call functions that take `const GlintModel*`
//!   concurrently.
//! * `GlintSession*` is **not** thread-safe. A session owns mutable KV-cache
//!   and RNG state; calling `glint_generate` / `glint_stream_generate` /
//!   `glint_snapshot_export` on the same session from two threads at once is a
//!   data race (undefined behaviour). Use one session per thread, or serialise
//!   access with your own lock.
//! * The error string returned by `glint_last_error()` is thread-local: each
//!   thread sees only errors from its own calls.
//!
//! # Example (C)
//! ```c
//! #include "glint.h"
//!
//! GlintModel* m = glint_model_load("model.gguf");
//! if (!m) { fprintf(stderr, "%s\n", glint_last_error()); return 1; }
//!
//! GlintSamplerOptions opts = { .temperature = 0.8, .seed = 42 };
//! GlintSession* s = glint_session_new(m, &opts, "f32");
//! uint32_t tokens[256];
//! int n = glint_generate(m, s, "Hello", 64, tokens, 256);
//!
//! glint_session_free(s);
//! glint_model_free(m);
//! ```

// The caller-obligation ("# Safety") contract for every entry point is stated
// once in the module header above (null-checking, ownership, borrow lifetimes,
// thread-safety, panic containment) rather than repeated on each `#[no_mangle]`
// function, so the per-function `missing_safety_doc` lint is allowed here.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use crate::api::{GenerationOptions, Model};
use crate::sampling::SamplerConfig;
use crate::session::snapshot::KvSnapshot;
use crate::session::CacheFormat;

// ── Thread-local error string ─────────────────────────────────────────────────

std::thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_error(msg: impl std::fmt::Display) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg.to_string()).ok();
    });
}

fn clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// ── Opaque handle types ───────────────────────────────────────────────────────

/// Opaque handle to a loaded model. Free with `glint_model_free`.
pub struct GlintModelHandle(pub Model);

/// Opaque handle to a generation session. Free with `glint_session_free`.
pub struct GlintSessionHandle {
    /// The underlying session state.
    session: crate::session::Session,
    /// The GenerationOptions used to create this session, needed for snapshot restore.
    opts: GenerationOptions,
}

/// Opaque handle to a KV snapshot. Free with `glint_snapshot_free`.
pub struct GlintSnapshotHandle {
    snap: KvSnapshot,
    bytes: Vec<u8>,
}

// ── C-visible struct for sampler options ─────────────────────────────────────

/// Sampler parameters passed from C.
///
/// Zero-initialise to use defaults:
/// ```c
/// GlintSamplerOptions opts = {0};
/// opts.temperature = 0.8;
/// ```
#[repr(C)]
pub struct GlintSamplerOptions {
    /// Sampling temperature. 0.0 = greedy, >0 = stochastic.
    pub temperature: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p nucleus filtering. 0.0 / 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty. 1.0 = disabled.
    pub repeat_penalty: f32,
    /// PRNG seed. 0 = seed from system time.
    pub seed: u64,
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
}

impl GlintSamplerOptions {
    fn to_generation_opts(&self, cache_fmt: CacheFormat) -> GenerationOptions {
        GenerationOptions {
            max_new_tokens: if self.max_new_tokens == 0 {
                256
            } else {
                self.max_new_tokens
            },
            sampler_cfg: SamplerConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: if self.top_p == 0.0 { 1.0 } else { self.top_p },
                repeat_penalty: if self.repeat_penalty == 0.0 {
                    1.0
                } else {
                    self.repeat_penalty
                },
                seed: if self.seed == 0 {
                    None
                } else {
                    Some(self.seed)
                },
                min_p: 0.0,
            },
            cache_format: cache_fmt,
            constraint: None,
            lora_adapter: None,
        }
    }
}

// ── Helper macros ─────────────────────────────────────────────────────────────

macro_rules! null_check {
    ($ptr:expr, $ret:expr) => {
        if $ptr.is_null() {
            set_error("null pointer argument");
            return $ret;
        }
    };
}

/// Run an FFI body under `catch_unwind`, returning `$default` if it panics.
///
/// Unwinding across an `extern "C"` boundary is undefined behaviour, so every
/// entry point funnels through here. `AssertUnwindSafe` is sound because on the
/// panic path we return an error sentinel and touch no partially-mutated state;
/// the caller observes only success-or-error, never a torn value.
fn ffi_guard<T>(default: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            set_error("internal panic caught at FFI boundary");
            default
        }
    }
}

fn parse_cache_format(s: &CStr) -> Option<CacheFormat> {
    match s.to_str().ok()? {
        "q8" | "Q8" => Some(CacheFormat::Q8),
        _ => Some(CacheFormat::F32), // default
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// Load a GGUF model from `path`.
///
/// Returns a heap-allocated handle on success, or `NULL` on failure (call
/// `glint_last_error()` for details).  Must be freed with `glint_model_free`.
#[no_mangle]
pub unsafe extern "C" fn glint_model_load(path: *const c_char) -> *mut GlintModelHandle {
    ffi_guard(std::ptr::null_mut(), || {
        null_check!(path, std::ptr::null_mut());
        clear_error();
        let path_cstr = unsafe { CStr::from_ptr(path) };
        let path_str = match path_cstr.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(e);
                return std::ptr::null_mut();
            }
        };
        match Model::load(std::path::Path::new(path_str)) {
            Ok(m) => Box::into_raw(Box::new(GlintModelHandle(m))),
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Free a model handle.  No-op if `model` is `NULL`.
#[no_mangle]
pub unsafe extern "C" fn glint_model_free(model: *mut GlintModelHandle) {
    ffi_guard((), || {
        if !model.is_null() {
            drop(unsafe { Box::from_raw(model) });
        }
    })
}

// ── Session ───────────────────────────────────────────────────────────────────

/// Create a new generation session.
///
/// `cache_format`: `"f32"` (default) or `"q8"`.
/// Returns `NULL` on failure.  Must be freed with `glint_session_free`.
#[no_mangle]
pub unsafe extern "C" fn glint_session_new(
    model: *const GlintModelHandle,
    sampler_opts: *const GlintSamplerOptions,
    cache_format: *const c_char,
) -> *mut GlintSessionHandle {
    ffi_guard(std::ptr::null_mut(), || {
        null_check!(model, std::ptr::null_mut());
        null_check!(sampler_opts, std::ptr::null_mut());
        clear_error();

        let fmt = if cache_format.is_null() {
            CacheFormat::F32
        } else {
            let cs = unsafe { CStr::from_ptr(cache_format) };
            parse_cache_format(cs).unwrap_or(CacheFormat::F32)
        };

        let sopts = unsafe { &*sampler_opts };
        let opts = sopts.to_generation_opts(fmt);
        let m = unsafe { &(*model).0 };
        let session = m.new_session(&opts);

        Box::into_raw(Box::new(GlintSessionHandle { session, opts }))
    })
}

/// Free a session handle.  No-op if `session` is `NULL`.
#[no_mangle]
pub unsafe extern "C" fn glint_session_free(session: *mut GlintSessionHandle) {
    ffi_guard((), || {
        if !session.is_null() {
            drop(unsafe { Box::from_raw(session) });
        }
    })
}

// ── Generation ────────────────────────────────────────────────────────────────

/// Prefill and decode up to `max_new_tokens` tokens into `out_tokens`.
///
/// Returns the number of tokens written, or -1 on error.
///
/// The session is reset for each call (prefill runs on `prompt`).
/// Generated token ids are written to `out_tokens[0..return_value]`.
#[no_mangle]
pub unsafe extern "C" fn glint_generate(
    model: *const GlintModelHandle,
    session: *mut GlintSessionHandle,
    prompt: *const c_char,
    max_new_tokens: usize,
    out_tokens: *mut u32,
    out_capacity: usize,
) -> c_int {
    ffi_guard(-1, || {
        null_check!(model, -1);
        null_check!(session, -1);
        null_check!(prompt, -1);
        null_check!(out_tokens, -1);
        clear_error();

        let m = unsafe { &(*model).0 };
        let s = unsafe { &mut *session };
        let cs = unsafe { CStr::from_ptr(prompt) };
        let txt = match cs.to_str() {
            Ok(t) => t,
            Err(e) => {
                set_error(e);
                return -1;
            }
        };

        let mut opts = s.opts.clone();
        if max_new_tokens > 0 {
            opts.max_new_tokens = max_new_tokens;
        }

        s.opts = opts.clone();
        s.session = m.new_session(&opts);
        match m.prefill(&mut s.session, txt, &mut None) {
            Ok(()) => {}
            Err(e) => {
                set_error(e);
                return -1;
            }
        }

        let mut tokens = Vec::new();
        while let Some(tok) = m.decode_one(&mut s.session, &mut None) {
            tokens.push(tok);
        }

        let n = tokens.len().min(out_capacity);
        let out = unsafe { std::slice::from_raw_parts_mut(out_tokens, n) };
        out.copy_from_slice(&tokens[..n]);
        n as c_int
    })
}

/// Streaming generation with a per-token callback.
///
/// `on_token(token_id, userdata)` is called for each generated token.
/// Return non-zero from the callback to stop early.
/// Returns total tokens generated, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn glint_stream_generate(
    model: *const GlintModelHandle,
    session: *mut GlintSessionHandle,
    prompt: *const c_char,
    max_new_tokens: usize,
    on_token: Option<unsafe extern "C" fn(u32, *mut c_void) -> c_int>,
    userdata: *mut c_void,
) -> c_int {
    ffi_guard(-1, || {
        null_check!(model, -1);
        null_check!(session, -1);
        null_check!(prompt, -1);
        clear_error();

        let callback = match on_token {
            Some(f) => f,
            None => {
                set_error("on_token callback is null");
                return -1;
            }
        };

        let m = unsafe { &(*model).0 };
        let s = unsafe { &mut *session };
        let cs = unsafe { CStr::from_ptr(prompt) };
        let txt = match cs.to_str() {
            Ok(t) => t,
            Err(e) => {
                set_error(e);
                return -1;
            }
        };

        let mut opts = s.opts.clone();
        if max_new_tokens > 0 {
            opts.max_new_tokens = max_new_tokens;
        }

        s.opts = opts.clone();
        s.session = m.new_session(&opts);
        if let Err(e) = m.prefill(&mut s.session, txt, &mut None) {
            set_error(e);
            return -1;
        }

        let mut count = 0i32;
        while let Some(tok) = m.decode_one(&mut s.session, &mut None) {
            count += 1;
            // SAFETY: callback and userdata are valid for the duration of this call.
            let stop = unsafe { callback(tok, userdata) };
            if stop != 0 {
                break;
            }
        }
        count
    })
}

// ── Snapshots ─────────────────────────────────────────────────────────────────

/// Export the current session state to a snapshot.
///
/// Returns an opaque handle on success, or `NULL` on failure.
/// Must be freed with `glint_snapshot_free`.
#[no_mangle]
pub unsafe extern "C" fn glint_snapshot_export(
    model: *const GlintModelHandle,
    session: *const GlintSessionHandle,
) -> *mut GlintSnapshotHandle {
    ffi_guard(std::ptr::null_mut(), || {
        null_check!(model, std::ptr::null_mut());
        null_check!(session, std::ptr::null_mut());
        clear_error();

        let m = unsafe { &(*model).0 };
        let s = unsafe { &(*session).session };

        match m
            .export_session(s)
            .and_then(|bytes| m.import_snapshot_bytes(&bytes).map(|snap| (bytes, snap)))
        {
            Ok((bytes, snap)) => Box::into_raw(Box::new(GlintSnapshotHandle { snap, bytes })),
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Restore a session from a snapshot.
///
/// Returns a new session handle on success, or `NULL` on failure.
/// Must be freed with `glint_session_free`.
#[no_mangle]
pub unsafe extern "C" fn glint_snapshot_import(
    model: *const GlintModelHandle,
    snapshot: *const GlintSnapshotHandle,
    sampler_opts: *const GlintSamplerOptions,
) -> *mut GlintSessionHandle {
    ffi_guard(std::ptr::null_mut(), || {
        null_check!(model, std::ptr::null_mut());
        null_check!(snapshot, std::ptr::null_mut());
        null_check!(sampler_opts, std::ptr::null_mut());
        clear_error();

        let m = unsafe { &(*model).0 };
        let snap = unsafe { &(*snapshot).snap };
        let so = unsafe { &*sampler_opts };
        let opts = so.to_generation_opts(snap.meta.cache_format);

        // Re-deserialise from the stored bytes so we get a fresh KvSnapshot to move.
        let meta = crate::session::snapshot::SnapshotMetadata {
            model_hash: m.model_hash,
            context_len: m.config.context_length,
            n_layers: m.config.block_count,
            n_kv_heads: m.config.head_count_kv,
            head_dim: m.config.head_dim(),
            cache_format: snap.meta.cache_format,
        };

        let bytes = unsafe { &(*snapshot).bytes };
        match crate::session::snapshot::import_snapshot(bytes, &meta) {
            Ok(fresh_snap) => match m.restore_session(fresh_snap, opts.clone()) {
                Ok(session) => Box::into_raw(Box::new(GlintSessionHandle { session, opts })),
                Err(e) => {
                    set_error(e);
                    std::ptr::null_mut()
                }
            },
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Serialise a snapshot to a caller-supplied byte buffer.
///
/// Returns the number of bytes written, or -1 if the buffer is too small.
/// Call with `buf = NULL` and `buf_len = 0` to query the required size.
#[no_mangle]
pub unsafe extern "C" fn glint_snapshot_serialize(
    snapshot: *const GlintSnapshotHandle,
    buf: *mut u8,
    buf_len: usize,
) -> c_int {
    ffi_guard(-1, || {
        null_check!(snapshot, -1);
        clear_error();

        let h = unsafe { &*snapshot };
        let src = &h.bytes;

        if buf.is_null() || buf_len == 0 {
            return src.len() as c_int;
        }
        if buf_len < src.len() {
            set_error(format!(
                "buffer too small: need {} bytes, got {}",
                src.len(),
                buf_len
            ));
            return -1;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(buf, src.len()) };
        dst.copy_from_slice(src);
        src.len() as c_int
    })
}

/// Deserialise a snapshot from bytes.
///
/// Returns an opaque handle on success, or `NULL` on failure.
/// Must be freed with `glint_snapshot_free`.
///
/// Note: pass the model handle to `glint_snapshot_import` after this to
/// restore a session — this function only parses the bytes without verifying
/// model identity.
#[no_mangle]
pub unsafe extern "C" fn glint_snapshot_deserialize(
    buf: *const u8,
    len: usize,
) -> *mut GlintSnapshotHandle {
    ffi_guard(std::ptr::null_mut(), || {
        null_check!(buf, std::ptr::null_mut());
        clear_error();

        let bytes = unsafe { std::slice::from_raw_parts(buf, len) };
        let stored = bytes.to_vec();

        // Read the header through the crate's single header parser (validates magic
        // and version) rather than re-deriving byte offsets here — the two must
        // never drift. The metadata is taken from the blob itself; full validation
        // against a specific model happens later in `glint_snapshot_import`.
        let meta = match crate::session::snapshot::peek_snapshot_metadata(&stored) {
            Ok(m) => m,
            Err(e) => {
                set_error(e);
                return std::ptr::null_mut();
            }
        };

        match crate::session::snapshot::import_snapshot(&stored, &meta) {
            Ok(snap) => Box::into_raw(Box::new(GlintSnapshotHandle {
                snap,
                bytes: stored,
            })),
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Free a snapshot handle.  No-op if `snapshot` is `NULL`.
#[no_mangle]
pub unsafe extern "C" fn glint_snapshot_free(snapshot: *mut GlintSnapshotHandle) {
    ffi_guard((), || {
        if !snapshot.is_null() {
            drop(unsafe { Box::from_raw(snapshot) });
        }
    })
}

// ── Error reporting ───────────────────────────────────────────────────────────

/// Return the last error message for the current thread.
///
/// Returns a `NULL`-terminated string, or `""` if no error has occurred.
/// The pointer is valid until the next Glint call on this thread.
#[no_mangle]
pub extern "C" fn glint_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|cs| cs.as_ptr())
            .unwrap_or(c"".as_ptr())
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_last_error_empty_initially() {
        clear_error();
        let ptr = glint_last_error();
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) };
        assert_eq!(s.to_str().unwrap(), "");
    }

    #[test]
    fn test_model_load_null_returns_null() {
        let ptr = unsafe { glint_model_load(std::ptr::null()) };
        assert!(ptr.is_null());
        let err = unsafe { CStr::from_ptr(glint_last_error()) };
        assert!(!err.to_str().unwrap().is_empty());
    }

    #[test]
    fn test_model_load_bad_path_returns_null() {
        let path = CString::new("/nonexistent/path/model.gguf").unwrap();
        let ptr = unsafe { glint_model_load(path.as_ptr()) };
        assert!(ptr.is_null());
        let err = unsafe { CStr::from_ptr(glint_last_error()) };
        assert!(!err.to_str().unwrap().is_empty());
    }

    #[test]
    fn test_session_new_null_model_returns_null() {
        let opts = GlintSamplerOptions {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: 0,
            max_new_tokens: 64,
        };
        let ptr = unsafe { glint_session_new(std::ptr::null(), &opts, std::ptr::null()) };
        assert!(ptr.is_null());
    }

    #[test]
    fn test_ffi_guard_catches_panic() {
        clear_error();
        // Silence the default panic hook so this expected panic doesn't spam
        // test output; restore it afterwards.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = ffi_guard(-1i32, || panic!("boom"));
        std::panic::set_hook(prev);

        assert_eq!(r, -1, "guard must return the default on panic, not unwind");
        let err = unsafe { CStr::from_ptr(glint_last_error()) };
        assert!(err.to_str().unwrap().contains("panic"));
    }

    #[test]
    fn test_snapshot_serialize_null_returns_neg1() {
        let n = unsafe { glint_snapshot_serialize(std::ptr::null(), std::ptr::null_mut(), 0) };
        assert_eq!(n, -1);
    }

    #[test]
    fn test_snapshot_deserialize_bad_magic() {
        let bad: &[u8] = b"BADMAGIC12345678901234567890";
        let ptr = unsafe { glint_snapshot_deserialize(bad.as_ptr(), bad.len()) };
        assert!(ptr.is_null());
    }

    #[test]
    fn test_snapshot_deserialize_truncated_header_returns_null() {
        let truncated = [0u8; 24];
        let ptr = unsafe { glint_snapshot_deserialize(truncated.as_ptr(), truncated.len()) };
        assert!(ptr.is_null());
    }
}
