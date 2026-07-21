# C FFI

Glint exposes a C-compatible ABI behind the `cffi` feature. The surface is handle-based: callers create opaque model/session/snapshot objects and interact through exported functions in `include/glint.h`.

Source: `src/ffi/mod.rs`, `include/glint.h`

---

## Build

```bash
cargo build --release --features cffi
```

This produces a `cdylib` plus the checked-in header:

- `include/glint.h`

---

## Handles

| Handle | Purpose |
|--------|---------|
| `GlintModel*` | Loaded GGUF model |
| `GlintSession*` | Generation session plus sampling options |
| `GlintSnapshot*` | Serialized session state |

Errors are reported through `glint_last_error()`.

---

## Typical Flow

```c
GlintModel* model = glint_model_load("model.gguf");
GlintSamplerOptions opts = {0};
GlintSession* session = glint_session_new(model, &opts, "f32");

uint32_t out[256];
int n = glint_generate(model, session, "Hello", 64, out, 256);

GlintSnapshot* snap = glint_snapshot_export(model, session);
glint_session_free(session);
session = glint_snapshot_import(model, snap, &opts);
```

`glint_generate()` and `glint_stream_generate()` reset the session for the provided prompt, run prefill, then update the stored session state as tokens are generated. That means exported snapshots reflect the actual post-generation state.

---

## Snapshot Functions

| Function | Purpose |
|----------|---------|
| `glint_snapshot_export()` | Export the current session state |
| `glint_snapshot_import()` | Restore a new session from a snapshot |
| `glint_snapshot_serialize()` | Copy snapshot bytes into a caller buffer |
| `glint_snapshot_deserialize()` | Parse a serialized snapshot blob |

Snapshot restore re-validates model identity and KV cache format before importing raw cache bytes.

---

## Panic and Thread Safety

Every exported `extern "C"` function runs its body under `catch_unwind`. A Rust
panic can therefore never unwind across the FFI boundary (which would be undefined
behaviour and abort the host process): a caught panic is reported as a `NULL` handle
or a `-1` return code, with the reason available from `glint_last_error()`. All
incoming pointers are null-checked before use.

Thread-safety follows handle ownership:

- `GlintModel*` is read-only after load and may be shared across threads — any
  number of threads may call `const GlintModel*` functions concurrently.
- `GlintSession*` is **not** thread-safe: it owns mutable KV-cache and RNG state,
  so driving one session from two threads at once is a data race. Use one session
  per thread, or serialize with your own lock.
- `glint_last_error()` is thread-local — each thread sees only its own errors.

The snapshot deserialize path parses fully untrusted bytes with bounded,
overflow-checked arithmetic (the same parser the fuzz suite exercises), so a
malformed blob yields an error rather than an over-allocation or out-of-bounds read.

---

## Notes

- Cache format strings are `"f32"` and `"q8"`.
- The FFI surface currently mirrors the synchronous runtime API.
- If you pass malformed input, check `glint_last_error()` after any `NULL` or `-1` return value.
