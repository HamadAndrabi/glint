# Session API & Snapshots

Glint's library-facing runtime lives in `src/api/mod.rs` and `src/session/`. It exposes a small in-process API for loading a model, creating a session, prefilling a prompt, decoding token by token, and snapshotting the session state for later resume.

Source: `src/api/mod.rs`, `src/session/mod.rs`, `src/session/snapshot.rs`

---

## Core Types

| Type | Purpose |
|------|---------|
| `Model` | Loaded weights (GGUF or SafeTensors), tokenizer, and config |
| `GenerationOptions` | Sampling, cache format, constraint, and LoRA options |
| `Session` | Tokens, KV cache, RNG state, last logits, and generation budget |
| `CacheFormat` | `F32` or `Q8` KV cache |
| `KvSnapshot` | Serialized session state ready for restore |

The `F32` format can additionally be backed by on-demand pages instead of one
contiguous allocation by supplying `SessionOptions::page_pool` — see
[KV Cache](./kv-cache.md) for the paged variant and its sharing semantics.

---

## Basic Flow

```rust
use glint::api::{GenerationOptions, Model};

let model = Model::load("model.gguf".as_ref())?;
let opts = GenerationOptions::default();
let mut session = model.new_session(&opts);

model.prefill(&mut session, "Hello", &mut None)?;
while let Some(tok) = model.decode_one(&mut session, &mut None) {
    print!("{}", model.decode(&[tok]));
}
# Ok::<(), glint::error::GlintError>(())
```

`prefill()` tokenizes the prompt, fills the KV cache, and stores the next-token logits in the session. `decode_one()` samples from those logits, advances the cache by one token, and returns the emitted token ID.

---

## Snapshots

Snapshots capture:

- model metadata needed for validation
- full token history
- prompt length (`prefill_len`)
- KV cache bytes (`f32` or `Q8`)
- sampler RNG state

Export and restore:

```rust
# use glint::api::{GenerationOptions, Model};
# let model = Model::load("model.gguf".as_ref())?;
# let opts = GenerationOptions::default();
# let mut session = model.new_session(&opts);
# model.prefill(&mut session, "Hello", &mut None)?;
let bytes = model.export_session(&session)?;
let snap = model.import_snapshot_bytes(&bytes)?;
let restored = model.restore_session(snap, opts)?;
# Ok::<(), glint::error::GlintError>(())
```

`restore_session()` rebuilds the last decode state so the next sampled token matches a non-snapshotted run when the same seed/options are used.

---

## Constraints and Resume

Structured-output constraints live in the session, not in the model. When restoring:

- pass the same `GenerationOptions::constraint` you used originally
- pass the same LoRA adapter if the session was using one
- keep the same model file; snapshots are validated against model identity and dimensions before cache data is loaded

Constraint state is rebuilt from the stored generated suffix, so constrained generation can continue after restore.

---

## Notes

- `CacheFormat::Q8` snapshots preserve the Q8 KV cache format; Glint validates the cache format before import.
- The snapshot format is versioned. Older blobs may need re-export if the format changes.
- The session API is synchronous; higher-level async/server surfaces build on top of it.

### Importing untrusted snapshots

A snapshot blob is treated as fully untrusted input (it may cross the C FFI / WASM
boundary). `import_snapshot_bytes()` reads every length and offset field from the
header with overflow-checked, saturating arithmetic and clamps any file-supplied
count to the blob's actual byte length before allocating, so a hostile or corrupt
header cannot wrap a bounds check or trigger an unbounded allocation. Use
`peek_snapshot_metadata()` to read a blob's identity/dimensions before committing to
a full import. This importer is covered by the `snapshot_import` fuzz target.
