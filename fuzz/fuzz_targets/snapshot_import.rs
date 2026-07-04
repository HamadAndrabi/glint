#![no_main]
//! Fuzz the KV-snapshot importer against arbitrary bytes.
//!
//! This mirrors the untrusted C-FFI `glint_snapshot_deserialize` path: read the
//! header metadata from the blob itself, then import against it. Both steps must
//! return `Ok`/`Err` on any input — never panic or attempt an unbounded
//! allocation from a hostile count/length field.
//!
//! Run: `cargo fuzz run snapshot_import`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    use glint::session::snapshot::{import_snapshot, peek_snapshot_metadata};

    // The FFI deserialize path takes the "expected" metadata from the blob, so
    // the only real defenses are the parser's bounds/overflow checks.
    if let Ok(meta) = peek_snapshot_metadata(data) {
        let _ = import_snapshot(data, &meta);
    }
});
