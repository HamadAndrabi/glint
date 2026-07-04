#![no_main]
//! Fuzz the GGUF parser against arbitrary bytes.
//!
//! `GgufModel::from_bytes` parses fully untrusted input (model files, or a
//! browser-supplied `ArrayBuffer` on WASM). The contract we assert here is
//! simple: it must return `Ok` or `Err` — never panic, abort, or OOM — no
//! matter what bytes it is handed.
//!
//! Run: `cargo fuzz run gguf_parse`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Ignore the result; we only care that parsing terminates without a panic.
    let _ = glint::model::gguf::GgufModel::from_bytes(data.to_vec());
});
