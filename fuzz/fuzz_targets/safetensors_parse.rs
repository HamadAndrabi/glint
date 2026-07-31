#![no_main]
//! Fuzz the SafeTensors parser against arbitrary bytes.
//!
//! `SafeTensorsFile::from_bytes` parses fully untrusted input: the 8-byte
//! header length, the JSON header, and every tensor's shape and byte range are
//! all attacker-controlled. The contract asserted here matches the GGUF
//! target's: it must return `Ok` or `Err` — never panic, abort, or OOM — no
//! matter what bytes it is handed.
//!
//! Run: `cargo fuzz run safetensors_parse`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = glint::model::safetensors::SafeTensorsFile::from_bytes(data.to_vec()) {
        // A file that parsed must also be readable: every descriptor it kept
        // has to yield an in-bounds slice of exactly the promised size.
        for view in file.tensor_infos() {
            let bytes = file
                .tensor_bytes(&view.name)
                .expect("a parsed descriptor must be readable");
            assert_eq!(bytes.len(), view.nbytes());
        }
    }
});
