# Phase 1.4 — Tokenizer: Technical Guide

This document explains the BPE tokenizer implementation.

**Source files:**

- [tokenizer.rs](../src/model/tokenizer.rs) — `Tokenizer` struct, encode/decode, GPT-2 byte mapping

---

## Overview

The tokenizer converts between text and token IDs:

- **Encode:** `"Hello, world!"` → `[15496, 11, 995, 0]`
- **Decode:** `[15496, 11, 995, 0]` → `"Hello, world!"`

SmolLM uses a GPT-2 style **Byte-Pair Encoding (BPE)** tokenizer with the vocabulary and merge rules stored directly in GGUF metadata — no external tokenizer files needed.

---

## BPE Encoding Algorithm

See [Tokenizer::encode](../src/model/tokenizer.rs#L83-L142).

### Step 1: Bytes to Initial Tokens

Each byte of the input text is mapped to a single-character token using the GPT-2 byte-to-unicode mapping ([gpt2_byte_to_char](../src/model/tokenizer.rs#L208-L225)). This avoids control characters in the vocabulary.

Example: `"Hi"` → bytes `[72, 105]` → tokens `["H", "i"]`

### Step 2: Iterative Merging

Repeatedly find the adjacent pair of tokens with the **lowest merge rank** and merge them into one token:

```
["H", "i"]          → merge rank of ("H","i") is found
["Hi"]              → done (only one token left)
```

For longer text, many rounds of merging occur, collapsing common byte sequences into single tokens. The merge rules are ordered by frequency in the training corpus — the most common pairs were merged first.

### Step 3: Token ID Lookup

Each final merged piece is looked up in the vocabulary hash map to get its integer ID. Unknown pieces fall back to token 0.

---

## GPT-2 Byte-to-Unicode Mapping

GPT-2's BPE operates on bytes, but the vocabulary contains unicode strings. There's a fixed mapping:

- Printable ASCII (33-126): identity (`!` → `!`, `A` → `A`)
- Extended Latin (161-172, 174-255): identity (`¡` → `¡`)
- Control chars / space / delete (0-32, 127-160, 173): mapped to U+0100+ range

This is implemented in [gpt2_byte_to_char](../src/model/tokenizer.rs#L208-L225) and reversed by [gpt2_char_to_byte](../src/model/tokenizer.rs#L228-L240).

The roundtrip test verifies all 256 bytes map correctly.

---

## Decoding

[Tokenizer::decode](../src/model/tokenizer.rs#L144-L158) reverses the process:

1. Look up each token ID in the vocabulary → string piece
2. Convert each piece's characters back to raw bytes via `gpt2_char_to_byte`
3. Assemble bytes into a UTF-8 string (with lossy fallback)

Special hex tokens like `<0x0A>` (newline) are handled separately.

---

## Special Tokens

| Token | ID  | Purpose                                          |
| ----- | --- | ------------------------------------------------ |
| BOS   | 1   | Beginning of sequence — prepended to prompts     |
| EOS   | 2   | End of sequence — signals generation should stop |
| UNK   | 0   | Unknown token — fallback for unrecognized pieces |

Read from GGUF metadata keys `tokenizer.ggml.bos_token_id` and `tokenizer.ggml.eos_token_id`.
