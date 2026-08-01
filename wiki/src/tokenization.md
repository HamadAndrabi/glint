# Tokenization

Glint implements BPE (Byte Pair Encoding) tokenization from scratch, reading vocabulary and merge rules directly from GGUF metadata. No external tokenizer library is needed.

When a model is loaded from a HuggingFace SafeTensors directory instead, the
same BPE structures are built from its `tokenizer.json` (byte-level BPE, the
LLaMA-3/SmolLM/Qwen style). SentencePiece-derived `tokenizer.json` files
(Metaspace `▁` pretokenizers, e.g. LLaMA-2 / Mistral-v0.1 HF repos) are
rejected with a clear error rather than mis-encoded — see
[SafeTensors & HF Models](./safetensors.md).

Source: `src/model/tokenizer.rs`

---

## What is BPE?

BPE builds a vocabulary of subword units by iteratively merging the most frequent adjacent pairs of tokens. Starting from individual bytes, the algorithm:

1. Represent the input as individual bytes
2. Find the most frequent adjacent pair across the corpus
3. Merge that pair into a new token
4. Repeat until vocabulary size is reached

The result: common words become single tokens, rare words are split into meaningful subpieces.

---

## Tokenizer Data in GGUF

The GGUF file embeds all tokenizer data in metadata:

| Key | Content |
|-----|---------|
| `tokenizer.ggml.model` | Model type: `"gpt2"`, `"llama"`, etc. |
| `tokenizer.ggml.tokens` | Array of token strings (the vocabulary) |
| `tokenizer.ggml.scores` | Per-token scores (for SentencePiece variants) |
| `tokenizer.ggml.token_type` | Token type flags (normal, special, BOS, EOS, etc.) |
| `tokenizer.ggml.merges` | Array of merge rules as `"a b"` strings |
| `tokenizer.ggml.bos_token_id` | Beginning-of-sequence token ID |
| `tokenizer.ggml.eos_token_id` | End-of-sequence token ID |

---

## The GPT-2 Byte-to-Unicode Mapping

GPT-2 (and models derived from it, including LLaMA) uses a byte-level encoding that maps all 256 byte values to printable Unicode characters. This avoids issues with:
- Control characters (bytes 0–31 and 127)
- Non-UTF-8 byte sequences
- Whitespace handling

The mapping assigns printable ASCII characters to themselves, and uses a range of Unicode characters (`Ā`, `ā`, `Ă`, ...) for the remaining bytes.

```
byte 0x20 (space)  →  'Ġ'   (Unicode 0x0120)
byte 0x00 (NUL)    →  'Ā'   (Unicode 0x0100)
byte 0x41 ('A')    →  'A'   (unchanged)
```

During encoding, the input string is first converted to bytes, then each byte is mapped through this table before BPE merges are applied. During decoding, the reverse mapping converts back to bytes, which are then interpreted as UTF-8.

---

## Encoding Algorithm

```
encode(text: &str) -> Vec<u32>:

1. Convert text bytes to unicode characters using byte-to-unicode map
2. Split into initial tokens (characters or Unicode code points)
3. Repeatedly apply merges:
   - Find the merge rule with highest priority that matches
     an adjacent pair in the current token sequence
   - Apply the merge (replace pair with combined token)
   - Repeat until no more merges apply
4. Convert final token strings to IDs using the vocabulary
```

Pre-tokenization splits on whitespace boundaries, adding the GPT-2 `Ġ` prefix character to tokens that follow a space (marking word boundaries).

---

## Decoding

Decoding is simpler: convert token IDs back to strings via vocabulary lookup, apply the inverse byte-to-unicode mapping, and concatenate.

```rust
pub fn decode(&self, token_ids: &[u32]) -> String {
    let mut bytes = Vec::new();
    for &id in token_ids {
        let token_str = &self.vocab[id as usize];
        for ch in token_str.chars() {
            bytes.push(self.unicode_to_byte[&ch]);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
```

---

## Special Tokens

| Token | Description | Default ID |
|-------|-------------|-----------|
| BOS | Beginning of sequence. Prepended to every prompt. | 1 |
| EOS | End of sequence. Generation stops when this is sampled. | 2 |
| UNK | Unknown token. Used when a byte sequence can't be mapped. | 0 |

The actual IDs are read from GGUF metadata and can vary by model. The `Tokenizer` struct exposes:
- `tokenizer.bos_token_id`
- `tokenizer.eos_token_id`
- `tokenizer.vocab_size()`

---

## Chat Templates

Chat models wrap messages in a specific prompt format. Glint detects the template from `tokenizer.chat_template` GGUF metadata and formats messages accordingly.

Source: `src/model/chat_template.rs`

Templates handled:
- **LLaMA-3** — `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
- **Mistral / ChatML** — `<s>[INST] ... [/INST]`
- **Generic** — Simple `User:` / `Assistant:` prefix format

The `glint chat` command applies the template automatically. The `/v1/chat/completions` endpoint also applies the chat template before tokenization.
