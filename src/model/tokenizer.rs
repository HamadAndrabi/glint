//! BPE tokenizer — encodes text to token IDs and decodes back.
//!
//! Reads vocabulary and merge rules directly from GGUF metadata, or from a
//! HuggingFace `tokenizer.json` (see [`Tokenizer::from_hf_json`]).

use std::collections::HashMap;

use crate::error::GlintError;
use crate::model::gguf::GgufModel;

/// Upper bound on the token id space accepted from a `tokenizer.json`.
///
/// The vocabulary is turned into a dense `Vec<String>` indexed by id, so a
/// hostile file claiming an id of `u32::MAX` would otherwise ask for a
/// multi-gigabyte allocation. No real vocabulary comes close to this.
const MAX_VOCAB_SIZE: usize = 4_000_000;

/// A byte-pair encoding tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// String piece → token ID
    token_to_id: HashMap<String, u32>,
    /// Ordered merge rules: (piece_a, piece_b) → merged piece
    merges: Vec<(String, String)>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// Whether prompts should be prefixed with BOS
    /// (`tokenizer.ggml.add_bos_token`; true when the model omits the key).
    ///
    /// llama.cpp honours this flag, so Glint must too or greedy decode
    /// diverges from the reference on models like SmolLM2 that set it false.
    pub add_bos: bool,
}

impl Tokenizer {
    /// Load tokenizer from GGUF model metadata.
    pub fn from_gguf(model: &GgufModel) -> Result<Self, GlintError> {
        // Extract vocabulary
        let tokens_meta = model
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or(GlintError::MissingVocabulary)?;

        let vocab: Vec<String> = tokens_meta
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Extract merge rules
        let merges = if let Some(merges_meta) = model
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            merges_meta
                .iter()
                .filter_map(|v| {
                    let s = v.as_str()?;
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next()?.to_string();
                    let b = parts.next()?.to_string();
                    Some((a, b))
                })
                .collect()
        } else {
            Vec::new()
        };

        let bos_token_id = model
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);

        let eos_token_id = model
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        let add_bos = model
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            add_bos,
        })
    }

    /// Load a tokenizer from a HuggingFace `tokenizer.json`.
    ///
    /// Only byte-level BPE is supported — the LLaMA-3 / SmolLM / Qwen family,
    /// and any other tokenizer whose pieces are GPT-2 byte characters. That is
    /// the same assumption [`Tokenizer::encode`] already makes for GGUF
    /// vocabularies, so a SentencePiece-derived tokenizer (Metaspace `▁`
    /// pieces, no `ByteLevel` stage) is refused rather than silently
    /// mis-encoding every prompt.
    ///
    /// * `tokenizer_json` — contents of `tokenizer.json` (required).
    /// * `tokenizer_config_json` — contents of `tokenizer_config.json`, which
    ///   carries `add_bos_token` and the BOS/EOS token *strings*.
    /// * `bos_from_config` / `eos_from_config` — ids from `config.json`, which
    ///   take precedence over the strings resolved from the tokenizer config.
    pub fn from_hf_json(
        tokenizer_json: &str,
        tokenizer_config_json: Option<&str>,
        bos_from_config: Option<u32>,
        eos_from_config: Option<u32>,
    ) -> Result<Self, GlintError> {
        let invalid = |detail: String| GlintError::HfInvalidJson {
            file: "tokenizer.json".to_string(),
            detail,
        };

        let root: serde_json::Value =
            serde_json::from_str(tokenizer_json).map_err(|e| invalid(e.to_string()))?;
        let root = root
            .as_object()
            .ok_or_else(|| invalid("top level value is not a JSON object".to_string()))?;

        let model = root
            .get("model")
            .and_then(|v| v.as_object())
            .ok_or_else(|| invalid("missing 'model' object".to_string()))?;

        match model.get("type").and_then(|v| v.as_str()) {
            None | Some("BPE") => {}
            Some(other) => {
                return Err(GlintError::HfUnsupported(format!(
                    "tokenizer model type '{other}' — Glint implements byte-level BPE"
                )))
            }
        }
        if !has_byte_level_stage(root) {
            return Err(GlintError::HfUnsupported(
                "tokenizer.json has no ByteLevel pre-tokenizer or decoder — only \
                 byte-level BPE vocabularies (LLaMA-3, SmolLM, Qwen, GPT-2 style) \
                 can be encoded correctly; convert the model to GGUF instead"
                    .to_string(),
            ));
        }

        // ── Vocabulary: `model.vocab` plus any `added_tokens` ────────────────
        let vocab_json = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| invalid("missing 'model.vocab' object".to_string()))?;

        let mut pairs: Vec<(String, u32)> = Vec::with_capacity(vocab_json.len());
        let mut push = |piece: &str, id: u64| -> Result<(), GlintError> {
            let id = usize::try_from(id).ok().filter(|&i| i < MAX_VOCAB_SIZE);
            let id = id.ok_or_else(|| {
                invalid(format!(
                    "token '{piece}' has an id beyond the {MAX_VOCAB_SIZE}-token limit"
                ))
            })?;
            pairs.push((piece.to_string(), id as u32));
            Ok(())
        };

        for (piece, id) in vocab_json {
            let id = id
                .as_u64()
                .ok_or_else(|| invalid(format!("vocab entry '{piece}' is not an integer id")))?;
            push(piece, id)?;
        }
        // Added tokens (specials such as <|endoftext|>) are listed separately
        // and are usually — but not always — mirrored in `model.vocab`.
        if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for entry in added {
                let content = entry
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| invalid("added_tokens entry has no 'content'".to_string()))?;
                let id = entry
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| invalid("added_tokens entry has no 'id'".to_string()))?;
                push(content, id)?;
            }
        }

        if pairs.is_empty() {
            return Err(GlintError::MissingVocabulary);
        }

        let size = pairs.iter().map(|(_, id)| *id as usize).max().unwrap() + 1;
        let mut vocab = vec![String::new(); size];
        let mut token_to_id = HashMap::with_capacity(pairs.len());
        for (piece, id) in pairs {
            vocab[id as usize] = piece.clone();
            token_to_id.insert(piece, id);
        }

        // ── Merge rules ──────────────────────────────────────────────────────
        // Written either as "a b" strings (tokenizers < 0.20) or as ["a", "b"]
        // pairs (>= 0.20). Byte-level pieces never contain a space — a literal
        // space byte is the piece "Ġ" — so splitting on the first space is
        // unambiguous.
        let mut merges = Vec::new();
        if let Some(list) = model.get("merges").and_then(|v| v.as_array()) {
            merges.reserve(list.len());
            for entry in list {
                if let Some(s) = entry.as_str() {
                    let mut parts = s.splitn(2, ' ');
                    match (parts.next(), parts.next()) {
                        (Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                        _ => return Err(invalid(format!("merge rule '{s}' is not a pair"))),
                    }
                } else if let Some(pair) = entry.as_array().filter(|p| p.len() == 2) {
                    match (pair[0].as_str(), pair[1].as_str()) {
                        (Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                        _ => return Err(invalid("merge rule pair is not a string".to_string())),
                    }
                } else {
                    return Err(invalid("merge rule is neither a string nor a pair".into()));
                }
            }
        }

        // ── Special tokens ───────────────────────────────────────────────────
        let tok_config: Option<serde_json::Value> =
            tokenizer_config_json.and_then(|text| serde_json::from_str(text).ok());
        let special_id = |key: &str| -> Option<u32> {
            let value = tok_config.as_ref()?.get(key)?;
            let content = value
                .as_str()
                .or_else(|| value.get("content").and_then(|c| c.as_str()))?;
            token_to_id.get(content).copied()
        };

        let eos = eos_from_config
            .or_else(|| special_id("eos_token"))
            .ok_or_else(|| {
                GlintError::HfUnsupported(
                    "no EOS token id in config.json or tokenizer_config.json — \
                     generation would never stop"
                        .to_string(),
                )
            })?;
        let bos = bos_from_config.or_else(|| special_id("bos_token"));

        let add_bos = tok_config
            .as_ref()
            .and_then(|c| c.get("add_bos_token"))
            .and_then(|v| v.as_bool())
            .unwrap_or(bos.is_some());
        if add_bos && bos.is_none() {
            return Err(GlintError::HfUnsupported(
                "add_bos_token is set but no BOS token id could be resolved".to_string(),
            ));
        }

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            // Never prepended while `add_bos` is false, so falling back to EOS
            // here is inert; it keeps the field non-optional for the GGUF path.
            bos_token_id: bos.unwrap_or(eos),
            eos_token_id: eos,
            add_bos,
        })
    }

    /// Encode a prompt for generation: BPE-encode and prepend BOS if (and
    /// only if) the model's metadata asks for it.
    ///
    /// Every generation entry point (CLI, server, runtime API, bindings)
    /// should tokenize prompts through this, not raw [`encode`](Self::encode),
    /// so BOS handling stays consistent with llama.cpp.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut tokens = self.encode(text);
        if self.add_bos {
            tokens.insert(0, self.bos_token_id);
        }
        tokens
    }

    /// Encode a string into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with individual bytes as initial tokens
        // GPT-2 BPE operates on bytes represented as unicode chars in the vocab
        let mut pieces: Vec<String> = text.bytes().map(|b| self.byte_to_token(b)).collect();

        // Build a lookup for merge priority (lower index = higher priority)
        let merge_rank: HashMap<(&str, &str), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.as_str(), b.as_str()), i))
            .collect();

        // Repeatedly apply the highest-priority merge
        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the pair with the lowest merge rank
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;

            for i in 0..pieces.len() - 1 {
                if let Some(&rank) = merge_rank.get(&(pieces[i].as_str(), pieces[i + 1].as_str())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break; // No more merges possible
            }

            // Apply the merge
            let merged = format!("{}{}", pieces[best_idx], pieces[best_idx + 1]);
            pieces[best_idx] = merged;
            pieces.remove(best_idx + 1);
        }

        // Convert pieces to token IDs
        pieces
            .iter()
            .map(|p| {
                self.token_to_id
                    .get(p)
                    .or_else(|| self.token_to_id.get("<unk>"))
                    .copied()
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if id as usize >= self.vocab.len() {
                continue;
            }
            let piece = &self.vocab[id as usize];
            // Convert token piece back to bytes
            for b in self.token_to_bytes(piece) {
                bytes.push(b);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> &str {
        if (id as usize) < self.vocab.len() {
            &self.vocab[id as usize]
        } else {
            "<unk>"
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Convert a byte to its GPT-2 BPE token representation.
    ///
    /// GPT-2 BPE uses a mapping where bytes 0-255 are represented as
    /// specific unicode characters to avoid control characters.
    fn byte_to_token(&self, byte: u8) -> String {
        let ch = gpt2_byte_to_char(byte);
        let s = ch.to_string();
        // If this single-char token exists in vocab, use it
        if self.token_to_id.contains_key(&s) {
            s
        } else {
            // Fallback to hex representation
            format!("<0x{:02X}>", byte)
        }
    }

    /// Convert a token piece back to raw bytes.
    fn token_to_bytes(&self, piece: &str) -> Vec<u8> {
        // Check for hex byte tokens like <0x0A>
        if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
            if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                return vec![byte];
            }
        }

        // Convert GPT-2 unicode chars back to bytes
        piece.chars().map(gpt2_char_to_byte).collect()
    }

    /// Minimal tokenizer for unit tests — no merge rules, identity vocab.
    #[cfg(test)]
    pub(crate) fn bare_for_test(vocab_size: usize, bos_token_id: u32, eos_token_id: u32) -> Self {
        let vocab: Vec<String> = (0..vocab_size).map(|i| format!("tok{}", i)).collect();
        let token_to_id = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Self {
            vocab,
            token_to_id,
            merges: Vec::new(),
            bos_token_id,
            eos_token_id,
            add_bos: true,
        }
    }
}

/// Does this `tokenizer.json` run a `ByteLevel` stage?
///
/// The pre-tokenizer and decoder may each be a single component or a
/// `Sequence` of them, so both subtrees are scanned for a `"type":"ByteLevel"`
/// node at any depth. Its presence is what makes the GPT-2 byte↔char mapping
/// in [`gpt2_byte_to_char`] the right way to turn text into vocabulary pieces.
fn has_byte_level_stage(root: &serde_json::Map<String, serde_json::Value>) -> bool {
    fn scan(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(map) => {
                if map.get("type").and_then(|t| t.as_str()) == Some("ByteLevel") {
                    return true;
                }
                map.values().any(scan)
            }
            serde_json::Value::Array(items) => items.iter().any(scan),
            _ => false,
        }
    }
    ["pre_tokenizer", "decoder"]
        .iter()
        .filter_map(|key| root.get(*key))
        .any(scan)
}

/// GPT-2 BPE byte-to-unicode mapping.
///
/// Maps bytes to unicode codepoints, avoiding control characters.
/// Printable ASCII bytes map to themselves; others get shifted to
/// the range starting at U+0100.
fn gpt2_byte_to_char(byte: u8) -> char {
    let b = byte as u32;
    let cp = match b {
        // Printable ASCII range (except space/delete edge cases)
        33..=126 => b,  // '!' to '~'
        161..=172 => b, // '¡' to '¬'
        174..=255 => b, // '®' to 'ÿ'
        // Everything else gets mapped to U+0100+
        _ => 256 + b - b.min(32), // offset to avoid collisions
    };
    // For the special cases, use a lookup
    match byte {
        0..=32 => char::from_u32(256 + byte as u32).unwrap(),
        127..=160 => char::from_u32(256 + 33 + (byte as u32 - 127)).unwrap(),
        173 => char::from_u32(256 + 33 + 34).unwrap(),
        _ => char::from_u32(cp).unwrap(),
    }
}

/// Reverse of gpt2_byte_to_char.
fn gpt2_char_to_byte(ch: char) -> u8 {
    let cp = ch as u32;
    match cp {
        33..=126 => cp as u8,
        161..=172 => cp as u8,
        174..=255 => cp as u8,
        // Reverse the special mappings
        256..=288 => (cp - 256) as u8,       // 0..=32
        289..=322 => (127 + cp - 289) as u8, // 127..=160
        323 => 173u8,
        _ => b'?', // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_byte_roundtrip() {
        // Every byte should roundtrip through the mapping
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            let back = gpt2_char_to_byte(ch);
            assert_eq!(
                b, back,
                "Byte {} -> char {:?} (U+{:04X}) -> {}",
                b, ch, ch as u32, back
            );
        }
    }

    #[test]
    fn test_ascii_identity() {
        // Printable ASCII should map to itself
        for b in 33..=126u8 {
            let ch = gpt2_byte_to_char(b);
            assert_eq!(ch, b as char);
        }
    }

    // ── HuggingFace tokenizer.json ───────────────────────────────────────────

    /// A minimal byte-level BPE `tokenizer.json`: every byte of "hi there"
    /// as a single-char piece, plus merges that build "hi" and "Ġthere".
    fn hf_tokenizer_json() -> String {
        // Byte-level pieces: printable ASCII maps to itself, space maps to 'Ġ'.
        let letters = [
            "h", "i", "Ġ", "t", "e", "r", "Ġt", "Ġth", "Ġthe", "Ġther", "Ġthere", "hi",
        ];
        let mut vocab = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        vocab.extend(letters.iter().map(|s| s.to_string()));
        let entries: Vec<String> = vocab
            .iter()
            .enumerate()
            .map(|(i, piece)| format!("\"{piece}\":{i}"))
            .collect();
        format!(
            r#"{{
                "added_tokens": [
                    {{"id": 0, "content": "<unk>", "special": true}},
                    {{"id": 1, "content": "<s>", "special": true}},
                    {{"id": 2, "content": "</s>", "special": true}}
                ],
                "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
                "decoder": {{"type": "ByteLevel"}},
                "model": {{
                    "type": "BPE",
                    "vocab": {{{}}},
                    "merges": ["h i", "Ġ t", "Ġt h", "Ġth e", "Ġthe r", "Ġther e"]
                }}
            }}"#,
            entries.join(",")
        )
    }

    fn hf_tokenizer_config_json() -> &'static str {
        r#"{
            "add_bos_token": true,
            "bos_token": {"content": "<s>"},
            "eos_token": "</s>",
            "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
        }"#
    }

    #[test]
    fn test_from_hf_json_encode_decode_roundtrip() {
        let tok = Tokenizer::from_hf_json(&hf_tokenizer_json(), None, Some(1), Some(2)).unwrap();
        assert_eq!(tok.bos_token_id, 1);
        assert_eq!(tok.eos_token_id, 2);

        let ids = tok.encode("hi there");
        // The merge chain collapses the string to exactly two pieces.
        assert_eq!(ids.len(), 2, "got {ids:?}");
        assert_eq!(tok.decode_token(ids[0]), "hi");
        assert_eq!(tok.decode_token(ids[1]), "Ġthere");
        assert_eq!(tok.decode(&ids), "hi there");
    }

    #[test]
    fn test_from_hf_json_resolves_specials_from_tokenizer_config() {
        let tok = Tokenizer::from_hf_json(
            &hf_tokenizer_json(),
            Some(hf_tokenizer_config_json()),
            None,
            None,
        )
        .unwrap();
        assert_eq!(tok.bos_token_id, 1);
        assert_eq!(tok.eos_token_id, 2);
        assert!(tok.add_bos);
        assert_eq!(tok.encode_prompt("hi")[0], 1);
    }

    #[test]
    fn test_from_hf_json_accepts_merge_pairs_array() {
        let json = hf_tokenizer_json().replace(
            r#"["h i", "Ġ t", "Ġt h", "Ġth e", "Ġthe r", "Ġther e"]"#,
            r#"[["h","i"],["Ġ","t"],["Ġt","h"],["Ġth","e"],["Ġthe","r"],["Ġther","e"]]"#,
        );
        let tok = Tokenizer::from_hf_json(&json, None, Some(1), Some(2)).unwrap();
        assert_eq!(tok.encode("hi there").len(), 2);
    }

    #[test]
    fn test_from_hf_json_vocab_is_dense_and_indexed_by_id() {
        let tok = Tokenizer::from_hf_json(&hf_tokenizer_json(), None, Some(1), Some(2)).unwrap();
        assert_eq!(tok.vocab_size(), 15);
        assert_eq!(tok.decode_token(0), "<unk>");
        assert_eq!(tok.decode_token(2), "</s>");
    }

    #[test]
    fn test_from_hf_json_rejects_non_byte_level_tokenizer() {
        let json = hf_tokenizer_json()
            .replace(
                r#""pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}"#,
                r#""pre_tokenizer": {"type": "Metaspace"}"#,
            )
            .replace(
                r#""decoder": {"type": "ByteLevel"}"#,
                r#""decoder": {"type": "Metaspace"}"#,
            );
        let err = Tokenizer::from_hf_json(&json, None, Some(1), Some(2))
            .err()
            .unwrap();
        assert!(err.to_string().contains("ByteLevel"), "got: {err}");
    }

    #[test]
    fn test_from_hf_json_rejects_non_bpe_model() {
        let json = hf_tokenizer_json().replace(r#""type": "BPE""#, r#""type": "Unigram""#);
        let err = Tokenizer::from_hf_json(&json, None, Some(1), Some(2))
            .err()
            .unwrap();
        assert!(err.to_string().contains("Unigram"), "got: {err}");
    }

    #[test]
    fn test_from_hf_json_requires_an_eos_token() {
        let err = Tokenizer::from_hf_json(&hf_tokenizer_json(), None, None, None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("EOS"), "got: {err}");
    }

    #[test]
    fn test_from_hf_json_rejects_malformed_input() {
        assert!(Tokenizer::from_hf_json("not json", None, Some(1), Some(2)).is_err());
        assert!(Tokenizer::from_hf_json("{}", None, Some(1), Some(2)).is_err());
        // A vocab id far beyond any real tokenizer must not be allocated for.
        let huge = r#"{
            "pre_tokenizer": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": {"a": 4294967295}, "merges": []}
        }"#;
        assert!(Tokenizer::from_hf_json(huge, None, Some(1), Some(2)).is_err());
    }
}
