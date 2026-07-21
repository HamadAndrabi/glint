//! BPE tokenizer — encodes text to token IDs and decodes back.
//!
//! Reads vocabulary and merge rules directly from GGUF metadata.

use std::collections::HashMap;

use crate::error::GlintError;
use crate::model::gguf::GgufModel;

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
}
