//! Constrained generation — token masking for structured outputs.
//!
//! [`TokenConstraint`] is the core trait: it computes a boolean mask over the
//! vocabulary at each decode step so the sampler only picks syntactically
//! valid tokens.  Masks are cached per parse state so the overhead of
//! character-level simulation is paid only once per unique state.
//!
//! # JSON Object mode
//!
//! [`JsonObjectConstraint`] guides the model to produce a valid JSON object:
//!
//! ```text
//! { "key": "value", "num": 42 }
//! ```
//!
//! It simulates every character of a candidate token through a simple JSON
//! state machine and rejects tokens that would produce ill-formed JSON.
//! Multi-character tokens are handled correctly: a token like `": "` is
//! allowed only if ALL three characters are legal in sequence.
//!
//! # Usage
//!
//! ```no_run
//! use glint::constrained::{JsonObjectConstraint, TokenConstraint, VocabIndex};
//! // Build the vocab index once from your tokenizer's vocabulary list.
//! let vocab: Vec<String> = /* tokenizer.vocab() */ vec![];
//! let vi  = VocabIndex::from_vocab(&vocab);
//! let mut constraint = JsonObjectConstraint::new(vi);
//! // Pass to Sampler::sample_constrained when decoding.
//! ```

use std::collections::HashMap;
use std::sync::Arc;

// ── TokenConstraint trait ─────────────────────────────────────────────────────

/// A stateful constraint that masks the vocabulary at each decode step.
///
/// Implement this to enforce any grammar during generation.  The constraint is
/// called once per token: `allowed_tokens` to compute the mask, then
/// `advance` to update internal state after the token is chosen.
pub trait TokenConstraint: Send + Sync {
    /// Return a boolean mask over the full vocabulary.
    ///
    /// `mask[i] == true` means token `i` is allowed in the current state.
    /// The mask length must equal the vocabulary size.
    ///
    /// `token_history` is the sequence of token ids generated so far; it can
    /// be used for context-sensitive constraints (e.g. "no repeated keys").
    ///
    /// The `VocabIndex` is provided for implementations that do not store their
    /// own vocab reference internally.  Most implementations will ignore it.
    fn allowed_tokens(&mut self, token_history: &[u32], vocab: &VocabIndex) -> Vec<bool>;

    /// Advance internal state after token `token_id` was chosen.
    fn advance(&mut self, token_id: u32);
}

// ── VocabIndex ────────────────────────────────────────────────────────────────

/// Pre-built lookup table from a tokenizer vocabulary.
///
/// Stores each token's decoded string and a map from leading character to
/// token ids so constraints can quickly build masks by character class.
pub struct VocabIndex {
    /// `strings[i]` is the decoded string for token id `i`.
    pub strings: Vec<String>,
    /// `char_to_ids[ch]` lists all token ids whose first decoded char is `ch`.
    pub char_to_ids: HashMap<char, Vec<u32>>,
}

impl VocabIndex {
    /// Build the index from a vocabulary list.
    ///
    /// `vocab[i]` should be the decoded string for token id `i`.  This is the
    /// same order as the tokenizer's internal vocab table.
    pub fn from_vocab(vocab: &[String]) -> Arc<Self> {
        let mut char_to_ids: HashMap<char, Vec<u32>> = HashMap::new();
        for (id, s) in vocab.iter().enumerate() {
            if let Some(ch) = s.chars().next() {
                char_to_ids.entry(ch).or_default().push(id as u32);
            }
        }
        Arc::new(Self {
            strings: vocab.to_vec(),
            char_to_ids,
        })
    }
}

pub mod gbnf;
pub mod grammar;
pub mod json_schema;

pub use gbnf::GbnfGrammar;
pub use grammar::{GrammarConstraint, JsonSchemaConstraint};
pub use json_schema::json_schema_to_gbnf;

// ── ConstraintSpec ────────────────────────────────────────────────────────────

/// Declarative constraint specification.
#[derive(Clone, Debug)]
pub enum ConstraintSpec {
    /// Force the model to produce a valid JSON object (`{…}`).
    JsonObject,
    /// Force the model to produce one of the given string literals.
    JsonEnum(Vec<String>),
    /// Force the model to produce JSON adhering to a JSON Schema.
    JsonSchema(serde_json::Value),
    /// Force the model to follow a custom GBNF grammar.
    Grammar(String),
}

// ── JSON parse state ──────────────────────────────────────────────────────────

/// Character-level JSON object parse state.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum JsonState {
    /// Expecting `{` to open the object.
    ExpectOpenBrace,
    /// Expecting `"` to start a key, or `}` if the object can be empty.
    ExpectKeyOrClose,
    /// Inside a quoted key string (after the opening `"`).
    InKey,
    /// After the closing `"` of a key; expecting `:`.
    ExpectColon,
    /// After `:`, expecting the start of a value.
    ExpectValue,
    /// Inside a quoted string value.
    InStringValue,
    /// Inside a number value (integer or float).
    InNumber,
    /// Inside a literal (true, false, null) — stores remaining chars.
    InLiteral(String),
    /// After a complete value; expecting `,` or `}`.
    AfterValue,
    /// Object closed; only EOS is valid.
    Done,
    /// Unrecoverable error — token must be rejected.
    Error,
}

impl JsonState {
    /// Advance one character through the JSON grammar.
    fn step(self, ch: char) -> Self {
        match self {
            Self::ExpectOpenBrace => {
                if ch == '{' {
                    Self::ExpectKeyOrClose
                } else {
                    Self::Error
                }
            }
            Self::ExpectKeyOrClose => {
                match ch {
                    '"' => Self::InKey,
                    '}' => Self::Done,
                    ' ' | '\t' | '\n' | '\r' => Self::ExpectKeyOrClose, // whitespace
                    _ => Self::Error,
                }
            }
            Self::InKey => {
                match ch {
                    '"' => Self::ExpectColon,
                    '\\' => Self::InKey,        // escape — next char is data
                    '\n' | '\r' => Self::Error, // raw newline in string
                    _ => Self::InKey,
                }
            }
            Self::ExpectColon => match ch {
                ':' => Self::ExpectValue,
                ' ' | '\t' => Self::ExpectColon,
                _ => Self::Error,
            },
            Self::ExpectValue => {
                match ch {
                    '"' => Self::InStringValue,
                    '-' | '0'..='9' => Self::InNumber,
                    't' => Self::InLiteral("rue".to_string()),
                    'f' => Self::InLiteral("alse".to_string()),
                    'n' => Self::InLiteral("ull".to_string()),
                    '[' | '{' => Self::AfterValue, // nested array/object — treat as opaque
                    ' ' | '\t' | '\n' | '\r' => Self::ExpectValue,
                    _ => Self::Error,
                }
            }
            Self::InStringValue => {
                match ch {
                    '"' => Self::AfterValue,
                    '\\' => Self::InStringValue, // escape
                    '\n' | '\r' => Self::Error,
                    _ => Self::InStringValue,
                }
            }
            Self::InNumber => match ch {
                '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => Self::InNumber,
                ',' => Self::ExpectKeyOrClose,
                '}' => Self::Done,
                ' ' | '\t' | '\n' | '\r' => Self::AfterValue,
                _ => Self::Error,
            },
            Self::InLiteral(mut remaining) => {
                if remaining.is_empty() {
                    // Literal complete; now expecting separator.
                    match ch {
                        ',' => Self::ExpectKeyOrClose,
                        '}' => Self::Done,
                        ' ' | '\t' | '\n' | '\r' => Self::AfterValue,
                        _ => Self::Error,
                    }
                } else {
                    let expected = remaining.remove(0);
                    if ch == expected {
                        Self::InLiteral(remaining)
                    } else {
                        Self::Error
                    }
                }
            }
            Self::AfterValue => match ch {
                ',' => Self::ExpectKeyOrClose,
                '}' => Self::Done,
                ' ' | '\t' | '\n' | '\r' => Self::AfterValue,
                _ => Self::Error,
            },
            Self::Done => Self::Error, // nothing valid after closing brace
            Self::Error => Self::Error,
        }
    }

    /// Apply every character of `s` through the state machine.
    ///
    /// Returns the final state, or `Error` if any intermediate step fails.
    fn step_str(self, s: &str) -> Self {
        let mut state = self;
        for ch in s.chars() {
            state = state.step(ch);
            if state == Self::Error {
                return Self::Error;
            }
        }
        state
    }

    /// True if this state represents a complete, well-formed JSON object.
    #[cfg(test)]
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Done)
    }
}

// ── JsonObjectConstraint ─────────────────────────────────────────────────────

/// Constrained sampler that forces valid JSON object output.
///
/// Builds a token mask for each decode step by simulating the JSON parse state
/// machine over each candidate token's decoded characters.  Masks are cached
/// per state so re-entry into the same parse state costs a map lookup.
pub struct JsonObjectConstraint {
    state: JsonState,
    vocab: Arc<VocabIndex>,
    mask_cache: HashMap<JsonState, Vec<bool>>,
}

impl JsonObjectConstraint {
    /// Create a new JSON object constraint.
    ///
    /// Call `VocabIndex::from_vocab` with the tokenizer's vocabulary to build
    /// the index, then pass it here.
    pub fn new(vocab: Arc<VocabIndex>) -> Self {
        Self {
            state: JsonState::ExpectOpenBrace,
            vocab,
            mask_cache: HashMap::new(),
        }
    }

    /// Build (or return cached) a boolean mask for the current state.
    fn build_mask(&mut self) -> &Vec<bool> {
        let state = self.state.clone();
        let strings = &self.vocab.strings;
        self.mask_cache.entry(state.clone()).or_insert_with(|| {
            strings
                .iter()
                .map(|s| {
                    if s.is_empty() {
                        return false;
                    }
                    // A token is allowed if applying all its characters leads to a
                    // non-Error state.  We do NOT require Done — the token may be a
                    // partial step (e.g. the `"` opening a string value is fine).
                    let next = state.clone().step_str(s);
                    next != JsonState::Error
                })
                .collect()
        })
    }
}

impl TokenConstraint for JsonObjectConstraint {
    fn allowed_tokens(&mut self, _token_history: &[u32], _vocab: &VocabIndex) -> Vec<bool> {
        self.build_mask().clone()
    }

    fn advance(&mut self, token_id: u32) {
        if let Some(s) = self.vocab.strings.get(token_id as usize) {
            self.state = self.state.clone().step_str(s);
        }
    }
}

// ── JsonEnumConstraint ────────────────────────────────────────────────────────

/// Forces the model to output exactly one of the given string literals.
///
/// Useful for classification tasks — constrains output to a known set of
/// categories, e.g. `["positive", "negative", "neutral"]`.
pub struct JsonEnumConstraint {
    candidates: Vec<String>,
    vocab: Arc<VocabIndex>,
    /// Characters produced so far.
    emitted: String,
}

impl JsonEnumConstraint {
    pub fn new(candidates: Vec<String>, vocab: Arc<VocabIndex>) -> Self {
        Self {
            candidates,
            vocab,
            emitted: String::new(),
        }
    }

    /// Find candidates that are still reachable.
    fn still_valid(&self) -> Vec<&str> {
        self.candidates
            .iter()
            .filter(|c| c.starts_with(&self.emitted as &str))
            .map(String::as_str)
            .collect()
    }
}

impl TokenConstraint for JsonEnumConstraint {
    fn allowed_tokens(&mut self, _token_history: &[u32], _vocab: &VocabIndex) -> Vec<bool> {
        let valid = self.still_valid();
        self.vocab
            .strings
            .iter()
            .map(|s| {
                let candidate = format!("{}{}", self.emitted, s);
                valid
                    .iter()
                    .any(|v| v.starts_with(&candidate as &str) || *v == &candidate as &str)
            })
            .collect()
    }

    fn advance(&mut self, token_id: u32) {
        if let Some(s) = self.vocab.strings.get(token_id as usize) {
            self.emitted.push_str(s);
        }
    }
}

/// Build a [`Box<dyn TokenConstraint>`] from a [`ConstraintSpec`].
pub fn build_constraint(spec: &ConstraintSpec, vocab: Arc<VocabIndex>) -> Box<dyn TokenConstraint> {
    match spec {
        ConstraintSpec::JsonObject => Box::new(JsonObjectConstraint::new(vocab)),
        ConstraintSpec::JsonEnum(opts) => Box::new(JsonEnumConstraint::new(opts.clone(), vocab)),
        ConstraintSpec::JsonSchema(schema) => {
            match JsonSchemaConstraint::from_json_schema(schema, Arc::clone(&vocab)) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    eprintln!("Warning: failed to compile JSON schema to GBNF ({e}), falling back to JsonObject");
                    Box::new(JsonObjectConstraint::new(vocab))
                }
            }
        }
        ConstraintSpec::Grammar(grammar_str) => {
            match GrammarConstraint::from_gbnf_str(grammar_str, Arc::clone(&vocab)) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    eprintln!("Warning: failed to parse GBNF grammar ({e}), falling back to JsonObject");
                    Box::new(JsonObjectConstraint::new(vocab))
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vocab() -> Arc<VocabIndex> {
        let vocab: Vec<String> = [
            "{", "}", "\"", ":", ",", "a", "b", "c", "1", "2", "true", "false", "null", " ", "\n",
            "key", "val", "hello", "world", "0", ".",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        VocabIndex::from_vocab(&vocab)
    }

    #[test]
    fn test_json_state_machine_valid_object() {
        // Simulate: { "key": "val" }
        let s = JsonState::ExpectOpenBrace;
        let s = s.step('{'); // -> ExpectKeyOrClose
        assert_ne!(s, JsonState::Error);
        let s = s.step('"'); // -> InKey
        assert_ne!(s, JsonState::Error);
        let s = s.step('k'); // -> InKey
        let s = s.step('e');
        let s = s.step('y');
        let s = s.step('"'); // -> ExpectColon
        assert_ne!(s, JsonState::Error);
        let s = s.step(':'); // -> ExpectValue
        let s = s.step('"'); // -> InStringValue
        let s = s.step('v');
        let s = s.step('a');
        let s = s.step('l');
        let s = s.step('"'); // -> AfterValue
        let s = s.step('}'); // -> Done
        assert!(s.is_terminal());
    }

    #[test]
    fn test_json_state_machine_rejects_plain_word() {
        let s = JsonState::ExpectOpenBrace;
        let s = s.step('h'); // error: must start with {
        assert_eq!(s, JsonState::Error);
    }

    #[test]
    fn test_json_state_number_value() {
        let s = JsonState::ExpectValue;
        let s = s.step_str("42");
        assert_eq!(s, JsonState::InNumber);
        let s = s.step('}');
        assert!(s.is_terminal());
    }

    #[test]
    fn test_json_state_literal_true() {
        let s = JsonState::ExpectValue;
        let s = s.step_str("true");
        // InLiteral("") → after full match, still need separator
        assert_ne!(s, JsonState::Error);
        let s = s.step('}');
        assert!(s.is_terminal());
    }

    #[test]
    fn test_constraint_allows_open_brace() {
        let vocab = tiny_vocab();
        let mut c = JsonObjectConstraint::new(Arc::clone(&vocab));
        let mask = c.allowed_tokens(&[], &vocab);
        // Token 0 is "{" — should be allowed
        assert!(mask[0], "{{ should be allowed in ExpectOpenBrace state");
        // Token 1 is "}" — should NOT be allowed at start
        assert!(!mask[1], "}} should not be allowed before opening {{");
    }

    #[test]
    fn test_constraint_advance_after_open_brace() {
        let vocab = tiny_vocab();
        let mut c = JsonObjectConstraint::new(Arc::clone(&vocab));
        c.advance(0); // emit "{"
        let mask = c.allowed_tokens(&[], &vocab);
        // Now in ExpectKeyOrClose — token 2 ('"') should be allowed
        assert!(mask[2], "quote should be allowed to start a key");
    }

    #[test]
    fn test_constraint_mask_cache_reused() {
        let vocab = tiny_vocab();
        let mut c = JsonObjectConstraint::new(Arc::clone(&vocab));
        let m1 = c.allowed_tokens(&[], &vocab).clone();
        let m2 = c.allowed_tokens(&[], &vocab).clone();
        assert_eq!(m1, m2);
        // Cache should have exactly one entry (ExpectOpenBrace)
        assert_eq!(c.mask_cache.len(), 1);
    }

    #[test]
    fn test_json_enum_constraint() {
        let vocab = Arc::new(VocabIndex::from_vocab(&[
            "yes".to_string(),
            "no".to_string(),
            "maybe".to_string(),
            "ye".to_string(),
            "s".to_string(),
        ]));
        let mut c = JsonEnumConstraint::new(
            vec!["yes".to_string(), "no".to_string()],
            Arc::clone(&vocab),
        );
        let mask = c.allowed_tokens(&[], &vocab);
        // "yes" (0) and "ye" (3) and "no" (1) should be allowed; "maybe" should not
        assert!(mask[0], "\"yes\" should be allowed");
        assert!(mask[1], "\"no\" should be allowed");
        assert!(!mask[2], "\"maybe\" should not be allowed");
    }
}
