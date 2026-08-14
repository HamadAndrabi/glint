//! Grammar and JSON Schema constrained generation implementations.

use std::sync::Arc;

use super::gbnf::{GbnfGrammar, GbnfMatcher};
use super::json_schema::json_schema_to_gbnf;
use super::{TokenConstraint, VocabIndex};

/// A stateful constraint that enforces arbitrary GBNF context-free grammars during generation.
pub struct GrammarConstraint {
    matcher: GbnfMatcher,
    vocab: Arc<VocabIndex>,
}

impl GrammarConstraint {
    /// Create a new grammar constraint from a GBNF grammar string.
    pub fn from_gbnf_str(grammar_str: &str, vocab: Arc<VocabIndex>) -> Result<Self, String> {
        let grammar = GbnfGrammar::from_str(grammar_str)?;
        let matcher = GbnfMatcher::new(Arc::new(grammar));
        Ok(Self { matcher, vocab })
    }

    /// Create from a parsed GbnfGrammar.
    pub fn new(grammar: Arc<GbnfGrammar>, vocab: Arc<VocabIndex>) -> Self {
        let matcher = GbnfMatcher::new(grammar);
        Self { matcher, vocab }
    }
}

impl TokenConstraint for GrammarConstraint {
    fn allowed_tokens(&mut self, _token_history: &[u32], _vocab: &VocabIndex) -> Vec<bool> {
        self.matcher.allowed_mask(&self.vocab, None)
    }

    fn advance(&mut self, token_id: u32) {
        if let Some(s) = self.vocab.strings.get(token_id as usize) {
            self.matcher.advance_str(s);
        }
    }
}

/// A stateful constraint that compiles a JSON Schema and enforces it during decode.
pub struct JsonSchemaConstraint {
    inner: GrammarConstraint,
}

impl JsonSchemaConstraint {
    /// Create a new JSON schema constraint.
    pub fn from_json_schema(
        schema: &serde_json::Value,
        vocab: Arc<VocabIndex>,
    ) -> Result<Self, String> {
        let gbnf = json_schema_to_gbnf(schema)?;
        let inner = GrammarConstraint::from_gbnf_str(&gbnf, vocab)?;
        Ok(Self { inner })
    }
}

impl TokenConstraint for JsonSchemaConstraint {
    fn allowed_tokens(&mut self, token_history: &[u32], vocab: &VocabIndex) -> Vec<bool> {
        self.inner.allowed_tokens(token_history, vocab)
    }

    fn advance(&mut self, token_id: u32) {
        self.inner.advance(token_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_vocab() -> Arc<VocabIndex> {
        let tokens: Vec<String> = vec![
            "{".into(),
            "}".into(),
            ":".into(),
            ",".into(),
            "\"".into(),
            "name".into(),
            "age".into(),
            "Alice".into(),
            "30".into(),
            "true".into(),
            "false".into(),
            " ".into(),
            "\n".into(),
        ];
        VocabIndex::from_vocab(&tokens)
    }

    #[test]
    fn test_json_schema_constraint_tokens() {
        let vocab = test_vocab();
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });

        let mut constraint = JsonSchemaConstraint::from_json_schema(&schema, Arc::clone(&vocab)).unwrap();
        let mask = constraint.allowed_tokens(&[], &vocab);
        assert!(!mask.is_empty());
    }
}
