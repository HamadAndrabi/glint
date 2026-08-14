//! GBNF (GGML BNF) grammar parser, AST, and match engine.
//!
//! Implements the GGML BNF standard used by llama.cpp and Ollama.
//! Allows defining context-free grammars with character classes, negated classes,
//! string literals with escape sequences, alternations, sequences, groupings,
//! and repetition operators (`*`, `+`, `?`).
//!
//! State transitions are driven by Brzozowski derivatives with algebraic
//! simplification and recursion guards, making grammar execution robust and fast.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::VocabIndex;

/// An expression in a GBNF grammar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GbnfExpr {
    /// Matches the empty string (epsilon).
    Empty,
    /// Unmatchable / reject state.
    Reject,
    /// Exact terminal string.
    Literal(String),
    /// Character class: ranges of allowed (or disallowed if `negated`) unicode chars.
    CharClass {
        ranges: Vec<(char, char)>,
        negated: bool,
    },
    /// Non-terminal rule reference by name.
    RuleRef(String),
    /// Sequence of expressions to match in order: `A B C`.
    Sequence(Vec<GbnfExpr>),
    /// Choice / Alternation: `A | B | C`.
    Choice(Vec<GbnfExpr>),
    /// Zero or one match: `A?`.
    Optional(Box<GbnfExpr>),
    /// Zero or more matches: `A*`.
    ZeroOrMore(Box<GbnfExpr>),
    /// One or more matches: `A+`.
    OneOrMore(Box<GbnfExpr>),
}

/// A parsed GBNF grammar consisting of named rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GbnfGrammar {
    pub rules: HashMap<String, GbnfExpr>,
    pub root_rule: String,
}

impl std::str::FromStr for GbnfGrammar {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rules = HashMap::new();
        let mut first_rule_name = None;
        let mut current_rule_name = None;
        let mut current_rule_lines = Vec::new();

        for raw_line in s.lines() {
            let line = strip_comments(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some((name, expr_part)) = line.split_once("::=") {
                let trimmed_name = name.trim().to_string();
                if first_rule_name.is_none() {
                    first_rule_name = Some(trimmed_name.clone());
                }
                if let Some(prev_name) = current_rule_name.take() {
                    let combined = current_rule_lines.join(" ");
                    let expr = parse_expression(&combined)?;
                    rules.insert(prev_name, expr);
                    current_rule_lines.clear();
                }
                current_rule_name = Some(trimmed_name);
                current_rule_lines.push(expr_part.trim().to_string());
            } else if current_rule_name.is_some() {
                current_rule_lines.push(line.to_string());
            } else {
                return Err(format!("expected rule definition ('name ::= ...'), got: {line}"));
            }
        }

        if let Some(last_name) = current_rule_name {
            let combined = current_rule_lines.join(" ");
            let expr = parse_expression(&combined)?;
            rules.insert(last_name, expr);
        }

        if rules.is_empty() {
            return Err("grammar contains no rules".to_string());
        }

        let root_rule = if rules.contains_key("root") {
            "root".to_string()
        } else {
            first_rule_name.unwrap()
        };

        Ok(Self { rules, root_rule })
    }
}

impl GbnfGrammar {
    /// Parse a GBNF grammar from text.
    ///
    /// The entry rule defaults to `"root"`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, String> {
        <Self as std::str::FromStr>::from_str(input)
    }

    /// Retrieve the entry expression.
    pub fn root_expr(&self) -> GbnfExpr {
        self.rules
            .get(&self.root_rule)
            .cloned()
            .unwrap_or(GbnfExpr::Reject)
    }
}

fn strip_comments(line: &str) -> &str {
    let mut in_quote = false;
    let mut in_bracket = false;
    let mut quote_char = ' ';
    let mut escaped = false;

    let chars: Vec<(usize, char)> = line.char_indices().collect();
    for i in 0..chars.len() {
        let (idx, ch) = chars[i];
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if in_quote {
            if ch == quote_char {
                in_quote = false;
            }
        } else if in_bracket {
            if ch == ']' {
                in_bracket = false;
            }
        } else if ch == '"' || ch == '\'' {
            in_quote = true;
            quote_char = ch;
        } else if ch == '[' {
            in_bracket = true;
        } else if ch == '#' || (ch == '/' && i + 1 < chars.len() && chars[i + 1].1 == '/') {
            return &line[..idx];
        }
    }
    line
}

// ── Parser ───────────────────────────────────────────────────────────────────

fn parse_expression(input: &str) -> Result<GbnfExpr, String> {
    let tokens = tokenize_gbnf(input)?;
    let mut pos = 0;
    let expr = parse_choice(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected token at end of rule: {:?}", tokens[pos]));
    }
    Ok(expr.simplify())
}

#[derive(Clone, Debug, PartialEq)]
enum GbnfToken {
    Ident(String),
    Literal(String),
    CharClass { ranges: Vec<(char, char)>, negated: bool },
    Pipe,       // |
    LParen,     // (
    RParen,     // )
    Question,   // ?
    Star,       // *
    Plus,       // +
}

fn tokenize_gbnf(input: &str) -> Result<Vec<GbnfToken>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        match c {
            '|' => {
                tokens.push(GbnfToken::Pipe);
                i += 1;
            }
            '(' => {
                tokens.push(GbnfToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(GbnfToken::RParen);
                i += 1;
            }
            '?' => {
                tokens.push(GbnfToken::Question);
                i += 1;
            }
            '*' => {
                tokens.push(GbnfToken::Star);
                i += 1;
            }
            '+' => {
                tokens.push(GbnfToken::Plus);
                i += 1;
            }
            '"' => {
                // String literal
                i += 1;
                let mut lit = String::new();
                let mut closed = false;
                while i < len {
                    let ch = chars[i];
                    if ch == '\\' {
                        i += 1;
                        if i >= len {
                            return Err("unterminated escape in string literal".to_string());
                        }
                        let esc = chars[i];
                        match esc {
                            'n' => lit.push('\n'),
                            'r' => lit.push('\r'),
                            't' => lit.push('\t'),
                            '"' => lit.push('"'),
                            '\\' => lit.push('\\'),
                            'x' => {
                                // 2-digit hex
                                if i + 2 < len {
                                    let hex_str: String = chars[i + 1..=i + 2].iter().collect();
                                    if let Ok(b) = u8::from_str_radix(&hex_str, 16) {
                                        lit.push(b as char);
                                        i += 2;
                                    } else {
                                        lit.push_str("\\x");
                                    }
                                } else {
                                    lit.push_str("\\x");
                                }
                            }
                            other => {
                                lit.push(other);
                            }
                        }
                    } else if ch == '"' {
                        closed = true;
                        i += 1;
                        break;
                    } else {
                        lit.push(ch);
                    }
                    i += 1;
                }
                if !closed {
                    return Err("unterminated string literal".to_string());
                }
                tokens.push(GbnfToken::Literal(lit));
            }
            '[' => {
                // Character class `[a-z0-9]` or `[^"\\\x00-\x1f]`
                i += 1;
                let mut negated = false;
                if i < len && chars[i] == '^' {
                    negated = true;
                    i += 1;
                }
                let mut ranges = Vec::new();
                let mut closed = false;

                while i < len {
                    if chars[i] == ']' && !ranges.is_empty() {
                        closed = true;
                        i += 1;
                        break;
                    }

                    let start_ch = parse_char_in_class(&chars, &mut i, len)?;
                    if i < len && chars[i] == '-' && i + 1 < len && chars[i + 1] != ']' {
                        i += 1; // skip '-'
                        let end_ch = parse_char_in_class(&chars, &mut i, len)?;
                        ranges.push((start_ch, end_ch));
                    } else {
                        ranges.push((start_ch, start_ch));
                    }
                }

                if !closed {
                    return Err("unterminated character class `[...]`".to_string());
                }
                tokens.push(GbnfToken::CharClass { ranges, negated });
            }
            _ if c.is_alphanumeric() || c == '_' || c == '-' => {
                // Identifier
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '-') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(GbnfToken::Ident(ident));
            }
            other => {
                return Err(format!("unexpected character in GBNF grammar: '{other}'"));
            }
        }
    }

    Ok(tokens)
}

fn parse_char_in_class(chars: &[char], i: &mut usize, len: usize) -> Result<char, String> {
    if *i >= len {
        return Err("unexpected end of character class".to_string());
    }
    let ch = chars[*i];
    *i += 1;
    if ch == '\\' {
        if *i >= len {
            return Err("unterminated escape in character class".to_string());
        }
        let esc = chars[*i];
        *i += 1;
        match esc {
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            '\\' => Ok('\\'),
            ']' => Ok(']'),
            '-' => Ok('-'),
            '"' => Ok('"'),
            'x' => {
                if *i + 1 < len {
                    let hex_str: String = chars[*i..=*i + 1].iter().collect();
                    if let Ok(b) = u8::from_str_radix(&hex_str, 16) {
                        *i += 2;
                        return Ok(b as char);
                    }
                }
                Ok('x')
            }
            other => Ok(other),
        }
    } else {
        Ok(ch)
    }
}

fn parse_choice(tokens: &[GbnfToken], pos: &mut usize) -> Result<GbnfExpr, String> {
    let mut branches = Vec::new();
    let first = parse_sequence(tokens, pos)?;
    branches.push(first);

    while *pos < tokens.len() && tokens[*pos] == GbnfToken::Pipe {
        *pos += 1; // consume '|'
        let next = parse_sequence(tokens, pos)?;
        branches.push(next);
    }

    if branches.len() == 1 {
        Ok(branches.remove(0))
    } else {
        Ok(GbnfExpr::Choice(branches))
    }
}

fn parse_sequence(tokens: &[GbnfToken], pos: &mut usize) -> Result<GbnfExpr, String> {
    let mut elements = Vec::new();

    while *pos < tokens.len() {
        match &tokens[*pos] {
            GbnfToken::Pipe | GbnfToken::RParen => break,
            _ => {
                let elem = parse_postfix(tokens, pos)?;
                elements.push(elem);
            }
        }
    }

    if elements.is_empty() {
        Ok(GbnfExpr::Empty)
    } else if elements.len() == 1 {
        Ok(elements.remove(0))
    } else {
        Ok(GbnfExpr::Sequence(elements))
    }
}

fn parse_postfix(tokens: &[GbnfToken], pos: &mut usize) -> Result<GbnfExpr, String> {
    let mut expr = parse_primary(tokens, pos)?;

    while *pos < tokens.len() {
        match &tokens[*pos] {
            GbnfToken::Question => {
                *pos += 1;
                expr = GbnfExpr::Optional(Box::new(expr));
            }
            GbnfToken::Star => {
                *pos += 1;
                expr = GbnfExpr::ZeroOrMore(Box::new(expr));
            }
            GbnfToken::Plus => {
                *pos += 1;
                expr = GbnfExpr::OneOrMore(Box::new(expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}

fn parse_primary(tokens: &[GbnfToken], pos: &mut usize) -> Result<GbnfExpr, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".to_string());
    }

    match &tokens[*pos] {
        GbnfToken::Ident(name) => {
            let expr = GbnfExpr::RuleRef(name.clone());
            *pos += 1;
            Ok(expr)
        }
        GbnfToken::Literal(s) => {
            let expr = GbnfExpr::Literal(s.clone());
            *pos += 1;
            Ok(expr)
        }
        GbnfToken::CharClass { ranges, negated } => {
            let expr = GbnfExpr::CharClass {
                ranges: ranges.clone(),
                negated: *negated,
            };
            *pos += 1;
            Ok(expr)
        }
        GbnfToken::LParen => {
            *pos += 1; // consume '('
            let inner = parse_choice(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != GbnfToken::RParen {
                return Err("expected closing ')'".to_string());
            }
            *pos += 1; // consume ')'
            Ok(inner)
        }
        other => Err(format!("unexpected token in primary expression: {other:?}")),
    }
}

// ── Simplification & Derivative Engine ───────────────────────────────────────

impl GbnfExpr {
    /// Simplify an expression by flattening choices/sequences, eliminating Empties and Rejects.
    pub fn simplify(self) -> Self {
        match self {
            Self::Sequence(elements) => {
                let mut flat = Vec::new();
                for e in elements {
                    let s = e.simplify();
                    match s {
                        Self::Reject => return Self::Reject,
                        Self::Empty => {}
                        Self::Sequence(sub) => flat.extend(sub),
                        other => flat.push(other),
                    }
                }
                if flat.is_empty() {
                    Self::Empty
                } else if flat.len() == 1 {
                    flat.remove(0)
                } else {
                    Self::Sequence(flat)
                }
            }
            Self::Choice(branches) => {
                let mut flat = Vec::new();
                for b in branches {
                    let s = b.simplify();
                    match s {
                        Self::Reject => {}
                        Self::Choice(sub) => flat.extend(sub),
                        other => {
                            if !flat.contains(&other) {
                                flat.push(other);
                            }
                        }
                    }
                }
                if flat.is_empty() {
                    Self::Reject
                } else if flat.len() == 1 {
                    flat.remove(0)
                } else {
                    Self::Choice(flat)
                }
            }
            Self::Optional(inner) => {
                let s = inner.simplify();
                match s {
                    Self::Empty | Self::Reject => Self::Empty,
                    Self::Optional(_) => s,
                    other => Self::Optional(Box::new(other)),
                }
            }
            Self::ZeroOrMore(inner) => {
                let s = inner.simplify();
                match s {
                    Self::Empty | Self::Reject => Self::Empty,
                    Self::ZeroOrMore(_) => s,
                    other => Self::ZeroOrMore(Box::new(other)),
                }
            }
            Self::OneOrMore(inner) => {
                let s = inner.simplify();
                match s {
                    Self::Empty => Self::Empty,
                    Self::Reject => Self::Reject,
                    other => Self::OneOrMore(Box::new(other)),
                }
            }
            Self::Literal(s) if s.is_empty() => Self::Empty,
            other => other,
        }
    }

    /// Check if this expression can match the empty string (epsilon).
    pub fn nullable(&self, grammar: &HashMap<String, GbnfExpr>) -> bool {
        let mut visited = HashSet::new();
        self.nullable_inner(grammar, &mut visited)
    }

    fn nullable_inner(
        &self,
        grammar: &HashMap<String, GbnfExpr>,
        visited: &mut HashSet<String>,
    ) -> bool {
        match self {
            Self::Empty => true,
            Self::Reject => false,
            Self::Literal(s) => s.is_empty(),
            Self::CharClass { .. } => false,
            Self::RuleRef(name) => {
                if !visited.insert(name.clone()) {
                    return false; // recursion guard
                }
                let res = grammar
                    .get(name)
                    .map(|e| e.nullable_inner(grammar, visited))
                    .unwrap_or(false);
                visited.remove(name);
                res
            }
            Self::Sequence(elements) => elements.iter().all(|e| e.nullable_inner(grammar, visited)),
            Self::Choice(branches) => branches.iter().any(|b| b.nullable_inner(grammar, visited)),
            Self::Optional(_) => true,
            Self::ZeroOrMore(_) => true,
            Self::OneOrMore(inner) => inner.nullable_inner(grammar, visited),
        }
    }

    /// Compute the Brzozowski derivative with respect to character `ch`.
    pub fn derivative(&self, ch: char, grammar: &HashMap<String, GbnfExpr>) -> Self {
        let mut visited = HashSet::new();
        self.derivative_inner(ch, grammar, &mut visited).simplify()
    }

    fn derivative_inner(
        &self,
        ch: char,
        grammar: &HashMap<String, GbnfExpr>,
        visited: &mut HashSet<String>,
    ) -> Self {
        match self {
            Self::Empty | Self::Reject => Self::Reject,
            Self::Literal(s) => {
                if let Some(first) = s.chars().next() {
                    if first == ch {
                        let remainder = &s[ch.len_utf8()..];
                        if remainder.is_empty() {
                            Self::Empty
                        } else {
                            Self::Literal(remainder.to_string())
                        }
                    } else {
                        Self::Reject
                    }
                } else {
                    Self::Reject
                }
            }
            Self::CharClass { ranges, negated } => {
                let in_ranges = ranges.iter().any(|&(start, end)| ch >= start && ch <= end);
                let matched = if *negated { !in_ranges } else { in_ranges };
                if matched {
                    Self::Empty
                } else {
                    Self::Reject
                }
            }
            Self::RuleRef(name) => {
                if !visited.insert(name.clone()) {
                    return Self::Reject; // recursion loop
                }
                let res = if let Some(target) = grammar.get(name) {
                    target.derivative_inner(ch, grammar, visited)
                } else {
                    Self::Reject
                };
                visited.remove(name);
                res
            }
            Self::Sequence(elements) => {
                if elements.is_empty() {
                    return Self::Reject;
                }
                let first = &elements[0];
                let rest = &elements[1..];

                let mut branches = Vec::new();

                // branch 1: derivative of first concatenated with rest
                let d_first = first.derivative_inner(ch, grammar, visited);
                if d_first != Self::Reject {
                    let mut seq = vec![d_first];
                    seq.extend_from_slice(rest);
                    branches.push(Self::Sequence(seq));
                }

                // branch 2: if first is nullable, derivative of rest
                if first.nullable(grammar) && !rest.is_empty() {
                    let d_rest = Self::Sequence(rest.to_vec()).derivative_inner(ch, grammar, visited);
                    if d_rest != Self::Reject {
                        branches.push(d_rest);
                    }
                }

                Self::Choice(branches)
            }
            Self::Choice(branches) => {
                let d_branches = branches
                    .iter()
                    .map(|b| b.derivative_inner(ch, grammar, visited))
                    .collect();
                Self::Choice(d_branches)
            }
            Self::Optional(inner) => inner.derivative_inner(ch, grammar, visited),
            Self::ZeroOrMore(inner) => {
                let d_inner = inner.derivative_inner(ch, grammar, visited);
                if d_inner == Self::Reject {
                    Self::Reject
                } else {
                    Self::Sequence(vec![d_inner, Self::ZeroOrMore(inner.clone())])
                }
            }
            Self::OneOrMore(inner) => {
                let d_inner = inner.derivative_inner(ch, grammar, visited);
                if d_inner == Self::Reject {
                    Self::Reject
                } else {
                    Self::Sequence(vec![d_inner, Self::ZeroOrMore(inner.clone())])
                }
            }
        }
    }

    /// Step through all characters of a string.
    pub fn step_str(self, s: &str, grammar: &HashMap<String, GbnfExpr>) -> Self {
        let mut curr = self;
        for ch in s.chars() {
            curr = curr.derivative(ch, grammar);
            if curr == Self::Reject {
                return Self::Reject;
            }
        }
        curr
    }
}

// ── Matcher & Mask Cache ─────────────────────────────────────────────────────

/// A stateful GBNF grammar match engine that caches vocabulary masks per grammar state.
pub struct GbnfMatcher {
    pub grammar: Arc<GbnfGrammar>,
    pub current_state: GbnfExpr,
    mask_cache: HashMap<GbnfExpr, Vec<bool>>,
}

impl GbnfMatcher {
    /// Create a new matcher initialized at the root rule of the grammar.
    pub fn new(grammar: Arc<GbnfGrammar>) -> Self {
        let root = grammar.root_expr();
        Self {
            grammar,
            current_state: root,
            mask_cache: HashMap::new(),
        }
    }

    /// Advance the match state by consuming the characters in token string `s`.
    pub fn advance_str(&mut self, s: &str) {
        self.current_state = self
            .current_state
            .clone()
            .step_str(s, &self.grammar.rules);
    }

    /// Check if the current state accepts completion (e.g. EOS).
    pub fn is_terminal(&self) -> bool {
        self.current_state.nullable(&self.grammar.rules)
    }

    /// Compute (or retrieve from cache) the boolean allowed-token mask across the vocabulary.
    pub fn allowed_mask(&mut self, vocab: &VocabIndex, eos_token_id: Option<u32>) -> Vec<bool> {
        let state = self.current_state.clone();
        let grammar_rules = &self.grammar.rules;
        let is_term = self.is_terminal();

        let mut mask = if let Some(cached) = self.mask_cache.get(&state) {
            cached.clone()
        } else {
            let mut mask = Vec::with_capacity(vocab.strings.len());
            for s in &vocab.strings {
                if s.is_empty() {
                    mask.push(false);
                    continue;
                }
                let next_state = state.clone().step_str(s, grammar_rules);
                mask.push(next_state != GbnfExpr::Reject);
            }
            self.mask_cache.insert(state, mask.clone());
            mask
        };

        if let Some(eos) = eos_token_id {
            if (eos as usize) < mask.len() {
                mask[eos as usize] = is_term;
            }
        }
        for eos in &vocab.eos_token_ids {
            if (*eos as usize) < mask.len() {
                mask[*eos as usize] = is_term;
            }
        }

        mask
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_grammar() {
        let gbnf = r#"
            root ::= "hello" " " name
            name ::= "alice" | "bob" | "charlie"
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        assert_eq!(grammar.root_rule, "root");
        assert_eq!(grammar.rules.len(), 2);
    }

    #[test]
    fn test_gbnf_derivatives_literal() {
        let gbnf = r#"
            root ::= "hello world"
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));

        assert!(!matcher.is_terminal());
        matcher.advance_str("hello ");
        assert_ne!(matcher.current_state, GbnfExpr::Reject);
        assert!(!matcher.is_terminal());

        matcher.advance_str("world");
        assert_ne!(matcher.current_state, GbnfExpr::Reject);
        assert!(matcher.is_terminal());

        // Any extra character after completion should reject
        matcher.advance_str("!");
        assert_eq!(matcher.current_state, GbnfExpr::Reject);
    }

    #[test]
    fn test_gbnf_character_classes_and_repetition() {
        let gbnf = r#"
            root ::= [0-9]+ ("." [0-9]+)?
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));

        matcher.advance_str("42");
        assert_ne!(matcher.current_state, GbnfExpr::Reject);
        assert!(matcher.is_terminal());

        matcher.advance_str(".125");
        assert_ne!(matcher.current_state, GbnfExpr::Reject);
        assert!(matcher.is_terminal());

        matcher.advance_str("abc");
        assert_eq!(matcher.current_state, GbnfExpr::Reject);
    }

    #[test]
    fn test_gbnf_negated_class() {
        let gbnf = r#"
            root ::= "\"" [^"\\]* "\""
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));

        matcher.advance_str("\"hello world\"");
        assert!(matcher.is_terminal());
    }

    #[test]
    fn test_gbnf_vocab_masking() {
        let gbnf = r#"
            root ::= "true" | "false"
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));

        let vocab_raw = vec![
            "t".to_string(),
            "tr".to_string(),
            "true".to_string(),
            "f".to_string(),
            "false".to_string(),
            "x".to_string(),
            "hello".to_string(),
        ];
        let vocab = VocabIndex::from_vocab(&vocab_raw);

        let mask = matcher.allowed_mask(&vocab, None);
        // "t", "tr", "true", "f", "false" should be true; "x", "hello" should be false
        assert!(mask[0]); // "t"
        assert!(mask[1]); // "tr"
        assert!(mask[2]); // "true"
        assert!(mask[3]); // "f"
        assert!(mask[4]); // "false"
        assert!(!mask[5]); // "x"
        assert!(!mask[6]); // "hello"

        matcher.advance_str("tr");
        let mask2 = matcher.allowed_mask(&vocab, None);
        // After "tr", only "u" / "ue" are valid
        assert!(!mask2[0]); // "t" invalid
        assert!(!mask2[3]); // "f" invalid
    }

    #[test]
    fn test_gbnf_derivative_choice_branches_independent() {
        let gbnf = r#"
            root ::= foo "a" | foo "b"
            foo  ::= "x"
        "#;
        let grammar = Arc::new(GbnfGrammar::from_str(gbnf).unwrap());

        let mut m1 = GbnfMatcher::new(Arc::clone(&grammar));
        m1.advance_str("xa");
        assert!(m1.is_terminal(), "xa should be accepted");

        let mut m2 = GbnfMatcher::new(Arc::clone(&grammar));
        m2.advance_str("xb");
        assert!(m2.is_terminal(), "xb should be accepted");
    }

    #[test]
    fn test_gbnf_nullable_sequence_branches_independent() {
        let gbnf = r#"
            root ::= opt opt
            opt  ::= "a"?
        "#;
        let grammar = Arc::new(GbnfGrammar::from_str(gbnf).unwrap());
        let mut m = GbnfMatcher::new(Arc::clone(&grammar));
        assert!(m.is_terminal(), "root should be nullable initially");
        m.advance_str("a");
        assert!(m.is_terminal(), "root should be nullable after 1 'a'");
        m.advance_str("a");
        assert!(m.is_terminal(), "root should be nullable after 2 'a's");
    }

    #[test]
    fn test_strip_comments_with_bracketed_quotes() {
        let line = r#"chars ::= [^"\\]+ # comment"#;
        assert_eq!(strip_comments(line).trim(), r#"chars ::= [^"\\]+"#);
    }

    #[test]
    fn test_root_rule_input_order() {
        let gbnf = r#"
            entry ::= "hello"
            other ::= "world"
        "#;
        let grammar = GbnfGrammar::from_str(gbnf).unwrap();
        assert_eq!(grammar.root_rule, "entry");
    }
}

