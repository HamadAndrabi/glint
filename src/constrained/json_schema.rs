//! JSON Schema to GBNF (GGML BNF) Grammar Compiler.
//!
//! Compiles standard JSON Schema objects (primitives, objects, required keys,
//! arrays, enums, anyOf/oneOf, nested structures, and $defs) into strict GBNF
//! grammars for grammar-guided LLM sampling.

use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Convert a JSON Schema `Value` into a GBNF grammar string.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String, String> {
    let mut compiler = SchemaCompiler::new();
    let root_expr = compiler.compile_schema(schema, "root_type")?;

    let mut gbnf = String::new();
    gbnf.push_str("root ::= ws ");
    gbnf.push_str(&root_expr);
    gbnf.push_str(" ws\n\n");

    // Emit standard primitives and helper whitespace rules
    gbnf.push_str(r#"ws ::= [ \t\n\r]*
string ::= "\"" ([^"\\\x00-\x1f] | "\\" [^\x00-\x1f])* "\""
number ::= ("-")? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
integer ::= ("-")? ([0-9] | [1-9] [0-9]*)
boolean ::= "true" | "false"
null ::= "null"
value ::= object | array | string | number | boolean | null
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
"#);

    // Emit all custom rules generated during compilation
    for (name, rule) in &compiler.generated_rules {
        gbnf.push('\n');
        gbnf.push_str(name);
        gbnf.push_str(" ::= ");
        gbnf.push_str(rule);
        gbnf.push('\n');
    }

    Ok(gbnf)
}

fn generate_permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.is_empty() {
        return vec![vec![]];
    }
    if items.len() == 1 {
        return vec![vec![items[0].clone()]];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let cur = rest.remove(i);
        let sub_perms = generate_permutations(&rest);
        for mut sub in sub_perms {
            let mut perm = vec![cur.clone()];
            perm.append(&mut sub);
            result.push(perm);
        }
    }
    result
}

fn sanitize_ident(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if s.is_empty() || s.chars().next().unwrap().is_ascii_digit() {
        format!("p_{s}")
    } else {
        s
    }
}

fn escape_gbnf_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

struct SchemaCompiler {
    rule_counter: usize,
    generated_rules: HashMap<String, String>,
    defs: HashMap<String, Value>,
}

impl SchemaCompiler {
    fn new() -> Self {
        Self {
            rule_counter: 0,
            generated_rules: HashMap::new(),
            defs: HashMap::new(),
        }
    }

    fn fresh_rule_name(&mut self, prefix: &str) -> String {
        self.rule_counter += 1;
        let clean_prefix = sanitize_ident(prefix);
        format!("{clean_prefix}_{}", self.rule_counter)
    }

    fn compile_schema(&mut self, schema: &Value, rule_hint: &str) -> Result<String, String> {
        // Collect $defs or definitions if present
        if let Some(defs_val) = schema.get("$defs").or_else(|| schema.get("definitions")) {
            if let Some(defs_obj) = defs_val.as_object() {
                for (k, v) in defs_obj {
                    self.defs.insert(k.clone(), v.clone());
                }
            }
        }

        // Handle $ref
        if let Some(ref_str) = schema.get("$ref").and_then(Value::as_str) {
            let ref_name = ref_str.trim_start_matches("#/$defs/").trim_start_matches("#/definitions/");
            if let Some(ref_schema) = self.defs.get(ref_name).cloned() {
                return self.compile_schema(&ref_schema, ref_name);
            }
        }

        // Handle anyOf / oneOf
        if let Some(any_of) = schema.get("anyOf").or_else(|| schema.get("oneOf")).and_then(Value::as_array) {
            let mut branches = Vec::new();
            for (i, sub) in any_of.iter().enumerate() {
                let sub_hint = format!("{rule_hint}_branch_{i}");
                branches.push(self.compile_schema(sub, &sub_hint)?);
            }
            return Ok(format!("({})", branches.join(" | ")));
        }

        // Handle enum
        if let Some(enum_vals) = schema.get("enum").and_then(Value::as_array) {
            let mut lits = Vec::new();
            for val in enum_vals {
                let json_lit = serde_json::to_string(val).unwrap_or_default();
                let escaped = escape_gbnf_string(&json_lit);
                lits.push(format!("\"{}\"", escaped));
            }
            if lits.is_empty() {
                return Ok("reject".to_string());
            }
            return Ok(format!("({})", lits.join(" | ")));
        }

        // Handle type
        let type_str = schema.get("type").and_then(Value::as_str).unwrap_or("any");
        match type_str {
            "string" => Ok("string".to_string()),
            "integer" => Ok("integer".to_string()),
            "number" => Ok("number".to_string()),
            "boolean" => Ok("boolean".to_string()),
            "null" => Ok("null".to_string()),
            "array" => self.compile_array(schema, rule_hint),
            "object" => self.compile_object(schema, rule_hint),
            _ => Ok("value".to_string()),
        }
    }

    fn compile_array(&mut self, schema: &Value, rule_hint: &str) -> Result<String, String> {
        if let Some(items) = schema.get("items") {
            let item_hint = format!("{rule_hint}_item");
            let item_expr = self.compile_schema(items, &item_hint)?;
            let rule_name = self.fresh_rule_name(rule_hint);
            let arr_rule = format!(r#""[" ws ({item_expr} ("," ws {item_expr})*)? ws "]""#);
            self.generated_rules.insert(rule_name.clone(), arr_rule);
            Ok(rule_name)
        } else {
            Ok("array".to_string())
        }
    }

    fn compile_object(&mut self, schema: &Value, rule_hint: &str) -> Result<String, String> {
        let properties = schema.get("properties").and_then(Value::as_object);
        let required: HashSet<String> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        if let Some(props) = properties {
            if props.is_empty() {
                return Ok(r#""{" ws "}""#.to_string());
            }

            let rule_name = self.fresh_rule_name(rule_hint);
            let mut prop_items: Vec<(String, String, bool)> = Vec::new();

            for (prop_name, prop_schema) in props {
                let clean_name = sanitize_ident(prop_name);
                let prop_hint = format!("{rule_name}_{clean_name}");
                let val_expr = self.compile_schema(prop_schema, &prop_hint)?;
                let key_lit = format!("\"\\\"{}\\\"\"", escape_gbnf_string(prop_name));
                let field_match = format!("{key_lit} ws \":\" ws {val_expr}");
                let is_req = required.contains(prop_name);
                prop_items.push((prop_name.clone(), field_match, is_req));
            }

            let n = prop_items.len();
            let body = if n <= 4 {
                let mut branches = Vec::new();
                let num_subsets = 1usize << n;
                // Iterate over all non-empty subsets
                for mask in 1..num_subsets {
                    // Check if subset contains all required properties
                    let mut valid = true;
                    for (i, (_, _, is_req)) in prop_items.iter().enumerate() {
                        if *is_req && (mask & (1 << i)) == 0 {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }

                    // Extract items in subset
                    let subset_items: Vec<String> = (0..n)
                        .filter(|i| (mask & (1 << i)) != 0)
                        .map(|i| prop_items[i].1.clone())
                        .collect();

                    let perms = generate_permutations(&subset_items);
                    for p in perms {
                        branches.push(p.join(" \",\" ws "));
                    }
                }

                if required.is_empty() {
                    if branches.is_empty() {
                        "".to_string()
                    } else {
                        format!("({})?", branches.join(" | "))
                    }
                } else if branches.len() == 1 {
                    branches.into_iter().next().unwrap()
                } else {
                    format!("({})", branches.join(" | "))
                }
            } else {
                let field_matches: Vec<String> = prop_items.into_iter().map(|(_, fm, _)| fm).collect();
                let any_prop = format!("({})", field_matches.join(" | "));
                if required.is_empty() {
                    format!("({any_prop} (\",\" ws {any_prop})*)?")
                } else {
                    format!("{any_prop} (\",\" ws {any_prop})*")
                }
            };

            let obj_rule = format!(r#""{{" ws {body} ws "}}""#);
            self.generated_rules.insert(rule_name.clone(), obj_rule);
            Ok(rule_name)
        } else {
            Ok("object".to_string())
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained::gbnf::{GbnfGrammar, GbnfMatcher};
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn test_compile_primitive_schema() {
        let schema = json!({
            "type": "string"
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root ::= ws string ws"));

        let grammar = GbnfGrammar::from_str(&gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));
        matcher.advance_str("\"hello world\"");
        assert!(matcher.is_terminal());
    }

    #[test]
    fn test_compile_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        let grammar = GbnfGrammar::from_str(&gbnf).unwrap();
        let mut matcher = GbnfMatcher::new(Arc::new(grammar));

        let valid_json = r#"{"name": "Alice", "age": 30}"#;
        matcher.advance_str(valid_json);
        assert!(matcher.is_terminal());
    }

    #[test]
    fn test_compile_object_schema_with_optional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        let grammar = Arc::new(GbnfGrammar::from_str(&gbnf).unwrap());

        // Test 1: required only
        let mut m1 = GbnfMatcher::new(Arc::clone(&grammar));
        m1.advance_str(r#"{"name": "Alice"}"#);
        assert!(m1.is_terminal(), "object with only required field must be terminal");

        // Test 2: required + optional
        let mut m2 = GbnfMatcher::new(Arc::clone(&grammar));
        m2.advance_str(r#"{"name": "Alice", "age": 30}"#);
        assert!(m2.is_terminal(), "object with required and optional field must be terminal");

        // Test 3: optional only should NOT be terminal
        let mut m3 = GbnfMatcher::new(Arc::clone(&grammar));
        m3.advance_str(r#"{"age": 30}"#);
        assert!(!m3.is_terminal(), "missing required field should not be terminal");
    }

    #[test]
    fn test_compile_unsanitized_property_names() {
        let schema = json!({
            "type": "object",
            "properties": {
                "first name": { "type": "string" },
                "user-age.val": { "type": "integer" }
            },
            "required": ["first name"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        let grammar = Arc::new(GbnfGrammar::from_str(&gbnf).unwrap());

        let mut matcher = GbnfMatcher::new(Arc::clone(&grammar));
        matcher.advance_str(r#"{"first name": "Alice"}"#);
        assert!(matcher.is_terminal());
    }

    #[test]
    fn test_compile_enum_schema() {
        let schema = json!({
            "type": "string",
            "enum": ["asc", "desc"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        let grammar = GbnfGrammar::from_str(&gbnf).unwrap();

        let mut matcher1 = GbnfMatcher::new(Arc::new(grammar.clone()));
        matcher1.advance_str("\"asc\"");
        assert!(matcher1.is_terminal());

        let mut matcher2 = GbnfMatcher::new(Arc::new(grammar));
        matcher2.advance_str("\"other\"");
        assert_eq!(matcher2.current_state, crate::constrained::gbnf::GbnfExpr::Reject);
    }
}
