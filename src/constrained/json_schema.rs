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
        format!("{prefix}_{}", self.rule_counter)
    }

    fn compile_schema(&mut self, schema: &Value, rule_hint: &str) -> Result<String, String> {
        // Collect $defs or definitions if present
        if let Some(defs_val) = schema.get("$defs").or_else(|| schema.get("definitions")) {
            if let Some(defs_obj) = defs_val.as_object() {
                for (k, v) in defs_obj {
                    self.defs.insert(format!("#/$defs/{k}"), v.clone());
                    self.defs.insert(format!("#/definitions/{k}"), v.clone());
                }
            }
        }

        // Handle $ref
        if let Some(ref_str) = schema.get("$ref").and_then(Value::as_str) {
            if let Some(target) = self.defs.get(ref_str).cloned() {
                return self.compile_schema(&target, rule_hint);
            }
        }

        // Handle anyOf / oneOf
        if let Some(any_of) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            if let Some(variants) = any_of.as_array() {
                let mut branch_exprs = Vec::new();
                for (idx, v) in variants.iter().enumerate() {
                    let hint = format!("{rule_hint}_v{idx}");
                    let expr = self.compile_schema(v, &hint)?;
                    branch_exprs.push(expr);
                }
                if !branch_exprs.is_empty() {
                    return Ok(format!("({})", branch_exprs.join(" | ")));
                }
            }
        }

        // Handle enum
        if let Some(enum_vals) = schema.get("enum").and_then(Value::as_array) {
            let mut choices = Vec::new();
            for v in enum_vals {
                match v {
                    Value::String(s) => choices.push(format!("\"\\\"{}\\\"\"", s)),
                    other => choices.push(format!("\"{}\"", other)),
                }
            }
            if !choices.is_empty() {
                return Ok(format!("({})", choices.join(" | ")));
            }
        }

        let type_name = schema.get("type").and_then(Value::as_str).unwrap_or("any");

        match type_name {
            "string" => Ok("string".to_string()),
            "integer" => Ok("integer".to_string()),
            "number" => Ok("number".to_string()),
            "boolean" => Ok("boolean".to_string()),
            "null" => Ok("null".to_string()),
            "array" => self.compile_array(schema, rule_hint),
            "object" => self.compile_object(schema, rule_hint),
            "any" => Ok("value".to_string()),
            other => Err(format!("unsupported JSON schema type '{other}'")),
        }
    }

    fn compile_array(&mut self, schema: &Value, rule_hint: &str) -> Result<String, String> {
        let item_expr = if let Some(items) = schema.get("items") {
            let hint = format!("{rule_hint}_item");
            self.compile_schema(items, &hint)?
        } else {
            "value".to_string()
        };

        let rule_name = self.fresh_rule_name(rule_hint);
        let array_rule = format!(r#""[" ws ({item_expr} ("," ws {item_expr})*)? ws "]""#);
        self.generated_rules.insert(rule_name.clone(), array_rule);
        Ok(rule_name)
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

            // Build object field rules
            let rule_name = self.fresh_rule_name(rule_hint);
            let mut field_exprs = Vec::new();

            for (prop_name, prop_schema) in props {
                let prop_hint = format!("{rule_name}_{prop_name}");
                let val_expr = self.compile_schema(prop_schema, &prop_hint)?;
                let key_lit = format!("\"\\\"{}\\\"\"", prop_name);
                let field_match = format!("{key_lit} ws \":\" ws {val_expr}");

                if required.contains(prop_name) {
                    field_exprs.push(field_match);
                } else {
                    field_exprs.push(format!("({field_match})?"));
                }
            }

            let perms = if field_exprs.len() <= 4 {
                generate_permutations(&field_exprs)
            } else {
                vec![field_exprs]
            };

            let perm_strings: Vec<String> = perms
                .into_iter()
                .map(|p| p.join(" \",\" ws "))
                .collect();

            let body = if perm_strings.len() == 1 {
                perm_strings.into_iter().next().unwrap()
            } else {
                format!("({})", perm_strings.join(" | "))
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
    fn test_compile_object_schema_with_required_fields() {
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
