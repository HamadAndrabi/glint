# Structured Output & Tool Calling

Glint includes a built-in Context-Free Grammar (GBNF) engine and JSON Schema compiler that forces model sampling to adhere strictly to defined schemas or grammar specifications.

---

## 1. GBNF Grammar Engine

The grammar engine (`src/constrained/gbnf.rs`) parses standard GBNF grammars into rule ASTs and dynamically masks token logits at every sampling step.

### Grammar Example

```text
root ::= "{" ws "\"name\":" ws string "," ws "\"age\":" ws number "}"
string ::= "\"" [a-zA-Z]+ "\""
number ::= [0-9]+
ws ::= [ \t\n\r]*
```

---

## 2. JSON Schema Compiler

Glint converts JSON Schema definitions (`src/constrained/json_schema.rs`) into equivalent GBNF grammar rules automatically, supporting:

- Primitive types: `string`, `integer`, `number`, `boolean`, `null`
- Objects with `properties` and `required` constraints
- Arrays with `items` definitions
- Enums (`"enum": ["option1", "option2"]`)
- Nested objects and arrays

---

## 3. OpenAI Tool Calling

Tool calling is supported natively in `/v1/chat/completions`:

```json
{
  "model": "model",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  }]
}
```

The response includes properly formatted `tool_calls` containing verified JSON arguments.

---

## 4. Multi-Language Bindings

Structured output is exposed across all Glint surfaces:

- **Python**: `m.generate_constrained(prompt, max_tokens, grammar=gbnf_string)`
- **WASM**: `model.generate_constrained(prompt, max_tokens, grammar)`
- **C FFI**: `glint_generate_constrained(model, prompt, max_tokens, grammar, callback, userdata)`
