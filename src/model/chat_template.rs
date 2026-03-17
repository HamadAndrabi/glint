//! Chat template detection and rendering.
//!
//! GGUF files store a Jinja2 template string in `tokenizer.chat_template`.
//! Rather than implementing a full Jinja parser, we detect which well-known
//! format the template matches and apply it directly. This covers the vast
//! majority of models published on HuggingFace.
//!
//! Supported formats:
//!   - **ChatML**:    `<|im_start|>role\ncontent<|im_end|>`
//!   - **Llama 3**:   `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
//!   - **Mistral**:   `[INST] content [/INST]`
//!   - **Zephyr**:    `<|role|>\ncontent</s>`
//!   - **Gemma**:     `<start_of_turn>role\ncontent<end_of_turn>`
//!   - **Generic**:   `role: content\n` (fallback for unrecognized templates)

/// A detected chat template format.
///
/// Created once at model-load time from the GGUF metadata string.
/// Used at request time to format chat messages into a prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\ncontent<|im_end|>\n`
    ChatML,
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// `[INST] user_content [/INST]` (system prepended inside first [INST])
    MistralInstruct,
    /// `<|role|>\ncontent</s>\n`
    Zephyr,
    /// `<start_of_turn>role\ncontent<end_of_turn>\n`
    Gemma,
    /// Plain `role: content\n` fallback.
    Generic,
}

/// One message in a chat conversation (borrowed version for rendering).
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

impl ChatTemplate {
    /// Detect the template format from a raw Jinja template string.
    ///
    /// Matches on distinctive marker tokens that uniquely identify each format.
    /// Falls back to `Generic` if no known pattern is found.
    pub fn detect(template: &str) -> Self {
        if template.contains("<|im_start|>") {
            ChatTemplate::ChatML
        } else if template.contains("<|start_header_id|>") {
            ChatTemplate::Llama3
        } else if template.contains("[INST]") {
            ChatTemplate::MistralInstruct
        } else if template.contains("<start_of_turn>") {
            ChatTemplate::Gemma
        } else if template.contains("<|system|>") || template.contains("<|user|>") {
            ChatTemplate::Zephyr
        } else {
            ChatTemplate::Generic
        }
    }

    /// Render a list of chat messages into a prompt string.
    ///
    /// The rendered prompt ends with the assistant's opening tag so that the
    /// model continues generating as the assistant. This is the equivalent of
    /// `add_generation_prompt=True` in HuggingFace's tokenizer.
    pub fn apply(&self, messages: &[Message<'_>]) -> String {
        match self {
            ChatTemplate::ChatML => apply_chatml(messages),
            ChatTemplate::Llama3 => apply_llama3(messages),
            ChatTemplate::MistralInstruct => apply_mistral(messages),
            ChatTemplate::Zephyr => apply_zephyr(messages),
            ChatTemplate::Gemma => apply_gemma(messages),
            ChatTemplate::Generic => apply_generic(messages),
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::ChatML => "ChatML",
            ChatTemplate::Llama3 => "Llama-3",
            ChatTemplate::MistralInstruct => "Mistral",
            ChatTemplate::Zephyr => "Zephyr",
            ChatTemplate::Gemma => "Gemma",
            ChatTemplate::Generic => "generic",
        }
    }
}

// ── Format implementations ──────────────────────────────────────────────────

/// ChatML: `<|im_start|>role\ncontent<|im_end|>\n`
///
/// Used by: Qwen, Yi, many community fine-tunes.
fn apply_chatml(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(msg.role);
        prompt.push('\n');
        prompt.push_str(msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    // Generation prompt: open the assistant turn
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Llama 3: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
///
/// Used by: Meta Llama 3, Llama 3.1, Llama 3.2.
fn apply_llama3(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(msg.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(msg.content);
        prompt.push_str("<|eot_id|>");
    }
    // Generation prompt
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Mistral Instruct: `[INST] content [/INST]`
///
/// System message is prepended inside the first [INST] block.
/// Used by: Mistral 7B Instruct, Mixtral.
fn apply_mistral(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();

    // Collect system message (if any) to prepend to first user message
    let mut system_text: Option<&str> = None;
    let mut first_user = true;

    for msg in messages {
        match msg.role {
            "system" => {
                system_text = Some(msg.content);
            }
            "user" => {
                prompt.push_str("[INST] ");
                if first_user {
                    if let Some(sys) = system_text.take() {
                        prompt.push_str(sys);
                        prompt.push_str("\n\n");
                    }
                    first_user = false;
                }
                prompt.push_str(msg.content);
                prompt.push_str(" [/INST]");
            }
            "assistant" => {
                prompt.push(' ');
                prompt.push_str(msg.content);
                prompt.push_str("</s>");
            }
            _ => {
                // Unknown role: treat as user
                prompt.push_str("[INST] ");
                prompt.push_str(msg.content);
                prompt.push_str(" [/INST]");
            }
        }
    }

    prompt
}

/// Zephyr: `<|role|>\ncontent</s>\n`
///
/// Used by: Zephyr, StableLM, some HuggingFace fine-tunes.
fn apply_zephyr(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|");
        prompt.push_str(msg.role);
        prompt.push_str("|>\n");
        prompt.push_str(msg.content);
        prompt.push_str("</s>\n");
    }
    // Generation prompt
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Gemma: `<start_of_turn>role\ncontent<end_of_turn>\n`
///
/// Gemma uses "model" instead of "assistant" for the AI role.
/// Used by: Google Gemma, Gemma 2.
fn apply_gemma(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<start_of_turn>");
        // Gemma convention: "assistant" → "model"
        let role = if msg.role == "assistant" { "model" } else { msg.role };
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(msg.content);
        prompt.push_str("<end_of_turn>\n");
    }
    // Generation prompt
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Generic fallback: `role: content\n` then `assistant:`
fn apply_generic(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(msg.role);
        prompt.push_str(": ");
        prompt.push_str(msg.content);
        prompt.push('\n');
    }
    prompt.push_str("assistant:");
    prompt
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<Message<'static>> {
        vec![
            Message { role: "system", content: "You are helpful." },
            Message { role: "user", content: "Hi!" },
        ]
    }

    fn multi_turn() -> Vec<Message<'static>> {
        vec![
            Message { role: "user", content: "Hello" },
            Message { role: "assistant", content: "Hi there!" },
            Message { role: "user", content: "How are you?" },
        ]
    }

    #[test]
    fn test_detect_chatml() {
        let tpl = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}";
        assert_eq!(ChatTemplate::detect(tpl), ChatTemplate::ChatML);
    }

    #[test]
    fn test_detect_llama3() {
        let tpl = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}";
        assert_eq!(ChatTemplate::detect(tpl), ChatTemplate::Llama3);
    }

    #[test]
    fn test_detect_mistral() {
        let tpl = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% endif %}{% endfor %}";
        assert_eq!(ChatTemplate::detect(tpl), ChatTemplate::MistralInstruct);
    }

    #[test]
    fn test_detect_zephyr() {
        // Real Zephyr templates use the literal markers in the Jinja string
        let tpl = "{% for message in messages %}\n<|system|>\n{{ message['content'] }}</s>\n{% endfor %}<|assistant|>\n";
        assert_eq!(ChatTemplate::detect(tpl), ChatTemplate::Zephyr);
    }

    #[test]
    fn test_detect_gemma() {
        let tpl = "{% for message in messages %}<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n{% endfor %}";
        assert_eq!(ChatTemplate::detect(tpl), ChatTemplate::Gemma);
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(ChatTemplate::detect("some unknown template"), ChatTemplate::Generic);
    }

    #[test]
    fn test_chatml_format() {
        let msgs = sample_messages();
        let result = ChatTemplate::ChatML.apply(&msgs);
        assert_eq!(
            result,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi!<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_llama3_format() {
        let msgs = sample_messages();
        let result = ChatTemplate::Llama3.apply(&msgs);
        assert_eq!(
            result,
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_mistral_format() {
        let msgs = sample_messages();
        let result = ChatTemplate::MistralInstruct.apply(&msgs);
        assert_eq!(result, "[INST] You are helpful.\n\nHi! [/INST]");
    }

    #[test]
    fn test_mistral_multi_turn() {
        let msgs = multi_turn();
        let result = ChatTemplate::MistralInstruct.apply(&msgs);
        assert_eq!(result, "[INST] Hello [/INST] Hi there!</s>[INST] How are you? [/INST]");
    }

    #[test]
    fn test_zephyr_format() {
        let msgs = sample_messages();
        let result = ChatTemplate::Zephyr.apply(&msgs);
        assert_eq!(
            result,
            "<|system|>\nYou are helpful.</s>\n\
             <|user|>\nHi!</s>\n\
             <|assistant|>\n"
        );
    }

    #[test]
    fn test_gemma_format() {
        let msgs = sample_messages();
        let result = ChatTemplate::Gemma.apply(&msgs);
        assert_eq!(
            result,
            "<start_of_turn>system\nYou are helpful.<end_of_turn>\n\
             <start_of_turn>user\nHi!<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn test_gemma_assistant_becomes_model() {
        let msgs = multi_turn();
        let result = ChatTemplate::Gemma.apply(&msgs);
        assert!(result.contains("<start_of_turn>model\nHi there!<end_of_turn>"));
    }

    #[test]
    fn test_generic_fallback() {
        let msgs = sample_messages();
        let result = ChatTemplate::Generic.apply(&msgs);
        assert_eq!(result, "system: You are helpful.\nuser: Hi!\nassistant:");
    }
}
