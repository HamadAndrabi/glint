//! OpenAI-compatible request and response types.
//!
//! These match the shape of the OpenAI API so that any existing client library
//! (the OpenAI SDK, LangChain, curl, etc.) works with Glint without changes.
//!
//! Reference: https://platform.openai.com/docs/api-reference

use serde::{Deserialize, Serialize};

// ── ResponseFormat ────────────────────────────────────────────────────────────

/// OpenAI-compatible `response_format` object.
///
/// Supported types:
/// * `"text"` — default, unconstrained (same as omitting the field)
/// * `"json_object"` — constrains output to a valid JSON object
///
/// Example:
/// ```json
/// { "response_format": { "type": "json_object" } }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

impl ResponseFormat {
    /// True when this format requires JSON object output.
    pub fn is_json_object(&self) -> bool {
        self.format_type == "json_object"
    }
}

// ── Requests ─────────────────────────────────────────────────────────────────

/// `POST /v1/completions` request body.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    /// Maximum tokens to generate (default: 100).
    pub max_tokens: Option<usize>,
    /// Sampling temperature (default: 1.0, 0.0 = greedy).
    pub temperature: Option<f32>,
    /// Top-p nucleus sampling (default: 1.0 = disabled).
    pub top_p: Option<f32>,
    /// Top-k filtering (default: 0 = disabled).
    pub top_k: Option<usize>,
    /// Repetition penalty (default: 1.0 = disabled).
    pub repeat_penalty: Option<f32>,
    /// Seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Whether to stream token-by-token via SSE (default: false).
    pub stream: Option<bool>,
    /// Optional output format constraint.
    pub response_format: Option<ResponseFormat>,
}

/// One message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    /// "system", "user", or "assistant".
    pub role: String,
    pub content: String,
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    /// Optional output format constraint.
    pub response_format: Option<ResponseFormat>,
}

/// `POST /v1/responses` request body.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponseInput,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
}

/// Minimal text-only `input` forms supported by Glint's responses endpoint.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<ResponseInputMessage>),
}

#[derive(Debug, Deserialize)]
pub struct ResponseInputMessage {
    pub role: String,
    pub content: ResponseInputContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInputContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ResponseInputContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl ResponseInput {
    pub fn to_messages(&self, instructions: Option<&str>) -> Result<Vec<ChatMessage>, String> {
        let mut messages = Vec::new();

        if let Some(system) = instructions.filter(|text| !text.trim().is_empty()) {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            });
        }

        match self {
            ResponseInput::Text(text) => {
                if text.trim().is_empty() {
                    return Err("input must not be empty".to_string());
                }
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: text.clone(),
                });
            }
            ResponseInput::Messages(items) => {
                if items.is_empty() {
                    return Err("input must not be empty".to_string());
                }
                for item in items {
                    messages.push(ChatMessage {
                        role: item.role.clone(),
                        content: item.content.as_text()?,
                    });
                }
            }
        }

        Ok(messages)
    }
}

impl ResponseInputContent {
    fn as_text(&self) -> Result<String, String> {
        match self {
            ResponseInputContent::Text(text) => Ok(text.clone()),
            ResponseInputContent::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part.part_type.as_str() {
                        "input_text" | "text" => {
                            let part_text = part
                                .text
                                .as_deref()
                                .ok_or_else(|| "text content part is missing `text`".to_string())?;
                            text.push_str(part_text);
                        }
                        other => {
                            return Err(format!(
                                "unsupported responses input content type '{other}'; only text is supported"
                            ));
                        }
                    }
                }
                Ok(text)
            }
        }
    }
}

// ── Non-streaming responses ───────────────────────────────────────────────────

/// `POST /v1/completions` non-streaming response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// `POST /v1/chat/completions` non-streaming response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageOut,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: String,
}

/// `POST /v1/responses` non-streaming response.
#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub output_text: String,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
pub struct ResponseOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub status: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponseOutputContent>,
}

#[derive(Debug, Serialize)]
pub struct ResponseOutputContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
    pub annotations: Vec<ResponseOutputAnnotation>,
}

#[derive(Debug, Serialize)]
pub struct ResponseOutputAnnotation {}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

// ── Streaming chunks ──────────────────────────────────────────────────────────

/// One SSE data chunk sent during streaming.
///
/// Each token is sent as: `data: {json}\n\n`
/// The final event is: `data: [DONE]\n\n`
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub text: String,
    pub index: usize,
    /// `null` mid-stream, `"stop"` on the last chunk.
    pub finish_reason: Option<&'static str>,
}

/// Chat streaming chunk — `object: "chat.completion.chunk"` with `delta` fields.
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Models endpoint ───────────────────────────────────────────────────────────

/// `GET /v1/models` response.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

// ── Embeddings ────────────────────────────────────────────────────────────────

/// Input accepted by `POST /v1/embeddings`.
///
/// Matches the OpenAI API: a single string or a batch of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text string — backwards-compatible default.
    Single(String),
    /// Batch of text strings — embeds all inputs in one request.
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Return the inputs as a slice of `&str`.
    pub fn as_strings(&self) -> Vec<&str> {
        match self {
            EmbeddingInput::Single(s) => vec![s.as_str()],
            EmbeddingInput::Batch(v)  => v.iter().map(String::as_str).collect(),
        }
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    /// Text to embed.  Accepts a single string or a batch of strings.
    pub input: EmbeddingInput,
}

/// `POST /v1/embeddings` response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

// ── Error response ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_string_input_becomes_user_message() {
        let input = ResponseInput::Text("Hello".to_string());
        let messages = input.to_messages(None).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hello");
    }

    #[test]
    fn response_instructions_prepend_system_message() {
        let input = ResponseInput::Text("Hello".to_string());
        let messages = input.to_messages(Some("Be concise")).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "Be concise");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn response_content_parts_concatenate_text() {
        let input = ResponseInput::Messages(vec![ResponseInputMessage {
            role: "user".to_string(),
            content: ResponseInputContent::Parts(vec![
                ResponseInputContentPart {
                    part_type: "input_text".to_string(),
                    text: Some("Hello".to_string()),
                },
                ResponseInputContentPart {
                    part_type: "text".to_string(),
                    text: Some(" world".to_string()),
                },
            ]),
        }]);

        let messages = input.to_messages(None).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Hello world");
    }

    #[test]
    fn response_rejects_non_text_parts() {
        let input = ResponseInput::Messages(vec![ResponseInputMessage {
            role: "user".to_string(),
            content: ResponseInputContent::Parts(vec![ResponseInputContentPart {
                part_type: "input_image".to_string(),
                text: None,
            }]),
        }]);

        let err = input.to_messages(None).unwrap_err();
        assert!(err.contains("only text is supported"));
    }
}
