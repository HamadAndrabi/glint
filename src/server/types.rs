//! OpenAI-compatible request and response types.
//!
//! These match the shape of the OpenAI API so that any existing client library
//! (the OpenAI SDK, LangChain, curl, etc.) works with Glint without changes.
//!
//! Reference: https://platform.openai.com/docs/api-reference/completions

use serde::{Deserialize, Serialize};

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
}

/// One message in a chat conversation.
#[derive(Debug, Deserialize)]
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

/// `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    /// Text to embed. Accepts a single string (OpenAI-compatible).
    pub input: String,
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
