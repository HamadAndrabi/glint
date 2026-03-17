//! Axum route handlers for the OpenAI-compatible HTTP API.
//!
//! Each handler receives an `Arc<AppState>` via axum's State extractor.
//!
//! Inference is CPU-bound, so it runs inside `tokio::task::spawn_blocking`.
//! This keeps the async runtime free to handle other requests while inference
//! is running. The async runtime has a fixed pool of blocking threads
//! (default: 512) managed by tokio.
//!
//! For streaming: we create an mpsc channel. The blocking task sends token IDs
//! down the channel; the async handler wraps the receiver in a Stream and feeds
//! it to axum's SSE response.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::model::chat_template::Message;
use crate::sampling::{Sampler, SamplerConfig};
use crate::transformer::{generate_cached, generate_streaming};

use super::state::AppState;
use super::types::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id() -> String {
    // Simple ID from timestamp + a few xorshift bits
    let ts = now_secs();
    let noise = {
        let mut s = ts ^ 0xdeadbeef_cafebabe;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s & 0xffffff
    };
    format!("cmpl-{ts:x}{noise:06x}")
}

fn sampler_config_from_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
) -> SamplerConfig {
    SamplerConfig {
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        repeat_penalty: repeat_penalty.unwrap_or(1.0),
        seed,
        ..Default::default()
    }
}

/// JSON error response with an HTTP status code.
fn api_error(status: StatusCode, msg: impl Into<String>) -> Response {
    let body = Json(ErrorResponse {
        error: ErrorDetail {
            message: msg.into(),
            error_type: "api_error",
            code: status.as_u16(),
        },
    });
    (status, body).into_response()
}

// ── GET /v1/models ────────────────────────────────────────────────────────────

/// List the loaded model.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model",
            created: 0,
            owned_by: "ferrite",
        }],
    })
}

// ── POST /v1/completions ──────────────────────────────────────────────────────

/// Text completion — streaming or non-streaming.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    // Tokenize the prompt
    let mut prompt_tokens = state.tokenizer.encode(&req.prompt);
    prompt_tokens.insert(0, state.tokenizer.bos_token_id);

    let max_tokens = req.max_tokens.unwrap_or(100);
    let stream = req.stream.unwrap_or(false);
    let sampler_cfg = sampler_config_from_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.repeat_penalty,
        req.seed,
    );
    let prompt_len = prompt_tokens.len();
    let model_name = req.model.clone();
    let eos = state.tokenizer.eos_token_id;

    if stream {
        streaming_completion(state, prompt_tokens, max_tokens, sampler_cfg, eos, model_name).await
    } else {
        non_streaming_completion(state, prompt_tokens, prompt_len, max_tokens, sampler_cfg, eos, model_name).await
    }
}

async fn non_streaming_completion(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    eos: u32,
    model_name: String,
) -> Response {
    // Clone Arcs so the blocking closure can own them
    let weights = Arc::clone(&state.weights);
    let config = Arc::clone(&state.config);

    let result = tokio::task::spawn_blocking(move || {
        let mut sampler = Sampler::new(sampler_cfg);
        generate_cached(&weights, &config, &prompt_tokens, max_tokens, &mut sampler, eos)
    })
    .await;

    match result {
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "Inference task panicked"),
        Ok(output_tokens) => {
            let generated = &output_tokens[prompt_len..];
            // Re-clone tokenizer to decode (we moved it into spawn_blocking)
            let text = state.tokenizer.decode(generated);
            let completion_tokens = generated.len();

            Json(CompletionResponse {
                id: gen_id(),
                object: "text_completion",
                created: now_secs(),
                model: model_name,
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason: "stop",
                }],
                usage: UsageInfo {
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    total_tokens: prompt_len + completion_tokens,
                },
            })
            .into_response()
        }
    }
}

async fn streaming_completion(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    eos: u32,
    model_name: String,
) -> Response {
    // Channel: blocking inference thread sends token IDs; async handler reads them
    let (tx, rx) = mpsc::channel::<u32>(64);

    // Spawn inference on the blocking thread pool
    let weights = Arc::clone(&state.weights);
    let config = Arc::clone(&state.config);
    tokio::task::spawn_blocking(move || {
        let mut sampler = Sampler::new(sampler_cfg);
        generate_streaming(
            &weights,
            &config,
            &prompt_tokens,
            max_tokens,
            &mut sampler,
            eos,
            |token_id| {
                // blocking_send returns Err if the receiver is dropped (client disconnected)
                tx.blocking_send(token_id).is_ok()
            },
        );
    });

    // Wrap the receiver in a Stream and map each token_id to an SSE Event
    let tokenizer = Arc::clone(&state.tokenizer);
    let id = gen_id();
    let created = now_secs();

    let stream = ReceiverStream::new(rx).map(move |token_id| {
        let text = tokenizer.decode(&[token_id]);
        let chunk = CompletionChunk {
            id: id.clone(),
            object: "text_completion",
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                text,
                index: 0,
                finish_reason: None,
            }],
        };
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok::<Event, Infallible>(Event::default().data(data))
    });

    // Append the final [DONE] sentinel
    let done_stream = stream.chain(tokio_stream::once(Ok::<Event, Infallible>(
        Event::default().data("[DONE]"),
    )));

    Sse::new(done_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ── POST /v1/chat/completions ─────────────────────────────────────────────────

/// Chat completion — converts messages to a prompt, then runs inference.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    // Apply the model's chat template to format messages into a prompt.
    let msgs: Vec<Message<'_>> = req.messages.iter()
        .map(|m| Message { role: &m.role, content: &m.content })
        .collect();
    let prompt = state.chat_template.apply(&msgs);

    let mut prompt_tokens = state.tokenizer.encode(&prompt);
    prompt_tokens.insert(0, state.tokenizer.bos_token_id);

    let max_tokens = req.max_tokens.unwrap_or(100);
    let stream = req.stream.unwrap_or(false);
    let sampler_cfg = sampler_config_from_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.repeat_penalty,
        req.seed,
    );
    let prompt_len = prompt_tokens.len();
    let model_name = req.model.clone();
    let eos = state.tokenizer.eos_token_id;

    if stream {
        // For streaming chat, reuse the completions streaming path
        streaming_completion(state, prompt_tokens, max_tokens, sampler_cfg, eos, model_name).await
    } else {
        let weights = Arc::clone(&state.weights);
        let config = Arc::clone(&state.config);

        let result = tokio::task::spawn_blocking(move || {
            let mut sampler = Sampler::new(sampler_cfg);
            generate_cached(&weights, &config, &prompt_tokens, max_tokens, &mut sampler, eos)
        })
        .await;

        match result {
            Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "Inference task panicked"),
            Ok(output_tokens) => {
                let generated = &output_tokens[prompt_len..];
                let text = state.tokenizer.decode(generated);
                let completion_tokens = generated.len();

                Json(ChatCompletionResponse {
                    id: gen_id(),
                    object: "chat.completion",
                    created: now_secs(),
                    model: model_name,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessageOut {
                            role: "assistant",
                            content: text,
                        },
                        finish_reason: "stop",
                    }],
                    usage: UsageInfo {
                        prompt_tokens: prompt_len,
                        completion_tokens,
                        total_tokens: prompt_len + completion_tokens,
                    },
                })
                .into_response()
            }
        }
    }
}

