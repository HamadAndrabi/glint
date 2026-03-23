//! Axum route handlers for the OpenAI-compatible HTTP API.
//!
//! Each handler receives an `Arc<AppState>` via axum's State extractor.
//!
//! Inference runs on a dedicated engine thread (see `engine.rs`). Route handlers
//! submit token sequences via `engine.submit()` and receive generated tokens
//! through an mpsc channel.
//!
//! For streaming: the token receiver is wrapped in a `ReceiverStream` and fed
//! directly to axum's SSE response.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::model::chat_template::Message;
use crate::sampling::SamplerConfig;
use crate::transformer::embed;

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

/// Validate that the requested model matches the loaded model.
fn validate_model(requested: &str, loaded: &str) -> Result<(), Response> {
    if requested != loaded {
        Err(api_error(
            StatusCode::NOT_FOUND,
            format!("model '{requested}' not found; available: {loaded}"),
        ))
    } else {
        Ok(())
    }
}

// ── GET /health ──────────────────────────────────────────────────────────────

/// Health check — returns 200 OK with a minimal JSON body.
///
/// Used by load balancers, orchestrators (k8s), and monitoring systems.
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

// ── GET /v1/metrics ──────────────────────────────────────────────────────────

/// Runtime metrics — requests, token throughput, average latency, uptime.
pub async fn server_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let requests = state.metrics.requests_total.load(Ordering::Relaxed);
    let tokens = state.metrics.tokens_generated.load(Ordering::Relaxed);
    let total_ms = state.metrics.total_latency_ms.load(Ordering::Relaxed);
    let uptime_secs = state.metrics.started_at.elapsed().as_secs();

    let avg_latency_ms = if requests > 0 {
        total_ms as f64 / requests as f64
    } else {
        0.0
    };

    Json(serde_json::json!({
        "requests_total":  requests,
        "tokens_generated": tokens,
        "avg_latency_ms":  avg_latency_ms,
        "uptime_secs":     uptime_secs,
    }))
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
            owned_by: "glint",
        }],
    })
}

// ── POST /v1/completions ──────────────────────────────────────────────────────

/// Text completion — streaming or non-streaming.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if let Err(e) = validate_model(&req.model, &state.model_name) {
        return e;
    }

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
    let model_name = state.model_name.clone();
    let eos = state.tokenizer.eos_token_id;

    // Validate against context window
    let context_limit = state.config.context_length as usize;
    if prompt_len >= context_limit {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("prompt ({prompt_len} tokens) exceeds model context window ({context_limit} tokens)"),
        );
    }
    let max_tokens = max_tokens.min(context_limit - prompt_len);

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
    let rx = match state.engine.submit(prompt_tokens, max_tokens, sampler_cfg, eos) {
        Some(rx) => rx,
        None => return api_error(StatusCode::SERVICE_UNAVAILABLE, "Inference engine unavailable"),
    };

    let t0 = Instant::now();
    let mut stream = ReceiverStream::new(rx);
    let mut generated_ids: Vec<u32> = Vec::new();
    while let Some(token_id) = stream.next().await {
        generated_ids.push(token_id);
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let text = state.tokenizer.decode(&generated_ids);
    let completion_tokens = generated_ids.len();
    state.metrics.record(completion_tokens as u64, elapsed_ms);

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

async fn streaming_completion(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    eos: u32,
    model_name: String,
) -> Response {
    let rx = match state.engine.submit(prompt_tokens, max_tokens, sampler_cfg, eos) {
        Some(rx) => rx,
        None => return api_error(StatusCode::SERVICE_UNAVAILABLE, "Inference engine unavailable"),
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let id = gen_id();
    let created = now_secs();

    // Clone for the finish chunk (originals are moved into the content closure)
    let finish_id = id.clone();
    let finish_model = model_name.clone();

    let tc = Arc::clone(&token_count);
    let stream = ReceiverStream::new(rx).map(move |token_id| {
        tc.fetch_add(1, Ordering::Relaxed);
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

    // Final chunk with finish_reason: "stop", then [DONE]
    let tc = Arc::clone(&token_count);
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).map(move |i| {
        if i == 0 {
            let chunk = CompletionChunk {
                id: finish_id.clone(),
                object: "text_completion",
                created,
                model: finish_model.clone(),
                choices: vec![ChunkChoice {
                    text: String::new(),
                    index: 0,
                    finish_reason: Some("stop"),
                }],
            };
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<Event, Infallible>(Event::default().data(data))
        } else {
            state.metrics.record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }
    });

    let done_stream = stream.chain(finish_stream);

    Sse::new(done_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn streaming_chat_completion(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    eos: u32,
    model_name: String,
) -> Response {
    let rx = match state.engine.submit(prompt_tokens, max_tokens, sampler_cfg, eos) {
        Some(rx) => rx,
        None => return api_error(StatusCode::SERVICE_UNAVAILABLE, "Inference engine unavailable"),
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let id = gen_id();
    let created = now_secs();

    // Clone for the finish chunk (originals are moved into the content closure)
    let finish_id = id.clone();
    let finish_model = model_name.clone();

    // First chunk: send the role
    let role_chunk = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_name.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let role_data = serde_json::to_string(&role_chunk).unwrap_or_default();
    let role_event = tokio_stream::once(Ok::<Event, Infallible>(
        Event::default().data(role_data),
    ));

    // Content chunks: one per token
    let tc = Arc::clone(&token_count);
    let content_stream = ReceiverStream::new(rx).map(move |token_id| {
        tc.fetch_add(1, Ordering::Relaxed);
        let text = tokenizer.decode(&[token_id]);
        let chunk = ChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some(text),
                },
                finish_reason: None,
            }],
        };
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok::<Event, Infallible>(Event::default().data(data))
    });

    // Final chunk with finish_reason: "stop", then [DONE]
    let tc = Arc::clone(&token_count);
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).map(move |i| {
        if i == 0 {
            let chunk = ChatChunk {
                id: finish_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: finish_model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop"),
                }],
            };
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<Event, Infallible>(Event::default().data(data))
        } else {
            state.metrics.record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }
    });

    let done_stream = role_event
        .chain(content_stream)
        .chain(finish_stream);

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
    if let Err(e) = validate_model(&req.model, &state.model_name) {
        return e;
    }

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
    let model_name = state.model_name.clone();
    let eos = state.tokenizer.eos_token_id;

    // Validate against context window
    let context_limit = state.config.context_length as usize;
    if prompt_len >= context_limit {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("prompt ({prompt_len} tokens) exceeds model context window ({context_limit} tokens)"),
        );
    }
    let max_tokens = max_tokens.min(context_limit - prompt_len);

    if stream {
        streaming_chat_completion(state, prompt_tokens, max_tokens, sampler_cfg, eos, model_name).await
    } else {
        let rx = match state.engine.submit(prompt_tokens, max_tokens, sampler_cfg, eos) {
            Some(rx) => rx,
            None => return api_error(StatusCode::SERVICE_UNAVAILABLE, "Inference engine unavailable"),
        };

        let t0 = Instant::now();
        let mut stream = ReceiverStream::new(rx);
        let mut generated_ids: Vec<u32> = Vec::new();
        while let Some(token_id) = stream.next().await {
            generated_ids.push(token_id);
        }

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        let text = state.tokenizer.decode(&generated_ids);
        let completion_tokens = generated_ids.len();
        state.metrics.record(completion_tokens as u64, elapsed_ms);

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

// ── POST /v1/embeddings ───────────────────────────────────────────────────────

/// Text embedding — tokenize the input, run the forward pass, mean-pool hidden states.
///
/// Returns an OpenAI-compatible embedding object. The vector dimension equals
/// the model's `embedding_length` (e.g. 576 for SmolLM-135M, 4096 for LLaMA-3-8B).
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    if let Err(e) = validate_model(&req.model, &state.model_name) {
        return e;
    }

    if req.input.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "input must not be empty");
    }

    // Tokenize (include BOS so the forward pass mirrors normal inference)
    let mut tokens = state.tokenizer.encode(&req.input);
    tokens.insert(0, state.tokenizer.bos_token_id);
    let n_tokens = tokens.len();
    let model_name = state.model_name.clone();

    let weights = Arc::clone(&state.weights);
    let config = Arc::clone(&state.config);

    let result = tokio::task::spawn_blocking(move || {
        embed(&weights, &config, &tokens)
    })
    .await;

    match result {
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "Embedding task panicked"),
        Ok(vector) => Json(EmbeddingResponse {
            object: "list",
            data: vec![EmbeddingData {
                object: "embedding",
                embedding: vector,
                index: 0,
            }],
            model: model_name,
            usage: UsageInfo {
                prompt_tokens: n_tokens,
                completion_tokens: 0,
                total_tokens: n_tokens,
            },
        })
        .into_response(),
    }
}
