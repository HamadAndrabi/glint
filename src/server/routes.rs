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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::constrained::ConstraintSpec;
use crate::model::chat_template::Message;
use crate::sampling::SamplerConfig;
use crate::transformer::embed_batch;

use super::state::AppState;
use super::types::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id_with_prefix(prefix: &str) -> String {
    // Simple ID from timestamp + a few xorshift bits
    let ts = now_secs();
    let noise = {
        let mut s = ts ^ 0xdeadbeef_cafebabe;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s & 0xffffff
    };
    format!("{prefix}-{ts:x}{noise:06x}")
}

fn gen_id() -> String {
    gen_id_with_prefix("cmpl")
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

struct PreparedGeneration {
    prompt_tokens: Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    eos: u32,
    model_name: String,
    /// Optional structured-output constraint.
    constraint: Option<ConstraintSpec>,
}

struct StartedGeneration {
    rx: tokio::sync::mpsc::Receiver<u32>,
    prompt_len: usize,
    model_name: String,
}

struct CompletedGeneration {
    text: String,
    completion_tokens: usize,
}

fn prepare_generation_from_prompt(
    state: &Arc<AppState>,
    requested_model: &str,
    prompt: &str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<PreparedGeneration, Response> {
    validate_model(requested_model, &state.model_name)?;

    let mut prompt_tokens = state.tokenizer.encode(prompt);
    prompt_tokens.insert(0, state.tokenizer.bos_token_id);

    let prompt_len = prompt_tokens.len();
    let context_limit = state.config.context_length as usize;
    if prompt_len >= context_limit {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt ({prompt_len} tokens) exceeds model context window ({context_limit} tokens)"
            ),
        ));
    }

    let max_tokens = max_tokens.unwrap_or(100).min(context_limit - prompt_len);
    let sampler_cfg = sampler_config_from_params(temperature, top_p, top_k, repeat_penalty, seed);

    Ok(PreparedGeneration {
        prompt_tokens,
        prompt_len,
        max_tokens,
        sampler_cfg,
        eos: state.tokenizer.eos_token_id,
        model_name: state.model_name.clone(),
        constraint: None,
    })
}

fn prepare_chat_generation(
    state: &Arc<AppState>,
    requested_model: &str,
    messages: &[ChatMessage],
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<PreparedGeneration, Response> {
    if messages.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
        ));
    }

    let msgs: Vec<Message<'_>> = messages
        .iter()
        .map(|m| Message {
            role: &m.role,
            content: &m.content,
        })
        .collect();
    let prompt = state.chat_template.apply(&msgs);

    prepare_generation_from_prompt(
        state,
        requested_model,
        &prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        seed,
    )
}

fn prepare_responses_generation(
    state: &Arc<AppState>,
    req: &ResponsesRequest,
) -> Result<PreparedGeneration, Response> {
    let messages = req
        .input
        .to_messages(req.instructions.as_deref())
        .map_err(|msg| api_error(StatusCode::BAD_REQUEST, msg))?;

    prepare_chat_generation(
        state,
        &req.model,
        &messages,
        req.max_output_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repeat_penalty,
        req.seed,
    )
}

fn start_generation(
    state: &Arc<AppState>,
    prepared: PreparedGeneration,
) -> Result<StartedGeneration, Response> {
    let PreparedGeneration {
        prompt_tokens,
        prompt_len,
        max_tokens,
        sampler_cfg,
        eos,
        model_name,
        constraint,
    } = prepared;

    let rx = state
        .engine
        .submit(prompt_tokens, max_tokens, sampler_cfg, eos, constraint, None)
        .ok_or_else(|| {
            state.metrics.record_failure();
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Inference engine unavailable",
            )
        })?;

    state.metrics.on_submit();

    Ok(StartedGeneration {
        rx,
        prompt_len,
        model_name,
    })
}

async fn collect_generation(
    state: Arc<AppState>,
    rx: tokio::sync::mpsc::Receiver<u32>,
) -> CompletedGeneration {
    let t0 = Instant::now();
    let mut stream = ReceiverStream::new(rx);
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut ttft_recorded = false;
    while let Some(token_id) = stream.next().await {
        if !ttft_recorded {
            state.metrics.record_ttft_us(t0.elapsed().as_micros() as u64);
            state.metrics.on_first_token();
            ttft_recorded = true;
        }
        generated_ids.push(token_id);
    }

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let text = state.tokenizer.decode(&generated_ids);
    let completion_tokens = generated_ids.len();
    state.metrics.record(completion_tokens as u64, elapsed_ms);
    state.metrics.on_complete();

    CompletedGeneration {
        text,
        completion_tokens,
    }
}

fn build_responses_response(
    id: String,
    message_id: String,
    created_at: u64,
    model: String,
    output_text: String,
    prompt_tokens: usize,
    output_tokens: usize,
    status: &'static str,
) -> ResponsesResponse {
    ResponsesResponse {
        id,
        object: "response",
        created_at,
        status,
        model,
        output: vec![ResponseOutputItem {
            id: message_id,
            item_type: "message",
            status,
            role: "assistant",
            content: vec![ResponseOutputContent {
                content_type: "output_text",
                text: output_text.clone(),
                annotations: vec![],
            }],
        }],
        output_text,
        usage: ResponsesUsage {
            input_tokens: prompt_tokens,
            output_tokens,
            total_tokens: prompt_tokens + output_tokens,
        },
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

/// Runtime metrics — requests, token throughput, latency, and concurrency.
pub async fn server_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let m = &state.metrics;
    let requests        = m.requests_total.load(Ordering::Relaxed);
    let tokens          = m.tokens_generated.load(Ordering::Relaxed);
    let failed          = m.requests_failed.load(Ordering::Relaxed);
    let active          = m.active_sessions.load(Ordering::Relaxed);
    let queued          = m.queue_depth.load(Ordering::Relaxed);
    let uptime_secs     = m.started_at.elapsed().as_secs();

    Json(serde_json::json!({
        // Throughput
        "requests_total":          requests,
        "requests_failed":         failed,
        "tokens_generated":        tokens,
        // Latency averages (0.0 until at least one request completes)
        "avg_latency_ms":          m.avg_latency_ms(),
        "avg_ttft_ms":             m.avg_ttft_ms(),
        "avg_decode_ms":           m.avg_decode_ms(),
        // Live concurrency
        "active_sessions":         active,
        "queue_depth":             queued,
        // Uptime
        "uptime_secs":             uptime_secs,
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
    let mut prepared = match prepare_generation_from_prompt(
        &state,
        &req.model,
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repeat_penalty,
        req.seed,
    ) {
        Ok(prepared) => prepared,
        Err(err) => return err,
    };

    if req.response_format.as_ref().map_or(false, |f| f.is_json_object()) {
        prepared.constraint = Some(ConstraintSpec::JsonObject);
    }

    if req.stream.unwrap_or(false) {
        streaming_completion(state, prepared).await
    } else {
        non_streaming_completion(state, prepared).await
    }
}

async fn non_streaming_completion(state: Arc<AppState>, prepared: PreparedGeneration) -> Response {
    let started = match start_generation(&state, prepared) {
        Ok(started) => started,
        Err(err) => return err,
    };

    let completed = collect_generation(Arc::clone(&state), started.rx).await;

    Json(CompletionResponse {
        id: gen_id(),
        object: "text_completion",
        created: now_secs(),
        model: started.model_name,
        choices: vec![CompletionChoice {
            text: completed.text,
            index: 0,
            finish_reason: "stop",
        }],
        usage: UsageInfo {
            prompt_tokens: started.prompt_len,
            completion_tokens: completed.completion_tokens,
            total_tokens: started.prompt_len + completed.completion_tokens,
        },
    })
    .into_response()
}

async fn streaming_completion(state: Arc<AppState>, prepared: PreparedGeneration) -> Response {
    let started = match start_generation(&state, prepared) {
        Ok(started) => started,
        Err(err) => return err,
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let ttft_instant = t0.clone();
    let id = gen_id();
    let created = now_secs();

    // Clone for the finish chunk (originals are moved into the content closure)
    let finish_id = id.clone();
    let stream_model = started.model_name.clone();
    let finish_model = started.model_name.clone();

    let tc = Arc::clone(&token_count);
    let ttft_sent = Arc::new(AtomicBool::new(false));
    let ttft_sent_clone = Arc::clone(&ttft_sent);
    let state_for_stream = Arc::clone(&state);
    let stream = ReceiverStream::new(started.rx).map(move |token_id| {
        if !ttft_sent_clone.swap(true, Ordering::Relaxed) {
            state_for_stream.metrics.record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
            state_for_stream.metrics.on_first_token();
        }
        tc.fetch_add(1, Ordering::Relaxed);
        let text = tokenizer.decode(&[token_id]);
        let chunk = CompletionChunk {
            id: id.clone(),
            object: "text_completion",
            created,
            model: stream_model.clone(),
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
            state
                .metrics
                .record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }
    });

    let done_stream = stream.chain(finish_stream);

    Sse::new(done_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn streaming_chat_completion(state: Arc<AppState>, prepared: PreparedGeneration) -> Response {
    let started = match start_generation(&state, prepared) {
        Ok(started) => started,
        Err(err) => return err,
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let ttft_instant = t0.clone();
    let id = gen_id();
    let created = now_secs();

    // Clone for the finish chunk (originals are moved into the content closure)
    let finish_id = id.clone();
    let stream_model = started.model_name.clone();
    let finish_model = started.model_name.clone();

    // First chunk: send the role
    let role_chunk = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: stream_model.clone(),
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
    let role_event = tokio_stream::once(Ok::<Event, Infallible>(Event::default().data(role_data)));

    // Content chunks: one per token
    let tc = Arc::clone(&token_count);
    let ttft_sent = Arc::new(AtomicBool::new(false));
    let ttft_sent_clone = Arc::clone(&ttft_sent);
    let state_for_stream = Arc::clone(&state);
    let content_stream = ReceiverStream::new(started.rx).map(move |token_id| {
        if !ttft_sent_clone.swap(true, Ordering::Relaxed) {
            state_for_stream.metrics.record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
            state_for_stream.metrics.on_first_token();
        }
        tc.fetch_add(1, Ordering::Relaxed);
        let text = tokenizer.decode(&[token_id]);
        let chunk = ChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: stream_model.clone(),
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
            state
                .metrics
                .record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }
    });

    let done_stream = role_event.chain(content_stream).chain(finish_stream);

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
    let mut prepared = match prepare_chat_generation(
        &state,
        &req.model,
        &req.messages,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repeat_penalty,
        req.seed,
    ) {
        Ok(prepared) => prepared,
        Err(err) => return err,
    };

    if req.response_format.as_ref().map_or(false, |f| f.is_json_object()) {
        prepared.constraint = Some(ConstraintSpec::JsonObject);
    }

    if req.stream.unwrap_or(false) {
        streaming_chat_completion(state, prepared).await
    } else {
        let started = match start_generation(&state, prepared) {
            Ok(started) => started,
            Err(err) => return err,
        };

        let completed = collect_generation(Arc::clone(&state), started.rx).await;

        Json(ChatCompletionResponse {
            id: gen_id(),
            object: "chat.completion",
            created: now_secs(),
            model: started.model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant",
                    content: completed.text,
                },
                finish_reason: "stop",
            }],
            usage: UsageInfo {
                prompt_tokens: started.prompt_len,
                completion_tokens: completed.completion_tokens,
                total_tokens: started.prompt_len + completed.completion_tokens,
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
// ── POST /v1/responses ────────────────────────────────────────────────────────

/// Text-only Responses API built on the same inference engine as chat/completions.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let prepared = match prepare_responses_generation(&state, &req) {
        Ok(prepared) => prepared,
        Err(err) => return err,
    };

    if req.stream.unwrap_or(false) {
        streaming_response(state, prepared).await
    } else {
        let started = match start_generation(&state, prepared) {
            Ok(started) => started,
            Err(err) => return err,
        };

        let id = gen_id_with_prefix("resp");
        let message_id = gen_id_with_prefix("msg");
        let created_at = now_secs();
        let completed = collect_generation(Arc::clone(&state), started.rx).await;

        Json(build_responses_response(
            id,
            message_id,
            created_at,
            started.model_name,
            completed.text,
            started.prompt_len,
            completed.completion_tokens,
            "completed",
        ))
        .into_response()
    }
}

async fn streaming_response(state: Arc<AppState>, prepared: PreparedGeneration) -> Response {
    let started = match start_generation(&state, prepared) {
        Ok(started) => started,
        Err(err) => return err,
    };

    let response_id = gen_id_with_prefix("resp");
    let message_id = gen_id_with_prefix("msg");
    let created_at = now_secs();
    let stream_model = started.model_name.clone();
    let finish_model = started.model_name.clone();
    let prompt_len = started.prompt_len;
    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let full_text = Arc::new(Mutex::new(String::new()));
    let t0 = Instant::now();

    let created_response = build_responses_response(
        response_id.clone(),
        message_id.clone(),
        created_at,
        stream_model,
        String::new(),
        prompt_len,
        0,
        "in_progress",
    );
    let created_data = serde_json::json!({
        "type": "response.created",
        "response": created_response,
    });
    let created_event = tokio_stream::once(Ok::<Event, Infallible>(
        Event::default()
            .event("response.created")
            .data(created_data.to_string()),
    ));

    let ttft_instant = t0.clone();
    let tc = Arc::clone(&token_count);
    let text_buf = Arc::clone(&full_text);
    let delta_response_id = response_id.clone();
    let ttft_sent = Arc::new(AtomicBool::new(false));
    let ttft_sent_clone = Arc::clone(&ttft_sent);
    let state_for_stream = Arc::clone(&state);
    let delta_stream = ReceiverStream::new(started.rx).map(move |token_id| {
        if !ttft_sent_clone.swap(true, Ordering::Relaxed) {
            state_for_stream.metrics.record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
            state_for_stream.metrics.on_first_token();
        }
        tc.fetch_add(1, Ordering::Relaxed);
        let text = tokenizer.decode(&[token_id]);
        if let Ok(mut buf) = text_buf.lock() {
            buf.push_str(&text);
        }

        let payload = serde_json::json!({
            "type": "response.output_text.delta",
            "response_id": delta_response_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        });

        Ok::<Event, Infallible>(
            Event::default()
                .event("response.output_text.delta")
                .data(payload.to_string()),
        )
    });

    let tc = Arc::clone(&token_count);
    let text_buf = Arc::clone(&full_text);
    let done_response_id = response_id.clone();
    let completed_response_id = response_id.clone();
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).map(move |i| {
        let text = text_buf.lock().map(|buf| buf.clone()).unwrap_or_default();

        if i == 0 {
            let payload = serde_json::json!({
                "type": "response.output_text.done",
                "response_id": done_response_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            });
            Ok::<Event, Infallible>(
                Event::default()
                    .event("response.output_text.done")
                    .data(payload.to_string()),
            )
        } else {
            let output_tokens = tc.load(Ordering::Relaxed) as usize;
            state
                .metrics
                .record(output_tokens as u64, t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();
            let response = build_responses_response(
                completed_response_id.clone(),
                message_id.clone(),
                created_at,
                finish_model.clone(),
                text,
                prompt_len,
                output_tokens,
                "completed",
            );
            let payload = serde_json::json!({
                "type": "response.completed",
                "response": response,
            });
            Ok::<Event, Infallible>(
                Event::default()
                    .event("response.completed")
                    .data(payload.to_string()),
            )
        }
    });

    let event_stream = created_event.chain(delta_stream).chain(finish_stream);

    Sse::new(event_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    if let Err(e) = validate_model(&req.model, &state.model_name) {
        return e;
    }

    let input_texts = req.input.as_strings();
    if input_texts.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "input must not be empty");
    }

    // Tokenize each input, prepending BOS to mirror normal inference.
    let all_tokens: Vec<Vec<u32>> = input_texts.iter().map(|text| {
        let mut tokens = state.tokenizer.encode(text);
        tokens.insert(0, state.tokenizer.bos_token_id);
        tokens
    }).collect();

    let total_tokens: usize = all_tokens.iter().map(|t| t.len()).sum();
    let model_name = state.model_name.clone();

    let weights = Arc::clone(&state.weights);
    let config  = Arc::clone(&state.config);

    let result = tokio::task::spawn_blocking(move || {
        let refs: Vec<&[u32]> = all_tokens.iter().map(Vec::as_slice).collect();
        embed_batch(&weights, &config, &refs)
    }).await;

    match result {
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "Embedding task panicked"),
        Ok(vectors) => {
            let data: Vec<EmbeddingData> = vectors.into_iter().enumerate()
                .map(|(i, embedding)| EmbeddingData {
                    object: "embedding",
                    embedding,
                    index: i,
                })
                .collect();
            Json(EmbeddingResponse {
                object: "list",
                data,
                model: model_name,
                usage: UsageInfo {
                    prompt_tokens: total_tokens,
                    completion_tokens: 0,
                    total_tokens,
                },
            }).into_response()
        }
    }
}
