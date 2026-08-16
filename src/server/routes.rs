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

use super::engine::{Finish, FinishSignal};
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
    if requested != loaded
        && requested != "default"
        && requested != "auto"
        && requested != "glint"
        && !requested.is_empty()
    {
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
    /// Why the sequence ended — only meaningful once `rx` has closed.
    finish: FinishSignal,
    prompt_len: usize,
    model_name: String,
}

struct CompletedGeneration {
    text: String,
    completion_tokens: usize,
    /// `Some(reason)` for a complete response, `None` if it was truncated.
    finish_reason: Option<&'static str>,
}

/// Map an engine outcome to its OpenAI `finish_reason`.
///
/// `None` means there is no honest value: the client received a strict prefix
/// of the completion, so the response must not be presented as finished. See
/// [`Finish::Truncated`].
fn finish_reason_for(finish: Finish) -> Option<&'static str> {
    match finish {
        Finish::Stop => Some("stop"),
        Finish::Length => Some("length"),
        Finish::Truncated | Finish::Incomplete => None,
    }
}

/// Body of the SSE `error` event that replaces the normal terminator when a
/// stream is cut short.
const TRUNCATED_SSE_BODY: &str = concat!(
    r#"{"error":{"message":"The response was truncated: the server dropped "#,
    r#"undelivered tokens because this client could not keep up with the "#,
    r#"stream. The content received is incomplete.","#,
    r#""type":"server_error","code":"truncated"}}"#
);

/// Terminal SSE event for a stream that was cut short.
///
/// Deliberately *not* followed by `[DONE]`: a client that only waits for the
/// terminator would otherwise treat a partial response as a complete one, which
/// is the whole failure this signalling exists to prevent.
fn truncated_sse_event() -> Event {
    Event::default().event("error").data(TRUNCATED_SSE_BODY)
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

    let prompt_tokens = state.tokenizer.encode_prompt(prompt);

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

fn format_tools_system_prompt(tools: &[Tool]) -> String {
    let mut s = String::from("\nYou have access to the following tools/functions:\n```json\n[\n");
    for (i, tool) in tools.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        let serialized = serde_json::to_string_pretty(tool).unwrap_or_default();
        s.push_str(&serialized);
    }
    s.push_str("\n]\n```\nTo call a tool, respond with a JSON object format: `{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}`.\n");
    s
}

fn resolve_constraint(response_format: Option<&ResponseFormat>) -> Option<ConstraintSpec> {
    let rf = response_format?;
    if rf.is_json_object() {
        Some(ConstraintSpec::JsonObject)
    } else if rf.is_json_schema() {
        rf.json_schema
            .as_ref()
            .and_then(|js| js.schema.clone())
            .map(ConstraintSpec::JsonSchema)
    } else if rf.is_grammar() {
        rf.grammar.clone().map(ConstraintSpec::Grammar)
    } else {
        None
    }
}

fn resolve_tool_constraint(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
) -> Option<ConstraintSpec> {
    match tool_choice {
        Some(ToolChoice::Mode(mode)) if mode == "none" => None,
        Some(ToolChoice::Function { function, .. }) => {
            let tool = tools.iter().find(|t| t.function.name == function.name)?;
            let params = tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({"type": "object"}));
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "enum": [function.name] },
                    "arguments": params
                },
                "required": ["name", "arguments"]
            });
            Some(ConstraintSpec::JsonSchema(schema))
        }
        Some(ToolChoice::Mode(mode)) if mode == "required" => {
            let mut variants = Vec::new();
            for t in tools {
                let params = t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object"}));
                variants.push(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "enum": [t.function.name] },
                        "arguments": params
                    },
                    "required": ["name", "arguments"]
                }));
            }
            if !variants.is_empty() {
                Some(ConstraintSpec::JsonSchema(
                    serde_json::json!({ "anyOf": variants }),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn maybe_parse_tool_call(text: &str, tools: Option<&[Tool]>) -> Option<Vec<ToolCall>> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    let trimmed = text.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let val: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let obj = val.as_object()?;
    let name = obj.get("name")?.as_str()?;
    if !tools.iter().any(|t| t.function.name == name) {
        return None;
    }
    let args_val = obj.get("arguments")?;
    let arguments = match args_val {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };
    Some(vec![ToolCall {
        id: gen_id_with_prefix("call"),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments,
        },
    }])
}

#[allow(clippy::too_many_arguments)]
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
    tools: Option<&[Tool]>,
) -> Result<PreparedGeneration, Response> {
    if messages.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
        ));
    }

    let mut owned_messages: Vec<(String, String)> = Vec::new();
    let mut has_system = false;

    for m in messages {
        let content_str = m.content_str();
        if m.role == "system" {
            has_system = true;
            if let Some(t) = tools {
                if !t.is_empty() {
                    let tool_prompt = format_tools_system_prompt(t);
                    owned_messages
                        .push((m.role.clone(), format!("{}{}", content_str, tool_prompt)));
                    continue;
                }
            }
        }
        owned_messages.push((m.role.clone(), content_str.to_string()));
    }

    if !has_system {
        if let Some(t) = tools {
            if !t.is_empty() {
                let tool_prompt = format_tools_system_prompt(t);
                owned_messages.insert(0, ("system".to_string(), tool_prompt));
            }
        }
    }

    let msgs: Vec<Message<'_>> = owned_messages
        .iter()
        .map(|(role, content)| Message {
            role: role.as_str(),
            content: content.as_str(),
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
        None,
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

    let submitted = state
        .engine
        .submit(
            prompt_tokens,
            max_tokens,
            sampler_cfg,
            eos,
            constraint,
            None,
        )
        .map_err(|e| {
            state.metrics.record_failure();
            match e {
                crate::server::engine::SubmitError::Busy => api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server is at capacity; retry shortly",
                ),
                crate::server::engine::SubmitError::Shutdown => api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Inference engine unavailable",
                ),
            }
        })?;

    state.metrics.on_submit();

    Ok(StartedGeneration {
        rx: submitted.rx,
        finish: submitted.finish,
        prompt_len,
        model_name,
    })
}

async fn collect_generation(
    state: Arc<AppState>,
    rx: tokio::sync::mpsc::Receiver<u32>,
    finish: FinishSignal,
) -> CompletedGeneration {
    let t0 = Instant::now();
    let mut stream = ReceiverStream::new(rx);
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut ttft_recorded = false;
    while let Some(token_id) = stream.next().await {
        if !ttft_recorded {
            state
                .metrics
                .record_ttft_us(t0.elapsed().as_micros() as u64);
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

    // Safe to read now: the stream above ran to completion, so the engine has
    // dropped the sender — and it records the outcome before doing so.
    CompletedGeneration {
        text,
        completion_tokens,
        finish_reason: finish_reason_for(finish.get()),
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
///
/// KV-memory reporting is **additive and optional**: `kv_pool` appears only
/// when the engine runs a paged pool, `prefix_cache` only when prefix reuse is
/// enabled. Every other field is present unconditionally, as before.
pub async fn server_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let m = &state.metrics;
    let requests = m.requests_total.load(Ordering::Relaxed);
    let tokens = m.tokens_generated.load(Ordering::Relaxed);
    let failed = m.requests_failed.load(Ordering::Relaxed);
    let active = m.active_sessions.load(Ordering::Relaxed);
    let queued = m.queue_depth.load(Ordering::Relaxed);
    let uptime_secs = m.started_at.elapsed().as_secs();

    let mut body = serde_json::json!({
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
    });

    // Sampled by the engine when sequences are admitted or retired, so these
    // are exact as of the last such boundary rather than continuously live.
    let kv = state.engine.kv_stats();
    if let Some(pool) = kv.pool {
        body["kv_pool"] = serde_json::json!({
            "capacity":   pool.capacity,
            "live":       pool.live,
            "peak_live":  pool.peak_live,
            "pooled":     pool.pooled,
        });
    }
    if let Some(prefix) = kv.prefix {
        body["prefix_cache"] = serde_json::json!({
            "hits":           prefix.hits,
            "misses":         prefix.misses,
            "evictions":      prefix.evictions,
            "tokens_reused":  prefix.tokens_reused,
            "entries":        prefix.entries,
            "pages":          prefix.pages,
        });
    }
    Json(body)
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

    if let Some(spec) = resolve_constraint(req.response_format.as_ref()) {
        if let Err(msg) = spec.validate() {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("invalid response_format: {msg}"),
            );
        }
        prepared.constraint = Some(spec);
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

    let completed = collect_generation(Arc::clone(&state), started.rx, started.finish).await;

    // A truncated buffer would serialize as a short-but-valid completion, which
    // the caller cannot tell from a real one. Fail the request instead.
    let Some(finish_reason) = completed.finish_reason else {
        state.metrics.record_failure();
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The response was truncated before it could be fully collected",
        );
    };

    Json(CompletionResponse {
        id: gen_id(),
        object: "text_completion",
        created: now_secs(),
        model: started.model_name,
        choices: vec![CompletionChoice {
            text: completed.text,
            index: 0,
            finish_reason,
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
    let ttft_instant = t0;
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
            state_for_stream
                .metrics
                .record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
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

    // Terminator. A completed stream gets the usual final chunk carrying
    // finish_reason ("stop" or "length") followed by [DONE]; a truncated one
    // gets an error event and no [DONE]. Evaluated lazily — these closures run
    // only after the token stream above has drained, which is exactly when the
    // engine's outcome becomes readable.
    let tc = Arc::clone(&token_count);
    let finish_signal = started.finish;
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).filter_map(move |i| {
        let reason = finish_reason_for(finish_signal.get());
        if i == 0 {
            let event = match reason {
                Some(reason) => {
                    let chunk = CompletionChunk {
                        id: finish_id.clone(),
                        object: "text_completion",
                        created,
                        model: finish_model.clone(),
                        choices: vec![ChunkChoice {
                            text: String::new(),
                            index: 0,
                            finish_reason: Some(reason),
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                None => truncated_sse_event(),
            };
            Some(Ok::<Event, Infallible>(event))
        } else {
            // Record metrics either way — a truncated stream still consumed
            // engine time, and skipping this would leak the in-flight gauge.
            state
                .metrics
                .record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();
            if reason.is_none() {
                state.metrics.record_failure();
            }
            reason.map(|_| Ok::<Event, Infallible>(Event::default().data("[DONE]")))
        }
    });

    let done_stream = stream.chain(finish_stream);

    Sse::new(done_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn streaming_chat_completion(
    state: Arc<AppState>,
    prepared: PreparedGeneration,
    is_tool_call_mode: bool,
    specific_tool_name: Option<String>,
) -> Response {
    let started = match start_generation(&state, prepared) {
        Ok(started) => started,
        Err(err) => return err,
    };

    let tokenizer = Arc::clone(&state.tokenizer);
    let token_count = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let ttft_instant = t0;
    let id = gen_id();
    let created = now_secs();
    let call_id = gen_id_with_prefix("call");

    // Clone for the finish chunk (originals are moved into the content closure)
    let finish_id = id.clone();
    let stream_model = started.model_name.clone();
    let finish_model = started.model_name.clone();

    // First chunk: send the role (+ initial tool call delta if in tool call mode)
    let role_chunk = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: stream_model.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: if is_tool_call_mode {
                ChatDelta {
                    role: Some("assistant"),
                    content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: Some(call_id.clone()),
                        tool_type: Some("function".to_string()),
                        function: Some(FunctionCallDelta {
                            name: specific_tool_name.clone(),
                            arguments: Some(String::new()),
                        }),
                    }]),
                }
            } else {
                ChatDelta {
                    role: Some("assistant"),
                    content: None,
                    tool_calls: None,
                }
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
            state_for_stream
                .metrics
                .record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
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
                delta: if is_tool_call_mode {
                    ChatDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ToolCallDelta {
                            index: 0,
                            id: None,
                            tool_type: None,
                            function: Some(FunctionCallDelta {
                                name: None,
                                arguments: Some(text),
                            }),
                        }]),
                    }
                } else {
                    ChatDelta {
                        role: None,
                        content: Some(text),
                        tool_calls: None,
                    }
                },
                finish_reason: None,
            }],
        };
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok::<Event, Infallible>(Event::default().data(data))
    });

    // Terminator — see `streaming_completion` for the contract: final chunk +
    // [DONE] when the completion finished, error event and no [DONE] when it
    // was truncated.
    let tc = Arc::clone(&token_count);
    let finish_signal = started.finish;
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).filter_map(move |i| {
        let raw_reason = finish_reason_for(finish_signal.get());
        let reason = if is_tool_call_mode && raw_reason == Some("stop") {
            Some("tool_calls")
        } else {
            raw_reason
        };
        if i == 0 {
            let event = match reason {
                Some(reason) => {
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
                                tool_calls: None,
                            },
                            finish_reason: Some(reason),
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                None => truncated_sse_event(),
            };
            Some(Ok::<Event, Infallible>(event))
        } else {
            state
                .metrics
                .record(tc.load(Ordering::Relaxed), t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();
            if reason.is_none() {
                state.metrics.record_failure();
            }
            reason.map(|_| Ok::<Event, Infallible>(Event::default().data("[DONE]")))
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
    let tools = match req.tool_choice.as_ref() {
        Some(ToolChoice::Mode(mode)) if mode == "none" => None,
        _ => req.tools.as_deref(),
    };

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
        tools,
    ) {
        Ok(prepared) => prepared,
        Err(err) => return err,
    };

    let mut is_tool_call_mode = false;
    let mut specific_tool_name = None;

    if let Some(spec) = resolve_constraint(req.response_format.as_ref()) {
        if let Err(msg) = spec.validate() {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("invalid response_format: {msg}"),
            );
        }
        prepared.constraint = Some(spec);
    } else if let Some(t) = tools {
        if let Some(tool_constraint) = resolve_tool_constraint(t, req.tool_choice.as_ref()) {
            if let Err(msg) = tool_constraint.validate() {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    format!("invalid tools schema: {msg}"),
                );
            }
            prepared.constraint = Some(tool_constraint);
            is_tool_call_mode = true;
            if let Some(ToolChoice::Function { function, .. }) = req.tool_choice.as_ref() {
                specific_tool_name = Some(function.name.clone());
            } else if t.len() == 1 {
                specific_tool_name = Some(t[0].function.name.clone());
            }
        }
    }

    if req.stream.unwrap_or(false) {
        streaming_chat_completion(state, prepared, is_tool_call_mode, specific_tool_name).await
    } else {
        let started = match start_generation(&state, prepared) {
            Ok(started) => started,
            Err(err) => return err,
        };

        let completed = collect_generation(Arc::clone(&state), started.rx, started.finish).await;

        let Some(finish_reason) = completed.finish_reason else {
            state.metrics.record_failure();
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "The response was truncated before it could be fully collected",
            );
        };

        let tool_calls = maybe_parse_tool_call(&completed.text, tools);
        let finish_reason = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason
        };

        let message = if let Some(tc) = tool_calls {
            ChatMessageOut {
                role: "assistant",
                content: None,
                tool_calls: Some(tc),
            }
        } else {
            ChatMessageOut {
                role: "assistant",
                content: Some(completed.text),
                tool_calls: None,
            }
        };

        Json(ChatCompletionResponse {
            id: gen_id(),
            object: "chat.completion",
            created: now_secs(),
            model: started.model_name,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason,
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
        let completed = collect_generation(Arc::clone(&state), started.rx, started.finish).await;

        // The Responses API carries outcome in `status` rather than
        // `finish_reason`: a budget/context stop is "incomplete", not a failure.
        // A truncated response is a genuine server error either way.
        let status = match completed.finish_reason {
            Some("length") => "incomplete",
            Some(_) => "completed",
            None => {
                state.metrics.record_failure();
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "The response was truncated before it could be fully collected",
                );
            }
        };

        Json(build_responses_response(
            id,
            message_id,
            created_at,
            started.model_name,
            completed.text,
            started.prompt_len,
            completed.completion_tokens,
            status,
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

    let ttft_instant = t0;
    let tc = Arc::clone(&token_count);
    let text_buf = Arc::clone(&full_text);
    let delta_response_id = response_id.clone();
    let ttft_sent = Arc::new(AtomicBool::new(false));
    let ttft_sent_clone = Arc::clone(&ttft_sent);
    let state_for_stream = Arc::clone(&state);
    let delta_stream = ReceiverStream::new(started.rx).map(move |token_id| {
        if !ttft_sent_clone.swap(true, Ordering::Relaxed) {
            state_for_stream
                .metrics
                .record_ttft_us(ttft_instant.elapsed().as_micros() as u64);
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
    // This surface signals completion with typed events rather than [DONE], so
    // a truncated stream emits `response.failed` in place of both terminal
    // events — never an `output_text.done` carrying a partial body, and never a
    // `response.completed`.
    let finish_signal = started.finish;
    let finish_stream = tokio_stream::iter(vec![0u8, 1u8]).filter_map(move |i| {
        let text = text_buf.lock().map(|buf| buf.clone()).unwrap_or_default();
        let reason = finish_reason_for(finish_signal.get());

        if i == 0 {
            // Truncated: suppress the "done" event entirely rather than
            // announce a partial body as final.
            reason?;
            let payload = serde_json::json!({
                "type": "response.output_text.done",
                "response_id": done_response_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            });
            Some(Ok::<Event, Infallible>(
                Event::default()
                    .event("response.output_text.done")
                    .data(payload.to_string()),
            ))
        } else {
            let output_tokens = tc.load(Ordering::Relaxed) as usize;
            state
                .metrics
                .record(output_tokens as u64, t0.elapsed().as_millis() as u64);
            state.metrics.on_complete();

            let Some(reason) = reason else {
                state.metrics.record_failure();
                let payload = serde_json::json!({
                    "type": "response.failed",
                    "response_id": completed_response_id.clone(),
                    "error": {
                        "message": "The response was truncated: the server dropped \
                                    undelivered tokens because this client could not \
                                    keep up with the stream.",
                        "type": "server_error",
                        "code": "truncated",
                    },
                });
                return Some(Ok::<Event, Infallible>(
                    Event::default()
                        .event("response.failed")
                        .data(payload.to_string()),
                ));
            };

            let response = build_responses_response(
                completed_response_id.clone(),
                message_id.clone(),
                created_at,
                finish_model.clone(),
                text,
                prompt_len,
                output_tokens,
                if reason == "length" {
                    "incomplete"
                } else {
                    "completed"
                },
            );
            let payload = serde_json::json!({
                "type": "response.completed",
                "response": response,
            });
            Some(Ok::<Event, Infallible>(
                Event::default()
                    .event("response.completed")
                    .data(payload.to_string()),
            ))
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
    let all_tokens: Vec<Vec<u32>> = input_texts
        .iter()
        .map(|text| state.tokenizer.encode_prompt(text))
        .collect();

    let total_tokens: usize = all_tokens.iter().map(|t| t.len()).sum();
    let model_name = state.model_name.clone();

    let weights = Arc::clone(&state.weights);
    let config = Arc::clone(&state.config);

    let result = tokio::task::spawn_blocking(move || {
        let refs: Vec<&[u32]> = all_tokens.iter().map(Vec::as_slice).collect();
        embed_batch(&weights, &config, &refs)
    })
    .await;

    match result {
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "Embedding task panicked"),
        Ok(vectors) => {
            let data: Vec<EmbeddingData> = vectors
                .into_iter()
                .enumerate()
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
            })
            .into_response()
        }
    }
}

// ── GET / and GET /ui ─────────────────────────────────────────────────────────

/// Embedded Single-Page Application (SPA) Web Dashboard.
pub async fn web_ui() -> Response {
    axum::response::Html(include_str!("web/index.html")).into_response()
}

// ── GET /assets/logo.svg ──────────────────────────────────────────────────────

/// Serves the official Glint logo SVG.
pub async fn logo_svg() -> Response {
    let svg = include_str!("../../assets/logo.svg");
    axum::response::Response::builder()
        .header("Content-Type", "image/svg+xml")
        .header("Cache-Control", "public, max-age=86400")
        .body(axum::body::Body::from(svg))
        .unwrap_or_else(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "failed to load logo"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_web_ui_returns_html() {
        let resp = web_ui().await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_logo_svg_returns_svg() {
        let resp = logo_svg().await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // The mapping that decides how a stream terminates. `None` is the load-
    // bearing case: it is what makes a truncated response fail loudly instead
    // of arriving as a short, complete-looking one.
    #[test]
    fn finish_reason_mapping() {
        assert_eq!(finish_reason_for(Finish::Stop), Some("stop"));
        assert_eq!(finish_reason_for(Finish::Length), Some("length"));
        assert_eq!(finish_reason_for(Finish::Truncated), None);
        assert_eq!(finish_reason_for(Finish::Incomplete), None);
    }

    // The truncation payload is assembled from string literals, so a stray
    // quote would ship malformed JSON to every truncated stream. Clients key
    // off `error.code`, not the prose, so pin both.
    #[test]
    fn truncated_sse_body_is_valid_json() {
        let parsed: serde_json::Value = serde_json::from_str(TRUNCATED_SSE_BODY)
            .expect("truncation payload must be valid JSON");
        assert_eq!(parsed["error"]["code"], "truncated");
        assert_eq!(parsed["error"]["type"], "server_error");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("truncated")),
            "message should say plainly that the response was truncated"
        );
    }
}
