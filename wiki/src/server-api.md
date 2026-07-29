# HTTP Server API

Glint exposes an OpenAI-compatible HTTP API. Any client that works with the OpenAI API works with Glint without modification.

Source: `src/server/routes.rs`, `src/server/types.rs`

---

## Starting the Server

```bash
glint serve -f model.gguf -p 8080
```

The model name is derived from the file stem. Use this name in API requests.

---

## Output Constraints

`POST /v1/completions` and `POST /v1/chat/completions` support OpenAI-style:

```json
{ "response_format": { "type": "json_object" } }
```

When present, Glint constrains decoding so the response is a valid JSON object. The `text` format is the default when `response_format` is omitted.

---

## Endpoints

### `GET /health`

Liveness check. Returns 200 OK immediately.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### `GET /v1/models`

List the loaded model.

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [{
    "id": "smollm-135m-instruct.Q8_0",
    "object": "model",
    "created": 0,
    "owned_by": "glint"
  }]
}
```

### `GET /v1/metrics`

Runtime metrics.

```bash
curl http://localhost:8080/v1/metrics
```

```json
{
  "requests_total": 42,
  "tokens_generated": 3871,
  "avg_latency_ms": 234.7,
  "uptime_secs": 1802
}
```

---

### `POST /v1/completions`

Text completion. Continues a prompt string.

**Request:**
```json
{
  "model": "smollm-135m-instruct.Q8_0",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "seed": 42,
  "stream": false,
  "response_format": { "type": "json_object" }
}
```

All fields except `model` and `prompt` are optional with sensible defaults.

**Non-streaming response:**
```json
{
  "id": "cmpl-17a3b2c4d5e6",
  "object": "text_completion",
  "created": 1712345678,
  "model": "smollm-135m-instruct.Q8_0",
  "choices": [{
    "text": "{\"capital\":\"Paris\"}",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 7,
    "total_tokens": 14
  }
}
```

**Streaming (`"stream": true`):**

Returns `text/event-stream` (SSE). Each token arrives as:
```
data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"...","choices":[{"text":" Paris","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"...","choices":[{"text":",","index":0,"finish_reason":null}]}

...

data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"...","choices":[{"text":"","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

---

### `POST /v1/chat/completions`

Chat completion. Accepts a list of messages and returns a response.

**Request:**
```json
{
  "model": "smollm-135m-instruct.Q8_0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"}
  ],
  "max_tokens": 100,
  "temperature": 0.5,
  "stream": false,
  "response_format": { "type": "json_object" }
}
```

**Non-streaming response:**
```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "smollm-135m-instruct.Q8_0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "{\"answer\":4}"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 23,
    "completion_tokens": 8,
    "total_tokens": 31
  }
}
```

**Streaming (`"stream": true`):**

Chat streaming follows the delta format. First chunk sends the role:
```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
```

Subsequent chunks send content tokens:
```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"2 + 2"},"finish_reason":null}]}
```

Final chunk:
```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### `POST /v1/embeddings`

Generate a text embedding. Returns a single float vector of dimension `embedding_length` (model's hidden size).

**Request:**
```json
{
  "model": "smollm-135m-instruct.Q8_0",
  "input": "The quick brown fox"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, 0.789, ...],
    "index": 0
  }],
  "model": "smollm-135m-instruct.Q8_0",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 0,
    "total_tokens": 5
  }
}
```

The embedding is the mean-pooled final hidden state across all token positions.

---

## Error Responses

All errors return a JSON body:

```json
{
  "error": {
    "message": "model 'unknown-model' not found; available: smollm-135m-instruct.Q8_0",
    "type": "api_error",
    "code": 404
  }
}
```

| HTTP status | Cause |
|-------------|-------|
| 400 | Invalid request (empty messages, prompt exceeds context window) |
| 404 | Model name doesn't match the loaded model |
| 503 | Inference engine unavailable (startup race) |
| 500 | Internal error (embedding task panic) |

---

## SSE Streaming Invariants

These behaviors are guaranteed and clients can rely on them:

1. Non-streaming is the default (`stream: false` if omitted)
2. A stream that ran to completion ends with a final chunk carrying a
   `finish_reason`, followed by `data: [DONE]`
3. `finish_reason` is `"stop"` when the model emitted its EOS token, and
   `"length"` when generation was cut short by `max_tokens` or by the model's
   context window
4. Content chunks have `finish_reason: null`; only the final chunk sets it
5. Chat streams always start with a role-only delta chunk

### Truncated streams

There is one case where a stream does **not** end with `[DONE]`.

The engine never blocks on delivery, so a client that falls far enough behind
is evicted and its undelivered tokens are dropped (see
[Backpressure](#backpressure-and-slow-clients) below). The content such a client
received is a strict *prefix* of what was generated. Terminating that stream
normally would make a partial response indistinguishable from a complete one —
including for clients that only wait for `[DONE]` and never inspect
`finish_reason` — so instead the stream ends with an SSE `error` event and no
terminator:

```text
event: error
data: {"error":{"message":"The response was truncated: ...","type":"server_error","code":"truncated"}}
```

Key off `error.code == "truncated"`. The absence of `[DONE]` is itself the
signal: **treat a stream that ends without `[DONE]` as incomplete.**

On the non-streaming endpoints the same condition returns **HTTP 500** rather
than a short 200, for the same reason. On `/v1/responses`, which signals
outcome through typed events instead of `[DONE]`, a truncated stream emits
`response.failed` in place of both `response.output_text.done` and
`response.completed`, and a non-streaming truncation is likewise a 500;
a `"length"` outcome maps to `status: "incomplete"`.

---

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="smollm-135m-instruct.Q8_0",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Inference Engine

The server uses a background inference engine (`src/server/engine.rs`) that:
- Runs on a dedicated OS thread (avoiding async runtime blocking)
- Maintains a request queue
- Processes requests sequentially (one KV cache per request)
- Sends tokens back through per-request `mpsc` channels

The engine is wrapped in `Arc<InferenceEngine>` and shared across all route handlers.

### Backpressure and slow clients

Token delivery to clients never blocks the decode loop. On each generated token the
engine performs a non-blocking `try_deliver` per active sequence, so one slow or
stalled reader can no longer freeze decoding for everyone else:

- **Healthy client** — the token is pushed onto the per-sequence outbound channel
  and decoding continues.
- **Disconnected client** — if the receiver has been dropped (the client hung up),
  the sequence is finished immediately and its work is reclaimed.
- **Too-slow client** — if a sequence's undelivered backlog grows past
  `MAX_PENDING_TOKENS` (4096), the client is treated as unable to keep up and its
  sequence is evicted. A finished sequence is also given a bounded `DRAIN_TIMEOUT`
  (10 s) to accept its remaining tokens before the undelivered tail is dropped.

Eviction costs the client the tail of its response, so it must be observable.
A closed token channel on its own cannot say *why* it closed, so every sequence
carries an out-of-band outcome — `Stop`, `Length`, `Truncated`, or `Incomplete`
— that the engine records **before** dropping the sender. The HTTP layer reads
it once the token stream ends and terminates the response accordingly: normally
for the first two, and as a [truncated stream](#truncated-streams) for the
others. Any sequence that leaves the engine without recording an outcome (a
rejected prompt, or a decode panic) reads `Incomplete` and is likewise reported
as a failure rather than as a successful empty completion.

### Fault isolation

The engine's decode loop runs under `catch_unwind`. If a decode step panics, the
panic is logged as `FATAL`, every in-flight sequence is dropped (each client
observes its stream ending), and the engine loop is respawned rather than silently
zombieing the server. Those dropped sequences never recorded an outcome, so their
clients see a truncation error rather than a clean end-of-stream.
