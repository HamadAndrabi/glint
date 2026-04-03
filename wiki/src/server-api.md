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
  "stream": false
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
    "text": " Paris, the city of light.",
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
  "stream": false
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
      "content": "2 + 2 = 4."
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
2. The final SSE chunk before `[DONE]` always carries `"finish_reason": "stop"`
3. Every SSE stream terminates with `data: [DONE]`
4. Content chunks have `finish_reason: null`; only the final chunk has `"stop"`
5. Chat streams always start with a role-only delta chunk

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
