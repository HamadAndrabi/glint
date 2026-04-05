/**
 * @file glint.h
 * @brief C API for the Glint LLM inference engine.
 *
 * Build the shared library with:
 *   cargo build --release --features cffi
 *
 * Typical usage:
 * @code
 *   GlintModel*   m = glint_model_load("model.gguf");
 *   GlintSamplerOptions opts = {0};
 *   opts.temperature    = 0.8;
 *   opts.max_new_tokens = 256;
 *   GlintSession* s = glint_session_new(m, &opts, "f32");
 *   uint32_t tokens[256];
 *   int n = glint_generate(m, s, "Hello", 0, tokens, 256);
 *   glint_session_free(s);
 *   glint_model_free(m);
 * @endcode
 */

#ifndef GLINT_H
#define GLINT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque types ──────────────────────────────────────────────────────────── */

/** Opaque handle to a loaded GGUF model. Free with glint_model_free(). */
typedef struct GlintModel GlintModel;

/** Opaque handle to a generation session. Free with glint_session_free(). */
typedef struct GlintSession GlintSession;

/** Opaque handle to a KV-cache snapshot. Free with glint_snapshot_free(). */
typedef struct GlintSnapshot GlintSnapshot;

/* ── Sampler options ───────────────────────────────────────────────────────── */

/**
 * Sampler parameters.  Zero-initialise to use defaults for all fields.
 *
 * @code
 *   GlintSamplerOptions opts = {0};
 *   opts.temperature    = 0.8;
 *   opts.max_new_tokens = 256;
 * @endcode
 */
typedef struct GlintSamplerOptions {
    float  temperature;      /**< 0.0 = greedy, >0 = stochastic. Default: 0. */
    size_t top_k;            /**< Top-k filtering. 0 = disabled.              */
    float  top_p;            /**< Top-p nucleus. 0.0/1.0 = disabled.          */
    float  repeat_penalty;   /**< Repetition penalty. 0.0/1.0 = disabled.     */
    uint64_t seed;           /**< PRNG seed. 0 = system time.                 */
    size_t max_new_tokens;   /**< Generation budget. 0 defaults to 256.       */
} GlintSamplerOptions;

/* ── Model lifecycle ───────────────────────────────────────────────────────── */

/**
 * Load a GGUF model from @p path.
 *
 * @return  Heap-allocated handle, or NULL on failure.
 *          Call glint_last_error() for the error message.
 */
GlintModel* glint_model_load(const char* path);

/** Free a model handle.  No-op if @p model is NULL. */
void glint_model_free(GlintModel* model);

/* ── Session lifecycle ─────────────────────────────────────────────────────── */

/**
 * Create a new generation session.
 *
 * @param model         Model to generate from (borrowed).
 * @param sampler_opts  Sampler configuration (borrowed).
 * @param cache_format  "f32" (default) or "q8".  NULL means "f32".
 * @return  Heap-allocated session, or NULL on failure.
 */
GlintSession* glint_session_new(
    const GlintModel*          model,
    const GlintSamplerOptions* sampler_opts,
    const char*                cache_format
);

/** Free a session handle.  No-op if @p session is NULL. */
void glint_session_free(GlintSession* session);

/* ── Generation ────────────────────────────────────────────────────────────── */

/**
 * Tokenise @p prompt, prefill, and decode into @p out_tokens.
 *
 * @param model          Model to run (borrowed).
 * @param session        Session to use — parameters are taken from the session.
 * @param prompt         NUL-terminated UTF-8 prompt.
 * @param max_new_tokens Maximum tokens to generate.  0 = use session setting.
 * @param out_tokens     Caller-allocated buffer for output token ids.
 * @param out_capacity   Length of @p out_tokens in uint32_t elements.
 * @return  Number of tokens written (≥ 0), or -1 on error.
 */
int glint_generate(
    const GlintModel*   model,
    GlintSession*       session,
    const char*         prompt,
    size_t              max_new_tokens,
    uint32_t*           out_tokens,
    size_t              out_capacity
);

/**
 * Streaming generation — calls @p on_token for each generated token.
 *
 * @param on_token   Callback invoked with (token_id, userdata).
 *                   Return non-zero from the callback to stop early.
 * @param userdata   Opaque pointer forwarded to @p on_token.
 * @return  Total tokens generated (≥ 0), or -1 on error.
 */
int glint_stream_generate(
    const GlintModel*   model,
    GlintSession*       session,
    const char*         prompt,
    size_t              max_new_tokens,
    int (*on_token)(uint32_t token_id, void* userdata),
    void*               userdata
);

/* ── Snapshots ─────────────────────────────────────────────────────────────── */

/**
 * Export the current session state to a snapshot.
 *
 * @return  Heap-allocated snapshot handle, or NULL on failure.
 */
GlintSnapshot* glint_snapshot_export(
    const GlintModel*   model,
    const GlintSession* session
);

/**
 * Restore a session from a snapshot.
 *
 * @param snapshot     Snapshot to restore from (model hash is verified).
 * @return  New session handle, or NULL on failure (e.g. model mismatch).
 */
GlintSession* glint_snapshot_import(
    const GlintModel*          model,
    const GlintSnapshot*       snapshot,
    const GlintSamplerOptions* sampler_opts
);

/**
 * Serialise a snapshot to a byte buffer.
 *
 * Pass buf=NULL and buf_len=0 to query the required size without writing.
 *
 * @return  Number of bytes written, or -1 if the buffer is too small.
 */
int glint_snapshot_serialize(
    const GlintSnapshot* snapshot,
    uint8_t*             buf,
    size_t               buf_len
);

/**
 * Deserialise a snapshot from @p len bytes at @p buf.
 *
 * @return  Heap-allocated snapshot handle, or NULL on failure.
 */
GlintSnapshot* glint_snapshot_deserialize(const uint8_t* buf, size_t len);

/** Free a snapshot handle.  No-op if @p snapshot is NULL. */
void glint_snapshot_free(GlintSnapshot* snapshot);

/* ── Error reporting ───────────────────────────────────────────────────────── */

/**
 * Return the last error message for the current thread.
 *
 * The returned pointer is valid until the next Glint call on this thread.
 * Returns "" (empty string) if no error has occurred.
 */
const char* glint_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GLINT_H */
