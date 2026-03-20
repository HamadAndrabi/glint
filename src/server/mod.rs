//! HTTP inference server — OpenAI-compatible API.
//!
//! Exposes these endpoints:
//!   GET  /health                    — health check (returns 200 OK)
//!   GET  /v1/models                 — list the loaded model
//!   GET  /v1/metrics                — runtime metrics (requests, tokens, latency, uptime)
//!   POST /v1/completions            — text completion (streaming or not)
//!   POST /v1/chat/completions       — chat completion (streaming or not)
//!   POST /v1/embeddings             — text embedding (mean-pooled hidden states)
//!
//! CORS is enabled for all origins so browser-based clients work out of the box.
//!
//! Start the server by calling `run_server(state, host, port).await`.

mod routes;
mod state;
mod types;

pub use state::{AppState, Metrics};

use std::sync::Arc;

use axum::http::Method;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

/// Start the HTTP server and block until shutdown.
///
/// Loads the model into `AppState` before calling this function, then pass
/// the state here. The server listens on `host:port` and handles requests
/// until the process exits.
pub async fn run_server(state: AppState, host: &str, port: u16) {
    let shared = Arc::new(state);

    // Allow any origin, common headers, and the methods we actually handle.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/metrics", get(routes::server_metrics))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/embeddings", post(routes::embeddings))
        .layer(cors)
        .with_state(shared);

    let addr = format!("{host}:{port}");
    eprintln!("Glint server listening on http://{addr}");
    eprintln!("  GET  http://{addr}/health");
    eprintln!("  GET  http://{addr}/v1/models");
    eprintln!("  GET  http://{addr}/v1/metrics");
    eprintln!("  POST http://{addr}/v1/completions");
    eprintln!("  POST http://{addr}/v1/chat/completions");
    eprintln!("  POST http://{addr}/v1/embeddings");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error: failed to bind {addr}: {e}");
            std::process::exit(1);
        });

    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error: server exited unexpectedly: {e}");
            std::process::exit(1);
        });
}
