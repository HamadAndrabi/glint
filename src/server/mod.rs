//! HTTP inference server — OpenAI-compatible API.
//!
//! Exposes three endpoints:
//!   GET  /v1/models                — list the loaded model
//!   POST /v1/completions           — text completion (streaming or not)
//!   POST /v1/chat/completions      — chat completion (streaming or not)
//!
//! Start the server by calling `run_server(state, host, port).await`.

mod routes;
mod state;
mod types;

pub use state::AppState;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

/// Start the HTTP server and block until shutdown.
///
/// Loads the model into `AppState` before calling this function, then pass
/// the state here. The server listens on `host:port` and handles requests
/// until the process exits.
pub async fn run_server(state: AppState, host: &str, port: u16) {
    let shared = Arc::new(state);

    let app = Router::new()
        .route("/v1/models", get(routes::list_models))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .with_state(shared);

    let addr = format!("{host}:{port}");
    eprintln!("Ferrite server listening on http://{addr}");
    eprintln!("  GET  http://{addr}/v1/models");
    eprintln!("  POST http://{addr}/v1/completions");
    eprintln!("  POST http://{addr}/v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind {addr}: {e}"));

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
