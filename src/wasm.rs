//! WebAssembly / browser bindings for Glint.
//!
//! Exposes a `GlintModel` class to JavaScript via `wasm-bindgen`.
//! The model is loaded from an `ArrayBuffer` (fetched by the JS caller),
//! which is passed as a `Uint8Array` and converted to `Vec<u8>` here.
//!
//! # Build
//! ```sh
//! wasm-pack build --target web --features wasm
//! ```
//!
//! # JavaScript usage
//! ```js
//! import init, { GlintModel } from './pkg/glint.js';
//!
//! await init();
//!
//! const resp  = await fetch('model.gguf');
//! const bytes = new Uint8Array(await resp.arrayBuffer());
//! const model = new GlintModel(bytes);
//!
//! const output = model.generate("The meaning of life is", 64, 0.8);
//! console.log(output);
//! ```
//!
//! # WASM compilation notes
//! The following are automatically handled for the wasm32 target:
//! - `GgufModel::from_bytes()` replaces `load()` (no filesystem)
//! - Sequential inference (rayon is not used on wasm32)
//! - The HTTP server and CLI are not compiled into the WASM binary
//!
//! To fully compile for `wasm32-unknown-unknown`, the rayon usage in
//! `forward_prefill_inner` must be replaced with sequential iterators
//! (`cfg(target_arch = "wasm32")` guards).  See `src/transformer/forward.rs`.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::{Sampler, SamplerConfig};
use crate::transformer::{TransformerWeights, generate_cached, generate_greedy_cached};

// ── Panic hook ────────────────────────────────────────────────────────────────

/// Install a panic hook that forwards Rust panics to `console.error`.
/// Call this once from JS before using the model.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// ── GlintModel ────────────────────────────────────────────────────────────────

/// A loaded LLM model ready for inference in the browser.
///
/// Construct with `new GlintModel(bytes)` where `bytes` is a `Uint8Array`
/// containing the raw `.gguf` file data.
#[wasm_bindgen]
pub struct GlintModel {
    weights:   TransformerWeights,
    config:    ModelConfig,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl GlintModel {
    /// Load a GGUF model from a byte array.
    ///
    /// ```js
    /// const resp  = await fetch('model.gguf');
    /// const bytes = new Uint8Array(await resp.arrayBuffer());
    /// const model = new GlintModel(bytes);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<GlintModel, JsValue> {
        let model = GgufModel::from_bytes(bytes.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let config = ModelConfig::from_metadata(&model.metadata)
            .ok_or_else(|| JsValue::from_str("could not read model config from GGUF metadata"))?;

        let tokenizer = Tokenizer::from_gguf(&model)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let weights = TransformerWeights::load(&model, &config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(GlintModel { weights, config, tokenizer })
    }

    /// Generate text continuing `prompt`.
    ///
    /// * `max_tokens`  — maximum new tokens to generate
    /// * `temperature` — sampling temperature; 0.0 = greedy (deterministic)
    ///
    /// Returns only the newly generated text (not the prompt).
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        tokens.insert(0, self.tokenizer.bos_token_id);

        let output = if temperature > 0.0 {
            let mut sampler = Sampler::new(SamplerConfig {
                temperature,
                ..Default::default()
            });
            generate_cached(
                &self.weights, &self.config, &tokens,
                max_tokens, &mut sampler, self.tokenizer.eos_token_id, &mut None,
            )
        } else {
            generate_greedy_cached(
                &self.weights, &self.config, &tokens, max_tokens, &mut None,
            )
        };

        self.tokenizer.decode(&output[tokens.len()..])
    }

    /// Return the model's context length (maximum tokens it can process).
    pub fn context_length(&self) -> u32 {
        self.config.context_length
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }

    /// Return the model architecture string (e.g. `"llama"`).
    pub fn architecture(&self) -> String {
        self.config.architecture.clone()
    }

    /// Generate text with per-token streaming.
    ///
    /// `on_token` is called with the decoded text of each new token as it is
    /// generated. Designed for use in a Web Worker — calling `postMessage`
    /// inside the callback queues messages for the main thread while inference
    /// runs, giving the appearance of streaming.
    ///
    /// ```js
    /// model.generate_streaming(prompt, 256, 0.8, (text) => {
    ///     self.postMessage({ type: 'token', text });
    /// });
    /// ```
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_token: &js_sys::Function,
    ) {
        let mut tokens = self.tokenizer.encode(prompt);
        tokens.insert(0, self.tokenizer.bos_token_id);

        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            ..Default::default()
        });

        let eos = self.tokenizer.eos_token_id;
        let tokenizer = &self.tokenizer;

        crate::transformer::generate_streaming(
            &self.weights,
            &self.config,
            &tokens,
            max_tokens,
            &mut sampler,
            eos,
            |token_id| {
                if token_id == eos {
                    return true;
                }
                let text = tokenizer.decode(&[token_id]);
                let _ = on_token.call1(&JsValue::null(), &JsValue::from_str(&text));
                true
            },
            &mut None,
        );
    }
}
