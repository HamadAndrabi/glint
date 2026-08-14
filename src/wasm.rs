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
//! `GlintModel` uses the `Model`-level session API internally.
//! It does NOT use the background `InferenceEngine` thread — WASM is
//! single-threaded and the engine requires `std::thread::spawn`.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use crate::api::{GenerationOptions, Model};
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::SamplerConfig;
use crate::session::CacheFormat;
use crate::transformer::TransformerWeights;

use std::sync::Arc;

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
    model: Model,
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
        let gguf =
            GgufModel::from_bytes(bytes.to_vec()).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let config = ModelConfig::from_metadata(&gguf.metadata)
            .ok_or_else(|| JsValue::from_str("could not read model config from GGUF metadata"))?;

        let tokenizer =
            Tokenizer::from_gguf(&gguf).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let weights = TransformerWeights::load(&gguf, &config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let model = Model {
            weights: Arc::new(weights),
            config: Arc::new(config),
            tokenizer: Arc::new(tokenizer),
            model_hash: 0,
            adapter_registry: crate::model::lora_registry::AdapterRegistry::new(),
        };

        Ok(GlintModel { model })
    }

    /// Generate text continuing `prompt`.
    ///
    /// * `max_tokens`  — maximum new tokens to generate
    /// * `temperature` — sampling temperature; 0.0 = greedy (deterministic)
    ///
    /// Returns only the newly generated text (not the prompt).
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, JsValue> {
        let opts = GenerationOptions {
            max_new_tokens: max_tokens,
            sampler_cfg: SamplerConfig {
                temperature,
                ..Default::default()
            },
            cache_format: CacheFormat::F32,
            constraint: None,
            lora_adapter: None,
        };
        let new_tokens = self
            .model
            .generate(prompt, &opts, &mut None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.model.decode(&new_tokens))
    }

    /// Return the model's context length (maximum tokens it can process).
    pub fn context_length(&self) -> u32 {
        self.model.config.context_length
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.model.config.vocab_size
    }

    /// Return the model architecture string (e.g. `"llama"`).
    pub fn architecture(&self) -> String {
        self.model.config.architecture.clone()
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
    ) -> Result<(), JsValue> {
        let opts = GenerationOptions {
            max_new_tokens: max_tokens,
            sampler_cfg: SamplerConfig {
                temperature,
                ..Default::default()
            },
            cache_format: CacheFormat::F32,
            constraint: None,
            lora_adapter: None,
        };

        self.model
            .generate_streaming(
                prompt,
                &opts,
                |tok| {
                    let text = self.model.decode(&[tok]);
                    let _ = on_token.call1(&JsValue::null(), &JsValue::from_str(&text));
                    true // always continue; caller stops via EOS or budget
                },
                &mut None,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }

    /// Generate output constrained to a valid JSON object (`{...}`).
    pub fn generate_json_object(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, JsValue> {
        let opts = GenerationOptions {
            max_new_tokens: max_tokens,
            sampler_cfg: SamplerConfig {
                temperature,
                ..Default::default()
            },
            cache_format: CacheFormat::F32,
            constraint: Some(crate::constrained::ConstraintSpec::JsonObject),
            lora_adapter: None,
        };
        let new_tokens = self
            .model
            .generate(prompt, &opts, &mut None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.model.decode(&new_tokens))
    }

    /// Generate output strictly conforming to a JSON Schema string.
    pub fn generate_json_schema(
        &self,
        prompt: &str,
        json_schema_str: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, JsValue> {
        let schema_val: serde_json::Value = serde_json::from_str(json_schema_str)
            .map_err(|e| JsValue::from_str(&format!("invalid JSON schema: {e}")))?;

        let opts = GenerationOptions {
            max_new_tokens: max_tokens,
            sampler_cfg: SamplerConfig {
                temperature,
                ..Default::default()
            },
            cache_format: CacheFormat::F32,
            constraint: Some(crate::constrained::ConstraintSpec::JsonSchema(schema_val)),
            lora_adapter: None,
        };
        let new_tokens = self
            .model
            .generate(prompt, &opts, &mut None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.model.decode(&new_tokens))
    }

    /// Generate output strictly conforming to a GBNF grammar string.
    pub fn generate_grammar(
        &self,
        prompt: &str,
        grammar_str: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, JsValue> {
        let opts = GenerationOptions {
            max_new_tokens: max_tokens,
            sampler_cfg: SamplerConfig {
                temperature,
                ..Default::default()
            },
            cache_format: CacheFormat::F32,
            constraint: Some(crate::constrained::ConstraintSpec::Grammar(
                grammar_str.to_string(),
            )),
            lora_adapter: None,
        };
        let new_tokens = self
            .model
            .generate(prompt, &opts, &mut None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.model.decode(&new_tokens))
    }
}
