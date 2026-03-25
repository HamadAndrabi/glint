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

use crate::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::{Sampler, SamplerConfig};
use crate::transformer::{TransformerWeights, forward_one, forward_prefill};

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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn validate_token_ids(token_ids: &[u32], embedding_rows: usize) -> Result<(), JsValue> {
    if let Some(&bad) = token_ids.iter().find(|&&id| id as usize >= embedding_rows) {
        return Err(JsValue::from_str(&format!(
            "token id {bad} is outside the embedding table (size {embedding_rows})"
        )));
    }
    Ok(())
}

fn prepare_generation(
    tokenizer: &Tokenizer,
    embedding_rows: usize,
    context_length: usize,
    prompt: &str,
    max_tokens: usize,
) -> Result<(Vec<u32>, usize), JsValue> {
    let mut tokens = tokenizer.encode(prompt);
    tokens.insert(0, tokenizer.bos_token_id);
    validate_token_ids(&tokens, embedding_rows)?;

    if tokens.len() >= context_length {
        return Err(JsValue::from_str(&format!(
            "prompt is {} tokens, but the model context length is {}",
            tokens.len(),
            context_length
        )));
    }

    let max_new_tokens = max_tokens.min(context_length - tokens.len());
    if max_new_tokens == 0 {
        return Err(JsValue::from_str("no room left to generate tokens for this prompt"));
    }

    Ok((tokens, max_new_tokens))
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
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String, JsValue> {
        let context_length = self.config.context_length as usize;
        let embedding_rows = self.weights.token_embedding.rows();
        let (mut tokens, max_new_tokens) = prepare_generation(
            &self.tokenizer,
            embedding_rows,
            context_length,
            prompt,
            max_tokens,
        )?;
        let prompt_len = tokens.len();

        // Allocate only enough KV-cache for this request instead of the model's
        // full advertised context length, which is too large for many browsers.
        let mut cache = KvCache::new(
            self.config.block_count as usize,
            prompt_len + max_new_tokens,
            self.config.head_count_kv as usize,
            self.config.head_dim() as usize,
        );

        let mut last_logits = forward_prefill(&self.weights, &self.config, &tokens, &mut cache, 0, &mut None);
        let mut sampler = (temperature > 0.0).then(|| {
            Sampler::new(SamplerConfig {
                temperature,
                ..Default::default()
            })
        });

        for _ in 0..max_new_tokens {
            let next = if let Some(sampler) = sampler.as_mut() {
                sampler.sample(last_logits.data(), &tokens)
            } else {
                argmax(last_logits.data())
            };

            if next as usize >= embedding_rows {
                return Err(JsValue::from_str(&format!(
                    "generated token id {next} is outside the embedding table (size {embedding_rows})"
                )));
            }

            tokens.push(next);
            if next == self.tokenizer.eos_token_id {
                break;
            }

            last_logits = forward_one(
                &self.weights,
                &self.config,
                next,
                tokens.len() - 1,
                &mut cache,
                &mut None,
            );
        }

        Ok(self.tokenizer.decode(&tokens[prompt_len..]))
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
    ) -> Result<(), JsValue> {
        let context_length = self.config.context_length as usize;
        let embedding_rows = self.weights.token_embedding.rows();
        let (mut tokens, max_new_tokens) = prepare_generation(
            &self.tokenizer,
            embedding_rows,
            context_length,
            prompt,
            max_tokens,
        )?;

        let mut cache = KvCache::new(
            self.config.block_count as usize,
            tokens.len() + max_new_tokens,
            self.config.head_count_kv as usize,
            self.config.head_dim() as usize,
        );
        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            ..Default::default()
        });
        let eos = self.tokenizer.eos_token_id;
        let mut last_logits = forward_prefill(&self.weights, &self.config, &tokens, &mut cache, 0, &mut None);

        for _ in 0..max_new_tokens {
            let next = sampler.sample(last_logits.data(), &tokens);
            if next as usize >= embedding_rows {
                return Err(JsValue::from_str(&format!(
                    "generated token id {next} is outside the embedding table (size {embedding_rows})"
                )));
            }

            tokens.push(next);
            if next == eos {
                break;
            }

            let text = self.tokenizer.decode(&[next]);
            let _ = on_token.call1(&JsValue::null(), &JsValue::from_str(&text));

            last_logits = forward_one(
                &self.weights,
                &self.config,
                next,
                tokens.len() - 1,
                &mut cache,
                &mut None,
            );
        }

        Ok(())
    }
}
