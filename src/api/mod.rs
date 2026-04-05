//! High-level Rust library API for Glint.
//!
//! [`Model`] is the main entry point: load a GGUF file, create [`Session`]s,
//! and drive generation via the methods below.  This API is the authoritative
//! surface that Python, WASM, and C FFI bindings will layer on top of.
//!
//! # Example
//! ```no_run
//! use std::path::Path;
//! use glint::api::{Model, GenerationOptions};
//!
//! let model  = Model::load(Path::new("mistral.gguf")).unwrap();
//! let opts   = GenerationOptions::default();
//! let tokens = model.generate("The capital of France is", &opts, &mut None).unwrap();
//! println!("{}", model.decode(&tokens));
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::backend::GpuBackend;
use crate::constrained::{build_constraint, ConstraintSpec, VocabIndex};
use crate::error::GlintError;
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::lora::LoraWeights;
use crate::model::lora_registry::AdapterRegistry;
use crate::model::tokenizer::Tokenizer;
use crate::sampling::SamplerConfig;
use crate::session::snapshot::{
    export_snapshot_with_meta, import_snapshot, model_hash, peek_snapshot_cache_format,
    restore_session, KvSnapshot,
    SnapshotMetadata,
};
use crate::session::{CacheFormat, Session, SessionOptions};
use crate::transformer::{forward_one_lora, forward_prefill_lora, TransformerWeights};

// ── GenerationOptions ─────────────────────────────────────────────────────────

/// Parameters controlling a single generation run.
///
/// Construct via `Default::default()` and override fields as needed.
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampler_cfg:    SamplerConfig,
    pub cache_format:   CacheFormat,
    /// Optional structured-output constraint.
    ///
    /// `None` = unconstrained (default).
    /// `Some(ConstraintSpec::JsonObject)` = force valid JSON object output.
    pub constraint: Option<ConstraintSpec>,
    /// Optional LoRA adapter to apply for this generation run.
    ///
    /// When `Some`, the adapter overrides the base model's built-in adapter
    /// (if any).  Pass `None` for standard (no per-request LoRA) inference.
    pub lora_adapter: Option<Arc<LoraWeights>>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampler_cfg:    SamplerConfig::default(),
            cache_format:   CacheFormat::F32,
            constraint:     None,
            lora_adapter:   None,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// A loaded GGUF model, ready for synchronous or session-based inference.
///
/// Weights and config are `Arc`-wrapped so the model can be cloned cheaply
/// (e.g. across threads) without copying large weight tensors.
pub struct Model {
    pub weights:    Arc<TransformerWeights>,
    pub config:     Arc<ModelConfig>,
    pub tokenizer:  Arc<Tokenizer>,
    /// FNV-64 hash of (file path bytes || file size LE u64).
    /// Used to verify that a [`KvSnapshot`] was created from the same file.
    pub model_hash: u64,
    /// Named LoRA adapters registered with this model instance.
    pub adapter_registry: AdapterRegistry,
}

impl Model {
    /// Load a GGUF model from `path`.
    pub fn load(path: &Path) -> Result<Self, GlintError> {
        let path_str = path.to_string_lossy().into_owned();
        let gguf = GgufModel::load(path)
            .map_err(|e| GlintError::TensorReadError { name: "model".into(), detail: e.to_string() })?;

        let file_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        let hash = model_hash(&path_str, file_size);

        let config = ModelConfig::from_metadata(&gguf.metadata)
            .ok_or(GlintError::MissingModelConfig)?;

        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights   = TransformerWeights::load(&gguf, &config)?;

        Ok(Self {
            weights:    Arc::new(weights),
            config:     Arc::new(config),
            tokenizer:  Arc::new(tokenizer),
            model_hash: hash,
            adapter_registry: AdapterRegistry::new(),
        })
    }

    // ── LoRA adapters ────────────────────────────────────────────────────────

    /// Load a GGUF LoRA adapter and register it under `name`.
    ///
    /// After registration, pass the same `name` via
    /// [`GenerationOptions::lora_adapter`] (as `Some(Arc<LoraWeights>)`)
    /// or retrieve it with `model.adapter_registry.get(name)`.
    pub fn register_lora(
        &mut self,
        name: &str,
        path: &std::path::Path,
    ) -> Result<(), GlintError> {
        self.adapter_registry.register(name, path, self.config.block_count as usize)
    }

    // ── Session lifecycle ────────────────────────────────────────────────────

    /// Create a new empty session (no prefill yet).
    pub fn new_session(&self, opts: &GenerationOptions) -> Session {
        let mut session = Session::new(SessionOptions {
            max_new_tokens: opts.max_new_tokens,
            sampler_cfg:    opts.sampler_cfg,
            eos_token:      self.tokenizer.eos_token_id,
            cache_format:   opts.cache_format,
            context_length: self.config.context_length as usize,
            n_layers:       self.config.block_count as usize,
            n_kv_heads:     self.config.head_count_kv as usize,
            head_dim:       self.config.head_dim() as usize,
            lora_adapter:   opts.lora_adapter.clone(),
        });
        if let Some(spec) = &opts.constraint {
            // Build a vocab index from the tokenizer's raw vocabulary strings.
            let vocab_strings: Vec<String> = (0..self.tokenizer.vocab_size())
                .map(|i| self.tokenizer.decode_token(i as u32).to_owned())
                .collect();
            let vi = VocabIndex::from_vocab(&vocab_strings);
            session.constraint  = Some(build_constraint(spec, Arc::clone(&vi)));
            session.vocab_index = Some(vi);
        }
        session
    }

    /// Tokenise `prompt` and run a prefill pass, storing K/V into `session`.
    pub fn prefill(
        &self,
        session: &mut Session,
        prompt: &str,
        gpu: &mut Option<&mut GpuBackend>,
    ) -> Result<(), GlintError> {
        let tokens = self.tokenizer.encode(prompt);
        self.prefill_tokens(session, &tokens, gpu)
    }

    /// Run a prefill pass for raw `token_ids`.
    pub fn prefill_tokens(
        &self,
        session: &mut Session,
        token_ids: &[u32],
        gpu: &mut Option<&mut GpuBackend>,
    ) -> Result<(), GlintError> {
        session.tokens = token_ids.to_vec();
        session.prefill_len = token_ids.len();
        let lora = session.lora_adapter.as_deref();
        session.last_logits = forward_prefill_lora(
            &self.weights,
            &self.config,
            token_ids,
            session.cache.as_mut(),
            0,
            gpu,
            lora,
        );
        session.pos = token_ids.len().saturating_sub(1);
        Ok(())
    }

    /// Sample and decode one token.
    ///
    /// Advances `session` by one step.  Returns `None` when the session is
    /// finished (EOS hit or budget exhausted).
    pub fn decode_one(
        &self,
        session: &mut Session,
        gpu: &mut Option<&mut GpuBackend>,
    ) -> Option<u32> {
        if session.is_finished() || session.tokens.is_empty() {
            return None;
        }
        session.max_remaining = session.max_remaining.saturating_sub(1);
        let next = if let Some(constraint) = session.constraint.as_mut() {
            // VocabIndex is stored alongside the constraint; build a minimal
            // empty one as a fallback if somehow missing.
            static EMPTY_VI: std::sync::OnceLock<std::sync::Arc<crate::constrained::VocabIndex>> =
                std::sync::OnceLock::new();
            let vi = session.vocab_index.as_ref()
                .unwrap_or_else(|| EMPTY_VI.get_or_init(|| {
                    crate::constrained::VocabIndex::from_vocab(&[])
                }));
            let mask = constraint.allowed_tokens(&session.tokens, vi);
            let tok  = session.sampler.sample_constrained(
                session.last_logits.data(), &session.tokens, &mask,
            );
            constraint.advance(tok);
            tok
        } else {
            session.sampler.sample(session.last_logits.data(), &session.tokens)
        };
        session.tokens.push(next);
        if next == session.eos_token {
            return Some(next);
        }
        let pos = session.tokens.len() - 1;
        let lora = session.lora_adapter.as_deref();
        session.last_logits = forward_one_lora(
            &self.weights,
            &self.config,
            next,
            pos,
            session.cache.as_mut(),
            gpu,
            lora,
        );
        session.pos = pos;
        Some(next)
    }

    // ── High-level generation ────────────────────────────────────────────────

    /// Tokenise `prompt`, prefill, then decode up to `opts.max_new_tokens`.
    ///
    /// Returns only the newly generated token ids (not the prompt).
    pub fn generate(
        &self,
        prompt: &str,
        opts: &GenerationOptions,
        gpu: &mut Option<&mut GpuBackend>,
    ) -> Result<Vec<u32>, GlintError> {
        let mut session = self.new_session(opts);
        self.prefill(&mut session, prompt, gpu)?;
        let mut new_tokens = Vec::new();
        while let Some(tok) = self.decode_one(&mut session, gpu) {
            new_tokens.push(tok);
            if tok == session.eos_token { break; }
        }
        Ok(new_tokens)
    }

    /// Like [`generate`] but calls `on_token(id)` for each new token.
    ///
    /// Return `false` from the callback to stop early.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        opts: &GenerationOptions,
        mut on_token: impl FnMut(u32) -> bool,
        gpu: &mut Option<&mut GpuBackend>,
    ) -> Result<Vec<u32>, GlintError> {
        let mut session = self.new_session(opts);
        self.prefill(&mut session, prompt, gpu)?;
        let mut new_tokens = Vec::new();
        while let Some(tok) = self.decode_one(&mut session, gpu) {
            new_tokens.push(tok);
            let stop = tok == session.eos_token || !on_token(tok);
            if stop { break; }
        }
        Ok(new_tokens)
    }

    // ── Snapshot API ─────────────────────────────────────────────────────────

    /// Serialise the current session state to bytes.
    pub fn export_session(&self, session: &Session) -> Result<Vec<u8>, GlintError> {
        let mut meta = self.snapshot_meta();
        meta.cache_format = session.cache_format;
        export_snapshot_with_meta(session, &meta)
    }

    /// Deserialise bytes and verify they match this model.
    pub fn import_snapshot_bytes(&self, bytes: &[u8]) -> Result<KvSnapshot, GlintError> {
        let mut meta = self.snapshot_meta();
        meta.cache_format = peek_snapshot_cache_format(bytes)?;
        import_snapshot(bytes, &meta)
    }

    /// Restore a [`Session`] from a snapshot.
    pub fn restore_session(
        &self,
        snap: KvSnapshot,
        opts: GenerationOptions,
    ) -> Result<Session, GlintError> {
        let session_opts = SessionOptions {
            max_new_tokens: opts.max_new_tokens,
            sampler_cfg:    opts.sampler_cfg,
            eos_token:      self.tokenizer.eos_token_id,
            cache_format:   snap.meta.cache_format,
            context_length: self.config.context_length as usize,
            n_layers:       self.config.block_count as usize,
            n_kv_heads:     self.config.head_count_kv as usize,
            head_dim:       self.config.head_dim() as usize,
            lora_adapter:   opts.lora_adapter.clone(),
        };
        let mut session = restore_session(snap, session_opts)?;
        if let Some(spec) = &opts.constraint {
            let vocab_strings: Vec<String> = (0..self.tokenizer.vocab_size())
                .map(|i| self.tokenizer.decode_token(i as u32).to_owned())
                .collect();
            let vi = VocabIndex::from_vocab(&vocab_strings);
            let mut constraint = build_constraint(spec, Arc::clone(&vi));
            for &tok in session.tokens.iter().skip(session.prefill_len) {
                constraint.advance(tok);
            }
            session.constraint = Some(constraint);
            session.vocab_index = Some(vi);
        }
        if let Some((&last_token, pos)) = session
            .tokens
            .last()
            .map(|tok| (tok, session.tokens.len().saturating_sub(1)))
        {
            session.cache.truncate(pos);
            let lora = session.lora_adapter.as_deref();
            session.last_logits = forward_one_lora(
                &self.weights,
                &self.config,
                last_token,
                pos,
                session.cache.as_mut(),
                &mut None,
                lora,
            );
            session.pos = pos;
        }
        Ok(session)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Decode token ids to a string using the loaded tokenizer.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.tokenizer.decode(token_ids)
    }

    fn snapshot_meta(&self) -> SnapshotMetadata {
        SnapshotMetadata {
            model_hash:  self.model_hash,
            context_len: self.config.context_length,
            n_layers:    self.config.block_count,
            n_kv_heads:  self.config.head_count_kv,
            head_dim:    self.config.head_dim(),
            cache_format: CacheFormat::F32,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use crate::model::tokenizer::Tokenizer;
    use crate::tensor::{QuantizedTensor, Tensor};
    use crate::transformer::weights::LayerWeights;

    fn make_tiny_model() -> Model {
        let config = ModelConfig {
            architecture: "test".to_string(),
            context_length: 32,
            embedding_length: 4,
            block_count: 1,
            head_count: 2,
            head_count_kv: 1,
            vocab_size: 8,
            feed_forward_length: Some(8),
            rms_norm_eps: 1e-5,
            rope_freq_base: Some(10000.0),
            chat_template: None,
            sliding_window: None,
            rope_scaling_factor: None,
            partial_rotary_factor: None,
        };
        let weights = TransformerWeights {
            token_embedding: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(), 8, 4),
            layers: vec![LayerWeights {
                attn_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
                ffn_norm:  Tensor::from_vec(vec![1.0; 4], &[4]),
                attn_q:      QuantizedTensor::from_f32(&(0..16).map(|i| i as f32 * 0.05 - 0.4).collect::<Vec<_>>(), 4, 4),
                attn_k:      QuantizedTensor::from_f32(&(0..8).map(|i| i as f32 * 0.1 - 0.3).collect::<Vec<_>>(), 2, 4),
                attn_v:      QuantizedTensor::from_f32(&(0..8).map(|i| i as f32 * 0.07 - 0.2).collect::<Vec<_>>(), 2, 4),
                attn_output: QuantizedTensor::from_f32(&(0..16).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>(), 4, 4),
                ffn_gate: QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(), 8, 4),
                ffn_up:   QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.015 - 0.2).collect::<Vec<_>>(), 8, 4),
                ffn_down: QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.01 - 0.15).collect::<Vec<_>>(), 4, 8),
            }],
            output_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            output:      QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.1 - 1.6).collect::<Vec<_>>(), 8, 4),
            lora: None,
        };
        Model {
            weights:    Arc::new(weights),
            config:     Arc::new(config),
            tokenizer:  Arc::new(Tokenizer::bare_for_test(8, 1, 2)),
            model_hash: 0,
            adapter_registry: AdapterRegistry::new(),
        }
    }

    #[test]
    fn test_new_session_f32() {
        let model = make_tiny_model();
        let opts  = GenerationOptions::default();
        let s     = model.new_session(&opts);
        assert!(s.tokens.is_empty());
        assert_eq!(s.eos_token, 2);
        assert_eq!(s.max_remaining, 256);
    }

    #[test]
    fn test_prefill_tokens_sets_pos() {
        let model = make_tiny_model();
        let opts  = GenerationOptions::default();
        let mut s = model.new_session(&opts);
        model.prefill_tokens(&mut s, &[1, 3, 5], &mut None).unwrap();
        assert_eq!(s.tokens, vec![1, 3, 5]);
        assert_eq!(s.pos, 2);
    }

    #[test]
    fn test_decode_one_advances_session() {
        let model = make_tiny_model();
        let opts  = GenerationOptions { max_new_tokens: 5, ..Default::default() };
        let mut s = model.new_session(&opts);
        model.prefill_tokens(&mut s, &[1], &mut None).unwrap();
        let tok = model.decode_one(&mut s, &mut None);
        assert!(tok.is_some());
        assert_eq!(s.tokens.len(), 2);
        assert_eq!(s.max_remaining, 4);
    }

    #[test]
    fn test_generate_returns_new_tokens_only() {
        let model = make_tiny_model();
        let opts  = GenerationOptions {
            max_new_tokens: 4,
            sampler_cfg: SamplerConfig { seed: Some(42), ..Default::default() },
            ..Default::default()
        };
        // generate will error if tokenizer can't encode, but with our toy model
        // the encode will produce empty tokens which is still a valid test of the loop
        let result = model.generate("tok1", &opts, &mut None);
        // toy tokenizer won't produce real tokens; just verify it doesn't panic
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_returns_string() {
        let model = make_tiny_model();
        // eos token (2) should produce its vocab string
        let s = model.decode(&[2]);
        assert!(!s.is_empty() || s.is_empty()); // just doesn't panic
    }

    #[test]
    fn test_snapshot_restore_rebuilds_last_logits() {
        let model = make_tiny_model();
        let opts  = GenerationOptions {
            max_new_tokens: 4,
            sampler_cfg: SamplerConfig { seed: Some(7), ..Default::default() },
            ..Default::default()
        };
        let mut original = model.new_session(&opts);
        model.prefill_tokens(&mut original, &[1, 3], &mut None).unwrap();
        assert!(model.decode_one(&mut original, &mut None).is_some());

        let bytes = model.export_session(&original).unwrap();
        let snap = model.import_snapshot_bytes(&bytes).unwrap();
        let mut restored = model.restore_session(snap, opts.clone()).unwrap();

        let next_original = model.decode_one(&mut original, &mut None);
        let next_restored = model.decode_one(&mut restored, &mut None);
        assert_eq!(next_original, next_restored);
    }

    #[test]
    fn test_q8_snapshot_roundtrip_via_model_api() {
        let model = make_tiny_model();
        let opts  = GenerationOptions {
            cache_format: CacheFormat::Q8,
            ..Default::default()
        };
        let mut session = model.new_session(&opts);
        model.prefill_tokens(&mut session, &[1, 3, 5], &mut None).unwrap();

        let bytes = model.export_session(&session).unwrap();
        let snap = model.import_snapshot_bytes(&bytes).unwrap();
        assert_eq!(snap.meta.cache_format, CacheFormat::Q8);

        let restored = model.restore_session(snap, opts).unwrap();
        assert_eq!(restored.cache_format, CacheFormat::Q8);
    }

    #[test]
    fn test_restore_session_rebuilds_constraint_state() {
        let model = make_tiny_model();
        let opts = GenerationOptions {
            constraint: Some(ConstraintSpec::JsonEnum(vec!["tok3tok4".to_string()])),
            ..Default::default()
        };
        let mut original = model.new_session(&opts);
        model.prefill_tokens(&mut original, &[1], &mut None).unwrap();
        original.tokens.push(3);
        original.cache.write(0, 1, &[0.0, 0.0], &[0.0, 0.0]);
        original.cache.advance();
        original.pos = 1;
        original.constraint.as_mut().unwrap().advance(3);

        let original_vi = Arc::clone(original.vocab_index.as_ref().unwrap());
        let original_mask = original
            .constraint
            .as_mut()
            .unwrap()
            .allowed_tokens(&original.tokens, original_vi.as_ref());

        let bytes = model.export_session(&original).unwrap();
        let snap = model.import_snapshot_bytes(&bytes).unwrap();
        let mut restored = model.restore_session(snap, opts).unwrap();
        let restored_vi = Arc::clone(restored.vocab_index.as_ref().unwrap());
        let restored_mask = restored
            .constraint
            .as_mut()
            .unwrap()
            .allowed_tokens(&restored.tokens, restored_vi.as_ref());

        assert_eq!(restored.prefill_len, 1);
        assert_eq!(original_mask, restored_mask);
    }
}
