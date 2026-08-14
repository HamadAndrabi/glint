//! Model hyperparameters extracted from GGUF metadata or a HuggingFace
//! `config.json`.

use std::collections::HashMap;

use super::gguf::MetadataValue;
use crate::error::GlintError;

/// Model hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub vocab_size: u32,
    pub feed_forward_length: Option<u32>,
    pub rms_norm_eps: f32,
    pub rope_freq_base: Option<f32>,
    /// Raw Jinja chat template from GGUF metadata (`tokenizer.chat_template`).
    pub chat_template: Option<String>,
    /// Sliding window attention size (Mistral, Gemma 2, some Qwen2 variants).
    /// When set, each token attends to at most this many past positions.
    pub sliding_window: Option<u32>,
    /// RoPE scaling factor for extended-context models (Phi-3, Qwen2-long).
    pub rope_scaling_factor: Option<f32>,
    /// Partial rotary factor — fraction of head_dim to apply RoPE to (Phi-3).
    pub partial_rotary_factor: Option<f32>,
    /// Explicit head dimension override (e.g. Gemma 2).
    pub head_dim_override: Option<u32>,
    /// Attention score logit soft-capping (Gemma 2).
    pub attn_logit_softcapping: Option<f32>,
    /// Final logits soft-capping (Gemma 2).
    pub final_logit_softcapping: Option<f32>,
    /// Attention query pre-attention scaling (Gemma 2).
    pub query_pre_attn_scalar: Option<f32>,
    /// Whether sliding window attention alternates every layer (e.g. Gemma 2 even layers).
    pub sliding_window_alternating: bool,
}

impl ModelConfig {
    /// Extract model configuration from GGUF metadata.
    pub fn from_metadata(metadata: &HashMap<String, MetadataValue>) -> Option<Self> {
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())?
            .to_string();

        let arch = &architecture;

        let get_u32 = |key: &str| -> Option<u32> {
            metadata
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.as_u32())
        };

        let get_f32 = |key: &str| -> Option<f32> {
            metadata
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.as_f32())
        };

        let context_length = get_u32("context_length")?;
        let embedding_length = get_u32("embedding_length")?;
        let block_count = get_u32("block_count")?;
        let head_count = get_u32("attention.head_count")?;
        let head_count_kv = get_u32("attention.head_count_kv").unwrap_or(head_count);

        let vocab_size = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len() as u32)
            .or_else(|| get_u32("vocab_size"))
            .unwrap_or(0);

        let feed_forward_length = get_u32("feed_forward_length");
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon")
            .or_else(|| get_f32("attention.layer_norm_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base = get_f32("rope.freq_base");
        let sliding_window =
            get_u32("sliding_window").or_else(|| get_u32("attention.sliding_window"));
        let rope_scaling_factor =
            get_f32("rope_scaling.factor").or_else(|| get_f32("rope.scaling.factor"));
        let partial_rotary_factor = get_f32("partial_rotary_factor");

        let head_dim_override = get_u32("attention.key_length").or_else(|| get_u32("head_dim"));
        let attn_logit_softcapping =
            get_f32("attention.logit_softcapping").or_else(|| get_f32("attn_logit_softcapping"));
        let final_logit_softcapping = get_f32("final_logit_softcapping");
        let query_pre_attn_scalar =
            get_f32("attention.query_pre_attn_scalar").map(|s| 1.0 / s.sqrt());
        let sliding_window_alternating = arch == "gemma2";

        let chat_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Some(Self {
            architecture,
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            vocab_size,
            feed_forward_length,
            rms_norm_eps,
            rope_freq_base,
            chat_template,
            sliding_window,
            rope_scaling_factor,
            partial_rotary_factor,
            head_dim_override,
            attn_logit_softcapping,
            final_logit_softcapping,
            query_pre_attn_scalar,
            sliding_window_alternating,
        })
    }

    /// True when this model uses the Gemma or Gemma 2 architecture family.
    pub fn is_gemma(&self) -> bool {
        self.architecture == "gemma" || self.architecture == "gemma2"
    }

    /// Dimension of each attention head: `head_dim_override` or `embedding_length / head_count`.
    pub fn head_dim(&self) -> u32 {
        self.head_dim_override
            .unwrap_or_else(|| self.embedding_length / self.head_count)
    }

    /// Map a HuggingFace `config.json` onto a [`ModelConfig`].
    ///
    /// Shorthand for [`HfConfig::from_json`] when the HF-only fields
    /// (`tie_word_embeddings`, BOS/EOS ids) are not needed.
    pub fn from_hf_json(text: &str) -> Result<Self, GlintError> {
        Ok(HfConfig::from_json(text)?.config)
    }
}

// ── HuggingFace config.json ──────────────────────────────────────────────────

/// Model families whose HF `config.json` maps cleanly onto Glint's transformer
/// forward pass (RMSNorm + SwiGLU MLP + RoPE + optional GQA / SWA / soft-capping).
const SUPPORTED_HF_ARCHITECTURES: &[&str] = &[
    "llama",
    "mistral",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "gemma",
    "gemma2",
];

/// A parsed HuggingFace `config.json`.
///
/// Everything Glint's forward pass needs lands in [`ModelConfig`]; the few
/// fields that only matter at load time (weight tying) or that belong to the
/// tokenizer (BOS/EOS ids) are kept alongside it.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub config: ModelConfig,
    /// `tie_word_embeddings` — when true the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl HfConfig {
    /// Parse a HuggingFace `config.json` (LlamaConfig-style keys).
    ///
    /// Required: `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `vocab_size`, `max_position_embeddings`, and an architecture
    /// (`model_type`, or the first entry of `architectures`).
    /// Everything else falls back to the HF default for that field.
    pub fn from_json(text: &str) -> Result<Self, GlintError> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| GlintError::HfInvalidJson {
                file: "config.json".to_string(),
                detail: e.to_string(),
            })?;
        let obj = value.as_object().ok_or_else(|| GlintError::HfInvalidJson {
            file: "config.json".to_string(),
            detail: "top level value is not a JSON object".to_string(),
        })?;

        let get_u32 = |key: &str| -> Option<u32> {
            obj.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|n| u32::try_from(n).ok())
        };
        let get_f32 =
            |key: &str| -> Option<f32> { obj.get(key).and_then(|v| v.as_f64()).map(|f| f as f32) };
        let req_u32 = |key: &'static str| -> Result<u32, GlintError> {
            get_u32(key).ok_or(GlintError::HfMissingConfigField(key))
        };

        // `model_type` is the canonical family name ("llama"); fall back to the
        // first `architectures` entry ("LlamaForCausalLM" → "llama") for the
        // handful of configs that omit it.
        let architecture = obj
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| {
                obj.get("architectures")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim_end_matches("ForCausalLM").to_ascii_lowercase())
            })
            .ok_or(GlintError::HfMissingConfigField("model_type"))?;

        if !SUPPORTED_HF_ARCHITECTURES.contains(&architecture.as_str()) {
            return Err(GlintError::HfUnsupported(format!(
                "model_type '{architecture}' — Glint's safetensors loader handles \
                 LLaMA-style models ({}); convert to GGUF for anything else",
                SUPPORTED_HF_ARCHITECTURES.join(", ")
            )));
        }

        let embedding_length = req_u32("hidden_size")?;
        let block_count = req_u32("num_hidden_layers")?;
        let head_count = req_u32("num_attention_heads")?;
        let vocab_size = req_u32("vocab_size")?;
        let context_length = req_u32("max_position_embeddings")?;
        let head_count_kv = get_u32("num_key_value_heads").unwrap_or(head_count);

        if head_count == 0 || head_count_kv == 0 {
            return Err(GlintError::HfUnsupported(
                "num_attention_heads / num_key_value_heads must be non-zero".to_string(),
            ));
        }
        if embedding_length % head_count != 0 {
            return Err(GlintError::HfUnsupported(format!(
                "hidden_size {embedding_length} is not divisible by \
                 num_attention_heads {head_count}"
            )));
        }
        if head_count % head_count_kv != 0 {
            return Err(GlintError::HfUnsupported(format!(
                "num_attention_heads {head_count} is not a multiple of \
                 num_key_value_heads {head_count_kv}"
            )));
        }
        // If explicit head_dim is specified (e.g. Gemma 2), keep it as head_dim_override.
        let head_dim_override = get_u32("head_dim");
        if head_dim_override.is_none() && embedding_length % head_count != 0 {
            return Err(GlintError::HfUnsupported(format!(
                "hidden_size {embedding_length} is not divisible by \
                 num_attention_heads {head_count}"
            )));
        }

        // Qwen2 sets `use_sliding_window: false` while still carrying a
        // `sliding_window` value — honour the switch.
        let sliding_window = match obj.get("use_sliding_window").and_then(|v| v.as_bool()) {
            Some(false) => None,
            _ => get_u32("sliding_window").filter(|&w| w > 0),
        };

        // Only linear RoPE scaling is implemented (`ops::rope` divides the
        // position by a constant). `llama3`/`dynamic`/`yarn` schedules would
        // need their own frequency tables, so they are refused rather than
        // approximated.
        let rope_scaling_factor = match obj.get("rope_scaling") {
            None | Some(serde_json::Value::Null) => None,
            Some(scaling) => {
                let kind = scaling
                    .get("rope_type")
                    .or_else(|| scaling.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("linear");
                let factor = scaling.get("factor").and_then(|v| v.as_f64());
                match (kind, factor) {
                    ("linear", Some(f)) => Some(f as f32),
                    ("default", _) => None,
                    _ => {
                        return Err(GlintError::HfUnsupported(format!(
                            "rope_scaling type '{kind}' — only linear scaling is implemented"
                        )))
                    }
                }
            }
        };

        let attn_logit_softcapping = get_f32("attn_logit_softcapping");
        let final_logit_softcapping = get_f32("final_logit_softcapping");
        let query_pre_attn_scalar =
            get_f32("query_pre_attn_scalar").map(|s| if s > 1.0 { 1.0 / s.sqrt() } else { s });
        let sliding_window_alternating = architecture == "gemma2";

        let config = ModelConfig {
            architecture,
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            vocab_size,
            feed_forward_length: get_u32("intermediate_size"),
            // HF's LlamaConfig default; GGUF's loader uses the same fallback.
            rms_norm_eps: get_f32("rms_norm_eps").unwrap_or(1e-5),
            rope_freq_base: get_f32("rope_theta"),
            // `config.json` has no chat template — it lives in
            // `tokenizer_config.json` and is filled in by the directory loader.
            chat_template: None,
            sliding_window,
            rope_scaling_factor,
            partial_rotary_factor: get_f32("partial_rotary_factor"),
            head_dim_override,
            attn_logit_softcapping,
            final_logit_softcapping,
            query_pre_attn_scalar,
            sliding_window_alternating,
        };

        // `eos_token_id` is a list on some LLaMA-3 derivatives (EOS + EOT);
        // the first entry is the one generation should stop on.
        let eos_token_id = get_u32("eos_token_id").or_else(|| {
            obj.get("eos_token_id")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
                .and_then(|n| u32::try_from(n).ok())
        });

        Ok(Self {
            config,
            tie_word_embeddings: obj
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            bos_token_id: get_u32("bos_token_id"),
            eos_token_id,
        })
    }
}

impl std::fmt::Display for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Architecture:       {}", self.architecture)?;
        writeln!(f, "Context length:     {}", self.context_length)?;
        writeln!(f, "Embedding size:     {}", self.embedding_length)?;
        writeln!(f, "Layers:             {}", self.block_count)?;
        writeln!(f, "Attention heads:    {}", self.head_count)?;
        writeln!(f, "KV heads:           {}", self.head_count_kv)?;
        writeln!(f, "Head dimension:     {}", self.head_dim())?;
        writeln!(f, "Vocab size:         {}", self.vocab_size)?;
        if let Some(ffn) = self.feed_forward_length {
            writeln!(f, "FFN hidden size:    {ffn}")?;
        }
        writeln!(f, "RMSNorm epsilon:    {:.0e}", self.rms_norm_eps)?;
        if let Some(base) = self.rope_freq_base {
            writeln!(f, "RoPE freq base:     {base}")?;
        }
        if let Some(w) = self.sliding_window {
            writeln!(f, "Sliding window:     {w}")?;
        }
        if let Some(sf) = self.rope_scaling_factor {
            writeln!(f, "RoPE scale factor:  {sf}")?;
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal but realistic LLaMA-family `config.json`.
    fn llama_config_json() -> &'static str {
        r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "intermediate_size": 1536,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-05,
            "rope_theta": 100000.0,
            "vocab_size": 49152,
            "tie_word_embeddings": true,
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#
    }

    #[test]
    fn test_hf_config_maps_every_model_config_field() {
        let hf = HfConfig::from_json(llama_config_json()).unwrap();
        let c = &hf.config;
        assert_eq!(c.architecture, "llama");
        assert_eq!(c.context_length, 8192);
        assert_eq!(c.embedding_length, 576);
        assert_eq!(c.block_count, 30);
        assert_eq!(c.head_count, 9);
        assert_eq!(c.head_count_kv, 3);
        assert_eq!(c.vocab_size, 49152);
        assert_eq!(c.feed_forward_length, Some(1536));
        assert!((c.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert_eq!(c.rope_freq_base, Some(100000.0));
        assert_eq!(c.chat_template, None);
        assert_eq!(c.sliding_window, None);
        assert_eq!(c.rope_scaling_factor, None);
        assert_eq!(c.partial_rotary_factor, None);
        assert_eq!(c.head_dim(), 64);

        assert!(hf.tie_word_embeddings);
        assert_eq!(hf.bos_token_id, Some(1));
        assert_eq!(hf.eos_token_id, Some(2));
    }

    #[test]
    fn test_hf_config_defaults_optional_fields() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32
        }"#;
        let hf = HfConfig::from_json(json).unwrap();
        // num_key_value_heads defaults to num_attention_heads (no GQA).
        assert_eq!(hf.config.head_count_kv, 2);
        assert_eq!(hf.config.feed_forward_length, None);
        assert!((hf.config.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert_eq!(hf.config.rope_freq_base, None);
        assert!(!hf.tie_word_embeddings);
        assert_eq!(hf.eos_token_id, None);
    }

    #[test]
    fn test_hf_config_missing_required_field_is_named() {
        let json = r#"{"model_type": "llama", "hidden_size": 8}"#;
        let err = HfConfig::from_json(json).unwrap_err();
        assert!(
            err.to_string().contains("num_hidden_layers"),
            "error should name the missing field, got: {err}"
        );
    }

    #[test]
    fn test_hf_config_rejects_unsupported_architecture() {
        let json = r#"{
            "model_type": "gpt2",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32
        }"#;
        let err = HfConfig::from_json(json).unwrap_err();
        assert!(err.to_string().contains("gpt2"), "got: {err}");
    }

    #[test]
    fn test_hf_config_parses_gemma2_and_qwen2() {
        let gemma_json = r#"{
            "model_type": "gemma2",
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "head_dim": 256,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256
        }"#;
        let gemma_cfg = HfConfig::from_json(gemma_json).unwrap().config;
        assert_eq!(gemma_cfg.architecture, "gemma2");
        assert_eq!(gemma_cfg.head_dim(), 256);
        assert_eq!(gemma_cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(gemma_cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(gemma_cfg.query_pre_attn_scalar, Some(1.0 / 16.0));
        assert!(gemma_cfg.sliding_window_alternating);

        let qwen_json = r#"{
            "model_type": "qwen2",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 32768
        }"#;
        let qwen_cfg = HfConfig::from_json(qwen_json).unwrap().config;
        assert_eq!(qwen_cfg.architecture, "qwen2");
        assert_eq!(qwen_cfg.head_dim(), 128);
        assert!(!qwen_cfg.sliding_window_alternating);
    }

    #[test]
    fn test_hf_config_architecture_falls_back_to_architectures_list() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32
        }"#;
        assert_eq!(
            HfConfig::from_json(json).unwrap().config.architecture,
            "mistral"
        );
    }

    #[test]
    fn test_hf_config_rejects_indivisible_hidden_size() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 9,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32
        }"#;
        assert!(HfConfig::from_json(json).is_err());
    }

    #[test]
    fn test_hf_config_linear_rope_scaling_and_sliding_window() {
        let json = r#"{
            "model_type": "mistral",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "sliding_window": 4096,
            "rope_scaling": {"type": "linear", "factor": 4.0}
        }"#;
        let c = HfConfig::from_json(json).unwrap().config;
        assert_eq!(c.sliding_window, Some(4096));
        assert_eq!(c.rope_scaling_factor, Some(4.0));
    }

    #[test]
    fn test_hf_config_rejects_unimplemented_rope_scaling() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0}
        }"#;
        let err = HfConfig::from_json(json).unwrap_err();
        assert!(err.to_string().contains("llama3"), "got: {err}");
    }

    #[test]
    fn test_hf_config_eos_token_id_list_takes_first() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "eos_token_id": [128001, 128009]
        }"#;
        assert_eq!(
            HfConfig::from_json(json).unwrap().eos_token_id,
            Some(128001)
        );
    }

    #[test]
    fn test_hf_config_rejects_garbage_json() {
        assert!(HfConfig::from_json("not json").is_err());
        assert!(HfConfig::from_json("[1, 2, 3]").is_err());
    }
}
