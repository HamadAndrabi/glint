//! Model hyperparameters extracted from GGUF metadata.

use std::collections::HashMap;

use super::gguf::MetadataValue;

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
    /// Sliding window attention size (Mistral, some Qwen2 variants).
    /// When set, each token attends to at most this many past positions.
    pub sliding_window: Option<u32>,
    /// RoPE scaling factor for extended-context models (Phi-3, Qwen2-long).
    pub rope_scaling_factor: Option<f32>,
    /// Partial rotary factor — fraction of head_dim to apply RoPE to (Phi-3).
    pub partial_rotary_factor: Option<f32>,
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
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_freq_base = get_f32("rope.freq_base");
        let sliding_window = get_u32("sliding_window");
        let rope_scaling_factor =
            get_f32("rope_scaling.factor").or_else(|| get_f32("rope.scaling.factor"));
        let partial_rotary_factor = get_f32("partial_rotary_factor");

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
        })
    }

    /// Dimension of each attention head: `embedding_length / head_count`.
    pub fn head_dim(&self) -> u32 {
        self.embedding_length / self.head_count
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
