//! Model weight loading from GGUF.
//!
//! Large weight matrices (attention projections, FFN) are stored as
//! `QuantizedTensor` — keeping raw quantized bytes in memory instead of
//! expanding to f32. Norm weights are tiny and stay as `Tensor` (f32).

use crate::error::FerriteError;
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::tensor::{load_tensor_f32, QuantizedTensor, Tensor};

/// All weights for a single transformer block (one layer).
pub struct LayerWeights {
    /// Pre-attention RMSNorm scale — F32 in GGUF, tiny (embed_dim floats).
    pub attn_norm: Tensor,
    /// Pre-FFN RMSNorm scale — F32 in GGUF, tiny.
    pub ffn_norm: Tensor,

    /// Query projection [n_heads*head_dim, embed_dim] — quantized.
    pub attn_q: QuantizedTensor,
    /// Key projection [n_kv_heads*head_dim, embed_dim] — quantized.
    pub attn_k: QuantizedTensor,
    /// Value projection [n_kv_heads*head_dim, embed_dim] — quantized.
    pub attn_v: QuantizedTensor,
    /// Attention output projection [embed_dim, embed_dim] — quantized.
    pub attn_output: QuantizedTensor,

    /// FFN gate projection [ffn_hidden, embed_dim] — quantized.
    pub ffn_gate: QuantizedTensor,
    /// FFN up projection [ffn_hidden, embed_dim] — quantized.
    pub ffn_up: QuantizedTensor,
    /// FFN down projection [embed_dim, ffn_hidden] — quantized.
    pub ffn_down: QuantizedTensor,
}

/// All weights for the full transformer model.
pub struct TransformerWeights {
    /// Token embedding table [vocab_size, embed_dim] — quantized.
    pub token_embedding: QuantizedTensor,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm scale — F32, tiny.
    pub output_norm: Tensor,
    /// LM head [vocab_size, embed_dim] — quantized.
    pub output: QuantizedTensor,
}

impl TransformerWeights {
    /// Load all weights from a GGUF model.
    ///
    /// Weight matrices are loaded as `QuantizedTensor` (raw bytes kept).
    /// Norm vectors are loaded as `Tensor` (f32, tiny — ~2 KB each).
    pub fn load(model: &GgufModel, config: &ModelConfig) -> Result<Self, FerriteError> {
        eprintln!("Loading token embedding...");
        let token_embedding = QuantizedTensor::load(model, "token_embd.weight")?;

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for i in 0..config.block_count as usize {
            eprint!("\rLoading layer {}/{}...", i + 1, config.block_count);
            layers.push(LayerWeights {
                attn_norm: load_tensor_f32(model, &format!("blk.{i}.attn_norm.weight"))?,
                ffn_norm:  load_tensor_f32(model, &format!("blk.{i}.ffn_norm.weight"))?,
                attn_q:      QuantizedTensor::load(model, &format!("blk.{i}.attn_q.weight"))?,
                attn_k:      QuantizedTensor::load(model, &format!("blk.{i}.attn_k.weight"))?,
                attn_v:      QuantizedTensor::load(model, &format!("blk.{i}.attn_v.weight"))?,
                attn_output: QuantizedTensor::load(model, &format!("blk.{i}.attn_output.weight"))?,
                ffn_gate:    QuantizedTensor::load(model, &format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up:      QuantizedTensor::load(model, &format!("blk.{i}.ffn_up.weight"))?,
                ffn_down:    QuantizedTensor::load(model, &format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        eprintln!("\rLoaded {}/{} layers.       ", config.block_count, config.block_count);

        eprintln!("Loading output weights...");
        let output_norm = load_tensor_f32(model, "output_norm.weight")?;

        // Some models tie the output projection to the token embedding
        let output = if model.get_tensor_info("output.weight").is_some() {
            QuantizedTensor::load(model, "output.weight")?
        } else {
            token_embedding.clone()
        };

        Ok(Self { token_embedding, layers, output_norm, output })
    }
}
