//! Model weight loading from GGUF into dequantized f32 tensors.

use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::tensor::{load_tensor_f32, Tensor};

/// All weights for a single transformer block (one layer).
pub struct LayerWeights {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub attn_q: Tensor,
    pub attn_k: Tensor,
    pub attn_v: Tensor,
    pub attn_output: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
}

/// All weights for the full transformer model.
pub struct TransformerWeights {
    pub token_embedding: Tensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Tensor,
    pub output: Tensor,
}

impl TransformerWeights {
    /// Load and dequantize all weights from a GGUF model.
    pub fn load(model: &GgufModel, config: &ModelConfig) -> Self {
        eprintln!("Loading token embedding...");
        let token_embedding = load_tensor_f32(model, "token_embd.weight");

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for i in 0..config.block_count as usize {
            eprint!("\rLoading layer {}/{}...", i + 1, config.block_count);
            layers.push(LayerWeights {
                attn_norm: load_tensor_f32(model, &format!("blk.{i}.attn_norm.weight")),
                ffn_norm: load_tensor_f32(model, &format!("blk.{i}.ffn_norm.weight")),
                attn_q: load_tensor_f32(model, &format!("blk.{i}.attn_q.weight")),
                attn_k: load_tensor_f32(model, &format!("blk.{i}.attn_k.weight")),
                attn_v: load_tensor_f32(model, &format!("blk.{i}.attn_v.weight")),
                attn_output: load_tensor_f32(model, &format!("blk.{i}.attn_output.weight")),
                ffn_gate: load_tensor_f32(model, &format!("blk.{i}.ffn_gate.weight")),
                ffn_up: load_tensor_f32(model, &format!("blk.{i}.ffn_up.weight")),
                ffn_down: load_tensor_f32(model, &format!("blk.{i}.ffn_down.weight")),
            });
        }
        eprintln!("\rLoaded {}/{} layers.       ", config.block_count, config.block_count);

        eprintln!("Loading output weights...");
        let output_norm = load_tensor_f32(model, "output_norm.weight");

        // Some models tie the output weight to the embedding weight
        let output = if model.get_tensor_info("output.weight").is_some() {
            load_tensor_f32(model, "output.weight")
        } else {
            token_embedding.clone()
        };

        Self {
            token_embedding,
            layers,
            output_norm,
            output,
        }
    }
}
