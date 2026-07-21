//! Model weight loading from GGUF.
//!
//! Large weight matrices (attention projections, FFN) are stored as
//! `QuantizedTensor` — keeping raw quantized bytes in memory instead of
//! expanding to f32. Norm weights are tiny and stay as `Tensor` (f32).

use crate::error::GlintError;
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::lora::LoraWeights;
use crate::tensor::quantized::WeightLoadMode;
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
    /// Optional LoRA adapters — `None` unless loaded via `with_lora()`.
    pub lora: Option<LoraWeights>,
}

impl TransformerWeights {
    /// Load all weights from a GGUF model.
    ///
    /// Weight matrices are loaded as `QuantizedTensor` (raw bytes kept).
    /// Norm vectors are loaded as `Tensor` (f32, tiny — ~2 KB each).
    /// Load all weights using the default (eager) mode.
    pub fn load(model: &GgufModel, config: &ModelConfig) -> Result<Self, GlintError> {
        Self::load_with_mode(model, config, WeightLoadMode::Eager)
    }

    /// Load all weights with explicit load mode.
    ///
    /// Use [`WeightLoadMode::Lazy`] to keep weight bytes in the mmap rather
    /// than copying — halves the peak RAM used during loading.  The mmap pages
    /// will be faulted in on first access.
    pub fn load_with_mode(
        model: &GgufModel,
        config: &ModelConfig,
        mode: WeightLoadMode,
    ) -> Result<Self, GlintError> {
        let load = |name: &str| QuantizedTensor::load_with_mode(model, name, mode);

        eprintln!("Loading token embedding...");
        let token_embedding = load("token_embd.weight")?;

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for i in 0..config.block_count as usize {
            eprint!("\rLoading layer {}/{}...", i + 1, config.block_count);
            layers.push(LayerWeights {
                attn_norm: load_tensor_f32(model, &format!("blk.{i}.attn_norm.weight"))?,
                ffn_norm: load_tensor_f32(model, &format!("blk.{i}.ffn_norm.weight"))?,
                attn_q: load(&format!("blk.{i}.attn_q.weight"))?,
                attn_k: load(&format!("blk.{i}.attn_k.weight"))?,
                attn_v: load(&format!("blk.{i}.attn_v.weight"))?,
                attn_output: load(&format!("blk.{i}.attn_output.weight"))?,
                ffn_gate: load(&format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: load(&format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: load(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        eprintln!(
            "\rLoaded {}/{} layers.       ",
            config.block_count, config.block_count
        );

        eprintln!("Loading output weights...");
        let output_norm = load_tensor_f32(model, "output_norm.weight")?;

        // Some models tie the output projection to the token embedding.
        let output = if model.get_tensor_info("output.weight").is_some() {
            load("output.weight")?
        } else {
            token_embedding.clone()
        };

        Ok(Self {
            token_embedding,
            layers,
            output_norm,
            output,
            lora: None,
        })
    }

    /// Upload all quantized weight tensors to the GPU.
    ///
    /// After this call, `matvec_gpu()` on any weight tensor will dispatch to
    /// the GPU instead of the CPU. Call once at startup before inference.
    #[cfg(feature = "vulkan")]
    pub fn upload_all_to_gpu(&mut self, gpu: &mut crate::backend::GpuBackend) {
        eprintln!("Uploading weights to GPU...");
        self.token_embedding.upload_to_gpu(gpu, "token_embd");
        self.output.upload_to_gpu(gpu, "output");
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            eprint!("\r  Layer {}/{}...", i + 1, n_layers);
            layer.attn_q.upload_to_gpu(gpu, &format!("blk.{i}.attn_q"));
            layer.attn_k.upload_to_gpu(gpu, &format!("blk.{i}.attn_k"));
            layer.attn_v.upload_to_gpu(gpu, &format!("blk.{i}.attn_v"));
            layer
                .attn_output
                .upload_to_gpu(gpu, &format!("blk.{i}.attn_output"));
            layer
                .ffn_gate
                .upload_to_gpu(gpu, &format!("blk.{i}.ffn_gate"));
            layer.ffn_up.upload_to_gpu(gpu, &format!("blk.{i}.ffn_up"));
            layer
                .ffn_down
                .upload_to_gpu(gpu, &format!("blk.{i}.ffn_down"));
        }
        eprintln!("\r  Uploaded all layers.       ");
    }

    /// Attach LoRA adapters loaded from a separate GGUF file.
    ///
    /// Returns `Self` unchanged if the file contains no LoRA tensors.
    pub fn with_lora(mut self, lora_model: &GgufModel) -> Self {
        let n_layers = self.layers.len();
        if let Some(lora) = LoraWeights::load(lora_model, n_layers) {
            eprintln!("LoRA adapters loaded ({} layers).", n_layers);
            self.lora = Some(lora);
        } else {
            eprintln!("Warning: no lora_a/lora_b tensors found in adapter file.");
        }
        self
    }
}
