//! Model weight loading from GGUF or HuggingFace SafeTensors.
//!
//! Large weight matrices (attention projections, FFN) are stored as
//! `QuantizedTensor` — keeping raw quantized bytes in memory instead of
//! expanding to f32. Norm weights are tiny and stay as `Tensor` (f32).

use crate::error::GlintError;
use crate::model::config::ModelConfig;
use crate::model::gguf::GgufModel;
use crate::model::lora::LoraWeights;
use crate::model::safetensors::SafeTensorsModel;
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

    /// Load all weights from a HuggingFace SafeTensors checkpoint.
    ///
    /// # Layout
    ///
    /// HF stores every `nn.Linear` weight row-major as
    /// `[out_features, in_features]`, which is *already* Glint's convention —
    /// `QuantizedTensor::matvec` walks `rows` dot products of length `cols`.
    /// No transpose is applied. (GGUF stores the same matrices column-major,
    /// which is the only reason `QuantizedTensor::load` reverses GGUF's dims:
    /// both paths converge on `[out, in]`.)
    ///
    /// The one reordering that *is* needed is the Q/K row permutation — see
    /// [`permute_qk_rows`].
    ///
    /// `tie_word_embeddings` comes from `config.json`; when set (or when
    /// `lm_head.weight` is simply absent) the embedding table doubles as the
    /// output projection, mirroring the GGUF path's `output.weight` fallback.
    pub fn from_safetensors(
        st: &SafeTensorsModel,
        config: &ModelConfig,
        tie_word_embeddings: bool,
    ) -> Result<Self, GlintError> {
        let embed_dim = config.embedding_length as usize;
        let n_heads = config.head_count as usize;
        let n_kv_heads = config.head_count_kv as usize;
        let head_dim = config.head_dim() as usize;
        let n_layers = config.block_count as usize;

        // Number of head dimensions RoPE actually rotates. Matches the
        // `rot_dim` computed in `transformer::forward`, so the permutation
        // covers exactly the rows the rotation will touch.
        let rot_dim = config
            .partial_rotary_factor
            .map(|f| (head_dim as f32 * f) as usize & !1)
            .unwrap_or(head_dim);

        reject_unsupported_layouts(st, n_layers)?;

        eprintln!("Loading token embedding...");
        let token_embedding = hf_matrix(st, "model.embed_tokens.weight", None, embed_dim)?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            eprint!("\rLoading layer {}/{}...", i + 1, n_layers);
            let p = format!("model.layers.{i}");

            let attn_q = hf_matrix(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                Some(n_heads * head_dim),
                embed_dim,
            )?;
            let attn_k = hf_matrix(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                Some(n_kv_heads * head_dim),
                embed_dim,
            )?;

            let ffn_gate = hf_matrix(st, &format!("{p}.mlp.gate_proj.weight"), None, embed_dim)?;
            let ffn_hidden = ffn_gate.rows();

            layers.push(LayerWeights {
                attn_norm: hf_vector(st, &format!("{p}.input_layernorm.weight"), embed_dim)?,
                ffn_norm: hf_vector(
                    st,
                    &format!("{p}.post_attention_layernorm.weight"),
                    embed_dim,
                )?,
                attn_q: permute_qk(attn_q, n_heads, head_dim, rot_dim),
                attn_k: permute_qk(attn_k, n_kv_heads, head_dim, rot_dim),
                attn_v: hf_matrix(
                    st,
                    &format!("{p}.self_attn.v_proj.weight"),
                    Some(n_kv_heads * head_dim),
                    embed_dim,
                )?,
                attn_output: hf_matrix(
                    st,
                    &format!("{p}.self_attn.o_proj.weight"),
                    Some(embed_dim),
                    n_heads * head_dim,
                )?,
                ffn_gate,
                ffn_up: hf_matrix(
                    st,
                    &format!("{p}.mlp.up_proj.weight"),
                    Some(ffn_hidden),
                    embed_dim,
                )?,
                ffn_down: hf_matrix(
                    st,
                    &format!("{p}.mlp.down_proj.weight"),
                    Some(embed_dim),
                    ffn_hidden,
                )?,
            });
        }
        eprintln!("\rLoaded {n_layers}/{n_layers} layers.       ");

        eprintln!("Loading output weights...");
        let output_norm = hf_vector(st, "model.norm.weight", embed_dim)?;

        let output = if tie_word_embeddings || !st.contains("lm_head.weight") {
            token_embedding.clone()
        } else {
            hf_matrix(st, "lm_head.weight", None, embed_dim)?
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

// ── SafeTensors helpers ──────────────────────────────────────────────────────

/// Load a 2-D HF weight as a `QuantizedTensor`, keeping its stored dtype.
///
/// The bytes are handed to the matvec kernels as-is: F32 goes through the F32
/// path, F16/BF16 through the per-row dequantizing fallback. Nothing is widened
/// to f32 at load time, so an F16 checkpoint costs the same RAM here as it does
/// on disk.
///
/// `expect_rows`/`expect_cols` are checked against the header so a checkpoint
/// that disagrees with its own `config.json` fails at load instead of producing
/// a shape panic inside `matvec`. `expect_rows` is `None` where the tensor is
/// the authority (vocabulary size, FFN hidden size).
fn hf_matrix(
    st: &SafeTensorsModel,
    name: &str,
    expect_rows: Option<usize>,
    expect_cols: usize,
) -> Result<QuantizedTensor, GlintError> {
    let view = st
        .get(name)
        .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;
    if view.shape.len() != 2 {
        return Err(GlintError::InvalidTensorShape {
            name: name.to_string(),
            ndim: view.shape.len(),
        });
    }
    // HF linear weights are [out_features, in_features] row-major — already
    // Glint's [rows, cols] convention, so no transpose.
    let (rows, cols) = (view.shape[0], view.shape[1]);
    if cols != expect_cols || expect_rows.is_some_and(|r| r != rows) {
        return Err(GlintError::TensorReadError {
            name: name.to_string(),
            detail: format!(
                "expected shape [{}, {expect_cols}], found [{rows}, {cols}]",
                expect_rows.map_or("*".to_string(), |r| r.to_string())
            ),
        });
    }
    let ggml_type =
        view.dtype
            .to_ggml()
            .ok_or_else(|| GlintError::SafeTensorsUnsupportedDtype {
                name: name.to_string(),
                dtype: view.dtype.name().to_string(),
            })?;
    let bytes = st.tensor_bytes(name)?.to_vec();
    Ok(QuantizedTensor::from_raw(bytes, rows, cols, ggml_type))
}

/// Load a 1-D HF weight (an RMSNorm scale) as an f32 `Tensor`.
fn hf_vector(st: &SafeTensorsModel, name: &str, expect_len: usize) -> Result<Tensor, GlintError> {
    let view = st
        .get(name)
        .ok_or_else(|| GlintError::TensorNotFound(name.to_string()))?;
    if view.shape.len() != 1 {
        return Err(GlintError::InvalidTensorShape {
            name: name.to_string(),
            ndim: view.shape.len(),
        });
    }
    if view.shape[0] != expect_len {
        return Err(GlintError::TensorReadError {
            name: name.to_string(),
            detail: format!("expected {expect_len} elements, found {}", view.shape[0]),
        });
    }
    Ok(Tensor::from_vec(st.tensor_f32(name)?, &[expect_len]))
}

/// Apply [`permute_qk_rows`] to a loaded Q/K projection.
fn permute_qk(
    weight: QuantizedTensor,
    n_heads: usize,
    head_dim: usize,
    rot_dim: usize,
) -> QuantizedTensor {
    let rows = weight.rows();
    let cols = weight.cols();
    let ggml_type = weight.ggml_type();
    // `rows` is validated as `n_heads * head_dim` by the caller, so the row
    // stride below is exact.
    let row_bytes = weight.raw_data().len() / rows.max(1);
    let permuted = permute_qk_rows(weight.raw_data(), n_heads, head_dim, rot_dim, row_bytes);
    QuantizedTensor::from_raw(permuted, rows, cols, ggml_type)
}

/// Reorder the rows of an HF Q/K projection into the RoPE layout Glint expects.
///
/// HF's `apply_rotary_pos_emb` uses `rotate_half`: inside a head, dimension `j`
/// is rotated against dimension `j + rot_dim/2`. Glint's [`crate::tensor::rope`]
/// (like ggml's `NORM` rope mode, which GGUF LLaMA models use) rotates
/// *adjacent* pairs `(2j, 2j+1)`. llama.cpp reconciles the two by permuting
/// `q_proj`/`k_proj` rows when it writes the GGUF file — `permute()` in
/// `convert_hf_to_gguf.py`:
///
/// ```text
/// w.reshape(n_head, 2, rot_dim/2, -1).swapaxes(1, 2).reshape(w.shape)
/// ```
///
/// Loading HF weights directly means doing the same permutation here, or every
/// rotation pairs the wrong dimensions. Written out per head:
///
/// ```text
/// new_row[2*j + a] = old_row[a * rot_dim/2 + j]      for a in {0,1}
/// ```
///
/// so `new_row[2j] = old_row[j]` and `new_row[2j+1] = old_row[j + rot_dim/2]`:
/// the two dimensions HF pairs end up adjacent, which is exactly the pair
/// Glint's `rope` rotates. The frequency also lines up — Glint applies
/// `base^(-2j/head_dim)` to the pair at offset `2j`, and HF applies the same
/// `base^(-2j/head_dim)` to `(j, j + head_dim/2)`.
///
/// Rows past `rot_dim` (partial rotary, e.g. Phi-3) are copied unchanged: RoPE
/// never touches them.
///
/// This is a pure row permutation, so it works on the raw bytes for any dtype
/// whose rows are contiguous — which is every dtype in the format.
fn permute_qk_rows(
    bytes: &[u8],
    n_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    row_bytes: usize,
) -> Vec<u8> {
    let half = rot_dim / 2;
    if half == 0 || row_bytes == 0 || bytes.len() != n_heads * head_dim * row_bytes {
        // Shape does not describe whole heads — leave the bytes alone rather
        // than scrambling them; the caller has already validated the shape, so
        // this is defence in depth.
        return bytes.to_vec();
    }
    let mut out = bytes.to_vec();
    for h in 0..n_heads {
        let head = h * head_dim;
        for j in 0..half {
            for a in 0..2 {
                let src = (head + a * half + j) * row_bytes;
                let dst = (head + 2 * j + a) * row_bytes;
                out[dst..dst + row_bytes].copy_from_slice(&bytes[src..src + row_bytes]);
            }
        }
    }
    out
}

/// Refuse HF layouts the forward pass cannot express, with a message that says
/// which one, instead of failing later on a missing tensor name.
fn reject_unsupported_layouts(st: &SafeTensorsModel, n_layers: usize) -> Result<(), GlintError> {
    for i in 0..n_layers {
        let p = format!("model.layers.{i}");
        // Phi-3 style fused projections.
        if st.contains(&format!("{p}.self_attn.qkv_proj.weight")) {
            return Err(GlintError::HfUnsupported(
                "fused self_attn.qkv_proj (Phi-3 layout) — Glint needs separate \
                 q/k/v projections"
                    .to_string(),
            ));
        }
        if st.contains(&format!("{p}.mlp.gate_up_proj.weight")) {
            return Err(GlintError::HfUnsupported(
                "fused mlp.gate_up_proj (Phi-3 layout) — Glint needs separate \
                 gate/up projections"
                    .to_string(),
            ));
        }
        // Qwen2-style attention biases: the forward pass has no bias term, so
        // loading these weights would silently drop them.
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            if st.contains(&format!("{p}.self_attn.{proj}.bias")) {
                return Err(GlintError::HfUnsupported(format!(
                    "attention bias '{p}.self_attn.{proj}.bias' — Glint's attention \
                     is bias-free"
                )));
            }
        }
        // Qwen3-style per-head query/key norms.
        for norm in ["q_norm", "k_norm"] {
            if st.contains(&format!("{p}.self_attn.{norm}.weight")) {
                return Err(GlintError::HfUnsupported(format!(
                    "per-head '{norm}' (Qwen3 layout) is not implemented"
                )));
            }
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gguf::GgmlType;
    use crate::model::safetensors::test_support::{
        build_f16, build_f32, pseudo_random, spec, TensorSpec,
    };
    use crate::model::safetensors::{HfModelDir, SafeTensorsFile};

    // ── RoPE row permutation ─────────────────────────────────────────────────

    /// Independent reference: the exact numpy expression llama.cpp's converter
    /// uses, `reshape(n_head, 2, half, -1).swapaxes(1, 2).reshape(...)`,
    /// written out index by index over whole rows.
    fn reference_permute(rows: &[Vec<f32>], n_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let half = head_dim / 2;
        let mut out = vec![Vec::new(); rows.len()];
        for h in 0..n_heads {
            for a in 0..2 {
                for j in 0..half {
                    // source index (h, a, j) → destination index (h, j, a)
                    out[h * head_dim + j * 2 + a] = rows[h * head_dim + a * half + j].clone();
                }
            }
        }
        out
    }

    fn rows_to_bytes(rows: &[Vec<f32>]) -> Vec<u8> {
        rows.iter()
            .flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    }

    #[test]
    fn test_permute_qk_rows_matches_numpy_swapaxes() {
        let (n_heads, head_dim, cols) = (3, 4, 2);
        let rows: Vec<Vec<f32>> = (0..n_heads * head_dim)
            .map(|r| (0..cols).map(|c| (r * cols + c) as f32).collect())
            .collect();

        let permuted =
            permute_qk_rows(&rows_to_bytes(&rows), n_heads, head_dim, head_dim, cols * 4);
        assert_eq!(
            permuted,
            rows_to_bytes(&reference_permute(&rows, n_heads, head_dim))
        );
    }

    /// The permutation's whole purpose: after it, Glint's adjacent-pair RoPE
    /// must produce exactly what HuggingFace's `rotate_half` RoPE produces on
    /// the original row order. This is the numerical-equivalence check — it
    /// exercises the pairing *and* the per-pair frequency.
    #[test]
    fn test_permuted_rope_matches_hf_rotate_half() {
        let (head_dim, base, pos) = (8usize, 10000.0f32, 5usize);
        let half = head_dim / 2;
        let q: Vec<f32> = (0..head_dim).map(|i| 0.3 * i as f32 - 1.0).collect();

        // HuggingFace: q_out = q*cos + rotate_half(q)*sin, where
        // rotate_half([x1, x2]) = [-x2, x1] over the two halves, and
        // cos/sin are indexed by j (mod half) with freq = base^(-2j/head_dim).
        let mut hf = vec![0.0f32; head_dim];
        for j in 0..half {
            let freq = base.powf(-2.0 * j as f32 / head_dim as f32);
            let (c, s) = ((pos as f32 * freq).cos(), (pos as f32 * freq).sin());
            hf[j] = q[j] * c - q[j + half] * s;
            hf[j + half] = q[j + half] * c + q[j] * s;
        }

        // Glint: permute the *weight rows* (here, one column's worth: the
        // vector itself), then rotate adjacent pairs.
        let rows: Vec<Vec<f32>> = q.iter().map(|&v| vec![v]).collect();
        let permuted = reference_permute(&rows, 1, head_dim);
        let q_permuted: Vec<f32> = permuted.iter().map(|r| r[0]).collect();
        let rotated = crate::tensor::rope(
            &Tensor::from_vec(q_permuted, &[head_dim]),
            pos,
            head_dim,
            base,
            1.0,
            head_dim,
        );

        // The result is the HF result in permuted row order.
        let hf_rows: Vec<Vec<f32>> = hf.iter().map(|&v| vec![v]).collect();
        let expected: Vec<f32> = reference_permute(&hf_rows, 1, head_dim)
            .iter()
            .map(|r| r[0])
            .collect();
        for (got, want) in rotated.data().iter().zip(&expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "rotated {:?} != HF {:?}",
                rotated.data(),
                expected
            );
        }
    }

    #[test]
    fn test_permute_leaves_non_rotary_rows_untouched() {
        // Partial rotary: only the first `rot_dim` rows of each head move.
        let (n_heads, head_dim, rot_dim, cols) = (2, 8, 4, 1);
        let rows: Vec<Vec<f32>> = (0..n_heads * head_dim).map(|r| vec![r as f32]).collect();
        let bytes = permute_qk_rows(&rows_to_bytes(&rows), n_heads, head_dim, rot_dim, cols * 4);
        let got: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Head 0: rows 0..4 permuted as (0,2,1,3); rows 4..8 unchanged.
        assert_eq!(got[0..8], [0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(got[8..16], [8.0, 10.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    // ── A micro model, written both ways ─────────────────────────────────────

    const EMBED: usize = 8;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = EMBED / HEADS;
    const FFN: usize = 16;
    const VOCAB: usize = 12;
    const LAYERS: usize = 2;

    fn micro_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            context_length: 32,
            embedding_length: EMBED as u32,
            block_count: LAYERS as u32,
            head_count: HEADS as u32,
            head_count_kv: KV_HEADS as u32,
            vocab_size: VOCAB as u32,
            feed_forward_length: Some(FFN as u32),
            rms_norm_eps: 1e-5,
            rope_freq_base: Some(10000.0),
            chat_template: None,
            sliding_window: None,
            rope_scaling_factor: None,
            partial_rotary_factor: None,
        }
    }

    /// Every tensor of the micro model as `(hf_name, ggml_name, rows, cols)`.
    fn tensor_plan() -> Vec<(String, String, usize, usize)> {
        let mut plan = vec![(
            "model.embed_tokens.weight".to_string(),
            "token_embd.weight".to_string(),
            VOCAB,
            EMBED,
        )];
        for i in 0..LAYERS {
            plan.extend([
                (
                    format!("model.layers.{i}.self_attn.q_proj.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    HEADS * HEAD_DIM,
                    EMBED,
                ),
                (
                    format!("model.layers.{i}.self_attn.k_proj.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    KV_HEADS * HEAD_DIM,
                    EMBED,
                ),
                (
                    format!("model.layers.{i}.self_attn.v_proj.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    KV_HEADS * HEAD_DIM,
                    EMBED,
                ),
                (
                    format!("model.layers.{i}.self_attn.o_proj.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    EMBED,
                    HEADS * HEAD_DIM,
                ),
                (
                    format!("model.layers.{i}.mlp.gate_proj.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    FFN,
                    EMBED,
                ),
                (
                    format!("model.layers.{i}.mlp.up_proj.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    FFN,
                    EMBED,
                ),
                (
                    format!("model.layers.{i}.mlp.down_proj.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                    EMBED,
                    FFN,
                ),
                (
                    format!("model.layers.{i}.input_layernorm.weight"),
                    format!("blk.{i}.attn_norm.weight"),
                    EMBED,
                    1,
                ),
                (
                    format!("model.layers.{i}.post_attention_layernorm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    EMBED,
                    1,
                ),
            ]);
        }
        plan.push((
            "model.norm.weight".to_string(),
            "output_norm.weight".to_string(),
            EMBED,
            1,
        ));
        plan.push((
            "lm_head.weight".to_string(),
            "output.weight".to_string(),
            VOCAB,
            EMBED,
        ));
        plan
    }

    /// Deterministic weights for the whole micro model, keyed by HF name.
    fn micro_weights() -> Vec<(String, String, usize, usize, Vec<f32>)> {
        tensor_plan()
            .into_iter()
            .enumerate()
            .map(|(seed, (hf, ggml, rows, cols))| {
                let data = pseudo_random(rows * cols, seed as u64 + 1);
                (hf, ggml, rows, cols, data)
            })
            .collect()
    }

    fn micro_safetensors() -> Vec<u8> {
        let specs: Vec<TensorSpec> = micro_weights()
            .into_iter()
            .map(|(hf, _, rows, cols, data)| {
                // 1-D norm vectors are stored as [n]; matrices as [out, in].
                let shape: Vec<usize> = if cols == 1 {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                spec(&hf, &shape, data)
            })
            .collect();
        build_f32(&specs)
    }

    /// Write the same weights as a GGUF file, exactly as llama.cpp's converter
    /// would: dimensions reversed (column-major) and q/k rows pre-permuted.
    fn micro_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        let metadata: Vec<(&str, MetaValue)> = vec![
            ("general.architecture", MetaValue::Str("llama")),
            ("llama.context_length", MetaValue::U32(32)),
            ("llama.embedding_length", MetaValue::U32(EMBED as u32)),
            ("llama.block_count", MetaValue::U32(LAYERS as u32)),
            ("llama.attention.head_count", MetaValue::U32(HEADS as u32)),
            (
                "llama.attention.head_count_kv",
                MetaValue::U32(KV_HEADS as u32),
            ),
            ("llama.feed_forward_length", MetaValue::U32(FFN as u32)),
            ("llama.vocab_size", MetaValue::U32(VOCAB as u32)),
        ];

        let tensors: Vec<(String, usize, usize, Vec<f32>)> = micro_weights()
            .into_iter()
            .map(|(hf, ggml, rows, cols, data)| {
                let data = if hf.ends_with("q_proj.weight") || hf.ends_with("k_proj.weight") {
                    let n_heads = if hf.ends_with("q_proj.weight") {
                        HEADS
                    } else {
                        KV_HEADS
                    };
                    let rows_vec: Vec<Vec<f32>> = data.chunks(cols).map(|c| c.to_vec()).collect();
                    reference_permute(&rows_vec, n_heads, HEAD_DIM)
                        .into_iter()
                        .flatten()
                        .collect()
                } else {
                    data
                };
                (ggml, rows, cols, data)
            })
            .collect();

        out.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in &metadata {
            write_gguf_string(&mut out, key);
            match value {
                MetaValue::Str(s) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    write_gguf_string(&mut out, s);
                }
                MetaValue::U32(v) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // Tensor descriptors. GGUF dims are column-major, so a row-major
        // [rows, cols] matrix is written as [cols, rows]; 1-D vectors as [n].
        let mut offset = 0u64;
        let mut payload: Vec<(u64, Vec<f32>)> = Vec::new();
        for (name, rows, cols, data) in &tensors {
            write_gguf_string(&mut out, name);
            let dims: Vec<u64> = if *cols == 1 {
                vec![*rows as u64]
            } else {
                vec![*cols as u64, *rows as u64]
            };
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in &dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            payload.push((offset, data.clone()));
            offset += (data.len() * 4) as u64;
            offset = offset.div_ceil(32) * 32; // per-tensor alignment
        }

        while out.len() % 32 != 0 {
            out.push(0);
        }
        let data_start = out.len();
        out.resize(data_start + offset as usize, 0);
        for (off, data) in payload {
            let mut at = data_start + off as usize;
            for v in data {
                out[at..at + 4].copy_from_slice(&v.to_le_bytes());
                at += 4;
            }
        }
        out
    }

    enum MetaValue {
        Str(&'static str),
        U32(u32),
    }

    fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn load_micro_safetensors() -> TransformerWeights {
        let st =
            SafeTensorsModel::from_files(vec![
                SafeTensorsFile::from_bytes(micro_safetensors()).unwrap()
            ])
            .unwrap();
        TransformerWeights::from_safetensors(&st, &micro_config(), false).unwrap()
    }

    /// The headline layout test: loading the same model through the GGUF path
    /// and the safetensors path must produce byte-identical weights. This pins
    /// down the row/column orientation *and* the Q/K RoPE permutation against
    /// the format Glint already runs correctly.
    #[test]
    fn test_from_safetensors_matches_gguf_load() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("micro.gguf");
        std::fs::write(&gguf_path, micro_gguf()).unwrap();
        let gguf = GgufModel::load(&gguf_path).unwrap();
        let config = ModelConfig::from_metadata(&gguf.metadata).unwrap();
        assert_eq!(config.embedding_length, EMBED as u32);

        let from_gguf = TransformerWeights::load(&gguf, &config).unwrap();
        let from_st = load_micro_safetensors();

        assert_eq!(
            from_st.token_embedding.raw_data(),
            from_gguf.token_embedding.raw_data()
        );
        assert_eq!(
            (
                from_st.token_embedding.rows(),
                from_st.token_embedding.cols()
            ),
            (VOCAB, EMBED)
        );
        assert_eq!(from_st.output.raw_data(), from_gguf.output.raw_data());
        assert_eq!(from_st.output_norm.data(), from_gguf.output_norm.data());
        assert_eq!(from_st.layers.len(), from_gguf.layers.len());

        for (i, (st, gg)) in from_st.layers.iter().zip(&from_gguf.layers).enumerate() {
            assert_eq!(st.attn_norm.data(), gg.attn_norm.data(), "layer {i} norm");
            assert_eq!(st.ffn_norm.data(), gg.ffn_norm.data(), "layer {i} ffn norm");
            for (name, a, b) in [
                ("attn_q", &st.attn_q, &gg.attn_q),
                ("attn_k", &st.attn_k, &gg.attn_k),
                ("attn_v", &st.attn_v, &gg.attn_v),
                ("attn_output", &st.attn_output, &gg.attn_output),
                ("ffn_gate", &st.ffn_gate, &gg.ffn_gate),
                ("ffn_up", &st.ffn_up, &gg.ffn_up),
                ("ffn_down", &st.ffn_down, &gg.ffn_down),
            ] {
                assert_eq!(
                    (a.rows(), a.cols()),
                    (b.rows(), b.cols()),
                    "layer {i} {name} shape"
                );
                assert_eq!(a.raw_data(), b.raw_data(), "layer {i} {name} data");
            }
        }
    }

    /// F16 weights keep their stored width (no widening to f32 at load) and
    /// still run: the permutation operates on 2-byte rows and the fallback
    /// matvec dequantizes per row, so the logits track the F32 model closely.
    #[test]
    fn test_f16_weights_load_natively_and_track_f32() {
        let specs: Vec<TensorSpec> = micro_weights()
            .into_iter()
            .map(|(hf, _, rows, cols, data)| {
                let shape: Vec<usize> = if cols == 1 {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                spec(&hf, &shape, data)
            })
            .collect();
        let st = SafeTensorsModel::from_files(vec![
            SafeTensorsFile::from_bytes(build_f16(&specs)).unwrap()
        ])
        .unwrap();
        let config = micro_config();
        let weights = TransformerWeights::from_safetensors(&st, &config, false).unwrap();

        assert_eq!(weights.layers[0].attn_q.ggml_type(), GgmlType::F16);
        assert_eq!(
            weights.layers[0].attn_q.raw_data().len(),
            HEADS * HEAD_DIM * EMBED * 2,
            "F16 weights must not be widened to f32"
        );

        let f16_logits = crate::transformer::forward(&weights, &config, &[1, 5, 3]);
        let f32_logits =
            crate::transformer::forward(&load_micro_safetensors(), &config, &[1, 5, 3]);
        for (a, b) in f16_logits.data().iter().zip(f32_logits.data()) {
            assert!(
                (a - b).abs() < 1e-2,
                "f16 {:?} vs f32 {:?}",
                f16_logits.data(),
                f32_logits.data()
            );
        }
    }

    #[test]
    fn test_from_safetensors_forward_pass_produces_finite_logits() {
        let weights = load_micro_safetensors();
        let config = micro_config();
        let logits = crate::transformer::forward(&weights, &config, &[1, 5, 3]);
        assert_eq!(logits.shape(), &[VOCAB]);
        assert!(
            logits.data().iter().all(|v| v.is_finite()),
            "logits: {:?}",
            logits.data()
        );
    }

    #[test]
    fn test_tie_word_embeddings_reuses_the_embedding_table() {
        let st =
            SafeTensorsModel::from_files(vec![
                SafeTensorsFile::from_bytes(micro_safetensors()).unwrap()
            ])
            .unwrap();
        let weights = TransformerWeights::from_safetensors(&st, &micro_config(), true).unwrap();
        assert_eq!(
            weights.output.raw_data(),
            weights.token_embedding.raw_data()
        );
    }

    #[test]
    fn test_missing_lm_head_falls_back_to_the_embedding_table() {
        let specs: Vec<TensorSpec> = micro_weights()
            .into_iter()
            .filter(|(hf, ..)| hf != "lm_head.weight")
            .map(|(hf, _, rows, cols, data)| {
                let shape: Vec<usize> = if cols == 1 {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                spec(&hf, &shape, data)
            })
            .collect();
        let st = SafeTensorsModel::from_files(vec![
            SafeTensorsFile::from_bytes(build_f32(&specs)).unwrap()
        ])
        .unwrap();
        let weights = TransformerWeights::from_safetensors(&st, &micro_config(), false).unwrap();
        assert_eq!(
            weights.output.raw_data(),
            weights.token_embedding.raw_data()
        );
    }

    #[test]
    fn test_missing_tensor_is_named() {
        let specs: Vec<TensorSpec> = micro_weights()
            .into_iter()
            .filter(|(hf, ..)| !hf.ends_with("layers.1.mlp.up_proj.weight"))
            .map(|(hf, _, rows, cols, data)| {
                let shape: Vec<usize> = if cols == 1 {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                spec(&hf, &shape, data)
            })
            .collect();
        let st = SafeTensorsModel::from_files(vec![
            SafeTensorsFile::from_bytes(build_f32(&specs)).unwrap()
        ])
        .unwrap();
        let err = TransformerWeights::from_safetensors(&st, &micro_config(), false)
            .err()
            .unwrap();
        assert!(
            err.to_string()
                .contains("model.layers.1.mlp.up_proj.weight"),
            "got: {err}"
        );
    }

    #[test]
    fn test_shape_disagreeing_with_config_is_rejected() {
        let mut config = micro_config();
        config.embedding_length = EMBED as u32 * 2;
        config.head_count = HEADS as u32; // keeps head_dim divisible
        let st =
            SafeTensorsModel::from_files(vec![
                SafeTensorsFile::from_bytes(micro_safetensors()).unwrap()
            ])
            .unwrap();
        let err = TransformerWeights::from_safetensors(&st, &config, false)
            .err()
            .unwrap();
        assert!(err.to_string().contains("expected shape"), "got: {err}");
    }

    #[test]
    fn test_attention_bias_is_rejected() {
        let mut specs: Vec<TensorSpec> = micro_weights()
            .into_iter()
            .map(|(hf, _, rows, cols, data)| {
                let shape: Vec<usize> = if cols == 1 {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                spec(&hf, &shape, data)
            })
            .collect();
        specs.push(spec(
            "model.layers.0.self_attn.q_proj.bias",
            &[EMBED],
            vec![0.1; EMBED],
        ));
        let st = SafeTensorsModel::from_files(vec![
            SafeTensorsFile::from_bytes(build_f32(&specs)).unwrap()
        ])
        .unwrap();
        let err = TransformerWeights::from_safetensors(&st, &micro_config(), false)
            .err()
            .unwrap();
        assert!(err.to_string().contains("bias-free"), "got: {err}");
    }

    // ── Full directory load ──────────────────────────────────────────────────

    /// Byte-level BPE tokenizer covering the ASCII letters the test prompt uses.
    fn micro_tokenizer_json() -> String {
        let pieces = ["<unk>", "<s>", "</s>", "h", "i", "Ġ", "t", "e", "r", "hi"];
        let entries: Vec<String> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| format!("\"{p}\":{i}"))
            .collect();
        format!(
            r#"{{
                "pre_tokenizer": {{"type": "ByteLevel"}},
                "decoder": {{"type": "ByteLevel"}},
                "model": {{"type": "BPE", "vocab": {{{}}}, "merges": ["h i"]}}
            }}"#,
            entries.join(",")
        )
    }

    fn micro_config_json() -> String {
        format!(
            r#"{{
                "model_type": "llama",
                "hidden_size": {EMBED},
                "num_hidden_layers": {LAYERS},
                "num_attention_heads": {HEADS},
                "num_key_value_heads": {KV_HEADS},
                "intermediate_size": {FFN},
                "vocab_size": {VOCAB},
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-05,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "bos_token_id": 1,
                "eos_token_id": 2
            }}"#
        )
    }

    /// End to end: a HuggingFace-shaped directory on disk → config, tokenizer,
    /// weights → a forward pass with finite logits.
    #[test]
    fn test_hf_model_directory_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), micro_config_json()).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), micro_tokenizer_json()).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"add_bos_token": true, "chat_template": "<|im_start|>{{ x }}<|im_end|>"}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors"), micro_safetensors()).unwrap();

        let hf = HfModelDir::open(dir.path()).unwrap();
        assert_eq!(hf.config.embedding_length, EMBED as u32);
        assert_eq!(hf.config.block_count, LAYERS as u32);
        assert_eq!(hf.tokenizer.eos_token_id, 2);
        assert!(hf
            .config
            .chat_template
            .as_deref()
            .unwrap()
            .contains("im_start"));
        assert!(!hf.tie_word_embeddings);

        let weights =
            TransformerWeights::from_safetensors(&hf.weights, &hf.config, hf.tie_word_embeddings)
                .unwrap();
        let tokens = hf.tokenizer.encode_prompt("hi");
        assert_eq!(tokens[0], hf.tokenizer.bos_token_id);

        let logits = crate::transformer::forward(&weights, &hf.config, &tokens);
        assert_eq!(logits.shape(), &[VOCAB]);
        assert!(logits.data().iter().all(|v| v.is_finite()));

        // Opening via the weight file rather than the directory is equivalent.
        let via_file = HfModelDir::open(&dir.path().join("model.safetensors")).unwrap();
        assert_eq!(via_file.config.block_count, LAYERS as u32);
    }
}
