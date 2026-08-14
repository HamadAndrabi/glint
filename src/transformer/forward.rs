//! Decoder-only transformer forward pass (LLaMA-style).

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::weights::{LayerWeights, TransformerWeights};
use crate::backend::GpuBackend;
use crate::cache::{KvCache, KvCacheQ8, KvStore, PagePool, PagedKvCache};
use crate::model::config::ModelConfig;
use crate::model::lora::{LoraLayerAdapters, LoraWeights};
use crate::sampling::Sampler;
use crate::tensor::{self, QuantizedTensor, Tensor};

// ── GPU dispatch helpers ──────────────────────────────────────────────────────

/// Reborrow `Option<&mut GpuBackend>` so it can be used multiple times.
fn reborrow<'a>(gpu: &'a mut Option<&mut GpuBackend>) -> Option<&'a mut GpuBackend> {
    gpu.as_mut().map(|g| &mut **g)
}

/// Matvec that dispatches to GPU when available, CPU otherwise.
fn matvec_maybe_gpu(qt: &QuantizedTensor, input: &[f32], gpu: Option<&mut GpuBackend>) -> Tensor {
    #[cfg(feature = "vulkan")]
    if let Some(g) = gpu {
        return qt.matvec_gpu(input, g);
    }
    let _ = gpu;
    qt.matvec(input)
}

/// RMS normalization — GPU when available, CPU otherwise.
#[cfg(feature = "vulkan")]
fn rms_norm_maybe_gpu(x: &Tensor, weight: &Tensor, eps: f32, gpu: Option<&GpuBackend>) -> Tensor {
    if let Some(g) = gpu {
        if let Ok(data) = g.rms_norm(x.data(), weight.data(), eps) {
            return Tensor::from_vec(data, x.shape());
        }
    }
    tensor::rms_norm(x, weight, eps)
}

/// Element-wise add — GPU when available, CPU otherwise.
#[cfg(feature = "vulkan")]
fn add_maybe_gpu(a: &Tensor, b: &Tensor, gpu: Option<&GpuBackend>) -> Tensor {
    if let Some(g) = gpu {
        if let Ok(data) = g.add(a.data(), b.data()) {
            return Tensor::from_vec(data, a.shape());
        }
    }
    tensor::add(a, b)
}

/// SiLU(gate) * up — GPU when available, CPU otherwise.
#[cfg(feature = "vulkan")]
fn silu_mul_maybe_gpu(gate: &Tensor, up: &Tensor, gpu: Option<&GpuBackend>) -> Tensor {
    if let Some(g) = gpu {
        if let Ok(data) = g.silu_mul(gate.data(), up.data()) {
            return Tensor::from_vec(data, gate.shape());
        }
    }
    tensor::mul(&tensor::silu(gate), up)
}

/// Run a full forward pass: token_ids → logits.
///
/// Recomputes attention over all positions each time (no KV-cache).
pub fn forward(weights: &TransformerWeights, config: &ModelConfig, token_ids: &[u32]) -> Tensor {
    let n_tokens = token_ids.len();
    let mut hidden_states: Vec<Tensor> = token_ids
        .iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize))
        .collect();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        eprint!("\r  Layer {}/{}...", layer_idx + 1, config.block_count);
        let mut new_hidden_states = Vec::with_capacity(n_tokens);
        for pos in 0..n_tokens {
            let normed =
                tensor::rms_norm(&hidden_states[pos], &layer.attn_norm, config.rms_norm_eps);
            let mut attn_out = attention(&normed, &hidden_states, layer, config, pos);
            if let Some(norm) = &layer.post_attn_norm {
                attn_out = tensor::rms_norm(&attn_out, norm, config.rms_norm_eps);
            }
            let after_attn = tensor::add(&hidden_states[pos], &attn_out);
            let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
            let mut ffn_out = feed_forward(&normed_ffn, layer, None, &mut None);
            if let Some(norm) = &layer.post_ffn_norm {
                ffn_out = tensor::rms_norm(&ffn_out, norm, config.rms_norm_eps);
            }
            new_hidden_states.push(tensor::add(&after_attn, &ffn_out));
        }
        hidden_states = new_hidden_states;
    }
    eprintln!();

    let last_hidden = &hidden_states[n_tokens - 1];
    let normed = tensor::rms_norm(last_hidden, &weights.output_norm, config.rms_norm_eps);
    let mut logits = weights.output.matvec(normed.data());
    if let Some(cap) = config.final_logit_softcapping {
        tensor::logit_softcap_in_place(logits.data_mut(), cap);
    }
    logits
}

/// Multi-head self-attention (no KV-cache; recomputes K/V for all positions).
fn attention(
    x: &Tensor,
    all_hidden: &[Tensor],
    layer: &LayerWeights,
    config: &ModelConfig,
    pos: usize,
) -> Tensor {
    let embed_dim = config.embedding_length as usize;
    let n_heads = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;
    let kv_group_size = n_heads / n_kv_heads;
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scaling = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim = config
        .partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);

    let mut q_all = layer.attn_q.matvec(x.data());
    let mut k_cur = layer.attn_k.matvec(x.data());
    let mut v_cur = layer.attn_v.matvec(x.data());
    if let Some(bias) = &layer.attn_q_bias {
        tensor::add_in_place(q_all.data_mut(), bias.data());
    }
    if let Some(bias) = &layer.attn_k_bias {
        tensor::add_in_place(k_cur.data_mut(), bias.data());
    }
    if let Some(bias) = &layer.attn_v_bias {
        tensor::add_in_place(v_cur.data_mut(), bias.data());
    }
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base, rope_scaling, rot_dim);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base, rope_scaling, rot_dim);

    let mut k_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    let mut v_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    for (p, hidden) in all_hidden.iter().enumerate().take(pos) {
        let normed_p = tensor::rms_norm(hidden, &layer.attn_norm, config.rms_norm_eps);
        let mut k_p = layer.attn_k.matvec(normed_p.data());
        let mut v_p = layer.attn_v.matvec(normed_p.data());
        if let Some(bias) = &layer.attn_k_bias {
            tensor::add_in_place(k_p.data_mut(), bias.data());
        }
        if let Some(bias) = &layer.attn_v_bias {
            tensor::add_in_place(v_p.data_mut(), bias.data());
        }
        k_cache.push(tensor::rope(
            &k_p,
            p,
            head_dim,
            freq_base,
            rope_scaling,
            rot_dim,
        ));
        v_cache.push(v_p);
    }
    k_cache.push(k_cur);
    v_cache.push(v_cur);

    let seq_len = k_cache.len();
    let scale = config.query_pre_attn_scalar.unwrap_or(1.0 / (head_dim as f32).sqrt());
    let mut attn_output = vec![0.0f32; embed_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q_all.data()[q_offset..q_offset + head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for (s, k_pos) in k_cache.iter().enumerate() {
            let k_head = &k_pos.data()[kv_h * head_dim..(kv_h + 1) * head_dim];
            let mut dot = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() * scale;
            if let Some(cap) = config.attn_logit_softcapping {
                dot = cap * (dot / cap).tanh();
            }
            scores[s] = dot;
        }
        let scores_t = Tensor::from_vec(scores, &[seq_len]);
        let attn_weights = tensor::softmax(&scores_t);
        for (s, v_pos) in v_cache.iter().enumerate() {
            let w = attn_weights.get_flat(s);
            let v_head = &v_pos.data()[kv_h * head_dim..(kv_h + 1) * head_dim];
            for d in 0..head_dim {
                attn_output[q_offset + d] += w * v_head[d];
            }
        }
    }

    let attn_vec = Tensor::from_vec(attn_output, &[embed_dim]);
    layer.attn_output.matvec(attn_vec.data())
}

/// CPU multi-head flash attention for the single query at `pos`.
///
/// Shared by the single-sequence decode ([`attention_cached`]) and the batched
/// decode step ([`forward_batch`]): a sequence must walk its KV cache exactly
/// the same way whether or not it shares the step with other sequences, so
/// there is one loop rather than two that have to be kept in agreement.
///
/// `out` is `n_heads * head_dim` long and must be pre-zeroed —
/// [`tensor::flash_attn_1d`] accumulates into it.
fn attn_heads_cpu(
    q_all: &[f32],
    cache: &dyn KvStore,
    layer_idx: usize,
    window_start: usize,
    attend_len: usize,
    n_heads: usize,
    kv_group_size: usize,
    head_dim: usize,
    scale: f32,
    softcap: Option<f32>,
    out: &mut [f32],
) {
    // Tiled online softmax — O(1) extra memory per head.
    for h in 0..n_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        tensor::flash_attn_1d_ext(
            &q_all[q_offset..q_offset + head_dim],
            cache,
            layer_idx,
            kv_h,
            window_start,
            attend_len,
            head_dim,
            scale,
            softcap,
            &mut out[q_offset..q_offset + head_dim],
        );
    }
}

/// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
fn feed_forward(
    x: &Tensor,
    layer: &LayerWeights,
    lora: Option<&LoraLayerAdapters>,
    gpu: &mut Option<&mut GpuBackend>,
) -> Tensor {
    let mut gate = matvec_maybe_gpu(&layer.ffn_gate, x.data(), reborrow(gpu));
    let mut up = matvec_maybe_gpu(&layer.ffn_up, x.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.ffn_gate {
            a.apply(x.data(), gate.data_mut());
        }
        if let Some(a) = &ll.ffn_up {
            a.apply(x.data(), up.data_mut());
        }
    }
    #[cfg(feature = "vulkan")]
    let hidden = silu_mul_maybe_gpu(&gate, &up, gpu.as_deref());
    #[cfg(not(feature = "vulkan"))]
    let hidden = tensor::mul(&tensor::silu(&gate), &up);
    let mut out = matvec_maybe_gpu(&layer.ffn_down, hidden.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.ffn_down {
            a.apply(hidden.data(), out.data_mut());
        }
    }
    out
}

/// Compute text embeddings for multiple token sequences in parallel.
///
/// Each input sequence is processed independently; rayon parallelises across
/// sequences when the `rayon` feature is enabled.
///
/// Returns one embedding vector per input sequence.  The single-input fast path
/// delegates to [`embed`] directly.
pub fn embed_batch(
    weights: &TransformerWeights,
    config: &ModelConfig,
    inputs: &[&[u32]],
) -> Vec<Vec<f32>> {
    if inputs.is_empty() {
        return vec![];
    }
    if inputs.len() == 1 {
        return vec![embed(weights, config, inputs[0])];
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        inputs
            .par_iter()
            .map(|token_ids| embed(weights, config, token_ids))
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    inputs
        .iter()
        .map(|token_ids| embed(weights, config, token_ids))
        .collect()
}

/// Compute a text embedding by mean-pooling the final hidden states.
pub fn embed(weights: &TransformerWeights, config: &ModelConfig, token_ids: &[u32]) -> Vec<f32> {
    let n_tokens = token_ids.len();
    let embed_dim = config.embedding_length as usize;
    let mut hidden_states: Vec<Tensor> = token_ids
        .iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize))
        .collect();
    for layer in weights.layers.iter() {
        let mut new_hs = Vec::with_capacity(n_tokens);
        for pos in 0..n_tokens {
            let normed =
                tensor::rms_norm(&hidden_states[pos], &layer.attn_norm, config.rms_norm_eps);
            let mut attn_out = attention(&normed, &hidden_states, layer, config, pos);
            if let Some(norm) = &layer.post_attn_norm {
                attn_out = tensor::rms_norm(&attn_out, norm, config.rms_norm_eps);
            }
            let after_attn = tensor::add(&hidden_states[pos], &attn_out);
            let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
            let mut ffn_out = feed_forward(&normed_ffn, layer, None, &mut None);
            if let Some(norm) = &layer.post_ffn_norm {
                ffn_out = tensor::rms_norm(&ffn_out, norm, config.rms_norm_eps);
            }
            new_hs.push(tensor::add(&after_attn, &ffn_out));
        }
        hidden_states = new_hs;
    }
    let mut embedding = vec![0.0f32; embed_dim];
    for hidden in &hidden_states {
        let normed = tensor::rms_norm(hidden, &weights.output_norm, config.rms_norm_eps);
        for (acc, &v) in embedding.iter_mut().zip(normed.data()) {
            *acc += v;
        }
    }
    for acc in &mut embedding {
        *acc /= n_tokens as f32;
    }
    embedding
}

/// Greedy decoding without KV-cache (recomputes full attention each step).
pub fn generate_greedy(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();
    for step in 0..max_new_tokens {
        eprintln!(
            "--- Step {}/{} (seq_len={}) ---",
            step + 1,
            max_new_tokens,
            tokens.len()
        );
        let logits = forward(weights, config, &tokens);
        let next = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tokens.push(next);
        if next == 2 {
            break;
        }
    }
    tokens
}

// ── KV-cached single-token forward pass ──────────────────────────────────────

/// Forward pass for a single token at position `pos`.
///
/// Writes K/V for this position into `cache` and returns the logits.
/// Use `forward_prefill` for the initial prompt; `forward_one` for decode.
pub fn forward_one(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_id: u32,
    pos: usize,
    cache: &mut dyn KvStore,
    gpu: &mut Option<&mut GpuBackend>,
) -> Tensor {
    forward_one_lora(weights, config, token_id, pos, cache, gpu, None)
}

/// Like [`forward_one`] but accepts an optional per-request LoRA override.
///
/// When `lora` is `Some`, it takes precedence over `weights.lora`; when
/// `None`, falls back to `weights.lora` as usual.
pub fn forward_one_lora(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_id: u32,
    pos: usize,
    cache: &mut dyn KvStore,
    gpu: &mut Option<&mut GpuBackend>,
    lora: Option<&LoraWeights>,
) -> Tensor {
    let mut hidden = weights.token_embedding.row_as_f32(token_id as usize);

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let effective_lora = lora.or(weights.lora.as_ref());
        let lora_layer = effective_lora.and_then(|l| l.layers.get(layer_idx));
        #[cfg(feature = "vulkan")]
        let normed = rms_norm_maybe_gpu(
            &hidden,
            &layer.attn_norm,
            config.rms_norm_eps,
            gpu.as_deref(),
        );
        #[cfg(not(feature = "vulkan"))]
        let normed = tensor::rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps);
        let mut attn_out = attention_cached(
            &normed, layer, lora_layer, config, pos, layer_idx, cache, gpu,
        );
        if let Some(norm) = &layer.post_attn_norm {
            attn_out = tensor::rms_norm(&attn_out, norm, config.rms_norm_eps);
        }
        #[cfg(feature = "vulkan")]
        let after_attn = add_maybe_gpu(&hidden, &attn_out, gpu.as_deref());
        #[cfg(not(feature = "vulkan"))]
        let after_attn = tensor::add(&hidden, &attn_out);
        #[cfg(feature = "vulkan")]
        let normed_ffn = rms_norm_maybe_gpu(
            &after_attn,
            &layer.ffn_norm,
            config.rms_norm_eps,
            gpu.as_deref(),
        );
        #[cfg(not(feature = "vulkan"))]
        let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
        let mut ffn_out = feed_forward(&normed_ffn, layer, lora_layer, gpu);
        if let Some(norm) = &layer.post_ffn_norm {
            ffn_out = tensor::rms_norm(&ffn_out, norm, config.rms_norm_eps);
        }
        #[cfg(feature = "vulkan")]
        {
            hidden = add_maybe_gpu(&after_attn, &ffn_out, gpu.as_deref());
        }
        #[cfg(not(feature = "vulkan"))]
        {
            hidden = tensor::add(&after_attn, &ffn_out);
        }
    }

    cache.advance();
    #[cfg(feature = "vulkan")]
    let normed = rms_norm_maybe_gpu(
        &hidden,
        &weights.output_norm,
        config.rms_norm_eps,
        gpu.as_deref(),
    );
    #[cfg(not(feature = "vulkan"))]
    let normed = tensor::rms_norm(&hidden, &weights.output_norm, config.rms_norm_eps);
    let mut logits = matvec_maybe_gpu(&weights.output, normed.data(), reborrow(gpu));
    if let Some(cap) = config.final_logit_softcapping {
        tensor::logit_softcap_in_place(logits.data_mut(), cap);
    }
    logits
}

fn attention_cached(
    x: &Tensor,
    layer: &LayerWeights,
    lora: Option<&LoraLayerAdapters>,
    config: &ModelConfig,
    pos: usize,
    layer_idx: usize,
    cache: &mut dyn KvStore,
    gpu: &mut Option<&mut GpuBackend>,
) -> Tensor {
    let embed_dim = config.embedding_length as usize;
    let n_heads = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;
    let kv_group_size = n_heads / n_kv_heads;
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scaling = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim = config
        .partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);

    let mut q_all = matvec_maybe_gpu(&layer.attn_q, x.data(), reborrow(gpu));
    let mut k_cur = matvec_maybe_gpu(&layer.attn_k, x.data(), reborrow(gpu));
    let mut v_cur = matvec_maybe_gpu(&layer.attn_v, x.data(), reborrow(gpu));
    if let Some(bias) = &layer.attn_q_bias {
        tensor::add_in_place(q_all.data_mut(), bias.data());
    }
    if let Some(bias) = &layer.attn_k_bias {
        tensor::add_in_place(k_cur.data_mut(), bias.data());
    }
    if let Some(bias) = &layer.attn_v_bias {
        tensor::add_in_place(v_cur.data_mut(), bias.data());
    }
    if let Some(ll) = lora {
        if let Some(a) = &ll.attn_q {
            a.apply(x.data(), q_all.data_mut());
        }
        if let Some(a) = &ll.attn_k {
            a.apply(x.data(), k_cur.data_mut());
        }
        if let Some(a) = &ll.attn_v {
            a.apply(x.data(), v_cur.data_mut());
        }
    }
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base, rope_scaling, rot_dim);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base, rope_scaling, rot_dim);

    cache.write(layer_idx, pos, k_cur.data(), v_cur.data());

    let window_start = if config.sliding_window_alternating && !layer_idx.is_multiple_of(2) {
        0 // Full context on odd layers (Gemma 2)
    } else {
        config
            .sliding_window
            .map(|w| (pos as i64 - w as i64 + 1).max(0) as usize)
            .unwrap_or(0)
    };
    let attend_len = pos + 1 - window_start;
    let scale = config.query_pre_attn_scalar.unwrap_or(1.0 / (head_dim as f32).sqrt());
    let mut attn_output = vec![0.0f32; embed_dim];

    let cache_ro: &dyn KvStore = &*cache;

    // ── GPU attention path ──────────────────────────────────────────────────
    #[cfg(feature = "vulkan")]
    let gpu_attn: Option<Tensor> = {
        if let Some(ref mut g) = *gpu {
            if attend_len <= 4096 {
                if let Some(kv_buf) = cache_ro.gpu_buffer() {
                    // Path 1: GPU-resident K/V — no full-context upload needed.
                    g.attention_resident(
                        q_all.data(),
                        kv_buf,
                        layer_idx,
                        window_start,
                        attend_len,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        scale,
                    )
                    .ok()
                    .map(|data| Tensor::from_vec(data, &[embed_dim]))
                } else {
                    // Path 2: CPU cache — copy full window to GPU, then dispatch.
                    let kv_stride = n_kv_heads * head_dim;
                    let mut k_flat = vec![0.0f32; attend_len * kv_stride];
                    let mut v_flat = vec![0.0f32; attend_len * kv_stride];
                    for i in 0..attend_len {
                        for kv_h in 0..n_kv_heads {
                            let off = i * kv_stride + kv_h * head_dim;
                            cache_ro.read_k_head(
                                layer_idx,
                                window_start + i,
                                kv_h,
                                head_dim,
                                &mut k_flat[off..off + head_dim],
                            );
                            cache_ro.read_v_head(
                                layer_idx,
                                window_start + i,
                                kv_h,
                                head_dim,
                                &mut v_flat[off..off + head_dim],
                            );
                        }
                    }
                    g.attention(
                        q_all.data(),
                        &k_flat,
                        &v_flat,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        attend_len as u32,
                        scale,
                    )
                    .ok()
                    .map(|data| Tensor::from_vec(data, &[embed_dim]))
                }
            } else {
                None
            }
        } else {
            None
        }
    };
    #[cfg(not(feature = "vulkan"))]
    let gpu_attn: Option<Tensor> = None;

    let attn_vec = if let Some(t) = gpu_attn {
        t
    } else {
        attn_heads_cpu(
            q_all.data(),
            cache_ro,
            layer_idx,
            window_start,
            attend_len,
            n_heads,
            kv_group_size,
            head_dim,
            scale,
            config.attn_logit_softcapping,
            &mut attn_output,
        );
        Tensor::from_vec(attn_output, &[embed_dim])
    };
    let mut out = matvec_maybe_gpu(&layer.attn_output, attn_vec.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.attn_output {
            a.apply(attn_vec.data(), out.data_mut());
        }
    }
    out
}

// ── Batched prefill (Phase 13) ────────────────────────────────────────────────

/// Process all `token_ids` through the transformer in one call.
///
/// Writes positions `pos_offset .. pos_offset + token_ids.len()` into `cache`.
/// Returns logits for the **last** token.
///
/// Q/K/V projections and FFN are parallelised across all positions via rayon,
/// which is significantly faster than calling `forward_one` in a loop for
/// long prompts.
pub fn forward_prefill(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
    cache: &mut dyn KvStore,
    pos_offset: usize,
    gpu: &mut Option<&mut GpuBackend>,
) -> Tensor {
    forward_prefill_lora(weights, config, token_ids, cache, pos_offset, gpu, None)
}

/// Like [`forward_prefill`] but accepts an optional per-request LoRA override.
pub fn forward_prefill_lora(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
    cache: &mut dyn KvStore,
    pos_offset: usize,
    gpu: &mut Option<&mut GpuBackend>,
    lora: Option<&LoraWeights>,
) -> Tensor {
    let all = forward_prefill_inner(weights, config, token_ids, cache, pos_offset, gpu, lora);
    all.into_iter()
        .last()
        .unwrap_or_else(|| Tensor::zeros(&[config.vocab_size as usize]))
}

/// Like `forward_prefill` but returns logits for **every** input position.
///
/// Used by speculative decoding to verify all draft tokens in one batched
/// target-model call, avoiding k separate `forward_one` calls.
pub fn forward_prefill_all(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
    cache: &mut dyn KvStore,
    pos_offset: usize,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<Tensor> {
    forward_prefill_inner(weights, config, token_ids, cache, pos_offset, gpu, None)
}

fn forward_prefill_inner(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
    cache: &mut dyn KvStore,
    pos_offset: usize,
    gpu: &mut Option<&mut GpuBackend>,
    lora: Option<&LoraWeights>,
) -> Vec<Tensor> {
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return vec![];
    }

    let embed_dim = config.embedding_length as usize;
    let n_heads = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;
    let kv_group = n_heads / n_kv_heads;
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scale = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim = config
        .partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);
    let scale = config.query_pre_attn_scalar.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Use sequential path when GPU is active (GPU parallelism replaces rayon)
    // or when rayon is not available (e.g. wasm32 builds).
    #[cfg(feature = "rayon")]
    let use_sequential = gpu.is_some();
    #[cfg(not(feature = "rayon"))]
    let use_sequential = true;

    // 1. Embed all tokens
    let mut hidden: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| {
            weights
                .token_embedding
                .row_as_f32(id as usize)
                .data()
                .to_vec()
        })
        .collect();

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let effective_lora = lora.or(weights.lora.as_ref());
        let lora_layer = effective_lora.and_then(|l| l.layers.get(layer_idx));

        // a. Q/K/V projections + RoPE
        // When GPU is active, run sequentially (GPU parallelism replaces rayon).
        let qkv: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = if use_sequential {
            hidden
                .iter()
                .enumerate()
                .map(|(lp, h)| {
                    let abs = pos_offset + lp;
                    let h_t = Tensor::from_vec(h.clone(), &[embed_dim]);
                    #[cfg(feature = "vulkan")]
                    let normed = rms_norm_maybe_gpu(
                        &h_t,
                        &layer.attn_norm,
                        config.rms_norm_eps,
                        gpu.as_deref(),
                    );
                    #[cfg(not(feature = "vulkan"))]
                    let normed = tensor::rms_norm(&h_t, &layer.attn_norm, config.rms_norm_eps);
                    let mut q = matvec_maybe_gpu(&layer.attn_q, normed.data(), reborrow(gpu));
                    let mut k = matvec_maybe_gpu(&layer.attn_k, normed.data(), reborrow(gpu));
                    let mut v = matvec_maybe_gpu(&layer.attn_v, normed.data(), reborrow(gpu));
                    if let Some(bias) = &layer.attn_q_bias {
                        tensor::add_in_place(q.data_mut(), bias.data());
                    }
                    if let Some(bias) = &layer.attn_k_bias {
                        tensor::add_in_place(k.data_mut(), bias.data());
                    }
                    if let Some(bias) = &layer.attn_v_bias {
                        tensor::add_in_place(v.data_mut(), bias.data());
                    }
                    if let Some(ll) = lora_layer {
                        if let Some(a) = &ll.attn_q {
                            a.apply(normed.data(), q.data_mut());
                        }
                        if let Some(a) = &ll.attn_k {
                            a.apply(normed.data(), k.data_mut());
                        }
                        if let Some(a) = &ll.attn_v {
                            a.apply(normed.data(), v.data_mut());
                        }
                    }
                    let q = tensor::rope(&q, abs, head_dim, freq_base, rope_scale, rot_dim);
                    let k = tensor::rope(&k, abs, head_dim, freq_base, rope_scale, rot_dim);
                    (q.data().to_vec(), k.data().to_vec(), v.data().to_vec())
                })
                .collect()
        } else {
            #[cfg(feature = "rayon")]
            {
                hidden
                    .par_iter()
                    .enumerate()
                    .map(|(lp, h)| {
                        let abs = pos_offset + lp;
                        let h_t = Tensor::from_vec(h.clone(), &[embed_dim]);
                        let normed = tensor::rms_norm(&h_t, &layer.attn_norm, config.rms_norm_eps);
                        let mut q = layer.attn_q.matvec(normed.data());
                        let mut k = layer.attn_k.matvec(normed.data());
                        let mut v = layer.attn_v.matvec(normed.data());
                        if let Some(bias) = &layer.attn_q_bias {
                            tensor::add_in_place(q.data_mut(), bias.data());
                        }
                        if let Some(bias) = &layer.attn_k_bias {
                            tensor::add_in_place(k.data_mut(), bias.data());
                        }
                        if let Some(bias) = &layer.attn_v_bias {
                            tensor::add_in_place(v.data_mut(), bias.data());
                        }
                        if let Some(ll) = lora_layer {
                            if let Some(a) = &ll.attn_q {
                                a.apply(normed.data(), q.data_mut());
                            }
                            if let Some(a) = &ll.attn_k {
                                a.apply(normed.data(), k.data_mut());
                            }
                            if let Some(a) = &ll.attn_v {
                                a.apply(normed.data(), v.data_mut());
                            }
                        }
                        let q = tensor::rope(&q, abs, head_dim, freq_base, rope_scale, rot_dim);
                        let k = tensor::rope(&k, abs, head_dim, freq_base, rope_scale, rot_dim);
                        (q.data().to_vec(), k.data().to_vec(), v.data().to_vec())
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            unreachable!()
        };

        // b. Write all K/V (sequential — must finish before reads below)
        for (lp, kv) in qkv.iter().enumerate() {
            cache.write(layer_idx, pos_offset + lp, &kv.1, &kv.2);
        }

        // c. Attention + output projection
        let cache_ro: &dyn KvStore = &*cache;
        let attn_outs: Vec<Vec<f32>> = if use_sequential {
            (0..seq_len)
                .map(|lp| {
                    let abs = pos_offset + lp;
                    let window = if config.sliding_window_alternating && !layer_idx.is_multiple_of(2) {
                        0
                    } else {
                        config
                            .sliding_window
                            .map(|w| (abs as i64 - w as i64 + 1).max(0) as usize)
                            .unwrap_or(0)
                    };
                    let attend_len = abs + 1 - window;
                    let mut out = vec![0.0f32; embed_dim];
                    for h in 0..n_heads {
                        let kv_h = h / kv_group;
                        let q_off = h * head_dim;
                        tensor::flash_attn_1d_ext(
                            &qkv[lp].0[q_off..q_off + head_dim],
                            cache_ro,
                            layer_idx,
                            kv_h,
                            window,
                            attend_len,
                            head_dim,
                            scale,
                            config.attn_logit_softcapping,
                            &mut out[q_off..q_off + head_dim],
                        );
                    }
                    let attn_vec = Tensor::from_vec(out, &[embed_dim]);
                    let mut proj =
                        matvec_maybe_gpu(&layer.attn_output, attn_vec.data(), reborrow(gpu));
                    if let Some(norm) = &layer.post_attn_norm {
                        proj = tensor::rms_norm(&proj, norm, config.rms_norm_eps);
                    }
                    if let Some(ll) = lora_layer {
                        if let Some(a) = &ll.attn_output {
                            a.apply(attn_vec.data(), proj.data_mut());
                        }
                    }
                    proj.data().to_vec()
                })
                .collect()
        } else {
            #[cfg(feature = "rayon")]
            {
                (0..seq_len)
                    .into_par_iter()
                    .map(|lp| {
                        let abs = pos_offset + lp;
                        let window = if config.sliding_window_alternating && !layer_idx.is_multiple_of(2) {
                            0
                        } else {
                            config
                                .sliding_window
                                .map(|w| (abs as i64 - w as i64 + 1).max(0) as usize)
                                .unwrap_or(0)
                        };
                        let attend_len = abs + 1 - window;
                        let mut out = vec![0.0f32; embed_dim];
                        for h in 0..n_heads {
                            let kv_h = h / kv_group;
                            let q_off = h * head_dim;
                            tensor::flash_attn_1d_ext(
                                &qkv[lp].0[q_off..q_off + head_dim],
                                cache_ro,
                                layer_idx,
                                kv_h,
                                window,
                                attend_len,
                                head_dim,
                                scale,
                                config.attn_logit_softcapping,
                                &mut out[q_off..q_off + head_dim],
                            );
                        }
                        let attn_vec = Tensor::from_vec(out, &[embed_dim]);
                        let mut proj = layer.attn_output.matvec(attn_vec.data());
                        if let Some(norm) = &layer.post_attn_norm {
                            proj = tensor::rms_norm(&proj, norm, config.rms_norm_eps);
                        }
                        if let Some(ll) = lora_layer {
                            if let Some(a) = &ll.attn_output {
                                a.apply(attn_vec.data(), proj.data_mut());
                            }
                        }
                        proj.data().to_vec()
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            unreachable!()
        };

        // d. Residual + FFN
        if use_sequential {
            hidden = hidden
                .into_iter()
                .zip(attn_outs)
                .map(|(h, ao)| {
                    let h_t = Tensor::from_vec(h, &[embed_dim]);
                    let a_t = Tensor::from_vec(ao, &[embed_dim]);
                    #[cfg(feature = "vulkan")]
                    let after_attn = add_maybe_gpu(&h_t, &a_t, gpu.as_deref());
                    #[cfg(not(feature = "vulkan"))]
                    let after_attn = tensor::add(&h_t, &a_t);
                    #[cfg(feature = "vulkan")]
                    let nf = rms_norm_maybe_gpu(
                        &after_attn,
                        &layer.ffn_norm,
                        config.rms_norm_eps,
                        gpu.as_deref(),
                    );
                    #[cfg(not(feature = "vulkan"))]
                    let nf = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
                    let mut ffn_out = feed_forward(&nf, layer, lora_layer, gpu);
                    if let Some(norm) = &layer.post_ffn_norm {
                        ffn_out = tensor::rms_norm(&ffn_out, norm, config.rms_norm_eps);
                    }
                    #[cfg(feature = "vulkan")]
                    {
                        add_maybe_gpu(&after_attn, &ffn_out, gpu.as_deref())
                            .data()
                            .to_vec()
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        tensor::add(&after_attn, &ffn_out).data().to_vec()
                    }
                })
                .collect();
        } else {
            #[cfg(feature = "rayon")]
            {
                hidden = hidden
                    .into_par_iter()
                    .zip(attn_outs.into_par_iter())
                    .map(|(h, ao)| {
                        let h_t = Tensor::from_vec(h, &[embed_dim]);
                        let a_t = Tensor::from_vec(ao, &[embed_dim]);
                        let after_attn = tensor::add(&h_t, &a_t);
                        let nf =
                            tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
                        let mut ffn_out = feed_forward(&nf, layer, lora_layer, &mut None);
                        if let Some(norm) = &layer.post_ffn_norm {
                            ffn_out = tensor::rms_norm(&ffn_out, norm, config.rms_norm_eps);
                        }
                        tensor::add(&after_attn, &ffn_out).data().to_vec()
                    })
                    .collect();
            }
            #[cfg(not(feature = "rayon"))]
            unreachable!();
        }
    }

    // Advance cache seq_len positions
    for _ in 0..seq_len {
        cache.advance();
    }

    // 3. Final norm + LM head for every position
    hidden
        .iter()
        .map(|h| {
            let h_t = Tensor::from_vec(h.clone(), &[embed_dim]);
            #[cfg(feature = "vulkan")]
            let normed = rms_norm_maybe_gpu(
                &h_t,
                &weights.output_norm,
                config.rms_norm_eps,
                gpu.as_deref(),
            );
            #[cfg(not(feature = "vulkan"))]
            let normed = tensor::rms_norm(&h_t, &weights.output_norm, config.rms_norm_eps);
            let mut logits = matvec_maybe_gpu(&weights.output, normed.data(), reborrow(gpu));
            if let Some(cap) = config.final_logit_softcapping {
                tensor::logit_softcap_in_place(logits.data_mut(), cap);
            }
            logits
        })
        .collect()
}

// ── Generation functions ──────────────────────────────────────────────────────

/// Greedy decoding with KV-cache (batched prefill + token-by-token decode).
pub fn generate_greedy_cached(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize,
        config.context_length as usize,
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0, gpu);
    for step in 0..max_new_tokens {
        let next = last_logits
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        eprintln!("  Decode {}/{}: token {next}", step + 1, max_new_tokens);
        tokens.push(next);
        if next == 2 {
            break;
        }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, &mut cache, gpu);
    }
    tokens
}

/// Sampler-based generation with f32 KV-cache (batched prefill).
pub fn generate_cached(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize,
        config.context_length as usize,
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    generate_with_cache(
        weights,
        config,
        prompt_tokens,
        max_new_tokens,
        sampler,
        eos_token,
        &mut cache,
        gpu,
    )
}

/// Sampler-based generation with a Q8_0-compressed KV-cache (~3.8× less RAM).
///
/// Drop-in replacement for `generate_cached`. Identical output quality for
/// most prompts; reduces KV-cache memory from ~4 bytes/element to ~1.1 bytes.
pub fn generate_cached_q8(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let mut cache = KvCacheQ8::new(
        config.block_count as usize,
        config.context_length as usize,
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    generate_with_cache(
        weights,
        config,
        prompt_tokens,
        max_new_tokens,
        sampler,
        eos_token,
        &mut cache,
        gpu,
    )
}

/// Sampler-based generation with a paged f32 KV-cache.
///
/// Drop-in replacement for [`generate_cached`] with bit-identical output: the
/// cache holds the same f32 rows, just in [`PAGE_SIZE`]-token pages taken from
/// a private pool as the sequence grows, instead of one `context_length`
/// allocation made up front.
///
/// Server-side callers should build one [`PagePool`] shared by every sequence
/// (see `EngineLimits::kv_pool_pages`) — that is where paging pays off.
pub fn generate_cached_paged(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let n_layers = config.block_count as usize;
    // Private pool sized for this one sequence's worst case, so page
    // allocation can never fail part-way through the generation.
    let pool = PagePool::new(
        PagePool::pages_for(config.context_length as usize, n_layers),
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    let mut cache = PagedKvCache::new(&pool, n_layers);
    generate_with_cache(
        weights,
        config,
        prompt_tokens,
        max_new_tokens,
        sampler,
        eos_token,
        &mut cache,
        gpu,
    )
}

fn generate_with_cache(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    cache: &mut dyn KvStore,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, cache, 0, gpu);
    for step in 0..max_new_tokens {
        let next = sampler.sample(last_logits.data(), &tokens);
        eprintln!("  Decode {}/{}: token {next}", step + 1, max_new_tokens);
        tokens.push(next);
        if next == eos_token {
            break;
        }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, cache, gpu);
    }
    tokens
}

/// Streaming generation with per-token callback.
///
/// Like `generate_cached` but calls `on_token(id)` for each generated token.
/// Return `false` from the callback to stop early (e.g., client disconnect).
pub fn generate_streaming(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    on_token: impl Fn(u32) -> bool,
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize,
        config.context_length as usize,
        config.head_count_kv as usize,
        config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0, gpu);
    for _ in 0..max_new_tokens {
        let next = sampler.sample(last_logits.data(), &tokens);
        tokens.push(next);
        if !on_token(next) || next == eos_token {
            break;
        }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, &mut cache, gpu);
    }
    tokens
}

// ── Batched decode (continuous batching) ─────────────────────────────────────

/// Interleaved `[rows, B]` kernel output → one owned `Tensor` per sequence.
///
/// `scratch` is reused across every matvec of a step, so a decode step
/// allocates its per-sequence activations and nothing else.
fn batch_matvec(
    weight: &QuantizedTensor,
    inputs: &[&[f32]],
    scratch: &mut Vec<f32>,
) -> Vec<Tensor> {
    let batch = inputs.len();
    let rows = weight.rows();
    scratch.clear();
    scratch.resize(rows * batch, 0.0);
    weight.matvec_batch_into(inputs, scratch);
    (0..batch)
        .map(|s| {
            let mut column = vec![0.0f32; rows];
            for (i, o) in column.iter_mut().enumerate() {
                *o = scratch[i * batch + s];
            }
            Tensor::from_vec(column, &[rows])
        })
        .collect()
}

/// Borrow every sequence's activation vector, in batch order.
fn data_refs(tensors: &[Tensor]) -> Vec<&[f32]> {
    tensors.iter().map(|t| t.data()).collect()
}

/// Decode one token for several sequences in a **single** forward pass.
///
/// Each element `i` provides:
/// - `tokens[i]`    — the token to decode
/// - `positions[i]` — its absolute position (0-based from the prompt start)
/// - `caches[i]`    — its own exclusive KV cache
///
/// Returns one logits [`Tensor`] per sequence, in the same order.
///
/// # Why this exists
///
/// Decoding is memory-bound: a step reads every weight matrix once and does
/// only one multiply-add per weight. Running `B` sequences as `B` separate
/// `forward_one` calls therefore streams the whole model from RAM `B` times to
/// do `B` multiply-adds per weight. This function streams it **once** and does
/// all `B` — the per-layer matvecs go through
/// [`QuantizedTensor::matvec_batch_into`], which applies each decoded weight
/// block to every sequence before moving on. That is the throughput multiplier
/// behind continuous batching in the server engine.
///
/// Attention stays per-sequence: each sequence has its own KV cache and its own
/// position, so there is nothing to share — those are run in parallel across
/// sequences instead (rayon), and the matvecs keep their usual row-parallelism.
///
/// # Batching is invisible
///
/// The logits returned here are **bit-identical** to what [`forward_one`] would
/// produce for the same sequence on its own: the batched kernels preserve each
/// sequence's accumulation order, and everything outside them (norms, RoPE,
/// attention, residuals) is per-sequence code shared with the single-sequence
/// path. A request must never get different tokens because the server happened
/// to be busy. `forward_batch_matches_forward_one` pins this.
///
/// A batch of one, or an active GPU backend (a single device context, so
/// nothing to interleave), falls through to `forward_one` unchanged — including
/// its allocation profile.
pub fn forward_batch(
    weights: &TransformerWeights,
    config: &ModelConfig,
    tokens: &[u32],
    positions: &[usize],
    caches: &mut [&mut dyn KvStore],
    gpu: &mut Option<&mut GpuBackend>,
) -> Vec<Tensor> {
    forward_batch_lora(weights, config, tokens, positions, caches, gpu, &[])
}

/// Like [`forward_batch`] but with a per-sequence LoRA override.
///
/// `loras` is either empty (every sequence uses the model's built-in adapter,
/// if any) or one entry per sequence — sequences carrying *different* adapters
/// can share a step, since the base weights are what the batch amortises and
/// each adapter's low-rank update is applied to its own sequence afterwards.
pub fn forward_batch_lora(
    weights: &TransformerWeights,
    config: &ModelConfig,
    tokens: &[u32],
    positions: &[usize],
    caches: &mut [&mut dyn KvStore],
    gpu: &mut Option<&mut GpuBackend>,
    loras: &[Option<&LoraWeights>],
) -> Vec<Tensor> {
    let n_seqs = tokens.len();
    assert_eq!(
        positions.len(),
        n_seqs,
        "forward_batch: positions/tokens len"
    );
    assert_eq!(caches.len(), n_seqs, "forward_batch: caches/tokens len");
    assert!(
        loras.is_empty() || loras.len() == n_seqs,
        "forward_batch: loras must be empty or one per sequence"
    );
    if n_seqs == 0 {
        return vec![];
    }
    let lora_for = |s: usize| loras.get(s).copied().flatten();

    // Nothing to amortise across (B = 1), or a GPU backend that owns a single
    // context: run the single-sequence path exactly as before.
    if n_seqs == 1 || gpu.is_some() {
        let mut logits = Vec::with_capacity(n_seqs);
        for (s, cache) in caches.iter_mut().enumerate() {
            logits.push(forward_one_lora(
                weights,
                config,
                tokens[s],
                positions[s],
                &mut **cache,
                gpu,
                lora_for(s),
            ));
        }
        return logits;
    }

    // ── CPU batch path ─────────────────────────────────────────────────────

    let embed_dim = config.embedding_length as usize;
    let n_heads = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;
    let kv_group_size = n_heads / n_kv_heads;
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scaling = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim = config
        .partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);
    let scale = config.query_pre_attn_scalar.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Interleaved matvec output, reused by every batched matvec in the step.
    let mut scratch: Vec<f32> = Vec::new();
    let mut hidden: Vec<Tensor> = tokens
        .iter()
        .map(|&t| weights.token_embedding.row_as_f32(t as usize))
        .collect();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let lora_layers: Vec<Option<&LoraLayerAdapters>> = (0..n_seqs)
            .map(|s| {
                lora_for(s)
                    .or(weights.lora.as_ref())
                    .and_then(|l| l.layers.get(layer_idx))
            })
            .collect();

        // ── Attention ──────────────────────────────────────────────────────
        let normed: Vec<Tensor> = hidden
            .iter()
            .map(|h| tensor::rms_norm(h, &layer.attn_norm, config.rms_norm_eps))
            .collect();
        let inputs = data_refs(&normed);
        let mut q = batch_matvec(&layer.attn_q, &inputs, &mut scratch);
        let mut k = batch_matvec(&layer.attn_k, &inputs, &mut scratch);
        let mut v = batch_matvec(&layer.attn_v, &inputs, &mut scratch);
        if let Some(bias) = &layer.attn_q_bias {
            for q_s in q.iter_mut() {
                tensor::add_in_place(q_s.data_mut(), bias.data());
            }
        }
        if let Some(bias) = &layer.attn_k_bias {
            for k_s in k.iter_mut() {
                tensor::add_in_place(k_s.data_mut(), bias.data());
            }
        }
        if let Some(bias) = &layer.attn_v_bias {
            for v_s in v.iter_mut() {
                tensor::add_in_place(v_s.data_mut(), bias.data());
            }
        }
        for (s, ll) in lora_layers.iter().enumerate() {
            if let Some(ll) = ll {
                if let Some(a) = &ll.attn_q {
                    a.apply(normed[s].data(), q[s].data_mut());
                }
                if let Some(a) = &ll.attn_k {
                    a.apply(normed[s].data(), k[s].data_mut());
                }
                if let Some(a) = &ll.attn_v {
                    a.apply(normed[s].data(), v[s].data_mut());
                }
            }
        }

        // RoPE, then publish this position's K/V into each sequence's cache.
        for s in 0..n_seqs {
            let pos = positions[s];
            q[s] = tensor::rope(&q[s], pos, head_dim, freq_base, rope_scaling, rot_dim);
            let k_roped = tensor::rope(&k[s], pos, head_dim, freq_base, rope_scaling, rot_dim);
            caches[s].write(layer_idx, pos, k_roped.data(), v[s].data());
        }

        // Attention itself is per-sequence — independent caches and positions,
        // so the sequences run in parallel rather than sharing a traversal.
        let attn_vecs: Vec<Tensor> = {
            let read_only: Vec<&dyn KvStore> = caches.iter().map(|c| &**c).collect();
            let attend = |s: usize| {
                let pos = positions[s];
                let window_start = if config.sliding_window_alternating && !layer_idx.is_multiple_of(2) {
                    0
                } else {
                    config
                        .sliding_window
                        .map(|w| (pos as i64 - w as i64 + 1).max(0) as usize)
                        .unwrap_or(0)
                };
                let mut out = vec![0.0f32; embed_dim];
                attn_heads_cpu(
                    q[s].data(),
                    read_only[s],
                    layer_idx,
                    window_start,
                    pos + 1 - window_start,
                    n_heads,
                    kv_group_size,
                    head_dim,
                    scale,
                    config.attn_logit_softcapping,
                    &mut out,
                );
                Tensor::from_vec(out, &[embed_dim])
            };
            #[cfg(feature = "rayon")]
            {
                (0..n_seqs).into_par_iter().map(attend).collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..n_seqs).map(attend).collect()
            }
        };

        let attn_inputs = data_refs(&attn_vecs);
        let mut proj = batch_matvec(&layer.attn_output, &attn_inputs, &mut scratch);
        if let Some(norm) = &layer.post_attn_norm {
            for p in proj.iter_mut() {
                *p = tensor::rms_norm(p, norm, config.rms_norm_eps);
            }
        }
        for (s, ll) in lora_layers.iter().enumerate() {
            if let Some(a) = ll.and_then(|l| l.attn_output.as_ref()) {
                a.apply(attn_vecs[s].data(), proj[s].data_mut());
            }
        }
        let after_attn: Vec<Tensor> = hidden
            .iter()
            .zip(&proj)
            .map(|(h, p)| tensor::add(h, p))
            .collect();

        // ── Feed-forward ───────────────────────────────────────────────────
        let normed_ffn: Vec<Tensor> = after_attn
            .iter()
            .map(|h| tensor::rms_norm(h, &layer.ffn_norm, config.rms_norm_eps))
            .collect();
        let ffn_inputs = data_refs(&normed_ffn);
        let mut gate = batch_matvec(&layer.ffn_gate, &ffn_inputs, &mut scratch);
        let mut up = batch_matvec(&layer.ffn_up, &ffn_inputs, &mut scratch);
        for (s, ll) in lora_layers.iter().enumerate() {
            if let Some(ll) = ll {
                if let Some(a) = &ll.ffn_gate {
                    a.apply(normed_ffn[s].data(), gate[s].data_mut());
                }
                if let Some(a) = &ll.ffn_up {
                    a.apply(normed_ffn[s].data(), up[s].data_mut());
                }
            }
        }
        let activated: Vec<Tensor> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| tensor::mul(&tensor::silu(g), u))
            .collect();
        let down_inputs = data_refs(&activated);
        let mut down = batch_matvec(&layer.ffn_down, &down_inputs, &mut scratch);
        if let Some(norm) = &layer.post_ffn_norm {
            for d in down.iter_mut() {
                *d = tensor::rms_norm(d, norm, config.rms_norm_eps);
            }
        }
        for (s, ll) in lora_layers.iter().enumerate() {
            if let Some(a) = ll.and_then(|l| l.ffn_down.as_ref()) {
                a.apply(activated[s].data(), down[s].data_mut());
            }
        }
        hidden = after_attn
            .iter()
            .zip(&down)
            .map(|(h, d)| tensor::add(h, d))
            .collect();
    }

    for cache in caches.iter_mut() {
        cache.advance();
    }

    // Final norm + LM head. The LM head is the widest matrix in the model, so
    // it is also where batching saves the most.
    let normed: Vec<Tensor> = hidden
        .iter()
        .map(|h| tensor::rms_norm(h, &weights.output_norm, config.rms_norm_eps))
        .collect();
    let inputs = data_refs(&normed);
    let mut logits = batch_matvec(&weights.output, &inputs, &mut scratch);
    if let Some(cap) = config.final_logit_softcapping {
        for logit in logits.iter_mut() {
            tensor::logit_softcap_in_place(logit.data_mut(), cap);
        }
    }
    logits
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Build a minimal one-layer model with deterministic weights.
///
/// Shared test fixture: exercises the full forward pass (GQA with 2 query /
/// 1 KV head, SwiGLU, RoPE) at a size where tests run in microseconds. Also
/// used by the server engine's end-to-end tests.
#[cfg(test)]
pub(crate) fn make_tiny_weights() -> (TransformerWeights, ModelConfig) {
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
        head_dim_override: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        query_pre_attn_scalar: None,
        sliding_window_alternating: false,
    };
    let weights = TransformerWeights {
        token_embedding: QuantizedTensor::from_f32(
            &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            8,
            4,
        ),
        layers: vec![LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            ffn_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            post_attn_norm: None,
            post_ffn_norm: None,
            attn_q: QuantizedTensor::from_f32(
                &(0..16).map(|i| i as f32 * 0.05 - 0.4).collect::<Vec<_>>(),
                4,
                4,
            ),
            attn_k: QuantizedTensor::from_f32(
                &(0..8).map(|i| i as f32 * 0.1 - 0.3).collect::<Vec<_>>(),
                2,
                4,
            ),
            attn_v: QuantizedTensor::from_f32(
                &(0..8).map(|i| i as f32 * 0.07 - 0.2).collect::<Vec<_>>(),
                2,
                4,
            ),
            attn_output: QuantizedTensor::from_f32(
                &(0..16).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>(),
                4,
                4,
            ),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            ffn_gate: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(),
                8,
                4,
            ),
            ffn_up: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.015 - 0.2).collect::<Vec<_>>(),
                8,
                4,
            ),
            ffn_down: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.025 - 0.4).collect::<Vec<_>>(),
                4,
                8,
            ),
        }],
        output_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
        output: QuantizedTensor::from_f32(
            &(0..32).map(|i| i as f32 * 0.1 - 1.6).collect::<Vec<_>>(),
            8,
            4,
        ),
        lora: None,
    };
    (weights, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::tensor::QuantizedTensor;

    #[test]
    fn test_cached_matches_uncached() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];
        let logits_uncached = forward(&weights, &config, &token_ids);
        let mut cache = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let mut logits_cached = Tensor::zeros(&[1]);
        for (pos, &tok) in token_ids.iter().enumerate() {
            logits_cached = forward_one(&weights, &config, tok, pos, &mut cache, &mut None);
        }
        assert_eq!(logits_uncached.shape(), logits_cached.shape());
        for (i, (&a, &b)) in logits_uncached
            .data()
            .iter()
            .zip(logits_cached.data())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-4,
                "index {i}: uncached={a:.6}, cached={b:.6}"
            );
        }
    }

    #[test]
    fn test_prefill_matches_sequential() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];

        // Sequential via forward_one
        let mut cache_seq = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let mut logits_seq = Tensor::zeros(&[1]);
        for (pos, &tok) in token_ids.iter().enumerate() {
            logits_seq = forward_one(&weights, &config, tok, pos, &mut cache_seq, &mut None);
        }

        // Batched prefill
        let mut cache_batch = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let logits_batch = forward_prefill(
            &weights,
            &config,
            &token_ids,
            &mut cache_batch,
            0,
            &mut None,
        );

        for (i, (&a, &b)) in logits_seq
            .data()
            .iter()
            .zip(logits_batch.data())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-4,
                "index {i}: sequential={a:.6}, prefill={b:.6}"
            );
        }
    }

    #[test]
    fn test_prefill_q8_close_to_f32() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];

        let mut c_f32 = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let l_f32 = forward_prefill(&weights, &config, &token_ids, &mut c_f32, 0, &mut None);

        let mut c_q8 = KvCacheQ8::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let l_q8 = forward_prefill(&weights, &config, &token_ids, &mut c_q8, 0, &mut None);

        for (i, (&a, &b)) in l_f32.data().iter().zip(l_q8.data()).enumerate() {
            assert!((a - b).abs() < 0.05, "index {i}: f32={a:.4}, q8={b:.4}");
        }
    }

    /// End-to-end check that the paged cache is invisible to the forward pass:
    /// a batched prefill followed by decode steps must produce exactly the same
    /// logits as the contiguous f32 cache, bit for bit.
    #[test]
    fn test_paged_cache_matches_kv_cache_end_to_end() {
        let (weights, config) = make_tiny_weights();
        let n_layers = config.block_count as usize;
        let prompt: Vec<u32> = vec![1, 3, 5, 2, 7];

        let mut plain = KvCache::new(
            n_layers,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        // A pool with room for the whole context, shared by nobody else.
        let pool = PagePool::new(
            PagePool::pages_for(config.context_length as usize, n_layers),
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let mut paged = PagedKvCache::new(&pool, n_layers);

        let mut l_plain = forward_prefill(&weights, &config, &prompt, &mut plain, 0, &mut None);
        let mut l_paged = forward_prefill(&weights, &config, &prompt, &mut paged, 0, &mut None);

        // Decode far enough to cross two page boundaries (PAGE_SIZE = 16).
        let mut tokens = prompt.clone();
        for step in 0..25 {
            for (i, (&a, &b)) in l_plain.data().iter().zip(l_paged.data()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {step} logit {i}: kv={a:.6} paged={b:.6}"
                );
            }
            let next = l_plain
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            tokens.push(next);
            let pos = tokens.len() - 1;
            l_plain = forward_one(&weights, &config, next, pos, &mut plain, &mut None);
            l_paged = forward_one(&weights, &config, next, pos, &mut paged, &mut None);
        }

        assert_eq!(plain.len(), paged.len());
        assert_eq!(paged.pages_per_layer(), tokens.len().div_ceil(16));
        assert_eq!(pool.live_pages(), paged.allocated_pages());
    }

    /// A sequence forked from a prefilled prompt shares that prompt's pages and
    /// still decodes to the same logits as one that computed the prefix itself
    /// — the property prefix caching will be built on.
    #[test]
    fn test_forked_cache_decodes_like_a_fresh_one() {
        let (weights, config) = make_tiny_weights();
        let n_layers = config.block_count as usize;
        let prompt: Vec<u32> = vec![1, 3, 5, 2, 7, 4];

        let pool = PagePool::new(
            64,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let mut base = PagedKvCache::new(&pool, n_layers);
        forward_prefill(&weights, &config, &prompt, &mut base, 0, &mut None);

        // Fork the whole prompt instead of prefilling it again.
        let mut forked = base.fork_from(prompt.len());
        assert!(forked.is_page_shared(0, 0), "the prefix page is shared");

        let mut fresh = PagedKvCache::new(&pool, n_layers);
        forward_prefill(&weights, &config, &prompt, &mut fresh, 0, &mut None);

        let mut tokens = prompt.clone();
        for _ in 0..8 {
            let pos = tokens.len();
            let tok = (pos as u32) % config.vocab_size;
            let l_fresh = forward_one(&weights, &config, tok, pos, &mut fresh, &mut None);
            let l_forked = forward_one(&weights, &config, tok, pos, &mut forked, &mut None);
            for (i, (&a, &b)) in l_fresh.data().iter().zip(l_forked.data()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "pos {pos} logit {i}");
            }
            tokens.push(tok);
        }
        // Writing into the forked tail page copied it; the base is untouched.
        assert_eq!(base.len(), prompt.len());
        assert!(!base.is_page_shared(0, 0));
    }

    #[test]
    fn test_feed_forward_shape() {
        let layer = LayerWeights {
            attn_norm: Tensor::zeros(&[4]),
            ffn_norm: Tensor::zeros(&[4]),
            post_attn_norm: None,
            post_ffn_norm: None,
            attn_q: QuantizedTensor::from_f32(&[0.0f32; 16], 4, 4),
            attn_k: QuantizedTensor::from_f32(&[0.0f32; 16], 4, 4),
            attn_v: QuantizedTensor::from_f32(&[0.0f32; 16], 4, 4),
            attn_output: QuantizedTensor::from_f32(&[0.0f32; 16], 4, 4),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            ffn_gate: QuantizedTensor::from_f32(&[0.0f32; 32], 8, 4),
            ffn_up: QuantizedTensor::from_f32(&[0.0f32; 32], 8, 4),
            ffn_down: QuantizedTensor::from_f32(&[0.0f32; 32], 4, 8),
        };
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        assert_eq!(feed_forward(&x, &layer, None, &mut None).shape(), &[4]);
    }

    // ── Batched decode ───────────────────────────────────────────────────
    //
    // Continuous batching is only acceptable if it is invisible to a request:
    // whatever the server's load, a sequence must produce the same tokens it
    // would have produced alone. Because sampling is a function of the logits,
    // that reduces to bit-identical logits — asserted here on the f32 bit
    // patterns, not on a tolerance.

    /// A KV cache holding `prompt` plus the prompt's final logits — ready for
    /// the next decode step at position `prompt.len()`.
    fn prefill_for_test(
        weights: &TransformerWeights,
        config: &ModelConfig,
        prompt: &[u32],
    ) -> (KvCache, Tensor) {
        let mut cache = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let logits = forward_prefill(weights, config, prompt, &mut cache, 0, &mut None);
        (cache, logits)
    }

    fn prefilled_cache(
        weights: &TransformerWeights,
        config: &ModelConfig,
        prompt: &[u32],
    ) -> KvCache {
        prefill_for_test(weights, config, prompt).0
    }

    fn assert_same_logits(expected: &Tensor, got: &Tensor, what: &str) {
        assert_eq!(expected.shape(), got.shape(), "{what}: shape");
        for (i, (&a, &b)) in expected.data().iter().zip(got.data()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: logit {i} single={a} batched={b}"
            );
        }
    }

    /// Sequences with different prompt lengths (so different positions and
    /// different cached histories) share one batched step; every sequence must
    /// get exactly the logits `forward_one` gives it on its own.
    #[test]
    fn forward_batch_matches_forward_one() {
        let (weights, config) = make_tiny_weights();
        let prompts: [&[u32]; 4] = [&[1, 2, 3], &[4], &[5, 0, 6, 7], &[2, 2]];
        let next: [u32; 4] = [3, 6, 1, 7];

        for batch in [1usize, 2, 4] {
            let expected: Vec<Tensor> = (0..batch)
                .map(|s| {
                    let mut cache = prefilled_cache(&weights, &config, prompts[s]);
                    forward_one(
                        &weights,
                        &config,
                        next[s],
                        prompts[s].len(),
                        &mut cache,
                        &mut None,
                    )
                })
                .collect();

            let mut caches: Vec<KvCache> = (0..batch)
                .map(|s| prefilled_cache(&weights, &config, prompts[s]))
                .collect();
            let mut cache_refs: Vec<&mut dyn KvStore> =
                caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
            let tokens: Vec<u32> = next[..batch].to_vec();
            let positions: Vec<usize> = (0..batch).map(|s| prompts[s].len()).collect();
            let got = forward_batch(
                &weights,
                &config,
                &tokens,
                &positions,
                &mut cache_refs,
                &mut None,
            );

            assert_eq!(got.len(), batch);
            for s in 0..batch {
                assert_same_logits(&expected[s], &got[s], &format!("batch={batch} seq={s}"));
            }
        }
    }

    /// The batched step must also leave each cache in the state the
    /// single-sequence step would have — otherwise the *next* step diverges
    /// even though this one matched.
    #[test]
    fn forward_batch_writes_the_same_cache_state() {
        let (weights, config) = make_tiny_weights();
        let prompts: [&[u32]; 2] = [&[1, 2, 3], &[5, 0]];
        let next: [u32; 2] = [4, 6];

        let solo: Vec<KvCache> = (0..2)
            .map(|s| {
                let mut cache = prefilled_cache(&weights, &config, prompts[s]);
                forward_one(
                    &weights,
                    &config,
                    next[s],
                    prompts[s].len(),
                    &mut cache,
                    &mut None,
                );
                cache
            })
            .collect();

        let mut batched: Vec<KvCache> = (0..2)
            .map(|s| prefilled_cache(&weights, &config, prompts[s]))
            .collect();
        let mut cache_refs: Vec<&mut dyn KvStore> =
            batched.iter_mut().map(|c| c as &mut dyn KvStore).collect();
        let positions: Vec<usize> = (0..2).map(|s| prompts[s].len()).collect();
        forward_batch(
            &weights,
            &config,
            &next,
            &positions,
            &mut cache_refs,
            &mut None,
        );

        for s in 0..2 {
            assert_eq!(solo[s].len(), batched[s].len(), "seq {s} cache length");
            for pos in 0..solo[s].len() {
                assert_eq!(
                    solo[s].k_at(0, pos),
                    batched[s].k_at(0, pos),
                    "seq {s} K@{pos}"
                );
                assert_eq!(
                    solo[s].v_at(0, pos),
                    batched[s].v_at(0, pos),
                    "seq {s} V@{pos}"
                );
            }
        }
    }

    /// Parity has to hold *step after step*, not just for the first token:
    /// a batch decodes for many steps, and each one reads the cache the
    /// previous one wrote.
    #[test]
    fn forward_batch_stays_identical_over_many_steps() {
        let (weights, config) = make_tiny_weights();
        let prompts: [&[u32]; 3] = [&[1, 2], &[3, 4, 5], &[6]];
        let steps = 6;

        let argmax = |t: &Tensor| {
            t.data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap()
        };

        // Reference: each sequence decoded entirely on its own.
        let solo: Vec<Vec<u32>> = (0..3)
            .map(|s| {
                let (mut cache, mut logits) = prefill_for_test(&weights, &config, prompts[s]);
                let mut produced = Vec::new();
                for pos in (prompts[s].len()..).take(steps) {
                    let tok = argmax(&logits);
                    produced.push(tok);
                    logits = forward_one(&weights, &config, tok, pos, &mut cache, &mut None);
                }
                produced
            })
            .collect();

        // Batched: all three sequences share every step.
        let mut caches: Vec<KvCache> = Vec::new();
        let mut logits: Vec<Tensor> = Vec::new();
        for prompt in prompts {
            let (cache, first) = prefill_for_test(&weights, &config, prompt);
            caches.push(cache);
            logits.push(first);
        }
        let mut positions: Vec<usize> = (0..3).map(|s| prompts[s].len()).collect();
        let mut produced: Vec<Vec<u32>> = vec![Vec::new(); 3];

        for _ in 0..steps {
            let tokens: Vec<u32> = logits.iter().map(argmax).collect();
            for (s, &tok) in tokens.iter().enumerate() {
                produced[s].push(tok);
            }
            let mut cache_refs: Vec<&mut dyn KvStore> =
                caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
            logits = forward_batch(
                &weights,
                &config,
                &tokens,
                &positions,
                &mut cache_refs,
                &mut None,
            );
            for p in positions.iter_mut() {
                *p += 1;
            }
        }

        assert_eq!(
            produced, solo,
            "batched decoding diverged from solo decoding"
        );
    }

    /// Quantize `rows × cols` samples of `f` into a Q8_0 tensor.
    ///
    /// The tiny fixture above is F32, which takes the dequantize-then-dot
    /// fallback. Real models are quantized and hit the SIMD kernels, so the
    /// batched forward pass needs a fixture that does too.
    fn q8_0_tensor(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> QuantizedTensor {
        use crate::model::gguf::GgmlType;
        assert_eq!(cols % 32, 0);
        let mut data = Vec::with_capacity(rows * (cols / 32) * 34);
        for r in 0..rows {
            for b in 0..cols / 32 {
                let values: Vec<f32> = (0..32).map(|j| f(r, b * 32 + j)).collect();
                let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = half::f16::from_f32(if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 });
                data.extend_from_slice(&d.to_le_bytes());
                let d = d.to_f32();
                for v in values {
                    data.push(((v / d).round().clamp(-127.0, 127.0) as i8) as u8);
                }
            }
        }
        QuantizedTensor::from_raw(data, rows, cols, GgmlType::Q8_0)
    }

    /// One-layer model with Q8_0 weights: embed 32, 2 query heads over 1 KV
    /// head (GQA), head_dim 16, FFN 64, vocab 32.
    fn make_tiny_q8_weights() -> (TransformerWeights, ModelConfig) {
        let config = ModelConfig {
            architecture: "test-q8".to_string(),
            context_length: 32,
            embedding_length: 32,
            block_count: 1,
            head_count: 2,
            head_count_kv: 1,
            vocab_size: 32,
            feed_forward_length: Some(64),
            rms_norm_eps: 1e-5,
            rope_freq_base: Some(10000.0),
            chat_template: None,
            sliding_window: None,
            rope_scaling_factor: None,
            partial_rotary_factor: None,
            head_dim_override: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window_alternating: false,
        };
        let wave = |k: f32| move |r: usize, c: usize| ((r * 31 + c * 7) as f32 * 0.021).sin() * k;
        let weights = TransformerWeights {
            token_embedding: q8_0_tensor(32, 32, wave(0.7)),
            layers: vec![LayerWeights {
                attn_norm: Tensor::from_vec(
                    (0..32).map(|i| 1.0 + i as f32 * 0.01).collect(),
                    &[32],
                ),
                ffn_norm: Tensor::from_vec(
                    (0..32).map(|i| 1.0 - i as f32 * 0.005).collect(),
                    &[32],
                ),
                post_attn_norm: None,
                post_ffn_norm: None,
                attn_q: q8_0_tensor(32, 32, wave(0.4)),
                attn_k: q8_0_tensor(16, 32, wave(0.35)),
                attn_v: q8_0_tensor(16, 32, wave(0.5)),
                attn_output: q8_0_tensor(32, 32, wave(0.3)),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                ffn_gate: q8_0_tensor(64, 32, wave(0.25)),
                ffn_up: q8_0_tensor(64, 32, wave(0.45)),
                ffn_down: q8_0_tensor(32, 64, wave(0.2)),
            }],
            output_norm: Tensor::from_vec(vec![1.0; 32], &[32]),
            output: q8_0_tensor(32, 32, wave(0.6)),
            lora: None,
        };
        (weights, config)
    }

    /// Parity on a quantized model — this is the path a real deployment takes
    /// (and, on x86, the batched AVX2 kernels).
    #[test]
    fn forward_batch_matches_forward_one_quantized() {
        let (weights, config) = make_tiny_q8_weights();
        let prompts: [&[u32]; 4] = [&[1, 2, 3], &[9], &[5, 0, 6, 7], &[11, 11]];
        let next: [u32; 4] = [3, 6, 1, 7];

        for batch in [2usize, 4] {
            let expected: Vec<Tensor> = (0..batch)
                .map(|s| {
                    let mut cache = prefilled_cache(&weights, &config, prompts[s]);
                    forward_one(
                        &weights,
                        &config,
                        next[s],
                        prompts[s].len(),
                        &mut cache,
                        &mut None,
                    )
                })
                .collect();

            let mut caches: Vec<KvCache> = (0..batch)
                .map(|s| prefilled_cache(&weights, &config, prompts[s]))
                .collect();
            let mut cache_refs: Vec<&mut dyn KvStore> =
                caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
            let tokens: Vec<u32> = next[..batch].to_vec();
            let positions: Vec<usize> = (0..batch).map(|s| prompts[s].len()).collect();
            let got = forward_batch(
                &weights,
                &config,
                &tokens,
                &positions,
                &mut cache_refs,
                &mut None,
            );

            for s in 0..batch {
                assert_same_logits(&expected[s], &got[s], &format!("q8 batch={batch} seq={s}"));
            }
        }
    }

    /// The same parity, with the Q8_0-compressed KV cache rather than f32 —
    /// the batched step must read a quantized cache exactly as the single one
    /// does.
    #[test]
    fn forward_batch_matches_forward_one_q8_cache() {
        let (weights, config) = make_tiny_q8_weights();
        let prompts: [&[u32]; 2] = [&[1, 2, 3], &[8, 4]];
        let next: [u32; 2] = [5, 6];

        let new_cache = || {
            KvCacheQ8::new(
                config.block_count as usize,
                config.context_length as usize,
                config.head_count_kv as usize,
                config.head_dim() as usize,
            )
        };
        let expected: Vec<Tensor> = (0..2)
            .map(|s| {
                let mut cache = new_cache();
                forward_prefill(&weights, &config, prompts[s], &mut cache, 0, &mut None);
                forward_one(
                    &weights,
                    &config,
                    next[s],
                    prompts[s].len(),
                    &mut cache,
                    &mut None,
                )
            })
            .collect();

        let mut caches: Vec<KvCacheQ8> = (0..2)
            .map(|s| {
                let mut cache = new_cache();
                forward_prefill(&weights, &config, prompts[s], &mut cache, 0, &mut None);
                cache
            })
            .collect();
        let mut cache_refs: Vec<&mut dyn KvStore> =
            caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
        let positions: Vec<usize> = (0..2).map(|s| prompts[s].len()).collect();
        let got = forward_batch(
            &weights,
            &config,
            &next,
            &positions,
            &mut cache_refs,
            &mut None,
        );

        for s in 0..2 {
            assert_same_logits(&expected[s], &got[s], &format!("q8-cache seq={s}"));
        }
    }

    /// Sequences carrying different LoRA adapters can share a step: the base
    /// weights are what the batch amortises, and each adapter's update is
    /// applied to its own sequence afterwards.
    #[test]
    fn forward_batch_applies_per_sequence_lora() {
        use crate::model::lora::{LoraAdapter, LoraLayerAdapters, LoraWeights};

        let (weights, config) = make_tiny_weights();
        let embed = config.embedding_length as usize;
        // Rank-1 adapter on the output projection of the single layer.
        let make_lora = |k: f32| LoraWeights {
            layers: vec![LoraLayerAdapters {
                attn_output: Some(LoraAdapter {
                    a: Tensor::from_vec(vec![k; embed], &[1, embed]),
                    b: Tensor::from_vec(vec![0.5; embed], &[embed, 1]),
                    scale: 2.0,
                }),
                ..Default::default()
            }],
        };
        let lora_a = make_lora(0.25);
        let lora_b = make_lora(-0.5);
        let prompts: [&[u32]; 2] = [&[1, 2], &[3]];
        let next: [u32; 2] = [4, 5];
        let loras: [Option<&LoraWeights>; 2] = [Some(&lora_a), Some(&lora_b)];

        let expected: Vec<Tensor> = (0..2)
            .map(|s| {
                let mut cache = prefilled_cache(&weights, &config, prompts[s]);
                forward_one_lora(
                    &weights,
                    &config,
                    next[s],
                    prompts[s].len(),
                    &mut cache,
                    &mut None,
                    loras[s],
                )
            })
            .collect();

        let mut caches: Vec<KvCache> = (0..2)
            .map(|s| prefilled_cache(&weights, &config, prompts[s]))
            .collect();
        let mut cache_refs: Vec<&mut dyn KvStore> =
            caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
        let positions: Vec<usize> = (0..2).map(|s| prompts[s].len()).collect();
        let got = forward_batch_lora(
            &weights,
            &config,
            &next,
            &positions,
            &mut cache_refs,
            &mut None,
            &loras,
        );

        for s in 0..2 {
            assert_same_logits(&expected[s], &got[s], &format!("lora seq={s}"));
        }
        // Sanity: the two adapters really do produce different logits, so the
        // test above is not comparing a sequence against itself.
        assert_ne!(got[0].data()[0].to_bits(), got[1].data()[0].to_bits());
    }

    /// Verify `embed_batch([a, b])` produces the same results as `[embed(a), embed(b)]`.
    #[test]
    fn test_embed_batch_matches_individual() {
        let (weights, config) = make_tiny_weights();
        let seq_a: &[u32] = &[1, 3];
        let seq_b: &[u32] = &[5, 2, 4];

        let individual: Vec<Vec<f32>> = vec![
            embed(&weights, &config, seq_a),
            embed(&weights, &config, seq_b),
        ];
        let batched = embed_batch(&weights, &config, &[seq_a, seq_b]);

        assert_eq!(batched.len(), 2);
        for (seq_idx, (ind, bat)) in individual.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                ind.len(),
                bat.len(),
                "embedding length mismatch for seq {seq_idx}"
            );
            for (i, (&a, &b)) in ind.iter().zip(bat).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "seq {seq_idx} index {i}: individual={a:.6}, batch={b:.6}"
                );
            }
        }
    }

    #[test]
    fn test_gemma2_forward_features() {
        let config = ModelConfig {
            architecture: "gemma2".to_string(),
            context_length: 32,
            embedding_length: 4,
            block_count: 2,
            head_count: 2,
            head_count_kv: 1,
            vocab_size: 8,
            feed_forward_length: Some(8),
            rms_norm_eps: 1e-5,
            rope_freq_base: Some(10000.0),
            chat_template: None,
            sliding_window: Some(2),
            rope_scaling_factor: None,
            partial_rotary_factor: None,
            head_dim_override: Some(2),
            attn_logit_softcapping: Some(10.0),
            final_logit_softcapping: Some(15.0),
            query_pre_attn_scalar: Some(1.0),
            sliding_window_alternating: true,
        };

        let make_layer = || LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0, 1.1, 0.9, 1.0], &[4]),
            ffn_norm: Tensor::from_vec(vec![1.0, 0.9, 1.1, 1.0], &[4]),
            post_attn_norm: Some(Tensor::from_vec(vec![1.05, 0.95, 1.0, 1.0], &[4])),
            post_ffn_norm: Some(Tensor::from_vec(vec![0.95, 1.05, 1.0, 1.0], &[4])),
            attn_q: QuantizedTensor::from_f32(
                &(0..16).map(|i| i as f32 * 0.05 - 0.4).collect::<Vec<_>>(),
                4,
                4,
            ),
            attn_k: QuantizedTensor::from_f32(
                &(0..8).map(|i| i as f32 * 0.1 - 0.3).collect::<Vec<_>>(),
                2,
                4,
            ),
            attn_v: QuantizedTensor::from_f32(
                &(0..8).map(|i| i as f32 * 0.07 - 0.2).collect::<Vec<_>>(),
                2,
                4,
            ),
            attn_output: QuantizedTensor::from_f32(
                &(0..16).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>(),
                4,
                4,
            ),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            ffn_gate: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(),
                8,
                4,
            ),
            ffn_up: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.015 - 0.2).collect::<Vec<_>>(),
                8,
                4,
            ),
            ffn_down: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.025 - 0.4).collect::<Vec<_>>(),
                4,
                8,
            ),
        };

        let weights = TransformerWeights {
            token_embedding: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                8,
                4,
            ),
            layers: vec![make_layer(), make_layer()],
            output_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            output: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.1 - 1.6).collect::<Vec<_>>(),
                8,
                4,
            ),
            lora: None,
        };

        let prompt = [1u32, 2, 3, 4];
        let next = 5u32;
        let mut cache = prefilled_cache(&weights, &config, &prompt);
        let solo = forward_one(&weights, &config, next, prompt.len(), &mut cache, &mut None);

        // Verify that logits are soft-capped (cannot exceed 15.0)
        for &val in solo.data() {
            assert!(val.abs() <= 15.0 + 1e-5, "val {val} exceeded softcap 15.0");
        }

        // Verify batched decode parity
        let mut caches: Vec<KvCache> = vec![prefilled_cache(&weights, &config, &prompt)];
        let mut cache_refs: Vec<&mut dyn KvStore> =
            caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
        let batched = forward_batch(
            &weights,
            &config,
            &[next],
            &[prompt.len()],
            &mut cache_refs,
            &mut None,
        );
        assert_same_logits(&solo, &batched[0], "gemma2 batch=1 parity");
    }

    #[test]
    fn test_qwen2_forward_features() {
        let config = ModelConfig {
            architecture: "qwen2".to_string(),
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
            head_dim_override: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window_alternating: false,
        };

        let weights = TransformerWeights {
            token_embedding: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                8,
                4,
            ),
            layers: vec![LayerWeights {
                attn_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
                ffn_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
                post_attn_norm: None,
                post_ffn_norm: None,
                attn_q: QuantizedTensor::from_f32(
                    &(0..16).map(|i| i as f32 * 0.05 - 0.4).collect::<Vec<_>>(),
                    4,
                    4,
                ),
                attn_k: QuantizedTensor::from_f32(
                    &(0..8).map(|i| i as f32 * 0.1 - 0.3).collect::<Vec<_>>(),
                    2,
                    4,
                ),
                attn_v: QuantizedTensor::from_f32(
                    &(0..8).map(|i| i as f32 * 0.07 - 0.2).collect::<Vec<_>>(),
                    2,
                    4,
                ),
                attn_output: QuantizedTensor::from_f32(
                    &(0..16).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>(),
                    4,
                    4,
                ),
                attn_q_bias: Some(Tensor::from_vec(vec![0.1, -0.1, 0.2, -0.2], &[4])),
                attn_k_bias: Some(Tensor::from_vec(vec![0.05, -0.05], &[2])),
                attn_v_bias: Some(Tensor::from_vec(vec![0.15, -0.15], &[2])),
                ffn_gate: QuantizedTensor::from_f32(
                    &(0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(),
                    8,
                    4,
                ),
                ffn_up: QuantizedTensor::from_f32(
                    &(0..32).map(|i| i as f32 * 0.015 - 0.2).collect::<Vec<_>>(),
                    8,
                    4,
                ),
                ffn_down: QuantizedTensor::from_f32(
                    &(0..32).map(|i| i as f32 * 0.025 - 0.4).collect::<Vec<_>>(),
                    4,
                    8,
                ),
            }],
            output_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            output: QuantizedTensor::from_f32(
                &(0..32).map(|i| i as f32 * 0.1 - 1.6).collect::<Vec<_>>(),
                8,
                4,
            ),
            lora: None,
        };

        let prompt = [1u32, 2, 3];
        let next = 4u32;
        let mut cache = prefilled_cache(&weights, &config, &prompt);
        let solo = forward_one(&weights, &config, next, prompt.len(), &mut cache, &mut None);

        let mut caches: Vec<KvCache> = vec![prefilled_cache(&weights, &config, &prompt)];
        let mut cache_refs: Vec<&mut dyn KvStore> =
            caches.iter_mut().map(|c| c as &mut dyn KvStore).collect();
        let batched = forward_batch(
            &weights,
            &config,
            &[next],
            &[prompt.len()],
            &mut cache_refs,
            &mut None,
        );
        assert_same_logits(&solo, &batched[0], "qwen2 batch=1 parity");
    }
}
