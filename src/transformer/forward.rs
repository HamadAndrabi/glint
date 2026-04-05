//! Decoder-only transformer forward pass (LLaMA-style).

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::backend::GpuBackend;
use crate::cache::{KvCache, KvCacheQ8, KvStore};
use crate::model::config::ModelConfig;
use crate::model::lora::{LoraLayerAdapters, LoraWeights};
use crate::sampling::Sampler;
use crate::tensor::{self, QuantizedTensor, Tensor};
use super::weights::{LayerWeights, TransformerWeights};

// ── GPU dispatch helpers ──────────────────────────────────────────────────────

/// Reborrow `Option<&mut GpuBackend>` so it can be used multiple times.
fn reborrow<'a>(gpu: &'a mut Option<&mut GpuBackend>) -> Option<&'a mut GpuBackend> {
    gpu.as_mut().map(|g| &mut **g)
}

/// Matvec that dispatches to GPU when available, CPU otherwise.
fn matvec_maybe_gpu(
    qt: &QuantizedTensor,
    input: &[f32],
    gpu: Option<&mut GpuBackend>,
) -> Tensor {
    #[cfg(feature = "vulkan")]
    if let Some(g) = gpu {
        return qt.matvec_gpu(input, g);
    }
    let _ = gpu;
    qt.matvec(input)
}

/// RMS normalization — GPU when available, CPU otherwise.
#[cfg(feature = "vulkan")]
fn rms_norm_maybe_gpu(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    gpu: Option<&GpuBackend>,
) -> Tensor {
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
pub fn forward(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Tensor {
    let n_tokens = token_ids.len();
    let mut hidden_states: Vec<Tensor> = token_ids
        .iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize))
        .collect();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        eprint!("\r  Layer {}/{}...", layer_idx + 1, config.block_count);
        let mut new_hidden_states = Vec::with_capacity(n_tokens);
        for pos in 0..n_tokens {
            let normed = tensor::rms_norm(&hidden_states[pos], &layer.attn_norm, config.rms_norm_eps);
            let attn_out = attention(&normed, &hidden_states, layer, config, pos);
            let after_attn = tensor::add(&hidden_states[pos], &attn_out);
            let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
            let ffn_out = feed_forward(&normed_ffn, layer, None, &mut None);
            new_hidden_states.push(tensor::add(&after_attn, &ffn_out));
        }
        hidden_states = new_hidden_states;
    }
    eprintln!();

    let last_hidden = &hidden_states[n_tokens - 1];
    let normed = tensor::rms_norm(last_hidden, &weights.output_norm, config.rms_norm_eps);
    weights.output.matvec(normed.data())
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
    let rot_dim = config.partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);

    let q_all = layer.attn_q.matvec(x.data());
    let k_cur = layer.attn_k.matvec(x.data());
    let v_cur = layer.attn_v.matvec(x.data());
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base, rope_scaling, rot_dim);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base, rope_scaling, rot_dim);

    let mut k_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    let mut v_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    for (p, hidden) in all_hidden.iter().enumerate().take(pos) {
        let normed_p = tensor::rms_norm(hidden, &layer.attn_norm, config.rms_norm_eps);
        let k_p = layer.attn_k.matvec(normed_p.data());
        let v_p = layer.attn_v.matvec(normed_p.data());
        k_cache.push(tensor::rope(&k_p, p, head_dim, freq_base, rope_scaling, rot_dim));
        v_cache.push(v_p);
    }
    k_cache.push(k_cur);
    v_cache.push(v_cur);

    let seq_len = k_cache.len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; embed_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q_all.data()[q_offset..q_offset + head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for (s, k_pos) in k_cache.iter().enumerate() {
            let k_head = &k_pos.data()[kv_h * head_dim..(kv_h + 1) * head_dim];
            scores[s] = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let scores_t = Tensor::from_vec(scores, &[seq_len]);
        let attn_weights = tensor::softmax(&scores_t);
        for (s, v_pos) in v_cache.iter().enumerate() {
            let w = attn_weights.get_flat(s);
            let v_head = &v_pos.data()[kv_h * head_dim..(kv_h + 1) * head_dim];
            for d in 0..head_dim { attn_output[q_offset + d] += w * v_head[d]; }
        }
    }

    let attn_vec = Tensor::from_vec(attn_output, &[embed_dim]);
    layer.attn_output.matvec(attn_vec.data())
}

/// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
fn feed_forward(
    x: &Tensor,
    layer: &LayerWeights,
    lora: Option<&LoraLayerAdapters>,
    gpu: &mut Option<&mut GpuBackend>,
) -> Tensor {
    let mut gate = matvec_maybe_gpu(&layer.ffn_gate, x.data(), reborrow(gpu));
    let mut up   = matvec_maybe_gpu(&layer.ffn_up, x.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.ffn_gate { a.apply(x.data(), gate.data_mut()); }
        if let Some(a) = &ll.ffn_up   { a.apply(x.data(),   up.data_mut()); }
    }
    #[cfg(feature = "vulkan")]
    let hidden = silu_mul_maybe_gpu(&gate, &up, gpu.as_deref());
    #[cfg(not(feature = "vulkan"))]
    let hidden = tensor::mul(&tensor::silu(&gate), &up);
    let mut out = matvec_maybe_gpu(&layer.ffn_down, hidden.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.ffn_down { a.apply(hidden.data(), out.data_mut()); }
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
        inputs.par_iter()
            .map(|token_ids| embed(weights, config, token_ids))
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    inputs.iter()
        .map(|token_ids| embed(weights, config, token_ids))
        .collect()
}

/// Compute a text embedding by mean-pooling the final hidden states.
pub fn embed(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Vec<f32> {
    let n_tokens = token_ids.len();
    let embed_dim = config.embedding_length as usize;
    let mut hidden_states: Vec<Tensor> = token_ids
        .iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize))
        .collect();
    for layer in weights.layers.iter() {
        let mut new_hs = Vec::with_capacity(n_tokens);
        for pos in 0..n_tokens {
            let normed = tensor::rms_norm(&hidden_states[pos], &layer.attn_norm, config.rms_norm_eps);
            let attn_out = attention(&normed, &hidden_states, layer, config, pos);
            let after_attn = tensor::add(&hidden_states[pos], &attn_out);
            let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
            let ffn_out = feed_forward(&normed_ffn, layer, None, &mut None);
            new_hs.push(tensor::add(&after_attn, &ffn_out));
        }
        hidden_states = new_hs;
    }
    let mut embedding = vec![0.0f32; embed_dim];
    for hidden in &hidden_states {
        let normed = tensor::rms_norm(hidden, &weights.output_norm, config.rms_norm_eps);
        for (acc, &v) in embedding.iter_mut().zip(normed.data()) { *acc += v; }
    }
    for acc in &mut embedding { *acc /= n_tokens as f32; }
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
        eprintln!("--- Step {}/{} (seq_len={}) ---", step + 1, max_new_tokens, tokens.len());
        let logits = forward(weights, config, &tokens);
        let next = logits.data().iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32).unwrap_or(0);
        tokens.push(next);
        if next == 2 { break; }
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
        let normed = rms_norm_maybe_gpu(&hidden, &layer.attn_norm, config.rms_norm_eps, gpu.as_deref());
        #[cfg(not(feature = "vulkan"))]
        let normed = tensor::rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps);
        let attn_out = attention_cached(&normed, layer, lora_layer, config, pos, layer_idx, cache, gpu);
        #[cfg(feature = "vulkan")]
        let after_attn = add_maybe_gpu(&hidden, &attn_out, gpu.as_deref());
        #[cfg(not(feature = "vulkan"))]
        let after_attn = tensor::add(&hidden, &attn_out);
        #[cfg(feature = "vulkan")]
        let normed_ffn = rms_norm_maybe_gpu(&after_attn, &layer.ffn_norm, config.rms_norm_eps, gpu.as_deref());
        #[cfg(not(feature = "vulkan"))]
        let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
        let ffn_out = feed_forward(&normed_ffn, layer, lora_layer, gpu);
        #[cfg(feature = "vulkan")]
        { hidden = add_maybe_gpu(&after_attn, &ffn_out, gpu.as_deref()); }
        #[cfg(not(feature = "vulkan"))]
        { hidden = tensor::add(&after_attn, &ffn_out); }
    }

    cache.advance();
    #[cfg(feature = "vulkan")]
    let normed = rms_norm_maybe_gpu(&hidden, &weights.output_norm, config.rms_norm_eps, gpu.as_deref());
    #[cfg(not(feature = "vulkan"))]
    let normed = tensor::rms_norm(&hidden, &weights.output_norm, config.rms_norm_eps);
    matvec_maybe_gpu(&weights.output, normed.data(), reborrow(gpu))
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
    let rot_dim = config.partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);

    let mut q_all = matvec_maybe_gpu(&layer.attn_q, x.data(), reborrow(gpu));
    let mut k_cur = matvec_maybe_gpu(&layer.attn_k, x.data(), reborrow(gpu));
    let mut v_cur = matvec_maybe_gpu(&layer.attn_v, x.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.attn_q { a.apply(x.data(), q_all.data_mut()); }
        if let Some(a) = &ll.attn_k { a.apply(x.data(), k_cur.data_mut()); }
        if let Some(a) = &ll.attn_v { a.apply(x.data(), v_cur.data_mut()); }
    }
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base, rope_scaling, rot_dim);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base, rope_scaling, rot_dim);

    cache.write(layer_idx, pos, k_cur.data(), v_cur.data());

    let window_start = config.sliding_window
        .map(|w| (pos as i64 - w as i64 + 1).max(0) as usize)
        .unwrap_or(0);
    let attend_len = pos + 1 - window_start;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; embed_dim];

    let cache_ro: &dyn KvStore = &*cache;

    // ── GPU attention path ──────────────────────────────────────────────────
    // Two sub-paths:
    //
    // 1. GPU-resident (GpuKvCache): K/V are already in a GPU buffer; bind
    //    them directly.  Only the new token's slice was uploaded this step.
    //    This is the efficient path: O(head_dim) CPU→GPU per decode step.
    //
    // 2. CPU-cache (KvCache / KvCacheQ8): extract the full K/V window into
    //    flat CPU buffers, then upload to GPU for the attention dispatch.
    //    O(seq_len * head_dim) upload per decode step (existing behaviour).
    //
    // Falls back to CPU flash attention if GPU is unavailable or seq_len > 4096.
    #[cfg(feature = "vulkan")]
    let gpu_attn: Option<Tensor> = {
        if let Some(ref mut g) = *gpu {
            if attend_len <= 4096 {
                if let Some(kv_buf) = cache_ro.gpu_buffer() {
                    // Path 1: GPU-resident K/V — no full-context upload needed.
                    g.attention_resident(
                        q_all.data(), kv_buf,
                        layer_idx, window_start, attend_len,
                        n_heads as u32, n_kv_heads as u32,
                        head_dim as u32, scale,
                    ).ok().map(|data| Tensor::from_vec(data, &[embed_dim]))
                } else {
                    // Path 2: CPU cache — copy full window to GPU, then dispatch.
                    let kv_stride = n_kv_heads * head_dim;
                    let mut k_flat = vec![0.0f32; attend_len * kv_stride];
                    let mut v_flat = vec![0.0f32; attend_len * kv_stride];
                    for i in 0..attend_len {
                        for kv_h in 0..n_kv_heads {
                            let off = i * kv_stride + kv_h * head_dim;
                            cache_ro.read_k_head(layer_idx, window_start + i, kv_h, head_dim, &mut k_flat[off..off + head_dim]);
                            cache_ro.read_v_head(layer_idx, window_start + i, kv_h, head_dim, &mut v_flat[off..off + head_dim]);
                        }
                    }
                    g.attention(
                        q_all.data(), &k_flat, &v_flat,
                        n_heads as u32, n_kv_heads as u32,
                        head_dim as u32, attend_len as u32, scale,
                    ).ok().map(|data| Tensor::from_vec(data, &[embed_dim]))
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
        // CPU flash attention (tiled online softmax, O(1) extra memory per head)
        for h in 0..n_heads {
            let kv_h = h / kv_group_size;
            let q_offset = h * head_dim;
            tensor::flash_attn_1d(
                &q_all.data()[q_offset..q_offset + head_dim],
                cache_ro, layer_idx, kv_h,
                window_start, attend_len, head_dim, scale,
                &mut attn_output[q_offset..q_offset + head_dim],
            );
        }
        Tensor::from_vec(attn_output, &[embed_dim])
    };
    let mut out = matvec_maybe_gpu(&layer.attn_output, attn_vec.data(), reborrow(gpu));
    if let Some(ll) = lora {
        if let Some(a) = &ll.attn_output { a.apply(attn_vec.data(), out.data_mut()); }
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
    all.into_iter().last()
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
    if seq_len == 0 { return vec![]; }

    let embed_dim  = config.embedding_length as usize;
    let n_heads    = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim   = config.head_dim() as usize;
    let kv_group   = n_heads / n_kv_heads;
    let freq_base  = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scale = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim    = config.partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use sequential path when GPU is active (GPU parallelism replaces rayon)
    // or when rayon is not available (e.g. wasm32 builds).
    #[cfg(feature = "rayon")]
    let use_sequential = gpu.is_some();
    #[cfg(not(feature = "rayon"))]
    let use_sequential = true;

    // 1. Embed all tokens
    let mut hidden: Vec<Vec<f32>> = token_ids.iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize).data().to_vec())
        .collect();

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let effective_lora = lora.or(weights.lora.as_ref());
        let lora_layer = effective_lora.and_then(|l| l.layers.get(layer_idx));

        // a. Q/K/V projections + RoPE
        // When GPU is active, run sequentially (GPU parallelism replaces rayon).
        let qkv: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = if use_sequential {
            hidden.iter()
                .enumerate()
                .map(|(lp, h)| {
                    let abs = pos_offset + lp;
                    let h_t    = Tensor::from_vec(h.clone(), &[embed_dim]);
                    #[cfg(feature = "vulkan")]
                    let normed = rms_norm_maybe_gpu(&h_t, &layer.attn_norm, config.rms_norm_eps, gpu.as_deref());
                    #[cfg(not(feature = "vulkan"))]
                    let normed = tensor::rms_norm(&h_t, &layer.attn_norm, config.rms_norm_eps);
                    let mut q = matvec_maybe_gpu(&layer.attn_q, normed.data(), reborrow(gpu));
                    let mut k = matvec_maybe_gpu(&layer.attn_k, normed.data(), reborrow(gpu));
                    let mut v = matvec_maybe_gpu(&layer.attn_v, normed.data(), reborrow(gpu));
                    if let Some(ll) = lora_layer {
                        if let Some(a) = &ll.attn_q { a.apply(normed.data(), q.data_mut()); }
                        if let Some(a) = &ll.attn_k { a.apply(normed.data(), k.data_mut()); }
                        if let Some(a) = &ll.attn_v { a.apply(normed.data(), v.data_mut()); }
                    }
                    let q = tensor::rope(&q, abs, head_dim, freq_base, rope_scale, rot_dim);
                    let k = tensor::rope(&k, abs, head_dim, freq_base, rope_scale, rot_dim);
                    (q.data().to_vec(), k.data().to_vec(), v.data().to_vec())
                })
                .collect()
        } else {
            #[cfg(feature = "rayon")]
            {
                hidden.par_iter()
                    .enumerate()
                    .map(|(lp, h)| {
                        let abs = pos_offset + lp;
                        let h_t    = Tensor::from_vec(h.clone(), &[embed_dim]);
                        let normed = tensor::rms_norm(&h_t, &layer.attn_norm, config.rms_norm_eps);
                        let mut q = layer.attn_q.matvec(normed.data());
                        let mut k = layer.attn_k.matvec(normed.data());
                        let mut v = layer.attn_v.matvec(normed.data());
                        if let Some(ll) = lora_layer {
                            if let Some(a) = &ll.attn_q { a.apply(normed.data(), q.data_mut()); }
                            if let Some(a) = &ll.attn_k { a.apply(normed.data(), k.data_mut()); }
                            if let Some(a) = &ll.attn_v { a.apply(normed.data(), v.data_mut()); }
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
            (0..seq_len).map(|lp| {
                let abs = pos_offset + lp;
                let window = config.sliding_window
                    .map(|w| (abs as i64 - w as i64 + 1).max(0) as usize)
                    .unwrap_or(0);
                let attend_len = abs + 1 - window;
                let mut out = vec![0.0f32; embed_dim];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;
                    let q_off = h * head_dim;
                    tensor::flash_attn_1d(
                        &qkv[lp].0[q_off..q_off + head_dim],
                        cache_ro, layer_idx, kv_h,
                        window, attend_len, head_dim, scale,
                        &mut out[q_off..q_off + head_dim],
                    );
                }
                let attn_vec = Tensor::from_vec(out, &[embed_dim]);
                let mut proj = matvec_maybe_gpu(&layer.attn_output, attn_vec.data(), reborrow(gpu));
                if let Some(ll) = lora_layer {
                    if let Some(a) = &ll.attn_output { a.apply(attn_vec.data(), proj.data_mut()); }
                }
                proj.data().to_vec()
            }).collect()
        } else {
            #[cfg(feature = "rayon")]
            {
                (0..seq_len).into_par_iter()
                    .map(|lp| {
                        let abs = pos_offset + lp;
                        let window = config.sliding_window
                            .map(|w| (abs as i64 - w as i64 + 1).max(0) as usize)
                            .unwrap_or(0);
                        let attend_len = abs + 1 - window;
                        let mut out = vec![0.0f32; embed_dim];
                        for h in 0..n_heads {
                            let kv_h = h / kv_group;
                            let q_off = h * head_dim;
                            tensor::flash_attn_1d(
                                &qkv[lp].0[q_off..q_off + head_dim],
                                cache_ro, layer_idx, kv_h,
                                window, attend_len, head_dim, scale,
                                &mut out[q_off..q_off + head_dim],
                            );
                        }
                        let attn_vec = Tensor::from_vec(out, &[embed_dim]);
                        let mut proj = layer.attn_output.matvec(attn_vec.data());
                        if let Some(ll) = lora_layer {
                            if let Some(a) = &ll.attn_output { a.apply(attn_vec.data(), proj.data_mut()); }
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
            hidden = hidden.into_iter()
                .zip(attn_outs.into_iter())
                .map(|(h, ao)| {
                    let h_t        = Tensor::from_vec(h,  &[embed_dim]);
                    let a_t        = Tensor::from_vec(ao, &[embed_dim]);
                    #[cfg(feature = "vulkan")]
                    let after_attn = add_maybe_gpu(&h_t, &a_t, gpu.as_deref());
                    #[cfg(not(feature = "vulkan"))]
                    let after_attn = tensor::add(&h_t, &a_t);
                    #[cfg(feature = "vulkan")]
                    let nf = rms_norm_maybe_gpu(&after_attn, &layer.ffn_norm, config.rms_norm_eps, gpu.as_deref());
                    #[cfg(not(feature = "vulkan"))]
                    let nf = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
                    let ffn_out = feed_forward(&nf, layer, lora_layer, gpu);
                    #[cfg(feature = "vulkan")]
                    { add_maybe_gpu(&after_attn, &ffn_out, gpu.as_deref()).data().to_vec() }
                    #[cfg(not(feature = "vulkan"))]
                    { tensor::add(&after_attn, &ffn_out).data().to_vec() }
                })
                .collect();
        } else {
            #[cfg(feature = "rayon")]
            {
                hidden = hidden.into_par_iter()
                    .zip(attn_outs.into_par_iter())
                    .map(|(h, ao)| {
                        let h_t        = Tensor::from_vec(h,  &[embed_dim]);
                        let a_t        = Tensor::from_vec(ao, &[embed_dim]);
                        let after_attn = tensor::add(&h_t, &a_t);
                        let nf         = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
                        let ffn_out    = feed_forward(&nf, layer, lora_layer, &mut None);
                        tensor::add(&after_attn, &ffn_out).data().to_vec()
                    })
                    .collect();
            }
            #[cfg(not(feature = "rayon"))]
            unreachable!();
        }
    }

    // Advance cache seq_len positions
    for _ in 0..seq_len { cache.advance(); }

    // 3. Final norm + LM head for every position
    hidden.iter()
        .map(|h| {
            let h_t    = Tensor::from_vec(h.clone(), &[embed_dim]);
            #[cfg(feature = "vulkan")]
            let normed = rms_norm_maybe_gpu(&h_t, &weights.output_norm, config.rms_norm_eps, gpu.as_deref());
            #[cfg(not(feature = "vulkan"))]
            let normed = tensor::rms_norm(&h_t, &weights.output_norm, config.rms_norm_eps);
            matvec_maybe_gpu(&weights.output, normed.data(), reborrow(gpu))
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
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0, gpu);
    for step in 0..max_new_tokens {
        let next = last_logits.data().iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32).unwrap_or(0);
        eprintln!("  Decode {}/{}: token {next}", step + 1, max_new_tokens);
        tokens.push(next);
        if next == 2 { break; }
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
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    generate_with_cache(weights, config, prompt_tokens, max_new_tokens, sampler, eos_token, &mut cache, gpu)
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
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    generate_with_cache(weights, config, prompt_tokens, max_new_tokens, sampler, eos_token, &mut cache, gpu)
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
        if next == eos_token { break; }
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
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0, gpu);
    for _ in 0..max_new_tokens {
        let next = sampler.sample(last_logits.data(), &tokens);
        tokens.push(next);
        if !on_token(next) || next == eos_token { break; }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, &mut cache, gpu);
    }
    tokens
}

// ── Batched decode (multi-sequence) ──────────────────────────────────────────

/// Decode one step for multiple sequences simultaneously.
///
/// Each element `i` provides:
/// - `tokens[i]`    — current token to decode
/// - `positions[i]` — current decode position (0-based from prompt start)
/// - `caches[i]`    — exclusive KV cache for sequence `i`
///
/// Returns one logits [`Tensor`] per sequence.
///
/// ## Throughput strategy (CPU)
///
/// The embedding lookup and Q/K/V projections for all N sequences are
/// dispatched with `rayon::par_iter` when the `rayon` feature is active,
/// giving N-way parallelism on multi-core hosts.  Attention is per-sequence
/// (independent caches) and likewise rayon-parallelised.  When GPU is active,
/// sequences are processed sequentially to avoid conflicting GPU borrows.
pub fn forward_batch(
    weights:   &TransformerWeights,
    config:    &ModelConfig,
    tokens:    &[u32],
    positions: &[usize],
    caches:    &mut [&mut dyn KvStore],
    gpu:       &mut Option<&mut GpuBackend>,
) -> Vec<Tensor> {
    let n_seqs = tokens.len();
    if n_seqs == 0 { return vec![]; }

    // Fast path: single sequence — use the battle-tested forward_one.
    if n_seqs == 1 {
        return vec![forward_one(weights, config, tokens[0], positions[0], caches[0], gpu)];
    }

    // GPU path: sequential to avoid simultaneous GPU borrows.
    // CPU path below uses rayon.
    if gpu.is_some() {
        return tokens.iter().zip(positions.iter()).zip(caches.iter_mut())
            .map(|((&tok, &pos), cache)| {
                forward_one(weights, config, tok, pos, *cache, gpu)
            })
            .collect();
    }

    // ── CPU batch path ─────────────────────────────────────────────────────

    // 1. Initial token embeddings for all sequences.
    let mut hiddens: Vec<Tensor> = tokens.iter()
        .map(|&t| weights.token_embedding.row_as_f32(t as usize))
        .collect();

    let embed_dim    = config.embedding_length as usize;
    let n_heads      = config.head_count as usize;
    let n_kv_heads   = config.head_count_kv as usize;
    let head_dim     = config.head_dim() as usize;
    let kv_group_sz  = n_heads / n_kv_heads;
    let freq_base    = config.rope_freq_base.unwrap_or(10000.0);
    let rope_scaling = config.rope_scaling_factor.unwrap_or(1.0);
    let rot_dim      = config.partial_rotary_factor
        .map(|f| (head_dim as f32 * f) as usize & !1)
        .unwrap_or(head_dim);
    let scale        = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let lora_layer = weights.lora.as_ref().and_then(|l| l.layers.get(layer_idx));

        // 2a. RMS-norm every hidden state — embarrassingly parallel.
        #[cfg(feature = "rayon")]
        let normed: Vec<Tensor> = hiddens.par_iter()
            .map(|h| tensor::rms_norm(h, &layer.attn_norm, config.rms_norm_eps))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let normed: Vec<Tensor> = hiddens.iter()
            .map(|h| tensor::rms_norm(h, &layer.attn_norm, config.rms_norm_eps))
            .collect();

        // 2b. Q/K/V projections — parallel across sequences.
        #[cfg(feature = "rayon")]
        let qkv: Vec<(Tensor, Tensor, Tensor)> = normed.par_iter()
            .map(|n| {
                let mut q = layer.attn_q.matvec(n.data());
                let mut k = layer.attn_k.matvec(n.data());
                let mut v = layer.attn_v.matvec(n.data());
                if let Some(ll) = lora_layer {
                    if let Some(a) = &ll.attn_q { a.apply(n.data(), q.data_mut()); }
                    if let Some(a) = &ll.attn_k { a.apply(n.data(), k.data_mut()); }
                    if let Some(a) = &ll.attn_v { a.apply(n.data(), v.data_mut()); }
                }
                (q, k, v)
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let qkv: Vec<(Tensor, Tensor, Tensor)> = normed.iter()
            .map(|n| {
                let mut q = layer.attn_q.matvec(n.data());
                let mut k = layer.attn_k.matvec(n.data());
                let mut v = layer.attn_v.matvec(n.data());
                if let Some(ll) = lora_layer {
                    if let Some(a) = &ll.attn_q { a.apply(n.data(), q.data_mut()); }
                    if let Some(a) = &ll.attn_k { a.apply(n.data(), k.data_mut()); }
                    if let Some(a) = &ll.attn_v { a.apply(n.data(), v.data_mut()); }
                }
                (q, k, v)
            })
            .collect();

        // 2c. RoPE (position-dependent, cheap) + K/V cache write (sequential, fast memcpy).
        let mut q_roped: Vec<Tensor> = Vec::with_capacity(n_seqs);
        for (i, (q, k, v)) in qkv.into_iter().enumerate() {
            let pos = positions[i];
            let q_r = tensor::rope(&q, pos, head_dim, freq_base, rope_scaling, rot_dim);
            let k_r = tensor::rope(&k, pos, head_dim, freq_base, rope_scaling, rot_dim);
            caches[i].write(layer_idx, pos, k_r.data(), v.data());
            q_roped.push(q_r);
        }

        // 2d. Attention — each sequence reads its own cache independently.
        //     Parallelise via rayon; each iteration captures only its own cache slice.
        //     We need immutable refs to caches for reading.
        let attn_vecs: Vec<Tensor> = {
            // Collect K/V reads into per-sequence buffers so we can release the
            // mutable cache borrow before parallelising.
            let per_seq: Vec<(Vec<f32>, usize, usize)> = (0..n_seqs).map(|i| {
                let pos = positions[i];
                let window_start = config.sliding_window
                    .map(|w| (pos as i64 - w as i64 + 1).max(0) as usize)
                    .unwrap_or(0);
                let attend_len = pos + 1 - window_start;
                let kv_stride = n_kv_heads * head_dim;
                let mut kv_flat = vec![0.0f32; attend_len * kv_stride * 2];
                let cache_ro: &dyn KvStore = caches[i];
                for t in 0..attend_len {
                    for kv_h in 0..n_kv_heads {
                        let off_k = t * kv_stride + kv_h * head_dim;
                        let off_v = attend_len * kv_stride + t * kv_stride + kv_h * head_dim;
                        cache_ro.read_k_head(layer_idx, window_start + t, kv_h, head_dim, &mut kv_flat[off_k..off_k + head_dim]);
                        cache_ro.read_v_head(layer_idx, window_start + t, kv_h, head_dim, &mut kv_flat[off_v..off_v + head_dim]);
                    }
                }
                (kv_flat, attend_len, window_start)
            }).collect();

            #[cfg(feature = "rayon")]
            let results: Vec<Tensor> = per_seq.into_par_iter().zip(q_roped.par_iter())
                .map(|((kv_flat, attend_len, _window_start), q)| {
                    let kv_stride = n_kv_heads * head_dim;
                    let mut output = vec![0.0f32; embed_dim];
                    for h in 0..n_heads {
                        let kv_h = h / kv_group_sz;
                        let q_off = h * head_dim;
                        let q_head = &q.data()[q_off..q_off + head_dim];
                        // Build per-head K/V slices from kv_flat
                        let mut k_buf = vec![0.0f32; attend_len * head_dim];
                        let mut v_buf = vec![0.0f32; attend_len * head_dim];
                        for t in 0..attend_len {
                            let k_src = &kv_flat[t * kv_stride + kv_h * head_dim..][..head_dim];
                            let v_src = &kv_flat[attend_len * kv_stride + t * kv_stride + kv_h * head_dim..][..head_dim];
                            k_buf[t * head_dim..(t + 1) * head_dim].copy_from_slice(k_src);
                            v_buf[t * head_dim..(t + 1) * head_dim].copy_from_slice(v_src);
                        }
                        // Scaled dot-product attention (online softmax)
                        let out_off = h * head_dim;
                        batch_attn_head(q_head, &k_buf, &v_buf, attend_len, head_dim, scale, &mut output[out_off..out_off + head_dim]);
                    }
                    Tensor::from_vec(output, &[embed_dim])
                })
                .collect();
            #[cfg(not(feature = "rayon"))]
            let results: Vec<Tensor> = per_seq.into_iter().zip(q_roped.iter())
                .map(|((kv_flat, attend_len, _window_start), q)| {
                    let kv_stride = n_kv_heads * head_dim;
                    let mut output = vec![0.0f32; embed_dim];
                    for h in 0..n_heads {
                        let kv_h = h / kv_group_sz;
                        let q_off = h * head_dim;
                        let q_head = &q.data()[q_off..q_off + head_dim];
                        let mut k_buf = vec![0.0f32; attend_len * head_dim];
                        let mut v_buf = vec![0.0f32; attend_len * head_dim];
                        for t in 0..attend_len {
                            let k_src = &kv_flat[t * kv_stride + kv_h * head_dim..][..head_dim];
                            let v_src = &kv_flat[attend_len * kv_stride + t * kv_stride + kv_h * head_dim..][..head_dim];
                            k_buf[t * head_dim..(t + 1) * head_dim].copy_from_slice(k_src);
                            v_buf[t * head_dim..(t + 1) * head_dim].copy_from_slice(v_src);
                        }
                        let out_off = h * head_dim;
                        batch_attn_head(q_head, &k_buf, &v_buf, attend_len, head_dim, scale, &mut output[out_off..out_off + head_dim]);
                    }
                    Tensor::from_vec(output, &[embed_dim])
                })
                .collect();
            results
        };

        // 2e. Output projection + residual (parallel across sequences).
        #[cfg(feature = "rayon")]
        let after_attn: Vec<Tensor> = attn_vecs.into_par_iter().zip(hiddens.par_iter())
            .map(|(attn_vec, h)| {
                let mut out = layer.attn_output.matvec(attn_vec.data());
                if let Some(ll) = lora_layer {
                    if let Some(a) = &ll.attn_output { a.apply(attn_vec.data(), out.data_mut()); }
                }
                tensor::add(h, &out)
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let after_attn: Vec<Tensor> = attn_vecs.into_iter().zip(hiddens.iter())
            .map(|(attn_vec, h)| {
                let mut out = layer.attn_output.matvec(attn_vec.data());
                if let Some(ll) = lora_layer {
                    if let Some(a) = &ll.attn_output { a.apply(attn_vec.data(), out.data_mut()); }
                }
                tensor::add(h, &out)
            })
            .collect();

        // 2f. FFN + residual (parallel).
        #[cfg(feature = "rayon")]
        {
            hiddens = after_attn.into_par_iter()
                .map(|h| {
                    let normed_ffn = tensor::rms_norm(&h, &layer.ffn_norm, config.rms_norm_eps);
                    let ffn_out = feed_forward_no_gpu(&normed_ffn, layer, lora_layer);
                    tensor::add(&h, &ffn_out)
                })
                .collect();
        }
        #[cfg(not(feature = "rayon"))]
        {
            hiddens = after_attn.into_iter()
                .map(|h| {
                    let normed_ffn = tensor::rms_norm(&h, &layer.ffn_norm, config.rms_norm_eps);
                    let ffn_out = feed_forward_no_gpu(&normed_ffn, layer, lora_layer);
                    tensor::add(&h, &ffn_out)
                })
                .collect();
        }
    }

    // 3. Advance all caches.
    for cache in caches.iter_mut() {
        cache.advance();
    }

    // 4. Final norm + LM head (parallel).
    #[cfg(feature = "rayon")]
    let logits: Vec<Tensor> = hiddens.into_par_iter()
        .map(|h| {
            let normed = tensor::rms_norm(&h, &weights.output_norm, config.rms_norm_eps);
            weights.output.matvec(normed.data())
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    let logits: Vec<Tensor> = hiddens.into_iter()
        .map(|h| {
            let normed = tensor::rms_norm(&h, &weights.output_norm, config.rms_norm_eps);
            weights.output.matvec(normed.data())
        })
        .collect();
    logits
}

/// Scaled dot-product attention for a single head — used by `forward_batch`.
///
/// `k_flat[t * head_dim .. (t+1)*head_dim]` holds the key for position `t`.
/// `v_flat` has the same layout.  Online softmax (numerically stable).
fn batch_attn_head(
    q:          &[f32],
    k_flat:     &[f32],
    v_flat:     &[f32],
    attend_len: usize,
    head_dim:   usize,
    scale:      f32,
    out:        &mut [f32],
) {
    // Online softmax: compute max, then weighted sum.
    let mut scores = vec![0.0f32; attend_len];
    for t in 0..attend_len {
        let k = &k_flat[t * head_dim..(t + 1) * head_dim];
        scores[t] = q.iter().zip(k).map(|(qi, ki)| qi * ki).sum::<f32>() * scale;
    }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in &mut scores { *s = (*s - max_s).exp(); exp_sum += *s; }
    if exp_sum > 0.0 { for s in &mut scores { *s /= exp_sum; } }

    for x in out.iter_mut() { *x = 0.0; }
    for t in 0..attend_len {
        let v = &v_flat[t * head_dim..(t + 1) * head_dim];
        for (o, vi) in out.iter_mut().zip(v) { *o += scores[t] * vi; }
    }
}

/// CPU-only FFN (no GPU dispatch) — used by `forward_batch`.
fn feed_forward_no_gpu(
    x:     &Tensor,
    layer: &LayerWeights,
    lora:  Option<&LoraLayerAdapters>,
) -> Tensor {
    feed_forward(x, layer, lora, &mut None)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::tensor::QuantizedTensor;

    fn make_tiny_weights() -> (TransformerWeights, ModelConfig) {
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
                ffn_gate:    QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.02 - 0.3).collect::<Vec<_>>(), 8, 4),
                ffn_up:      QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.015 - 0.2).collect::<Vec<_>>(), 8, 4),
                ffn_down:    QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.025 - 0.4).collect::<Vec<_>>(), 4, 8),
            }],
            output_norm: Tensor::from_vec(vec![1.0; 4], &[4]),
            output: QuantizedTensor::from_f32(&(0..32).map(|i| i as f32 * 0.1 - 1.6).collect::<Vec<_>>(), 8, 4),
            lora: None,
        };
        (weights, config)
    }

    #[test]
    fn test_cached_matches_uncached() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];
        let logits_uncached = forward(&weights, &config, &token_ids);
        let mut cache = KvCache::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let mut logits_cached = Tensor::zeros(&[1]);
        for (pos, &tok) in token_ids.iter().enumerate() {
            logits_cached = forward_one(&weights, &config, tok, pos, &mut cache, &mut None);
        }
        assert_eq!(logits_uncached.shape(), logits_cached.shape());
        for (i, (&a, &b)) in logits_uncached.data().iter().zip(logits_cached.data()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i}: uncached={a:.6}, cached={b:.6}");
        }
    }

    #[test]
    fn test_prefill_matches_sequential() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];

        // Sequential via forward_one
        let mut cache_seq = KvCache::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let mut logits_seq = Tensor::zeros(&[1]);
        for (pos, &tok) in token_ids.iter().enumerate() {
            logits_seq = forward_one(&weights, &config, tok, pos, &mut cache_seq, &mut None);
        }

        // Batched prefill
        let mut cache_batch = KvCache::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let logits_batch = forward_prefill(&weights, &config, &token_ids, &mut cache_batch, 0, &mut None);

        for (i, (&a, &b)) in logits_seq.data().iter().zip(logits_batch.data()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i}: sequential={a:.6}, prefill={b:.6}");
        }
    }

    #[test]
    fn test_prefill_q8_close_to_f32() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];

        let mut c_f32 = KvCache::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let l_f32 = forward_prefill(&weights, &config, &token_ids, &mut c_f32, 0, &mut None);

        let mut c_q8 = KvCacheQ8::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let l_q8 = forward_prefill(&weights, &config, &token_ids, &mut c_q8, 0, &mut None);

        for (i, (&a, &b)) in l_f32.data().iter().zip(l_q8.data()).enumerate() {
            assert!((a - b).abs() < 0.05, "index {i}: f32={a:.4}, q8={b:.4}");
        }
    }

    #[test]
    fn test_feed_forward_shape() {
        let layer = LayerWeights {
            attn_norm: Tensor::zeros(&[4]),
            ffn_norm:  Tensor::zeros(&[4]),
            attn_q:      QuantizedTensor::from_f32(&vec![0.0f32; 16], 4, 4),
            attn_k:      QuantizedTensor::from_f32(&vec![0.0f32; 16], 4, 4),
            attn_v:      QuantizedTensor::from_f32(&vec![0.0f32; 16], 4, 4),
            attn_output: QuantizedTensor::from_f32(&vec![0.0f32; 16], 4, 4),
            ffn_gate:    QuantizedTensor::from_f32(&vec![0.0f32; 32], 8, 4),
            ffn_up:      QuantizedTensor::from_f32(&vec![0.0f32; 32], 8, 4),
            ffn_down:    QuantizedTensor::from_f32(&vec![0.0f32; 32], 4, 8),
        };
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        assert_eq!(feed_forward(&x, &layer, None, &mut None).shape(), &[4]);
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
            assert_eq!(ind.len(), bat.len(), "embedding length mismatch for seq {seq_idx}");
            for (i, (&a, &b)) in ind.iter().zip(bat).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "seq {seq_idx} index {i}: individual={a:.6}, batch={b:.6}"
                );
            }
        }
    }
}
