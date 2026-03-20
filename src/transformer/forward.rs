//! Decoder-only transformer forward pass (LLaMA-style).

use rayon::prelude::*;

use crate::cache::{KvCache, KvCacheQ8, KvStore};
use crate::model::config::ModelConfig;
use crate::sampling::Sampler;
use crate::tensor::{self, Tensor};
use super::weights::{LayerWeights, TransformerWeights};

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
            let ffn_out = feed_forward(&normed_ffn, layer);
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
fn feed_forward(x: &Tensor, layer: &LayerWeights) -> Tensor {
    let gate   = layer.ffn_gate.matvec(x.data());
    let up     = layer.ffn_up.matvec(x.data());
    let hidden = tensor::mul(&tensor::silu(&gate), &up);
    layer.ffn_down.matvec(hidden.data())
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
            let ffn_out = feed_forward(&normed_ffn, layer);
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
) -> Tensor {
    let mut hidden = weights.token_embedding.row_as_f32(token_id as usize);

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let normed = tensor::rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps);
        let attn_out = attention_cached(&normed, layer, config, pos, layer_idx, cache);
        let after_attn = tensor::add(&hidden, &attn_out);
        let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
        let ffn_out = feed_forward(&normed_ffn, layer);
        hidden = tensor::add(&after_attn, &ffn_out);
    }

    cache.advance();
    let normed = tensor::rms_norm(&hidden, &weights.output_norm, config.rms_norm_eps);
    weights.output.matvec(normed.data())
}

fn attention_cached(
    x: &Tensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    pos: usize,
    layer_idx: usize,
    cache: &mut dyn KvStore,
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

    cache.write(layer_idx, pos, k_cur.data(), v_cur.data());

    let window_start = config.sliding_window
        .map(|w| (pos as i64 - w as i64 + 1).max(0) as usize)
        .unwrap_or(0);
    let attend_len = pos + 1 - window_start;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; embed_dim];

    let cache_ro: &dyn KvStore = &*cache;
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

    let attn_vec = Tensor::from_vec(attn_output, &[embed_dim]);
    layer.attn_output.matvec(attn_vec.data())
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
) -> Tensor {
    let all = forward_prefill_inner(weights, config, token_ids, cache, pos_offset);
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
) -> Vec<Tensor> {
    forward_prefill_inner(weights, config, token_ids, cache, pos_offset)
}

fn forward_prefill_inner(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
    cache: &mut dyn KvStore,
    pos_offset: usize,
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

    // 1. Embed all tokens
    let mut hidden: Vec<Vec<f32>> = token_ids.iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize).data().to_vec())
        .collect();

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // a. Q/K/V projections + RoPE — parallel across positions
        let qkv: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = hidden.par_iter()
            .enumerate()
            .map(|(lp, h)| {
                let abs = pos_offset + lp;
                let h_t    = Tensor::from_vec(h.clone(), &[embed_dim]);
                let normed = tensor::rms_norm(&h_t, &layer.attn_norm, config.rms_norm_eps);
                let q = layer.attn_q.matvec(normed.data());
                let k = layer.attn_k.matvec(normed.data());
                let v = layer.attn_v.matvec(normed.data());
                let q = tensor::rope(&q, abs, head_dim, freq_base, rope_scale, rot_dim);
                let k = tensor::rope(&k, abs, head_dim, freq_base, rope_scale, rot_dim);
                (q.data().to_vec(), k.data().to_vec(), v.data().to_vec())
            })
            .collect();

        // b. Write all K/V (sequential — must finish before reads below)
        for (lp, kv) in qkv.iter().enumerate() {
            cache.write(layer_idx, pos_offset + lp, &kv.1, &kv.2);
        }

        // c. Attention — parallel across positions (cache is read-only now)
        let cache_ro: &dyn KvStore = &*cache;
        let attn_outs: Vec<Vec<f32>> = (0..seq_len).into_par_iter()
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
                layer.attn_output.matvec(attn_vec.data()).data().to_vec()
            })
            .collect();

        // d. Residual + FFN — parallel across positions
        hidden = hidden.into_par_iter()
            .zip(attn_outs.into_par_iter())
            .map(|(h, ao)| {
                let h_t        = Tensor::from_vec(h,  &[embed_dim]);
                let a_t        = Tensor::from_vec(ao, &[embed_dim]);
                let after_attn = tensor::add(&h_t, &a_t);
                let nf         = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
                let ffn_out    = feed_forward(&nf, layer);
                tensor::add(&after_attn, &ffn_out).data().to_vec()
            })
            .collect();
    }

    // Advance cache seq_len positions
    for _ in 0..seq_len { cache.advance(); }

    // 3. Final norm + LM head for every position
    hidden.iter()
        .map(|h| {
            let h_t    = Tensor::from_vec(h.clone(), &[embed_dim]);
            let normed = tensor::rms_norm(&h_t, &weights.output_norm, config.rms_norm_eps);
            weights.output.matvec(normed.data())
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
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0);
    for step in 0..max_new_tokens {
        let next = last_logits.data().iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32).unwrap_or(0);
        eprintln!("  Decode {}/{}: token {next}", step + 1, max_new_tokens);
        tokens.push(next);
        if next == 2 { break; }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, &mut cache);
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
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    generate_with_cache(weights, config, prompt_tokens, max_new_tokens, sampler, eos_token, &mut cache)
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
) -> Vec<u32> {
    let mut cache = KvCacheQ8::new(
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    generate_with_cache(weights, config, prompt_tokens, max_new_tokens, sampler, eos_token, &mut cache)
}

fn generate_with_cache(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    cache: &mut dyn KvStore,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, cache, 0);
    for step in 0..max_new_tokens {
        let next = sampler.sample(last_logits.data(), &tokens);
        eprintln!("  Decode {}/{}: token {next}", step + 1, max_new_tokens);
        tokens.push(next);
        if next == eos_token { break; }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, cache);
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
) -> Vec<u32> {
    let mut cache = KvCache::new(
        config.block_count as usize, config.context_length as usize,
        config.head_count_kv as usize, config.head_dim() as usize,
    );
    let mut tokens = prompt_tokens.to_vec();
    let mut last_logits = forward_prefill(weights, config, prompt_tokens, &mut cache, 0);
    for _ in 0..max_new_tokens {
        let next = sampler.sample(last_logits.data(), &tokens);
        tokens.push(next);
        if !on_token(next) || next == eos_token { break; }
        last_logits = forward_one(weights, config, next, tokens.len() - 1, &mut cache);
    }
    tokens
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
            logits_cached = forward_one(&weights, &config, tok, pos, &mut cache);
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
            logits_seq = forward_one(&weights, &config, tok, pos, &mut cache_seq);
        }

        // Batched prefill
        let mut cache_batch = KvCache::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let logits_batch = forward_prefill(&weights, &config, &token_ids, &mut cache_batch, 0);

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
        let l_f32 = forward_prefill(&weights, &config, &token_ids, &mut c_f32, 0);

        let mut c_q8 = KvCacheQ8::new(
            config.block_count as usize, config.context_length as usize,
            config.head_count_kv as usize, config.head_dim() as usize,
        );
        let l_q8 = forward_prefill(&weights, &config, &token_ids, &mut c_q8, 0);

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
        assert_eq!(feed_forward(&x, &layer).shape(), &[4]);
    }
}
