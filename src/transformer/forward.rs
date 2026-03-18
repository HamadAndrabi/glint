//! Decoder-only transformer forward pass (LLaMA-style).

use crate::cache::KvCache;
use crate::model::config::ModelConfig;
use crate::sampling::Sampler;
use crate::tensor::{self, Tensor};
use super::weights::{LayerWeights, TransformerWeights};

/// Run a full forward pass: token_ids → logits.
///
/// This recomputes attention over all positions each time (no KV-cache).
/// Phase 2 adds KV-cache for efficient incremental decoding.
pub fn forward(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Tensor {
    let n_tokens = token_ids.len();

    // 1. Token embeddings: look up each token row from the quantized embedding table
    let mut hidden_states: Vec<Tensor> = token_ids
        .iter()
        .map(|&id| weights.token_embedding.row_as_f32(id as usize))
        .collect();

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        eprint!("\r  Layer {}/{}...", layer_idx + 1, config.block_count);

        let mut new_hidden_states = Vec::with_capacity(n_tokens);
        for pos in 0..n_tokens {
            // --- Attention block ---
            let normed = tensor::rms_norm(&hidden_states[pos], &layer.attn_norm, config.rms_norm_eps);
            let attn_out = attention(&normed, &hidden_states, layer, config, pos);
            let after_attn = tensor::add(&hidden_states[pos], &attn_out);

            // --- FFN block ---
            let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
            let ffn_out = feed_forward(&normed_ffn, layer);
            let after_ffn = tensor::add(&after_attn, &ffn_out);

            new_hidden_states.push(after_ffn);
        }
        hidden_states = new_hidden_states;
    }
    eprintln!();

    // 3. Final norm + LM head (only for last position)
    let last_hidden = &hidden_states[n_tokens - 1];
    let normed = tensor::rms_norm(last_hidden, &weights.output_norm, config.rms_norm_eps);

    // 4. Project to vocabulary: [vocab_size, embed_dim] × [embed_dim] → [vocab_size]
    weights.output.matvec(normed.data())
}

/// Multi-head self-attention for one position, attending to all positions ≤ pos.
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
    let kv_group_size = n_heads / n_kv_heads; // how many Q heads share one KV head

    // Q/K/V projections: weight [out_dim, embed_dim] × x [embed_dim] → [out_dim]
    let q_all = layer.attn_q.matvec(x.data());
    let k_cur = layer.attn_k.matvec(x.data());
    let v_cur = layer.attn_v.matvec(x.data());

    // Apply RoPE to Q and K for current position
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base);

    // Compute K and V for all positions we need to attend to (0..=pos)
    // In the no-cache version, we recompute K/V for all previous positions too
    let mut k_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    let mut v_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);

    for (p, hidden) in all_hidden.iter().enumerate().take(pos) {
        let normed_p = tensor::rms_norm(hidden, &layer.attn_norm, config.rms_norm_eps);
        let k_p = layer.attn_k.matvec(normed_p.data());
        let v_p = layer.attn_v.matvec(normed_p.data());
        k_cache.push(tensor::rope(&k_p, p, head_dim, freq_base));
        v_cache.push(v_p);
    }
    k_cache.push(k_cur);
    v_cache.push(v_cur);

    let seq_len = k_cache.len(); // = pos + 1
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut attn_output = vec![0.0f32; embed_dim];

    // Process each Q head
    for h in 0..n_heads {
        let kv_h = h / kv_group_size; // which KV head this Q head uses

        // Extract this head's Q vector
        let q_offset = h * head_dim;
        let q_head = &q_all.data()[q_offset..q_offset + head_dim];

        // Compute attention scores: dot(Q, K) * scale for each position
        let mut scores = vec![0.0f32; seq_len];
        for (s, k_pos) in k_cache.iter().enumerate() {
            let k_offset = kv_h * head_dim;
            let k_head = &k_pos.data()[k_offset..k_offset + head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores[s] = dot * scale;
        }

        // Softmax over scores
        let scores_tensor = Tensor::from_vec(scores, &[seq_len]);
        let attn_weights = tensor::softmax(&scores_tensor);

        // Weighted sum of V vectors
        let v_offset = kv_h * head_dim;
        for (s, v_pos) in v_cache.iter().enumerate() {
            let w = attn_weights.get_flat(s);
            let v_head = &v_pos.data()[v_offset..v_offset + head_dim];
            for d in 0..head_dim {
                attn_output[q_offset + d] += w * v_head[d];
            }
        }
    }

    // Output projection
    let attn_vec = Tensor::from_vec(attn_output, &[embed_dim]);
    layer.attn_output.matvec(attn_vec.data())
}

/// SwiGLU feed-forward network.
///
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
fn feed_forward(x: &Tensor, layer: &LayerWeights) -> Tensor {
    let gate = layer.ffn_gate.matvec(x.data());
    let up = layer.ffn_up.matvec(x.data());
    let activated = tensor::silu(&gate);
    let hidden = tensor::mul(&activated, &up);
    layer.ffn_down.matvec(hidden.data())
}

/// Greedy decoding: iteratively pick the highest-probability next token.
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

        // Argmax — total_cmp handles NaN safely (NaN sorts last)
        let next_token = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        eprintln!("  → token {next_token}");
        tokens.push(next_token);

        // Stop if we hit EOS (token ID 2 is common, but model-dependent)
        if next_token == 2 {
            break;
        }
    }

    tokens
}

// ── KV-cached forward pass (Phase 2.1) ─────────────────────────────────────

/// Forward pass for a single token at a given position, using the KV cache.
///
/// Reads K/V for positions 0..pos from the cache, computes K/V for the
/// current token and writes them to the cache, then returns logits.
///
/// For prefill: call in a loop for pos = 0, 1, ..., prompt_len - 1.
/// For decode: call once per generated token.
pub fn forward_one(
    weights: &TransformerWeights,
    config: &ModelConfig,
    token_id: u32,
    pos: usize,
    cache: &mut KvCache,
) -> Tensor {
    // 1. Embed one token — dequantize its row from the quantized embedding table
    let mut hidden = weights.token_embedding.row_as_f32(token_id as usize);

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // --- Attention block ---
        let normed = tensor::rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps);
        let attn_out = attention_cached(&normed, layer, config, pos, layer_idx, cache);
        let after_attn = tensor::add(&hidden, &attn_out);

        // --- FFN block ---
        let normed_ffn = tensor::rms_norm(&after_attn, &layer.ffn_norm, config.rms_norm_eps);
        let ffn_out = feed_forward(&normed_ffn, layer);
        hidden = tensor::add(&after_attn, &ffn_out);
    }

    // All layers have written this position — advance the cache
    cache.advance();

    // 3. Final norm + LM head
    let normed = tensor::rms_norm(&hidden, &weights.output_norm, config.rms_norm_eps);
    weights.output.matvec(normed.data())
}

/// Multi-head self-attention for one position, reading past K/V from cache.
fn attention_cached(
    x: &Tensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    pos: usize,
    layer_idx: usize,
    cache: &mut KvCache,
) -> Tensor {
    let embed_dim = config.embedding_length as usize;
    let n_heads = config.head_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;
    let kv_group_size = n_heads / n_kv_heads;

    // Q/K/V projections for current token only
    let q_all = layer.attn_q.matvec(x.data());
    let k_cur = layer.attn_k.matvec(x.data());
    let v_cur = layer.attn_v.matvec(x.data());

    // Apply RoPE to Q and K
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base);

    // Write current K/V into cache (before reading, so pos is included in the loop)
    cache.write(layer_idx, pos, k_cur.data(), v_cur.data());

    let seq_len = pos + 1; // attend to positions 0..=pos
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut attn_output = vec![0.0f32; embed_dim];

    for h in 0..n_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q_all.data()[q_offset..q_offset + head_dim];

        // Score against all cached K vectors (including current)
        let mut scores = vec![0.0f32; seq_len];
        for (s, score) in scores.iter_mut().enumerate() {
            let k_row = cache.k_at(layer_idx, s);
            let k_head = &k_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            *score = dot * scale;
        }

        // Softmax over scores
        let scores_tensor = Tensor::from_vec(scores, &[seq_len]);
        let attn_weights = tensor::softmax(&scores_tensor);

        // Weighted sum of cached V vectors
        for s in 0..seq_len {
            let w = attn_weights.get_flat(s);
            let v_row = cache.v_at(layer_idx, s);
            let v_head = &v_row[kv_h * head_dim..(kv_h + 1) * head_dim];
            for d in 0..head_dim {
                attn_output[q_offset + d] += w * v_head[d];
            }
        }
    }

    let attn_vec = Tensor::from_vec(attn_output, &[embed_dim]);
    layer.attn_output.matvec(attn_vec.data())
}

/// Greedy decoding with KV cache.
///
/// Two phases:
/// 1. **Prefill** — process each prompt token, populating the cache.
/// 2. **Decode** — generate new tokens one at a time using cached K/V.
pub fn generate_greedy_cached(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> Vec<u32> {
    let max_seq_len = config.context_length as usize;
    let n_layers = config.block_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;

    let mut cache = KvCache::new(n_layers, max_seq_len, n_kv_heads, head_dim);
    let mut tokens = prompt_tokens.to_vec();

    // Phase 1: Prefill — process all prompt tokens
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = Tensor::zeros(&[1]); // placeholder
    for (i, &token_id) in prompt_tokens.iter().enumerate() {
        eprint!("\r  Prefill token {}/{}...", i + 1, prompt_tokens.len());
        last_logits = forward_one(weights, config, token_id, i, &mut cache);
    }
    eprintln!();

    // Phase 2: Decode — generate new tokens
    for step in 0..max_new_tokens {
        // Argmax over logits — total_cmp handles NaN safely (NaN sorts last)
        let next_token = last_logits
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        eprintln!("  Decode step {}/{}: token {next_token}", step + 1, max_new_tokens);
        tokens.push(next_token);

        if next_token == 2 {
            break;
        }

        let pos = tokens.len() - 1;
        last_logits = forward_one(weights, config, next_token, pos, &mut cache);
    }

    tokens
}

/// Token generation with configurable sampling and a per-token callback.
///
/// Like `generate_cached`, but calls `on_token(id)` for every generated token.
/// If `on_token` returns `false`, generation stops early — used by the HTTP
/// server to detect client disconnects.
///
/// Two phases:
/// 1. **Prefill** — process each prompt token, populating the cache.
/// 2. **Decode** — generate new tokens; call `on_token` after each one.
pub fn generate_streaming(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
    on_token: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let max_seq_len = config.context_length as usize;
    let n_layers = config.block_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;

    let mut cache = KvCache::new(n_layers, max_seq_len, n_kv_heads, head_dim);
    let mut tokens = prompt_tokens.to_vec();

    // Prefill
    let mut last_logits = Tensor::zeros(&[1]);
    for (i, &token_id) in prompt_tokens.iter().enumerate() {
        last_logits = forward_one(weights, config, token_id, i, &mut cache);
    }

    // Decode
    for _ in 0..max_new_tokens {
        let next_token = sampler.sample(last_logits.data(), &tokens);
        tokens.push(next_token);

        let keep_going = on_token(next_token);
        if !keep_going || next_token == eos_token {
            break;
        }

        let pos = tokens.len() - 1;
        last_logits = forward_one(weights, config, next_token, pos, &mut cache);
    }

    tokens
}

/// Token generation with configurable sampling.
///
/// Like `generate_greedy_cached`, but uses a `Sampler` to select each token
/// instead of hardcoded argmax. This enables temperature, top-k, top-p,
/// repetition penalty, and other sampling strategies.
///
/// Two phases:
/// 1. **Prefill** — process each prompt token, populating the cache.
/// 2. **Decode** — generate new tokens using the sampler.
pub fn generate_cached(
    weights: &TransformerWeights,
    config: &ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampler: &mut Sampler,
    eos_token: u32,
) -> Vec<u32> {
    let max_seq_len = config.context_length as usize;
    let n_layers = config.block_count as usize;
    let n_kv_heads = config.head_count_kv as usize;
    let head_dim = config.head_dim() as usize;

    let mut cache = KvCache::new(n_layers, max_seq_len, n_kv_heads, head_dim);
    let mut tokens = prompt_tokens.to_vec();

    // Phase 1: Prefill — process all prompt tokens
    eprintln!("Prefill: {} tokens", prompt_tokens.len());
    let mut last_logits = Tensor::zeros(&[1]); // placeholder
    for (i, &token_id) in prompt_tokens.iter().enumerate() {
        eprint!("\r  Prefill token {}/{}...", i + 1, prompt_tokens.len());
        last_logits = forward_one(weights, config, token_id, i, &mut cache);
    }
    eprintln!();

    // Phase 2: Decode — generate new tokens using the sampler
    for step in 0..max_new_tokens {
        let next_token = sampler.sample(last_logits.data(), &tokens);

        eprintln!("  Decode step {}/{}: token {next_token}", step + 1, max_new_tokens);
        tokens.push(next_token);

        if next_token == eos_token {
            break;
        }

        let pos = tokens.len() - 1;
        last_logits = forward_one(weights, config, next_token, pos, &mut cache);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::tensor::QuantizedTensor;

    fn make_tiny_weights() -> (TransformerWeights, ModelConfig) {
        // Tiny model: embed_dim=4, 1 layer, 2 heads, 1 kv head, head_dim=2
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
        };
        // Use QuantizedTensor::from_f32 which encodes as F32 bytes — exact round-trip.
        let weights = TransformerWeights {
            token_embedding: QuantizedTensor::from_f32(
                &(0..32).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
                8, 4,
            ),
            layers: vec![LayerWeights {
                attn_norm: Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]),
                ffn_norm:  Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]),
                attn_q: QuantizedTensor::from_f32(
                    &(0..16).map(|i| (i as f32) * 0.05 - 0.4).collect::<Vec<_>>(),
                    4, 4,
                ),
                attn_k: QuantizedTensor::from_f32(
                    &(0..8).map(|i| (i as f32) * 0.1 - 0.3).collect::<Vec<_>>(),
                    2, 4,
                ),
                attn_v: QuantizedTensor::from_f32(
                    &(0..8).map(|i| (i as f32) * 0.07 - 0.2).collect::<Vec<_>>(),
                    2, 4,
                ),
                attn_output: QuantizedTensor::from_f32(
                    &(0..16).map(|i| (i as f32) * 0.03 - 0.2).collect::<Vec<_>>(),
                    4, 4,
                ),
                ffn_gate: QuantizedTensor::from_f32(
                    &(0..32).map(|i| (i as f32) * 0.02 - 0.3).collect::<Vec<_>>(),
                    8, 4,
                ),
                ffn_up: QuantizedTensor::from_f32(
                    &(0..32).map(|i| (i as f32) * 0.015 - 0.2).collect::<Vec<_>>(),
                    8, 4,
                ),
                ffn_down: QuantizedTensor::from_f32(
                    &(0..32).map(|i| (i as f32) * 0.025 - 0.4).collect::<Vec<_>>(),
                    4, 8,
                ),
            }],
            output_norm: Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]),
            output: QuantizedTensor::from_f32(
                &(0..32).map(|i| (i as f32) * 0.1 - 1.6).collect::<Vec<_>>(),
                8, 4,
            ),
        };
        (weights, config)
    }

    #[test]
    fn test_cached_matches_uncached() {
        let (weights, config) = make_tiny_weights();
        let token_ids: Vec<u32> = vec![1, 3, 5];

        // Uncached: full forward pass
        let logits_uncached = forward(&weights, &config, &token_ids);

        // Cached: process one token at a time
        let mut cache = KvCache::new(
            config.block_count as usize,
            config.context_length as usize,
            config.head_count_kv as usize,
            config.head_dim() as usize,
        );
        let mut logits_cached = Tensor::zeros(&[1]);
        for (pos, &tok) in token_ids.iter().enumerate() {
            logits_cached = forward_one(&weights, &config, tok, pos, &mut cache);
        }

        // Compare final logits
        assert_eq!(logits_uncached.shape(), logits_cached.shape());
        for (i, (&a, &b)) in logits_uncached.data().iter().zip(logits_cached.data()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "Logit mismatch at index {i}: uncached={a}, cached={b}, diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_feed_forward_shape() {
        // Minimal FFN: embed_dim=4, ffn_hidden=8
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
        let result = feed_forward(&x, &layer);
        assert_eq!(result.shape(), &[4]);
    }
}
