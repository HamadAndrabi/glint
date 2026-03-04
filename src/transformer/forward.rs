//! Decoder-only transformer forward pass (LLaMA-style).

use crate::model::config::ModelConfig;
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

    // 1. Token embedding: [n_tokens, embed_dim]
    let embedded = tensor::embedding(&weights.token_embedding, token_ids);

    // Process one token at a time (last position is what we want logits for)
    // For the naive version, we only compute logits for the last token
    // but still need all positions for attention context.
    let embed_dim = config.embedding_length as usize;

    // Start with the full embedded sequence, but we'll only track per-position hidden states
    let mut hidden_states: Vec<Tensor> = (0..n_tokens)
        .map(|t| {
            let start = t * embed_dim;
            Tensor::from_slice(&embedded.data()[start..start + embed_dim], &[embed_dim])
        })
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
    tensor::matvec(&weights.output, &normed)
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
    let q_all = tensor::matvec(&layer.attn_q, x);
    let k_cur = tensor::matvec(&layer.attn_k, x);
    let v_cur = tensor::matvec(&layer.attn_v, x);

    // Apply RoPE to Q and K for current position
    let freq_base = config.rope_freq_base.unwrap_or(10000.0);
    let q_all = tensor::rope(&q_all, pos, head_dim, freq_base);
    let k_cur = tensor::rope(&k_cur, pos, head_dim, freq_base);

    // Compute K and V for all positions we need to attend to (0..=pos)
    // In the no-cache version, we recompute K/V for all previous positions too
    let mut k_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);
    let mut v_cache: Vec<Tensor> = Vec::with_capacity(pos + 1);

    for p in 0..pos {
        let normed_p = tensor::rms_norm(&all_hidden[p], &layer.attn_norm, config.rms_norm_eps);
        let k_p = tensor::matvec(&layer.attn_k, &normed_p);
        let v_p = tensor::matvec(&layer.attn_v, &normed_p);
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
    tensor::matvec(&layer.attn_output, &attn_vec)
}

/// SwiGLU feed-forward network.
///
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
fn feed_forward(x: &Tensor, layer: &LayerWeights) -> Tensor {
    let gate = tensor::matvec(&layer.ffn_gate, x);
    let up = tensor::matvec(&layer.ffn_up, x);
    let activated = tensor::silu(&gate);
    let hidden = tensor::mul(&activated, &up);
    tensor::matvec(&layer.ffn_down, &hidden)
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

        // Argmax
        let next_token = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        eprintln!("  → token {next_token}");
        tokens.push(next_token);

        // Stop if we hit EOS (token ID 2 is common, but model-dependent)
        if next_token == 2 {
            break;
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_shape() {
        // Minimal FFN: embed_dim=4, ffn_hidden=8
        let layer = LayerWeights {
            attn_norm: Tensor::zeros(&[4]),
            ffn_norm: Tensor::zeros(&[4]),
            attn_q: Tensor::zeros(&[4, 4]),
            attn_k: Tensor::zeros(&[4, 4]),
            attn_v: Tensor::zeros(&[4, 4]),
            attn_output: Tensor::zeros(&[4, 4]),
            ffn_gate: Tensor::zeros(&[8, 4]),
            ffn_up: Tensor::zeros(&[8, 4]),
            ffn_down: Tensor::zeros(&[4, 8]),
        };
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let result = feed_forward(&x, &layer);
        assert_eq!(result.shape(), &[4]);
    }
}
