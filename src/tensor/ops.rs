//! Tensor math operations — naive f32 implementations.
//!
//! All operations are correctness-first. SIMD/threading optimizations come in Phase 2.

use super::tensor::Tensor;

/// Matrix multiplication: `[M, K] × [K, N] → [M, N]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul requires 2D tensors");
    assert_eq!(b.ndim(), 2, "matmul requires 2D tensors");
    let m = a.shape()[0];
    let k = a.shape()[1];
    assert_eq!(
        k,
        b.shape()[0],
        "Inner dimensions must match: {} vs {}",
        k,
        b.shape()[0]
    );
    let n = b.shape()[1];

    let mut out = Tensor::zeros(&[m, n]);
    let out_data = out.data_mut();
    let a_data = a.data();
    let b_data = b.data();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
    out
}

/// Matrix-vector multiplication: `[M, K] × [K] → [M]`.
pub fn matvec(mat: &Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(mat.ndim(), 2);
    assert_eq!(vec.ndim(), 1);
    let m = mat.shape()[0];
    let k = mat.shape()[1];
    assert_eq!(k, vec.shape()[0]);

    let mut out = Tensor::zeros(&[m]);
    let out_data = out.data_mut();
    let mat_data = mat.data();
    let vec_data = vec.data();

    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += mat_data[i * k + j] * vec_data[j];
        }
        out_data[i] = sum;
    }
    out
}

/// Element-wise addition. Shapes must match.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for add");
    let data: Vec<f32> = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    Tensor::from_vec(data, a.shape())
}

/// Element-wise addition in place: `a[i] += b[i]`.
pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Lengths must match for add_in_place");
    for (dst, &src) in a.iter_mut().zip(b) {
        *dst += src;
    }
}

/// Element-wise multiplication. Shapes must match.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for mul");
    let data: Vec<f32> = a.data().iter().zip(b.data()).map(|(x, y)| x * y).collect();
    Tensor::from_vec(data, a.shape())
}

/// Logit soft-capping: `cap * tanh(x / cap)`.
///
/// Used in Gemma 2 for attention logits and final logits to bound logit dynamics.
pub fn logit_softcap(x: &Tensor, cap: f32) -> Tensor {
    let inv_cap = 1.0 / cap;
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| cap * (v * inv_cap).tanh())
        .collect();
    Tensor::from_vec(data, x.shape())
}

/// Logit soft-capping in-place: `x[i] = cap * tanh(x[i] / cap)`.
pub fn logit_softcap_in_place(x: &mut [f32], cap: f32) {
    let inv_cap = 1.0 / cap;
    for v in x.iter_mut() {
        *v = cap * (*v * inv_cap).tanh();
    }
}

/// RMSNorm: `x / sqrt(mean(x²) + eps) * weight`.
///
/// Normalizes the input vector by its root-mean-square, then scales by a
/// learned weight vector. Used in LLaMA instead of LayerNorm.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.ndim(), 1);
    assert_eq!(weight.ndim(), 1);
    assert_eq!(x.shape()[0], weight.shape()[0]);

    let x_data = x.data();
    let w_data = weight.data();
    let n = x_data.len();

    // mean(x²)
    let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();

    let data: Vec<f32> = x_data
        .iter()
        .zip(w_data)
        .map(|(&x, &w)| x * rsqrt * w)
        .collect();
    Tensor::from_vec(data, x.shape())
}

/// Gemma RMSNorm: `x / sqrt(mean(x²) + eps) * (1.0 + weight)`.
///
/// In Gemma models, the learned scale weights are stored as offsets (Δw) centered at 0,
/// so the scaling factor is `(1.0 + weight)`.
pub fn rms_norm_gemma(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.ndim(), 1);
    assert_eq!(weight.ndim(), 1);
    assert_eq!(x.shape()[0], weight.shape()[0]);

    let x_data = x.data();
    let w_data = weight.data();
    let n = x_data.len();

    let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();

    let data: Vec<f32> = x_data
        .iter()
        .zip(w_data)
        .map(|(&x, &w)| x * rsqrt * (1.0 + w))
        .collect();
    Tensor::from_vec(data, x.shape())
}

/// SiLU activation: `x * sigmoid(x)`.
///
/// Also called "swish". Used in LLaMA's feed-forward network (SwiGLU variant).
pub fn silu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect();
    Tensor::from_vec(data, x.shape())
}

/// GeLU activation (tanh approximation): `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// Used in Gemma / Gemma 2 feed-forward networks (GeGLU variant).
pub fn gelu(x: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2.0 / PI)
    const COEF: f32 = 0.044715;
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| 0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + COEF * v * v * v)).tanh()))
        .collect();
    Tensor::from_vec(data, x.shape())
}

/// Softmax along a 1D tensor: `exp(x - max(x)) / sum(exp(x - max(x)))`.
///
/// The `max(x)` subtraction prevents overflow in exp().
pub fn softmax(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 1);
    let x_data = x.data();

    let max_val = x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x_data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let data: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
    Tensor::from_vec(data, x.shape())
}

/// Rotary Positional Embeddings (RoPE) — apply in-place to q and k vectors.
///
/// Rotates pairs of dimensions by position-dependent angles. This encodes
/// positional information directly into the attention computation without
/// needing additive position embeddings.
///
/// For each pair `(x0, x1)` at dimension index `i`:
///   freq = 1 / (base ^ (2i / head_dim))
///   angle = (pos / scaling_factor) * freq
///   x0' = x0 * cos(angle) - x1 * sin(angle)
///   x1' = x0 * sin(angle) + x1 * cos(angle)
///
/// # Arguments
/// * `scaling_factor` — linear RoPE scaling for extended-context models (Phi-3, Qwen2-long).
///   Use `1.0` for standard RoPE. Values >1 extend the effective context window.
/// * `rot_dim` — number of head dimensions to rotate. Use `head_dim` for full RoPE.
///   Phi-3 uses `partial_rotary_factor = 0.5`, so `rot_dim = head_dim / 2`; the
///   remaining dimensions are left unchanged.
pub fn rope(
    x: &Tensor,
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    scaling_factor: f32,
    rot_dim: usize,
) -> Tensor {
    assert_eq!(x.ndim(), 1);
    let data = x.data();
    // Start from a copy so dimensions outside rot_dim are already correct.
    let mut out = data.to_vec();

    let pos_scaled = pos as f32 / scaling_factor;
    let rot = (rot_dim.min(head_dim)) & !1;

    for i in (0..rot).step_by(2) {
        let freq = 1.0 / freq_base.powf(i as f32 / head_dim as f32);
        let angle = pos_scaled * freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        // Apply rotation to each head's pair at position i
        let mut offset = 0;
        while offset + head_dim <= data.len() {
            let x0 = data[offset + i];
            let x1 = data[offset + i + 1];
            out[offset + i] = x0 * cos_val - x1 * sin_val;
            out[offset + i + 1] = x0 * sin_val + x1 * cos_val;
            offset += head_dim;
        }
    }
    Tensor::from_vec(out, x.shape())
}

/// Embedding lookup: select rows from a weight matrix by token IDs.
///
/// weight: `[vocab_size, embed_dim]`, token_ids: list of token indices.
/// Returns `[n_tokens, embed_dim]`.
pub fn embedding(weight: &Tensor, token_ids: &[u32]) -> Tensor {
    assert_eq!(weight.ndim(), 2);
    let embed_dim = weight.shape()[1];
    let n_tokens = token_ids.len();

    let mut data = Vec::with_capacity(n_tokens * embed_dim);
    let w_data = weight.data();

    for &id in token_ids {
        let start = id as usize * embed_dim;
        data.extend_from_slice(&w_data[start..start + embed_dim]);
    }
    Tensor::from_vec(data, &[n_tokens, embed_dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "Length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "Mismatch at index {}: {} vs {} (diff {})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_matmul_identity() {
        // Multiplying by identity should return the original
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let result = matmul(&a, &identity);
        approx_eq(result.data(), a.data(), 1e-6);
    }

    #[test]
    fn test_matmul_known() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        approx_eq(c.data(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    #[test]
    fn test_matmul_non_square() {
        // [1,2,3] × [[1],[2],[3]] = [14]  (1×3 × 3×1 = 1×1)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[1, 1]);
        approx_eq(c.data(), &[14.0], 1e-6);
    }

    #[test]
    fn test_matvec() {
        let mat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let v = Tensor::from_vec(vec![1.0, 1.0], &[2]);
        let result = matvec(&mat, &v);
        approx_eq(result.data(), &[3.0, 7.0], 1e-6);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        approx_eq(c.data(), &[5.0, 7.0, 9.0], 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 0
        // mean(x²) = (1+4+9)/3 = 14/3
        // rsqrt = 1/sqrt(14/3) ≈ 0.46291
        // result ≈ [0.46291, 0.92582, 1.38873]
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let w = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]);
        let result = rms_norm(&x, &w, 0.0);
        let expected_rsqrt = 1.0 / (14.0f32 / 3.0).sqrt();
        approx_eq(
            result.data(),
            &[
                1.0 * expected_rsqrt,
                2.0 * expected_rsqrt,
                3.0 * expected_rsqrt,
            ],
            1e-5,
        );
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        // silu(1) = 1 * sigmoid(1) ≈ 0.7310586
        let x = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
        let result = silu(&x);
        let expected = vec![
            0.0,
            1.0 / (1.0 + (-1.0f32).exp()),
            -1.0 / (1.0 + 1.0f32.exp()),
        ];
        approx_eq(result.data(), &expected, 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let result = softmax(&x);

        // Should sum to 1
        let sum: f32 = result.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Last element should be largest
        assert!(result.data()[2] > result.data()[1]);
        assert!(result.data()[1] > result.data()[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values — should not overflow thanks to max subtraction
        let x = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]);
        let result = softmax(&x);
        let sum: f32 = result.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_basic() {
        // At position 0, all angles are 0, so cos=1, sin=0 → no change
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let result = rope(&x, 0, 4, 10000.0, 1.0, 4);
        approx_eq(result.data(), x.data(), 1e-6);
    }

    #[test]
    fn test_rope_partial_rotary() {
        // rot_dim=2 on a 4-dim head: first pair rotated, second pair unchanged.
        // At pos=0 all angles are 0, so rotation is identity — both pairs unchanged.
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let result = rope(&x, 0, 4, 10000.0, 1.0, 2);
        approx_eq(result.data(), x.data(), 1e-6);
    }

    #[test]
    fn test_rope_scaling_factor() {
        // scaling_factor=2 halves the effective position, so pos=2 with scale=2
        // should equal pos=1 with scale=1.
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let scaled = rope(&x, 2, 4, 10000.0, 2.0, 4);
        let unscaled = rope(&x, 1, 4, 10000.0, 1.0, 4);

        // Both calls reduce to a bit-identical angle (2.0/2.0 and 1.0/1.0 are
        // both exactly 1.0), so on real hardware `sin`/`cos` are pure and the
        // difference is exactly zero — keep the strict tolerance there.
        //
        // Miri deliberately returns non-deterministic results for transcendental
        // functions, within an error margin, to catch code that relies on exact
        // libm output. Under it the two calls can differ by a few ULP (~1.2e-6
        // observed), which says nothing about whether `scaling_factor` is
        // applied correctly: any real bug here changes the angle outright and
        // shows up as an O(1) difference, not a rounding-scale one.
        let tol = if cfg!(miri) { 1e-4 } else { 1e-6 };
        approx_eq(scaled.data(), unscaled.data(), tol);
    }

    #[test]
    fn test_embedding() {
        // 4-token vocab, 3-dim embeddings
        let weight = Tensor::from_vec(
            vec![
                0.1, 0.2, 0.3, // token 0
                0.4, 0.5, 0.6, // token 1
                0.7, 0.8, 0.9, // token 2
                1.0, 1.1, 1.2, // token 3
            ],
            &[4, 3],
        );
        let result = embedding(&weight, &[2, 0]);
        assert_eq!(result.shape(), &[2, 3]);
        approx_eq(&result.data()[0..3], &[0.7, 0.8, 0.9], 1e-6);
        approx_eq(&result.data()[3..6], &[0.1, 0.2, 0.3], 1e-6);
    }

    #[test]
    fn test_add_in_place() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        add_in_place(&mut a, &b);
        approx_eq(&a, &[5.0, 7.0, 9.0], 1e-6);
    }

    #[test]
    fn test_logit_softcap() {
        let x = Tensor::from_vec(vec![0.0, 50.0, -50.0, 1000.0, -1000.0], &[5]);
        let capped = logit_softcap(&x, 50.0);
        // 50 * tanh(0) = 0
        // 50 * tanh(1) ≈ 50 * 0.761594156 = 38.0797
        // 50 * tanh(-1) ≈ -38.0797
        // 50 * tanh(20) ≈ 50.0
        // 50 * tanh(-20) ≈ -50.0
        approx_eq(&[capped.data()[0]], &[0.0], 1e-6);
        approx_eq(&[capped.data()[1]], &[50.0 * 1.0f32.tanh()], 1e-6);
        approx_eq(&[capped.data()[2]], &[-50.0 * 1.0f32.tanh()], 1e-6);
        assert!((capped.data()[3] - 50.0).abs() < 1e-5);
        assert!((capped.data()[4] + 50.0).abs() < 1e-5);
    }
}
