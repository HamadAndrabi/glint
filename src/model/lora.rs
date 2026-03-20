//! LoRA (Low-Rank Adaptation) weight loading and application.
//!
//! Loads adapter weights from a GGUF adapter file and applies the update
//! `ΔW·x = scale * B @ (A @ x)` after each base projection matvec.
//!
//! Standard GGUF LoRA tensor naming convention:
//!   `blk.{i}.attn_q.weight.lora_a`  — shape [rank, in_dim]
//!   `blk.{i}.attn_q.weight.lora_b`  — shape [out_dim, rank]
//!
//! The scaling factor is `alpha / rank`.  `adapter.lora.alpha` (f32) is read
//! from GGUF metadata; if absent the scale defaults to 1.0 (alpha == rank).

use std::collections::HashMap;

use crate::model::gguf::GgufModel;
use crate::tensor::{load_tensor_f32, Tensor};

/// One low-rank adapter pair for a single projection matrix.
///
/// Applies `out += scale * B @ (A @ x)` in place.
pub struct LoraAdapter {
    /// A matrix, shape [rank, in_dim].
    pub a: Tensor,
    /// B matrix, shape [out_dim, rank].
    pub b: Tensor,
    /// Pre-computed `alpha / rank`.
    pub scale: f32,
}

impl LoraAdapter {
    /// Add `scale * B @ (A @ x)` into `out`.
    ///
    /// `x.len()` must equal `in_dim`; `out.len()` must equal `out_dim`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        let rank    = self.a.shape()[0];
        let in_dim  = self.a.shape()[1];
        let out_dim = self.b.shape()[0];
        debug_assert_eq!(x.len(),   in_dim,  "lora A in_dim mismatch");
        debug_assert_eq!(out.len(), out_dim, "lora B out_dim mismatch");

        let a_data = self.a.data();
        let b_data = self.b.data();

        // tmp = A @ x  →  [rank]
        let mut tmp = vec![0.0f32; rank];
        for r in 0..rank {
            let row = &a_data[r * in_dim..(r + 1) * in_dim];
            tmp[r] = row.iter().zip(x).map(|(a, xi)| a * xi).sum();
        }

        // out += scale * B @ tmp
        for o in 0..out_dim {
            let row   = &b_data[o * rank..(o + 1) * rank];
            let delta: f32 = row.iter().zip(&tmp).map(|(b, t)| b * t).sum();
            out[o] += self.scale * delta;
        }
    }
}

/// All LoRA adapters for a single transformer layer.
///
/// Each field is `None` when the adapter file does not target that projection.
#[derive(Default)]
pub struct LoraLayerAdapters {
    pub attn_q:      Option<LoraAdapter>,
    pub attn_k:      Option<LoraAdapter>,
    pub attn_v:      Option<LoraAdapter>,
    pub attn_output: Option<LoraAdapter>,
    pub ffn_gate:    Option<LoraAdapter>,
    pub ffn_up:      Option<LoraAdapter>,
    pub ffn_down:    Option<LoraAdapter>,
}

/// All LoRA adapters loaded from a GGUF adapter file.
pub struct LoraWeights {
    /// One entry per transformer layer (same length as `TransformerWeights::layers`).
    pub layers: Vec<LoraLayerAdapters>,
}

impl LoraWeights {
    /// Load LoRA adapters from a GGUF model.
    ///
    /// Returns `None` if the file contains no `lora_a`/`lora_b` tensors.
    pub fn load(model: &GgufModel, n_layers: usize) -> Option<Self> {
        // Optional alpha from metadata — 0 means "use rank" (scale = 1.0).
        let alpha: f32 = model.metadata
            .get("adapter.lora.alpha")
            .and_then(|v| v.as_f32())
            .unwrap_or(0.0);

        // Collect A and B tensors keyed by their base weight name.
        let mut a_map: HashMap<String, Tensor> = HashMap::new();
        let mut b_map: HashMap<String, Tensor> = HashMap::new();

        for info in &model.tensor_infos {
            if let Some(base) = info.name.strip_suffix(".lora_a") {
                if let Ok(t) = load_tensor_f32(model, &info.name) {
                    a_map.insert(base.to_string(), t);
                }
            } else if let Some(base) = info.name.strip_suffix(".lora_b") {
                if let Ok(t) = load_tensor_f32(model, &info.name) {
                    b_map.insert(base.to_string(), t);
                }
            }
        }

        if a_map.is_empty() {
            return None;
        }

        let mut layers: Vec<LoraLayerAdapters> = (0..n_layers)
            .map(|_| LoraLayerAdapters::default())
            .collect();

        for (base, a) in a_map {
            let Some(b) = b_map.remove(&base) else { continue };
            let rank  = a.shape()[0];
            let scale = if alpha > 0.0 { alpha / rank as f32 } else { 1.0 };
            let adapter = LoraAdapter { a, b, scale };

            // Parse "blk.{i}.{proj}.weight" or "blk.{i}.{proj}"
            let parts: Vec<&str> = base.split('.').collect();
            if parts.len() < 3 || parts[0] != "blk" { continue }
            let Ok(layer_idx) = parts[1].parse::<usize>() else { continue };
            if layer_idx >= n_layers { continue }

            let proj = parts[2..].join(".");
            let ll   = &mut layers[layer_idx];
            match proj.as_str() {
                "attn_q.weight"      | "attn_q"      => ll.attn_q      = Some(adapter),
                "attn_k.weight"      | "attn_k"      => ll.attn_k      = Some(adapter),
                "attn_v.weight"      | "attn_v"      => ll.attn_v      = Some(adapter),
                "attn_output.weight" | "attn_output" => ll.attn_output = Some(adapter),
                "ffn_gate.weight"    | "ffn_gate"    => ll.ffn_gate    = Some(adapter),
                "ffn_up.weight"      | "ffn_up"      => ll.ffn_up      = Some(adapter),
                "ffn_down.weight"    | "ffn_down"    => ll.ffn_down    = Some(adapter),
                _ => {}
            }
        }

        Some(LoraWeights { layers })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(a_data: Vec<f32>, a_shape: &[usize], b_data: Vec<f32>, b_shape: &[usize], scale: f32) -> LoraAdapter {
        LoraAdapter {
            a: Tensor::from_vec(a_data, a_shape),
            b: Tensor::from_vec(b_data, b_shape),
            scale,
        }
    }

    #[test]
    fn test_lora_apply_identity() {
        // rank=1, in_dim=2, out_dim=2
        // A = [[1,0]], B = [[1],[0]], scale=1.0  =>  B@A = [[1,0],[0,0]]
        // x=[3,4] => A@x=[3], B@[3]=[3,0]
        let ad = adapter(vec![1.0, 0.0], &[1, 2], vec![1.0, 0.0], &[2, 1], 1.0);
        let x  = [3.0f32, 4.0];
        let mut out = [0.0f32, 0.0];
        ad.apply(&x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-6, "out[1]={}", out[1]);
    }

    #[test]
    fn test_lora_apply_scale() {
        // rank=1, in_dim=1, out_dim=1 — A=[[1]], B=[[1]], scale=0.5
        // apply(x=[2]) => scale * B@(A@x) = 0.5 * 2 = 1
        let ad = adapter(vec![1.0], &[1, 1], vec![1.0], &[1, 1], 0.5);
        let x = [2.0f32];
        let mut out = [5.0f32]; // starts at 5, should become 6
        ad.apply(&x, &mut out);
        assert!((out[0] - 6.0).abs() < 1e-6, "out[0]={}", out[0]);
    }

    #[test]
    fn test_lora_apply_rank2() {
        // rank=2, in_dim=2, out_dim=2
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]], scale=1 => identity
        let ad = adapter(
            vec![1.0, 0.0,  0.0, 1.0], &[2, 2],
            vec![1.0, 0.0,  0.0, 1.0], &[2, 2],
            1.0,
        );
        let x = [3.0f32, 7.0];
        let mut out = [0.0f32; 2];
        ad.apply(&x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-5, "out[1]={}", out[1]);
    }
}
