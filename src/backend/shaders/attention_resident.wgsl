// Single-query multi-head attention for the KV-cached decode step
// using GPU-resident KV buffers (GpuKvCache).
//
// This shader is identical to attention.wgsl in logic, but K and V are
// the full GPU-resident sequence buffers covering all layers and positions.
// The extra params `kv_layer_off` and `window_start` address the correct slice.
//
// Bindings:
//   0 — Params uniform
//   1 — Q flat [n_heads * head_dim] f32
//   2 — K resident [n_layers * max_seq_len * n_kv_heads * head_dim] f32
//   3 — V resident (same layout as K)
//   4 — output  [n_heads * head_dim] f32
//
// Memory layout in K/V buffers:
//   [layer 0: max_seq_len * n_kv_heads * head_dim]
//   [layer 1: ...]  ...
//   The shader indexes layer l, position p as:
//     kv_layer_off + (window_start + i) * n_kv_heads * head_dim + kv_h * head_dim

struct Params {
    n_heads:      u32,
    n_kv_heads:   u32,
    head_dim:     u32,
    seq_len:      u32,  // attend_len (number of positions to attend to)
    scale:        f32,
    kv_layer_off: u32,  // offset in floats to this layer's start in the KV buffer
    window_start: u32,  // index of the first position in the attention window
}

@group(0) @binding(0) var<uniform>            params:  Params;
@group(0) @binding(1) var<storage, read>      q_vec:   array<f32>;
@group(0) @binding(2) var<storage, read>      k_mat:   array<f32>;
@group(0) @binding(3) var<storage, read>      v_mat:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out_vec: array<f32>;

const WG_SIZE: u32 = 128u;
const MAX_SEQ: u32 = 4096u;

var<workgroup> scores:  array<f32, MAX_SEQ>;
var<workgroup> partial: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let h   = wid.x;
    let tid = lid.x;

    if (h >= params.n_heads) { return; }

    let kv_group = params.n_heads / params.n_kv_heads;
    let kv_h     = h / kv_group;
    let q_off    = h * params.head_dim;
    let seq_len  = params.seq_len;

    // ── Phase 1: Q · K^T → scores ───────────────────────────────────────────
    for (var i = tid; i < seq_len; i += WG_SIZE) {
        let k_base = params.kv_layer_off
                   + (params.window_start + i) * params.n_kv_heads * params.head_dim
                   + kv_h * params.head_dim;
        var dot = 0.0f;
        for (var d = 0u; d < params.head_dim; d += 1u) {
            dot += q_vec[q_off + d] * k_mat[k_base + d];
        }
        scores[i] = dot * params.scale;
    }
    workgroupBarrier();

    // ── Phase 2: max reduction for numerical stability ───────────────────────
    var local_max = -1e38f;
    for (var i = tid; i < seq_len; i += WG_SIZE) {
        if (scores[i] > local_max) { local_max = scores[i]; }
    }
    partial[tid] = local_max;
    workgroupBarrier();

    var stride = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            if (partial[tid + stride] > partial[tid]) {
                partial[tid] = partial[tid + stride];
            }
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let max_score = partial[0];
    workgroupBarrier();

    // ── Phase 3: exp(score - max) and sum ───────────────────────────────────
    var local_sum = 0.0f;
    for (var i = tid; i < seq_len; i += WG_SIZE) {
        let e = exp(scores[i] - max_score);
        scores[i]  = e;
        local_sum += e;
    }
    partial[tid] = local_sum;
    workgroupBarrier();

    stride = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let inv_total = 1.0f / partial[0];
    workgroupBarrier();

    for (var i = tid; i < seq_len; i += WG_SIZE) {
        scores[i] *= inv_total;
    }
    workgroupBarrier();

    // ── Phase 4: weighted V sum ──────────────────────────────────────────────
    for (var d = tid; d < params.head_dim; d += WG_SIZE) {
        var acc = 0.0f;
        for (var i = 0u; i < seq_len; i += 1u) {
            let v_base = params.kv_layer_off
                       + (params.window_start + i) * params.n_kv_heads * params.head_dim
                       + kv_h * params.head_dim;
            acc += scores[i] * v_mat[v_base + d];
        }
        out_vec[q_off + d] = acc;
    }
}
