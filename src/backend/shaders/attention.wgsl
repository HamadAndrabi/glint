// Single-query multi-head attention for the KV-cached decode step.
//
// Computes:  output[h] = softmax(Q[h] · K^T * scale) · V
// for every query head h in parallel (one workgroup per head).
//
// Bindings:
//   0 — Params uniform
//   1 — Q flat [n_heads * head_dim] f32
//   2 — K flat [seq_len * n_kv_heads * head_dim] f32  (layout: [pos][kv_h][dim])
//   3 — V flat [seq_len * n_kv_heads * head_dim] f32
//   4 — output  [n_heads * head_dim] f32
//
// GQA (grouped query attention): kv_h = h / kv_group,  kv_group = n_heads / n_kv_heads
//
// Workgroup memory layout (must fit in max_compute_workgroup_storage_size):
//   scores[MAX_SEQ]   — attention scores / softmax weights
//   partial[WG_SIZE]  — per-thread reduction scratch
//
// Total: 4096*4 + 128*4 = 16896 bytes < 32768 (requested limit in GpuBackend::new).
// Graceful limit: seq_len must be ≤ MAX_SEQ; larger sequences fall back to CPU.

struct Params {
    n_heads:    u32,
    n_kv_heads: u32,
    head_dim:   u32,
    seq_len:    u32,
    scale:      f32,
}

@group(0) @binding(0) var<uniform>            params:  Params;
@group(0) @binding(1) var<storage, read>      q_vec:   array<f32>;
@group(0) @binding(2) var<storage, read>      k_mat:   array<f32>;
@group(0) @binding(3) var<storage, read>      v_mat:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out_vec: array<f32>;

const WG_SIZE: u32 = 128u;
const MAX_SEQ: u32 = 4096u;

var<workgroup> scores:  array<f32, MAX_SEQ>;  // 16 384 bytes
var<workgroup> partial: array<f32, WG_SIZE>;  //    512 bytes

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let h   = wid.x;   // query head index
    let tid = lid.x;

    if (h >= params.n_heads) { return; }

    let kv_group = params.n_heads / params.n_kv_heads;
    let kv_h     = h / kv_group;
    let q_off    = h * params.head_dim;
    let seq_len  = params.seq_len;

    // ── Phase 1: Q · K^T → scores ───────────────────────────────────────────
    // Each thread handles a stripe of seq_len positions.
    for (var i = tid; i < seq_len; i += WG_SIZE) {
        let k_base = i * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;
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

    // Normalise scores in-place to softmax weights
    for (var i = tid; i < seq_len; i += WG_SIZE) {
        scores[i] *= inv_total;
    }
    workgroupBarrier();

    // ── Phase 4: weighted V sum ──────────────────────────────────────────────
    // Each thread owns one or more output dimensions.
    for (var d = tid; d < params.head_dim; d += WG_SIZE) {
        var acc = 0.0f;
        for (var i = 0u; i < seq_len; i += 1u) {
            let v_base = i * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;
            acc += scores[i] * v_mat[v_base + d];
        }
        out_vec[q_off + d] = acc;
    }
}
