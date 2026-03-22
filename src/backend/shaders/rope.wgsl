// Rotary Position Embedding (RoPE).
//
// For each head, rotate pairs of elements by position-dependent angles:
//   x[2k]'   =  x[2k] * cos(θ) - x[2k+1] * sin(θ)
//   x[2k+1]' =  x[2k] * sin(θ) + x[2k+1] * cos(θ)
// where θ = pos / (freq_base ^ (2k / head_dim))

struct Params {
    n: u32,           // total number of elements
    pos: u32,         // token position
    head_dim: u32,    // dimension per head
    freq_base: f32,   // typically 10000.0
    rope_scale: f32,  // scaling factor (usually 1.0)
    rot_dim: u32,     // how many dims per head to rotate
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x; // index of the pair within the whole vector
    let n_pairs = params.n / 2u;
    if pair_idx >= n_pairs { return; }

    let i = pair_idx * 2u;
    let dim_in_head = i % params.head_dim;

    // Only rotate the first rot_dim dimensions of each head
    if dim_in_head >= params.rot_dim {
        output[i]     = input[i];
        output[i + 1u] = input[i + 1u];
        return;
    }

    let half_dim = dim_in_head / 2u;
    let pos_f = f32(params.pos) / params.rope_scale;
    let freq = pos_f / pow(params.freq_base, f32(dim_in_head) / f32(params.head_dim));
    let cos_f = cos(freq);
    let sin_f = sin(freq);

    let x0 = input[i];
    let x1 = input[i + 1u];
    output[i]     = x0 * cos_f - x1 * sin_f;
    output[i + 1u] = x0 * sin_f + x1 * cos_f;
}
