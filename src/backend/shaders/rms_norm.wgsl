// RMS normalization: out[i] = (x[i] / rms(x)) * weight[i]
// where rms(x) = sqrt(mean(x²) + eps)
//
// Two-pass: first reduce to compute sum-of-squares, then normalize.

struct Params {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> wg_buf: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn reduce_ss(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var sum_sq: f32 = 0.0;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        let v = x[i];
        sum_sq += v * v;
    }
    wg_buf[tid] = sum_sq;
    workgroupBarrier();

    var stride = WG_SIZE >> 1u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_buf[tid] += wg_buf[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Thread 0 computes 1/rms and stores in wg_buf[0]
    if tid == 0u {
        let mean_sq = wg_buf[0] / f32(params.n);
        wg_buf[0] = 1.0 / sqrt(mean_sq + params.eps);
    }
    workgroupBarrier();

    // All threads normalize their elements
    let inv_rms = wg_buf[0];
    for (var i = tid; i < params.n; i += WG_SIZE) {
        output[i] = x[i] * inv_rms * weight[i];
    }
}
