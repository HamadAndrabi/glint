// Softmax: output[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//
// Three-pass workgroup reduction: max, exp-sum, normalize.
// Designed for a single workgroup processing one vector.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> wg_buf: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Pass 1: find max
    var local_max: f32 = -1e38;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        local_max = max(local_max, input[i]);
    }
    wg_buf[tid] = local_max;
    workgroupBarrier();

    var stride = WG_SIZE >> 1u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_buf[tid] = max(wg_buf[tid], wg_buf[tid + stride]);
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let max_val = wg_buf[0];
    workgroupBarrier();

    // Pass 2: compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        let e = exp(input[i] - max_val);
        output[i] = e;
        local_sum += e;
    }
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    stride = WG_SIZE >> 1u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_buf[tid] += wg_buf[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let total_sum = wg_buf[0];
    workgroupBarrier();

    // Pass 3: normalize
    let inv_sum = 1.0 / total_sum;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        output[i] = output[i] * inv_sum;
    }
}
