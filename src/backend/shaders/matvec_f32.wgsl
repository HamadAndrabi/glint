// f32 matrix-vector multiply: output[row] = dot(weights[row], input_vec)
//
// Each workgroup handles one row. Threads cooperate on the dot product,
// then reduce via shared memory.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let row_offset = row * params.cols;

    var partial_sum: f32 = 0.0;
    for (var i = tid; i < params.cols; i += WG_SIZE) {
        partial_sum += weights[row_offset + i] * input_vec[i];
    }

    shared_sums[tid] = partial_sum;
    workgroupBarrier();

    var stride = WG_SIZE >> 1u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        output_vec[row] = shared_sums[0];
    }
}
