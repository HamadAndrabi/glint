// Q4_0 quantized matrix-vector multiply.
//
// Block layout (18 bytes per 32 elements):
//   [f16 scale (2 bytes)] [16 bytes of packed nibbles, 2 per byte]
//
// Nibble values are unsigned 0–15, centered by subtracting 8 → [-8, +7].
// The nibbles are split-plane (matching ggml's dequantize_row_q4_0): the low
// nibble of byte j is element j and its high nibble is element j + 16.
// Each workgroup computes one output row.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 18u;
const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let n_blocks = params.cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;
    let row_byte_offset = row * bytes_per_row;

    var partial_sum: f32 = 0.0;

    for (var b = tid; b < n_blocks; b += WG_SIZE) {
        let block_byte = row_byte_offset + b * BLOCK_BYTES;

        // Read f16 scale
        let scale_word_idx = block_byte >> 2u;
        let scale_lane = block_byte & 3u;
        let w0 = weights[scale_word_idx];
        var scale_bits: u32;
        if scale_lane == 0u {
            scale_bits = w0 & 0xFFFFu;
        } else if scale_lane == 1u {
            scale_bits = (w0 >> 8u) & 0xFFFFu;
        } else if scale_lane == 2u {
            scale_bits = (w0 >> 16u) & 0xFFFFu;
        } else {
            let w1 = weights[scale_word_idx + 1u];
            scale_bits = ((w0 >> 24u) & 0xFFu) | ((w1 & 0xFFu) << 8u);
        }
        let scale = unpack2x16float(scale_bits).x;

        // Read 16 packed nibble bytes → 32 4-bit values
        var block_sum: f32 = 0.0;
        let nibble_byte_start = block_byte + 2u;
        let vec_offset = b * BLOCK_ELEMS;

        for (var j = 0u; j < 16u; j += 1u) {
            let nb = nibble_byte_start + j;
            let wi = nb >> 2u;
            let lane = nb & 3u;
            let word = weights[wi];
            let packed_byte = (word >> (lane * 8u)) & 0xFFu;

            // Low nibble → element j
            let lo = packed_byte & 0xFu;
            let lo_val = f32(i32(lo) - 8);
            block_sum += lo_val * input_vec[vec_offset + j];

            // High nibble → element j + 16
            let hi = (packed_byte >> 4u) & 0xFu;
            let hi_val = f32(i32(hi) - 8);
            block_sum += hi_val * input_vec[vec_offset + j + 16u];
        }
        partial_sum += block_sum * scale;
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
