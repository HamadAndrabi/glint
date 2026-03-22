// Q8_0 quantized matrix-vector multiply.
//
// Block layout (34 bytes per 32 elements):
//   [f16 scale packed into u32 low halfword] [32 × i8 values packed as u32s]
//
// Each workgroup computes one output row. Threads within a workgroup
// cooperate on blocks, then reduce partial sums via shared memory.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Quantized weight matrix — raw bytes viewed as u32 array
@group(0) @binding(1) var<storage, read> weights: array<u32>;
// Input vector (f32)
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
// Output vector (f32)
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

const BLOCK_ELEMS: u32 = 32u;
// 34 bytes per block = 8.5 u32s → we use 9 u32s (36 bytes) with padding
// Actually: we read raw bytes, so 34 bytes = ceil(34/4) = 9 u32s per block.
// Layout in u32s: [scale_u16_in_low_half | 8 × u32 of i8 quants]
// But since WGSL storage buffers are u32-aligned, the GGUF data has blocks
// packed tightly at 34 bytes. We'll index carefully.

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

// Extract an i8 from a packed byte array addressed as u32s.
// byte_offset is relative to the start of the data array.
fn read_i8(base_u32: u32, byte_offset: u32) -> f32 {
    let word_idx = base_u32 + (byte_offset >> 2u);
    let lane = byte_offset & 3u;
    let word = weights[word_idx];
    let byte_val = (word >> (lane * 8u)) & 0xFFu;
    // Interpret as signed i8: if >= 128, subtract 256
    if byte_val >= 128u {
        return f32(i32(byte_val) - 256);
    }
    return f32(byte_val);
}

// Read an f16 scale stored as 2 bytes at a byte offset.
fn read_f16_scale(base_u32: u32, byte_offset: u32) -> f32 {
    let word_idx = base_u32 + (byte_offset >> 2u);
    let lane = byte_offset & 3u;
    let word = weights[word_idx];
    // Extract 16 bits
    let bits = (word >> (lane * 8u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if row >= params.rows { return; }

    let tid = lid.x;
    let n_blocks = params.cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * 34u;
    // Base byte offset for this row in the weight buffer
    let row_byte_offset = row * bytes_per_row;
    // Convert to u32 base (integer division — we handle sub-word alignment in read fns)
    let row_base_u32 = row_byte_offset >> 2u;
    let row_byte_mod = row_byte_offset & 3u;

    var partial_sum: f32 = 0.0;

    // Each thread processes a subset of blocks
    for (var b = tid; b < n_blocks; b += WG_SIZE) {
        let block_byte = row_byte_offset + b * 34u;
        let block_base_u32 = block_byte >> 2u;
        let block_byte_mod = block_byte & 3u;

        // Read f16 scale (first 2 bytes of block)
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
            // scale spans two u32s
            let w1 = weights[scale_word_idx + 1u];
            scale_bits = ((w0 >> 24u) & 0xFFu) | ((w1 & 0xFFu) << 8u);
        }
        let scale = unpack2x16float(scale_bits).x;

        // Read 32 i8 quantized values (bytes 2..33 of the block)
        var block_sum: f32 = 0.0;
        let quant_byte_start = block_byte + 2u;
        let vec_offset = b * BLOCK_ELEMS;
        for (var j = 0u; j < BLOCK_ELEMS; j += 1u) {
            let qb = quant_byte_start + j;
            let wi = qb >> 2u;
            let lane = qb & 3u;
            let word = weights[wi];
            let byte_val = (word >> (lane * 8u)) & 0xFFu;
            var signed_val: f32;
            if byte_val >= 128u {
                signed_val = f32(i32(byte_val) - 256);
            } else {
                signed_val = f32(byte_val);
            }
            block_sum += signed_val * input_vec[vec_offset + j];
        }
        partial_sum += block_sum * scale;
    }

    shared_sums[tid] = partial_sum;
    workgroupBarrier();

    // Parallel reduction
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
