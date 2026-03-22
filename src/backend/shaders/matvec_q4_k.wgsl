// Q4_K quantized matrix-vector multiply.
//
// Super-block layout (144 bytes per 256 elements):
//   [f16 d (2B)] [f16 dmin (2B)] [scales u8×12] [qs u8×128]
//
// 8 sub-blocks of 32 elements. Sub-block scale/min pairs are packed 6-bit
// values in the 12 `scales` bytes (decoded via get_scale_min below).
// The 128 qs bytes hold 256 nibbles: low nibbles → even sub-blocks (0,2,4,6),
// high nibbles → odd sub-blocks (1,3,5,7).
//
// Output[row] = Σ_superblock  Σ_element  (d*sc*nibble - dmin*mn) * input[element]
//
// Each workgroup computes one output row; threads cooperate over super-blocks.

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

const SUPER_BLOCK_ELEMS: u32 = 256u;
const SUPER_BLOCK_BYTES: u32 = 144u;  // 4 + 12 + 128 = 144, also 36×4 so always 4-aligned
const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

// Extract byte at global byte-index `bi` from the weight storage buffer.
fn load_byte(bi: u32) -> u32 {
    return (weights[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xFFu;
}

// Decode (scale, min) for sub-block `j` (0..8) from the 12-byte scales region
// starting at byte offset `sc_base` in the weight buffer.
// Mirrors get_scale_min_k4 from llama.cpp / ggml-quants.c.
fn get_scale_min(j: u32, sc_base: u32) -> vec2<u32> {
    if (j < 4u) {
        let sc = load_byte(sc_base + j) & 63u;
        let mn = load_byte(sc_base + j + 4u) & 63u;
        return vec2<u32>(sc, mn);
    } else {
        let sc = (load_byte(sc_base + j + 4u) & 0x0Fu)
               | ((load_byte(sc_base + j - 4u) >> 6u) << 4u);
        let mn = (load_byte(sc_base + j + 4u) >> 4u)
               | ((load_byte(sc_base + j    ) >> 6u) << 4u);
        return vec2<u32>(sc, mn);
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let n_blocks = params.cols / SUPER_BLOCK_ELEMS;
    let bytes_per_row = n_blocks * SUPER_BLOCK_BYTES;
    let row_byte_offset = row * bytes_per_row;

    var partial_sum: f32 = 0.0;

    for (var b = tid; b < n_blocks; b += WG_SIZE) {
        let bb = row_byte_offset + b * SUPER_BLOCK_BYTES;

        // d and dmin are at bb+0 and bb+2 as f16 LE.
        // SUPER_BLOCK_BYTES=144=36×4 so bb is always 4-aligned.
        let dm   = unpack2x16float(weights[bb >> 2u]);
        let d    = dm.x;
        let dmin = dm.y;

        let sc_base  = bb + 4u;   // 12-byte scales region
        let qs_base  = bb + 16u;  // 128-byte nibbles region
        let elem_base = b * SUPER_BLOCK_ELEMS;

        // 4 groups × 2 sub-blocks = 8 sub-blocks × 32 elements = 256 elements
        for (var group = 0u; group < 4u; group += 1u) {
            let sm0 = get_scale_min(group * 2u,      sc_base);
            let sm1 = get_scale_min(group * 2u + 1u, sc_base);

            let d0 = d    * f32(sm0.x);   // effective scale for even sub-block
            let m0 = dmin * f32(sm0.y);   // effective min   for even sub-block
            let d1 = d    * f32(sm1.x);
            let m1 = dmin * f32(sm1.y);

            let q_off = qs_base + group * 32u;     // byte start of this group's nibbles
            let e_off = elem_base + group * 64u;   // input element index for this group

            for (var l = 0u; l < 32u; l += 1u) {
                let q_byte = load_byte(q_off + l);
                let lo = f32(q_byte & 0x0Fu);       // low  nibble → even sub-block
                let hi = f32(q_byte >> 4u);          // high nibble → odd  sub-block

                // even sub-block: elements at e_off + l
                partial_sum += (d0 * lo - m0) * input_vec[e_off + l];
                // odd sub-block: elements at e_off + 32 + l
                partial_sum += (d1 * hi - m1) * input_vec[e_off + 32u + l];
            }
        }
    }

    // Parallel reduction across the workgroup
    shared_sums[tid] = partial_sum;
    workgroupBarrier();

    var stride = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if (tid == 0u) {
        output_vec[row] = shared_sums[0];
    }
}
