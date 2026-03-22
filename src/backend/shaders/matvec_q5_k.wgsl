// Q5_K quantized matrix-vector multiply.
//
// Super-block layout (176 bytes per 256 elements):
//   [f16 d (2B)] [f16 dmin (2B)] [scales u8×12] [qh u8×32] [qs u8×128]
//
// Same 8 sub-block structure as Q4_K, but each nibble gains a 5th bit from
// the packed high-bit array `qh` (one bit per element, packed 8 per byte).
//
// For group g (0..4), sub-block 2g:
//   bit mask u1 = 1 << (g*2)  — selects high bit for low-nibble elements
//   value = (qs_lo_nibble + u1_bit * 16) → 5-bit unsigned [0..31]
//   dequant = d*sc * value - dmin*mn
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
const SUPER_BLOCK_BYTES: u32 = 176u;  // 4 + 12 + 32 + 128 = 176, also 44×4 so always 4-aligned
const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

fn load_byte(bi: u32) -> u32 {
    return (weights[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xFFu;
}

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

        // SUPER_BLOCK_BYTES=176=44×4, so bb is always 4-aligned.
        let dm   = unpack2x16float(weights[bb >> 2u]);
        let d    = dm.x;
        let dmin = dm.y;

        let sc_base  = bb + 4u;   // 12 bytes: scale/min pairs
        let qh_base  = bb + 16u;  // 32 bytes: high bits (one bit per element, 8 packed per byte)
        let qs_base  = bb + 48u;  // 128 bytes: low nibbles
        let elem_base = b * SUPER_BLOCK_ELEMS;

        for (var group = 0u; group < 4u; group += 1u) {
            let sm0 = get_scale_min(group * 2u,      sc_base);
            let sm1 = get_scale_min(group * 2u + 1u, sc_base);

            let d0 = d    * f32(sm0.x);
            let m0 = dmin * f32(sm0.y);
            let d1 = d    * f32(sm1.x);
            let m1 = dmin * f32(sm1.y);

            // High-bit masks for this group (shift 2 bits left per group)
            let u1 = 1u << (group * 2u);   // bit for low-nibble  elements
            let u2 = 2u << (group * 2u);   // bit for high-nibble elements

            let q_off = qs_base + group * 32u;
            let e_off = elem_base + group * 64u;

            for (var l = 0u; l < 32u; l += 1u) {
                let q_byte  = load_byte(q_off  + l);
                let qh_byte = load_byte(qh_base + l);  // qh is indexed by l (0..32), same for all groups

                let lo4 = q_byte & 0x0Fu;
                let hi4 = q_byte >> 4u;

                // Add the 5th bit from qh if the corresponding bit is set
                let hi_bit0 = select(0u, 16u, (qh_byte & u1) != 0u);
                let hi_bit1 = select(0u, 16u, (qh_byte & u2) != 0u);

                let val0 = f32(lo4 + hi_bit0);   // 5-bit value for even sub-block
                let val1 = f32(hi4 + hi_bit1);   // 5-bit value for odd  sub-block

                partial_sum += (d0 * val0 - m0) * input_vec[e_off + l];
                partial_sum += (d1 * val1 - m1) * input_vec[e_off + 32u + l];
            }
        }
    }

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
