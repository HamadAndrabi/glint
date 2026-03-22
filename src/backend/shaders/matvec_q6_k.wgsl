// Q6_K quantized matrix-vector multiply.
//
// Super-block layout (210 bytes per 256 elements):
//   [ql u8×128] [qh u8×64] [scales i8×16] [f16 d (2B)]
//
// 16 sub-blocks of 16 elements each. Each element is a 6-bit signed value
// assembled from 4 bits of `ql` and 2 bits of `qh`, centered by subtracting 32.
//
// Assembly for group g (0..2), l (0..32):
//   ql_off = g*64,  qh_off = 128 + g*32,  sc_off = 192 + g*8
//   qhl = qh[qh_off + l]
//   v1 = (ql[ql_off+l   ] & 0x0F) | ((qhl & 0x03) << 4)    → out[g*128 + l]
//   v2 = (ql[ql_off+l+32] & 0x0F) | (((qhl>>2)&0x03) << 4)  → out[g*128 + l+32]
//   v3 = (ql[ql_off+l   ] >>   4) | (((qhl>>4)&0x03) << 4)  → out[g*128 + l+64]
//   v4 = (ql[ql_off+l+32] >>   4) | (((qhl>>6)&0x03) << 4)  → out[g*128 + l+96]
//   sc0..sc6 = scales[sc_off + is + {0,2,4,6}]   where is = l/16
//   value = d * sc * (v - 32)
//
// NOTE: SUPER_BLOCK_BYTES=210 is NOT 4-aligned (210 mod 4 = 2), so all byte
// reads use the load_byte helper which handles arbitrary alignment.
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
const SUPER_BLOCK_BYTES: u32 = 210u;
const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, WG_SIZE>;

fn load_byte(bi: u32) -> u32 {
    return (weights[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xFFu;
}

// Load an i8 stored at byte index `bi`, returned as f32.
// Sign-extend: shift byte into MSB of u32, then arithmetic right-shift back.
fn load_i8_f32(bi: u32) -> f32 {
    let raw = load_byte(bi);
    return f32(i32(raw << 24u) >> 24u);
}

// Load f16 little-endian at byte index `bi` (handles any alignment).
fn load_f16(bi: u32) -> f32 {
    let lo = load_byte(bi);
    let hi = load_byte(bi + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
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

        // d is at bb+208 (f16 LE), may be unaligned — use load_f16
        let d = load_f16(bb + 208u);

        // Two groups of 128 elements each (group = 0 or 1)
        for (var group = 0u; group < 2u; group += 1u) {
            let ql_off = bb + group * 64u;          // low 4-bit data
            let qh_off = bb + 128u + group * 32u;   // high 2-bit data
            let sc_off = bb + 192u + group * 8u;    // i8 sub-block scales
            let e_base = b * SUPER_BLOCK_ELEMS + group * 128u;

            for (var l = 0u; l < 32u; l += 1u) {
                let is = l / 16u;   // which pair of sub-blocks (0 or 1 within the group)

                let ql_a = load_byte(ql_off + l);
                let ql_b = load_byte(ql_off + l + 32u);
                let qhl  = load_byte(qh_off + l);

                // Assemble four 6-bit unsigned values
                let v1 = (ql_a & 0x0Fu) | ((qhl & 0x03u) << 4u);
                let v2 = (ql_b & 0x0Fu) | (((qhl >> 2u) & 0x03u) << 4u);
                let v3 = (ql_a >> 4u)   | (((qhl >> 4u) & 0x03u) << 4u);
                let v4 = (ql_b >> 4u)   | (((qhl >> 6u) & 0x03u) << 4u);

                // Sub-block scales (i8) — four distinct scales for the four output positions
                let sc0 = load_i8_f32(sc_off + is);
                let sc2 = load_i8_f32(sc_off + is + 2u);
                let sc4 = load_i8_f32(sc_off + is + 4u);
                let sc6 = load_i8_f32(sc_off + is + 6u);

                // Dequantized values: d * scale * (v - 32)
                let dq1 = d * sc0 * f32(i32(v1) - 32);
                let dq2 = d * sc2 * f32(i32(v2) - 32);
                let dq3 = d * sc4 * f32(i32(v3) - 32);
                let dq4 = d * sc6 * f32(i32(v4) - 32);

                // Dot product with input vector (elements at scattered positions)
                partial_sum += dq1 * input_vec[e_base + l];
                partial_sum += dq2 * input_vec[e_base + l + 32u];
                partial_sum += dq3 * input_vec[e_base + l + 64u];
                partial_sum += dq4 * input_vec[e_base + l + 96u];
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
