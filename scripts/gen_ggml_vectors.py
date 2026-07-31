#!/usr/bin/env python3
"""Generate golden dequantization vectors from the ggml reference algorithms.

Each dequantize_* below is a line-by-line transcription of the corresponding
dequantize_row_* function in llama.cpp's ggml/src/ggml-quants.c (fetched
2026-07-04), together with the block byte layouts from ggml-common.h:

  block_q2_K : scales[16] qs[64]  d(f16) dmin(f16)                  =  84 B
  block_q3_K : hmask[32]  qs[64]  scales[12] d(f16)                 = 110 B
  block_q4_K : d(f16) dmin(f16) scales[12] qs[128]                  = 144 B
  block_q5_K : d(f16) dmin(f16) scales[12] qh[32] qs[128]           = 176 B
  block_q6_K : ql[128] qh[64] scales[16:int8] d(f16)                = 210 B
  block_q4_0 : d(f16) qs[16]                                        =  18 B
  block_q5_0 : d(f16) qh[4] qs[16]                                  =  22 B
  block_q5_1 : d(f16) m(f16) qh[4] qs[16]                           =  24 B
  block_iq4nl: d(f16) qs[16]                                        =  18 B

Inputs are procedurally generated (same formulas the Rust test uses), with
d = 0.5 and dmin = 0.25 so every product is a small dyadic rational --
exactly representable in both f32 and f64, making expected values exact.

The output is Rust source: paste into the `ggml_reference_tests` module in
src/tensor/dequantize.rs. On its first run this generator exposed three real
layout bugs (Q2_K, Q3_K, IQ4_NL) that Glint's circular kernel-vs-dequantize
tests could not see.
"""
import struct


def f16(v):  # value -> 2 little-endian bytes
    return struct.pack('<e', v)


def i8(b):  # unsigned byte -> int8
    return b - 256 if b >= 128 else b


def pat(n, start, a, c):
    """Deterministic byte pattern shared with the Rust test."""
    return bytes(((start + i) * a + c) & 0xFF for i in range(n))


KV_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)
    return d, m


def dequantize_q4_k(scales, qs, d, dmin):
    y = []
    is_ = 0
    q = 0
    for _j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        for l in range(32):
            y.append(d1 * (qs[q + l] & 0xF) - m1)
        for l in range(32):
            y.append(d2 * (qs[q + l] >> 4) - m2)
        q += 32
        is_ += 2
    return y


def dequantize_q5_k(scales, qh, qs, d, dmin):
    y = []
    is_ = 0
    ql = 0
    u1, u2 = 1, 2
    for _j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        for l in range(32):
            y.append(d1 * ((qs[ql + l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1)
        for l in range(32):
            y.append(d2 * ((qs[ql + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2)
        ql += 32
        is_ += 2
        u1 <<= 2
        u2 <<= 2
    return y


def dequantize_q6_k(ql, qh, scales, d):
    y = [0.0] * 256
    yo, qlo, qho, sco = 0, 0, 0, 0
    for _n in range(0, 256, 128):
        for l in range(32):
            is_ = l // 16
            q1 = i8(((ql[qlo + l + 0] & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) & 0xFF) - 32
            q2 = i8(((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) & 0xFF) - 32
            q3 = i8(((ql[qlo + l + 0] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) & 0xFF) - 32
            q4 = i8(((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) & 0xFF) - 32
            y[yo + l + 0] = d * i8(scales[sco + is_ + 0]) * q1
            y[yo + l + 32] = d * i8(scales[sco + is_ + 2]) * q2
            y[yo + l + 64] = d * i8(scales[sco + is_ + 4]) * q3
            y[yo + l + 96] = d * i8(scales[sco + is_ + 6]) * q4
        yo += 128
        qlo += 64
        qho += 32
        sco += 8
    return y


def dequantize_q2_k(scales, qs, d, dmin):
    y = []
    is_ = 0
    q = 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            sc = scales[is_]; is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for l in range(16):
                y.append(dl * ((qs[q + l] >> shift) & 3) - ml)
            sc = scales[is_]; is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for l in range(16):
                y.append(dl * ((qs[q + l + 16] >> shift) & 3) - ml)
            shift += 2
        q += 32
    return y


def dequantize_q3_k(hmask, qs, scales_raw, d_all):
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    aux = [int.from_bytes(scales_raw[i * 4:i * 4 + 4], 'little') for i in range(3)] + [0]
    tmp = aux[2]
    aux2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    sbytes = b''.join(a.to_bytes(4, 'little') for a in (aux0, aux1, aux2, aux3))
    scales = [i8(b) for b in sbytes]

    y = []
    is_ = 0
    m = 1
    q = 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            dl = d_all * (scales[is_] - 32); is_ += 1
            for l in range(16):
                y.append(dl * (((qs[q + l + 0] >> shift) & 3) - (0 if hmask[l + 0] & m else 4)))
            dl = d_all * (scales[is_] - 32); is_ += 1
            for l in range(16):
                y.append(dl * (((qs[q + l + 16] >> shift) & 3) - (0 if hmask[l + 16] & m else 4)))
            shift += 2
            m <<= 1
        q += 32
    return y


def dequantize_iq4_nl(qs, d):
    y = [0.0] * 32
    for j in range(16):
        y[j] = d * KV_IQ4NL[qs[j] & 0xF]
        y[j + 16] = d * KV_IQ4NL[qs[j] >> 4]
    return y


D, DMIN = 0.5, 0.25


def emit(name, vals, positions):
    print(f"// ── {name} ──")
    print(f"const {name}_EXPECTED: &[(usize, f32)] = &[")
    row = []
    for p in positions:
        row.append(f"({p}, {vals[p]!r})")
        if len(row) == 4:
            print("    " + ", ".join(row) + ",")
            row = []
    if row:
        print("    " + ", ".join(row) + ",")
    print("];")
    total = sum(vals)
    print(f"const {name}_SUM: f64 = {total!r};")
    print()


kq_positions = [17 * t for t in range(16)] + [256 + 17 * t for t in range(16)]

# Two super-blocks per format; field patterns continue across blocks.
vals = []
for k in range(2):
    vals += dequantize_q4_k(pat(12, k * 12, 83, 29), pat(128, k * 128, 37, 11), D, DMIN)
emit("Q4K", vals, kq_positions)

vals = []
for k in range(2):
    vals += dequantize_q5_k(pat(12, k * 12, 83, 29), pat(32, k * 32, 59, 17),
                            pat(128, k * 128, 37, 11), D, DMIN)
emit("Q5K", vals, kq_positions)

vals = []
for k in range(2):
    vals += dequantize_q6_k(pat(128, k * 128, 37, 11), pat(64, k * 64, 59, 17),
                            pat(16, k * 16, 83, 29), D)
emit("Q6K", vals, kq_positions)

vals = []
for k in range(2):
    vals += dequantize_q2_k(pat(16, k * 16, 83, 29), pat(64, k * 64, 37, 11), D, DMIN)
emit("Q2K", vals, kq_positions)

vals = []
for k in range(2):
    vals += dequantize_q3_k(pat(32, k * 32, 59, 17), pat(64, k * 64, 37, 11),
                            pat(12, k * 12, 83, 29), D)
emit("Q3K", vals, kq_positions)

def dequantize_q4_0(qs, d):
    # dequantize_row_q4_0, ggml/src/ggml-quants.c:
    #
    #     for (int j = 0; j < qk/2; ++j) {
    #         const int x0 = (x[i].qs[j] & 0x0F) - 8;
    #         const int x1 = (x[i].qs[j] >>   4) - 8;
    #         y[i*qk + j + 0   ] = x0*d;
    #         y[i*qk + j + qk/2] = x1*d;
    #     }
    #
    # Split-plane, exactly like Q5_0/Q5_1/IQ4_NL: byte j holds element j in its
    # low nibble and element j+16 in its high nibble — NOT elements 2j/2j+1.
    # quantize_row_q4_0_ref packs it the same way
    # (`y[i].qs[j] = xi0; y[i].qs[j] |= xi1 << 4;` with
    #  x0 = x[i*qk + 0 + j], x1 = x[i*qk + qk/2 + j]).
    y = [0.0] * 32
    for j in range(16):
        y[j] = ((qs[j] & 0x0F) - 8) * d
        y[j + 16] = ((qs[j] >> 4) - 8) * d
    return y


def dequantize_q5_0(qh_bytes, qs, d):
    qh = int.from_bytes(qh_bytes, 'little')
    y = [0.0] * 32
    for j in range(16):
        xh0 = ((qh >> (j + 0)) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        y[j] = (((qs[j] & 0xF) | xh0) - 16) * d
        y[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d
    return y


def dequantize_q5_1(qh_bytes, qs, d, m):
    qh = int.from_bytes(qh_bytes, 'little')
    y = [0.0] * 32
    for j in range(16):
        xh0 = ((qh >> (j + 0)) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        y[j] = ((qs[j] & 0xF) | xh0) * d + m
        y[j + 16] = ((qs[j] >> 4) | xh1) * d + m
    return y


vals = []
for k in range(2):
    vals += dequantize_q4_0(pat(16, k * 16, 37, 11), D)
emit("Q4_0", vals, list(range(64)))

vals = []
for k in range(2):
    vals += dequantize_iq4_nl(pat(16, k * 16, 37, 11), D)
emit("IQ4NL", vals, list(range(64)))

vals = []
for k in range(2):
    vals += dequantize_q5_0(pat(4, k * 4, 59, 17), pat(16, k * 16, 37, 11), D)
emit("Q5_0", vals, list(range(64)))

vals = []
for k in range(2):
    vals += dequantize_q5_1(pat(4, k * 4, 59, 17), pat(16, k * 16, 37, 11), D, DMIN)
emit("Q5_1", vals, list(range(64)))

# Sanity: confirm f16 round-trips are exact for D and DMIN.
assert struct.unpack('<e', f16(D))[0] == D
assert struct.unpack('<e', f16(DMIN))[0] == DMIN
print("// d/dmin f16-exact: OK")
