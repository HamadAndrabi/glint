# Quantization

Quantization compresses model weights from 32-bit floats into smaller integer representations. For LLM inference this is essential: a 7B parameter model in f32 requires 28 GB, but in Q4_K fits in ~4 GB.

Glint supports 8 quantization formats across two families: simple block quants (Q8_0, Q4_0, IQ4_NL) and K-quants (Q2_K through Q6_K).

Source: `src/tensor/quantized.rs`, `src/tensor/dequantize.rs`

---

## Simple Block Quants

### Q8_0

The simplest format: 8-bit signed integers with a per-block f16 scale.

```
Block (34 bytes, 32 elements):
┌────────────┬────────────────────────────────┐
│  d (f16)   │  32 × i8 values               │
│  2 bytes   │  32 bytes                      │
└────────────┴────────────────────────────────┘
```

**Encoding:** For each block of 32 f32 values:
1. Find `max_abs = max(|x|)`
2. `d = max_abs / 127`
3. `q[i] = round(x[i] / d)`, clamped to `[-127, 127]`

**Decoding:** `x[i] = q[i] * d`

Compression ratio: 128 bytes (f32) → 34 bytes = **3.76×**

### Q4_0

4-bit signed integers (nibble-packed) with a per-block f16 scale.

```
Block (18 bytes, 32 elements):
┌────────────┬────────────────────────────────┐
│  d (f16)   │  16 bytes (32 × 4-bit nibbles) │
│  2 bytes   │  packed two per byte           │
└────────────┴────────────────────────────────┘
```

Values range from -8 to +7 (4-bit signed, offset by -8 in storage).

Compression ratio: 128 bytes → 18 bytes = **7.1×**

### IQ4_NL

A non-linear variant of Q4_0 that maps 4-bit codes to values from a fixed lookup table rather than a linear scale. The lookup table is tuned to better approximate the distribution of transformer weights. Same storage layout as Q4_0.

---

## K-Quants

K-quants use 256-element super-blocks that contain nested sub-blocks. This allows a higher-precision shared scale across more elements, improving quality at the same bit width.

### Super-Block Structure (Q4_K example)

```
Super-block (144 bytes, 256 elements):
┌─────────┬─────────┬──────────────────────────────────────────┐
│ d (f16) │ dmin    │ 8 sub-blocks × (6-bit scale + 4-bit min) │
│ 2 bytes │ (f16)   │ + 128 bytes of 4-bit weights              │
└─────────┴─────────┴──────────────────────────────────────────┘
```

Each sub-block of 32 elements has its own 6-bit scale and 4-bit minimum, quantized relative to the super-block's `d` and `dmin`.

### Format Comparison

| Format | Bits/weight | Block layout | Notes |
|--------|-------------|-------------|-------|
| Q2_K | ~2.6 | 256-elem super-block, 4-bit scales | Lowest quality, smallest |
| Q3_K | ~3.4 | 256-elem super-block, 6-bit scales | |
| Q4_K | ~4.5 | 256-elem super-block, 6-bit scales + 4-bit mins | Recommended balance |
| Q5_K | ~5.5 | 256-elem super-block, 6-bit scales | High quality |
| Q6_K | ~6.5 | 256-elem super-block, 8-bit scales | Near-lossless |

---

## Dispatch Architecture

```rust
// src/tensor/quantized.rs (simplified)
pub fn matvec(weights: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "rayon"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { simd::matvec_q8_0_avx2(weights, x, out) };
    }
    matvec_q8_0_scalar(weights, x, out)  // scalar fallback
}
```

Every optimized path has a scalar fallback. The scalar implementation in `dequantize.rs` is the correctness reference.

---

## Memory Layout Invariants

These invariants are checked by the quantization tests and must hold across all code paths:

1. **Block boundaries.** Weights are stored in contiguous blocks; all accesses must respect block boundaries. Never read partial blocks.

2. **Scale type.** All simple block quants use **f16** scales (stored as raw `u16` bytes, read with `half::f16::from_le_bytes`).

3. **Nibble packing.** In Q4_0/Q4_K, nibbles are packed low-nibble-first: byte `b` contains elements `2i` (low) and `2i+1` (high):
   ```rust
   let lo = (byte & 0x0F) as i8 - 8;
   let hi = (byte >> 4)   as i8 - 8;
   ```

4. **K-quant alignment.** K-quant super-blocks must start at 256-element boundaries. The row dimension must be a multiple of 256 for K-quant weights.

5. **Row-major matvec.** The matvec signature is `W × x → out` where `W` is `[out_dim, in_dim]` in row-major order. Each row of `W` is dotted with `x` to produce one element of `out`.

---

## Choosing a Quantization Level

| Use case | Recommended |
|----------|------------|
| Maximum quality / debugging | Q8_0 or F16 |
| Production (7–8B models) | Q4_K or Q5_K |
| Memory-constrained | Q4_0 or Q3_K |
| KV-cache compression | Q8_0 (built-in via `KvCacheQ8`) |

For the KV cache specifically, Glint uses Q8_0 internally (`KvCacheQ8`) to achieve ~3.8× memory reduction while keeping dequantization overhead minimal (only one block per head per attention step).
