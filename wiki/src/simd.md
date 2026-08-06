# SIMD Optimizations

Glint uses AVX2 and FMA SIMD intrinsics for the hot-path quantized matvec operations, achieving 4–8× speedups over scalar code on modern x86_64 CPUs.

Source: `src/tensor/simd.rs`

---

## Background: Why SIMD Matters

LLM inference at batch size 1 is **memory-bandwidth bound**, not compute bound. The bottleneck is loading weight bytes from RAM. SIMD helps by:

1. Processing 8 f32 values or 32 int8 values per instruction
2. Keeping the arithmetic pipeline full while data is in flight
3. Using FMA (fused multiply-add) to avoid intermediate rounding

For a Q4_0 matvec on a 4096×4096 matrix, the scalar path touches ~8 MB of weight data per call. The AVX2 path processes this with 256-bit registers, effectively 8× wider memory access patterns.

---

## Compilation Guard

SIMD code is only compiled when both conditions hold:
- Architecture is `x86_64`
- The `rayon` feature is active

```rust
#[cfg(all(target_arch = "x86_64", feature = "rayon"))]
```

At runtime, dispatch checks CPU capabilities:

```rust
#[cfg(all(target_arch = "x86_64", feature = "rayon"))]
if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    return unsafe { simd::matvec_q8_0_avx2(weights, x, out) };
}
matvec_q8_0_scalar(weights, x, out)  // always available
```

This means a binary built on x86_64 with `rayon` will use AVX2 when run on a supporting CPU (virtually all CPUs since ~2013), and fall back to scalar on older hardware or other architectures.

---

## AVX2 Matvec (Q8_0)

The Q8_0 matvec computes `W × x → out` where `W` is stored as Q8_0 blocks.

Key operations:
- `_mm256_loadu_si256` — load 32 int8 values at once
- `_mm256_set1_epi32` — broadcast a scalar to all 8 int32 lanes
- `_mm256_madd_epi16` — multiply-accumulate 16-bit integers
- `_mm256_fmadd_ps` — fused multiply-add for f32 (for scale application)

The inner loop processes one Q8_0 block (32 elements) per iteration:

```rust
// SAFETY: only called when avx2 + fma are detected; pointer alignment
// is guaranteed by the Vec allocation from QuantizedTensor.
unsafe fn dot_q8_0_block_avx2(w_block: &[u8; 34], x_block: &[f32; 32]) -> f32 {
    let d = f16::from_le_bytes([w_block[0], w_block[1]]).to_f32();
    let w_ints = _mm256_loadu_si256(w_block[2..].as_ptr() as *const __m256i);
    // ... vectorized dot product ...
    d * horizontal_sum(acc)
}
```

---

## AVX2 Matvec (Q4_0)

Q4_0 requires nibble unpacking as an extra step. Each byte encodes two 4-bit values:

```rust
let lo = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));   // low nibbles
let hi = _mm256_srli_epi16(bytes, 4);                        // high nibbles (shift right)
// offset by -8 for signed representation
let lo = _mm256_sub_epi8(lo, _mm256_set1_epi8(8));
let hi = _mm256_sub_epi8(hi, _mm256_set1_epi8(8));
```

After unpacking, the computation is similar to Q8_0 but operates on 4-bit quantities extended to 8-bit.

The two nibble planes stay separate: the 16 low nibbles are elements 0–15 and the 16 high nibbles are elements 16–31 (ggml's split-plane layout — see [Quantization](./quantization.md)), so they drop straight into the low and high 128-bit halves of the `__m256i` with no per-byte interleave.

---

## K-Quant SIMD

K-quant matvecs (Q4_K, Q5_K, Q6_K) use the same AVX2 register width but require decoding the multi-level scale structure:

1. Load super-block scale `d` and minimum `dmin` (f16)
2. For each sub-block, load its 6-bit scale and 4-bit min from the packed metadata bytes
3. Compute `effective_scale = d * sub_scale` and `effective_min = dmin * sub_min`
4. Unpack and multiply with effective scale, subtract effective min

The extra bookkeeping makes K-quant kernels about 20–30% slower than Q8_0 per element, which is expected given the more complex encoding.

---

## Coverage

AVX2+FMA kernels exist for eight formats: `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`,
`Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`. The 5-bit formats assemble their fifth bit
from the shared `qh` word with a broadcast + `shuffle_epi8` bit-spread
(`qh_fifth_bits`), and all the simple 4/5-bit formats share the split-plane
nibble expansion (`split_plane_nibbles`). `Q2_K`, `Q3_K`, and `IQ4_NL` use the
scalar path.

Continuous batching adds *batched* AVX2 kernels (`matvec_*_batch_avx2`) for
`Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K` that walk each weight row once for up
to 16 input vectors. Formats whose only AVX2 kernel is single-vector
(`Q4_1`/`Q5_0`/`Q5_1`) are batched by delegating per lane to that same kernel,
keeping batched output bit-identical to the single path.

---

## Unsafe Code Discipline

All SIMD functions are `unsafe`. Every unsafe block must have a `// SAFETY:` comment explaining the invariants maintained:

```rust
// SAFETY: called only when avx2 + fma detected at runtime via
// is_x86_feature_detected!(); all pointer reads are within
// bounds established by block_count × block_size layout of
// QuantizedTensor::data.
unsafe fn matvec_q8_0_avx2(qt: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
    ...
}
```

Common invariants to document:
- CPU feature requirements (`avx2`, `fma` detected at dispatch site)
- Pointer alignment (Vec<u8> is heap-allocated, 1-byte aligned; loads use `_loadu` variants)
- Buffer bounds (row count × block size equals total data length)

---

## Performance

Measured on LLaMA-3 8B scale (4096 × 4096 matrix), single thread + rayon:

| Format | Throughput | Time per call |
|--------|-----------|---------------|
| Q4_0 | 24.4 Gelem/s | 687 µs |
| Q8_0 | 22.7 Gelem/s | 739 µs |
| Q4_K | 20.8 Gelem/s | 808 µs |
| Q6_K | 18.7 Gelem/s | 899 µs |
| Q5_K | 17.0 Gelem/s | 984 µs |

Run benchmarks with:
```bash
cargo bench --bench matvec
```

See [Benchmarks](./benchmarks.md) for more details.

---

## Scalar Fallback

Every SIMD path has a corresponding scalar fallback in `src/tensor/dequantize.rs` and `src/tensor/quantized.rs`. The scalar path:
- Is the correctness reference
- Runs on all architectures (ARM, WASM, etc.)
- Is tested independently
- Should produce results within floating-point tolerance of the SIMD path

When adding a new quantization format, always implement the scalar path first, write tests comparing against known-good values, then implement the SIMD acceleration.
