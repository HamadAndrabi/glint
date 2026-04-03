# Benchmarks

Glint includes micro-benchmarks for the hot-path matvec operations, measured with `criterion`.

Source: `benches/matvec.rs`

---

## Running Benchmarks

```bash
# Run all matvec benchmarks
cargo bench --bench matvec

# Run a specific benchmark by name
cargo bench --bench matvec -- q4_0

# Generate HTML report (opens in browser)
cargo bench --bench matvec -- --output-format html
```

Results are saved to `target/criterion/`. The HTML report includes plots of the distribution and historical comparisons.

---

## Matvec Throughput (4096 × 4096)

Measured on LLaMA-3 8B scale (matrix representing a single attention or FFN projection layer), single thread plus rayon parallel dispatch. CPU: AMD Ryzen 7 with AVX2 + FMA.

| Format | Throughput | Time per call |
|--------|-----------|---------------|
| Q4_0 | 24.4 Gelem/s | 687 µs |
| Q8_0 | 22.7 Gelem/s | 739 µs |
| Q4_K | 20.8 Gelem/s | 808 µs |
| Q6_K | 18.7 Gelem/s | 899 µs |
| Q5_K | 17.0 Gelem/s | 984 µs |

**Note:** Q4_0 outperforms Q8_0 because it moves half as much data through the memory bus (inference is memory-bandwidth bound). The unpacking overhead is cheaper than the extra memory access.

---

## Understanding the Numbers

### Arithmetic Intensity

LLM inference at batch size 1 is **memory-bandwidth bound**: the time to generate one token is dominated by reading weight bytes from RAM, not by arithmetic operations.

```
Arithmetic intensity = FLOPs / bytes_loaded
For matvec: = 2 × M × N / (M × N × bytes_per_elem)
            = 2 / bytes_per_elem

Q4_0: ~0.5 FLOPs/byte → clearly memory-bound
Q8_0: ~0.25 FLOPs/byte → even more memory-bound
```

On a typical CPU with 50 GB/s memory bandwidth, the theoretical peak for Q4_0:
```
50 GB/s × (2 FLOPs/byte) = 100 GFLOPS peak
Actual: ~24 Gelem/s × 2 = 48 GFLOPS  → ~50% efficiency (good)
```

### Why Rayon Helps

Rayon parallelizes across output rows. With 8 cores:
- Each core handles `4096/8 = 512` rows
- Memory bandwidth scales approximately linearly with cores (up to memory controller limits)
- Typical speedup: 4–6× over single-threaded

### Profiling Tools

For deeper investigation:

```bash
# Linux perf
cargo build --release && perf stat ./target/release/glint run -f model.gguf -p "x" -m 1

# Flamegraph
cargo install flamegraph
cargo flamegraph --bench matvec -- q8_0

# Memory bandwidth saturation
perf stat -e cache-misses,cache-references ./target/release/glint ...
```

---

## End-to-End Inference Speed

The matvec benchmarks measure individual ops. For real token generation speed, use the `run` subcommand:

```bash
glint run -f model.gguf -p "benchmark prompt" -m 100
# (100 tokens in 8.3s — 12.0 tok/s)
```

Factors that affect tokens/sec:
1. **Model size** — larger models have more layers and wider projections
2. **Context length** — longer contexts increase KV-cache read cost
3. **Quantization format** — Q4_K is 2× faster than Q8_0 at the same quality level
4. **Core count** — rayon scales near-linearly up to ~8 cores
5. **Cache warmth** — first run is slower (OS page faults); subsequent runs are faster

---

## Regression Testing

Before merging performance-sensitive changes:

1. Run `cargo bench --bench matvec` on the baseline branch and save results
2. Make your change
3. Run again and compare
4. `criterion` will flag regressions with a red "Performance has regressed" message

For forward-pass changes, time a full generation:
```bash
time glint run -f model.gguf -p "..." -m 200
```
