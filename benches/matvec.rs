//! Benchmarks for quantized matrix-vector multiplication.
//!
//! Measures throughput across all supported quantization formats at
//! representative model dimensions:
//!   - 576×576:   SmolLM-135M layer dimensions
//!   - 2048×2048: TinyLlama-1.1B scale
//!   - 4096×4096: Llama-3-8B scale
//!
//! Each format dispatches through `QuantizedTensor::matvec()`, which uses
//! the fastest available path (AVX2+FMA SIMD on x86_64, scalar fallback
//! otherwise).
//!
//! Run:  cargo bench --bench matvec

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use glint::model::gguf::GgmlType;
use glint::tensor::QuantizedTensor;

const SIZES: &[(usize, usize)] = &[
    (576, 576),
    (2048, 2048),
    (4096, 4096),
];

// ── Matrix builders ─────────────────────────────────────────────────────────

fn make_matrix(rows: usize, cols: usize, ggml_type: GgmlType) -> QuantizedTensor {
    match ggml_type {
        GgmlType::Q8_0 => make_q8_0(rows, cols),
        GgmlType::Q4_0 => make_q4_0(rows, cols),
        GgmlType::Q4K  => make_k_quant(rows, cols, GgmlType::Q4K, 144),
        GgmlType::Q5K  => make_k_quant(rows, cols, GgmlType::Q5K, 176),
        GgmlType::Q6K  => make_k_quant(rows, cols, GgmlType::Q6K, 210),
        _ => unimplemented!(),
    }
}

fn make_q8_0(rows: usize, cols: usize) -> QuantizedTensor {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    let mut data = vec![0u8; rows * bytes_per_row];
    for i in 0..rows {
        for b in 0..n_blocks {
            let off = i * bytes_per_row + b * BLOCK_BYTES;
            let scale = half::f16::from_f32(0.01);
            data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            for j in 0..BLOCK_ELEMS {
                data[off + 2 + j] = ((i.wrapping_add(b).wrapping_add(j)) % 200) as u8;
            }
        }
    }
    QuantizedTensor::from_raw(data, rows, cols, GgmlType::Q8_0)
}

fn make_q4_0(rows: usize, cols: usize) -> QuantizedTensor {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    let mut data = vec![0u8; rows * bytes_per_row];
    for i in 0..rows {
        for b in 0..n_blocks {
            let off = i * bytes_per_row + b * BLOCK_BYTES;
            let scale = half::f16::from_f32(0.02);
            data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            for j in 0..16 {
                let lo = ((i + b + j) % 16) as u8;
                let hi = ((i + b + j + 3) % 16) as u8;
                data[off + 2 + j] = lo | (hi << 4);
            }
        }
    }
    QuantizedTensor::from_raw(data, rows, cols, GgmlType::Q4_0)
}

/// Build a k-quant matrix (Q4_K, Q5_K, or Q6_K) with deterministic data.
fn make_k_quant(rows: usize, cols: usize, ggml_type: GgmlType, block_bytes: usize) -> QuantizedTensor {
    const SUPER_BLOCK: usize = 256;
    let n_super = cols / SUPER_BLOCK;
    let bytes_per_row = n_super * block_bytes;

    let mut data = vec![0u8; rows * bytes_per_row];
    for i in 0..rows {
        for sb in 0..n_super {
            let off = i * bytes_per_row + sb * block_bytes;
            let block = &mut data[off..off + block_bytes];

            match ggml_type {
                GgmlType::Q4K => {
                    // [f16 d][f16 dmin][scales×12][qs×128]
                    block[0..2].copy_from_slice(&half::f16::from_f32(0.015).to_le_bytes());
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
                    for j in 0..4 { block[4 + j] = ((i + sb + j) % 63 + 1) as u8; }
                    for j in 0..4 { block[8 + j] = ((i + sb + j + 7) % 63 + 1) as u8; }
                    for j in 4..8 { block[4 + j + 4] = ((i + sb + j) % 15 + 1) as u8; }
                    for k in 16..144 {
                        let lo = ((i + sb + k) % 16) as u8;
                        let hi = ((i + sb + k + 5) % 16) as u8;
                        block[k] = lo | (hi << 4);
                    }
                }
                GgmlType::Q5K => {
                    // [f16 d][f16 dmin][scales×12][qh×32][qs×128]
                    block[0..2].copy_from_slice(&half::f16::from_f32(0.015).to_le_bytes());
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
                    for j in 0..4 { block[4 + j] = ((i + sb + j) % 63 + 1) as u8; }
                    for j in 0..4 { block[8 + j] = ((i + sb + j + 7) % 63 + 1) as u8; }
                    for j in 4..8 { block[4 + j + 4] = ((i + sb + j) % 15 + 1) as u8; }
                    for k in 0..32 { block[16 + k] = ((i + sb + k * 3) % 256) as u8; }
                    for k in 0..128 {
                        let lo = ((i + sb + k) % 16) as u8;
                        let hi = ((i + sb + k + 5) % 16) as u8;
                        block[48 + k] = lo | (hi << 4);
                    }
                }
                GgmlType::Q6K => {
                    // [ql×128][qh×64][scales i8×16][f16 d]
                    for k in 0..128 { block[k] = ((i + sb + k * 7) % 256) as u8; }
                    for k in 0..64 { block[128 + k] = ((i + sb + k * 3) % 256) as u8; }
                    for k in 0..16 { block[192 + k] = ((i + sb + k + 1) % 127 + 1) as u8; }
                    block[208..210].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }
    QuantizedTensor::from_raw(data, rows, cols, ggml_type)
}

fn make_input(cols: usize) -> Vec<f32> {
    (0..cols).map(|i| (i as f32) * 0.001).collect()
}

// ── Benchmark groups ────────────────────────────────────────────────────────

fn bench_format(c: &mut Criterion, name: &str, ggml_type: GgmlType) {
    let mut group = c.benchmark_group(name);

    for &(rows, cols) in SIZES {
        let qt = make_matrix(rows, cols, ggml_type);
        let input = make_input(cols);

        // Throughput: number of multiply-accumulate operations per matvec
        // (rows × cols fused multiply-adds)
        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{cols}")),
            &(&qt, &input),
            |b, (qt, input)| {
                b.iter(|| {
                    black_box(qt.matvec(black_box(input)));
                });
            },
        );
    }

    group.finish();
}

fn bench_q8_0(c: &mut Criterion) { bench_format(c, "matvec_Q8_0", GgmlType::Q8_0); }
fn bench_q4_0(c: &mut Criterion) { bench_format(c, "matvec_Q4_0", GgmlType::Q4_0); }
fn bench_q4_k(c: &mut Criterion) { bench_format(c, "matvec_Q4_K", GgmlType::Q4K); }
fn bench_q5_k(c: &mut Criterion) { bench_format(c, "matvec_Q5_K", GgmlType::Q5K); }
fn bench_q6_k(c: &mut Criterion) { bench_format(c, "matvec_Q6_K", GgmlType::Q6K); }

/// Cross-format comparison at a single representative size (4096×4096).
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_comparison_4096");
    let (rows, cols) = (4096, 4096);
    let input = make_input(cols);

    group.throughput(Throughput::Elements((rows * cols) as u64));

    for (name, ggml_type) in [
        ("Q8_0", GgmlType::Q8_0),
        ("Q4_0", GgmlType::Q4_0),
        ("Q4_K", GgmlType::Q4K),
        ("Q5_K", GgmlType::Q5K),
        ("Q6_K", GgmlType::Q6K),
    ] {
        let qt = make_matrix(rows, cols, ggml_type);
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&qt, &input),
            |b, (qt, input)| {
                b.iter(|| {
                    black_box(qt.matvec(black_box(input)));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_q8_0,
    bench_q4_0,
    bench_q4_k,
    bench_q5_k,
    bench_q6_k,
    bench_comparison,
);
criterion_main!(benches);
