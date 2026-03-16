//! Benchmarks for quantized matrix-vector multiplication.
//!
//! Measures throughput at representative model sizes:
//!   - 576×576:   SmolLM-135M layer dimensions
//!   - 2048×2048: TinyLlama-1.1B scale
//!   - 4096×4096: Llama-3-8B scale

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ferrite::model::gguf::GgmlType;
use ferrite::tensor::QuantizedTensor;

/// Build a Q8_0 QuantizedTensor with deterministic data.
fn make_q8_0_matrix(rows: usize, cols: usize) -> QuantizedTensor {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    assert!(cols % BLOCK_ELEMS == 0);
    let n_blocks = cols / BLOCK_ELEMS;
    let bytes_per_row = n_blocks * BLOCK_BYTES;

    let mut data = vec![0u8; rows * bytes_per_row];
    for i in 0..rows {
        for b in 0..n_blocks {
            let off = i * bytes_per_row + b * BLOCK_BYTES;
            // scale = 0.01 as f16
            let scale = half::f16::from_f32(0.01);
            data[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            // weights: repeating pattern of signed i8 values
            for j in 0..BLOCK_ELEMS {
                data[off + 2 + j] = ((i.wrapping_add(b).wrapping_add(j)) % 200) as u8;
            }
        }
    }
    QuantizedTensor::from_raw(data, rows, cols, GgmlType::Q8_0)
}

fn bench_matvec_q8_0(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[
        (576, 576),
        (2048, 2048),
        (4096, 4096),
    ];

    let mut group = c.benchmark_group("matvec_q8_0");

    for &(rows, cols) in sizes {
        let qt = make_q8_0_matrix(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("simd_dispatch", format!("{rows}x{cols}")),
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

criterion_group!(benches, bench_matvec_q8_0);
criterion_main!(benches);
