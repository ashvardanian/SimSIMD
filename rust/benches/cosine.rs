use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simsimd::SimSIMD;

mod native;

const DIMENSIONS: usize = 1536;

pub fn cosine_benchmark(c: &mut Criterion) {
    let inputs: (Vec<f32>, Vec<f32>) = (
        native::generate_random_vector(DIMENSIONS),
        native::generate_random_vector(DIMENSIONS),
    );

    let mut group = c.benchmark_group("SIMD Cosine");

    for i in 0..=5 {
        group.bench_with_input(BenchmarkId::new("SimSIMD", i), &i, |b, _| {
            b.iter(|| SimSIMD::cosine(&inputs.0, &inputs.1))
        });
        group.bench_with_input(BenchmarkId::new("Rust Native", i), &i, |b, _| {
            b.iter(|| native::cosine_similarity_cpu(&inputs.0, &inputs.1))
        });
    }
}

criterion_group!(benches, cosine_benchmark);
criterion_main!(benches);
