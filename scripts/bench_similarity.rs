#![allow(unused)]
use rand::Rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simsimd::SpatialSimilarity as SimSIMD;

const DIMENSIONS: usize = 1536;

pub(crate) fn generate_random_vector_f32(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

pub(crate) fn generate_random_vector_i8(dim: usize) -> Vec<i8> {
    (0..dim).map(|_| rand::thread_rng().gen_range(-128..=127)).collect()
}

pub(crate) fn generate_random_vector_u8(dim: usize) -> Vec<u8> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

// Baseline functions for f32
pub(crate) fn baseline_cos_functional_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let (dot_product, norm_a, norm_b) = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a * b, a * a, b * b))
        .fold((0.0, 0.0, 0.0), |acc, x| {
            (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
        });

    Some(1.0 - (dot_product / (norm_a.sqrt() * norm_b.sqrt())))
}

pub(crate) fn baseline_cos_unrolled_i8(a: &[i8], b: &[i8]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc1 = 0i32;
    let mut acc2 = 0i32;
    let mut acc3 = 0i32;
    let mut acc4 = 0i32;
    let mut acc5 = 0i32;
    let mut acc6 = 0i32;
    let mut acc7 = 0i32;
    let mut acc8 = 0i32;

    let mut norm_a1 = 0i32;
    let mut norm_a2 = 0i32;
    let mut norm_b1 = 0i32;
    let mut norm_b2 = 0i32;

    while i < (a.len() - remainder) {
        unsafe {
            let a1 = *a.get_unchecked(i) as i32;
            let a2 = *a.get_unchecked(i + 1) as i32;
            let a3 = *a.get_unchecked(i + 2) as i32;
            let a4 = *a.get_unchecked(i + 3) as i32;
            let a5 = *a.get_unchecked(i + 4) as i32;
            let a6 = *a.get_unchecked(i + 5) as i32;
            let a7 = *a.get_unchecked(i + 6) as i32;
            let a8 = *a.get_unchecked(i + 7) as i32;

            let b1 = *b.get_unchecked(i) as i32;
            let b2 = *b.get_unchecked(i + 1) as i32;
            let b3 = *b.get_unchecked(i + 2) as i32;
            let b4 = *b.get_unchecked(i + 3) as i32;
            let b5 = *b.get_unchecked(i + 4) as i32;
            let b6 = *b.get_unchecked(i + 5) as i32;
            let b7 = *b.get_unchecked(i + 6) as i32;
            let b8 = *b.get_unchecked(i + 7) as i32;

            acc1 += a1 * b1;
            acc2 += a2 * b2;
            acc3 += a3 * b3;
            acc4 += a4 * b4;
            acc5 += a5 * b5;
            acc6 += a6 * b6;
            acc7 += a7 * b7;
            acc8 += a8 * b8;

            norm_a1 += a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4;
            norm_b1 += b1 * b1 + b2 * b2 + b3 * b3 + b4 * b4;

            norm_a2 += a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8;
            norm_b2 += b5 * b5 + b6 * b6 + b7 * b7 + b8 * b8;
        }

        i += 8;
    }

    let dot_product = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    let norm_a = (norm_a1 + norm_a2) as f32;
    let norm_b = (norm_b1 + norm_b2) as f32;

    Some(1.0 - (dot_product as f32 / (norm_a.sqrt() * norm_b.sqrt())))
}

// Benchmarks
pub fn l2sq_benchmark(c: &mut Criterion) {
    let inputs_f32 = (
        generate_random_vector_f32(DIMENSIONS),
        generate_random_vector_f32(DIMENSIONS),
    );
    let inputs_i8 = (
        generate_random_vector_i8(DIMENSIONS),
        generate_random_vector_i8(DIMENSIONS),
    );
    let inputs_u8 = (
        generate_random_vector_u8(DIMENSIONS),
        generate_random_vector_u8(DIMENSIONS),
    );

    let mut group = c.benchmark_group("Squared Euclidean Distance");

    for i in 0..=5 {
        group.bench_with_input(BenchmarkId::new("SimSIMD_f32", i), &i, |b, _| {
            b.iter(|| SimSIMD::sqeuclidean(&inputs_f32.0, &inputs_f32.1))
        });
        group.bench_with_input(BenchmarkId::new("SimSIMD_i8", i), &i, |b, _| {
            b.iter(|| SimSIMD::sqeuclidean(&inputs_i8.0, &inputs_i8.1))
        });
        group.bench_with_input(BenchmarkId::new("SimSIMD_u8", i), &i, |b, _| {
            b.iter(|| SimSIMD::sqeuclidean(&inputs_u8.0, &inputs_u8.1))
        });
        group.bench_with_input(BenchmarkId::new("Rust Procedural i8", i), &i, |b, _| {
            b.iter(|| baseline_cos_unrolled_i8(&inputs_i8.0, &inputs_i8.1))
        });
    }
}

pub fn cos_benchmark(c: &mut Criterion) {
    let inputs_f32 = (
        generate_random_vector_f32(DIMENSIONS),
        generate_random_vector_f32(DIMENSIONS),
    );
    let inputs_i8 = (
        generate_random_vector_i8(DIMENSIONS),
        generate_random_vector_i8(DIMENSIONS),
    );
    let inputs_u8 = (
        generate_random_vector_u8(DIMENSIONS),
        generate_random_vector_u8(DIMENSIONS),
    );

    let mut group = c.benchmark_group("SIMD Cosine");

    for i in 0..=5 {
        group.bench_with_input(BenchmarkId::new("SimSIMD_f32", i), &i, |b, _| {
            b.iter(|| SimSIMD::cosine(&inputs_f32.0, &inputs_f32.1))
        });
        group.bench_with_input(BenchmarkId::new("SimSIMD_i8", i), &i, |b, _| {
            b.iter(|| SimSIMD::cosine(&inputs_i8.0, &inputs_i8.1))
        });
        group.bench_with_input(BenchmarkId::new("SimSIMD_u8", i), &i, |b, _| {
            b.iter(|| SimSIMD::cosine(&inputs_u8.0, &inputs_u8.1))
        });
        group.bench_with_input(BenchmarkId::new("Rust Procedural i8", i), &i, |b, _| {
            b.iter(|| baseline_cos_unrolled_i8(&inputs_i8.0, &inputs_i8.1))
        });
    }
}

// Criterion group definitions
criterion_group!(cos_benches, cos_benchmark);
criterion_group!(l2sq_benches, l2sq_benchmark);
criterion_main!(cos_benches, l2sq_benches);
