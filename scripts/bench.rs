//! Unified SimSIMD Benchmark Suite
//!
//! Compares SimSIMD vs native Rust implementations using Criterion.
//! Reports both performance and accuracy metrics.
//!
//! Run with:
//! ```bash
//! cargo bench --bench bench -- --quiet --noplot
//!
//! # Or with custom dimensions:
//! SIMSIMD_BENCH_DENSE_DIMENSIONS=2048 cargo bench --bench bench -- --quiet --noplot
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use simsimd::SpatialSimilarity as SimSIMD;
use std::time::Duration;

const PAIRS_COUNT: usize = 128;

fn get_dense_dimensions() -> usize {
    std::env::var("SIMSIMD_BENCH_DENSE_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536)
}

#[allow(dead_code)]
fn get_curved_dimensions() -> usize {
    std::env::var("SIMSIMD_BENCH_CURVED_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

fn generate_f32_pairs(dimensions: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..PAIRS_COUNT)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(i as u64);
            let a: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect()
}

fn generate_i8_pairs(dimensions: usize) -> Vec<(Vec<i8>, Vec<i8>)> {
    (0..PAIRS_COUNT)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(i as u64);
            let a: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect()
}

fn calculate_errors(baseline: &[f64], contender: &[f64]) -> (f64, f64) {
    let mut mean_delta = 0.0f64;
    let mut mean_relative_error = 0.0f64;

    for (&b, &c) in baseline.iter().zip(contender.iter()) {
        let abs_delta = (b - c).abs();
        mean_delta += abs_delta;
        if abs_delta != 0.0 && b != 0.0 {
            mean_relative_error += abs_delta / b.abs();
        }
    }

    mean_delta /= baseline.len() as f64;
    mean_relative_error /= baseline.len() as f64;
    (mean_delta, mean_relative_error)
}

fn print_error_table(name: &str, errors: &[(&str, f64, f64)]) {
    println!("\n{} Accuracy Report:", name);
    println!(
        "{:<20} {:>15} {:>15}",
        "Implementation", "Abs Error", "Rel Error"
    );
    println!("{:-<50}", "");
    for (impl_name, abs_err, rel_err) in errors {
        println!("{:<20} {:>15.6e} {:>15.6e}", impl_name, abs_err, rel_err);
    }
    println!();
}

// region: Baseline Implementations - Cosine Similarity (f32 -> f64)

fn baseline_cos_procedural(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        let (av, bv) = (a[i], b[i]);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    (1.0 - dot / (norm_a * norm_b).sqrt()) as f64
}

fn baseline_cos_functional(a: &[f32], b: &[f32]) -> f64 {
    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b)
        .map(|(a, b)| (*a * *b, *a * *a, *b * *b))
        .fold((0.0f32, 0.0f32, 0.0f32), |acc, x| {
            (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
        });
    (1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))) as f64
}

#[rustfmt::skip]
fn baseline_cos_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];
    let mut norm_a = [0.0f32; 8];
    let mut norm_b = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1,a2,a3,a4,a5,a6,a7,a8] = [*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3), *a.get_unchecked(i+4), *a.get_unchecked(i+5), *a.get_unchecked(i+6), *a.get_unchecked(i+7)];
            let [b1,b2,b3,b4,b5,b6,b7,b8] = [*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3), *b.get_unchecked(i+4), *b.get_unchecked(i+5), *b.get_unchecked(i+6), *b.get_unchecked(i+7)];

            acc[0] += a1*b1; acc[1] += a2*b2; acc[2] += a3*b3; acc[3] += a4*b4;
            acc[4] += a5*b5; acc[5] += a6*b6; acc[6] += a7*b7; acc[7] += a8*b8;
            norm_a[0] += a1*a1; norm_a[1] += a2*a2; norm_a[2] += a3*a3; norm_a[3] += a4*a4;
            norm_a[4] += a5*a5; norm_a[5] += a6*a6; norm_a[6] += a7*a7; norm_a[7] += a8*a8;
            norm_b[0] += b1*b1; norm_b[1] += b2*b2; norm_b[2] += b3*b3; norm_b[3] += b4*b4;
            norm_b[4] += b5*b5; norm_b[5] += b6*b6; norm_b[6] += b7*b7; norm_b[7] += b8*b8;
            i += 8;
        }
        while i < a.len() {
            let (av, bv) = (*a.get_unchecked(i), *b.get_unchecked(i));
            acc[0] += av * bv; norm_a[0] += av * av; norm_b[0] += bv * bv; i += 1;
        }
    }

    let dot: f32 = acc.iter().sum();
    let na: f32 = norm_a.iter().sum();
    let nb: f32 = norm_b.iter().sum();
    (1.0 - (dot / (na.sqrt() * nb.sqrt()))) as f64
}

// endregion: Baseline Implementations - Cosine Similarity (f32 -> f64)

// region: Baseline Implementations - Squared Euclidean Distance (f32 -> f64)

fn baseline_l2sq_procedural(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum as f64
}

fn baseline_l2sq_functional(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = *x - *y;
            diff * diff
        })
        .sum::<f32>() as f64
}

#[rustfmt::skip]
fn baseline_l2sq_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1,a2,a3,a4,a5,a6,a7,a8] = [*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3), *a.get_unchecked(i+4), *a.get_unchecked(i+5), *a.get_unchecked(i+6), *a.get_unchecked(i+7)];
            let [b1,b2,b3,b4,b5,b6,b7,b8] = [*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3), *b.get_unchecked(i+4), *b.get_unchecked(i+5), *b.get_unchecked(i+6), *b.get_unchecked(i+7)];

            let [d1,d2,d3,d4,d5,d6,d7,d8] = [a1-b1, a2-b2, a3-b3, a4-b4, a5-b5, a6-b6, a7-b7, a8-b8];
            acc[0] += d1*d1; acc[1] += d2*d2; acc[2] += d3*d3; acc[3] += d4*d4;
            acc[4] += d5*d5; acc[5] += d6*d6; acc[6] += d7*d7; acc[7] += d8*d8;
            i += 8;
        }
        while i < a.len() {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            acc[0] += diff * diff; i += 1;
        }
    }

    acc.iter().sum::<f32>() as f64
}

// endregion: Baseline Implementations - Squared Euclidean Distance (f32 -> f64)

// region: Baseline Implementations - Dot Product (f32 -> f64)

fn baseline_dot_procedural(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    -(sum as f64)
}

fn baseline_dot_functional(a: &[f32], b: &[f32]) -> f64 {
    -(a.iter().zip(b).map(|(x, y)| *x * *y).sum::<f32>() as f64)
}

#[rustfmt::skip]
fn baseline_dot_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1,a2,a3,a4,a5,a6,a7,a8] = [*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3), *a.get_unchecked(i+4), *a.get_unchecked(i+5), *a.get_unchecked(i+6), *a.get_unchecked(i+7)];
            let [b1,b2,b3,b4,b5,b6,b7,b8] = [*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3), *b.get_unchecked(i+4), *b.get_unchecked(i+5), *b.get_unchecked(i+6), *b.get_unchecked(i+7)];

            acc[0] += a1*b1; acc[1] += a2*b2; acc[2] += a3*b3; acc[3] += a4*b4;
            acc[4] += a5*b5; acc[5] += a6*b6; acc[6] += a7*b7; acc[7] += a8*b8;
            i += 8;
        }
        while i < a.len() {
            acc[0] += *a.get_unchecked(i) * *b.get_unchecked(i); i += 1;
        }
    }

    -(acc.iter().sum::<f32>() as f64)
}

// endregion: Baseline Implementations - Dot Product (f32 -> f64)

// region: Baseline Implementations - Euclidean Distance (f32 -> f64)

fn baseline_l2_procedural(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_procedural(a, b).sqrt()
}

fn baseline_l2_functional(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_functional(a, b).sqrt()
}

fn baseline_l2_unrolled(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_unrolled(a, b).sqrt()
}

// endregion: Baseline Implementations - Euclidean Distance (f32 -> f64)

// region: Baseline Implementations - i8 (i8 -> f64)

fn baseline_i8_cos_procedural(a: &[i8], b: &[i8]) -> f64 {
    let mut dot = 0i32;
    let mut norm_a = 0i32;
    let mut norm_b = 0i32;
    for i in 0..a.len() {
        let (av, bv) = (a[i] as i32, b[i] as i32);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    let dot_f = dot as f32;
    let norm_a_f = norm_a as f32;
    let norm_b_f = norm_b as f32;
    (1.0 - dot_f / (norm_a_f * norm_b_f).sqrt()) as f64
}

fn baseline_i8_dot_procedural(a: &[i8], b: &[i8]) -> f64 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    -(sum as f64)
}

fn baseline_i8_l2sq_procedural(a: &[i8], b: &[i8]) -> f64 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]) as i32;
        sum += diff * diff;
    }
    sum as f64
}

fn baseline_i8_l2_procedural(a: &[i8], b: &[i8]) -> f64 {
    baseline_i8_l2sq_procedural(a, b).sqrt()
}

// endregion: Baseline Implementations - i8 (i8 -> f64)

// region: Benchmark Functions

pub fn cosine_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_f32_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_cos_procedural(a, b))
        .collect();
    let functional: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_cos_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_cos_unrolled(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::cosine(a, b).unwrap())
        .collect();

    let errors = vec![
        ("SimSIMD", calculate_errors(&procedural, &simsimd)),
        ("Functional", calculate_errors(&procedural, &functional)),
        ("Unrolled", calculate_errors(&procedural, &unrolled)),
    ];
    print_error_table(
        &format!("Cosine <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("Cosine <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::cosine(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_cos_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Functional", |b| {
        b.iter(|| baseline_cos_functional(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Unrolled", |b| {
        b.iter(|| baseline_cos_unrolled(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn sqeuclidean_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_f32_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2sq_procedural(a, b))
        .collect();
    let functional: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2sq_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2sq_unrolled(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::sqeuclidean(a, b).unwrap())
        .collect();

    let errors = vec![
        ("SimSIMD", calculate_errors(&procedural, &simsimd)),
        ("Functional", calculate_errors(&procedural, &functional)),
        ("Unrolled", calculate_errors(&procedural, &unrolled)),
    ];
    print_error_table(
        &format!("SqEuclidean <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("SqEuclidean <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::sqeuclidean(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_l2sq_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Functional", |b| {
        b.iter(|| baseline_l2sq_functional(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Unrolled", |b| {
        b.iter(|| baseline_l2sq_unrolled(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn dot_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_f32_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_dot_procedural(a, b))
        .collect();
    let functional: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_dot_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_dot_unrolled(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::dot(a, b).unwrap())
        .collect();

    let errors = vec![
        ("SimSIMD", calculate_errors(&procedural, &simsimd)),
        ("Functional", calculate_errors(&procedural, &functional)),
        ("Unrolled", calculate_errors(&procedural, &unrolled)),
    ];
    print_error_table(
        &format!("Dot Product <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("Dot Product <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::dot(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_dot_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Functional", |b| {
        b.iter(|| baseline_dot_functional(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Unrolled", |b| {
        b.iter(|| baseline_dot_unrolled(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn euclidean_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_f32_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2_procedural(a, b))
        .collect();
    let functional: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_l2_unrolled(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::euclidean(a, b).unwrap())
        .collect();

    let errors = vec![
        ("SimSIMD", calculate_errors(&procedural, &simsimd)),
        ("Functional", calculate_errors(&procedural, &functional)),
        ("Unrolled", calculate_errors(&procedural, &unrolled)),
    ];
    print_error_table(
        &format!("Euclidean <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("Euclidean <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::euclidean(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_l2_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Functional", |b| {
        b.iter(|| baseline_l2_functional(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Unrolled", |b| {
        b.iter(|| baseline_l2_unrolled(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn i8_cosine_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_i8_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_i8_cos_procedural(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::cosine(a, b).unwrap())
        .collect();

    let errors = vec![("SimSIMD", calculate_errors(&procedural, &simsimd))];
    print_error_table(
        &format!("i8 Cosine <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("i8 Cosine <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::cosine(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_i8_cos_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn i8_dot_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_i8_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_i8_dot_procedural(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::dot(a, b).unwrap())
        .collect();

    let errors = vec![("SimSIMD", calculate_errors(&procedural, &simsimd))];
    print_error_table(
        &format!("i8 Dot Product <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("i8 Dot Product <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::dot(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_i8_dot_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn i8_sqeuclidean_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_i8_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_i8_l2sq_procedural(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::sqeuclidean(a, b).unwrap())
        .collect();

    let errors = vec![("SimSIMD", calculate_errors(&procedural, &simsimd))];
    print_error_table(
        &format!("i8 SqEuclidean <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("i8 SqEuclidean <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::sqeuclidean(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_i8_l2sq_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

pub fn i8_euclidean_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let pairs = generate_i8_pairs(dimensions);

    let procedural: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| baseline_i8_l2_procedural(a, b))
        .collect();
    let simsimd: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| SimSIMD::euclidean(a, b).unwrap())
        .collect();

    let errors = vec![("SimSIMD", calculate_errors(&procedural, &simsimd))];
    print_error_table(
        &format!("i8 Euclidean <{}d>", dimensions),
        &errors
            .iter()
            .map(|(n, (a, r))| (*n, *a, *r))
            .collect::<Vec<_>>(),
    );

    let mut group = c.benchmark_group(format!("i8 Euclidean <{}d>", dimensions));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);
    group.bench_function("SimSIMD", |b| {
        b.iter(|| SimSIMD::euclidean(&pairs[0].0, &pairs[0].1))
    });
    group.bench_function("Procedural", |b| {
        b.iter(|| baseline_i8_l2_procedural(&pairs[0].0, &pairs[0].1))
    });
    group.finish();
}

// endregion: Benchmark Functions

fn custom_criterion() -> Criterion {
    Criterion::default().without_plots().noise_threshold(0.05)
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = cosine_benchmark, sqeuclidean_benchmark, dot_benchmark, euclidean_benchmark,
              i8_cosine_benchmark, i8_dot_benchmark, i8_sqeuclidean_benchmark, i8_euclidean_benchmark
}
criterion_main!(benches);
