//! Unified NumKong Benchmark Suite
//!
//! Compares NumKong vs native Rust implementations using Criterion.
//! Reports both performance and accuracy metrics.
//!
//! Run with:
//! ```bash
//! cargo bench --bench bench -- --quiet --noplot
//!
//! # Or with custom dimensions:
//! NK_BENCH_DENSE_DIMENSIONS=2048 cargo bench --bench bench -- --quiet --noplot
//! ```
use std::hint::black_box;
use std::mem::size_of;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use numkong::SpatialSimilarity as NumKong;

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn get_dense_dimensions() -> usize {
    std::env::var("NK_BENCH_DENSE_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536)
}

#[allow(dead_code)]
fn get_curved_dimensions() -> usize {
    std::env::var("NK_BENCH_CURVED_DIMENSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
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

fn print_error(abs_err: f64, rel_err: f64) {
    println!(
        "                        abs error: {}{:.6e}{}",
        BOLD, abs_err, RESET
    );
    println!(
        "                        rel error: {}{:.6e}{}",
        BOLD, rel_err, RESET
    );
}

// region: Accurate Baseline Implementations (f64)

fn accurate_cos(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let (av, bv) = (a[i] as f64, b[i] as f64);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    1.0 - dot / (norm_a * norm_b).sqrt()
}

fn accurate_l2sq(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]) as f64;
        sum += diff * diff;
    }
    sum
}

fn accurate_dot(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        sum += (a[i] as f64) * (b[i] as f64);
    }
    -sum
}

fn accurate_l2(a: &[f32], b: &[f32]) -> f64 {
    accurate_l2sq(a, b).sqrt()
}

fn accurate_i8_cos(a: &[i8], b: &[i8]) -> f64 {
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

fn accurate_i8_dot(a: &[i8], b: &[i8]) -> f64 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    -(sum as f64)
}

fn accurate_i8_l2sq(a: &[i8], b: &[i8]) -> f64 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]) as i32;
        sum += diff * diff;
    }
    sum as f64
}

fn accurate_i8_l2(a: &[i8], b: &[i8]) -> f64 {
    accurate_i8_l2sq(a, b).sqrt()
}

// endregion: Accurate Baseline Implementations (f64)

// region: Baseline Implementations - Cosine Similarity

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

fn baseline_cos_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];
    let mut norm_a = [0.0f32; 8];
    let mut norm_b = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1, a2, a3, a4, a5, a6, a7, a8] = [
                *a.get_unchecked(i),
                *a.get_unchecked(i + 1),
                *a.get_unchecked(i + 2),
                *a.get_unchecked(i + 3),
                *a.get_unchecked(i + 4),
                *a.get_unchecked(i + 5),
                *a.get_unchecked(i + 6),
                *a.get_unchecked(i + 7),
            ];
            let [b1, b2, b3, b4, b5, b6, b7, b8] = [
                *b.get_unchecked(i),
                *b.get_unchecked(i + 1),
                *b.get_unchecked(i + 2),
                *b.get_unchecked(i + 3),
                *b.get_unchecked(i + 4),
                *b.get_unchecked(i + 5),
                *b.get_unchecked(i + 6),
                *b.get_unchecked(i + 7),
            ];

            acc[0] += a1 * b1;
            acc[1] += a2 * b2;
            acc[2] += a3 * b3;
            acc[3] += a4 * b4;
            acc[4] += a5 * b5;
            acc[5] += a6 * b6;
            acc[6] += a7 * b7;
            acc[7] += a8 * b8;
            norm_a[0] += a1 * a1;
            norm_a[1] += a2 * a2;
            norm_a[2] += a3 * a3;
            norm_a[3] += a4 * a4;
            norm_a[4] += a5 * a5;
            norm_a[5] += a6 * a6;
            norm_a[6] += a7 * a7;
            norm_a[7] += a8 * a8;
            norm_b[0] += b1 * b1;
            norm_b[1] += b2 * b2;
            norm_b[2] += b3 * b3;
            norm_b[3] += b4 * b4;
            norm_b[4] += b5 * b5;
            norm_b[5] += b6 * b6;
            norm_b[6] += b7 * b7;
            norm_b[7] += b8 * b8;
            i += 8;
        }
        while i < a.len() {
            let (av, bv) = (*a.get_unchecked(i), *b.get_unchecked(i));
            acc[0] += av * bv;
            norm_a[0] += av * av;
            norm_b[0] += bv * bv;
            i += 1;
        }
    }

    let dot: f32 = acc.iter().sum();
    let na: f32 = norm_a.iter().sum();
    let nb: f32 = norm_b.iter().sum();
    (1.0 - (dot / (na.sqrt() * nb.sqrt()))) as f64
}

// endregion: Baseline Implementations - Cosine Similarity

// region: Baseline Implementations - Squared Euclidean Distance

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

fn baseline_l2sq_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1, a2, a3, a4, a5, a6, a7, a8] = [
                *a.get_unchecked(i),
                *a.get_unchecked(i + 1),
                *a.get_unchecked(i + 2),
                *a.get_unchecked(i + 3),
                *a.get_unchecked(i + 4),
                *a.get_unchecked(i + 5),
                *a.get_unchecked(i + 6),
                *a.get_unchecked(i + 7),
            ];
            let [b1, b2, b3, b4, b5, b6, b7, b8] = [
                *b.get_unchecked(i),
                *b.get_unchecked(i + 1),
                *b.get_unchecked(i + 2),
                *b.get_unchecked(i + 3),
                *b.get_unchecked(i + 4),
                *b.get_unchecked(i + 5),
                *b.get_unchecked(i + 6),
                *b.get_unchecked(i + 7),
            ];

            let [d1, d2, d3, d4, d5, d6, d7, d8] = [
                a1 - b1,
                a2 - b2,
                a3 - b3,
                a4 - b4,
                a5 - b5,
                a6 - b6,
                a7 - b7,
                a8 - b8,
            ];
            acc[0] += d1 * d1;
            acc[1] += d2 * d2;
            acc[2] += d3 * d3;
            acc[3] += d4 * d4;
            acc[4] += d5 * d5;
            acc[5] += d6 * d6;
            acc[6] += d7 * d7;
            acc[7] += d8 * d8;
            i += 8;
        }
        while i < a.len() {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            acc[0] += diff * diff;
            i += 1;
        }
    }

    acc.iter().sum::<f32>() as f64
}

// endregion: Baseline Implementations - Squared Euclidean Distance

// region: Baseline Implementations - Dot Product

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

fn baseline_dot_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc = [0.0f32; 8];

    unsafe {
        while i < a.len() - remainder {
            let [a1, a2, a3, a4, a5, a6, a7, a8] = [
                *a.get_unchecked(i),
                *a.get_unchecked(i + 1),
                *a.get_unchecked(i + 2),
                *a.get_unchecked(i + 3),
                *a.get_unchecked(i + 4),
                *a.get_unchecked(i + 5),
                *a.get_unchecked(i + 6),
                *a.get_unchecked(i + 7),
            ];
            let [b1, b2, b3, b4, b5, b6, b7, b8] = [
                *b.get_unchecked(i),
                *b.get_unchecked(i + 1),
                *b.get_unchecked(i + 2),
                *b.get_unchecked(i + 3),
                *b.get_unchecked(i + 4),
                *b.get_unchecked(i + 5),
                *b.get_unchecked(i + 6),
                *b.get_unchecked(i + 7),
            ];

            acc[0] += a1 * b1;
            acc[1] += a2 * b2;
            acc[2] += a3 * b3;
            acc[3] += a4 * b4;
            acc[4] += a5 * b5;
            acc[5] += a6 * b6;
            acc[6] += a7 * b7;
            acc[7] += a8 * b8;
            i += 8;
        }
        while i < a.len() {
            acc[0] += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }

    -(acc.iter().sum::<f32>() as f64)
}

// endregion: Baseline Implementations - Dot Product

// region: Baseline Implementations - Euclidean Distance

fn baseline_l2_procedural(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_procedural(a, b).sqrt()
}

fn baseline_l2_functional(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_functional(a, b).sqrt()
}

fn baseline_l2_unrolled(a: &[f32], b: &[f32]) -> f64 {
    baseline_l2sq_unrolled(a, b).sqrt()
}

// endregion: Baseline Implementations - Euclidean Distance

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

pub fn f32_cosine_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<f32>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| accurate_cos(a, b))
        .collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_cos_procedural(a, b))
        .collect();
    let functional: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_cos_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_cos_unrolled(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::cosine(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);
    let functional_err = calculate_errors(&accurate, &functional);
    let unrolled_err = calculate_errors(&accurate, &unrolled);

    let mut group = c.benchmark_group(format!("f32/{}d/cosine", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<f32>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::cosine(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_cos_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);

    group.bench_function("functional", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_cos_functional(a, b));
            }
        })
    });
    print_error(functional_err.0, functional_err.1);

    group.bench_function("unrolled", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_cos_unrolled(a, b));
            }
        })
    });
    print_error(unrolled_err.0, unrolled_err.1);
    group.finish();
}

pub fn f32_dot_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<f32>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| accurate_dot(a, b))
        .collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_dot_procedural(a, b))
        .collect();
    let functional: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_dot_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_dot_unrolled(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::dot(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);
    let functional_err = calculate_errors(&accurate, &functional);
    let unrolled_err = calculate_errors(&accurate, &unrolled);

    let mut group = c.benchmark_group(format!("f32/{}d/dot", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<f32>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::dot(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_dot_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);

    group.bench_function("functional", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_dot_functional(a, b));
            }
        })
    });
    print_error(functional_err.0, functional_err.1);

    group.bench_function("unrolled", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_dot_unrolled(a, b));
            }
        })
    });
    print_error(unrolled_err.0, unrolled_err.1);
    group.finish();
}

pub fn f32_l2_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<f32>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<f32> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs.iter().map(|(a, b)| accurate_l2(a, b)).collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_l2_procedural(a, b))
        .collect();
    let functional: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_l2_functional(a, b))
        .collect();
    let unrolled: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_l2_unrolled(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::euclidean(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);
    let functional_err = calculate_errors(&accurate, &functional);
    let unrolled_err = calculate_errors(&accurate, &unrolled);

    let mut group = c.benchmark_group(format!("f32/{}d/l2", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<f32>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::euclidean(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_l2_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);

    group.bench_function("functional", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_l2_functional(a, b));
            }
        })
    });
    print_error(functional_err.0, functional_err.1);

    group.bench_function("unrolled", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_l2_unrolled(a, b));
            }
        })
    });
    print_error(unrolled_err.0, unrolled_err.1);
    group.finish();
}

pub fn i8_cosine_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<i8>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| accurate_i8_cos(a, b))
        .collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_i8_cos_procedural(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::cosine(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);

    let mut group = c.benchmark_group(format!("i8/{}d/cosine", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<i8>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::cosine(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_i8_cos_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);
    group.finish();
}

pub fn i8_dot_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<i8>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| accurate_i8_dot(a, b))
        .collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_i8_dot_procedural(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::dot(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);

    let mut group = c.benchmark_group(format!("i8/{}d/dot", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<i8>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::dot(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_i8_dot_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);
    group.finish();
}

pub fn i8_l2_benchmark(c: &mut Criterion) {
    let dimensions = get_dense_dimensions();
    let bench_pairs_count = (128 * 1024 * 1024) / (dimensions * size_of::<i8>() * 2);
    let bench_pairs: Vec<_> = (0..bench_pairs_count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(1000 + i as u64);
            let a: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            let b: Vec<i8> = (0..dimensions).map(|_| rng.random()).collect();
            (a, b)
        })
        .collect();

    let accurate: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| accurate_i8_l2(a, b))
        .collect();
    let procedural: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| baseline_i8_l2_procedural(a, b))
        .collect();
    let numkong: Vec<f64> = bench_pairs
        .iter()
        .map(|(a, b)| NumKong::euclidean(a, b).unwrap())
        .collect();

    let nk_err = calculate_errors(&accurate, &numkong);
    let procedural_err = calculate_errors(&accurate, &procedural);

    let mut group = c.benchmark_group(format!("i8/{}d/l2", dimensions));
    let bytes_per_op = (2 * dimensions * size_of::<i8>() * bench_pairs_count) as u64;
    group.throughput(Throughput::Bytes(bytes_per_op));
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.05);

    group.bench_function("numkong", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(NumKong::euclidean(a, b));
            }
        })
    });
    print_error(nk_err.0, nk_err.1);

    group.bench_function("procedural", |b| {
        b.iter(|| {
            for (a, b) in &bench_pairs {
                black_box(baseline_i8_l2_procedural(a, b));
            }
        })
    });
    print_error(procedural_err.0, procedural_err.1);
    group.finish();
}

// endregion: Benchmark Functions

fn custom_criterion() -> Criterion {
    Criterion::default().without_plots().noise_threshold(0.05)
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = f32_cosine_benchmark, f32_dot_benchmark, f32_l2_benchmark,
              i8_cosine_benchmark, i8_dot_benchmark, i8_l2_benchmark
}
criterion_main!(benches);
