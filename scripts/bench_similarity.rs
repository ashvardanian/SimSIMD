//! Benchmark for vector similarity functions
//!
//! Compares NumKong vs native Rust implementations using Criterion.
//! Run with:
//!
//! ```bash
//! cargo bench --bench bench_sqeuclidean
//! ```
#![allow(unused)]
use rand::Rng;
use std::ops::{AddAssign, Mul};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_traits::{AsPrimitive, Num, NumCast};
use numkong::SpatialSimilarity as NumKong;

const DIMENSIONS: usize = 1536;

pub(crate) fn generate_random_vector_f32(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

pub(crate) fn generate_random_vector_i8(dim: usize) -> Vec<i8> {
    (0..dim)
        .map(|_| rand::thread_rng().gen_range(-128..=127))
        .collect()
}

pub(crate) fn generate_random_vector_u8(dim: usize) -> Vec<u8> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

pub(crate) fn baseline_cos_unrolled<T, Acc>(a: &[T], b: &[T]) -> Option<f32>
where
    T: Num + Copy + NumCast + AsPrimitive<f32>,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    T: AsPrimitive<Acc>,
{
    if a.len() != b.len() {
        return None;
    }

    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc1 = Acc::zero();
    let mut acc2 = Acc::zero();
    let mut acc3 = Acc::zero();
    let mut acc4 = Acc::zero();
    let mut acc5 = Acc::zero();
    let mut acc6 = Acc::zero();
    let mut acc7 = Acc::zero();
    let mut acc8 = Acc::zero();

    let mut norm_a1 = Acc::zero();
    let mut norm_a2 = Acc::zero();
    let mut norm_b1 = Acc::zero();
    let mut norm_b2 = Acc::zero();

    while i < (a.len() - remainder) {
        unsafe {
            let a1 = *a.get_unchecked(i);
            let a2 = *a.get_unchecked(i + 1);
            let a3 = *a.get_unchecked(i + 2);
            let a4 = *a.get_unchecked(i + 3);
            let a5 = *a.get_unchecked(i + 4);
            let a6 = *a.get_unchecked(i + 5);
            let a7 = *a.get_unchecked(i + 6);
            let a8 = *a.get_unchecked(i + 7);

            let b1 = *b.get_unchecked(i);
            let b2 = *b.get_unchecked(i + 1);
            let b3 = *b.get_unchecked(i + 2);
            let b4 = *b.get_unchecked(i + 3);
            let b5 = *b.get_unchecked(i + 4);
            let b6 = *b.get_unchecked(i + 5);
            let b7 = *b.get_unchecked(i + 6);
            let b8 = *b.get_unchecked(i + 7);

            let a1_acc: Acc = NumCast::from(a1).unwrap();
            let a2_acc: Acc = NumCast::from(a2).unwrap();
            let a3_acc: Acc = NumCast::from(a3).unwrap();
            let a4_acc: Acc = NumCast::from(a4).unwrap();
            let a5_acc: Acc = NumCast::from(a5).unwrap();
            let a6_acc: Acc = NumCast::from(a6).unwrap();
            let a7_acc: Acc = NumCast::from(a7).unwrap();
            let a8_acc: Acc = NumCast::from(a8).unwrap();

            let b1_acc: Acc = NumCast::from(b1).unwrap();
            let b2_acc: Acc = NumCast::from(b2).unwrap();
            let b3_acc: Acc = NumCast::from(b3).unwrap();
            let b4_acc: Acc = NumCast::from(b4).unwrap();
            let b5_acc: Acc = NumCast::from(b5).unwrap();
            let b6_acc: Acc = NumCast::from(b6).unwrap();
            let b7_acc: Acc = NumCast::from(b7).unwrap();
            let b8_acc: Acc = NumCast::from(b8).unwrap();

            acc1 += a1_acc * b1_acc;
            acc2 += a2_acc * b2_acc;
            acc3 += a3_acc * b3_acc;
            acc4 += a4_acc * b4_acc;
            acc5 += a5_acc * b5_acc;
            acc6 += a6_acc * b6_acc;
            acc7 += a7_acc * b7_acc;
            acc8 += a8_acc * b8_acc;

            norm_a1 += a1_acc * a1_acc + a2_acc * a2_acc + a3_acc * a3_acc + a4_acc * a4_acc;
            norm_b1 += b1_acc * b1_acc + b2_acc * b2_acc + b3_acc * b3_acc + b4_acc * b4_acc;

            norm_a2 += a5_acc * a5_acc + a6_acc * a6_acc + a7_acc * a7_acc + a8_acc * a8_acc;
            norm_b2 += b5_acc * b5_acc + b6_acc * b6_acc + b7_acc * b7_acc + b8_acc * b8_acc;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        unsafe {
            let a_acc: Acc = NumCast::from(*a.get_unchecked(i)).unwrap();
            let b_acc: Acc = NumCast::from(*b.get_unchecked(i)).unwrap();
            acc1 += a_acc * b_acc;
            norm_a1 += a_acc * a_acc;
            norm_b1 += b_acc * b_acc;
        }
        i += 1;
    }

    let dot_product = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    let norm_a = norm_a1 + norm_a2;
    let norm_b = norm_b1 + norm_b2;

    let dot_product_f32: f32 = NumCast::from(dot_product).unwrap();
    let norm_a_f32: f32 = NumCast::from(norm_a).unwrap();
    let norm_b_f32: f32 = NumCast::from(norm_b).unwrap();

    Some(1.0 - (dot_product_f32 / (norm_a_f32.sqrt() * norm_b_f32.sqrt())))
}

pub(crate) fn baseline_l2sq_unrolled<T, Acc>(a: &[T], b: &[T]) -> Option<f32>
where
    T: Num + Copy + NumCast,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    T: AsPrimitive<Acc>,
{
    if a.len() != b.len() {
        return None;
    }
    let mut i = 0;
    let remainder = a.len() % 8;

    let mut acc1 = Acc::zero();
    let mut acc2 = Acc::zero();
    let mut acc3 = Acc::zero();
    let mut acc4 = Acc::zero();
    let mut acc5 = Acc::zero();
    let mut acc6 = Acc::zero();
    let mut acc7 = Acc::zero();
    let mut acc8 = Acc::zero();

    while i < (a.len() - remainder) {
        unsafe {
            let a1 = *a.get_unchecked(i);
            let a2 = *a.get_unchecked(i + 1);
            let a3 = *a.get_unchecked(i + 2);
            let a4 = *a.get_unchecked(i + 3);
            let a5 = *a.get_unchecked(i + 4);
            let a6 = *a.get_unchecked(i + 5);
            let a7 = *a.get_unchecked(i + 6);
            let a8 = *a.get_unchecked(i + 7);

            let b1 = *b.get_unchecked(i);
            let b2 = *b.get_unchecked(i + 1);
            let b3 = *b.get_unchecked(i + 2);
            let b4 = *b.get_unchecked(i + 3);
            let b5 = *b.get_unchecked(i + 4);
            let b6 = *b.get_unchecked(i + 5);
            let b7 = *b.get_unchecked(i + 6);
            let b8 = *b.get_unchecked(i + 7);

            let diff1 = <Acc as NumCast>::from(a1).unwrap() - <Acc as NumCast>::from(b1).unwrap();
            let diff2 = <Acc as NumCast>::from(a2).unwrap() - <Acc as NumCast>::from(b2).unwrap();
            let diff3 = <Acc as NumCast>::from(a3).unwrap() - <Acc as NumCast>::from(b3).unwrap();
            let diff4 = <Acc as NumCast>::from(a4).unwrap() - <Acc as NumCast>::from(b4).unwrap();
            let diff5 = <Acc as NumCast>::from(a5).unwrap() - <Acc as NumCast>::from(b5).unwrap();
            let diff6 = <Acc as NumCast>::from(a6).unwrap() - <Acc as NumCast>::from(b6).unwrap();
            let diff7 = <Acc as NumCast>::from(a7).unwrap() - <Acc as NumCast>::from(b7).unwrap();
            let diff8 = <Acc as NumCast>::from(a8).unwrap() - <Acc as NumCast>::from(b8).unwrap();

            acc1 += diff1 * diff1;
            acc2 += diff2 * diff2;
            acc3 += diff3 * diff3;
            acc4 += diff4 * diff4;
            acc5 += diff5 * diff5;
            acc6 += diff6 * diff6;
            acc7 += diff7 * diff7;
            acc8 += diff8 * diff8;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        unsafe {
            let a_val = <Acc as NumCast>::from(*a.get_unchecked(i)).unwrap();
            let b_val = <Acc as NumCast>::from(*b.get_unchecked(i)).unwrap();
            let diff = a_val - b_val;
            acc1 += diff * diff;
        }
        i += 1;
    }

    let sum = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    let sum_f32: f32 = NumCast::from(sum).unwrap();

    Some(sum_f32)
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

    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("Squared Euclidean Distance");
    group.throughput(Throughput::Elements(DIMENSIONS as u64));

    // f32
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<f32>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong f32", 0), &0, |b, _| {
        b.iter(|| NumKong::sqeuclidean(&inputs_f32.0, &inputs_f32.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled f32", 0), &0, |b, _| {
        b.iter(|| baseline_l2sq_unrolled::<f32, f32>(&inputs_f32.0, &inputs_f32.1))
    });

    // i8
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<i8>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong i8", 0), &0, |b, _| {
        b.iter(|| NumKong::sqeuclidean(&inputs_i8.0, &inputs_i8.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled i8", 0), &0, |b, _| {
        b.iter(|| baseline_l2sq_unrolled::<i8, i32>(&inputs_i8.0, &inputs_i8.1))
    });

    // u8
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<u8>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong u8", 0), &0, |b, _| {
        b.iter(|| NumKong::sqeuclidean(&inputs_u8.0, &inputs_u8.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled u8", 0), &0, |b, _| {
        b.iter(|| baseline_l2sq_unrolled::<u8, u32>(&inputs_u8.0, &inputs_u8.1))
    });
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

    let mut group = c.benchmark_group("Cosine Similarity");
    group.throughput(Throughput::Elements(DIMENSIONS as u64));

    // f32
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<f32>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong f32", 0), &0, |b, _| {
        b.iter(|| NumKong::cosine(&inputs_f32.0, &inputs_f32.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled u8", 0), &0, |b, _| {
        b.iter(|| baseline_cos_unrolled::<u8, u32>(&inputs_u8.0, &inputs_u8.1))
    });

    // i8
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<i8>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong i8", 0), &0, |b, _| {
        b.iter(|| NumKong::cosine(&inputs_i8.0, &inputs_i8.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled i8", 0), &0, |b, _| {
        b.iter(|| baseline_cos_unrolled::<i8, i32>(&inputs_i8.0, &inputs_i8.1))
    });

    // u8
    group.throughput(Throughput::Bytes((DIMENSIONS * size_of::<u8>()) as u64));
    group.bench_with_input(BenchmarkId::new("NumKong u8", 0), &0, |b, _| {
        b.iter(|| NumKong::cosine(&inputs_u8.0, &inputs_u8.1))
    });
    group.bench_with_input(BenchmarkId::new("Rust Unrolled f32", 0), &0, |b, _| {
        b.iter(|| baseline_cos_unrolled::<f32, f32>(&inputs_f32.0, &inputs_f32.1))
    });
}

// Criterion group definitions
criterion_group!(cos_benches, cos_benchmark);
criterion_group!(l2sq_benches, l2sq_benchmark);
criterion_main!(cos_benches, l2sq_benches);
