//! Numeric traits and implementations for vector operations.
//!
//! This module provides hardware-accelerated implementations of:
//!
//! - **Spatial similarity**: [`Dot`], [`Angular`], [`Euclidean`]
//! - **Binary similarity**: [`Hamming`], [`Jaccard`]
//! - **Probability divergence**: [`KullbackLeibler`], [`JensenShannon`]
//! - **Complex products**: [`ComplexDot`], [`ComplexVDot`], [`ComplexBilinear`]
//! - **Curved metrics**: [`Bilinear`], [`Mahalanobis`]
//! - **Elementwise operations**: [`EachScale`], [`EachSum`], [`EachBlend`], [`EachFMA`]
//! - **Trigonometry**: [`EachSin`], [`EachCos`], [`EachATan`]
//! - **Reductions**: [`ReduceMoments`], [`ReduceMinMax`]
//! - **Geospatial**: [`Haversine`], [`Vincenty`]
//! - **Mesh alignment**: [`MeshAlignment`]
//! - **Sparse sets**: [`SparseIntersect`], [`SparseDot`]
//! - **Type casting**: [`CastDtype`], [`cast`]
//! - **Capabilities**: [`cap`] module for runtime SIMD feature detection

use crate::scalars::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};

pub type ComplexProductF32 = (f32, f32);
pub type ComplexProductF64 = (f64, f64);

/// Size type used in C FFI to match `nk_size_t` which is always `uint64_t`.
type u64size = u64;

#[link(name = "numkong")]
extern "C" {
    // Capability detection
    fn nk_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_dynamic_dispatch() -> i32;
    fn nk_capabilities() -> u64;

    // Vector dot products
    fn nk_dot_i8(a: *const i8, b: *const i8, c: u64size, d: *mut i32);
    fn nk_dot_u8(a: *const u8, b: *const u8, c: u64size, d: *mut u32);
    fn nk_dot_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_e2m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_e3m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_dot_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_dot_f16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_bf16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_f32c(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_dot_f64c(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_vdot_f16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_vdot_bf16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_vdot_f32c(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_vdot_f64c(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    // Spatial similarity/distance functions
    fn nk_angular_i8(a: *const i8, b: *const i8, c: u64size, d: *mut f32);
    fn nk_angular_u8(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_angular_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_angular_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_angular_e2m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_angular_e3m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_angular_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_angular_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_sqeuclidean_i8(a: *const i8, b: *const i8, c: u64size, d: *mut u32);
    fn nk_sqeuclidean_u8(a: *const u8, b: *const u8, c: u64size, d: *mut u32);
    fn nk_sqeuclidean_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_e2m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_e3m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_sqeuclidean_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_euclidean_i8(a: *const i8, b: *const i8, c: u64size, d: *mut f32);
    fn nk_euclidean_u8(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_euclidean_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_euclidean_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_euclidean_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_euclidean_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_euclidean_e2m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_euclidean_e3m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_euclidean_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_euclidean_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_hamming_u1(a: *const u8, b: *const u8, c: u64size, d: *mut u32);
    fn nk_jaccard_u1(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_hamming_u8(a: *const u8, b: *const u8, n: u64size, result: *mut u32);
    fn nk_jaccard_u16(a: *const u16, b: *const u16, n: u64size, result: *mut f32);
    fn nk_jaccard_u32(a: *const u32, b: *const u32, n: u64size, result: *mut f32);

    // 4-bit integer kernels
    fn nk_dot_i4(a: *const u8, b: *const u8, n: u64size, result: *mut i32);
    fn nk_dot_u4(a: *const u8, b: *const u8, n: u64size, result: *mut u32);
    fn nk_sqeuclidean_i4(a: *const u8, b: *const u8, n: u64size, result: *mut u32);
    fn nk_sqeuclidean_u4(a: *const u8, b: *const u8, n: u64size, result: *mut u32);
    fn nk_euclidean_i4(a: *const u8, b: *const u8, n: u64size, result: *mut f32);
    fn nk_euclidean_u4(a: *const u8, b: *const u8, n: u64size, result: *mut f32);
    fn nk_angular_i4(a: *const u8, b: *const u8, n: u64size, result: *mut f32);
    fn nk_angular_u4(a: *const u8, b: *const u8, n: u64size, result: *mut f32);

    // Probability distribution distances/divergences
    fn nk_jsd_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_jsd_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_jsd_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_jsd_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_kld_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_kld_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_kld_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_kld_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    // Sparse sets
    fn nk_sparse_intersect_u16(
        a: *const u16,
        b: *const u16,
        a_length: u64size,
        b_length: u64size,
        result: *mut u16,
        count: *mut u64size,
    );
    fn nk_sparse_intersect_u32(
        a: *const u32,
        b: *const u32,
        a_length: u64size,
        b_length: u64size,
        result: *mut u32,
        count: *mut u64size,
    );
    fn nk_sparse_intersect_u64(
        a: *const u64,
        b: *const u64,
        a_length: u64size,
        b_length: u64size,
        result: *mut u64,
        count: *mut u64size,
    );
    fn nk_sparse_dot_u16bf16(
        a: *const u16,
        b: *const u16,
        a_weights: *const u16,
        b_weights: *const u16,
        a_length: u64size,
        b_length: u64size,
        product: *mut f32,
    );
    fn nk_sparse_dot_u32f32(
        a: *const u32,
        b: *const u32,
        a_weights: *const f32,
        b_weights: *const f32,
        a_length: u64size,
        b_length: u64size,
        product: *mut f32,
    );

    // Trigonometry functions
    fn nk_each_sin_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_each_sin_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_each_sin_f16(inputs: *const u16, n: u64size, outputs: *mut u16);
    fn nk_each_cos_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_each_cos_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_each_cos_f16(inputs: *const u16, n: u64size, outputs: *mut u16);
    fn nk_each_atan_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_each_atan_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_each_atan_f16(inputs: *const u16, n: u64size, outputs: *mut u16);

    // Elementwise operations
    fn nk_each_scale_f64(
        a: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_scale_f32(
        a: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_scale_f16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_bf16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_i8(
        a: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_scale_u8(
        a: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_i16(
        a: *const i16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i16,
    );
    fn nk_each_scale_u16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_i32(
        a: *const i32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i32,
    );
    fn nk_each_scale_u32(
        a: *const u32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u32,
    );
    fn nk_each_scale_i64(
        a: *const i64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i64,
    );
    fn nk_each_scale_u64(
        a: *const u64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u64,
    );
    fn nk_each_scale_e4m3(
        a: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e5m2(
        a: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e2m3(
        a: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e3m2(
        a: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_each_sum_f64(a: *const f64, b: *const f64, n: u64size, result: *mut f64);
    fn nk_each_sum_f32(a: *const f32, b: *const f32, n: u64size, result: *mut f32);
    fn nk_each_sum_f16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_each_sum_bf16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_each_sum_i8(a: *const i8, b: *const i8, n: u64size, result: *mut i8);
    fn nk_each_sum_u8(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_each_sum_i16(a: *const i16, b: *const i16, n: u64size, result: *mut i16);
    fn nk_each_sum_u16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_each_sum_i32(a: *const i32, b: *const i32, n: u64size, result: *mut i32);
    fn nk_each_sum_u32(a: *const u32, b: *const u32, n: u64size, result: *mut u32);
    fn nk_each_sum_i64(a: *const i64, b: *const i64, n: u64size, result: *mut i64);
    fn nk_each_sum_u64(a: *const u64, b: *const u64, n: u64size, result: *mut u64);
    fn nk_each_sum_e4m3(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_each_sum_e5m2(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_each_sum_e2m3(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_each_sum_e3m2(a: *const u8, b: *const u8, n: u64size, result: *mut u8);

    fn nk_each_blend_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_blend_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_blend_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_i8(
        a: *const i8,
        b: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_blend_u8(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_each_fma_f64(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_fma_f32(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_fma_f16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_fma_bf16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_fma_i8(
        a: *const i8,
        b: *const i8,
        c: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_fma_u8(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    // Complex elementwise operations (interleaved real/imag layout, n = number of complex pairs)
    fn nk_each_sum_f32c(a: *const f32, b: *const f32, n: u64size, result: *mut f32);
    fn nk_each_sum_f64c(a: *const f64, b: *const f64, n: u64size, result: *mut f64);
    fn nk_each_scale_f32c(
        a: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_scale_f64c(
        a: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_blend_f32c(
        a: *const f32,
        b: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_blend_f64c(
        a: *const f64,
        b: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_fma_f32c(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_fma_f64c(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );

    // Reductions: moments (sum + sum-of-squares)
    fn nk_reduce_moments_f64(
        data: *const f64,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f64,
        sumsq: *mut f64,
    );
    fn nk_reduce_moments_f32(
        data: *const f32,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f64,
        sumsq: *mut f64,
    );
    fn nk_reduce_moments_i8(
        data: *const i8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u8(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i16(
        data: *const i16,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i32(
        data: *const i32,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u32(
        data: *const u32,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i64(
        data: *const i64,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u64(
        data: *const u64,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );

    // Reductions: minmax (min + max + argmin + argmax)
    fn nk_reduce_minmax_f64(
        data: *const f64,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut f64,
        min_idx: *mut u64size,
        max_val: *mut f64,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_f32(
        data: *const f32,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut f32,
        min_idx: *mut u64size,
        max_val: *mut f32,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_i8(
        data: *const i8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut i8,
        min_idx: *mut u64size,
        max_val: *mut i8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u8(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_i16(
        data: *const i16,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut i16,
        min_idx: *mut u64size,
        max_val: *mut i16,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u16,
        min_idx: *mut u64size,
        max_val: *mut u16,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_i32(
        data: *const i32,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut i32,
        min_idx: *mut u64size,
        max_val: *mut i32,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u32(
        data: *const u32,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u32,
        min_idx: *mut u64size,
        max_val: *mut u32,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_i64(
        data: *const i64,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut i64,
        min_idx: *mut u64size,
        max_val: *mut i64,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u64(
        data: *const u64,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u64,
        min_idx: *mut u64size,
        max_val: *mut u64,
        max_idx: *mut u64size,
    );

    // Mesh superposition metrics
    fn nk_rmsd_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_rmsd_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_rmsd_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_rmsd_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_kabsch_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_umeyama_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );

    // Geospatial distance functions
    fn nk_haversine_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: u64size,
        results: *mut f32,
    );
    fn nk_haversine_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: u64size,
        results: *mut f64,
    );
    fn nk_vincenty_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: u64size,
        results: *mut f32,
    );
    fn nk_vincenty_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: u64size,
        results: *mut f64,
    );

    // Batch type casting
    fn nk_cast(
        from: *const core::ffi::c_void,
        from_type: u32,
        n: u64size,
        to: *mut core::ffi::c_void,
        to_type: u32,
    );

    // Bilinear form: aᵀ × C × b
    fn nk_bilinear_f64(a: *const f64, b: *const f64, c: *const f64, n: u64size, result: *mut f64);
    fn nk_bilinear_f32(a: *const f32, b: *const f32, c: *const f32, n: u64size, result: *mut f32);
    fn nk_bilinear_f16(a: *const u16, b: *const u16, c: *const u16, n: u64size, result: *mut f32);
    fn nk_bilinear_bf16(a: *const u16, b: *const u16, c: *const u16, n: u64size, result: *mut f32);
    fn nk_bilinear_f64c(a: *const f64, b: *const f64, c: *const f64, n: u64size, results: *mut f64);
    fn nk_bilinear_f32c(a: *const f32, b: *const f32, c: *const f32, n: u64size, results: *mut f32);
    fn nk_bilinear_f16c(a: *const u16, b: *const u16, c: *const u16, n: u64size, results: *mut f32);
    fn nk_bilinear_bf16c(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        results: *mut f32,
    );

    // Mahalanobis distance
    fn nk_mahalanobis_f64(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: u64size,
        result: *mut f64,
    );
    fn nk_mahalanobis_f32(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: u64size,
        result: *mut f32,
    );
    fn nk_mahalanobis_f16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        result: *mut f32,
    );
    fn nk_mahalanobis_bf16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        result: *mut f32,
    );

    fn nk_each_blend_e4m3(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e5m2(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e2m3(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e3m2(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_each_fma_e4m3(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e5m2(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e2m3(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e3m2(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_each_fma_i16(
        a: *const i16,
        b: *const i16,
        c: *const i16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        r: *mut i16,
    );
    fn nk_each_fma_u16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        r: *mut u16,
    );
    fn nk_each_fma_i32(
        a: *const i32,
        b: *const i32,
        c: *const i32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        r: *mut i32,
    );
    fn nk_each_fma_u32(
        a: *const u32,
        b: *const u32,
        c: *const u32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        r: *mut u32,
    );
    fn nk_each_fma_i64(
        a: *const i64,
        b: *const i64,
        c: *const i64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        r: *mut i64,
    );
    fn nk_each_fma_u64(
        a: *const u64,
        b: *const u64,
        c: *const u64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        r: *mut u64,
    );

    fn nk_reduce_moments_f16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_bf16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e4m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e5m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e2m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e3m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_i4(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u4(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u1(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        sum: *mut u64,
        sumsq: *mut u64,
    );

    fn nk_reduce_minmax_f16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u16,
        min_idx: *mut u64size,
        max_val: *mut u16,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_bf16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u16,
        min_idx: *mut u64size,
        max_val: *mut u16,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_e4m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_e5m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_e2m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_e3m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_i4(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut i8,
        min_idx: *mut u64size,
        max_val: *mut i8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u4(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );
    fn nk_reduce_minmax_u1(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_val: *mut u8,
        min_idx: *mut u64size,
        max_val: *mut u8,
        max_idx: *mut u64size,
    );

}

// region: Capabilities

/// Hardware capability detection functions.
pub mod capabilities {
    /// Returns the bitmask of available CPU capabilities.
    /// Use with `cap::*` constants to check for specific features.
    ///
    /// # Example
    /// ```
    /// use numkong::{capabilities, cap};
    ///
    /// let caps = capabilities::available();
    /// if caps & cap::NEON != 0 {
    ///     println!("NEON is available");
    /// }
    /// if caps & cap::SKYLAKE != 0 {
    ///     println!("AVX-512 (Skylake) is available");
    /// }
    /// ```
    pub fn available() -> u64 {
        unsafe { super::nk_capabilities() }
    }

    /// Configures the current thread for optimal SIMD performance.
    /// This includes flushing denormalized numbers to zero and enabling AMX on supported CPUs.
    /// Must be called once per thread before using AMX (Advanced Matrix Extensions) operations.
    pub fn configure_thread() -> bool {
        // Pass !0 to enable all capabilities including AMX
        unsafe { super::nk_configure_thread(!0) != 0 }
    }

    /// Returns `true` if the library uses dynamic dispatch for function selection.
    pub fn uses_dynamic_dispatch() -> bool {
        unsafe { super::nk_uses_dynamic_dispatch() != 0 }
    }
}

/// Capability bit masks in chronological order (by first commercial silicon).
pub mod cap {
    pub const SERIAL: u64 = 1 << 0; // Always: Fallback
    pub const NEON: u64 = 1 << 1; // 2013: ARM NEON
    pub const HASWELL: u64 = 1 << 2; // 2013: Intel AVX2
    pub const SKYLAKE: u64 = 1 << 3; // 2017: Intel AVX-512
    pub const NEONHALF: u64 = 1 << 4; // 2017: ARM NEON FP16
    pub const NEONSDOT: u64 = 1 << 5; // 2017: ARM NEON i8 dot
    pub const NEONFHM: u64 = 1 << 6; // 2018: ARM NEON FP16 FML
    pub const ICELAKE: u64 = 1 << 7; // 2019: Intel AVX-512 VNNI
    pub const GENOA: u64 = 1 << 8; // 2020: Intel/AMD AVX-512 BF16
    pub const NEONBFDOT: u64 = 1 << 9; // 2020: ARM NEON BF16
    pub const SVE: u64 = 1 << 10; // 2020: ARM SVE
    pub const SVEHALF: u64 = 1 << 11; // 2020: ARM SVE FP16
    pub const SVESDOT: u64 = 1 << 12; // 2020: ARM SVE i8 dot
    pub const SIERRA: u64 = 1 << 13; // 2021: Intel AVX2+VNNI
    pub const SVEBFDOT: u64 = 1 << 14; // 2021: ARM SVE BF16
    pub const SVE2: u64 = 1 << 15; // 2022: ARM SVE2
    pub const V128RELAXED: u64 = 1 << 16; // 2022: WASM Relaxed SIMD
    pub const SAPPHIRE: u64 = 1 << 17; // 2023: Intel AVX-512 FP16
    pub const SAPPHIREAMX: u64 = 1 << 18; // 2023: Intel Sapphire AMX
    pub const RVV: u64 = 1 << 19; // 2023: RISC-V Vector
    pub const RVVHALF: u64 = 1 << 20; // 2023: RISC-V Zvfh
    pub const RVVBF16: u64 = 1 << 21; // 2023: RISC-V Zvfbfwma
    pub const RVVBB: u64 = 1 << 33; // 2024: RISC-V Zvbb
    pub const GRANITEAMX: u64 = 1 << 22; // 2024: Intel Granite AMX FP16
    pub const TURIN: u64 = 1 << 23; // 2024: AMD Turin AVX-512 CD
    pub const SME: u64 = 1 << 24; // 2024: ARM SME
    pub const SME2: u64 = 1 << 25; // 2024: ARM SME2
    pub const SMEF64: u64 = 1 << 26; // 2024: ARM SME F64
    pub const SMEFA64: u64 = 1 << 27; // 2024: ARM SME FA64
    pub const SVE2P1: u64 = 1 << 28; // 2025+: ARM SVE2.1
    pub const SME2P1: u64 = 1 << 29; // 2025+: ARM SME2.1
    pub const SMEHALF: u64 = 1 << 30; // 2025+: ARM SME F16F16
    pub const SMEBF16: u64 = 1 << 31; // 2025+: ARM SME B16B16
    pub const SMELUT2: u64 = 1 << 32; // 2025+: ARM SME LUTv2
}

// endregion: Capabilities

// region: dtype (internal)

/// Internal dtype codes matching `nk_dtype_t` from C.
/// Not exposed to users.
mod dtype {
    pub(crate) const F64: u32 = 1 << 10;
    pub(crate) const F32: u32 = 1 << 11;
    pub(crate) const F16: u32 = 1 << 12;
    pub(crate) const BF16: u32 = 1 << 13;
    pub(crate) const E4M3: u32 = 1 << 14;
    pub(crate) const E5M2: u32 = 1 << 15;
    pub(crate) const E2M3: u32 = 1 << 18;
    pub(crate) const E3M2: u32 = 1 << 19;
    pub(crate) const I8: u32 = 1 << 2;
    pub(crate) const I16: u32 = 1 << 3;
    pub(crate) const I32: u32 = 1 << 4;
    pub(crate) const I64: u32 = 1 << 5;
    pub(crate) const U8: u32 = 1 << 6;
    pub(crate) const U16: u32 = 1 << 7;
    pub(crate) const U32: u32 = 1 << 8;
    pub(crate) const U64: u32 = 1 << 9;
}

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for super::f16 {}
    impl Sealed for super::bf16 {}
    impl Sealed for super::e4m3 {}
    impl Sealed for super::e5m2 {}
    impl Sealed for super::e2m3 {}
    impl Sealed for super::e3m2 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Trait for types that can participate in cast operations.
///
/// This trait is sealed - users cannot implement it for their own types.
pub trait CastDtype: private::Sealed {
    #[doc(hidden)]
    fn dtype_code() -> u32;
}

impl CastDtype for f64 {
    fn dtype_code() -> u32 {
        dtype::F64
    }
}
impl CastDtype for f32 {
    fn dtype_code() -> u32 {
        dtype::F32
    }
}
impl CastDtype for f16 {
    fn dtype_code() -> u32 {
        dtype::F16
    }
}
impl CastDtype for bf16 {
    fn dtype_code() -> u32 {
        dtype::BF16
    }
}
impl CastDtype for e4m3 {
    fn dtype_code() -> u32 {
        dtype::E4M3
    }
}
impl CastDtype for e5m2 {
    fn dtype_code() -> u32 {
        dtype::E5M2
    }
}
impl CastDtype for e2m3 {
    fn dtype_code() -> u32 {
        dtype::E2M3
    }
}
impl CastDtype for e3m2 {
    fn dtype_code() -> u32 {
        dtype::E3M2
    }
}
impl CastDtype for i8 {
    fn dtype_code() -> u32 {
        dtype::I8
    }
}
impl CastDtype for i16 {
    fn dtype_code() -> u32 {
        dtype::I16
    }
}
impl CastDtype for i32 {
    fn dtype_code() -> u32 {
        dtype::I32
    }
}
impl CastDtype for i64 {
    fn dtype_code() -> u32 {
        dtype::I64
    }
}
impl CastDtype for u8 {
    fn dtype_code() -> u32 {
        dtype::U8
    }
}
impl CastDtype for u16 {
    fn dtype_code() -> u32 {
        dtype::U16
    }
}
impl CastDtype for u32 {
    fn dtype_code() -> u32 {
        dtype::U32
    }
}
impl CastDtype for u64 {
    fn dtype_code() -> u32 {
        dtype::U64
    }
}

/// Cast source slice elements to destination slice.
///
/// Converts elements from source type `S` to destination type `D` using
/// hardware-accelerated SIMD operations when available.
///
/// # Arguments
/// * `source` - Source slice of elements to cast
/// * `dest` - Destination slice to receive cast elements (must be same length as source)
///
/// # Returns
/// * `Some(())` if successful
/// * `None` if slices have different lengths
///
/// # Example
/// ```ignore
/// use numkong::{f16, cast};
///
/// let f16_data: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
/// let mut f32_data: Vec<f32> = vec![0.0; f16_data.len()];
/// cast(&f16_data, &mut f32_data);
/// ```
pub fn cast<S: CastDtype, D: CastDtype>(source: &[S], dest: &mut [D]) -> Option<()> {
    if source.len() != dest.len() {
        return None;
    }
    unsafe {
        nk_cast(
            source.as_ptr() as *const core::ffi::c_void,
            S::dtype_code(),
            source.len() as u64size,
            dest.as_mut_ptr() as *mut core::ffi::c_void,
            D::dtype_code(),
        );
    }
    Some(())
}

// endregion: dtype (internal)

// region: Dot

/// Computes the **dot product** (inner product) between two vectors.
///
/// d = ∑ᵢ aᵢ × bᵢ
///
/// Range: unbounded. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
///
/// # Example
/// ```
/// use numkong::Dot;
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// let result = f32::dot(&a, &b).unwrap();
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
pub trait Dot: Sized {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `dot`.
    fn inner(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::dot(a, b)
    }
}

impl Dot for f64 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Dot for f32 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Dot for f16 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for bf16 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for i8 {
    type Output = i32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_dot_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Dot for u8 {
    type Output = u32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_dot_u8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Dot for e4m3 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for e5m2 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for e2m3 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for e3m2 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_dot_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for i4x2 {
    type Output = i32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let n = (a.len() * 2) as u64size; // Each i4x2 contains 2 elements
        unsafe {
            nk_dot_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for u4x2 {
    type Output = u32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let n = (a.len() * 2) as u64size; // Each u4x2 contains 2 elements
        unsafe {
            nk_dot_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Dot

// region: Angular

/// Computes the **angular distance** (cosine distance) between two vectors.
///
/// d = 1 − (a · b) / (‖a‖ × ‖b‖)
///
/// Range: \[0, 2\]. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
pub trait Angular: Sized {
    type Output;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `angular`.
    fn cosine(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::angular(a, b)
    }
}

impl Angular for f64 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Angular for f32 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Angular for f16 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for bf16 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for i8 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Angular for u8 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_u8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Angular for e4m3 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for e5m2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for e2m3 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for e3m2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_angular_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for i4x2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let n = (a.len() * 2) as u64size; // Each i4x2 contains 2 elements
        unsafe {
            nk_angular_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for u4x2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let n = (a.len() * 2) as u64size; // Each u4x2 contains 2 elements
        unsafe {
            nk_angular_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Angular

// region: Euclidean

/// Computes the **Euclidean distance** (L2) between two vectors.
///
/// d = √(∑ᵢ (aᵢ − bᵢ)²)
///
/// Range: \[0, ∞). Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
pub trait Euclidean: Sized {
    type SqEuclideanOutput;
    type EuclideanOutput;

    /// Squared Euclidean distance (L2²). Faster than `euclidean` for comparisons.
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput>;

    /// Euclidean distance (L2). True metric distance.
    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput>;
}

impl Euclidean for f64 {
    type SqEuclideanOutput = f64;
    type EuclideanOutput = f64;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f32 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f16 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for bf16 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for i8 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        unsafe { nk_sqeuclidean_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for u8 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        unsafe { nk_sqeuclidean_u8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_u8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for e4m3 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for e5m2 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for e2m3 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for e3m2 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe {
            nk_sqeuclidean_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe {
            nk_euclidean_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for i4x2 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        let n = (a.len() * 2) as u64size; // Each i4x2 contains 2 elements
        unsafe {
            nk_sqeuclidean_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        let n = (a.len() * 2) as u64size; // Each i4x2 contains 2 elements
        unsafe {
            nk_euclidean_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for u4x2 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        let n = (a.len() * 2) as u64size; // Each u4x2 contains 2 elements
        unsafe {
            nk_sqeuclidean_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        let n = (a.len() * 2) as u64size; // Each u4x2 contains 2 elements
        unsafe {
            nk_euclidean_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Euclidean

// region: Geospatial

/// Computes **great-circle distances** between geographic coordinates on Earth.
///
/// Uses the Haversine formula for spherical Earth approximation:
///
/// - `a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)`
/// - `c = 2 × atan2(√a, √(1−a))`
/// - `d = R × c`
///
/// Where φ = latitude, λ = longitude, R = Earth's radius (6335 km).
/// Inputs are in radians, outputs in meters.
pub trait Haversine: Sized {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Computes **Vincenty geodesic distances** on the WGS84 ellipsoid.
///
/// Uses Vincenty's iterative formula for oblate spheroid geodesics:
///
/// 1. Reduced latitudes: `tan(U) = (1−f) × tan(φ)`
/// 2. Iterate until convergence: `λ → L + (1−C) × f × sin(α) × [σ + C × sin(σ) × ...]`
/// 3. Compute: `u² = cos²(α) × (a² − b²)/b²`
/// 4. Series coefficients A, B from u²
/// 5. Distance: `s = b × A × (σ − Δσ)`
///
/// Where a = equatorial radius, b = polar radius, f = flattening.
/// ~20× more accurate than Haversine for long distances.
/// Inputs are in radians, outputs in meters.
pub trait Vincenty: Sized {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Combined trait for all geospatial distance computations.
pub trait Geospatial: Haversine + Vincenty {}

impl Haversine for f64 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f64 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f64 {}

impl Haversine for f32 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f32 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f32 {}

// endregion: Geospatial

// region: Hamming

/// Computes the **Hamming distance** between two binary vectors.
///
/// Counts differing bits (for `u1x8`) or differing bytes (for `u8`).
///
/// Range: \[0, n\]. Returns `None` if lengths differ.
///
/// Implemented for: `u1x8`, `u8`.
pub trait Hamming: Sized {
    type Output;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Hamming for u1x8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let n_bits = (a.len() * 8) as u64size; // Each u1x8 contains 8 bits
        unsafe {
            nk_hamming_u1(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Hamming for u8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_hamming_u8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Hamming

// region: Jaccard

/// Computes the **Jaccard distance** between two sets represented as bit/integer vectors.
///
/// d = 1 − |A ∩ B| / |A ∪ B|
///
/// Range: \[0, 1\]. Returns `None` if lengths differ.
///
/// Implemented for: `u1x8`, `u16`, `u32`.
pub trait Jaccard: Sized {
    type Output;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Jaccard for u1x8 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let n_bits = (a.len() * 8) as u64size; // Each u1x8 contains 8 bits
        unsafe {
            nk_jaccard_u1(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Jaccard for u16 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_u16(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Jaccard for u32 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_u32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Jaccard

// region: KullbackLeibler

/// Computes the **Kullback-Leibler divergence** between two probability distributions.
///
/// D_KL(P‖Q) = ∑ᵢ pᵢ × ln(pᵢ / qᵢ)
///
/// Range: \[0, ∞). Not symmetric. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait KullbackLeibler: Sized {
    type Output;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `kullbackleibler`.
    fn kl(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::kullbackleibler(a, b)
    }
}

impl KullbackLeibler for f64 {
    type Output = f64;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl KullbackLeibler for f32 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl KullbackLeibler for f16 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_kld_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl KullbackLeibler for bf16 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_kld_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: KullbackLeibler

// region: JensenShannon

/// Computes the **Jensen-Shannon divergence** between two probability distributions.
///
/// JS(P, Q) = ½(D_KL(P‖M) + D_KL(Q‖M)), where M = (P + Q) / 2
///
/// Range: \[0, ln2\]. Symmetric. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait JensenShannon: Sized {
    type Output;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `jensenshannon`.
    fn js(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::jensenshannon(a, b)
    }
}

impl JensenShannon for f64 {
    type Output = f64;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl JensenShannon for f32 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl JensenShannon for f16 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jsd_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl JensenShannon for bf16 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jsd_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: JensenShannon

// region: ComplexDot

/// Computes the **complex dot product** between two complex vectors.
///
/// z = ∑ᵢ aᵢ × bᵢ (complex multiplication, interleaved real/imag layout)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait ComplexDot: Sized {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl ComplexDot for f64 {
    type Output = ComplexProductF64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f64; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for f32 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for f16 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for bf16 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

// endregion: ComplexDot

// region: ComplexVDot

/// Computes the **conjugate dot product** (Hermitian inner product) between complex vectors.
///
/// z = ∑ᵢ conj(aᵢ) × bᵢ (complex multiplication with conjugated first operand)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait ComplexVDot: Sized {
    type Output;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl ComplexVDot for f64 {
    type Output = ComplexProductF64;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f64; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for f32 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for f16 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for bf16 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

// endregion: ComplexVDot

// region: SparseIntersect

/// Computes set operations on sorted sparse vectors.
pub trait SparseIntersect: Sized {
    /// Returns the intersection size between two sorted sparse vectors.
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize;

    /// Computes intersection and writes matching elements to output buffer.
    /// Buffer must be at least `min(a.len(), b.len())` in size.
    /// Returns `Some(count)` with number of elements written, or `None` if buffer too small.
    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize>;
}

impl SparseIntersect for u16 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count as usize
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count as usize)
    }
}

impl SparseIntersect for u32 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count as usize
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count as usize)
    }
}

impl SparseIntersect for u64 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count as usize
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: u64size = 0;
        unsafe {
            nk_sparse_intersect_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count as usize)
    }
}

// endregion: SparseIntersect

// region: SparseDot

/// Computes sparse dot product between two sorted sparse vectors with weights.
///
/// Each vector consists of sorted indices and corresponding weights. The dot product
/// is computed over the intersection of indices, summing the products of weights.
pub trait SparseDot: Sized {
    /// Weight type for this sparse dot product.
    type Weight;

    /// Computes sparse dot product.
    ///
    /// Returns the sum of `a_weights[i] × b_weights[j]` for all pairs where `a_indices[i] == b_indices[j]`.
    fn sparse_dot(
        a_indices: &[Self],
        b_indices: &[Self],
        a_weights: &[Self::Weight],
        b_weights: &[Self::Weight],
    ) -> f32;
}

impl SparseDot for u16 {
    type Weight = bf16;

    fn sparse_dot(
        a_indices: &[Self],
        b_indices: &[Self],
        a_weights: &[bf16],
        b_weights: &[bf16],
    ) -> f32 {
        let mut product: f32 = 0.0;
        unsafe {
            nk_sparse_dot_u16bf16(
                a_indices.as_ptr(),
                b_indices.as_ptr(),
                a_weights.as_ptr() as *const u16,
                b_weights.as_ptr() as *const u16,
                a_indices.len() as u64size,
                b_indices.len() as u64size,
                &mut product,
            );
        }
        product
    }
}

impl SparseDot for u32 {
    type Weight = f32;

    fn sparse_dot(
        a_indices: &[Self],
        b_indices: &[Self],
        a_weights: &[f32],
        b_weights: &[f32],
    ) -> f32 {
        let mut product: f32 = 0.0;
        unsafe {
            nk_sparse_dot_u32f32(
                a_indices.as_ptr(),
                b_indices.as_ptr(),
                a_weights.as_ptr(),
                b_weights.as_ptr(),
                a_indices.len() as u64size,
                b_indices.len() as u64size,
                &mut product,
            );
        }
        product
    }
}

// endregion: SparseDot

// region: EachSin

/// Computes **element-wise sine** of a vector.
pub trait EachSin: Sized {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachSin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_sin_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_sin_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSin for f16 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_sin_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachSin

// region: EachCos

/// Computes **element-wise cosine** of a vector.
pub trait EachCos: Sized {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachCos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_cos_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachCos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_cos_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachCos for f16 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_cos_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachCos

// region: EachATan

/// Computes **element-wise arctangent** (inverse tangent) of a vector.
pub trait EachATan: Sized {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachATan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_atan_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachATan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_atan_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachATan for f16 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_atan_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachATan

// region: Scale

/// Applies an **element-wise affine transform** (scale and shift).
///
/// rᵢ = α × aᵢ + β
///
/// Returns `None` if `a` and `result` lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachScale: Sized {
    type Scalar;
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl EachScale for f64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for f32 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for f16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f16(
                a.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachScale for bf16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_bf16(
                a.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachScale for i8 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_i8(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for u8 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_u8(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for i16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_i16(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for u16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_u16(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for i32 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_i32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for u32 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_u32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for i64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_i64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for u64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_u64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachScale for e4m3 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e4m3(
                a.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachScale for e5m2 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e5m2(
                a.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachScale for e2m3 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e2m3(
                a.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachScale for e3m2 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e3m2(
                a.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

// endregion: Scale

// region: Sum

/// Applies **element-wise addition** of two vectors.
///
/// rᵢ = aᵢ + bᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachSum: Sized {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()>;
}

impl EachSum for f64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for f32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for f16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachSum for bf16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachSum for i8 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for u8 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for i16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_i16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for u16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for i32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_i32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for u32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for i64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_i64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for u64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachSum for e4m3 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachSum for e5m2 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachSum for e2m3 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachSum for e3m2 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

// endregion: Sum

// region: WSum

/// Applies **element-wise weighted sum** (blend) of two vectors.
///
/// rᵢ = α × aᵢ + β × bᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachBlend: Sized {
    type Scalar;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl EachBlend for f64 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for f32 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for f16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachBlend for bf16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachBlend for i8 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for u8 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for e4m3 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e5m2 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e2m3 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e3m2 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

// endregion: WSum

// region: FMA

/// Applies **fused multiply-add** element-wise across three vectors.
///
/// rᵢ = α × aᵢ × bᵢ + β × cᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachFMA: Sized {
    type Scalar;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl EachFMA for f64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for f32 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for f16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachFMA for bf16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl EachFMA for i8 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for u8 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for e4m3 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e5m2 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e2m3 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e3m2 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for i16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i16(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for u16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u16(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for i32 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for u32 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for i64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachFMA for u64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: FMA

// region: Reductions

/// Compute first and second moments (sum and sum-of-squares) with stride support.
///
/// Returns `(sum, sum_of_squares)` for all elements in a slice, with optional striding.
/// The output types may be wider than the input to avoid overflow.
pub trait ReduceMoments: Sized {
    /// Type for the sum output.
    type SumOutput;
    /// Type for the sum-of-squares output.
    type SumSqOutput;
    /// Compute `(sum, sum_of_squares)` for `data` with the given stride (in bytes).
    /// Use `stride_bytes = size_of::<Self>()` for contiguous data.
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput);
}

impl ReduceMoments for f64 {
    type SumOutput = f64;
    type SumSqOutput = f64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f64 = 0.0;
        let mut sumsq: f64 = 0.0;
        unsafe {
            nk_reduce_moments_f64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for f32 {
    type SumOutput = f64;
    type SumSqOutput = f64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f64 = 0.0;
        let mut sumsq: f64 = 0.0;
        unsafe {
            nk_reduce_moments_f32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for i8 {
    type SumOutput = i64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: i64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_i8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u8 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for i16 {
    type SumOutput = i64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: i64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_i16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u16 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for i32 {
    type SumOutput = i64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: i64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_i32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u32 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for i64 {
    type SumOutput = i64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: i64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_i64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u64 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for f16 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_f16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for bf16 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_bf16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for e4m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_e4m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for e5m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_e5m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for e2m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_e2m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for e3m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: f32 = 0.0;
        let mut sumsq: f32 = 0.0;
        unsafe {
            nk_reduce_moments_e3m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for i4x2 {
    type SumOutput = i64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: i64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_i4(
                data.as_ptr() as *const u8,
                (data.len() * 2) as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u4x2 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u4(
                data.as_ptr() as *const u8,
                (data.len() * 2) as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

impl ReduceMoments for u1x8 {
    type SumOutput = u64;
    type SumSqOutput = u64;
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        let mut sum: u64 = 0;
        let mut sumsq: u64 = 0;
        unsafe {
            nk_reduce_moments_u1(
                data.as_ptr() as *const u8,
                (data.len() * 8) as u64size,
                stride_bytes as u64size,
                &mut sum,
                &mut sumsq,
            );
        }
        (sum, sumsq)
    }
}

/// Find minimum and maximum values with their indices, with stride support.
///
/// Returns `(min_value, min_index, max_value, max_index)` for all elements in a slice.
/// The value output type may be widened for half-precision types.
pub trait ReduceMinMax: Sized {
    /// Output type for the min/max values — matches the C layer's native type.
    type Output;
    /// Returns `(min_value, min_index, max_value, max_index)` for the given data with the specified stride.
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize);
}

impl ReduceMinMax for f64 {
    type Output = f64;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: f64 = 0.0;
        let mut min_idx: u64size = 0;
        let mut max_val: f64 = 0.0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_f64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for f32 {
    type Output = f32;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: f32 = 0.0;
        let mut min_idx: u64size = 0;
        let mut max_val: f32 = 0.0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_f32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for i8 {
    type Output = i8;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: i8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: i8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_i8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u8 {
    type Output = u8;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for i16 {
    type Output = i16;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: i16 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: i16 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_i16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u16 {
    type Output = u16;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u16 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u16 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for i32 {
    type Output = i32;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: i32 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: i32 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_i32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u32 {
    type Output = u32;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u32 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u32 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for i64 {
    type Output = i64;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: i64 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: i64 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_i64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u64 {
    type Output = u64;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u64 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u64 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for f16 {
    type Output = f16;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u16 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u16 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_f16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            f16(min_raw),
            min_idx as usize,
            f16(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for bf16 {
    type Output = bf16;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u16 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u16 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_bf16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            bf16(min_raw),
            min_idx as usize,
            bf16(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for e4m3 {
    type Output = e4m3;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_e4m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            e4m3(min_raw),
            min_idx as usize,
            e4m3(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for e5m2 {
    type Output = e5m2;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_e5m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            e5m2(min_raw),
            min_idx as usize,
            e5m2(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for e2m3 {
    type Output = e2m3;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_e2m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            e2m3(min_raw),
            min_idx as usize,
            e2m3(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for e3m2 {
    type Output = e3m2;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_raw: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_raw: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_e3m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_raw,
                &mut min_idx,
                &mut max_raw,
                &mut max_idx,
            );
        }
        (
            e3m2(min_raw),
            min_idx as usize,
            e3m2(max_raw),
            max_idx as usize,
        )
    }
}

impl ReduceMinMax for i4x2 {
    type Output = i8;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: i8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: i8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_i4(
                data.as_ptr() as *const u8,
                (data.len() * 2) as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u4x2 {
    type Output = u8;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u4(
                data.as_ptr() as *const u8,
                (data.len() * 2) as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

impl ReduceMinMax for u1x8 {
    type Output = u8;
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> (Self::Output, usize, Self::Output, usize) {
        let mut min_val: u8 = 0;
        let mut min_idx: u64size = 0;
        let mut max_val: u8 = 0;
        let mut max_idx: u64size = 0;
        unsafe {
            nk_reduce_minmax_u1(
                data.as_ptr() as *const u8,
                (data.len() * 8) as u64size,
                stride_bytes as u64size,
                &mut min_val,
                &mut min_idx,
                &mut max_val,
                &mut max_idx,
            );
        }
        (min_val, min_idx as usize, max_val, max_idx as usize)
    }
}

// endregion: Reductions

// region: MeshAlignment

/// Result of mesh alignment operations (RMSD, Kabsch, Umeyama).
///
/// Contains the rigid-body transformation (rotation, scale, translation)
/// that best aligns point cloud A onto point cloud B, along with the
/// root-mean-square deviation of the aligned points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAlignmentResult<T> {
    /// 3×3 rotation matrix in row-major order.
    pub rotation_matrix: [T; 9],
    /// Uniform scale factor (1.0 for Kabsch, free for Umeyama).
    pub scale: T,
    /// Root-mean-square deviation after alignment.
    pub rmsd: T,
    /// Centroid of point cloud A before alignment.
    pub a_centroid: [T; 3],
    /// Centroid of point cloud B (target).
    pub b_centroid: [T; 3],
}

impl MeshAlignmentResult<f64> {
    #[inline]
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

impl MeshAlignmentResult<f32> {
    #[inline]
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Mesh alignment operations for 3D point clouds.
pub trait MeshAlignment: Sized {
    /// Output type for results. f64/f32 use themselves, f16/bf16 use f32.
    type Output: Default + Copy;

    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>>;
    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>>;
    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>>;
}

impl MeshAlignment for f64 {
    type Output = f64;

    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for f32 {
    type Output = f32;

    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for f16 {
    type Output = f32;

    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for bf16 {
    type Output = f32;

    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self::Output>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

// endregion: MeshAlignment

// region: Bilinear Form

/// Bilinear form computation: aᵀ × C × b where C is a metric tensor.
///
/// Computes the bilinear form of two vectors `a` and `b` with respect to
/// a symmetric matrix `C` (given in row-major order as a flat slice of length n²).
pub trait Bilinear: Sized {
    /// Output type for results. f64/f32 use themselves, f16/bf16 use f32.
    type Output;

    /// Computes the bilinear form aᵀ × C × b.
    ///
    /// # Arguments
    /// * `a` - First vector of length n
    /// * `b` - Second vector of length n
    /// * `c` - Metric tensor (n×n matrix in row-major order, length n²)
    ///
    /// # Returns
    /// `Some(result)` if inputs are valid, `None` if lengths are incompatible.
    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output>;
}

impl Bilinear for f64 {
    type Output = f64;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_bilinear_f64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Bilinear for f32 {
    type Output = f32;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_bilinear_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Bilinear for f16 {
    type Output = f32;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_bilinear_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Bilinear for bf16 {
    type Output = f32;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_bilinear_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

// endregion: Bilinear Form

// region: Complex Bilinear Form

/// Complex bilinear form computation: aᴴ × C × b where inputs are interleaved complex vectors.
///
/// Input data is interleaved `[real, imag, real, imag, ...]`. Returns `(real, imag)`.
/// The `n` parameter to the C function is the number of complex elements (half the slice length).
pub trait ComplexBilinear: Sized {
    type Output;
    fn complex_bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output>;
}

impl ComplexBilinear for f64 {
    type Output = ComplexProductF64;

    fn complex_bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || n != b.len() || n % 2 != 0 || c.len() != (n / 2) * (n / 2) * 2 {
            return None;
        }
        let mut result: [f64; 2] = [0.0, 0.0];
        unsafe {
            nk_bilinear_f64c(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                (n / 2) as u64size,
                result.as_mut_ptr(),
            );
        }
        Some((result[0], result[1]))
    }
}

impl ComplexBilinear for f32 {
    type Output = ComplexProductF32;

    fn complex_bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || n != b.len() || n % 2 != 0 || c.len() != (n / 2) * (n / 2) * 2 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_bilinear_f32c(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                (n / 2) as u64size,
                result.as_mut_ptr(),
            );
        }
        Some((result[0], result[1]))
    }
}

impl ComplexBilinear for f16 {
    type Output = ComplexProductF32;

    fn complex_bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || n != b.len() || n % 2 != 0 || c.len() != (n / 2) * (n / 2) * 2 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_bilinear_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                (n / 2) as u64size,
                result.as_mut_ptr(),
            );
        }
        Some((result[0], result[1]))
    }
}

impl ComplexBilinear for bf16 {
    type Output = ComplexProductF32;

    fn complex_bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || n != b.len() || n % 2 != 0 || c.len() != (n / 2) * (n / 2) * 2 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_bilinear_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                (n / 2) as u64size,
                result.as_mut_ptr(),
            );
        }
        Some((result[0], result[1]))
    }
}

// endregion: Complex Bilinear Form

// region: Complex Elementwise

/// Applies **complex element-wise addition** of two interleaved complex vectors.
///
/// rᵢ = aᵢ + bᵢ (complex addition)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`.
pub trait ComplexEachSum: Sized {
    fn complex_each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()>;
}

impl ComplexEachSum for f64 {
    fn complex_each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_sum_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ComplexEachSum for f32 {
    fn complex_each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_sum_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

/// Applies a **complex element-wise affine transform** (scale and shift).
///
/// rᵢ = α × aᵢ + β (complex multiply and add)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Coefficients `alpha` and `beta` are `[real, imag]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`.
pub trait ComplexEachScale: Sized {
    fn complex_each_scale(
        a: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()>;
}

impl ComplexEachScale for f64 {
    fn complex_each_scale(
        a: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_scale_f64c(
                a.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ComplexEachScale for f32 {
    fn complex_each_scale(
        a: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_scale_f32c(
                a.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

/// Applies **complex element-wise weighted sum** (blend) of two interleaved complex vectors.
///
/// rᵢ = α × aᵢ + β × bᵢ (complex multiply and add)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Coefficients `alpha` and `beta` are `[real, imag]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`.
pub trait ComplexEachBlend: Sized {
    fn complex_each_blend(
        a: &[Self],
        b: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()>;
}

impl ComplexEachBlend for f64 {
    fn complex_each_blend(
        a: &[Self],
        b: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_blend_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ComplexEachBlend for f32 {
    fn complex_each_blend(
        a: &[Self],
        b: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_blend_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

/// Applies **complex fused multiply-add** element-wise across three interleaved complex vectors.
///
/// rᵢ = α × aᵢ × bᵢ + β × cᵢ (complex multiply chain)
///
/// Input slices contain interleaved `[re₀, im₀, re₁, im₁, ...]` pairs.
/// Coefficients `alpha` and `beta` are `[real, imag]` pairs.
/// Returns `None` if lengths differ or are odd.
///
/// Implemented for: `f64`, `f32`.
pub trait ComplexEachFMA: Sized {
    fn complex_each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()>;
}

impl ComplexEachFMA for f64 {
    fn complex_each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_fma_f64c(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ComplexEachFMA for f32 {
    fn complex_each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: [Self; 2],
        beta: [Self; 2],
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() || a.len() % 2 != 0 {
            return None;
        }
        unsafe {
            nk_each_fma_f32c(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size / 2,
                alpha.as_ptr(),
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Complex Elementwise

// region: Mahalanobis Distance

/// Mahalanobis distance: √((a−b)ᵀ × C × (a−b)).
///
/// Computes the Mahalanobis distance between two vectors `a` and `b` with respect
/// to an inverse covariance matrix `C` (given in row-major order as a flat slice of length n²).
pub trait Mahalanobis: Sized {
    /// Output type for results. f64/f32 use themselves, f16/bf16 use f32.
    type Output;

    /// Computes the Mahalanobis distance √((a−b)ᵀ × C × (a−b)).
    ///
    /// # Arguments
    /// * `a` - First vector of length n
    /// * `b` - Second vector of length n
    /// * `c` - Inverse covariance matrix (n×n matrix in row-major order, length n²)
    ///
    /// # Returns
    /// `Some(result)` if inputs are valid, `None` if lengths are incompatible.
    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output>;
}

impl Mahalanobis for f64 {
    type Output = f64;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_mahalanobis_f64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Mahalanobis for f32 {
    type Output = f32;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_mahalanobis_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Mahalanobis for f16 {
    type Output = f32;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_mahalanobis_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Mahalanobis for bf16 {
    type Output = f32;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let n = a.len();
        if n == 0 || b.len() != n || c.len() != n * n {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_mahalanobis_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                n as u64size,
                &mut result,
            );
        }
        Some(result)
    }
}

// endregion: Mahalanobis Distance

// region: Convenience Trait Aliases

/// `SpatialSimilarity` bundles spatial distance metrics: Dot, Angular, and Euclidean.
pub trait SpatialSimilarity: Dot + Angular + Euclidean {}
impl<T: Dot + Angular + Euclidean> SpatialSimilarity for T {}

/// `BinarySimilarity` bundles binary distance metrics: Hamming and Jaccard.
pub trait BinarySimilarity: Hamming + Jaccard {}
impl<T: Hamming + Jaccard> BinarySimilarity for T {}

/// `ProbabilitySimilarity` bundles probability divergence metrics: KullbackLeibler and JensenShannon.
pub trait ProbabilitySimilarity: KullbackLeibler + JensenShannon {}
impl<T: KullbackLeibler + JensenShannon> ProbabilitySimilarity for T {}

/// `ComplexProducts` bundles complex number products: ComplexDot and ComplexVDot.
pub trait ComplexProducts: ComplexDot + ComplexVDot {}
impl<T: ComplexDot + ComplexVDot> ComplexProducts for T {}

/// `Trigonometry` bundles trigonometric functions: EachSin, EachCos, and EachATan.
pub trait Trigonometry: EachSin + EachCos + EachATan {}
impl<T: EachSin + EachCos + EachATan> Trigonometry for T {}

/// `Reductions` bundles reduction operations: ReduceMoments and ReduceMinMax.
pub trait Reductions: ReduceMoments + ReduceMinMax {}
impl<T: ReduceMoments + ReduceMinMax> Reductions for T {}

// endregion: Convenience Trait Aliases

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalars::{assert_close, FloatLike, TestableType};

    // region: Core Test Helpers

    /// Test a two-input metric: convert f32 inputs to T, call `op`, compare to `expected`.
    fn check_binary<T, R, F>(a_vals: &[f32], b_vals: &[f32], op: F, expected: f64, label: &str)
    where
        T: FloatLike + TestableType,
        R: FloatLike,
        F: FnOnce(&[T], &[T]) -> Option<R>,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result = op(&a, &b).unwrap().to_f64();
        assert_close(
            result,
            expected,
            T::atol(),
            T::rtol(),
            &format!("{}<{}>", label, core::any::type_name::<T>()),
        );
    }

    /// Test a binary elementwise op: convert inputs, apply `op`, compare element-wise.
    fn check_each_binary<T, F>(
        a_vals: &[f32],
        b_vals: &[f32],
        op: F,
        expected_fn: fn(f64, f64) -> f64,
        label: &str,
    ) where
        T: FloatLike + TestableType,
        F: FnOnce(&[T], &[T], &mut [T]) -> Option<()>,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let mut result = vec![T::zero(); a.len()];
        op(&a, &b, &mut result).unwrap();
        for (i, &r) in result.iter().enumerate() {
            let expected = expected_fn(a_vals[i] as f64, b_vals[i] as f64);
            assert_close(
                r.to_f64(),
                expected,
                T::atol(),
                T::rtol(),
                &format!("{}<{}>[{i}]", label, core::any::type_name::<T>()),
            );
        }
    }

    /// Test a unary elementwise op over generated values.
    fn check_each_unary<T, F>(
        count: usize,
        gen_fn: fn(usize, usize) -> f64,
        op: F,
        ref_fn: fn(f64) -> f64,
        label: &str,
    ) where
        T: FloatLike + TestableType,
        F: FnOnce(&[T], &mut [T]) -> Option<()>,
    {
        let values: Vec<f64> = (0..count).map(|i| gen_fn(i, count)).collect();
        let a: Vec<T> = values.iter().map(|&v| T::from_f32(v as f32)).collect();
        let mut result = vec![T::zero(); count];
        op(&a, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = ref_fn(values[i]);
            assert_close(
                r.to_f64(),
                expected,
                T::atol() * 10000.0,
                T::rtol() * 10000.0,
                &format!("{}<{}>[{}]", label, core::any::type_name::<T>(), i),
            );
        }
    }

    /// Build an identity matrix of size n*n.
    fn make_identity<T: FloatLike>(n: usize) -> Vec<T> {
        let mut v = vec![T::zero(); n * n];
        for i in 0..n {
            v[i * n + i] = T::one();
        }
        v
    }

    /// Convert a point cloud from f32 to T.
    fn convert_cloud<T: FloatLike>(cloud: &[[f32; 3]]) -> Vec<[T; 3]> {
        cloud
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect()
    }

    // endregion

    // region: Binary Distances

    #[test]
    fn hamming() {
        // u1x8
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b00001111), u1x8(0b01010101)];
        assert_eq!(u1x8::hamming(&a, &b).unwrap(), 16);

        // u8
        let a: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let b: Vec<u8> = vec![0, 1, 2, 3, 0, 0, 0, 0];
        assert_eq!(u8::hamming(&a, &b).unwrap(), 4);
    }

    #[test]
    fn jaccard() {
        // u1x8 — identical
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b11110000), u1x8(0b10101010)];
        assert_close(
            u1x8::jaccard(&a, &b).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u1x8",
        );

        // u16 — identical
        let a: Vec<u16> = vec![1, 2, 3, 4];
        let b: Vec<u16> = vec![1, 2, 3, 4];
        assert_close(
            u16::jaccard(&a, &b).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u16 identical",
        );
        // u16 — disjoint
        let c: Vec<u16> = vec![5, 6, 7, 8];
        assert_close(
            u16::jaccard(&a, &c).unwrap() as f64,
            1.0,
            0.01,
            0.0,
            "jaccard_u16 disjoint",
        );

        // u32 — partial overlap
        let a: Vec<u32> = vec![1, 2, 3, 4];
        let b: Vec<u32> = vec![1, 2, 5, 6];
        assert_close(
            u32::jaccard(&a, &b).unwrap() as f64,
            0.5,
            0.01,
            0.0,
            "jaccard_u32",
        );
    }

    // endregion

    // region: Sparse Intersections

    #[test]
    fn sparse_intersection() {
        // u16 — intersection size
        let a: Vec<u16> = vec![1, 3, 5, 7, 9];
        let b: Vec<u16> = vec![2, 3, 5, 8, 9];
        assert_eq!(u16::sparse_intersection_size(&a, &b), 3);

        // u16 — intersect into buffer
        let mut result: Vec<u16> = vec![0; 5];
        let count = u16::sparse_intersect_into(&a, &b, &mut result).unwrap();
        assert_eq!(count, 3);
        assert_eq!(&result[..count], &[3, 5, 9]);

        // u32 — intersect into buffer
        let a: Vec<u32> = vec![10, 20, 30, 40];
        let b: Vec<u32> = vec![15, 20, 30, 45];
        let mut result: Vec<u32> = vec![0; 4];
        let count = u32::sparse_intersect_into(&a, &b, &mut result).unwrap();
        assert_eq!(count, 2);
        assert_eq!(&result[..count], &[20, 30]);

        // u64 — intersection size
        let a: Vec<u64> = vec![100, 200, 300];
        let b: Vec<u64> = vec![200, 300, 400];
        assert_eq!(u64::sparse_intersection_size(&a, &b), 2);
    }

    #[test]
    fn sparse_intersect_into_buffer_too_small() {
        let a: Vec<u16> = vec![1, 2, 3, 4, 5];
        let b: Vec<u16> = vec![3, 4, 5, 6, 7];
        let mut result: Vec<u16> = vec![0; 2];
        assert!(u16::sparse_intersect_into(&a, &b, &mut result).is_none());
    }

    // endregion

    // region: Dot Products

    fn check_dot<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Dot,
        T::Output: FloatLike,
    {
        check_binary::<T, T::Output, _>(a_vals, b_vals, T::dot, expected, "dot");
    }

    #[test]
    fn dot() {
        check_dot::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 6.0);
        check_dot::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
        check_dot::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
    }

    // endregion

    // region: Angular Distances

    fn check_angular<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Angular,
        T::Output: FloatLike,
    {
        check_binary::<T, T::Output, _>(a_vals, b_vals, T::angular, expected, "angular");
    }

    #[test]
    fn angular() {
        // angular([1,2,3],[4,5,6]) = 1 - 32/sqrt(14*77) ≈ 0.025368
        let expected = 1.0 - 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        check_angular::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        // e2m3 max is 7.5, use values in range
        let expected_e2m3 = 1.0 - 6.0 / (14.0_f64.sqrt() * 3.0_f64.sqrt());
        check_angular::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
    }

    // endregion

    // region: Euclidean Distances

    fn check_sqeuclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::SqEuclideanOutput: FloatLike,
    {
        check_binary::<T, T::SqEuclideanOutput, _>(
            a_vals,
            b_vals,
            T::sqeuclidean,
            expected,
            "sqeuclidean",
        );
    }

    fn check_euclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::EuclideanOutput: FloatLike,
    {
        check_binary::<T, T::EuclideanOutput, _>(
            a_vals,
            b_vals,
            T::euclidean,
            expected,
            "euclidean",
        );
    }

    #[test]
    fn sqeuclidean() {
        check_sqeuclidean::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e2m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
    }

    #[test]
    fn euclidean() {
        let expected = 27.0_f64.sqrt();
        check_euclidean::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e2m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        let expected_packed = 54.0_f64.sqrt(); // i4x2 duplicates each value into both nibbles
        check_euclidean::<i4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
        check_euclidean::<u4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
    }

    // endregion

    // region: Probability Divergences

    fn check_kld<T>(a: &[f32], b: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + KullbackLeibler,
        T::Output: FloatLike,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = T::kullbackleibler(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        assert_close(
            result,
            expected,
            T::atol().max(1e-6),
            T::rtol().max(1e-6),
            &format!("kld<{}>", core::any::type_name::<T>()),
        );
    }

    fn check_jsd<T>(a: &[f32], b: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + JensenShannon,
        T::Output: FloatLike,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = T::jensenshannon(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        assert_close(
            result,
            expected,
            T::atol().max(1e-6),
            T::rtol().max(1e-6),
            &format!("jsd<{}>", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn divergences() {
        let a = &[0.1_f32, 0.9, 0.0];
        let b = &[0.2_f32, 0.8, 0.0];

        // KL(a||b) = 0.1*ln(0.1/0.2) + 0.9*ln(0.9/0.8)
        let kld_expected = 0.1_f64 * (0.1_f64 / 0.2).ln() + 0.9_f64 * (0.9_f64 / 0.8).ln();
        check_kld::<f64>(a, b, kld_expected);
        check_kld::<f32>(a, b, kld_expected);
        check_kld::<f16>(a, b, kld_expected);
        check_kld::<bf16>(a, b, kld_expected);

        // JS distance = sqrt(0.5 * (KL(a||m) + KL(b||m))) where m = (a+b)/2
        let kl_am = 0.1_f64 * (0.1_f64 / 0.15).ln() + 0.9 * (0.9_f64 / 0.85).ln();
        let kl_bm = 0.2_f64 * (0.2_f64 / 0.15).ln() + 0.8 * (0.8_f64 / 0.85).ln();
        let jsd_expected = (0.5 * (kl_am + kl_bm)).sqrt();
        check_jsd::<f64>(a, b, jsd_expected);
        check_jsd::<f32>(a, b, jsd_expected);
        check_jsd::<f16>(a, b, jsd_expected);
        check_jsd::<bf16>(a, b, jsd_expected);
    }

    // endregion

    // region: Complex Products

    trait ComplexOutput {
        fn real(&self) -> f64;
        fn imag(&self) -> f64;
    }
    impl ComplexOutput for (f32, f32) {
        fn real(&self) -> f64 {
            self.0 as f64
        }
        fn imag(&self) -> f64 {
            self.1 as f64
        }
    }
    impl ComplexOutput for (f64, f64) {
        fn real(&self) -> f64 {
            self.0
        }
        fn imag(&self) -> f64 {
            self.1
        }
    }

    /// Test a complex two-input operation with real + imaginary expected outputs.
    fn check_complex<T, R, F>(
        a: &[f32],
        b: &[f32],
        op: F,
        expected_re: f64,
        expected_im: f64,
        label: &str,
    ) where
        T: FloatLike + TestableType,
        R: ComplexOutput,
        F: FnOnce(&[T], &[T]) -> Option<R>,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = op(&a_t, &b_t).unwrap();
        let tol = T::atol() + T::rtol() * expected_re.abs().max(expected_im.abs());
        assert_close(
            result.real(),
            expected_re,
            tol,
            0.0,
            &format!("{}<{}> real", label, core::any::type_name::<T>()),
        );
        assert_close(
            result.imag(),
            expected_im,
            tol,
            0.0,
            &format!("{}<{}> imag", label, core::any::type_name::<T>()),
        );
    }

    fn check_complex_dot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: FloatLike + TestableType + ComplexDot,
        T::Output: ComplexOutput,
    {
        check_complex::<T, T::Output, _>(
            a,
            b,
            <T as ComplexDot>::dot,
            expected_re,
            expected_im,
            "complex_dot",
        );
    }

    fn check_complex_vdot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: FloatLike + TestableType + ComplexVDot,
        T::Output: ComplexOutput,
    {
        check_complex::<T, T::Output, _>(a, b, T::vdot, expected_re, expected_im, "complex_vdot");
    }

    fn check_complex_bilinear_identity<T>(n: usize)
    where
        T: FloatLike + TestableType + ComplexBilinear,
        T::Output: ComplexOutput,
    {
        // a = [1+0i, 0...], b = [1+0i, 0...], C = identity
        let mut a = vec![T::zero(); n * 2];
        let mut b = vec![T::zero(); n * 2];
        a[0] = T::one();
        b[0] = T::one();
        let mut c = vec![T::zero(); n * n * 2];
        for i in 0..n {
            c[(i * n + i) * 2] = T::one();
        }
        let result = T::complex_bilinear(&a, &b, &c).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            result.real(),
            1.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> real", core::any::type_name::<T>()),
        );
        assert_close(
            result.imag(),
            0.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> imag", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn complex_products() {
        // [1+2i, 3+4i] · [5+6i, 7+8i]
        let a = &[1.0_f32, 2.0, 3.0, 4.0];
        let b = &[5.0_f32, 6.0, 7.0, 8.0];

        // dot: (-18, 68)
        check_complex_dot::<f64>(a, b, -18.0, 68.0);
        check_complex_dot::<f32>(a, b, -18.0, 68.0);
        check_complex_dot::<f16>(a, b, -18.0, 68.0);
        check_complex_dot::<bf16>(a, b, -18.0, 68.0);

        // vdot (conjugate): (70, -8)
        check_complex_vdot::<f64>(a, b, 70.0, -8.0);
        check_complex_vdot::<f32>(a, b, 70.0, -8.0);
        check_complex_vdot::<f16>(a, b, 70.0, -8.0);
        check_complex_vdot::<bf16>(a, b, 70.0, -8.0);

        // bilinear: identity matrix, unit vector → (1, 0)
        check_complex_bilinear_identity::<f64>(4);
        check_complex_bilinear_identity::<f32>(4);
        check_complex_bilinear_identity::<f16>(4);
        check_complex_bilinear_identity::<bf16>(4);
    }

    // endregion

    // region: Intersection Tests

    fn reference_intersect<T: Ord>(a: &[T], b: &[T]) -> usize {
        let mut a_iter = a.iter();
        let mut b_iter = b.iter();
        let mut a_current = a_iter.next();
        let mut b_current = b_iter.next();
        let mut count = 0;
        while let (Some(a_val), Some(b_val)) = (a_current, b_current) {
            match a_val.cmp(b_val) {
                core::cmp::Ordering::Less => a_current = a_iter.next(),
                core::cmp::Ordering::Greater => b_current = b_iter.next(),
                core::cmp::Ordering::Equal => {
                    count += 1;
                    a_current = a_iter.next();
                    b_current = b_iter.next();
                }
            }
        }
        count
    }

    fn generate_intersection_test_arrays<T>() -> Vec<Vec<T>>
    where
        T: core::convert::TryFrom<u32> + Copy,
        <T as core::convert::TryFrom<u32>>::Error: core::fmt::Debug,
    {
        vec![
            vec![],
            vec![T::try_from(42).unwrap()],
            vec![
                T::try_from(1).unwrap(),
                T::try_from(5).unwrap(),
                T::try_from(10).unwrap(),
            ],
            vec![
                T::try_from(2).unwrap(),
                T::try_from(4).unwrap(),
                T::try_from(6).unwrap(),
                T::try_from(8).unwrap(),
                T::try_from(10).unwrap(),
                T::try_from(12).unwrap(),
                T::try_from(14).unwrap(),
            ],
            (0..14).map(|x| T::try_from(x * 10).unwrap()).collect(),
            (5..20).map(|x| T::try_from(x * 10).unwrap()).collect(),
            (0..40).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (10..50).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (0..45).map(|x| T::try_from(x * 3).unwrap()).collect(),
            (0..100).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (50..150).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (0..100).map(|x| T::try_from(x * 5).unwrap()).collect(),
            (0..150)
                .filter(|x| x % 7 == 0)
                .map(|x| T::try_from(x).unwrap())
                .collect(),
            (0..500).map(|x| T::try_from(x * 3).unwrap()).collect(),
            (100..600).map(|x| T::try_from(x * 3).unwrap()).collect(),
            (0..600).map(|x| T::try_from(x * 7).unwrap()).collect(),
            (0..50).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (1000..1050).map(|x| T::try_from(x * 2).unwrap()).collect(),
            (0..16).map(|x| T::try_from(x).unwrap()).collect(),
            (0..32).map(|x| T::try_from(x).unwrap()).collect(),
            (0..64).map(|x| T::try_from(x).unwrap()).collect(),
        ]
    }

    #[test]
    fn intersect_u32_comprehensive() {
        let test_arrays: Vec<Vec<u32>> = generate_intersection_test_arrays();
        for (i, array_a) in test_arrays.iter().enumerate() {
            for (j, array_b) in test_arrays.iter().enumerate() {
                let expected = reference_intersect(array_a, array_b);
                let result = u32::sparse_intersection_size(array_a.as_slice(), array_b.as_slice());
                assert_eq!(
                    expected,
                    result,
                    "Intersection mismatch for arrays[{}] (len={}) and arrays[{}] (len={})",
                    i,
                    array_a.len(),
                    j,
                    array_b.len()
                );
            }
        }
    }

    #[test]
    fn intersect_u16_comprehensive() {
        let test_arrays: Vec<Vec<u16>> = generate_intersection_test_arrays();
        for (i, array_a) in test_arrays.iter().enumerate() {
            for (j, array_b) in test_arrays.iter().enumerate() {
                let expected = reference_intersect(array_a, array_b);
                let result = u16::sparse_intersection_size(array_a.as_slice(), array_b.as_slice());
                assert_eq!(
                    expected,
                    result,
                    "Intersection mismatch for arrays[{}] (len={}) and arrays[{}] (len={})",
                    i,
                    array_a.len(),
                    j,
                    array_b.len()
                );
            }
        }
    }

    #[test]
    fn intersect_edge_cases() {
        let empty: &[u32] = &[];
        let non_empty: &[u32] = &[1, 2, 3];
        assert_eq!(u32::sparse_intersection_size(empty, empty), 0);
        assert_eq!(u32::sparse_intersection_size(empty, non_empty), 0);
        assert_eq!(u32::sparse_intersection_size(non_empty, empty), 0);

        assert_eq!(u32::sparse_intersection_size(&[42u32], &[42u32]), 1);
        assert_eq!(u32::sparse_intersection_size(&[42u32], &[43u32]), 0);

        let a: &[u32] = &[1, 2, 3, 4, 5];
        let b: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::sparse_intersection_size(a, b), 0);

        let c: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::sparse_intersection_size(c, c), 5);

        let boundary_16: Vec<u32> = (0..16).collect();
        let boundary_32: Vec<u32> = (0..32).collect();
        let boundary_64: Vec<u32> = (0..64).collect();
        assert_eq!(
            u32::sparse_intersection_size(&boundary_16, &boundary_16),
            16
        );
        assert_eq!(
            u32::sparse_intersection_size(&boundary_32, &boundary_32),
            32
        );
        assert_eq!(
            u32::sparse_intersection_size(&boundary_64, &boundary_64),
            64
        );

        let first_half: Vec<u32> = (0..32).collect();
        let second_half: Vec<u32> = (16..48).collect();
        assert_eq!(u32::sparse_intersection_size(&first_half, &second_half), 16);
    }

    // endregion

    // region: Cast Operations

    fn check_cast_roundtrip<T: FloatLike + TestableType + CastDtype>(values: &[f32]) {
        let src: Vec<T> = values.iter().map(|&v| T::from_f32(v)).collect();
        let mut dst = vec![0.0f32; src.len()];
        cast(&src, &mut dst).unwrap();
        for (i, (&expected, &actual)) in values.iter().zip(dst.iter()).enumerate() {
            assert_close(
                actual as f64,
                expected as f64,
                T::atol(),
                T::rtol(),
                &format!("cast_roundtrip<{}>[{i}]", core::any::type_name::<T>()),
            );
        }
    }

    #[test]
    fn cast_roundtrip() {
        check_cast_roundtrip::<f16>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<bf16>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e4m3>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e5m2>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e2m3>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e3m2>(&[1.0, 0.5, -1.0]);
    }

    #[test]
    fn cast_f32_to_f16() {
        let src = [1.0f32, -1.0];
        let mut dst = [f16(0); 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!([dst[0].0, dst[1].0], [0x3C00, 0xBC00]);
    }

    #[test]
    fn cast_length_mismatch() {
        let src = [f16(0x3C00)];
        let mut dst = [0.0f32; 2];
        assert!(cast(&src, &mut dst).is_none());
    }

    // endregion

    // region: Elementwise Operations

    fn check_each_scale<T>(values: &[f32], alpha: f32, beta: f32)
    where
        T: FloatLike + TestableType + EachScale,
        <T as EachScale>::Scalar: FloatLike,
    {
        let a: Vec<T> = values.iter().map(|&v| T::from_f32(v)).collect();
        let mut result = vec![T::zero(); a.len()];
        let alpha_s = <<T as EachScale>::Scalar>::from_f32(alpha);
        let beta_s = <<T as EachScale>::Scalar>::from_f32(beta);
        T::each_scale(&a, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values[i] as f64 + beta as f64;
            assert_close(
                r.to_f64(),
                expected,
                T::atol(),
                T::rtol(),
                &format!("each_scale<{}>[{i}]", core::any::type_name::<T>()),
            );
        }
    }

    fn check_each_sum<T>(values_a: &[f32], values_b: &[f32])
    where
        T: FloatLike + TestableType + EachSum,
    {
        check_each_binary::<T, _>(values_a, values_b, T::each_sum, |a, b| a + b, "each_sum");
    }

    fn check_each_blend<T>(values_a: &[f32], values_b: &[f32], alpha: f32, beta: f32)
    where
        T: FloatLike + TestableType + EachBlend,
        <T as EachBlend>::Scalar: FloatLike,
    {
        let a: Vec<T> = values_a.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = values_b.iter().map(|&v| T::from_f32(v)).collect();
        let mut result = vec![T::zero(); a.len()];
        let alpha_s = <<T as EachBlend>::Scalar>::from_f32(alpha);
        let beta_s = <<T as EachBlend>::Scalar>::from_f32(beta);
        T::each_blend(&a, &b, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values_a[i] as f64 + beta as f64 * values_b[i] as f64;
            assert_close(
                r.to_f64(),
                expected,
                T::atol(),
                T::rtol(),
                &format!("each_blend<{}>[{i}]", core::any::type_name::<T>()),
            );
        }
    }

    fn check_each_fma<T>(
        values_a: &[f32],
        values_b: &[f32],
        values_c: &[f32],
        alpha: f32,
        beta: f32,
    ) where
        T: FloatLike + TestableType + EachFMA,
        <T as EachFMA>::Scalar: FloatLike,
    {
        let a: Vec<T> = values_a.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = values_b.iter().map(|&v| T::from_f32(v)).collect();
        let c: Vec<T> = values_c.iter().map(|&v| T::from_f32(v)).collect();
        let mut result = vec![T::zero(); a.len()];
        let alpha_s = <<T as EachFMA>::Scalar>::from_f32(alpha);
        let beta_s = <<T as EachFMA>::Scalar>::from_f32(beta);
        T::each_fma(&a, &b, &c, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values_a[i] as f64 * values_b[i] as f64
                + beta as f64 * values_c[i] as f64;
            assert_close(
                r.to_f64(),
                expected,
                T::atol(),
                T::rtol(),
                &format!("each_fma<{}>[{i}]", core::any::type_name::<T>()),
            );
        }
    }

    #[test]
    fn each_scale() {
        check_each_scale::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<f16>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<bf16>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<e2m3>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<e4m3>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<e5m2>(&[1.0, 2.0], 2.0, 0.0);
        check_each_scale::<e3m2>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i8>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<u8>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<u32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<i16>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<u16>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<u64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
    }

    #[test]
    fn each_sum() {
        check_each_sum::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<e4m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<e5m2>(&[1.0, 2.0], &[1.0, 1.0]);
        check_each_sum::<e3m2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn each_sum_length_mismatch() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0];
        let mut result = vec![0.0f32; a.len()];
        assert!(f32::each_sum(&a, &b, &mut result).is_none());
    }

    #[test]
    fn each_blend() {
        check_each_blend::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e4m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e5m2>(&[1.0, 2.0], &[1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e3m2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
    }

    #[test]
    fn each_fma() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[2.0, 3.0, 4.0];
        let c = &[1.0, 1.0, 1.0];
        check_each_fma::<f32>(a, b, c, 1.0, 1.0);
        check_each_fma::<f64>(a, b, c, 1.0, 1.0);
        check_each_fma::<f16>(a, b, c, 1.0, 1.0);
        check_each_fma::<bf16>(a, b, c, 1.0, 1.0);
        // e2m3 max is 7.5, so use small inputs that stay in range: 1*1+1=2
        check_each_fma::<e2m3>(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], c, 1.0, 1.0);
        check_each_fma::<e4m3>(a, b, c, 1.0, 1.0);
        let a2 = &[1.0, 2.0];
        let b2 = &[2.0, 3.0];
        let c2 = &[1.0, 1.0];
        check_each_fma::<e5m2>(a2, b2, c2, 1.0, 1.0);
        check_each_fma::<e3m2>(a, b, c, 1.0, 1.0);
        check_each_fma::<i8>(a, b, c, 1.0, 1.0);
        check_each_fma::<u8>(a, b, c, 1.0, 1.0);
        check_each_fma::<i32>(a, b, c, 1.0, 1.0);
        check_each_fma::<u32>(a, b, c, 1.0, 1.0);
        check_each_fma::<i16>(a, b, c, 1.0, 1.0);
        check_each_fma::<u16>(a, b, c, 1.0, 1.0);
        check_each_fma::<i64>(a, b, c, 1.0, 1.0);
        check_each_fma::<u64>(a, b, c, 1.0, 1.0);
    }

    #[test]
    fn each_large() {
        let values: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        check_each_scale::<f32>(&values, 2.0, 0.5);

        let b: Vec<f32> = (0..1536).map(|i| (i as f32) * 2.0).collect();
        check_each_sum::<f32>(&values, &b);
    }

    // endregion

    // region: Trigonometry

    fn check_each_sin<T>(count: usize)
    where
        T: FloatLike + TestableType + EachSin,
    {
        use core::f64::consts::PI;
        check_each_unary::<T, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            T::sin,
            f64::sin,
            "sin",
        );
    }

    fn check_each_cos<T>(count: usize)
    where
        T: FloatLike + TestableType + EachCos,
    {
        use core::f64::consts::PI;
        check_each_unary::<T, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            T::cos,
            f64::cos,
            "cos",
        );
    }

    fn check_each_atan<T>(count: usize)
    where
        T: FloatLike + TestableType + EachATan,
    {
        check_each_unary::<T, _>(
            count,
            |i, n| -5.0 + 10.0 * (i as f64) / (n as f64),
            T::atan,
            f64::atan,
            "atan",
        );
    }

    #[test]
    fn each_sin() {
        check_each_sin::<f32>(97);
        check_each_sin::<f64>(97);
        check_each_sin::<f16>(97);
    }

    #[test]
    fn each_cos() {
        check_each_cos::<f32>(97);
        check_each_cos::<f64>(97);
        check_each_cos::<f16>(97);
    }

    #[test]
    fn each_atan() {
        check_each_atan::<f32>(100);
        check_each_atan::<f64>(100);
        check_each_atan::<f16>(100);
    }

    // endregion

    // region: Mesh Alignment

    fn check_kabsch_identical<T>(cloud: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Output: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let result = T::kabsch(&cloud_t, &cloud_t).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            FloatLike::to_f64(result.scale),
            1.0,
            tol,
            0.0,
            &format!("kabsch<{}> scale", core::any::type_name::<T>()),
        );
        assert_close(
            FloatLike::to_f64(result.rmsd),
            0.0,
            tol,
            0.0,
            &format!("kabsch<{}> rmsd", core::any::type_name::<T>()),
        );
    }

    fn check_umeyama_scaled<T>(cloud: &[[f32; 3]], scaled: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Output: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let scaled_t = convert_cloud::<T>(scaled);
        let result = T::umeyama(&cloud_t, &scaled_t).unwrap();
        let scale = FloatLike::to_f64(result.scale);
        assert!(
            scale > 1.0 && scale < 3.0,
            "umeyama<{}> scale: expected ~2.0, got {}",
            core::any::type_name::<T>(),
            scale
        );
    }

    fn check_rmsd_identical<T>(cloud: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Output: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let result = T::rmsd(&cloud_t, &cloud_t).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            FloatLike::to_f64(result.scale),
            1.0,
            tol,
            0.0,
            &format!("rmsd<{}> scale", core::any::type_name::<T>()),
        );
        assert_close(
            FloatLike::to_f64(result.rmsd),
            0.0,
            tol,
            0.0,
            &format!("rmsd<{}> rmsd", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn mesh_alignment() {
        let cloud: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let scaled: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ];
        let tri: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // Kabsch — identical clouds
        check_kabsch_identical::<f64>(cloud);
        check_kabsch_identical::<f32>(cloud);

        // Umeyama — 2x scaled
        check_umeyama_scaled::<f64>(cloud, scaled);
        check_umeyama_scaled::<f32>(cloud, scaled);

        // RMSD — identical
        check_rmsd_identical::<f64>(tri);
        check_rmsd_identical::<f32>(tri);
    }

    #[test]
    fn mesh_alignment_edge_cases() {
        let tri: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pair: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        // Length mismatch
        assert!(f64::kabsch(tri, pair).is_none());
        assert!(f64::rmsd(tri, pair).is_none());
        assert!(f64::umeyama(tri, pair).is_none());

        // Too few points
        assert!(f64::kabsch(pair, pair).is_none());

        // Transform point — identical clouds → point stays approximately the same
        let result = f64::kabsch(tri, tri).unwrap();
        let transformed = result.transform_point([1.0, 0.0, 0.0]);
        assert!(
            (transformed[0] - 1.0).abs() < 0.01,
            "Expected x ~1.0, got {}",
            transformed[0]
        );
        assert!(
            transformed[1].abs() < 0.01,
            "Expected y ~0.0, got {}",
            transformed[1]
        );
        assert!(
            transformed[2].abs() < 0.01,
            "Expected z ~0.0, got {}",
            transformed[2]
        );

        // Rotation determinant
        let cloud5: &[[f64; 3]] = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let result = f64::kabsch(cloud5, cloud5).unwrap();
        let r = &result.rotation_matrix;
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!(
            (det.abs() - 1.0).abs() < 0.001,
            "Expected det(R) ~±1.0, got {}",
            det
        );
    }

    // endregion

    // region: Bilinear

    fn check_bilinear<T>(first_values: &[f32], second_values: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Bilinear,
        T::Output: FloatLike,
    {
        let first: Vec<T> = first_values.iter().map(|&v| T::from_f32(v)).collect();
        let second: Vec<T> = second_values.iter().map(|&v| T::from_f32(v)).collect();
        let identity = make_identity::<T>(first.len());
        let result = T::bilinear(&first, &second, &identity).unwrap();
        assert_close(
            result.to_f64(),
            expected,
            T::atol(),
            T::rtol(),
            &format!("bilinear<{}>", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn bilinear() {
        // first=[1,2,3], second=[4,5,6], identity matrix → dot = 32
        let first_values = &[1.0, 2.0, 3.0];
        let second_values = &[4.0, 5.0, 6.0];
        check_bilinear::<f64>(first_values, second_values, 32.0);
        check_bilinear::<f32>(first_values, second_values, 32.0);
        check_bilinear::<f16>(first_values, second_values, 32.0);
        check_bilinear::<bf16>(first_values, second_values, 32.0);
    }

    // endregion

    // region: Mahalanobis Distance

    fn check_mahalanobis<T>(first_values: &[f32], second_values: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Mahalanobis,
        T::Output: FloatLike,
    {
        let first: Vec<T> = first_values.iter().map(|&v| T::from_f32(v)).collect();
        let second: Vec<T> = second_values.iter().map(|&v| T::from_f32(v)).collect();
        let identity = make_identity::<T>(first.len());
        let result = T::mahalanobis(&first, &second, &identity).unwrap();
        assert_close(
            result.to_f64(),
            expected,
            T::atol(),
            T::rtol(),
            &format!("mahalanobis<{}>", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn mahalanobis() {
        // first=[1,2,3], second=[4,5,6], identity → sqrt(27)
        let first_values = &[1.0, 2.0, 3.0];
        let second_values = &[4.0, 5.0, 6.0];
        let expected = (27.0_f64).sqrt();
        check_mahalanobis::<f64>(first_values, second_values, expected);
        check_mahalanobis::<f32>(first_values, second_values, expected);
        check_mahalanobis::<f16>(first_values, second_values, expected);
        check_mahalanobis::<bf16>(first_values, second_values, expected);
    }

    // endregion

    // region: Geospatial

    fn check_haversine<T>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        T: FloatLike + TestableType + Haversine,
    {
        let a_lat = [T::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [T::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [T::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [T::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [T::zero()];
        T::haversine(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("haversine<{}>", core::any::type_name::<T>()),
        );
    }

    fn check_vincenty<T>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        T: FloatLike + TestableType + Vincenty,
    {
        let a_lat = [T::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [T::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [T::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [T::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [T::zero()];
        T::vincenty(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("vincenty<{}>", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn geospatial() {
        // New York → Los Angeles
        // Haversine uses NK_EARTH_MEDIATORIAL_RADIUS (6,335,439m) → ~3,913,778m
        // Vincenty uses the WGS-84 ellipsoid → ~3,944,422m
        let hav_expected = 3_914_000.0;
        let vin_expected = 3_944_000.0;
        check_haversine::<f64>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            hav_expected,
            20_000.0,
        );
        check_haversine::<f32>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            hav_expected,
            50_000.0,
        );
        check_vincenty::<f64>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            vin_expected,
            20_000.0,
        );
        check_vincenty::<f32>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            vin_expected,
            50_000.0,
        );
    }

    // endregion

    // region: ReduceMoments

    fn check_reduce_moments<T>(input_values: &[f32])
    where
        T: FloatLike + TestableType + ReduceMoments,
        T::SumOutput: FloatLike,
        T::SumSqOutput: FloatLike,
    {
        let data: Vec<T> = input_values.iter().map(|&v| T::from_f32(v)).collect();
        let stride_bytes = core::mem::size_of::<T>();
        let (actual_sum, actual_sumsq) = T::reduce_moments(&data, stride_bytes);
        let expected_sum: f64 = input_values.iter().map(|&v| v as f64).sum();
        let expected_sumsq: f64 = input_values.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let n = input_values.len() as f64;
        assert_close(
            actual_sum.to_f64(),
            expected_sum,
            T::atol() * n,
            T::rtol(),
            &format!("reduce_moments<{}> sum", core::any::type_name::<T>()),
        );
        assert_close(
            actual_sumsq.to_f64(),
            expected_sumsq,
            T::atol() * n,
            T::rtol(),
            &format!("reduce_moments<{}> sumsq", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn reduce_moments() {
        // Float types — SumOutput/SumSqOutput are FloatLike (f32 or f64)
        let input_values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<f64>(input_values);
        check_reduce_moments::<f32>(input_values);
        check_reduce_moments::<f16>(input_values);
        check_reduce_moments::<bf16>(input_values);
        check_reduce_moments::<e4m3>(input_values);
        check_reduce_moments::<e5m2>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e2m3>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e3m2>(&[1.0, 2.0, 3.0]);

        // Integer types — now also go through generics with FloatLike for i64/u64
        let signed = &[1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let unsigned = &[1.0_f32, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<i8>(signed);
        check_reduce_moments::<u8>(unsigned);
        check_reduce_moments::<i16>(signed);
        check_reduce_moments::<u16>(unsigned);
        check_reduce_moments::<i32>(signed);
        check_reduce_moments::<u32>(unsigned);
        check_reduce_moments::<i64>(signed);
        check_reduce_moments::<u64>(unsigned);
    }

    // endregion

    // region: ReduceMinMax

    fn check_reduce_minmax<T>(input_values: &[f32])
    where
        T: FloatLike + TestableType + ReduceMinMax,
        T::Output: FloatLike,
    {
        let data: Vec<T> = input_values.iter().map(|&v| T::from_f32(v)).collect();
        let stride_bytes = core::mem::size_of::<T>();
        let (actual_min, actual_min_idx, actual_max, actual_max_idx) =
            T::reduce_minmax(&data, stride_bytes);
        let (exp_min_idx, exp_min) = input_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (exp_max_idx, exp_max) = input_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_close(
            actual_min.to_f64(),
            *exp_min as f64,
            T::atol(),
            0.0,
            &format!("reduce_minmax<{}> min", core::any::type_name::<T>()),
        );
        assert_eq!(
            actual_min_idx,
            exp_min_idx,
            "reduce_minmax<{}> min_index",
            core::any::type_name::<T>()
        );
        assert_close(
            actual_max.to_f64(),
            *exp_max as f64,
            T::atol(),
            0.0,
            &format!("reduce_minmax<{}> max", core::any::type_name::<T>()),
        );
        assert_eq!(
            actual_max_idx,
            exp_max_idx,
            "reduce_minmax<{}> max_index",
            core::any::type_name::<T>()
        );
    }

    #[test]
    fn reduce_minmax() {
        // All FloatLike types — Output is also FloatLike
        let input_values = &[3.0, 1.0, 4.0, 1.5, 5.0, 2.0];
        check_reduce_minmax::<f64>(input_values);
        check_reduce_minmax::<f32>(input_values);
        check_reduce_minmax::<f16>(input_values);
        check_reduce_minmax::<bf16>(input_values);
        check_reduce_minmax::<e4m3>(input_values);
        check_reduce_minmax::<e5m2>(input_values);
        check_reduce_minmax::<e2m3>(&[3.0, 1.0, 4.0, 1.5, 5.0, 2.0]);
        check_reduce_minmax::<e3m2>(input_values);
        check_reduce_minmax::<i8>(input_values);
        check_reduce_minmax::<u8>(input_values);
        check_reduce_minmax::<i32>(input_values);
        check_reduce_minmax::<u32>(input_values);

        // i16, u16, i64, u64 — now also go through generics with FloatLike
        check_reduce_minmax::<i16>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u16>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
        check_reduce_minmax::<i64>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u64>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
    }

    // endregion

    // region: SparseDot

    #[test]
    fn sparse_dot() {
        // u32 indices with f32 weights
        let first_indices: Vec<u32> = vec![1, 3, 5];
        let second_indices: Vec<u32> = vec![2, 3, 5, 7];
        let first_weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let second_weights: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        // Overlap at indices 3 and 5: 2.0*5.0 + 3.0*6.0 = 28.0
        let result = u32::sparse_dot(
            &first_indices,
            &second_indices,
            &first_weights,
            &second_weights,
        );
        assert!((result - 28.0).abs() < 0.01, "sparse_dot u32f32: {result}");

        // u16 indices with bf16 weights
        let first_indices_u16: Vec<u16> = vec![1, 3, 5];
        let second_indices_u16: Vec<u16> = vec![2, 3, 5, 7];
        let first_weights_bf16: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect();
        let second_weights_bf16: Vec<bf16> = vec![4.0, 5.0, 6.0, 7.0]
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect();
        let result = u16::sparse_dot(
            &first_indices_u16,
            &second_indices_u16,
            &first_weights_bf16,
            &second_weights_bf16,
        );
        assert!((result - 28.0).abs() < 1.0, "sparse_dot u16bf16: {result}");

        // Disjoint sets → 0
        let result = u32::sparse_dot(&[1, 2], &[3, 4], &[1.0, 1.0], &[1.0, 1.0]);
        assert!(result.abs() < 0.01, "sparse_dot disjoint: {result}");
    }

    // endregion
}
