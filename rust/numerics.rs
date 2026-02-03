//! Numeric traits and implementations for vector operations.
//!
//! This module provides hardware-accelerated implementations of:
//!
//! - **Spatial similarity**: [`Dot`], [`Angular`], [`Euclidean`]
//! - **Binary similarity**: [`Hamming`], [`Jaccard`]
//! - **Probability divergence**: [`KullbackLeibler`], [`JensenShannon`]
//! - **Complex products**: [`ComplexDot`], [`ComplexVDot`]
//! - **Elementwise operations**: [`EachScale`], [`EachSum`], [`EachBlend`], [`EachFMA`]
//! - **Trigonometry**: [`Sin`], [`Cos`], [`ATan`]
//! - **Reductions**: [`ReduceAdd`], [`ReduceMin`], [`ReduceMax`]
//! - **Geospatial**: [`Haversine`], [`Vincenty`]
//! - **Mesh alignment**: [`MeshAlignment`]
//! - **Sparse sets**: [`SparseIntersect`], [`SparseDot`]

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
    fn nk_sin_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_sin_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_sin_f16(inputs: *const u16, n: u64size, outputs: *mut u16);
    fn nk_cos_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_cos_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_cos_f16(inputs: *const u16, n: u64size, outputs: *mut u16);
    fn nk_atan_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_atan_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_atan_f16(inputs: *const u16, n: u64size, outputs: *mut u16);

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

    // Reductions
    fn nk_reduce_add_f64(data: *const f64, count: u64size, stride_bytes: u64size, result: *mut f64);
    fn nk_reduce_add_f32(data: *const f32, count: u64size, stride_bytes: u64size, result: *mut f64);
    fn nk_reduce_add_i8(data: *const i8, count: u64size, stride_bytes: u64size, result: *mut i64);
    fn nk_reduce_add_u8(data: *const u8, count: u64size, stride_bytes: u64size, result: *mut u64);
    fn nk_reduce_add_i16(data: *const i16, count: u64size, stride_bytes: u64size, result: *mut i64);
    fn nk_reduce_add_u16(data: *const u16, count: u64size, stride_bytes: u64size, result: *mut u64);
    fn nk_reduce_add_i32(data: *const i32, count: u64size, stride_bytes: u64size, result: *mut i64);
    fn nk_reduce_add_u32(data: *const u32, count: u64size, stride_bytes: u64size, result: *mut u64);
    fn nk_reduce_add_i64(data: *const i64, count: u64size, stride_bytes: u64size, result: *mut i64);
    fn nk_reduce_add_u64(data: *const u64, count: u64size, stride_bytes: u64size, result: *mut u64);

    fn nk_reduce_min_f64(
        data: *const f64,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f64,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_f32(
        data: *const f32,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_i8(
        data: *const i8,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut i8,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_u8(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut u8,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_i16(
        data: *const i16,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut i16,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_u16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut u16,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_i32(
        data: *const i32,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut i32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_u32(
        data: *const u32,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut u32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_i64(
        data: *const i64,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut i64,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_u64(
        data: *const u64,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut u64,
        min_index: *mut u64size,
    );

    fn nk_reduce_max_f64(
        data: *const f64,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f64,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_f32(
        data: *const f32,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_i8(
        data: *const i8,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut i8,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_u8(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut u8,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_i16(
        data: *const i16,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut i16,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_u16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut u16,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_i32(
        data: *const i32,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut i32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_u32(
        data: *const u32,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut u32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_i64(
        data: *const i64,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut i64,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_u64(
        data: *const u64,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut u64,
        max_index: *mut u64size,
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

    // FP8 EachBlend
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

    // FP8 EachFMA
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

    // Integer EachFMA (i16/u16 use f32 coefficients, i32/u32/i64/u64 use f64)
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

    // Half-precision ReduceAdd (output to f32)
    fn nk_reduce_add_f16(data: *const u16, count: u64size, stride_bytes: u64size, result: *mut f32);
    fn nk_reduce_add_bf16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        result: *mut f32,
    );
    fn nk_reduce_add_e4m3(data: *const u8, count: u64size, stride_bytes: u64size, result: *mut f32);
    fn nk_reduce_add_e5m2(data: *const u8, count: u64size, stride_bytes: u64size, result: *mut f32);

    // Half-precision ReduceMin (output to f32)
    fn nk_reduce_min_f16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_bf16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_e4m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f32,
        min_index: *mut u64size,
    );
    fn nk_reduce_min_e5m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        min_value: *mut f32,
        min_index: *mut u64size,
    );

    // Half-precision ReduceMax (output to f32)
    fn nk_reduce_max_f16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_bf16(
        data: *const u16,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_e4m3(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f32,
        max_index: *mut u64size,
    );
    fn nk_reduce_max_e5m2(
        data: *const u8,
        count: u64size,
        stride_bytes: u64size,
        max_value: *mut f32,
        max_index: *mut u64size,
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

/// Computes the **Jaccard distance** between two binary vectors.
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

// region: Sin

/// Computes **element-wise sine** of a vector.
pub trait Sin: Sized {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl Sin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_sin_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_sin_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sin for f16 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_sin_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: Sin

// region: Cos

/// Computes **element-wise cosine** of a vector.
pub trait Cos: Sized {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl Cos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_cos_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Cos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_cos_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Cos for f16 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_cos_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: Cos

// region: ATan

/// Computes **element-wise arctangent** (inverse tangent) of a vector.
pub trait ATan: Sized {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl ATan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_atan_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ATan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_atan_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ATan for f16 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_atan_f16(
                inputs.as_ptr() as *const u16,
                inputs.len() as u64size,
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: ATan

// region: Scale

/// Computes **element-wise affine transform** (scale and shift).
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

// endregion: Scale

// region: Sum

/// Computes **element-wise addition** of two vectors.
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

// endregion: Sum

// region: WSum

/// Computes **element-wise weighted sum** of two vectors.
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

// endregion: WSum

// region: FMA

/// Computes **fused multiply-add** across three vectors.
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

/// Horizontal sum reduction with stride support.
///
/// Computes the sum of all elements in a slice, with optional striding.
/// The `Output` type may be wider than the input to avoid overflow.
pub trait ReduceAdd: Sized {
    type Output;
    /// Sum all elements in `data` with the given stride (in bytes).
    /// Use `stride_bytes = size_of::<Self>()` for contiguous data.
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output;
}

impl ReduceAdd for f64 {
    type Output = f64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f64 = 0.0;
        unsafe {
            nk_reduce_add_f64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for f32 {
    type Output = f64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f64 = 0.0;
        unsafe {
            nk_reduce_add_f32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for i8 {
    type Output = i64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: i64 = 0;
        unsafe {
            nk_reduce_add_i8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for u8 {
    type Output = u64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: u64 = 0;
        unsafe {
            nk_reduce_add_u8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for i16 {
    type Output = i64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: i64 = 0;
        unsafe {
            nk_reduce_add_i16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for u16 {
    type Output = u64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: u64 = 0;
        unsafe {
            nk_reduce_add_u16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for i32 {
    type Output = i64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: i64 = 0;
        unsafe {
            nk_reduce_add_i32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for u32 {
    type Output = u64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: u64 = 0;
        unsafe {
            nk_reduce_add_u32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for i64 {
    type Output = i64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: i64 = 0;
        unsafe {
            nk_reduce_add_i64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for u64 {
    type Output = u64;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: u64 = 0;
        unsafe {
            nk_reduce_add_u64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for f16 {
    type Output = f32;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f32 = 0.0;
        unsafe {
            nk_reduce_add_f16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for bf16 {
    type Output = f32;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f32 = 0.0;
        unsafe {
            nk_reduce_add_bf16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for e4m3 {
    type Output = f32;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f32 = 0.0;
        unsafe {
            nk_reduce_add_e4m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

impl ReduceAdd for e5m2 {
    type Output = f32;
    fn reduce_add(data: &[Self], stride_bytes: usize) -> Self::Output {
        let mut result: f32 = 0.0;
        unsafe {
            nk_reduce_add_e5m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut result,
            );
        }
        result
    }
}

/// Find minimum value and its index with stride support.
pub trait ReduceMin: Sized {
    /// Output type for the minimum value. Usually Self, but f32 for half-precision types.
    type Output;
    /// Returns (min_value, min_index) for the given data with the specified stride.
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize);
}

impl ReduceMin for f64 {
    type Output = f64;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f64 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_f64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for f32 {
    type Output = f32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f32 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_f32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for i8 {
    type Output = i8;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: i8 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_i8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for u8 {
    type Output = u8;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: u8 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_u8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for i16 {
    type Output = i16;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: i16 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_i16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for u16 {
    type Output = u16;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: u16 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_u16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for i32 {
    type Output = i32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: i32 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_i32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for u32 {
    type Output = u32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: u32 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_u32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for i64 {
    type Output = i64;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: i64 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_i64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for u64 {
    type Output = u64;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: u64 = 0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_u64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for f16 {
    type Output = f32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f32 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_f16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for bf16 {
    type Output = f32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f32 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_bf16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for e4m3 {
    type Output = f32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f32 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_e4m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

impl ReduceMin for e5m2 {
    type Output = f32;
    fn reduce_min(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut min_value: f32 = 0.0;
        let mut min_index: u64size = 0;
        unsafe {
            nk_reduce_min_e5m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut min_value,
                &mut min_index,
            );
        }
        (min_value, min_index as usize)
    }
}

/// Find maximum value and its index with stride support.
pub trait ReduceMax: Sized {
    /// Output type for the maximum value. Usually Self, but f32 for half-precision types.
    type Output;
    /// Returns (max_value, max_index) for the given data with the specified stride.
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize);
}

impl ReduceMax for f64 {
    type Output = f64;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f64 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_f64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for f32 {
    type Output = f32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f32 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_f32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for i8 {
    type Output = i8;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: i8 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_i8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for u8 {
    type Output = u8;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: u8 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_u8(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for i16 {
    type Output = i16;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: i16 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_i16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for u16 {
    type Output = u16;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: u16 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_u16(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for i32 {
    type Output = i32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: i32 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_i32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for u32 {
    type Output = u32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: u32 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_u32(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for i64 {
    type Output = i64;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: i64 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_i64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for u64 {
    type Output = u64;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: u64 = 0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_u64(
                data.as_ptr(),
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for f16 {
    type Output = f32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f32 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_f16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for bf16 {
    type Output = f32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f32 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_bf16(
                data.as_ptr() as *const u16,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for e4m3 {
    type Output = f32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f32 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_e4m3(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

impl ReduceMax for e5m2 {
    type Output = f32;
    fn reduce_max(data: &[Self], stride_bytes: usize) -> (Self::Output, usize) {
        let mut max_value: f32 = 0.0;
        let mut max_index: u64size = 0;
        unsafe {
            nk_reduce_max_e5m2(
                data.as_ptr() as *const u8,
                data.len() as u64size,
                stride_bytes as u64size,
                &mut max_value,
                &mut max_index,
            );
        }
        (max_value, max_index as usize)
    }
}

// endregion: Reductions

// region: MeshAlignment

/// Result of mesh alignment operations (RMSD, Kabsch, Umeyama).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAlignmentResult<T> {
    pub rotation_matrix: [T; 9],
    pub scale: T,
    pub rmsd: T,
    pub a_centroid: [T; 3],
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

/// `Elementwise` bundles element-wise operations: EachScale, EachSum, EachBlend, and EachFMA.
pub trait Elementwise: EachScale + EachSum + EachBlend + EachFMA {}
impl<T: EachScale + EachSum + EachBlend + EachFMA> Elementwise for T {}

/// `Trigonometry` bundles trigonometric functions: Sin, Cos, and ATan.
pub trait Trigonometry: Sin + Cos + ATan {}
impl<T: Sin + Cos + ATan> Trigonometry for T {}

/// `Reductions` bundles reduction operations: ReduceAdd, ReduceMin, and ReduceMax.
pub trait Reductions: ReduceAdd + ReduceMin + ReduceMax {}
impl<T: ReduceAdd + ReduceMin + ReduceMax> Reductions for T {}

// endregion: Convenience Trait Aliases

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = <f32 as Dot>::dot(&a, &b).unwrap();
        assert!((result - 32.0).abs() < 0.01);
    }

    #[test]
    fn dot_i8() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let result = i8::dot(&a, &b).unwrap();
        assert_eq!(result, 32);
    }

    #[test]
    fn euclidean_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = f32::sqeuclidean(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 0.01);
    }

    #[test]
    fn euclidean_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let result = f64::sqeuclidean(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 0.01);
    }

    #[test]
    fn euclidean_f16() {
        let a: Vec<f16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let b: Vec<f16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let result = f16::sqeuclidean(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 1.0);
    }

    #[test]
    fn hamming_u1x8() {
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b00001111), u1x8(0b01010101)];
        let result = u1x8::hamming(&a, &b).unwrap();
        assert_eq!(result, 16);
    }

    #[test]
    fn jaccard_u1x8() {
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let result = u1x8::jaccard(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn js_f32() {
        let a = vec![0.25f32, 0.25, 0.25, 0.25];
        let b = vec![0.25f32, 0.25, 0.25, 0.25];
        let result = f32::jensenshannon(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn kl_f32() {
        let a = vec![0.25f32, 0.25, 0.25, 0.25];
        let b = vec![0.25f32, 0.25, 0.25, 0.25];
        let result = f32::kullbackleibler(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn scale_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut result = vec![0.0f32; 3];
        f32::each_scale(&a, 2.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn scale_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let mut result = vec![0.0f64; 3];
        f64::each_scale(&a, 2.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn wsum_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut result = vec![0.0f32; 3];
        f32::each_blend(&a, &b, 0.5, 0.5, &mut result).unwrap();
        assert!((result[0] - 2.5).abs() < 0.01);
        assert!((result[1] - 3.5).abs() < 0.01);
        assert!((result[2] - 4.5).abs() < 0.01);
    }

    #[test]
    fn fma_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![2.0f32, 2.0, 2.0];
        let c = vec![1.0f32, 1.0, 1.0];
        let mut result = vec![0.0f32; 3];
        f32::each_fma(&a, &b, &c, 1.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn sin_f32_small() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..11).map(|i| (i as f32) * PI / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.sin()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Sin>::sin(&inputs, &mut result).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 0.1, "sin mismatch: {} vs {}", r, e);
        }
    }

    #[test]
    fn cos_f32_test() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..11).map(|i| (i as f32) * PI / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.cos()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Cos>::cos(&inputs, &mut result).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 0.1, "cos mismatch: {} vs {}", r, e);
        }
    }

    #[test]
    fn bf16_dot() {
        let brain_a: Vec<bf16> = vec![1.0, 2.0, 3.0, 1.0, 2.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let brain_b: Vec<bf16> = vec![4.0, 5.0, 6.0, 4.0, 5.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        if let Some(result) = <bf16 as Dot>::dot(&brain_a, &brain_b) {
            assert_eq!(46.0, result);
        }
    }

    #[test]
    fn mesh_alignment_length_mismatch() {
        let a: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(f64::kabsch(a, b).is_none());
    }

    #[test]
    fn mesh_alignment_too_few_points() {
        let a: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(f64::kabsch(a, b).is_none());
    }

    #[test]
    fn hamming_u8() {
        let a: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let b: Vec<u8> = vec![0, 1, 2, 3, 0, 0, 0, 0];
        let result = u8::hamming(&a, &b).unwrap();
        assert_eq!(result, 4);
    }

    #[test]
    fn jaccard_u16() {
        let a: Vec<u16> = vec![1, 2, 3, 4];
        let b: Vec<u16> = vec![1, 2, 3, 4];
        let result = u16::jaccard(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);

        let c: Vec<u16> = vec![5, 6, 7, 8];
        let result2 = u16::jaccard(&a, &c).unwrap();
        assert!((result2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn jaccard_u32() {
        let a: Vec<u32> = vec![1, 2, 3, 4];
        let b: Vec<u32> = vec![1, 2, 5, 6];
        let result = u32::jaccard(&a, &b).unwrap();
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn sparse_intersection_size_u16() {
        let a: Vec<u16> = vec![1, 3, 5, 7, 9];
        let b: Vec<u16> = vec![2, 3, 5, 8, 9];
        let count = u16::sparse_intersection_size(&a, &b);
        assert_eq!(count, 3);
    }

    #[test]
    fn sparse_intersect_into_u16() {
        let a: Vec<u16> = vec![1, 3, 5, 7, 9];
        let b: Vec<u16> = vec![2, 3, 5, 8, 9];
        let mut result: Vec<u16> = vec![0; 5];
        let count = u16::sparse_intersect_into(&a, &b, &mut result).unwrap();
        assert_eq!(count, 3);
        assert_eq!(&result[..count], &[3, 5, 9]);
    }

    #[test]
    fn sparse_intersect_into_u32() {
        let a: Vec<u32> = vec![10, 20, 30, 40];
        let b: Vec<u32> = vec![15, 20, 30, 45];
        let mut result: Vec<u32> = vec![0; 4];
        let count = u32::sparse_intersect_into(&a, &b, &mut result).unwrap();
        assert_eq!(count, 2);
        assert_eq!(&result[..count], &[20, 30]);
    }

    #[test]
    fn sparse_intersection_size_u64() {
        let a: Vec<u64> = vec![100, 200, 300];
        let b: Vec<u64> = vec![200, 300, 400];
        let count = u64::sparse_intersection_size(&a, &b);
        assert_eq!(count, 2);
    }

    #[test]
    fn sparse_intersect_into_buffer_too_small() {
        let a: Vec<u16> = vec![1, 2, 3, 4, 5];
        let b: Vec<u16> = vec![3, 4, 5, 6, 7];
        let mut result: Vec<u16> = vec![0; 2];
        assert!(u16::sparse_intersect_into(&a, &b, &mut result).is_none());
    }
}
