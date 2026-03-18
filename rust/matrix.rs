//! Batch matrix operations: GEMM, packed spatial distances.
//!
//! This module provides:
//!
//! - [`Dots`]: Batch dot-product (GEMM-like) between two matrices
//! - [`Angulars`]: Batch angular (cosine) distances
//! - [`Euclideans`]: Batch squared-L2 distances
//! - [`Hammings`]: Batch Hamming distances
//! - [`Jaccards`]: Batch Jaccard distances
//! - [`PackedMatrix`]: Pre-packed matrix for accelerated batch operations

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::tensor::{
    Allocator, Global, ShapeDescriptor, Tensor, TensorError, TensorRef, TensorView, SIMD_ALIGNMENT,
};
use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_packed_size_f32(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f32(
        b: *const f32,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_f64(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f64(
        b: *const f64,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_f16(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f16(
        b: *const u16,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_bf16(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_bf16(
        b: *const u16,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_i8(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_i8(b: *const i8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_u8(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_u8(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e4m3(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e4m3(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e5m2(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e5m2(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e2m3(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e2m3(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e3m2(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e3m2(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
    fn nk_dots_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_u4(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_u4(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_i4(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_i4(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut i32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    // Symmetric Gram matrix (C = A × Aᵀ)
    fn nk_dots_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut i32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut i32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    fn nk_dots_packed_size_u1(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_u1(
        q: *const u8,
        width: usize,
        depth: usize,
        q_stride: usize,
        q_packed: *mut u8,
    );
    fn nk_dots_packed_u1(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_dots_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_hammings_packed_u1(
        a: *const u8,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );
    fn nk_hammings_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    fn nk_jaccards_packed_u1(
        v: *const u8,
        q_packed: *const u8,
        result: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );
    fn nk_jaccards_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    // Batched angular distances
    fn nk_angulars_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    // Batched euclidean distances
    fn nk_euclideans_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

// region: Dots Trait

/// Low-level trait for batched **dot product** computation using pre-packed matrices.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = aᵢ · bⱼ
///
/// B is pre-packed into a backend-specific layout for optimal memory access.
/// All strides are in bytes.
pub trait Dots: Sized + Clone {
    /// Accumulator type for the multiplication.
    type Accumulator: Clone + Default;

    /// Returns the size in bytes needed for the packed B matrix buffer.
    fn dots_packed_size(width: usize, depth: usize) -> usize;

    /// Packs the B matrix into an optimized backend-specific layout.
    ///
    /// # Safety
    /// - `b` must point to valid memory for `width * depth` elements
    /// - `packed` must point to a buffer of at least `dots_packed_size(width, depth)` bytes
    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );

    /// Computes C = A × Bᵀ using packed B.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `dots_pack`
    /// - `c` must point to valid memory for `height * width` elements with given stride
    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes C = A × Aᵀ where C is symmetric.
    ///
    /// Given input matrix A of shape [n, k], computes the symmetric matrix of all pairwise
    /// dot products. Only the upper triangle is computed, then mirrored to the lower triangle.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors × depth` elements with given stride
    /// - `result` must point to valid memory for `n_vectors × n_vectors` elements with given stride
    /// - Strides are in bytes, not elements
    /// - `row_start + row_count` must be <= `n_vectors`
    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Dots for f32 {
    type Accumulator = f64;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f32(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_f32(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f32(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f32(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for f64 {
    type Accumulator = f64;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f64(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_f64(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f64(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f64(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for f16 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f16(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_f16(b as *const u16, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for bf16 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_bf16(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_bf16(b as *const u16, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_bf16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_bf16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for i8 {
    type Accumulator = i32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_i8(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_i8(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_i8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for u8 {
    type Accumulator = u32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u8(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_u8(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for e4m3 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e4m3(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_e4m3(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e4m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e4m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for e5m2 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e5m2(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_e5m2(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e5m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e5m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for e2m3 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e2m3(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_e2m3(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e2m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e2m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for e3m2 {
    type Accumulator = f32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e3m2(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_e3m2(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e3m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e3m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for u4x2 {
    type Accumulator = u32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u4(width, depth * 2) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_u4(b as *const u8, width, depth * 2, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for i4x2 {
    type Accumulator = i32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_i4(width, depth * 2) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_i4(b as *const u8, width, depth * 2, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_i4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Dots for u1x8 {
    type Accumulator = u32;

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u1(width, depth * 8) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
        nk_dots_pack_u1(b as *const u8, width, depth * 8, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u1(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 8,
            a_stride,
            c_stride,
        )
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u1(
            vectors as *const u8,
            n_vectors,
            depth * 8,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

// endregion: Dots Trait

// region: Hammings Trait

/// Low-level trait for batched **Hamming distance** operations.
///
/// Given A ∈ {0,1}ᵐˣᵏ and packed B ∈ {0,1}ⁿˣᵏ, computes C ∈ ℕᵐˣⁿ where:
/// Cᵢⱼ = popcount(aᵢ ⊕ bⱼ)
///
/// Packing is inherited from the `Dots` supertrait.
pub trait Hammings: Dots {
    /// Computes Hamming distances between values matrix rows and packed query rows.
    ///
    /// # Safety
    /// - `a` must point to valid memory for the values matrix
    /// - `q_packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `result` must point to valid memory for `height * width` u32 elements
    unsafe fn hammings_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );

    /// Computes symmetric Gram matrix of Hamming distances: C = A × Aᵀ.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for the input matrix
    /// - `result` must point to valid memory for `n_vectors * n_vectors` u32 elements
    unsafe fn hammings_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Hammings for u1x8 {
    unsafe fn hammings_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_hammings_packed_u1(
            a as *const u8,
            q_packed,
            result,
            height,
            width,
            depth * 8,
            v_stride,
            r_stride,
        )
    }

    unsafe fn hammings_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_hammings_symmetric_u1(
            vectors as *const u8,
            n_vectors,
            depth * 8,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

// endregion: Hammings Trait

// region: Jaccards Trait

/// Low-level trait for batched **Jaccard distance** operations.
///
/// Given A ∈ {0,1}ᵐˣᵏ and packed B ∈ {0,1}ⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = 1 − popcount(aᵢ ∧ bⱼ) / popcount(aᵢ ∨ bⱼ)
///
/// Packing is inherited from the `Dots` supertrait.
pub trait Jaccards: Dots {
    /// Result type for Jaccard distances.
    type JaccardResult: Clone + Default;

    /// Computes Jaccard distances between values matrix rows and packed query rows.
    ///
    /// # Safety
    /// - `a` must point to valid memory for the values matrix
    /// - `q_packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `result` must point to valid memory for `height * width` elements
    unsafe fn jaccards_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut Self::JaccardResult,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );

    /// Computes symmetric Gram matrix of Jaccard distances.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for the input matrix
    /// - `result` must point to valid memory for `n_vectors * n_vectors` elements
    unsafe fn jaccards_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::JaccardResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Jaccards for u1x8 {
    type JaccardResult = f32;

    unsafe fn jaccards_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut Self::JaccardResult,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_jaccards_packed_u1(
            a as *const u8,
            q_packed,
            result,
            height,
            width,
            depth * 8,
            v_stride,
            r_stride,
        )
    }

    unsafe fn jaccards_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::JaccardResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_jaccards_symmetric_u1(
            vectors as *const u8,
            n_vectors,
            depth * 8,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

// endregion: Jaccards Trait

// region: Angulars Trait

/// Low-level trait for batched **angular distance** operations.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = 1 − cos(θᵢⱼ) = 1 − (aᵢ · bⱼ) / (‖aᵢ‖ × ‖bⱼ‖)
///
/// Packing reuses `Dots::dots_pack` for optimal memory layout.
pub trait Angulars: Dots {
    /// Result type for angular distances.
    type SpatialResult: Clone + Default;

    /// Computes angular distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `height * width` result elements with given stride
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes symmetric angular distance matrix.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors * depth` elements
    /// - `result` must point to valid memory for `n_vectors * n_vectors` result elements
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

/// Low-level trait for batched **euclidean distance** operations.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = √(max(0, ‖aᵢ‖² + ‖bⱼ‖² − 2 · aᵢ · bⱼ))
///
/// Packing reuses `Dots::dots_pack` for optimal memory layout.
pub trait Euclideans: Dots {
    /// Result type for euclidean distances.
    type SpatialResult: Clone + Default;

    /// Computes euclidean distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `height * width` result elements with given stride
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes symmetric euclidean distance matrix.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors * depth` elements
    /// - `result` must point to valid memory for `n_vectors * n_vectors` result elements
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

macro_rules! impl_spatial_traits {
    ($rust_ty:ty, $result_ty:ty, $ptr_ty:ty, $cast:expr,
     $ang_packed:ident, $ang_sym:ident, $euc_packed:ident, $euc_sym:ident) => {
        impl Angulars for $rust_ty {
            type SpatialResult = $result_ty;

            unsafe fn angulars_packed(
                a: *const Self,
                packed: *const u8,
                c: *mut Self::SpatialResult,
                height: usize,
                width: usize,
                depth: usize,
                a_stride: usize,
                c_stride: usize,
            ) {
                $ang_packed(
                    $cast(a),
                    packed,
                    c,
                    height,
                    width,
                    depth,
                    a_stride,
                    c_stride,
                )
            }

            unsafe fn angulars_symmetric(
                vectors: *const Self,
                n_vectors: usize,
                depth: usize,
                stride: usize,
                result: *mut Self::SpatialResult,
                result_stride: usize,
                row_start: usize,
                row_count: usize,
            ) {
                $ang_sym(
                    $cast(vectors),
                    n_vectors,
                    depth,
                    stride,
                    result,
                    result_stride,
                    row_start,
                    row_count,
                )
            }
        }

        impl Euclideans for $rust_ty {
            type SpatialResult = $result_ty;

            unsafe fn euclideans_packed(
                a: *const Self,
                packed: *const u8,
                c: *mut Self::SpatialResult,
                height: usize,
                width: usize,
                depth: usize,
                a_stride: usize,
                c_stride: usize,
            ) {
                $euc_packed(
                    $cast(a),
                    packed,
                    c,
                    height,
                    width,
                    depth,
                    a_stride,
                    c_stride,
                )
            }

            unsafe fn euclideans_symmetric(
                vectors: *const Self,
                n_vectors: usize,
                depth: usize,
                stride: usize,
                result: *mut Self::SpatialResult,
                result_stride: usize,
                row_start: usize,
                row_count: usize,
            ) {
                $euc_sym(
                    $cast(vectors),
                    n_vectors,
                    depth,
                    stride,
                    result,
                    result_stride,
                    row_start,
                    row_count,
                )
            }
        }
    };
}

#[inline(always)]
fn identity_f32(p: *const f32) -> *const f32 { p }
#[inline(always)]
fn identity_f64(p: *const f64) -> *const f64 { p }
#[inline(always)]
fn identity_i8(p: *const i8) -> *const i8 { p }
#[inline(always)]
fn identity_u8(p: *const u8) -> *const u8 { p }
#[inline(always)]
fn cast_to_u16<T>(p: *const T) -> *const u16 { p as *const u16 }
#[inline(always)]
fn cast_to_u8<T>(p: *const T) -> *const u8 { p as *const u8 }

impl_spatial_traits!(
    f32,
    f64,
    *const f32,
    identity_f32,
    nk_angulars_packed_f32,
    nk_angulars_symmetric_f32,
    nk_euclideans_packed_f32,
    nk_euclideans_symmetric_f32
);
impl_spatial_traits!(
    f64,
    f64,
    *const f64,
    identity_f64,
    nk_angulars_packed_f64,
    nk_angulars_symmetric_f64,
    nk_euclideans_packed_f64,
    nk_euclideans_symmetric_f64
);
impl_spatial_traits!(
    f16,
    f32,
    *const u16,
    cast_to_u16,
    nk_angulars_packed_f16,
    nk_angulars_symmetric_f16,
    nk_euclideans_packed_f16,
    nk_euclideans_symmetric_f16
);
impl_spatial_traits!(
    bf16,
    f32,
    *const u16,
    cast_to_u16,
    nk_angulars_packed_bf16,
    nk_angulars_symmetric_bf16,
    nk_euclideans_packed_bf16,
    nk_euclideans_symmetric_bf16
);
impl_spatial_traits!(
    i8,
    f32,
    *const i8,
    identity_i8,
    nk_angulars_packed_i8,
    nk_angulars_symmetric_i8,
    nk_euclideans_packed_i8,
    nk_euclideans_symmetric_i8
);
impl_spatial_traits!(
    u8,
    f32,
    *const u8,
    identity_u8,
    nk_angulars_packed_u8,
    nk_angulars_symmetric_u8,
    nk_euclideans_packed_u8,
    nk_euclideans_symmetric_u8
);
impl_spatial_traits!(
    e4m3,
    f32,
    *const u8,
    cast_to_u8,
    nk_angulars_packed_e4m3,
    nk_angulars_symmetric_e4m3,
    nk_euclideans_packed_e4m3,
    nk_euclideans_symmetric_e4m3
);
impl_spatial_traits!(
    e5m2,
    f32,
    *const u8,
    cast_to_u8,
    nk_angulars_packed_e5m2,
    nk_angulars_symmetric_e5m2,
    nk_euclideans_packed_e5m2,
    nk_euclideans_symmetric_e5m2
);
impl_spatial_traits!(
    e2m3,
    f32,
    *const u8,
    cast_to_u8,
    nk_angulars_packed_e2m3,
    nk_angulars_symmetric_e2m3,
    nk_euclideans_packed_e2m3,
    nk_euclideans_symmetric_e2m3
);
impl_spatial_traits!(
    e3m2,
    f32,
    *const u8,
    cast_to_u8,
    nk_angulars_packed_e3m2,
    nk_angulars_symmetric_e3m2,
    nk_euclideans_packed_e3m2,
    nk_euclideans_symmetric_e3m2
);
// Manual impls for 4-bit packed types: k must be multiplied by 2 (storage → nibbles).

impl Angulars for u4x2 {
    type SpatialResult = f32;
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_u4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Euclideans for u4x2 {
    type SpatialResult = f32;
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_u4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Angulars for i4x2 {
    type SpatialResult = f32;
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_i4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Euclideans for i4x2 {
    type SpatialResult = f32;
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth * 2,
            a_stride,
            c_stride,
        )
    }
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_i4(
            vectors as *const u8,
            n_vectors,
            depth * 2,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

// endregion: Angulars Trait

// region: PackedMatrix

/// Pre-packed B matrix for efficient repeated GEMM operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// When multiplying A × Bᵀ multiple times with the same B matrix,
/// packing B once and reusing it is much faster than packing each time.
///
/// # Usage
///
/// For C = A × Bᵀ where B is (n × k):
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = PackedMatrix::try_pack(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
///
/// For C = A × B where B is (k × n) (standard GEMM layout):
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = PackedMatrix::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
pub struct PackedMatrix<T: Dots, A: Allocator = Global> {
    /// Raw pointer to packed data buffer.
    data: NonNull<u8>,
    /// Size of the packed buffer in bytes.
    size: usize,
    /// Output columns (B width).
    width: usize,
    /// Inner dimension (depth).
    depth: usize,
    /// Allocator instance.
    alloc: A,
    _marker: PhantomData<T>,
}

// Safety: PackedMatrix owns its data and is just bytes
unsafe impl<T: Dots + Send, A: Allocator + Send> Send for PackedMatrix<T, A> {}
unsafe impl<T: Dots + Sync, A: Allocator + Sync> Sync for PackedMatrix<T, A> {}

impl<T: Dots, A: Allocator> Drop for PackedMatrix<T, A> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                let layout =
                    alloc::alloc::Layout::from_size_align_unchecked(self.size, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<T: Dots, A: Allocator + Clone> PackedMatrix<T, A> {
    /// Try to clone this packed matrix, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        if self.size == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                size: 0,
                width: self.width,
                depth: self.depth,
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            });
        }

        let layout = alloc::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = self
            .alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size);
        }
        Ok(Self {
            data: ptr,
            size: self.size,
            width: self.width,
            depth: self.depth,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: Dots, A: Allocator + Clone> Clone for PackedMatrix<T, A> {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("PackedMatrix clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<T: Dots, A: Allocator> PackedMatrix<T, A> {
    /// Pack B matrix where B is (n × k) row-major using a custom allocator.
    ///
    /// Result computes: C = A × Bᵀ
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - allocation fails
    pub fn try_pack_in<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        let (width, depth) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(width, depth);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            // Allocate with SIMD alignment
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Zero the memory
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            unsafe {
                T::dots_pack(
                    b.as_ptr(),
                    width,
                    depth,
                    b.stride_bytes(0) as usize,
                    data.as_ptr(),
                );
            }
        }

        Ok(Self {
            data,
            size,
            width,
            depth,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using a custom allocator.
    ///
    /// Internally transposes then packs.
    /// Result computes: C = A × B
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - allocation fails
    pub fn try_pack_transposed_in<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        let (depth, width) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(width, depth);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            // Allocate with SIMD alignment
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Zero the memory
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            // Pack with transposed view: column stride becomes row stride
            unsafe {
                T::dots_pack(
                    b.as_ptr(),
                    width,
                    depth,
                    core::mem::size_of::<T>(),
                    data.as_ptr(),
                );
            }
        }

        Ok(Self {
            data,
            size,
            width,
            depth,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A { &self.alloc }

    /// Returns dimensions (width, depth) of the original B matrix.
    pub fn dims(&self) -> (usize, usize) { (self.width, self.depth) }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
}

// Convenience methods using Global allocator
impl<T: Dots> PackedMatrix<T, Global> {
    /// Pack B matrix where B is (n × k) row-major using the global allocator.
    ///
    /// Result computes: C = A × Bᵀ
    pub fn try_pack<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_in(b, Global)
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using the global allocator.
    ///
    /// Result computes: C = A × B
    pub fn try_pack_transposed<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_transposed_in(b, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<BA: Allocator, const MAX_RANK: usize>(b: &Tensor<T, BA, MAX_RANK>) -> Self {
        Self::try_pack(b).expect("PackedMatrix::pack failed")
    }

    /// Convenience constructor that panics on error.
    pub fn pack_transposed<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Self {
        Self::try_pack_transposed(b).expect("PackedMatrix::pack_transposed failed")
    }
}

// endregion: PackedMatrix

// region: Tensor GEMM

impl<T: Dots, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK>
where
    T::Accumulator: Clone + Default,
{
    /// Dot-product multiply: C = self × packed_bᵀ
    ///
    /// self must be 2D (m × k) with contiguous rows.
    /// packed_b contains B (n × k) packed.
    /// Returns C (m × n) using the same allocator as self.
    ///
    /// Returns `Err` if:
    /// - self is not 2D
    /// - self has non-contiguous rows
    /// - inner dimensions don't match
    /// - output allocation fails
    pub fn try_dots_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<Tensor<T::Accumulator, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }

        let mut c = Tensor::try_full_in(&[m, n], T::Accumulator::default(), self.alloc.clone())?;
        unsafe {
            T::dots_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn dots_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Tensor<T::Accumulator, A, MAX_RANK> {
        self.try_dots_packed(packed_b).expect("dots_packed failed")
    }
}

impl<T: Dots, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Dot-product multiply into existing output (avoids allocation).
    pub fn try_dots_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::Accumulator, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        unsafe {
            T::dots_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

// Parallel dots_packed implementations, if ForkUnion is available
#[cfg(feature = "parallel")]
impl<T: Dots + Clone + Send + Sync, A: Allocator + Clone, const MAX_RANK: usize>
    Tensor<T, A, MAX_RANK>
where
    T::Accumulator: Clone + Default + Send + Sync,
{
    /// Parallel dot-product multiply into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    /// This is a non-allocating interface - you provide the output tensor.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `PackedMatrix::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, PackedMatrix};
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
    /// let b_packed = PackedMatrix::try_pack(&b).unwrap();
    /// let mut c = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    /// ```
    pub fn try_dots_packed_parallel_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::Accumulator, CA, CA_MAX_RANK>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;

        // Get actual thread count from pool
        let num_threads = pool.threads().max(1);
        let rows_per_thread = (m + num_threads - 1) / num_threads;

        // Distribute rows across threads using fork_union
        // Safety: Each thread writes to disjoint rows of C, so no data races.
        pool.for_threads(move |thread_idx, _colocation_idx| {
            // Configure each worker thread for optimal SIMD (including AMX)
            // This is idempotent and safe to call multiple times
            crate::capabilities::configure_thread();

            let row_start = thread_idx * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(m);

            if row_start < m {
                unsafe {
                    T::dots_packed(
                        a_ptr.as_ptr().add(row_start * k),
                        packed_ptr.as_ptr(),
                        c_ptr.as_ptr().add(row_start * n),
                        row_end - row_start,
                        n,
                        k,
                        a_stride,
                        c_stride,
                    );
                }
            }
        })
        .join();

        Ok(())
    }

    /// Parallel dot-product multiply with allocation.
    ///
    /// Convenience wrapper that allocates the output tensor.
    /// Prefer `try_dots_packed_parallel_into` for performance-critical code.
    pub fn try_dots_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::Accumulator, Global, MAX_RANK>::try_full(
            &[m, n],
            T::Accumulator::default(),
        )?;
        self.try_dots_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn dots_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::Accumulator, Global, MAX_RANK> {
        self.try_dots_packed_parallel(packed_b, pool)
            .expect("parallel dots_packed failed")
    }
}

/// Compute row assignment for a thread without allocation
///
/// For a symmetric matrix, cumulative work up to row r is: r*(2n - r + 1)/2
/// Solving r*(2n - r + 1)/2 = work using quadratic formula gives exact row.
#[cfg(feature = "std")]
#[inline]
fn compute_thread_rows(thread_idx: usize, num_threads: usize, n: usize) -> (usize, usize) {
    let total_work = n * (n + 1) / 2;
    let work_per_thread = (total_work + num_threads - 1) / num_threads;

    let work_start = thread_idx * work_per_thread;
    let work_end = ((thread_idx + 1) * work_per_thread).min(total_work);

    // Solve: r^2 - r(2n + 1) + 2*work = 0
    // Using quadratic formula: r = (2n + 1 - sqrt((2n + 1)^2 - 8*work)) / 2
    let start_row = if work_start == 0 {
        0
    } else {
        let n_f64 = n as f64;
        let work_f64 = work_start as f64;
        let discriminant = (2.0 * n_f64 + 1.0).powi(2) - 8.0 * work_f64;
        let row_f64 = (2.0 * n_f64 + 1.0 - discriminant.sqrt()) / 2.0;
        row_f64.floor() as usize
    };

    let end_row = if work_end >= total_work {
        n
    } else {
        let n_f64 = n as f64;
        let work_f64 = work_end as f64;
        let discriminant = (2.0 * n_f64 + 1.0).powi(2) - 8.0 * work_f64;
        let row_f64 = (2.0 * n_f64 + 1.0 - discriminant.sqrt()) / 2.0;
        row_f64.ceil() as usize
    };

    (start_row, end_row - start_row)
}

#[cfg(feature = "parallel")]
impl<T: Dots + Clone + Send + Sync, A: Allocator + Clone, const MAX_RANK: usize>
    Tensor<T, A, MAX_RANK>
where
    T::Accumulator: Clone + Default + Send + Sync,
{
    /// Parallel computation of symmetric Gram matrix C = A × Aᵀ.
    ///
    /// Distributes rows across threads with balanced work distribution based on the
    /// triangular structure of symmetric matrix computation.
    ///
    /// # Arguments
    /// * `pool` - Pre-constructed thread pool
    ///
    /// # Example
    /// ```ignore
    /// use numkong::Tensor;
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let vectors = Tensor::<f32>::try_full(&[100, 768], 1.0).unwrap();
    /// let gram = vectors.try_dots_symmetric_parallel(&mut pool).unwrap();
    /// assert_eq!(gram.shape(), &[100, 100]);
    /// ```
    pub fn try_dots_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }

        let (n, k) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::<T::Accumulator, Global, MAX_RANK>::try_full(
            &[n, n],
            T::Accumulator::default(),
        )?;

        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = result.stride_bytes(0) as usize;

        pool.for_threads(move |thread_idx, _colocation_idx| {
            crate::capabilities::configure_thread();

            // Compute row assignment inline (no heap allocation)
            let (row_start, row_count) = compute_thread_rows(thread_idx, num_threads, n);

            unsafe {
                T::dots_symmetric(
                    vectors_ptr.as_ptr(),
                    n,
                    k,
                    stride,
                    result_ptr.as_ptr(),
                    result_stride,
                    row_start,
                    row_count,
                );
            }
        })
        .join();

        Ok(result)
    }

    /// Parallel computation of symmetric dot-product matrix (unwrapping version).
    ///
    /// # Panics
    /// Panics if the operation fails (e.g., wrong tensor rank).
    pub fn dots_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::Accumulator, Global, MAX_RANK> {
        self.try_dots_symmetric_parallel(pool)
            .expect("parallel dots_symmetric failed")
    }
}

// endregion: Tensor GEMM

// region: Tensor Spatial Distances

impl<T: Angulars, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes angular distances between rows of self and packed B matrix.
    pub fn try_angulars_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<Tensor<T::SpatialResult, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        let mut c = Tensor::try_full_in(&[m, n], T::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            T::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn angulars_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Tensor<T::SpatialResult, A, MAX_RANK> {
        self.try_angulars_packed(packed_b)
            .expect("angulars_packed failed")
    }
}

impl<T: Angulars, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes angular distances into pre-allocated output.
    pub fn try_angulars_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::SpatialResult, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        unsafe {
            T::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

impl<T: Euclideans, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes euclidean distances between rows of self and packed B matrix.
    pub fn try_euclideans_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<Tensor<T::SpatialResult, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        let mut c = Tensor::try_full_in(&[m, n], T::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            T::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Tensor<T::SpatialResult, A, MAX_RANK> {
        self.try_euclideans_packed(packed_b)
            .expect("euclideans_packed failed")
    }
}

impl<T: Euclideans, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes euclidean distances into pre-allocated output.
    pub fn try_euclideans_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::SpatialResult, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        unsafe {
            T::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

// Parallel spatial distance implementations
#[cfg(feature = "parallel")]
impl<T: Angulars + Clone + Send + Sync, A: Allocator + Clone, const MAX_RANK: usize>
    Tensor<T, A, MAX_RANK>
where
    T::SpatialResult: Clone + Default + Send + Sync,
{
    /// Parallel angular distances into pre-allocated output.
    pub fn try_angulars_packed_parallel_into<
        BA: Allocator,
        CA: Allocator,
        const CA_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::SpatialResult, CA, CA_MAX_RANK>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = (m + num_threads - 1) / num_threads;

        pool.for_threads(move |thread_idx, _colocation_idx| {
            crate::capabilities::configure_thread();
            let row_start = thread_idx * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(m);
            if row_start < m {
                unsafe {
                    T::angulars_packed(
                        a_ptr.as_ptr().add(row_start * k),
                        packed_ptr.as_ptr(),
                        c_ptr.as_ptr().add(row_start * n),
                        row_end - row_start,
                        n,
                        k,
                        a_stride,
                        c_stride,
                    );
                }
            }
        })
        .join();
        Ok(())
    }

    /// Parallel angular distances with allocation.
    pub fn try_angulars_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[m, n],
            T::SpatialResult::default(),
        )?;
        self.try_angulars_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn angulars_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_packed_parallel(packed_b, pool)
            .expect("parallel angulars_packed failed")
    }

    /// Parallel symmetric angular distance matrix.
    pub fn try_angulars_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (n, k) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = result.stride_bytes(0) as usize;

        pool.for_threads(move |thread_idx, _colocation_idx| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_idx, num_threads, n);
            unsafe {
                T::angulars_symmetric(
                    vectors_ptr.as_ptr(),
                    n,
                    k,
                    stride,
                    result_ptr.as_ptr(),
                    result_stride,
                    row_start,
                    row_count,
                );
            }
        })
        .join();
        Ok(result)
    }

    /// Convenience method that panics on error.
    pub fn angulars_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_symmetric_parallel(pool)
            .expect("parallel angulars_symmetric failed")
    }
}

#[cfg(feature = "parallel")]
impl<T: Euclideans + Clone + Send + Sync, A: Allocator + Clone, const MAX_RANK: usize>
    Tensor<T, A, MAX_RANK>
where
    T::SpatialResult: Clone + Default + Send + Sync,
{
    /// Parallel euclidean distances into pre-allocated output.
    pub fn try_euclideans_packed_parallel_into<
        BA: Allocator,
        CA: Allocator,
        const CA_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::SpatialResult, CA, CA_MAX_RANK>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = (m + num_threads - 1) / num_threads;

        pool.for_threads(move |thread_idx, _colocation_idx| {
            crate::capabilities::configure_thread();
            let row_start = thread_idx * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(m);
            if row_start < m {
                unsafe {
                    T::euclideans_packed(
                        a_ptr.as_ptr().add(row_start * k),
                        packed_ptr.as_ptr(),
                        c_ptr.as_ptr().add(row_start * n),
                        row_end - row_start,
                        n,
                        k,
                        a_stride,
                        c_stride,
                    );
                }
            }
        })
        .join();
        Ok(())
    }

    /// Parallel euclidean distances with allocation.
    pub fn try_euclideans_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[m, n],
            T::SpatialResult::default(),
        )?;
        self.try_euclideans_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_packed_parallel(packed_b, pool)
            .expect("parallel euclideans_packed failed")
    }

    /// Parallel symmetric euclidean distance matrix.
    pub fn try_euclideans_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (n, k) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = result.stride_bytes(0) as usize;

        pool.for_threads(move |thread_idx, _colocation_idx| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_idx, num_threads, n);
            unsafe {
                T::euclideans_symmetric(
                    vectors_ptr.as_ptr(),
                    n,
                    k,
                    stride,
                    result_ptr.as_ptr(),
                    result_stride,
                    row_start,
                    row_count,
                );
            }
        })
        .join();
        Ok(result)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_symmetric_parallel(pool)
            .expect("parallel euclideans_symmetric failed")
    }
}

// endregion: Tensor Spatial Distances

// region: Tensor Hammings/Jaccards

impl<T: Hammings, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Hamming distances between rows of self and packed B matrix.
    pub fn try_hammings_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<Tensor<u32, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        let mut c = Tensor::try_full_in(&[m, n], u32::default(), self.alloc.clone())?;
        unsafe {
            T::hammings_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn hammings_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Tensor<u32, A, MAX_RANK> {
        self.try_hammings_packed(packed_b)
            .expect("hammings_packed failed")
    }
}

impl<T: Hammings, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Hamming distances into pre-allocated output.
    pub fn try_hammings_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<u32, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        unsafe {
            T::hammings_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

impl<T: Jaccards, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Jaccard distances between rows of self and packed B matrix.
    pub fn try_jaccards_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<Tensor<T::JaccardResult, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        let mut c = Tensor::try_full_in(&[m, n], T::JaccardResult::default(), self.alloc.clone())?;
        unsafe {
            T::jaccards_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn jaccards_packed<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Tensor<T::JaccardResult, A, MAX_RANK> {
        self.try_jaccards_packed(packed_b)
            .expect("jaccards_packed failed")
    }
}

impl<T: Jaccards, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Jaccard distances into pre-allocated output.
    pub fn try_jaccards_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut Tensor<T::JaccardResult, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        unsafe {
            T::jaccards_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

// endregion: Tensor Hammings/Jaccards

impl<'a, T: Dots, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Accumulator: Clone + Default + 'static,
{
    /// Computes the symmetric dot-product matrix C = A × Aᵀ.
    ///
    /// Given a matrix of row vectors, computes the matrix of all pairwise dot products.
    /// The result is a symmetric n×n matrix where result\[i,j\] = dot(row_i, row_j).
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, TensorView};
    ///
    /// // 100 vectors of dimension 768
    /// let vectors = Tensor::<f32>::try_full(&[100, 768], 0.0)?;
    ///
    /// // Compute 100×100 symmetric matrix
    /// let gram = vectors.view().try_dots_symmetric()?;
    /// assert_eq!(gram.shape(), &[100, 100]);
    /// ```
    pub fn try_dots_symmetric(
        &self,
    ) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "try_dots_symmetric requires 2D tensor",
            });
        }

        let n_vectors = shape[0];
        let depth = shape[1];

        // Result is n_vectors × n_vectors symmetric matrix
        let result_shape = &[n_vectors, n_vectors];
        let mut result = Tensor::<T::Accumulator, Global, MAX_RANK>::try_full(
            result_shape,
            T::Accumulator::default(),
        )?;

        unsafe {
            T::dots_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }

        Ok(result)
    }
}

impl<'a, T: Angulars, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Computes symmetric angular distance matrix for a set of vectors.
    pub fn try_angulars_symmetric(
        &self,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "try_angulars_symmetric requires 2D tensor",
            });
        }
        let n_vectors = shape[0];
        let depth = shape[1];
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::angulars_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }

        Ok(result)
    }
}

impl<'a, T: Euclideans, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Computes symmetric euclidean distance matrix for a set of vectors.
    pub fn try_euclideans_symmetric(
        &self,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "try_euclideans_symmetric requires 2D tensor",
            });
        }
        let n_vectors = shape[0];
        let depth = shape[1];
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::euclideans_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }

        Ok(result)
    }
}

impl<'a, T: Hammings, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Computes symmetric Hamming distance matrix for a set of binary vectors.
    pub fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "try_hammings_symmetric requires 2D tensor",
            });
        }
        let n_vectors = shape[0];
        let depth = shape[1];
        let mut result =
            Tensor::<u32, Global, MAX_RANK>::try_full(&[n_vectors, n_vectors], u32::default())?;
        unsafe {
            T::hammings_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }
        Ok(result)
    }
}

impl<'a, T: Jaccards, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Computes symmetric Jaccard distance matrix for a set of binary vectors.
    pub fn try_jaccards_symmetric(
        &self,
    ) -> Result<Tensor<T::JaccardResult, Global, MAX_RANK>, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "try_jaccards_symmetric requires 2D tensor",
            });
        }
        let n_vectors = shape[0];
        let depth = shape[1];
        let mut result = Tensor::<T::JaccardResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            T::JaccardResult::default(),
        )?;
        unsafe {
            T::jaccards_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }
        Ok(result)
    }
}

// endregion: TensorView

// region: Symmetric Extension Traits

/// Extension trait: symmetric dot-product matrix for any [`TensorRef`] implementor.
pub trait SymmetricDots<T: Dots, const MAX_RANK: usize>: TensorRef<T, MAX_RANK>
where
    T::Accumulator: Clone + Default + 'static,
{
    fn try_dots_symmetric(&self) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        self.view().try_dots_symmetric()
    }
}

impl<T: Dots, const R: usize, C: TensorRef<T, R>> SymmetricDots<T, R> for C where
    T::Accumulator: Clone + Default + 'static
{
}

/// Extension trait: symmetric angular distance matrix for any [`TensorRef`] implementor.
pub trait SymmetricAngulars<T: Angulars, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_angulars_symmetric(
        &self,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_angulars_symmetric()
    }
}

impl<T: Angulars, const R: usize, C: TensorRef<T, R>> SymmetricAngulars<T, R> for C {}

/// Extension trait: symmetric euclidean distance matrix for any [`TensorRef`] implementor.
pub trait SymmetricEuclideans<T: Euclideans, const MAX_RANK: usize>:
    TensorRef<T, MAX_RANK>
{
    fn try_euclideans_symmetric(
        &self,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_euclideans_symmetric()
    }
}

impl<T: Euclideans, const R: usize, C: TensorRef<T, R>> SymmetricEuclideans<T, R> for C {}

/// Extension trait: symmetric Hamming distance matrix for any [`TensorRef`] implementor.
pub trait SymmetricHammings<T: Hammings, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        self.view().try_hammings_symmetric()
    }
}

impl<T: Hammings, const R: usize, C: TensorRef<T, R>> SymmetricHammings<T, R> for C {}

/// Extension trait: symmetric Jaccard distance matrix for any [`TensorRef`] implementor.
pub trait SymmetricJaccards<T: Jaccards, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_jaccards_symmetric(
        &self,
    ) -> Result<Tensor<T::JaccardResult, Global, MAX_RANK>, TensorError> {
        self.view().try_jaccards_symmetric()
    }
}

impl<T: Jaccards, const R: usize, C: TensorRef<T, R>> SymmetricJaccards<T, R> for C {}

// endregion: Symmetric Extension Traits

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FloatLike, NumberLike, TestableType};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_thread() {
        INIT.call_once(|| {
            crate::capabilities::configure_thread();
        });
    }

    const DIMS: &[(usize, usize, usize)] =
        &[(1, 1, 1), (1, 8, 3), (3, 1, 7), (7, 5, 3), (33, 17, 65)];

    fn check_dots_packed<T: TestableType + Dots>()
    where
        T::Accumulator: Clone + Default + Copy + FloatLike,
    {
        init_thread();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(c.shape(), &[m, n], "shape @ ({m},{n},{k})");
            let expected = T::dimensions_per_value() as f64 * k as f64;
            let tol = T::atol() + T::rtol() * expected.abs();
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    (v.to_f64() - expected).abs() <= tol,
                    "({m},{n},{k})[{i}]: {} vs {expected} (tol={tol})",
                    v.to_f64()
                );
            }
        }
    }

    fn check_dots_packed_transposed<T: TestableType + Dots>()
    where
        T::Accumulator: Clone + Default + Copy + FloatLike,
    {
        init_thread();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b_t = Tensor::<T>::try_full(&[k, n], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack_transposed(&b_t).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(c.shape(), &[m, n], "shape @ ({m},{n},{k})");
            let expected = T::dimensions_per_value() as f64 * k as f64;
            let tol = T::atol() + T::rtol() * expected.abs();
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    (v.to_f64() - expected).abs() <= tol,
                    "({m},{n},{k})[{i}]: {} vs {expected} (tol={tol})",
                    v.to_f64()
                );
            }
        }
    }

    fn check_angulars_packed<T: TestableType + Angulars>()
    where
        T::SpatialResult: Clone + Default + Copy + FloatLike,
    {
        init_thread();
        let tol = T::atol();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.angulars_packed(&b_packed);
            assert_eq!(c.shape(), &[m, n], "shape @ ({m},{n},{k})");
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({m},{n},{k})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
        }
    }

    fn check_euclideans_packed<T: TestableType + Euclideans>()
    where
        T::SpatialResult: Clone + Default + Copy + FloatLike,
    {
        init_thread();
        let tol = T::atol();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.euclideans_packed(&b_packed);
            assert_eq!(c.shape(), &[m, n], "shape @ ({m},{n},{k})");
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({m},{n},{k})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn check_dots_packed_parallel<T: TestableType + Dots + Send + Sync>()
    where
        T::Accumulator: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let serial = a.dots_packed(&b_packed);
            let parallel = a.dots_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({m},{n},{k})"
            );
        }
    }

    #[cfg(feature = "parallel")]
    fn check_angulars_packed_parallel<T: TestableType + Angulars + Send + Sync>()
    where
        T::SpatialResult: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let serial = a.angulars_packed(&b_packed);
            let parallel = a.angulars_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({m},{n},{k})"
            );
        }
    }

    #[cfg(feature = "parallel")]
    fn check_euclideans_packed_parallel<T: TestableType + Euclideans + Send + Sync>()
    where
        T::SpatialResult: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(m, n, k) in DIMS {
            let a = Tensor::<T>::try_full(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_full(&[n, k], T::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let serial = a.euclideans_packed(&b_packed);
            let parallel = a.euclideans_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({m},{n},{k})"
            );
        }
    }

    #[test]
    fn dots_packed() {
        check_dots_packed::<f32>();
        check_dots_packed::<f64>();
        check_dots_packed::<f16>();
        check_dots_packed::<bf16>();
        check_dots_packed::<e4m3>();
        check_dots_packed::<e5m2>();
        check_dots_packed::<e2m3>();
        check_dots_packed::<e3m2>();
        check_dots_packed::<i8>();
        check_dots_packed::<u8>();
        check_dots_packed::<i4x2>();
        check_dots_packed::<u4x2>();
    }

    #[test]
    fn dots_packed_transposed() {
        check_dots_packed_transposed::<f32>();
        check_dots_packed_transposed::<f64>();
        check_dots_packed_transposed::<f16>();
        check_dots_packed_transposed::<bf16>();
        check_dots_packed_transposed::<e4m3>();
        check_dots_packed_transposed::<e5m2>();
        check_dots_packed_transposed::<e2m3>();
        check_dots_packed_transposed::<e3m2>();
        check_dots_packed_transposed::<i8>();
        check_dots_packed_transposed::<u8>();
        check_dots_packed_transposed::<i4x2>();
        check_dots_packed_transposed::<u4x2>();
    }

    #[test]
    fn angulars_packed() {
        check_angulars_packed::<f32>();
        check_angulars_packed::<f64>();
        check_angulars_packed::<f16>();
        check_angulars_packed::<bf16>();
        check_angulars_packed::<e4m3>();
        check_angulars_packed::<e5m2>();
        check_angulars_packed::<e2m3>();
        check_angulars_packed::<e3m2>();
        check_angulars_packed::<i8>();
        check_angulars_packed::<u8>();
        check_angulars_packed::<i4x2>();
        check_angulars_packed::<u4x2>();
    }

    #[test]
    fn euclideans_packed() {
        check_euclideans_packed::<f32>();
        check_euclideans_packed::<f64>();
        check_euclideans_packed::<f16>();
        check_euclideans_packed::<bf16>();
        check_euclideans_packed::<e4m3>();
        check_euclideans_packed::<e5m2>();
        check_euclideans_packed::<e2m3>();
        check_euclideans_packed::<e3m2>();
        check_euclideans_packed::<i8>();
        check_euclideans_packed::<u8>();
        check_euclideans_packed::<i4x2>();
        check_euclideans_packed::<u4x2>();
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn packed_parallel() {
        check_dots_packed_parallel::<f32>();
        check_dots_packed_parallel::<bf16>();
        check_angulars_packed_parallel::<f32>();
        check_euclideans_packed_parallel::<f32>();
    }

    fn check_dots_symmetric<T: TestableType + Dots>()
    where
        T::Accumulator: Clone + Default + Copy + FloatLike + 'static,
    {
        init_thread();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let vectors = Tensor::<T>::try_full(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_dots_symmetric().unwrap();
            assert_eq!(
                gram_matrix.shape(),
                &[num_vectors, num_vectors],
                "shape @ ({num_vectors},{depth})"
            );
            let expected = T::dimensions_per_value() as f64 * depth as f64;
            let tolerance = T::atol() + T::rtol() * expected.abs();
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        (value.to_f64() - expected).abs() <= tolerance,
                        "({num_vectors},{depth})[{i},{j}]: {} vs {expected}",
                        value.to_f64()
                    );
                }
            }
        }
    }

    fn check_angulars_symmetric<T: TestableType + Angulars>()
    where
        T::SpatialResult: Clone + Default + Copy + FloatLike + 'static,
    {
        init_thread();
        let tolerance = T::atol();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let vectors = Tensor::<T>::try_full(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_angulars_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        value.to_f64().abs() <= tolerance,
                        "angular symmetric [{i},{j}]: {}",
                        value.to_f64()
                    );
                }
            }
        }
    }

    fn check_euclideans_symmetric<T: TestableType + Euclideans>()
    where
        T::SpatialResult: Clone + Default + Copy + FloatLike + 'static,
    {
        init_thread();
        let tolerance = T::atol();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let vectors = Tensor::<T>::try_full(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_euclideans_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        value.to_f64().abs() <= tolerance,
                        "euclidean symmetric [{i},{j}]: {}",
                        value.to_f64()
                    );
                }
            }
        }
    }

    #[test]
    fn dots_symmetric() {
        check_dots_symmetric::<f32>();
        check_dots_symmetric::<f64>();
        check_dots_symmetric::<f16>();
        check_dots_symmetric::<bf16>();
        check_dots_symmetric::<e4m3>();
        check_dots_symmetric::<e5m2>();
        check_dots_symmetric::<e2m3>();
        check_dots_symmetric::<e3m2>();
        check_dots_symmetric::<i8>();
        check_dots_symmetric::<u8>();
        check_dots_symmetric::<i4x2>();
        check_dots_symmetric::<u4x2>();
    }

    #[test]
    fn angulars_symmetric() {
        check_angulars_symmetric::<f32>();
        check_angulars_symmetric::<f64>();
        check_angulars_symmetric::<f16>();
        check_angulars_symmetric::<bf16>();
        check_angulars_symmetric::<e4m3>();
        check_angulars_symmetric::<e5m2>();
        check_angulars_symmetric::<e2m3>();
        check_angulars_symmetric::<e3m2>();
        check_angulars_symmetric::<i8>();
        check_angulars_symmetric::<u8>();
        check_angulars_symmetric::<i4x2>();
        check_angulars_symmetric::<u4x2>();
    }

    #[test]
    fn euclideans_symmetric() {
        check_euclideans_symmetric::<f32>();
        check_euclideans_symmetric::<f64>();
        check_euclideans_symmetric::<f16>();
        check_euclideans_symmetric::<bf16>();
        check_euclideans_symmetric::<e4m3>();
        check_euclideans_symmetric::<e5m2>();
        check_euclideans_symmetric::<e2m3>();
        check_euclideans_symmetric::<e3m2>();
        check_euclideans_symmetric::<i8>();
        check_euclideans_symmetric::<u8>();
        check_euclideans_symmetric::<i4x2>();
        check_euclideans_symmetric::<u4x2>();
    }

    #[test]
    fn binary_packed_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 8], u1x8(0xFF)).unwrap();
        let b = Tensor::<u1x8>::try_full(&[16, 8], u1x8(0xFF)).unwrap();
        let b_packed = PackedMatrix::try_pack(&b).unwrap();
        let c = a.dots_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 64);
        let c = a.hammings_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 0);
        let c = a.jaccards_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert!(c.as_slice()[0].abs() < 1e-5);
    }

    #[test]
    fn binary_symmetric_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 8], u1x8(0xFF)).unwrap();
        let gram = a.view().try_dots_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert_eq!(gram.as_slice()[0], 64);
        let gram = a.try_hammings_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert_eq!(gram.as_slice()[0], 0);
        let gram = a.try_jaccards_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert!(gram.as_slice()[0].abs() < 1e-5);
    }
}
