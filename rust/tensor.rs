//! N-dimensional tensor types, GEMM, and spatial distance operations.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorSpan`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//! - [`PackedMatrix`]: Pre-packed matrix for efficient GEMM
//!
//! ## Packed spatial operations
//!
//! - [`Dots`]: Matrix dot-product with pre-packing for cache efficiency
//! - [`Angulars`]: Angular/cosine distances on packed matrices
//! - [`Euclideans`]: Euclidean distances on packed matrices
//! - [`Hammings`]: Hamming distances for binary (u1x8) matrices
//! - [`Jaccards`]: Jaccard distances for binary (u1x8) matrices
//!
//! # Example
//!
//! ```rust,ignore
//! // Requires linking against libnumkong C library
//! use numkong::{Tensor, PackedMatrix};
//!
//! let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
//! let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
//!
//! // Pack B once, multiply many times
//! let b_packed = PackedMatrix::try_pack(&b).unwrap();
//! let c = a.dots_packed(&b_packed);  // Returns (1024 × 256)
//! ```

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::numerics::{
    cast, CastDtype, Dot, EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum,
    ReduceMinMax, ReduceMoments, Roots,
};
use crate::scalar::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};
use crate::vector::VecIndex;

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_packed_size_f32(n: usize, k: usize) -> usize;
    fn nk_dots_pack_f32(b: *const f32, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_f64(n: usize, k: usize) -> usize;
    fn nk_dots_pack_f64(b: *const f64, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_f16(n: usize, k: usize) -> usize;
    fn nk_dots_pack_f16(b: *const u16, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_bf16(n: usize, k: usize) -> usize;
    fn nk_dots_pack_bf16(b: *const u16, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_i8(n: usize, k: usize) -> usize;
    fn nk_dots_pack_i8(b: *const i8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_u8(n: usize, k: usize) -> usize;
    fn nk_dots_pack_u8(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e4m3(n: usize, k: usize) -> usize;
    fn nk_dots_pack_e4m3(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e5m2(n: usize, k: usize) -> usize;
    fn nk_dots_pack_e5m2(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e2m3(n: usize, k: usize) -> usize;
    fn nk_dots_pack_e2m3(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_e3m2(n: usize, k: usize) -> usize;
    fn nk_dots_pack_e3m2(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_u4(n: usize, k: usize) -> usize;
    fn nk_dots_pack_u4(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_packed_size_i4(n: usize, k: usize) -> usize;
    fn nk_dots_pack_i4(b: *const u8, n: usize, k: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut i32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );

    // Symmetric Gram matrix (C = A × Aᵀ)
    fn nk_dots_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
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

    fn nk_dots_packed_size_u1(n: usize, d: usize) -> usize;
    fn nk_dots_pack_u1(q: *const u8, n: usize, d: usize, q_stride: usize, q_packed: *mut u8);
    fn nk_dots_packed_u1(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: usize,
        n: usize,
        k: usize,
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
        rows: usize,
        cols: usize,
        d: usize,
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
        rows: usize,
        cols: usize,
        d: usize,
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
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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
        m: usize,
        n: usize,
        k: usize,
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

// region: Constants and Allocator

/// Default maximum rank for tensors.
pub const DEFAULT_MAX_RANK: usize = 8;

/// Alignment for SIMD-friendly allocations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Memory allocator trait for custom allocation strategies.
///
/// Implement this trait to use custom allocators (arena, pool, etc.) with
/// [`Tensor`] and [`PackedMatrix`].
///
/// # Safety
///
/// Implementations must ensure:
/// - `allocate` returns a valid, properly aligned pointer on success
/// - `deallocate` is called with the same `Layout` used in `allocate`
/// - The returned memory is not aliased
pub unsafe trait Allocator {
    /// Allocates memory with the given layout.
    ///
    /// Returns `None` if allocation fails.
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>>;

    /// Deallocates memory previously allocated with `allocate`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a previous call to `allocate`
    /// - `layout` must be the same as the one used for allocation
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout);
}

/// Global allocator using the system allocator.
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            // Return a dangling but aligned pointer for zero-size allocations
            return Some(NonNull::new(layout.align() as *mut u8).unwrap_or(NonNull::dangling()));
        }
        unsafe {
            let ptr = alloc::alloc::alloc(layout);
            NonNull::new(ptr)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout) {
        if layout.size() > 0 {
            alloc::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}

// endregion: Constants and Allocator

// region: Error Types

/// Fixed-size shape descriptor for error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeDescriptor {
    dims: [usize; DEFAULT_MAX_RANK],
    ndim: usize,
}

impl ShapeDescriptor {
    /// Create from a slice (truncates if > DEFAULT_MAX_RANK).
    pub fn from_slice(shape: &[usize]) -> Self {
        let mut dims = [0usize; DEFAULT_MAX_RANK];
        let ndim = shape.len().min(DEFAULT_MAX_RANK);
        dims[..ndim].copy_from_slice(&shape[..ndim]);
        Self { dims, ndim }
    }

    /// Return as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.dims[..self.ndim]
    }
}

impl core::fmt::Display for ShapeDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, &d) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Error type for Tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Memory allocation failed.
    AllocationFailed,
    /// Shape mismatch between arrays.
    ShapeMismatch {
        expected: ShapeDescriptor,
        got: ShapeDescriptor,
    },
    /// Invalid shape specification.
    InvalidShape {
        shape: ShapeDescriptor,
        reason: &'static str,
    },
    /// Operation requires contiguous rows but array has non-contiguous rows.
    NonContiguousRows,
    /// Expected a specific number of dimensions.
    DimensionMismatch { expected: usize, got: usize },
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, size: usize },
    /// Too many dimensions (exceeds MAX_RANK).
    TooManyRanks { got: usize },
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::AllocationFailed => write!(f, "memory allocation failed"),
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, got)
            }
            TensorError::InvalidShape { shape, reason } => {
                write!(f, "invalid shape {}: {}", shape, reason)
            }
            TensorError::NonContiguousRows => {
                write!(f, "operation requires contiguous rows")
            }
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "expected {} dimensions, got {}", expected, got)
            }
            TensorError::IndexOutOfBounds { index, size } => {
                write!(f, "index {} out of bounds for size {}", index, size)
            }
            TensorError::TooManyRanks { got } => {
                write!(f, "too many ranks: {}", got)
            }
        }
    }
}

// endregion: Error Types

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
    fn dots_packed_size(n: usize, k: usize) -> usize;

    /// Packs the B matrix into an optimized backend-specific layout.
    ///
    /// # Safety
    /// - `b` must point to valid memory for `n * k` elements
    /// - `packed` must point to a buffer of at least `dots_packed_size(n, k)` bytes
    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8);

    /// Computes C = A × Bᵀ using packed B.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `m * k` elements with given stride
    /// - `packed` must be a buffer previously filled by `dots_pack`
    /// - `c` must point to valid memory for `m * n` elements with given stride
    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
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
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f32(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f32(b, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f32(a, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f64(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f64(b, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f64(a, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f16(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f16(b as *const u16, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f16(a as *const u16, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_bf16(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_bf16(b as *const u16, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_bf16(a as *const u16, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_i8(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i8(b, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i8(a, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u8(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u8(b, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u8(a, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e4m3(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e4m3(b as *const u8, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e4m3(a as *const u8, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e5m2(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e5m2(b as *const u8, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e5m2(a as *const u8, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e2m3(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e2m3(b as *const u8, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e2m3(a as *const u8, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e3m2(n, k) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e3m2(b as *const u8, n, k, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e3m2(a as *const u8, packed, c, m, n, k, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u4(n, k * 2) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u4(b as *const u8, n, k * 2, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_i4(n, k * 2) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i4(b as *const u8, n, k * 2, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u1(n, k * 8) }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u1(b as *const u8, n, k * 8, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u1(a as *const u8, packed, c, m, n, k * 8, a_stride, c_stride)
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
    /// - `result` must point to valid memory for `rows * cols` u32 elements
    unsafe fn hammings_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut u32,
        rows: usize,
        cols: usize,
        d: usize,
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
        d: usize,
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
        rows: usize,
        cols: usize,
        d: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_hammings_packed_u1(
            a as *const u8,
            q_packed,
            result,
            rows,
            cols,
            d * 8,
            v_stride,
            r_stride,
        )
    }

    unsafe fn hammings_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_hammings_symmetric_u1(
            vectors as *const u8,
            n_vectors,
            d * 8,
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
    /// - `result` must point to valid memory for `rows * cols` elements
    unsafe fn jaccards_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut Self::JaccardResult,
        rows: usize,
        cols: usize,
        d: usize,
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
        d: usize,
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
        rows: usize,
        cols: usize,
        d: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_jaccards_packed_u1(
            a as *const u8,
            q_packed,
            result,
            rows,
            cols,
            d * 8,
            v_stride,
            r_stride,
        )
    }

    unsafe fn jaccards_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut Self::JaccardResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_jaccards_symmetric_u1(
            vectors as *const u8,
            n_vectors,
            d * 8,
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
    /// Result type for angular distances (always f32 except f64 → f64).
    type SpatialResult: Clone + Default;

    /// Computes angular distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `m * k` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `m * n` result elements with given stride
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        m: usize,
        n: usize,
        k: usize,
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
    /// Result type for euclidean distances (always f32 except f64 → f64).
    type SpatialResult: Clone + Default;

    /// Computes euclidean distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `m * k` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `m * n` result elements with given stride
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        m: usize,
        n: usize,
        k: usize,
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
                m: usize,
                n: usize,
                k: usize,
                a_stride: usize,
                c_stride: usize,
            ) {
                $ang_packed($cast(a), packed, c, m, n, k, a_stride, c_stride)
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
                m: usize,
                n: usize,
                k: usize,
                a_stride: usize,
                c_stride: usize,
            ) {
                $euc_packed($cast(a), packed, c, m, n, k, a_stride, c_stride)
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
fn identity_f32(p: *const f32) -> *const f32 {
    p
}
#[inline(always)]
fn identity_f64(p: *const f64) -> *const f64 {
    p
}
#[inline(always)]
fn identity_i8(p: *const i8) -> *const i8 {
    p
}
#[inline(always)]
fn identity_u8(p: *const u8) -> *const u8 {
    p
}
#[inline(always)]
fn cast_to_u16<T>(p: *const T) -> *const u16 {
    p as *const u16
}
#[inline(always)]
fn cast_to_u8<T>(p: *const T) -> *const u8 {
    p as *const u8
}

impl_spatial_traits!(
    f32,
    f32,
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
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_u4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_u4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_i4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_i4(a as *const u8, packed, c, m, n, k * 2, a_stride, c_stride)
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

// region: Tensor

/// N-dimensional array with NumKong-accelerated operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// Supports:
/// - Slicing and subviews (zero-copy)
/// - Dot-product multiplication with [`PackedMatrix`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, blend, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// use numkong::{Tensor, PackedMatrix};
///
/// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let b_packed = PackedMatrix::try_pack(&b).unwrap();
/// let c = a.dots_packed(&b_packed);  // Returns (1024 × 256)
/// ```
pub struct Tensor<T, A: Allocator = Global, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Raw pointer to data buffer.
    data: NonNull<T>,
    /// Total number of elements.
    len: usize,
    /// Shape dimensions.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Allocator instance.
    alloc: A,
}

// Safety: Tensor owns its data and T: Send implies the array is Send
unsafe impl<T: Send, A: Allocator + Send, const MAX_RANK: usize> Send for Tensor<T, A, MAX_RANK> {}
// Safety: Tensor has no interior mutability, &Tensor<T> is safe to share if T: Sync
unsafe impl<T: Sync, A: Allocator + Sync, const MAX_RANK: usize> Sync for Tensor<T, A, MAX_RANK> {}

impl<T, A: Allocator, const MAX_RANK: usize> Drop for Tensor<T, A, MAX_RANK> {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                // Drop all elements
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.data.as_ptr(),
                    self.len,
                ));
                // Deallocate buffer using matching SIMD-aligned layout
                let layout = alloc::alloc::Layout::from_size_align(
                    self.len * core::mem::size_of::<T>(),
                    SIMD_ALIGNMENT,
                )
                .unwrap();
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

impl<T: Clone, A: Allocator + Clone, const MAX_RANK: usize> Clone for Tensor<T, A, MAX_RANK> {
    fn clone(&self) -> Self {
        Self::try_from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
            .expect("clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<T: Clone, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Creates a new Tensor filled with a value using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full_in(shape: &[usize], value: T, alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() && shape.iter().any(|&d| d == 0) {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "zero-sized dimension",
            });
        }

        // Allocate SIMD-aligned raw buffer using our allocator
        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Initialize all elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..total {
                    core::ptr::write(ptr.add(i), value.clone());
                }
                NonNull::new_unchecked(ptr)
            }
        };

        // Build shape and strides arrays
        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data,
            len: total,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a zero-initialized Tensor using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_zeros_in(shape: &[usize], alloc: A) -> Result<Self, TensorError>
    where
        T: Default,
    {
        Self::try_full_in(shape, T::default(), alloc)
    }

    /// Creates a Tensor filled with ones using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_ones_in(shape: &[usize], alloc: A) -> Result<Self, TensorError>
    where
        T: crate::scalar::NumberLike,
    {
        Self::try_full_in(shape, T::one(), alloc)
    }

    /// Creates an uninitialized Tensor using a custom allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty_in(shape: &[usize], alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() && shape.iter().any(|&d| d == 0) {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "zero-sized dimension",
            });
        }

        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data,
            len: total,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a Tensor from existing slice data using a custom allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice_in(data: &[T], shape: &[usize], alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(shape),
                got: ShapeDescriptor::from_slice(&[data.len()]),
            });
        }

        // Allocate SIMD-aligned buffer and copy using our allocator
        let ptr = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Clone all elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..total {
                    core::ptr::write(ptr.add(i), data[i].clone());
                }
                NonNull::new_unchecked(ptr)
            }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data: ptr,
            len: total,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    fn compute_strides_into(shape: &[usize], strides: &mut [isize; MAX_RANK]) {
        let elem_size = core::mem::size_of::<T>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size as isize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i] as isize;
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }
}

// Methods that don't require Clone
impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.len
    }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    /// Returns the underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Check if rows are contiguous (required for GEMM A matrix).
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        // Last dimension stride should be element size
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Returns a row of a 2D array.
    pub fn row(&self, i: usize) -> Option<&[T]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&self.as_slice()[start..start + cols])
    }

    /// Returns a mutable row of a 2D array.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [T]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&mut self.as_mut_slice()[start..start + cols])
    }
}

// Convenience methods using Global allocator
impl<T: Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Creates a new Tensor filled with a value using the global allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full(shape: &[usize], value: T) -> Result<Self, TensorError> {
        Self::try_full_in(shape, value, Global)
    }

    /// Creates a zero-initialized Tensor using the global allocator.
    pub fn try_zeros(shape: &[usize]) -> Result<Self, TensorError>
    where
        T: Default,
    {
        Self::try_zeros_in(shape, Global)
    }

    /// Creates a Tensor filled with ones using the global allocator.
    pub fn try_ones(shape: &[usize]) -> Result<Self, TensorError>
    where
        T: crate::scalar::NumberLike,
    {
        Self::try_ones_in(shape, Global)
    }

    /// Creates an uninitialized Tensor using the global allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty(shape: &[usize]) -> Result<Self, TensorError> {
        unsafe { Self::try_empty_in(shape, Global) }
    }

    /// Creates a Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[T], shape: &[usize]) -> Result<Self, TensorError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self {
        Self::try_from_slice(data, shape).expect("Tensor::from_slice failed")
    }
}

// endregion: Tensor

// region: SliceRange

/// Represents a range specification for slicing along one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceRange {
    /// Full range (equivalent to `..`)
    Full,
    /// Single index (reduces dimension)
    Index(usize),
    /// Range from start to end exclusive (equivalent to `start..end`)
    Range { start: usize, end: usize },
    /// Range from start to end with step (equivalent to `start..end;step`)
    RangeStep {
        start: usize,
        end: usize,
        step: isize,
    },
}

impl SliceRange {
    /// Create a full range.
    pub fn full() -> Self {
        Self::Full
    }

    /// Create a single index.
    pub fn index(i: usize) -> Self {
        Self::Index(i)
    }

    /// Create a range from start to end.
    pub fn range(start: usize, end: usize) -> Self {
        Self::Range { start, end }
    }

    /// Create a range with step.
    pub fn range_step(start: usize, end: usize, step: isize) -> Self {
        Self::RangeStep { start, end, step }
    }
}

// endregion: SliceRange

// region: TensorView

/// A read-only view into a Tensor (doesn't own data).
///
/// Views provide zero-copy access to array subregions with potentially
/// different strides than the original array.
pub struct TensorView<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *const T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Get element at flat index (only valid for contiguous views).
    ///
    /// # Safety
    /// Caller must ensure the view is contiguous and index is in bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.data.add(index)
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
        } else {
            None
        }
    }
}

impl<'a, T: Clone, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Copy the view contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.len) };
            Tensor::try_from_slice(slice, self.shape())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = Tensor::try_full(self.shape(), unsafe { (*self.data).clone() })?;
            self.copy_to_contiguous(result.as_mut_slice());
            Ok(result)
        }
    }

    fn copy_to_contiguous(&self, dest: &mut [T]) {
        // For 2D case, optimize the copy
        if self.ndim == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let row_stride = self.strides[0];
            let col_stride = self.strides[1];
            let mut dest_idx = 0;
            for r in 0..rows {
                let row_ptr =
                    unsafe { (self.data as *const u8).offset(r as isize * row_stride) as *const T };
                for c in 0..cols {
                    let elem_ptr = unsafe {
                        (row_ptr as *const u8).offset(c as isize * col_stride) as *const T
                    };
                    dest[dest_idx] = unsafe { (*elem_ptr).clone() };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_idx in 0..self.len {
                // Compute pointer offset
                let mut offset = 0isize;
                for d in 0..self.ndim {
                    offset += indices[d] as isize * self.strides[d];
                }
                let elem_ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };
                dest[dest_idx] = unsafe { (*elem_ptr).clone() };

                // Increment indices (row-major order)
                for d in (0..self.ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < self.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }
    }
}

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

// region: TensorSpan

/// A mutable view into a Tensor.
pub struct TensorSpan<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *mut T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Returns a mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
        } else {
            None
        }
    }

    /// Convert to mutable slice (only valid for contiguous views).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.len) })
        } else {
            None
        }
    }

    /// Reborrow as immutable view.
    pub fn as_view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data,
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

// endregion: TensorSpan

// region: AxisIterator

/// Iterator over sub-tensor views along a given axis.
///
/// Each item is a `TensorView` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_views(0)` yields row views.
pub struct AxisIterator<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *const T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> Iterator for AxisIterator<'a, T, MAX_RANK> {
    type Item = TensorView<'a, T, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };

        // Build sub-shape/strides with the axis dimension removed
        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;
        let sub_len: usize = sub_shape[..sub_ndim].iter().product();

        self.current += 1;
        Some(TensorView {
            data: sub_ptr,
            len: sub_len,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const MAX_RANK: usize> ExactSizeIterator for AxisIterator<'a, T, MAX_RANK> {}

/// Mutable iterator over sub-tensor spans along a given axis.
///
/// Each item is a `TensorSpan` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_spans(0)` yields mutable row spans.
pub struct AxisIteratorMut<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *mut T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> Iterator for AxisIteratorMut<'a, T, MAX_RANK> {
    type Item = TensorSpan<'a, T, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *mut u8).offset(offset) as *mut T };

        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;
        let sub_len: usize = sub_shape[..sub_ndim].iter().product();

        self.current += 1;
        Some(TensorSpan {
            data: sub_ptr,
            len: sub_len,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const MAX_RANK: usize> ExactSizeIterator for AxisIteratorMut<'a, T, MAX_RANK> {}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<I: VecIndex>(
        &self,
        axis: I,
    ) -> Result<AxisIterator<'a, T, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIterator {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

impl<T: Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<I: VecIndex>(
        &self,
        axis: I,
    ) -> Result<AxisIterator<'_, T, MAX_RANK>, TensorError> {
        self.view().axis_views(axis)
    }

    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<I: VecIndex>(
        &mut self,
        axis: I,
    ) -> Result<AxisIteratorMut<'_, T, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIteratorMut {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

// endregion: AxisIterator

// region: Tensor View and Slice Methods

impl<T: Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Create a view of the entire array.
    pub fn view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of the entire tensor.
    pub fn span(&mut self) -> TensorSpan<'_, T, MAX_RANK> {
        TensorSpan {
            data: self.data.as_ptr(),
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Slice the array along multiple dimensions.
    ///
    /// # Arguments
    /// * `ranges` - Slice specification for each dimension. Length must match ndim.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, SliceRange};
    ///
    /// let arr = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
    ///
    /// // Get rows 0..2, all columns
    /// let view = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    ///
    /// // Get row 1 (reduces to 1D)
    /// let row = arr.slice(&[SliceRange::index(1), SliceRange::full()]).unwrap();
    /// ```
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if ranges.len() != self.ndim {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0isize;

        for (dim, range) in ranges.iter().enumerate() {
            let dim_size = self.shape[dim];
            let dim_stride = self.strides[dim];

            match *range {
                SliceRange::Full => {
                    new_shape[new_ndim] = dim_size;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                }
                SliceRange::Index(i) => {
                    if i >= dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    // Single index reduces dimension (doesn't add to new shape)
                    offset += i as isize * dim_stride;
                }
                SliceRange::Range { start, end } => {
                    if start > end || end > dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: end,
                            size: dim_size,
                        });
                    }
                    new_shape[new_ndim] = end - start;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                    offset += start as isize * dim_stride;
                }
                SliceRange::RangeStep { start, end, step } => {
                    if start >= dim_size || (end > dim_size && step > 0) {
                        return Err(TensorError::IndexOutOfBounds {
                            index: if start >= dim_size { start } else { end },
                            size: dim_size,
                        });
                    }
                    if step == 0 {
                        return Err(TensorError::InvalidShape {
                            shape: ShapeDescriptor::from_slice(self.shape()),
                            reason: "step cannot be zero",
                        });
                    }
                    let count = if step > 0 {
                        (end.saturating_sub(start) + (step as usize) - 1) / (step as usize)
                    } else {
                        let abs_step = (-step) as usize;
                        (start.saturating_sub(end) + abs_step - 1) / abs_step
                    };
                    new_shape[new_ndim] = count;
                    // Stride can be negative for reversed views
                    new_strides[new_ndim] = dim_stride * step;
                    new_ndim += 1;
                    offset += start as isize * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *const u8).offset(offset) as *const T };

        Ok(TensorView {
            data: new_ptr,
            len: new_len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the array mutably along multiple dimensions.
    ///
    /// Each range selects a sub-range along the corresponding axis. The resulting
    /// span has the same rank, with extents and strides adjusted per the ranges.
    /// Strided ranges produce non-contiguous spans.
    pub fn slice_mut(
        &mut self,
        ranges: &[SliceRange],
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        if ranges.len() != self.ndim {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0isize;

        for (dim, range) in ranges.iter().enumerate() {
            let dim_size = self.shape[dim];
            let dim_stride = self.strides[dim];

            match *range {
                SliceRange::Full => {
                    new_shape[new_ndim] = dim_size;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                }
                SliceRange::Index(i) => {
                    if i >= dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    offset += i as isize * dim_stride;
                }
                SliceRange::Range { start, end } => {
                    if start > end || end > dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: end,
                            size: dim_size,
                        });
                    }
                    new_shape[new_ndim] = end - start;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                    offset += start as isize * dim_stride;
                }
                SliceRange::RangeStep { start, end, step } => {
                    if start >= dim_size || (end > dim_size && step > 0) {
                        return Err(TensorError::IndexOutOfBounds {
                            index: if start >= dim_size { start } else { end },
                            size: dim_size,
                        });
                    }
                    if step == 0 {
                        return Err(TensorError::InvalidShape {
                            shape: ShapeDescriptor::from_slice(self.shape()),
                            reason: "step cannot be zero",
                        });
                    }
                    let count = if step > 0 {
                        (end.saturating_sub(start) + (step as usize) - 1) / (step as usize)
                    } else {
                        let abs_step = (-step) as usize;
                        (start.saturating_sub(end) + abs_step - 1) / abs_step
                    };
                    new_shape[new_ndim] = count;
                    new_strides[new_ndim] = dim_stride * step;
                    new_ndim += 1;
                    offset += start as isize * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut T };

        Ok(TensorSpan {
            data: new_ptr,
            len: new_len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        })
    }

    /// Transpose a 2D array (swaps strides, no data copy).
    pub fn t(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim,
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        new_shape[0] = self.shape[1];
        new_shape[1] = self.shape[0];
        new_strides[0] = self.strides[1];
        new_strides[1] = self.strides[0];

        Ok(TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: 2,
            _marker: PhantomData,
        })
    }

    /// Reshape the array (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if new_shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks {
                got: new_shape.len(),
            });
        }

        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(new_shape),
                got: ShapeDescriptor::from_slice(self.shape()),
            });
        }

        // Check if currently contiguous
        if !self.view().is_contiguous() {
            return Err(TensorError::NonContiguousRows);
        }

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Tensor::<T, Global, MAX_RANK>::compute_strides_into(new_shape, &mut strides_arr);

        Ok(TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }
}

// endregion: Tensor View and Slice Methods

// region: Type Aliases

/// Type alias for a 2D matrix (Tensor with MAX_RANK=2).
pub type Matrix<T, A = Global> = Tensor<T, A, 2>;

/// Type alias for an immutable 2D matrix view.
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;

/// Type alias for a mutable 2D matrix view.
pub type MatrixSpan<'a, T> = TensorSpan<'a, T, 2>;

// endregion: Type Aliases

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
    /// Output columns (B rows).
    n: usize,
    /// Inner dimension.
    k: usize,
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

impl<T: Dots, A: Allocator + Clone> Clone for PackedMatrix<T, A> {
    fn clone(&self) -> Self {
        if self.size == 0 {
            return Self {
                data: NonNull::dangling(),
                size: 0,
                n: self.n,
                k: self.k,
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            };
        }

        let layout = alloc::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
            .expect("invalid layout");
        let ptr = self
            .alloc
            .allocate(layout)
            .expect("clone allocation failed");
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size);
        }
        Self {
            data: ptr,
            size: self.size,
            n: self.n,
            k: self.k,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        }
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
        let (n, k) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(n, k);

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
                T::dots_pack(b.as_ptr(), n, k, b.stride_bytes(0) as usize, data.as_ptr());
            }
        }

        Ok(Self {
            data,
            size,
            n,
            k,
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
        let (k, n) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(n, k);

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
                T::dots_pack(b.as_ptr(), n, k, core::mem::size_of::<T>(), data.as_ptr());
            }
        }

        Ok(Self {
            data,
            size,
            n,
            k,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns dimensions (n, k) of the original B matrix.
    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
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

// Parallel dots_packed implementations, if Fork Union is available
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

    /// Computes symmetric angular distance matrix.
    pub fn try_angulars_symmetric(
        &self,
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
        unsafe {
            T::angulars_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n,
            );
        }
        Ok(result)
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

    /// Computes symmetric euclidean distance matrix.
    pub fn try_euclideans_symmetric(
        &self,
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
        unsafe {
            T::euclideans_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n,
            );
        }
        Ok(result)
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

    /// Computes symmetric Hamming distance matrix.
    pub fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (n, k) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::<u32, Global, MAX_RANK>::try_full(&[n, n], u32::default())?;
        unsafe {
            T::hammings_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n,
            );
        }
        Ok(result)
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

    /// Computes symmetric Jaccard distance matrix.
    pub fn try_jaccards_symmetric(
        &self,
    ) -> Result<Tensor<T::JaccardResult, Global, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (n, k) = (self.shape()[0], self.shape()[1]);
        let mut result = Tensor::<T::JaccardResult, Global, MAX_RANK>::try_full(
            &[n, n],
            T::JaccardResult::default(),
        )?;
        unsafe {
            T::jaccards_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride_bytes(0) as usize,
                result.as_mut_ptr(),
                result.stride_bytes(0) as usize,
                0,
                n,
            );
        }
        Ok(result)
    }
}

// endregion: Tensor Hammings/Jaccards

// region: Tensor Internal Helpers

#[inline]
fn validate_same_shape(lhs: &[usize], rhs: &[usize]) -> Result<(), TensorError> {
    if lhs == rhs {
        return Ok(());
    }
    Err(TensorError::ShapeMismatch {
        expected: ShapeDescriptor::from_slice(lhs),
        got: ShapeDescriptor::from_slice(rhs),
    })
}

#[inline]
fn normalize_axis<I: VecIndex>(axis: I, ndim: usize) -> Result<usize, TensorError> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds {
            index: 0,
            size: ndim,
        });
    }
    axis.resolve(ndim).ok_or(TensorError::IndexOutOfBounds {
        index: 0,
        size: ndim,
    })
}

fn reduced_shape(shape: &[usize], axis: usize, keep_dims: bool) -> Vec<usize> {
    let mut result = Vec::with_capacity(if keep_dims {
        shape.len()
    } else {
        shape.len() - 1
    });
    for (dim_index, &dim_size) in shape.iter().enumerate() {
        if dim_index == axis {
            if keep_dims {
                result.push(1);
            }
        } else {
            result.push(dim_size);
        }
    }
    result
}

fn shared_contiguous_tail_2(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_3(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_4(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
    fourth_strides: &[isize],
    fourth_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    let mut expected_fourth = fourth_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
            && fourth_strides[dim_index] == expected_fourth
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
            expected_fourth = expected_fourth.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

unsafe fn walk_contiguous_blocks_2<TIn, TOut, F>(
    source_ptr: *const TIn,
    source_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TIn, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_2(
        shape,
        source_strides,
        core::mem::size_of::<TIn>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TIn, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        source_ptr: *const u8,
        source_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TIn, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(source_ptr as *const TIn, target_ptr as *mut TOut, tail_len);
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let source_child = source_ptr.offset(offset_index as isize * source_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TIn, TOut, F>(
                dim_index + 1,
                outer_dims,
                source_child,
                source_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TIn, TOut, F>(
        0,
        outer_dims,
        source_ptr as *const u8,
        source_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_3<TFirst, TSecond, TOut, F>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_3(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TOut, F>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TOut, F>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_4<TFirst, TSecond, TThird, TOut, F>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    third_ptr: *const TThird,
    third_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_4(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        third_strides,
        core::mem::size_of::<TThird>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TThird, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        third_ptr: *const u8,
        third_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                third_ptr as *const TThird,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let third_child = third_ptr.offset(offset_index as isize * third_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TThird, TOut, F>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                third_child,
                third_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TThird, TOut, F>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        third_ptr as *const u8,
        third_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

fn for_each_axis_lane<T, const MAX_RANK: usize, F>(
    view: &TensorView<'_, T, MAX_RANK>,
    axis: usize,
    mut callback: F,
) where
    F: FnMut(*const T, usize, isize, usize),
{
    let lane_len = view.shape[axis];
    let lane_stride = view.strides[axis];
    let mut other_dims = [0usize; MAX_RANK];
    let mut other_ndim = 0usize;
    for dim_index in 0..view.ndim {
        if dim_index != axis {
            other_dims[other_ndim] = dim_index;
            other_ndim += 1;
        }
    }

    if other_ndim == 0 {
        callback(view.data, lane_len, lane_stride, 0);
        return;
    }

    let mut coords = [0usize; MAX_RANK];
    let total_lanes: usize = other_dims[..other_ndim]
        .iter()
        .map(|&dim_index| view.shape[dim_index])
        .product();

    for lane_index in 0..total_lanes {
        let mut lane_offset = 0isize;
        for idx in 0..other_ndim {
            let dim_index = other_dims[idx];
            lane_offset += coords[idx] as isize * view.strides[dim_index];
        }
        let lane_ptr = unsafe { (view.data as *const u8).offset(lane_offset) as *const T };
        callback(lane_ptr, lane_len, lane_stride, lane_index);

        for idx in (0..other_ndim).rev() {
            coords[idx] += 1;
            if coords[idx] < view.shape[other_dims[idx]] {
                break;
            }
            coords[idx] = 0;
        }
    }
}

unsafe fn normalize_reduction_lane<T>(
    lane_ptr: *const T,
    lane_len: usize,
    lane_stride: isize,
) -> (*const T, usize, usize, bool) {
    if lane_len == 0 {
        return (lane_ptr, 0, core::mem::size_of::<T>(), false);
    }
    if lane_stride >= 0 {
        return (lane_ptr, lane_len, lane_stride as usize, false);
    }
    let last_ptr =
        (lane_ptr as *const u8).offset((lane_len as isize - 1) * lane_stride) as *const T;
    (last_ptr, lane_len, (-lane_stride) as usize, true)
}

unsafe fn reduce_moments_recursive<T>(
    data: *const T,
    shape: &[usize],
    strides: &[isize],
) -> (T::SumOutput, T::SumSqOutput)
where
    T: ReduceMoments,
    T::SumOutput: Default + core::ops::AddAssign,
    T::SumSqOutput: Default + core::ops::AddAssign,
{
    if shape.is_empty() {
        return T::reduce_moments_raw(data, 1, core::mem::size_of::<T>());
    }
    if shape[0] == 0 {
        return (T::SumOutput::default(), T::SumSqOutput::default());
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, _) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return T::reduce_moments_raw(lane_ptr, lane_len, lane_stride);
    }

    let mut sum = T::SumOutput::default();
    let mut sumsq = T::SumSqOutput::default();
    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const T;
        let (child_sum, child_sumsq) =
            reduce_moments_recursive::<T>(child_ptr, &shape[1..], &strides[1..]);
        sum += child_sum;
        sumsq += child_sumsq;
    }
    (sum, sumsq)
}

unsafe fn reduce_minmax_recursive<T>(
    data: *const T,
    shape: &[usize],
    strides: &[isize],
    logical_offset: usize,
) -> Option<(T::Output, usize, T::Output, usize)>
where
    T: ReduceMinMax,
    T::Output: Clone + PartialOrd,
{
    if shape.is_empty() {
        return T::reduce_minmax_raw(data, 1, core::mem::size_of::<T>()).map(
            |(min_value, _, max_value, _)| (min_value, logical_offset, max_value, logical_offset),
        );
    }
    if shape[0] == 0 {
        return None;
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, reversed) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return T::reduce_minmax_raw(lane_ptr, lane_len, lane_stride).map(
            |(min_value, min_index, max_value, max_index)| {
                let min_index = if reversed {
                    lane_len - 1 - min_index
                } else {
                    min_index
                };
                let max_index = if reversed {
                    lane_len - 1 - max_index
                } else {
                    max_index
                };
                (
                    min_value,
                    logical_offset + min_index,
                    max_value,
                    logical_offset + max_index,
                )
            },
        );
    }

    let inner_len: usize = shape[1..].iter().product();
    let mut best_min: Option<(T::Output, usize)> = None;
    let mut best_max: Option<(T::Output, usize)> = None;

    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const T;
        let child_offset = logical_offset + index * inner_len;
        if let Some((child_min, child_min_index, child_max, child_max_index)) =
            reduce_minmax_recursive::<T>(child_ptr, &shape[1..], &strides[1..], child_offset)
        {
            match &best_min {
                Some((best_value, _))
                    if child_min.partial_cmp(best_value) != Some(core::cmp::Ordering::Less) => {}
                _ => best_min = Some((child_min, child_min_index)),
            }
            match &best_max {
                Some((best_value, _))
                    if child_max.partial_cmp(best_value) != Some(core::cmp::Ordering::Greater) => {}
                _ => best_max = Some((child_max, child_max_index)),
            }
        }
    }

    match (best_min, best_max) {
        (Some((min_value, min_index)), Some((max_value, max_index))) => {
            Some((min_value, min_index, max_value, max_index))
        }
        _ => None,
    }
}

#[doc(hidden)]
pub trait SumSqToF64 {
    fn to_f64(self) -> f64;
}

impl SumSqToF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl SumSqToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl SumSqToF64 for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl SumSqToF64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

fn try_alloc_output_like<D: Clone, F, const MAX_RANK: usize>(
    shape: &[usize],
    fill: F,
) -> Result<Tensor<D, Global, MAX_RANK>, TensorError>
where
    F: FnOnce(&mut TensorSpan<'_, D, MAX_RANK>) -> Result<(), TensorError>,
{
    let mut result = unsafe { Tensor::<D, Global, MAX_RANK>::try_empty(shape) }?;
    {
        let mut span = result.span();
        fill(&mut span)?;
    }
    Ok(result)
}

fn try_reborrow_tensor_into<T: Clone, D: Clone, F, const MAX_RANK: usize>(
    source: &Tensor<T, Global, MAX_RANK>,
    out: &mut Tensor<D, Global, MAX_RANK>,
    apply: F,
) -> Result<(), TensorError>
where
    F: FnOnce(
        &TensorView<'_, T, MAX_RANK>,
        &mut TensorSpan<'_, D, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = source.view();
    let mut span = out.span();
    apply(&view, &mut span)
}

fn try_reborrow_tensor_inplace<T: Clone, F, const MAX_RANK: usize>(
    tensor: &mut Tensor<T, Global, MAX_RANK>,
    apply: F,
) -> Result<(), TensorError>
where
    F: FnOnce(
        &TensorView<'_, T, MAX_RANK>,
        &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = TensorView {
        data: tensor.data.as_ptr(),
        len: tensor.len,
        shape: tensor.shape,
        strides: tensor.strides,
        ndim: tensor.ndim,
        _marker: PhantomData,
    };
    let mut span = tensor.span();
    apply(&view, &mut span)
}

fn rebind_view_rank<'a, T, const TARGET_MAX_RANK: usize, const SOURCE_MAX_RANK: usize>(
    view: &TensorView<'a, T, SOURCE_MAX_RANK>,
) -> Result<TensorView<'a, T, TARGET_MAX_RANK>, TensorError> {
    if view.ndim > TARGET_MAX_RANK {
        return Err(TensorError::DimensionMismatch {
            expected: TARGET_MAX_RANK,
            got: view.ndim,
        });
    }
    let mut shape = [0usize; TARGET_MAX_RANK];
    let mut strides = [0isize; TARGET_MAX_RANK];
    shape[..view.ndim].copy_from_slice(&view.shape[..view.ndim]);
    strides[..view.ndim].copy_from_slice(&view.strides[..view.ndim]);
    Ok(TensorView {
        data: view.data,
        len: view.len,
        shape,
        strides,
        ndim: view.ndim,
        _marker: PhantomData,
    })
}

fn try_unary_kernel_into<S, D, F, const MAX_RANK: usize>(
    source: &TensorView<'_, S, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[S], &mut [D]),
{
    validate_same_shape(source.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_2(
            source.data,
            &source.strides[..source.ndim],
            out.data,
            &out.strides[..out.ndim],
            source.shape(),
            |source_ptr, target_ptr, tail_len| {
                let source = core::slice::from_raw_parts(source_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(source, target);
            },
        );
    }
    Ok(())
}

fn try_binary_kernel_into<A, B, D, F, const MAX_RANK: usize>(
    first: &TensorView<'_, A, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[A], &[B], &mut [D]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_3(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, target);
            },
        );
    }
    Ok(())
}

fn try_ternary_kernel_into<A, B, C, D, F, const MAX_RANK: usize>(
    first: &TensorView<'_, A, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    third: &TensorView<'_, C, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[A], &[B], &[C], &mut [D]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), third.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_4(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            third.data,
            &third.strides[..third.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, third_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let third = core::slice::from_raw_parts(third_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, third, target);
            },
        );
    }
    Ok(())
}

// endregion: Tensor Internal Helpers

// region: Tensor Elementwise Operations

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    /// Apply element-wise scale: result\[i\] = α × self\[i\] + β
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_scale_tensor(alpha, beta)
    }

    /// Apply element-wise scale in-place: self\[i\] = α × self\[i\] + β
    pub fn scale_inplace(&mut self, alpha: T::Scalar, beta: T::Scalar) {
        let _ = try_reborrow_tensor_inplace(self, |view, span| {
            view.try_scale_tensor_into(alpha, beta, span)
        });
    }
}

impl<T: Clone + EachSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sum: result\[i\] = self\[i\] + other\[i\]
    ///
    /// Returns a new array with the summed values.
    pub fn add<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_add_tensor(&other_view)
    }

    /// Element-wise sum in-place: self\[i\] = self\[i\] + other\[i\]
    pub fn add_inplace<const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<(), TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other_view, span)
        })
    }
}

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    /// Blend: result\[i\] = α × self\[i\] + β × other\[i\]
    ///
    /// Returns a new array with the blend.
    pub fn blend<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_blend_tensor(&other_view, alpha, beta)
    }
}

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    /// Fused multiply-add: result\[i\] = α × self\[i\] × b\[i\] + β × c\[i\]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma<const B_MAX_RANK: usize, const C_MAX_RANK: usize>(
        &self,
        b: &Tensor<T, Global, B_MAX_RANK>,
        c: &Tensor<T, Global, C_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), b.shape())?;
        validate_same_shape(self.shape(), c.shape())?;
        let b_view = rebind_view_rank::<T, MAX_RANK, B_MAX_RANK>(&b.view())?;
        let c_view = rebind_view_rank::<T, MAX_RANK, C_MAX_RANK>(&c.view())?;
        self.view().try_fma_tensors(&b_view, &c_view, alpha, beta)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Explicit Elementwise + Cast

impl<'a, T: Clone + EachScale, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    pub fn try_scale_tensor(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_scale_tensor_into(alpha, beta, span)
        })
    }

    pub fn try_scale_tensor_into(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(alpha, beta, out)
    }

    fn try_affine_into(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::each_scale(source, alpha, beta, target);
        })
    }

    pub fn try_add_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_scalar_into(scalar, span))
    }

    pub fn try_sub_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_scalar_into(scalar, span))
    }

    pub fn try_mul_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_scalar_into(scalar, span))
    }

    pub fn try_add_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(T::Scalar::from(1.0f32), scalar, out)
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(
            T::Scalar::from(1.0f32),
            T::Scalar::from(-1.0f32) * scalar,
            out,
        )
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(scalar, T::Scalar::from(0.0f32), out)
    }
}

impl<'a, T: Clone + EachSum, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_add_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_tensor_into(other, span))
    }

    pub fn try_add_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            T::each_sum(first, second, target);
        })
    }
}

impl<'a, T: Clone + EachBlend, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_blend_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_blend_tensor_into(other, alpha, beta, span)
        })
    }

    pub fn try_blend_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            T::each_blend(first, second, alpha, beta, target);
        })
    }

    pub fn try_sub_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_tensor_into(other, span))
    }

    pub fn try_sub_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_blend_tensor_into(
            other,
            T::Scalar::from(1.0f32),
            T::Scalar::from(-1.0f32),
            out,
        )
    }
}

impl<'a, T: Clone + EachFMA, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_fma_tensors(
        &self,
        b: &TensorView<'_, T, MAX_RANK>,
        c: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_fma_tensors_into(b, c, alpha, beta, span)
        })
    }

    pub fn try_fma_tensors_into(
        &self,
        b: &TensorView<'_, T, MAX_RANK>,
        c: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_ternary_kernel_into(self, b, c, out, |first, second, third, target| {
            T::each_fma(first, second, third, alpha, beta, target);
        })
    }

    pub fn try_mul_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_tensor_into(other, span))
    }

    pub fn try_mul_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_fma_tensors_into(
            other,
            self,
            T::Scalar::from(1.0f32),
            T::Scalar::from(0.0f32),
            out,
        )
    }
}

impl<'a, S: Clone + CastDtype, const MAX_RANK: usize> TensorView<'a, S, MAX_RANK> {
    pub fn try_cast_dtype<D: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<D, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cast_dtype_into(span))
    }

    pub fn try_cast_dtype_into<D: Clone + CastDtype>(
        &self,
        out: &mut TensorSpan<'_, D, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            let _ = cast(source, target);
        })
    }
}

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    pub fn try_add_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_scalar(scalar)
    }

    pub fn try_sub_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_scalar(scalar)
    }

    pub fn try_mul_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_scalar(scalar)
    }

    pub fn try_add_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_scalar_into(scalar, span)
        })
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_scalar_into(scalar, span)
        })
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_scalar_into(scalar, span)
        })
    }

    pub fn try_add_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_add_scalar_into(scalar, span))
    }

    pub fn try_sub_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sub_scalar_into(scalar, span))
    }

    pub fn try_mul_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_mul_scalar_into(scalar, span))
    }
}

impl<T: Clone + EachSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    pub fn try_add_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_tensor(&other.view())
    }

    pub fn try_add_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }

    pub fn try_add_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }
}

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_sub_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_tensor(&other.view())
    }

    pub fn try_sub_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }

    pub fn try_sub_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }
}

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_mul_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_tensor(&other.view())
    }

    pub fn try_mul_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }

    pub fn try_mul_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }
}

impl<S: Clone + CastDtype, const MAX_RANK: usize> Tensor<S, Global, MAX_RANK> {
    pub fn try_cast_dtype<D: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<D, Global, MAX_RANK>, TensorError> {
        self.view().try_cast_dtype()
    }

    pub fn try_cast_dtype_into<D: Clone + CastDtype>(
        &self,
        out: &mut Tensor<D, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| view.try_cast_dtype_into(span))
    }
}

// endregion: Tensor Explicit Elementwise + Cast

// region: Tensor Trigonometry

impl<'a, T: Clone + EachSin, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sin_into(span))
    }

    pub fn try_sin_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::sin(source, target);
        })
    }
}

impl<'a, T: Clone + EachCos, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cos_into(span))
    }

    pub fn try_cos_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::cos(source, target);
        })
    }
}

impl<'a, T: Clone + EachATan, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_atan_into(span))
    }

    pub fn try_atan_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::atan(source, target);
        })
    }
}

impl<T: Clone + EachSin, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sine: result\[i\] = sin(self\[i\])
    ///
    /// Input values are in radians.
    pub fn sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sin()
    }

    pub fn try_sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sin()
    }

    /// Element-wise sine in-place: self\[i\] = sin(self\[i\])
    pub fn sin_inplace(&mut self) {
        let _ = self.try_sin_inplace();
    }

    pub fn try_sin_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sin_into(span))
    }
}

impl<T: Clone + EachCos, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise cosine: result\[i\] = cos(self\[i\])
    ///
    /// Input values are in radians.
    pub fn cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_cos()
    }

    pub fn try_cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_cos()
    }

    /// Element-wise cosine in-place: self\[i\] = cos(self\[i\])
    pub fn cos_inplace(&mut self) {
        let _ = self.try_cos_inplace();
    }

    pub fn try_cos_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_cos_into(span))
    }
}

impl<T: Clone + EachATan, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise arctangent: result\[i\] = atan(self\[i\])
    ///
    /// Output values are in radians in the range (-π/2, π/2).
    pub fn atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    pub fn try_atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    /// Element-wise arctangent in-place: self\[i\] = atan(self\[i\])
    pub fn atan_inplace(&mut self) {
        let _ = self.try_atan_inplace();
    }

    pub fn try_atan_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_atan_into(span))
    }
}

// endregion: Tensor Trigonometry

// region: Tensor Reductions

impl<T: Clone + Dot, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Compute the dot product of this array with another.
    ///
    /// Both arrays must be 1D with the same length.
    pub fn dot_product<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<T::Output, TensorError> {
        if self.ndim != 1 || other.ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: if self.ndim != 1 {
                    self.ndim
                } else {
                    other.ndim
                },
            });
        }
        if self.len != other.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        // Dot::dot returns Option, unwrap since we verified lengths match
        Ok(T::dot(self.as_slice(), other.as_slice()).expect("dot product failed"))
    }
}

impl<'a, T: Clone + ReduceMoments, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        Ok(unsafe {
            reduce_moments_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim])
        })
    }

    pub fn try_moments_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::SumOutput, Global, MAX_RANK>,
            Tensor<T::SumSqOutput, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut sums = Tensor::<T::SumOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            T::SumOutput::default(),
        )?;
        let mut sumsqs = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut sums, &mut sumsqs)?;
        Ok((sums, sumsqs))
    }

    pub fn try_moments_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        sum_out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<T::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, sum_out.shape())?;
        validate_same_shape(&expected_shape, sumsq_out.shape())?;

        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                let (lane_ptr, lane_len, lane_stride, _) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                let (sum, sumsq) =
                    unsafe { T::reduce_moments_raw(lane_ptr, lane_len, lane_stride) };
                sum_out.as_mut_slice()[output_index] = sum;
                sumsq_out.as_mut_slice()[output_index] = sumsq;
            },
        );
        Ok(())
    }

    pub fn try_sum_all(&self) -> Result<T::SumOutput, TensorError> {
        Ok(self.try_moments_all()?.0)
    }

    pub fn try_sum_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        let (sums, _) = self.try_moments_axis(axis, keep_dims)?;
        Ok(sums)
    }

    pub fn try_sum_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, out, &mut scratch)
    }

    pub fn try_norm_all(&self) -> Result<f64, TensorError> {
        let (_, sumsq) = self.try_moments_all()?;
        Ok(Roots::sqrt(sumsq.to_f64()))
    }

    pub fn try_norm_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        let (_, sumsqs) = self.try_moments_axis(axis, keep_dims)?;
        let mut norms = Tensor::<f64, Global, MAX_RANK>::try_full(sumsqs.shape(), 0.0)?;
        for (target, value) in norms
            .as_mut_slice()
            .iter_mut()
            .zip(sumsqs.as_slice().iter())
        {
            *target = Roots::sqrt(value.clone().to_f64());
        }
        Ok(norms)
    }

    pub fn try_norm_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch_sum = Tensor::<T::SumOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumOutput::default(),
        )?;
        let mut scratch_sumsq = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut scratch_sum, &mut scratch_sumsq)?;
        for (target, value) in out
            .as_mut_slice()
            .iter_mut()
            .zip(scratch_sumsq.as_slice().iter())
        {
            *target = Roots::sqrt(value.clone().to_f64());
        }
        Ok(())
    }
}

impl<'a, T: Clone + ReduceMinMax, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<(T::Output, usize, T::Output, usize), TensorError> {
        unsafe {
            reduce_minmax_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim], 0)
        }
        .ok_or(TensorError::InvalidShape {
            shape: ShapeDescriptor::from_slice(self.shape()),
            reason: "min/max reduction undefined for empty or NaN-only input",
        })
    }

    pub fn try_minmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut min_values =
            Tensor::<T::Output, Global, MAX_RANK>::try_full(&output_shape, T::Output::default())?;
        let mut min_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        let mut max_values =
            Tensor::<T::Output, Global, MAX_RANK>::try_full(&output_shape, T::Output::default())?;
        let mut max_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        self.try_minmax_axis_into(
            axis,
            keep_dims,
            &mut min_values,
            &mut min_indices,
            &mut max_values,
            &mut max_indices,
        )?;
        Ok((min_values, min_indices, max_values, max_indices))
    }

    pub fn try_minmax_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        min_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmin_out: &mut Tensor<usize, Global, MAX_RANK>,
        max_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmax_out: &mut Tensor<usize, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, min_out.shape())?;
        validate_same_shape(&expected_shape, argmin_out.shape())?;
        validate_same_shape(&expected_shape, max_out.shape())?;
        validate_same_shape(&expected_shape, argmax_out.shape())?;

        let mut invalid_lane = false;
        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                if invalid_lane {
                    return;
                }
                let (lane_ptr, lane_len, lane_stride, reversed) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                if let Some((min_value, min_index, max_value, max_index)) =
                    unsafe { T::reduce_minmax_raw(lane_ptr, lane_len, lane_stride) }
                {
                    min_out.as_mut_slice()[output_index] = min_value;
                    argmin_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - min_index
                    } else {
                        min_index
                    };
                    max_out.as_mut_slice()[output_index] = max_value;
                    argmax_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - max_index
                    } else {
                        max_index
                    };
                } else {
                    invalid_lane = true;
                }
            },
        );

        if invalid_lane {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(self.shape()),
                reason: "min/max reduction undefined for empty or NaN-only lanes",
            });
        }
        Ok(())
    }

    pub fn try_min_all(&self) -> Result<T::Output, TensorError> {
        Ok(self.try_minmax_all()?.0)
    }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.1)
    }

    pub fn try_max_all(&self) -> Result<T::Output, TensorError> {
        Ok(self.try_minmax_all()?.2)
    }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.3)
    }

    pub fn try_min_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        let (min_values, _, _, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(min_values)
    }

    pub fn try_argmin_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        let (_, argmin_values, _, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(argmin_values)
    }

    pub fn try_max_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        let (_, _, max_values, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(max_values)
    }

    pub fn try_argmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        let (_, _, _, argmax_values) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(argmax_values)
    }
}

impl<T: Clone + ReduceMoments, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        self.view().try_moments_all()
    }

    pub fn try_moments_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::SumOutput, Global, MAX_RANK>,
            Tensor<T::SumSqOutput, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        self.view().try_moments_axis(axis, keep_dims)
    }

    pub fn try_moments_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        sum_out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<T::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view()
            .try_moments_axis_into(axis, keep_dims, sum_out, sumsq_out)
    }

    pub fn try_sum_all(&self) -> Result<T::SumOutput, TensorError> {
        self.view().try_sum_all()
    }

    pub fn try_sum_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        self.view().try_sum_axis(axis, keep_dims)
    }

    pub fn try_sum_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_sum_axis_into(axis, keep_dims, out)
    }

    pub fn try_norm_all(&self) -> Result<f64, TensorError> {
        self.view().try_norm_all()
    }

    pub fn try_norm_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        self.view().try_norm_axis(axis, keep_dims)
    }

    pub fn try_norm_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_norm_axis_into(axis, keep_dims, out)
    }
}

impl<T: Clone + ReduceMinMax, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<(T::Output, usize, T::Output, usize), TensorError> {
        self.view().try_minmax_all()
    }

    pub fn try_minmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        self.view().try_minmax_axis(axis, keep_dims)
    }

    pub fn try_minmax_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        min_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmin_out: &mut Tensor<usize, Global, MAX_RANK>,
        max_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmax_out: &mut Tensor<usize, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view()
            .try_minmax_axis_into(axis, keep_dims, min_out, argmin_out, max_out, argmax_out)
    }

    pub fn try_min_all(&self) -> Result<T::Output, TensorError> {
        self.view().try_min_all()
    }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> {
        self.view().try_argmin_all()
    }

    pub fn try_max_all(&self) -> Result<T::Output, TensorError> {
        self.view().try_max_all()
    }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> {
        self.view().try_argmax_all()
    }

    pub fn try_min_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_min_axis(axis, keep_dims)
    }

    pub fn try_argmin_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmin_axis(axis, keep_dims)
    }

    pub fn try_max_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_max_axis(axis, keep_dims)
    }

    pub fn try_argmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmax_axis(axis, keep_dims)
    }
}

impl<const MAX_RANK: usize> Tensor<f32, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f32 {
        self.try_sum_all().unwrap_or(0.0) as f32
    }
}

impl<const MAX_RANK: usize> Tensor<f64, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f64 {
        self.try_sum_all().unwrap_or(0.0)
    }
}

// endregion: Tensor Reductions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{bf16c, f16c, f32c, NumberLike};
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Initialize thread for AMX and other SIMD features.
    /// Safe to call multiple times - only executes once.
    fn init_thread() {
        INIT.call_once(|| {
            crate::capabilities::configure_thread();
            let caps = crate::capabilities::available();
            eprintln!("NumKong Rust Test Suite v{}", env!("CARGO_PKG_VERSION"));
            eprintln!(
                "  Dynamic dispatch: {}",
                crate::capabilities::uses_dynamic_dispatch()
            );
            eprintln!("ISA:");
            let isas: &[(&str, u64)] = &[
                // x86
                ("Haswell", crate::cap::HASWELL),
                ("Skylake", crate::cap::SKYLAKE),
                ("Ice Lake", crate::cap::ICELAKE),
                ("Genoa", crate::cap::GENOA),
                ("Sapphire", crate::cap::SAPPHIRE),
                ("Sapphire AMX", crate::cap::SAPPHIREAMX),
                ("Granite AMX", crate::cap::GRANITEAMX),
                ("Turin", crate::cap::TURIN),
                ("Sierra", crate::cap::SIERRA),
                // Arm
                ("NEON", crate::cap::NEON),
                ("NEON F16", crate::cap::NEONHALF),
                ("NEON BF16", crate::cap::NEONBFDOT),
                ("NEON I8", crate::cap::NEONSDOT),
                ("NEON FHM", crate::cap::NEONFHM),
                ("SVE", crate::cap::SVE),
                ("SVE F16", crate::cap::SVEHALF),
                ("SVE BF16", crate::cap::SVEBFDOT),
                ("SVE I8", crate::cap::SVESDOT),
                ("SVE2", crate::cap::SVE2),
                ("SVE2P1", crate::cap::SVE2P1),
                ("SME", crate::cap::SME),
                ("SME2", crate::cap::SME2),
                ("SME2P1", crate::cap::SME2P1),
                ("SME F64", crate::cap::SMEF64),
                ("SME F16", crate::cap::SMEHALF),
                ("SME BF16", crate::cap::SMEBF16),
                ("SME FA64", crate::cap::SMEFA64),
                ("SME LUT2", crate::cap::SMELUT2),
                // RISC-V
                ("RVV", crate::cap::RVV),
                ("RVV HALF", crate::cap::RVVHALF),
                ("RVV BF16", crate::cap::RVVBF16),
                ("RVV BB", crate::cap::RVVBB),
                // WASM
                ("V128 Relaxed", crate::cap::V128RELAXED),
            ];
            for &(name, cap) in isas {
                let indicator = if caps & cap != 0 {
                    "\u{25CF}"
                } else {
                    "\u{25CB}"
                };
                eprintln!("- {} {}", name, indicator);
            }
            eprintln!();
        });
    }

    use crate::scalar::{FloatLike, TestableType};

    // Dimension combos for generic tensor tests: (m, n, k).
    const DIMS: &[(usize, usize, usize)] =
        &[(1, 1, 1), (1, 8, 3), (3, 1, 7), (7, 5, 3), (33, 17, 65)];

    /// Sweeps DIMS, packs B, checks shape + all elements against expected = k.
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

    /// Same but with B given as [k, n] and packed via try_pack_transposed.
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

    /// For all-ones vectors: angular distance = 0.
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

    /// For all-ones vectors: euclidean distance = 0.
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
    fn tensor_construction() {
        // Creation
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.numel(), 12);
        assert!(!arr.is_empty());

        // From slice
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);

        // Clone
        let arr = Tensor::<f32>::try_full(&[3, 4], 2.5f32).unwrap();
        let cloned = arr.clone();
        assert_eq!(cloned.shape(), arr.shape());
        assert_eq!(cloned.as_slice(), arr.as_slice());

        // Error display
        let err = TensorError::AllocationFailed;
        assert_eq!(format!("{}", err), "memory allocation failed");
        let err = TensorError::TooManyRanks { got: 10 };
        assert_eq!(format!("{}", err), "too many ranks: 10");
    }

    #[test]
    fn tensor_views() {
        // Row access
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.row(0), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(arr.row(1), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(arr.row(2), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        assert_eq!(arr.row(3), None);

        // Slicing
        let arr = Tensor::<f32>::try_full(&[4, 5], 1.0f32).unwrap();
        let view = arr
            .slice(&[SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[4, 5]);
        let view = arr
            .slice(&[SliceRange::range(1, 3), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[2, 5]);
        let view = arr
            .slice(&[SliceRange::index(0), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[5]);
        assert_eq!(view.ndim(), 1);

        // Transpose
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.t().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);

        // Contiguous check
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());

        // Matrix alias
        let mat: Matrix<f32> = Matrix::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_ops() {
        // Reshape
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.numel(), 12);

        // Sum f32
        let arr = Tensor::<f32>::try_full(&[100], 1.0f32).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.001);

        // Sum f64
        let arr = Tensor::<f64>::try_full(&[100], 1.0f64).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 1e-9);
    }

    #[test]
    fn elementwise_and_cast() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let b = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();

        let a_even = a
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let b_even = b
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let added = a_even.try_add_tensor(&b_even).unwrap();
        assert_eq!(added.shape(), &[3, 2]);
        assert_eq!(added.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        let scaled = a_even.try_mul_scalar(0.5).unwrap();
        assert_eq!(scaled.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        let casted = a_even.try_cast_dtype::<f64>().unwrap();
        assert_eq!(casted.shape(), &[3, 2]);
        assert_eq!(casted.as_slice(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        let complex = a_even.try_cast_dtype::<f32c>().unwrap();
        assert_eq!(complex.shape(), &[3, 2]);
        assert_eq!(complex.as_slice()[0], f32c::from_real_imag(0.0, 0.0));
        assert_eq!(complex.as_slice()[5], f32c::from_real_imag(10.0, 0.0));

        let mut out = Tensor::<f32>::try_full(&[3, 4], 0.0).unwrap();
        a.try_add_tensor_into(&b, &mut out).unwrap();
        assert_eq!(out.as_slice()[0], 2.0);
        assert_eq!(out.as_slice()[11], 13.0);

        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.try_add_scalar_inplace(1.0).unwrap();
        assert_eq!(inplace.as_slice()[0], 1.0);
        assert_eq!(inplace.as_slice()[11], 12.0);

        let mut trig_out = Tensor::<f32>::try_full(&[3, 2], 0.0).unwrap();
        {
            let mut span = trig_out.span();
            a_even.try_sin_into(&mut span).unwrap();
        }
        assert_eq!(trig_out.shape(), &[3, 2]);
        assert!((trig_out.as_slice()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn reductions_axis_and_strided_views() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let a_even = a
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let sum_all = a_even.try_sum_all().unwrap();
        assert!((sum_all - 30.0).abs() < 1e-6);

        let norm_all = a_even.try_norm_all().unwrap();
        assert!((norm_all - 14.832396974191326).abs() < 1e-9);

        let (sum_axis0, sumsq_axis0) = a_even.try_moments_axis(0, false).unwrap();
        assert_eq!(sum_axis0.shape(), &[2]);
        assert!((sum_axis0.as_slice()[0] - 12.0).abs() < 1e-6);
        assert!((sum_axis0.as_slice()[1] - 18.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[0] - 80.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[1] - 140.0).abs() < 1e-6);

        let sum_axis1_keep = a_even.try_sum_axis(-1_i32, true).unwrap();
        assert_eq!(sum_axis1_keep.shape(), &[3, 1]);
        assert!((sum_axis1_keep.as_slice()[0] - 2.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[1] - 10.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[2] - 18.0).abs() < 1e-6);

        let (min_axis0, argmin_axis0, max_axis0, argmax_axis0) =
            a_even.try_minmax_axis(0, false).unwrap();
        assert_eq!(min_axis0.as_slice(), &[0.0, 2.0]);
        assert_eq!(max_axis0.as_slice(), &[8.0, 10.0]);
        assert_eq!(argmin_axis0.as_slice(), &[0, 0]);
        assert_eq!(argmax_axis0.as_slice(), &[2, 2]);

        let reversed = a
            .slice(&[SliceRange::full(), SliceRange::range_step(3, 0, -1)])
            .unwrap();
        let reversed_sum = reversed.try_sum_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_sum.shape(), &[3]);
        assert_eq!(reversed_sum.as_slice(), &[6.0, 18.0, 30.0]);

        let reversed_argmin = reversed.try_argmin_axis(-1_i32, false).unwrap();
        let reversed_argmax = reversed.try_argmax_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_argmin.as_slice(), &[2, 2, 2]);
        assert_eq!(reversed_argmax.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn complex_elementwise_view_and_owner_paths() {
        let a_values = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
        let b_values = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];
        let zeros = Tensor::<f32c>::try_full(&[2], f32c { re: 0.0, im: 0.0 }).unwrap();
        let a = Tensor::<f32c>::try_from_slice(&a_values, &[2]).unwrap();
        let b = Tensor::<f32c>::try_from_slice(&b_values, &[2]).unwrap();

        let added = a.try_add_tensor(&b).unwrap();
        assert_eq!(
            added.as_slice(),
            &[f32c { re: 6.0, im: 8.0 }, f32c { re: 10.0, im: 12.0 }]
        );

        let scaled = a
            .scale(f32c { re: 1.0, im: 0.0 }, f32c { re: 1.0, im: 0.0 })
            .unwrap();
        assert_eq!(
            scaled.as_slice(),
            &[f32c { re: 2.0, im: 2.0 }, f32c { re: 4.0, im: 4.0 }]
        );

        let blended = a
            .view()
            .try_blend_tensor(
                &b.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: -1.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            blended.as_slice(),
            &[f32c { re: -4.0, im: -4.0 }, f32c { re: -4.0, im: -4.0 }]
        );

        let fma = a
            .view()
            .try_fma_tensors(
                &b.view(),
                &zeros.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: 0.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            fma.as_slice(),
            &[
                f32c { re: -7.0, im: 16.0 },
                f32c {
                    re: -11.0,
                    im: 52.0
                }
            ]
        );

        let mut inplace = Tensor::<f32c>::try_from_slice(&a_values, &[2]).unwrap();
        inplace.try_add_tensor_inplace(&b).unwrap();
        assert_eq!(inplace.as_slice(), added.as_slice());

        let widened = a.try_cast_dtype::<bf16c>().unwrap();
        assert_eq!(widened.as_slice()[0].re.to_f32(), 1.0);
        assert_eq!(widened.as_slice()[0].im.to_f32(), 2.0);

        let strided = Tensor::<f16c>::try_from_slice(
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(2.0),
                },
                f16c {
                    re: f16::from_f32(100.0),
                    im: f16::from_f32(101.0),
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(4.0),
                },
                f16c {
                    re: f16::from_f32(102.0),
                    im: f16::from_f32(103.0),
                },
            ],
            &[2, 2],
        )
        .unwrap();
        let complex_column = strided
            .slice(&[SliceRange::full(), SliceRange::range(0, 1)])
            .unwrap();
        let mut out = Tensor::<f16c>::try_full(
            &[2, 1],
            f16c {
                re: f16::ZERO,
                im: f16::ZERO,
            },
        )
        .unwrap();
        {
            let mut span = out.span();
            complex_column
                .try_scale_tensor_into(
                    f16c {
                        re: f16::ONE,
                        im: f16::ZERO,
                    },
                    f16c {
                        re: f16::ZERO,
                        im: f16::ONE,
                    },
                    &mut span,
                )
                .unwrap();
        }
        assert_eq!(
            out.as_slice(),
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(3.0)
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(5.0)
                }
            ]
        );
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
    /// Sweeps DIMS, checks shape + all elements of symmetric dot gram matrix.
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
            // All-ones vectors: every upper-triangle entry = self-dot = expected.
            // The C implementation only populates the upper triangle (j >= i).
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

    /// For all-ones vectors: angular distance = 0 (symmetric).
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
            // The C implementation only populates the upper triangle (j >= i).
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

    /// For all-ones vectors: euclidean distance = 0 (symmetric).
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
            // The C implementation only populates the upper triangle (j >= i).
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

        // Dots
        let b_packed = PackedMatrix::try_pack(&b).unwrap();
        let c = a.dots_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 64); // All bits set → popcount of AND = 64

        // Hammings
        let c = a.hammings_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 0); // All bits equal → Hamming distance = 0

        // Jaccards
        let c = a.jaccards_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert!(c.as_slice()[0].abs() < 1e-5); // All bits equal → Jaccard distance = 0
    }

    #[test]
    fn binary_symmetric_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 8], u1x8(0xFF)).unwrap();

        // Dots symmetric
        let gram = a.view().try_dots_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert_eq!(gram.as_slice()[0], 64); // Self-dot = popcount of all bits = 64

        // Hammings symmetric
        let gram = a.try_hammings_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert_eq!(gram.as_slice()[0], 0); // All vectors identical → Hamming distance = 0

        // Jaccards symmetric
        let gram = a.try_jaccards_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert!(gram.as_slice()[0].abs() < 1e-5); // All vectors identical → Jaccard distance = 0
    }
}

// endregion: Tests
