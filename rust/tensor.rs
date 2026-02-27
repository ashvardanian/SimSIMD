//! N-dimensional tensor types, GEMM, and spatial distance operations.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorViewMut`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//! - [`TransposedMatrixMultiplier`]: Pre-packed matrix for efficient GEMM
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
//! use numkong::{Tensor, TransposedMatrixMultiplier};
//!
//! let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
//! let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
//!
//! // Pack B once, multiply many times
//! let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
//! let c = a.dots_packed(&b_packed);  // Returns (1024 × 256)
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::numerics::{Dot, EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum};
use crate::scalars::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};

/// Size type used in C FFI to match `nk_size_t` which is always `uint64_t`.
type u64size = u64;

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_packed_size_f32(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f32(b: *const f32, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_f64(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f64(b: *const f64, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_f16(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f16(b: *const u16, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_bf16(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_bf16(b: *const u16, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_i8(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_i8(b: *const i8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_u8(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_u8(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e4m3(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e4m3(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e5m2(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e5m2(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e2m3(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e2m3(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e3m2(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e3m2(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_u4(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_u4(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_i4(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_i4(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut i32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // Symmetric Gram matrix (C = A × Aᵀ)
    fn nk_dots_symmetric_f32(
        vectors: *const f32,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_f64(
        vectors: *const f64,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f64,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_f16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_bf16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_i8(
        vectors: *const i8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut i32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_u8(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut u32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_u4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut u32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_dots_symmetric_i4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut i32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );

    fn nk_dots_packed_size_u1(n: u64size, d: u64size) -> u64size;
    fn nk_dots_pack_u1(q: *const u8, n: u64size, d: u64size, q_stride: u64size, q_packed: *mut u8);
    fn nk_dots_packed_u1(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_dots_symmetric_u1(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut u32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_hammings_packed_u1(
        a: *const u8,
        q_packed: *const u8,
        result: *mut u32,
        rows: u64size,
        cols: u64size,
        d: u64size,
        v_stride: u64size,
        r_stride: u64size,
    );
    fn nk_hammings_symmetric_u1(
        vectors: *const u8,
        n_vectors: u64size,
        d: u64size,
        stride: u64size,
        result: *mut u32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );

    fn nk_jaccards_packed_u1(
        v: *const u8,
        q_packed: *const u8,
        result: *mut f32,
        rows: u64size,
        cols: u64size,
        d: u64size,
        v_stride: u64size,
        r_stride: u64size,
    );
    fn nk_jaccards_symmetric_u1(
        vectors: *const u8,
        n_vectors: u64size,
        d: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );

    // Batched angular distances
    fn nk_angulars_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_f32(
        vectors: *const f32,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_f64(
        vectors: *const f64,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f64,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_f16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_bf16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_i8(
        vectors: *const i8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_u8(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_i4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_angulars_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_angulars_symmetric_u4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );

    // Batched euclidean distances
    fn nk_euclideans_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_f32(
        vectors: *const f32,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_f64(
        vectors: *const f64,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f64,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_f16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_bf16(
        vectors: *const u16,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_i8(
        vectors: *const i8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_u8(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_i4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
    );
    fn nk_euclideans_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
    fn nk_euclideans_symmetric_u4(
        vectors: *const u8,
        n_vectors: u64size,
        depth: u64size,
        stride: u64size,
        result: *mut f32,
        result_stride: u64size,
        row_start: u64size,
        row_count: u64size,
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
/// [`Tensor`] and [`TransposedMatrixMultiplier`].
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
        unsafe { nk_dots_packed_size_f32(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f32(b, n as u64size, k as u64size, b_stride as u64size, packed)
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
        nk_dots_packed_f32(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
        nk_dots_symmetric_f32(
            vectors,
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for f64 {
    type Accumulator = f64;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f64(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f64(b, n as u64size, k as u64size, b_stride as u64size, packed)
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
        nk_dots_packed_f64(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
        nk_dots_symmetric_f64(
            vectors,
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for f16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f16(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f16(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_f16(
            a as *const u16,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for bf16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_bf16(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_bf16(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_bf16(
            a as *const u16,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for i8 {
    type Accumulator = i32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_i8(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i8(b, n as u64size, k as u64size, b_stride as u64size, packed)
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
        nk_dots_packed_i8(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
        nk_dots_symmetric_i8(
            vectors,
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for u8 {
    type Accumulator = u32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u8(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u8(b, n as u64size, k as u64size, b_stride as u64size, packed)
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
        nk_dots_packed_u8(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
        nk_dots_symmetric_u8(
            vectors,
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for e4m3 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e4m3(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e4m3(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_e4m3(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for e5m2 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e5m2(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e5m2(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_e5m2(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for e2m3 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e2m3(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e2m3(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_e2m3(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for e3m2 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e3m2(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e3m2(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_e3m2(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            depth as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for u4x2 {
    type Accumulator = u32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u4(n as u64size, (k * 2) as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u4(
            b as *const u8,
            n as u64size,
            (k * 2) as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_u4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for i4x2 {
    type Accumulator = i32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_i4(n as u64size, (k * 2) as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i4(
            b as *const u8,
            n as u64size,
            (k * 2) as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_i4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
        )
    }
}

impl Dots for u1x8 {
    type Accumulator = u32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u1(n as u64size, (k * 8) as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u1(
            b as *const u8,
            n as u64size,
            (k * 8) as u64size,
            b_stride as u64size,
            packed,
        )
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
        nk_dots_packed_u1(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 8) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 8) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
            rows as u64size,
            cols as u64size,
            (d * 8) as u64size,
            v_stride as u64size,
            r_stride as u64size,
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
            n_vectors as u64size,
            (d * 8) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
            rows as u64size,
            cols as u64size,
            (d * 8) as u64size,
            v_stride as u64size,
            r_stride as u64size,
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
            n_vectors as u64size,
            (d * 8) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
                $ang_packed(
                    $cast(a),
                    packed,
                    c,
                    m as u64size,
                    n as u64size,
                    k as u64size,
                    a_stride as u64size,
                    c_stride as u64size,
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
                    n_vectors as u64size,
                    depth as u64size,
                    stride as u64size,
                    result,
                    result_stride as u64size,
                    row_start as u64size,
                    row_count as u64size,
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
                $euc_packed(
                    $cast(a),
                    packed,
                    c,
                    m as u64size,
                    n as u64size,
                    k as u64size,
                    a_stride as u64size,
                    c_stride as u64size,
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
                    n_vectors as u64size,
                    depth as u64size,
                    stride as u64size,
                    result,
                    result_stride as u64size,
                    row_start as u64size,
                    row_count as u64size,
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
        nk_angulars_packed_u4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
        nk_euclideans_packed_u4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
        nk_angulars_packed_i4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
        nk_euclideans_packed_i4(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            (k * 2) as u64size,
            a_stride as u64size,
            c_stride as u64size,
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
            n_vectors as u64size,
            (depth * 2) as u64size,
            stride as u64size,
            result,
            result_stride as u64size,
            row_start as u64size,
            row_count as u64size,
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
/// - Dot-product multiplication with [`TransposedMatrixMultiplier`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, wsum, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// use numkong::{Tensor, TransposedMatrixMultiplier};
///
/// let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
    strides: [usize; MAX_RANK],
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
                // Deallocate buffer using our allocator
                let layout = alloc::alloc::Layout::array::<T>(self.len).unwrap();
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
    pub fn try_new_in(shape: &[usize], value: T, alloc: A) -> Result<Self, TensorError> {
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

        // Allocate raw buffer using our allocator
        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
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

        let mut strides_arr = [0usize; MAX_RANK];
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

    /// Creates an Tensor from existing slice data using a custom allocator.
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

        // Allocate and copy using our allocator
        let ptr = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
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

        let mut strides_arr = [0usize; MAX_RANK];
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

    fn compute_strides_into(shape: &[usize], strides: &mut [usize; MAX_RANK]) {
        let elem_size = core::mem::size_of::<T>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
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

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
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
        self.strides[1] == core::mem::size_of::<T>()
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
    pub fn try_new(shape: &[usize], value: T) -> Result<Self, TensorError> {
        Self::try_new_in(shape, value, Global)
    }

    /// Creates an Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[T], shape: &[usize]) -> Result<Self, TensorError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn new(shape: &[usize], value: T) -> Self {
        Self::try_new(shape, value).expect("Tensor::new failed")
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
    strides: [usize; MAX_RANK],
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

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
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
        self.strides[1] == core::mem::size_of::<T>()
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>();
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
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
            let mut result = Tensor::try_new(self.shape(), unsafe { (*self.data).clone() })?;
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
                let row_ptr = unsafe { (self.data as *const u8).add(r * row_stride) as *const T };
                for c in 0..cols {
                    let elem_ptr =
                        unsafe { (row_ptr as *const u8).add(c * col_stride) as *const T };
                    dest[dest_idx] = unsafe { (*elem_ptr).clone() };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_idx in 0..self.len {
                // Compute pointer offset
                let mut offset = 0usize;
                for d in 0..self.ndim {
                    offset += indices[d] * self.strides[d];
                }
                let elem_ptr = unsafe { (self.data as *const u8).add(offset) as *const T };
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
    /// let vectors = Tensor::<f32>::try_new(&[100, 768], 0.0)?;
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
        let mut result = Tensor::<T::Accumulator, Global, MAX_RANK>::try_new(
            result_shape,
            T::Accumulator::default(),
        )?;

        unsafe {
            T::dots_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n_vectors, n_vectors],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::angulars_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n_vectors, n_vectors],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::euclideans_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
            Tensor::<u32, Global, MAX_RANK>::try_new(&[n_vectors, n_vectors], u32::default())?;
        unsafe {
            T::hammings_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        let mut result = Tensor::<T::JaccardResult, Global, MAX_RANK>::try_new(
            &[n_vectors, n_vectors],
            T::JaccardResult::default(),
        )?;
        unsafe {
            T::jaccards_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
                0,
                n_vectors,
            );
        }
        Ok(result)
    }
}

// endregion: TensorView

// region: TensorViewMut

/// A mutable view into a Tensor.
pub struct TensorViewMut<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *mut T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [usize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> TensorViewMut<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
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
        self.strides[1] == core::mem::size_of::<T>()
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>();
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
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

// endregion: TensorViewMut

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

    /// Create a mutable view of the entire array.
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T, MAX_RANK> {
        TensorViewMut {
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
    /// let arr = Tensor::<f32>::try_new(&[4, 5], 1.0).unwrap();
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
        let mut new_strides = [0usize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0usize;

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
                    offset += i * dim_stride;
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
                    offset += start * dim_stride;
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
                    new_strides[new_ndim] = (dim_stride as isize * step) as usize;
                    new_ndim += 1;
                    offset += start * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *const u8).add(offset) as *const T };

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
    pub fn slice_mut(
        &mut self,
        ranges: &[SliceRange],
    ) -> Result<TensorViewMut<'_, T, MAX_RANK>, TensorError> {
        if ranges.len() != self.ndim {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0usize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0usize;

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
                    offset += i * dim_stride;
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
                    offset += start * dim_stride;
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
                    new_strides[new_ndim] = (dim_stride as isize * step) as usize;
                    new_ndim += 1;
                    offset += start * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *mut u8).add(offset) as *mut T };

        Ok(TensorViewMut {
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
        let mut new_strides = [0usize; MAX_RANK];
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

        let mut strides_arr = [0usize; MAX_RANK];
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
pub type MatrixViewMut<'a, T> = TensorViewMut<'a, T, 2>;

// endregion: Type Aliases

// region: TransposedMatrixMultiplier

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
/// let b_packed = TransposedMatrixMultiplier::try_pack(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
///
/// For C = A × B where B is (k × n) (standard GEMM layout):
/// ```rust,ignore
/// let b_packed = TransposedMatrixMultiplier::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
pub struct TransposedMatrixMultiplier<T: Dots, A: Allocator = Global> {
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

// Safety: TransposedMatrixMultiplier owns its data and is just bytes
unsafe impl<T: Dots + Send, A: Allocator + Send> Send for TransposedMatrixMultiplier<T, A> {}
unsafe impl<T: Dots + Sync, A: Allocator + Sync> Sync for TransposedMatrixMultiplier<T, A> {}

impl<T: Dots, A: Allocator> Drop for TransposedMatrixMultiplier<T, A> {
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

impl<T: Dots, A: Allocator + Clone> Clone for TransposedMatrixMultiplier<T, A> {
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
impl<T: Dots, A: Allocator> TransposedMatrixMultiplier<T, A> {
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
                T::dots_pack(b.as_ptr(), n, k, b.stride(0), data.as_ptr());
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
impl<T: Dots> TransposedMatrixMultiplier<T, Global> {
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
        Self::try_pack(b).expect("TransposedMatrixMultiplier::pack failed")
    }

    /// Convenience constructor that panics on error.
    pub fn pack_transposed<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Self {
        Self::try_pack_transposed(b).expect("TransposedMatrixMultiplier::pack_transposed failed")
    }
}

// endregion: TransposedMatrixMultiplier

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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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

        let mut c = Tensor::try_new_in(&[m, n], T::Accumulator::default(), self.alloc.clone())?;
        unsafe {
            T::dots_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn dots_packed<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
    ) -> Tensor<T::Accumulator, A, MAX_RANK> {
        self.try_dots_packed(packed_b).expect("dots_packed failed")
    }
}

impl<T: Dots, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Dot-product multiply into existing output (avoids allocation).
    pub fn try_dots_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
                self.stride(0),
                c.stride(0),
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
    /// * `packed_b` - Pre-packed B matrix from `TransposedMatrixMultiplier::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, TransposedMatrixMultiplier};
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
    /// let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
    /// let mut c = Tensor::<f32>::try_new(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    /// ```
    pub fn try_dots_packed_parallel_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let a_stride = self.stride(0);
        let c_stride = c.stride(0);

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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::Accumulator, Global, MAX_RANK>::try_new(
            &[m, n],
            T::Accumulator::default(),
        )?;
        self.try_dots_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn dots_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
    /// let vectors = Tensor::<f32>::try_new(&[100, 768], 1.0).unwrap();
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
        let mut result = Tensor::<T::Accumulator, Global, MAX_RANK>::try_new(
            &[n, n],
            T::Accumulator::default(),
        )?;

        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride(0);
        let result_stride = result.stride(0);

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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut c = Tensor::try_new_in(&[m, n], T::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            T::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn angulars_packed<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
    ) -> Tensor<T::SpatialResult, A, MAX_RANK> {
        self.try_angulars_packed(packed_b)
            .expect("angulars_packed failed")
    }
}

impl<T: Angulars, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes angular distances into pre-allocated output.
    pub fn try_angulars_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
                self.stride(0),
                c.stride(0),
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::angulars_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut c = Tensor::try_new_in(&[m, n], T::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            T::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_packed<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
    ) -> Tensor<T::SpatialResult, A, MAX_RANK> {
        self.try_euclideans_packed(packed_b)
            .expect("euclideans_packed failed")
    }
}

impl<T: Euclideans, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes euclidean distances into pre-allocated output.
    pub fn try_euclideans_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
                self.stride(0),
                c.stride(0),
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        unsafe {
            T::euclideans_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let a_stride = self.stride(0);
        let c_stride = c.stride(0);
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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[m, n],
            T::SpatialResult::default(),
        )?;
        self.try_angulars_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn angulars_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride(0);
        let result_stride = result.stride(0);

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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let a_stride = self.stride(0);
        let c_stride = c.stride(0);
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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::SpatialResult, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[m, n],
            T::SpatialResult::default(),
        )?;
        self.try_euclideans_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_packed_parallel<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut result = Tensor::<T::SpatialResult, Global, MAX_RANK>::try_new(
            &[n, n],
            T::SpatialResult::default(),
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(result.as_mut_ptr());
        let stride = self.stride(0);
        let result_stride = result.stride(0);

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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut c = Tensor::try_new_in(&[m, n], u32::default(), self.alloc.clone())?;
        unsafe {
            T::hammings_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn hammings_packed<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
    ) -> Tensor<u32, A, MAX_RANK> {
        self.try_hammings_packed(packed_b)
            .expect("hammings_packed failed")
    }
}

impl<T: Hammings, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Hamming distances into pre-allocated output.
    pub fn try_hammings_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
                self.stride(0),
                c.stride(0),
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
        let mut result = Tensor::<u32, Global, MAX_RANK>::try_new(&[n, n], u32::default())?;
        unsafe {
            T::hammings_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
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
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
        let mut c = Tensor::try_new_in(&[m, n], T::JaccardResult::default(), self.alloc.clone())?;
        unsafe {
            T::jaccards_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn jaccards_packed<BA: Allocator>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
    ) -> Tensor<T::JaccardResult, A, MAX_RANK> {
        self.try_jaccards_packed(packed_b)
            .expect("jaccards_packed failed")
    }
}

impl<T: Jaccards, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Computes Jaccard distances into pre-allocated output.
    pub fn try_jaccards_packed_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &TransposedMatrixMultiplier<T, BA>,
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
                self.stride(0),
                c.stride(0),
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
        let mut result = Tensor::<T::JaccardResult, Global, MAX_RANK>::try_new(
            &[n, n],
            T::JaccardResult::default(),
        )?;
        unsafe {
            T::jaccards_symmetric(
                self.as_ptr(),
                n,
                k,
                self.stride(0),
                result.as_mut_ptr(),
                result.stride(0),
                0,
                n,
            );
        }
        Ok(result)
    }
}

// endregion: Tensor Hammings/Jaccards

// region: Tensor Elementwise Operations

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Apply element-wise scale: result\[i\] = α × self\[i\] + β
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::each_scale(self.as_slice(), alpha, beta, result.as_mut_slice());
        Ok(result)
    }

    /// Apply element-wise scale in-place: self\[i\] = α × self\[i\] + β
    pub fn scale_inplace(&mut self, alpha: T::Scalar, beta: T::Scalar) {
        // Need a temporary for in-place operation since input and output overlap
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::each_scale(slice, alpha, beta, self.as_mut_slice());
        }
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
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::each_sum(self.as_slice(), other.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sum in-place: self\[i\] = self\[i\] + other\[i\]
    pub fn add_inplace<const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::each_sum(slice, other.as_slice(), self.as_mut_slice());
        }
        Ok(())
    }
}

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Weighted sum: result\[i\] = α × self\[i\] + β × other\[i\]
    ///
    /// Returns a new array with the weighted sum.
    pub fn wsum<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::each_blend(
            self.as_slice(),
            other.as_slice(),
            alpha,
            beta,
            result.as_mut_slice(),
        );
        Ok(result)
    }
}

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
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
        if self.shape() != b.shape() || self.shape() != c.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(b.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::each_fma(
            self.as_slice(),
            b.as_slice(),
            c.as_slice(),
            alpha,
            beta,
            result.as_mut_slice(),
        );
        Ok(result)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Trigonometry

impl<T: Clone + EachSin, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sine: result\[i\] = sin(self\[i\])
    ///
    /// Input values are in radians.
    pub fn sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::sin(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sine in-place: self\[i\] = sin(self\[i\])
    pub fn sin_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::sin(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + EachCos, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise cosine: result\[i\] = cos(self\[i\])
    ///
    /// Input values are in radians.
    pub fn cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::cos(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise cosine in-place: self\[i\] = cos(self\[i\])
    pub fn cos_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::cos(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + EachATan, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise arctangent: result\[i\] = atan(self\[i\])
    ///
    /// Output values are in radians in the range (-π/2, π/2).
    pub fn atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::atan(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise arctangent in-place: self\[i\] = atan(self\[i\])
    pub fn atan_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::atan(slice, self.as_mut_slice());
        }
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

impl<const MAX_RANK: usize> Tensor<f32, Global, MAX_RANK> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f32 {
        let ones: Tensor<f32, Global, MAX_RANK> =
            Tensor::try_new(self.shape(), 1.0f32).expect("allocation failed");
        <f32 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

impl<const MAX_RANK: usize> Tensor<f64, Global, MAX_RANK> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f64 {
        let ones: Tensor<f64, Global, MAX_RANK> =
            Tensor::try_new(self.shape(), 1.0f64).expect("allocation failed");
        <f64 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

// endregion: Tensor Reductions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
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

    use crate::scalars::{FloatLike, TestableType};

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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b_t = Tensor::<T>::try_new(&[k, n], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack_transposed(&b_t).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
            let a = Tensor::<T>::try_new(&[m, k], T::one()).unwrap();
            let b = Tensor::<T>::try_new(&[n, k], T::one()).unwrap();
            let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 12);
        assert!(!arr.is_empty());

        // From slice
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);

        // Clone
        let arr = Tensor::<f32>::try_new(&[3, 4], 2.5f32).unwrap();
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
        let arr = Tensor::<f32>::try_new(&[4, 5], 1.0f32).unwrap();
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
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.t().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);

        // Contiguous check
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());

        // Matrix alias
        let mat: Matrix<f32> = Matrix::try_new(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_ops() {
        // Reshape
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.len(), 12);

        // Sum f32
        let arr = Tensor::<f32>::try_new(&[100], 1.0f32).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.001);

        // Sum f64
        let arr = Tensor::<f64>::try_new(&[100], 1.0f64).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 1e-9);
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
            let vectors = Tensor::<T>::try_new(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_dots_symmetric().unwrap();
            assert_eq!(
                gram_matrix.shape(),
                &[num_vectors, num_vectors],
                "shape @ ({num_vectors},{depth})"
            );
            let expected = T::dimensions_per_value() as f64 * depth as f64;
            let tolerance = T::atol() + T::rtol() * expected.abs();
            // All-ones vectors: every entry = self-dot = expected
            for (index, &value) in gram_matrix.as_slice().iter().enumerate() {
                assert!(
                    (value.to_f64() - expected).abs() <= tolerance,
                    "({num_vectors},{depth})[{index}]: {} vs {expected}",
                    value.to_f64()
                );
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
            let vectors = Tensor::<T>::try_new(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_angulars_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for &value in gram_matrix.as_slice() {
                assert!(
                    value.to_f64().abs() <= tolerance,
                    "angular symmetric: {}",
                    value.to_f64()
                );
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
            let vectors = Tensor::<T>::try_new(&[num_vectors, depth], T::one()).unwrap();
            let gram_matrix = vectors.view().try_euclideans_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for &value in gram_matrix.as_slice() {
                assert!(
                    value.to_f64().abs() <= tolerance,
                    "euclidean symmetric: {}",
                    value.to_f64()
                );
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
        let a = Tensor::<u1x8>::try_new(&[4, 8], u1x8(0xFF)).unwrap();
        let b = Tensor::<u1x8>::try_new(&[16, 8], u1x8(0xFF)).unwrap();

        // Dots
        let b_packed = TransposedMatrixMultiplier::try_pack(&b).unwrap();
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
        let a = Tensor::<u1x8>::try_new(&[4, 8], u1x8(0xFF)).unwrap();

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
