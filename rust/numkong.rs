//! # NumKong - Hardware-Accelerated Similarity Metrics and Distance Functions
//!
//! * Targets ARM NEON, SVE, x86 AVX2, AVX-512 (VNNI, FP16) hardware backends.
//! * Handles `f64` double- and `f32` single-precision, integral, and binary vectors.
//! * Exposes half-precision (`f16`) and brain floating point (`bf16`) types.
//! * Zero-dependency header-only C 99 library with bindings for Rust and other languages.
//!
//! ## Implemented distance functions include:
//!
//! * Euclidean (L2), inner product, and angular (cosine) spatial distances.
//! * Hamming (~ Manhattan) and Jaccard (~ Tanimoto) binary distances.
//! * Kullback-Leibler and Jensen-Shannon divergences for probability distributions.
//!
//! ## Example
//!
//! ```rust
//! use numkong::{Dot, Angular, Euclidean};
//!
//! let a = &[1.0_f32, 2.0, 3.0];
//! let b = &[4.0_f32, 5.0, 6.0];
//!
//! let dot_product = f32::dot(a, b);
//! let angular_dist = f32::angular(a, b);
//! let l2sq_dist = f32::l2sq(a, b);
//!
//! // Optimize performance by flushing denormals
//! numkong::capabilities::configure_thread();
//! ```
//!
//! ## Mixed Precision Support
//!
//! ```rust
//! use numkong::{Angular, f16, bf16};
//!
//! // Work with half-precision floats
//! let half_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_angular_dist = f16::angular(&half_a, &half_b);
//!
//! // Work with brain floats
//! let brain_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_angular_dist = bf16::angular(&brain_a, &brain_b);
//!
//! // Direct bit manipulation
//! let half = f16::from_f32(3.14);
//! let bits = half.0; // Access raw u16 representation
//! let reconstructed = f16(bits);
//! ```
//!
//! ## Traits
//!
//! The `SpatialSimilarity` trait (combining `Dot`, `Angular`, `Euclidean`) covers:
//!
//! - `dot(a, b)`: Computes dot product between two slices.
//! - `angular(a, b)` / `cosine(a, b)`: Computes angular distance (1 - cosine similarity).
//! - `l2sq(a, b)` / `sqeuclidean(a, b)`: Computes squared Euclidean distance.
//! - `l2(a, b)` / `euclidean(a, b)`: Computes Euclidean distance.
//!
//! The `BinarySimilarity` trait (combining `Hamming`, `Jaccard`) covers:
//!
//! - `hamming(a, b)`: Computes Hamming distance between two slices.
//! - `jaccard(a, b)`: Computes Jaccard distance between two slices.
//!
//! The `ProbabilitySimilarity` trait (combining `KullbackLeibler`, `JensenShannon`) covers:
//!
//! - `jensenshannon(a, b)`: Computes Jensen-Shannon divergence.
//! - `kullbackleibler(a, b)`: Computes Kullback-Leibler divergence.
//!
//! The `Elementwise` trait (combining `Scale`, `Sum`, `WSum`, `FMA`) covers:
//!
//! - `scale(a, alpha, beta, result)`: Element-wise `result[i] = alpha * a[i] + beta`.
//! - `sum(a, b, result)`: Element-wise `result[i] = a[i] + b[i]`.
//! - `wsum(a, b, alpha, beta, result)`: Weighted sum `result[i] = alpha * a[i] + beta * b[i]`.
//! - `fma(a, b, c, alpha, beta, result)`: Fused multiply-add `result[i] = alpha * a[i] * b[i] + beta * c[i]`.
//!
//! The `Trigonometry` trait (combining `Sin`, `Cos`, `ATan`) covers:
//!
//! - `sin(input, result)`: Element-wise sine.
//! - `cos(input, result)`: Element-wise cosine.
//! - `atan(input, result)`: Element-wise arctangent.
//!
//! Additional traits: `ComplexDot`, `ComplexVDot`, `Sparse`.
//!
#![allow(non_camel_case_types)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

extern crate alloc;

pub type ComplexProductF32 = (f32, f32);
pub type ComplexProductF64 = (f64, f64);

/// Size type used in C FFI to match `nk_size_t` which is always `uint64_t`.
type u64size = u64;

/// Compatibility function for pre 1.85 Rust versions lacking `f32::abs`.
#[inline(always)]
fn f32_abs_compat(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[link(name = "numkong")]
extern "C" {

    fn nk_uses_neon() -> i32;
    fn nk_uses_neonhalf() -> i32;
    fn nk_uses_neonbfdot() -> i32;
    fn nk_uses_neonsdot() -> i32;
    fn nk_uses_sve() -> i32;
    fn nk_uses_svehalf() -> i32;
    fn nk_uses_svebfdot() -> i32;
    fn nk_uses_svesdot() -> i32;
    fn nk_uses_sve2() -> i32;
    fn nk_uses_sve2p1() -> i32;
    fn nk_uses_neonfhm() -> i32;
    fn nk_uses_sme() -> i32;
    fn nk_uses_sme2() -> i32;
    fn nk_uses_sme2p1() -> i32;
    fn nk_uses_smef64() -> i32;
    fn nk_uses_smehalf() -> i32;
    fn nk_uses_smebf16() -> i32;
    fn nk_uses_smelut2() -> i32;
    fn nk_uses_smefa64() -> i32;
    fn nk_uses_haswell() -> i32;
    fn nk_uses_skylake() -> i32;
    fn nk_uses_ice() -> i32;
    fn nk_uses_genoa() -> i32;
    fn nk_uses_sapphire() -> i32;
    fn nk_uses_turin() -> i32;
    fn nk_uses_sierra() -> i32;
    fn nk_uses_sapphire_amx() -> i32;
    fn nk_uses_granite_amx() -> i32;

    fn nk_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_dynamic_dispatch() -> i32;

    fn nk_f32_to_f16(src: *const f32, dest: *mut u16);
    fn nk_f16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_bf16(src: *const f32, dest: *mut u16);
    fn nk_bf16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_e4m3(src: *const f32, dest: *mut u8);
    fn nk_e4m3_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e5m2(src: *const f32, dest: *mut u8);
    fn nk_e5m2_to_f32(src: *const u8, dest: *mut f32);

    // Vector dot products
    fn nk_dot_i8(a: *const i8, b: *const i8, c: u64size, d: *mut i32);
    fn nk_dot_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
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
    fn nk_angular_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_angular_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_l2sq_i8(a: *const i8, b: *const i8, c: u64size, d: *mut u32);
    fn nk_l2sq_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2sq_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2sq_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_l2sq_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_l2_i8(a: *const i8, b: *const i8, c: u64size, d: *mut f32);
    fn nk_l2_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_l2_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_hamming_b8(a: *const u8, b: *const u8, c: u64size, d: *mut u32);
    fn nk_jaccard_b8(a: *const u8, b: *const u8, c: u64size, d: *mut f32);

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
    fn nk_intersect_u16(
        a: *const u16,
        b: *const u16,
        a_length: u64size,
        b_length: u64size,
        d: *mut u32,
    );
    fn nk_intersect_u32(
        a: *const u32,
        b: *const u32,
        a_length: u64size,
        b_length: u64size,
        d: *mut u32,
    );

    // Trigonometry functions
    fn nk_sin_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_sin_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_cos_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_cos_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_atan_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_atan_f64(inputs: *const f64, n: u64size, outputs: *mut f64);

    // Elementwise operations
    fn nk_scale_f64(
        a: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_scale_f32(
        a: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_scale_f16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_bf16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_i8(a: *const i8, n: u64size, alpha: *const f32, beta: *const f32, result: *mut i8);
    fn nk_scale_u8(a: *const u8, n: u64size, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_scale_i16(
        a: *const i16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i16,
    );
    fn nk_scale_u16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_i32(
        a: *const i32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i32,
    );
    fn nk_scale_u32(
        a: *const u32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u32,
    );
    fn nk_scale_i64(
        a: *const i64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i64,
    );
    fn nk_scale_u64(
        a: *const u64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u64,
    );

    fn nk_sum_f64(a: *const f64, b: *const f64, n: u64size, result: *mut f64);
    fn nk_sum_f32(a: *const f32, b: *const f32, n: u64size, result: *mut f32);
    fn nk_sum_f16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_bf16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_i8(a: *const i8, b: *const i8, n: u64size, result: *mut i8);
    fn nk_sum_u8(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_sum_i16(a: *const i16, b: *const i16, n: u64size, result: *mut i16);
    fn nk_sum_u16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_i32(a: *const i32, b: *const i32, n: u64size, result: *mut i32);
    fn nk_sum_u32(a: *const u32, b: *const u32, n: u64size, result: *mut u32);
    fn nk_sum_i64(a: *const i64, b: *const i64, n: u64size, result: *mut i64);
    fn nk_sum_u64(a: *const u64, b: *const u64, n: u64size, result: *mut u64);

    fn nk_wsum_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_wsum_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_wsum_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_wsum_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_wsum_i8(
        a: *const i8,
        b: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_wsum_u8(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_fma_f64(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_fma_f32(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_fma_f16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_fma_bf16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_fma_i8(
        a: *const i8,
        b: *const i8,
        c: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_fma_u8(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
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

    // Matrix multiplication (GEMM) with packed B matrix
    // F32: C[m×n] = A[m×k] × B[n×k]ᵀ
    fn nk_dots_f32f32f32_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_f32f32f32_pack(
        b: *const f32,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_f32f32f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // F64 GEMM
    fn nk_dots_f64f64f64_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_f64f64f64_pack(
        b: *const f64,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_f64f64f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // F16 GEMM (accumulates to F32)
    fn nk_dots_f16f16f32_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_f16f16f32_pack(
        b: *const u16,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_f16f16f32(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // BF16 GEMM (accumulates to F32)
    fn nk_dots_bf16bf16f32_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_bf16bf16f32_pack(
        b: *const u16,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_bf16bf16f32(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // I8 GEMM (accumulates to I32)
    fn nk_dots_i8i8i32_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_i8i8i32_pack(
        b: *const i8,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_i8i8i32(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    // U8 GEMM (accumulates to U32)
    fn nk_dots_u8u8u32_packed_size(n: u64size, k: u64size) -> u64size;
    fn nk_dots_u8u8u32_pack(
        b: *const u8,
        n: u64size,
        k: u64size,
        b_stride: u64size,
        packed: *mut u8,
    );
    fn nk_dots_u8u8u32(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
}

// region: f16 Type

/// A half-precision (16-bit) floating point number.
///
/// This type represents IEEE 754 half-precision binary floating-point format.
/// It provides conversion methods to and from f32, and the underlying u16
/// representation is publicly accessible for direct bit manipulation.
///
/// # Examples
///
/// ```
/// use numkong::f16;
///
/// // Create from f32
/// let half = f16::from_f32(3.14);
///
/// // Convert back to f32
/// let float = half.to_f32();
///
/// // Direct access to bits
/// let bits = half.0;
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct f16(pub u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = f16(0);
    /// Positive one.
    pub const ONE: Self = f16(0x3C00);
    /// Negative one.
    pub const NEG_ONE: Self = f16(0xBC00);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_f16(&value, &mut result) };
        f16(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_f16_to_f32(&self.0, &mut result) };
        result
    }

    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.to_f32().is_nan()
    }

    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        self.to_f32().is_infinite()
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.to_f32().is_finite()
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for f16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for f16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for f16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for f16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for f16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for f16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: f16 Type

// region: bf16 Type

/// A brain floating point (bfloat16) number.
///
/// Google's bfloat16 format truncates IEEE 754 single-precision to 16 bits,
/// keeping the sign bit, 8 exponent bits, and 7 mantissa bits. This provides
/// wider dynamic range than f16 but lower precision.
///
/// # Examples
///
/// ```
/// use numkong::bf16;
///
/// let brain = bf16::from_f32(3.14);
/// let float = brain.to_f32();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct bf16(pub u16);

impl bf16 {
    /// Positive zero.
    pub const ZERO: Self = bf16(0);

    /// Positive one.
    pub const ONE: Self = bf16(0x3F80);

    /// Negative one.
    pub const NEG_ONE: Self = bf16(0xBF80);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_bf16(&value, &mut result) };
        bf16(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_bf16_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.to_f32().is_nan()
    }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        self.to_f32().is_infinite()
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.to_f32().is_finite()
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for bf16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for bf16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for bf16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for bf16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for bf16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for bf16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: bf16 Type

// region: e4m3 Type

/// An 8-bit floating point number in E4M3 format (OCP FP8).
///
/// E4M3 uses 1 sign bit, 4 exponent bits, and 3 mantissa bits. It provides
/// wider dynamic range than E5M2 but lower precision. Note: E4M3 has no
/// infinities, using those bit patterns for NaN instead.
///
/// # Examples
///
/// ```
/// use numkong::e4m3;
///
/// let fp8 = e4m3::from_f32(2.5);
/// let float = fp8.to_f32();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct e4m3(pub u8);

impl e4m3 {
    pub const ZERO: Self = e4m3(0x00);
    pub const ONE: Self = e4m3(0x38);
    pub const NEG_ONE: Self = e4m3(0xB8);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e4m3(&value, &mut result) };
        e4m3(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e4m3_to_f32(&self.0, &mut result) };
        result
    }

    #[inline(always)]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7F) == 0x7F
    }

    /// Returns true if this number is neither infinite nor NaN.
    /// Note: E4M3 format has no infinities.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        !self.is_nan()
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for e4m3 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: e4m3 Type

// region: e5m2 Type

/// An 8-bit floating point number in E5M2 format (OCP FP8).
///
/// E5M2 uses 1 sign bit, 5 exponent bits, and 2 mantissa bits. It has
/// a similar structure to IEEE half-precision but reduced precision.
/// Unlike E4M3, E5M2 supports infinities.
///
/// # Examples
///
/// ```
/// use numkong::e5m2;
///
/// let fp8 = e5m2::from_f32(2.5);
/// let float = fp8.to_f32();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct e5m2(pub u8);

impl e5m2 {
    /// Positive zero.
    pub const ZERO: Self = e5m2(0x00);

    /// Positive one.
    pub const ONE: Self = e5m2(0x3C);

    /// Negative one.
    pub const NEG_ONE: Self = e5m2(0xBC);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e5m2(&value, &mut result) };
        e5m2(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e5m2_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        exp == 0x1F && mant != 0
    }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        exp == 0x1F && mant == 0
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        exp != 0x1F
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for e5m2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: e5m2 Type

// region: Capabilities

/// Hardware capability detection functions.
pub mod capabilities {
    pub fn uses_neon() -> bool {
        unsafe { crate::nk_uses_neon() != 0 }
    }
    pub fn uses_neonhalf() -> bool {
        unsafe { crate::nk_uses_neonhalf() != 0 }
    }
    pub fn uses_neonbfdot() -> bool {
        unsafe { crate::nk_uses_neonbfdot() != 0 }
    }
    pub fn uses_neonsdot() -> bool {
        unsafe { crate::nk_uses_neonsdot() != 0 }
    }
    pub fn uses_sve() -> bool {
        unsafe { crate::nk_uses_sve() != 0 }
    }
    pub fn uses_svehalf() -> bool {
        unsafe { crate::nk_uses_svehalf() != 0 }
    }
    pub fn uses_svebfdot() -> bool {
        unsafe { crate::nk_uses_svebfdot() != 0 }
    }
    pub fn uses_svesdot() -> bool {
        unsafe { crate::nk_uses_svesdot() != 0 }
    }
    pub fn uses_sve2() -> bool {
        unsafe { crate::nk_uses_sve2() != 0 }
    }
    pub fn uses_sve2p1() -> bool {
        unsafe { crate::nk_uses_sve2p1() != 0 }
    }
    pub fn uses_neonfhm() -> bool {
        unsafe { crate::nk_uses_neonfhm() != 0 }
    }
    pub fn uses_sme() -> bool {
        unsafe { crate::nk_uses_sme() != 0 }
    }
    pub fn uses_sme2() -> bool {
        unsafe { crate::nk_uses_sme2() != 0 }
    }
    pub fn uses_sme2p1() -> bool {
        unsafe { crate::nk_uses_sme2p1() != 0 }
    }
    pub fn uses_smef64() -> bool {
        unsafe { crate::nk_uses_smef64() != 0 }
    }
    pub fn uses_smehalf() -> bool {
        unsafe { crate::nk_uses_smehalf() != 0 }
    }
    pub fn uses_smebf16() -> bool {
        unsafe { crate::nk_uses_smebf16() != 0 }
    }
    pub fn uses_smelut2() -> bool {
        unsafe { crate::nk_uses_smelut2() != 0 }
    }
    pub fn uses_smefa64() -> bool {
        unsafe { crate::nk_uses_smefa64() != 0 }
    }
    pub fn uses_haswell() -> bool {
        unsafe { crate::nk_uses_haswell() != 0 }
    }
    pub fn uses_skylake() -> bool {
        unsafe { crate::nk_uses_skylake() != 0 }
    }
    pub fn uses_ice() -> bool {
        unsafe { crate::nk_uses_ice() != 0 }
    }
    pub fn uses_genoa() -> bool {
        unsafe { crate::nk_uses_genoa() != 0 }
    }
    pub fn uses_sapphire() -> bool {
        unsafe { crate::nk_uses_sapphire() != 0 }
    }
    pub fn uses_turin() -> bool {
        unsafe { crate::nk_uses_turin() != 0 }
    }
    pub fn uses_sierra() -> bool {
        unsafe { crate::nk_uses_sierra() != 0 }
    }
    pub fn uses_sapphire_amx() -> bool {
        unsafe { crate::nk_uses_sapphire_amx() != 0 }
    }
    pub fn uses_granite_amx() -> bool {
        unsafe { crate::nk_uses_granite_amx() != 0 }
    }

    /// Flushes denormalized numbers to zero on the current CPU.
    ///
    /// Call this on each thread before performing SIMD operations to avoid
    /// significant performance penalties. FMA operations can be up to 30x
    /// slower when operating on denormalized values.
    ///
    /// Returns `true` if the flush was successful.
    pub fn configure_thread() -> bool {
        unsafe { crate::nk_configure_thread(0) != 0 }
    }

    /// Returns `true` if the library uses dynamic dispatch for function selection.
    ///
    /// When compiled with dynamic dispatch, the library selects the optimal
    /// SIMD implementation at runtime based on detected CPU capabilities.
    pub fn uses_dynamic_dispatch() -> bool {
        unsafe { crate::nk_uses_dynamic_dispatch() != 0 }
    }
}

// endregion: Capabilities

/// region: Dot

/// Computes the **dot product** (inner product) between two vectors.
///
/// # Formula
///
/// For vectors **a** and **b** of length *n*:
///
/// ```text
/// dot(a, b) = Σᵢ aᵢ · bᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
/// ```
///
/// # Properties
///
/// - **Symmetric**: `dot(a, b) = dot(b, a)`
/// - **Range**: `(-∞, +∞)` for real vectors
/// - **Self-dot**: `dot(a, a) = ||a||²` (squared L2 norm)
///
/// # Use Cases
///
/// - **Similarity search**: Higher dot product indicates more similar vectors
///   (when vectors are normalized, equals cosine similarity)
/// - **Neural networks**: Core operation in dense layers, attention mechanisms
/// - **Projections**: Computing component of one vector along another
/// - **Physics**: Work = Force · Displacement
///
/// # Relationship to Other Metrics
///
/// - For unit vectors: `dot(a, b) = cos(θ)` where θ is the angle between them
/// - `angular(a, b) = 1 - dot(a, b) / (||a|| · ||b||)`
/// - `l2sq(a, b) = dot(a, a) + dot(b, b) - 2·dot(a, b)`
///
/// # Returns
///
/// `None` if the slices differ in length.
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

// endregion: Dot

// region: Angular

/// Computes the **angular distance** (cosine distance) between two vectors.
///
/// # Formula
///
/// ```text
/// angular(a, b) = 1 - cos(θ) = 1 - (a · b) / (||a|| · ||b||)
/// ```
///
/// Where `θ` is the angle between vectors **a** and **b**.
///
/// # Properties
///
/// - **Symmetric**: `angular(a, b) = angular(b, a)`
/// - **Range**: `[0, 2]` — 0 means identical direction, 1 means orthogonal, 2 means opposite
/// - **Scale-invariant**: `angular(a, b) = angular(k·a, b)` for any `k > 0`
/// - **Not a true metric**: Violates triangle inequality (use `arccos` for true angular metric)
///
/// # Use Cases
///
/// - **Semantic search**: Comparing text/document embeddings (direction matters, not magnitude)
/// - **Recommendation systems**: User/item similarity in collaborative filtering
/// - **Image retrieval**: Comparing image feature vectors
/// - **Clustering**: K-means with cosine similarity
///
/// # When to Use Angular vs Euclidean
///
/// - Use **Angular** when vector magnitude is meaningless (e.g., TF-IDF, word embeddings)
/// - Use **Euclidean** when magnitude carries information (e.g., physical coordinates)
/// - Angular is more robust to varying vector lengths in high dimensions
///
/// # Relationship to Other Metrics
///
/// - `cosine_similarity = 1 - angular(a, b)`
/// - For unit vectors: `angular(a, b) = l2sq(a, b) / 2`
///
/// # Returns
///
/// `None` if the slices differ in length.
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

// endregion: Angular

// region: Euclidean

/// Computes the **Euclidean distance** (L2) between two vectors.
///
/// # Formula
///
/// **Squared Euclidean** (faster, avoids `sqrt`):
/// ```text
/// l2sq(a, b) = Σᵢ (aᵢ - bᵢ)² = ||a - b||²
/// ```
///
/// **Euclidean** (true distance):
/// ```text
/// l2(a, b) = √(Σᵢ (aᵢ - bᵢ)²) = ||a - b||
/// ```
///
/// # Properties
///
/// - **Symmetric**: `l2(a, b) = l2(b, a)`
/// - **Range**: `[0, +∞)`
/// - **Identity**: `l2(a, a) = 0`
/// - **True metric**: Satisfies triangle inequality `l2(a, c) ≤ l2(a, b) + l2(b, c)`
///
/// # Use Cases
///
/// - **K-Nearest Neighbors (KNN)**: Finding closest points in feature space
/// - **Clustering**: K-means, DBSCAN, hierarchical clustering
/// - **Anomaly detection**: Points far from cluster centers
/// - **Physical simulations**: Actual spatial distance
///
/// # Performance Notes
///
/// - Use `l2sq` when you only need to compare distances (avoids `sqrt`)
/// - `l2sq` is ~2x faster than `l2` for ranking/comparison tasks
/// - For normalized vectors: `l2sq(a, b) = 2 · angular(a, b)`
///
/// # When to Use L2 vs L2sq
///
/// - Use **l2sq** for: comparisons, rankings, nearest neighbor search
/// - Use **l2** for: actual distance values, radius searches, physical meaning
///
/// # Returns
///
/// `None` if the slices differ in length.
pub trait Euclidean: Sized {
    type L2sqOutput;
    type L2Output;

    /// Squared Euclidean distance (L2²). Faster than `l2` for comparisons.
    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput>;

    /// Euclidean distance (L2). True metric distance.
    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output>;

    /// Alias for `l2sq` (SciPy compatibility).
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        Self::l2sq(a, b)
    }
}

impl Euclidean for f64 {
    type L2sqOutput = f64;
    type L2Output = f64;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe { nk_l2sq_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f32 {
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe { nk_l2sq_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f16 {
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe {
            nk_l2sq_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe {
            nk_l2_f16(
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
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe {
            nk_l2sq_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe {
            nk_l2_bf16(
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
    type L2sqOutput = u32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0;
        unsafe { nk_l2sq_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Euclidean

// region: Geospatial

/// Computes **great-circle distances** between geographic coordinates on Earth.
///
/// # Formula
///
/// **Haversine** (fast approximation assuming spherical Earth):
/// ```text
/// a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
/// c = 2·atan2(√a, √(1-a))
/// distance = R·c
/// ```
///
/// # Input Format
///
/// Coordinates must be in **radians**, not degrees.
/// Convert with: `radians = degrees × π/180`
///
/// # Output
///
/// Writes distances in **meters** to the provided output buffer.
///
/// # Returns
///
/// `None` if input slices have mismatched lengths.
pub trait Haversine: Sized {
    /// Compute Haversine (spherical) great-circle distances.
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
/// More accurate than Haversine (~0.5mm accuracy vs ~0.3% error).
///
/// # Input Format
///
/// Coordinates must be in **radians**, not degrees.
///
/// # Output
///
/// Writes distances in **meters** to the provided output buffer.
///
/// # Returns
///
/// `None` if input slices have mismatched lengths.
pub trait Vincenty: Sized {
    /// Compute Vincenty (ellipsoidal) geodesic distances.
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
            );
        }
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
            );
        }
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
            );
        }
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
            );
        }
        Some(())
    }
}

impl Geospatial for f32 {}

// endregion: Geospatial

// region: Hamming

/// Computes the **Hamming distance** between two binary vectors.
///
/// # Formula
///
/// For binary vectors (packed as bytes):
/// ```text
/// hamming(a, b) = popcount(a ⊕ b) = number of differing bits
/// ```
///
/// Where `⊕` is bitwise XOR and `popcount` counts set bits.
///
/// # Properties
///
/// - **Symmetric**: `hamming(a, b) = hamming(b, a)`
/// - **Range**: `[0, n·8]` where n is number of bytes
/// - **True metric**: Satisfies triangle inequality
/// - **Integral result**: Always returns whole numbers
///
/// # Use Cases
///
/// - **Binary embeddings**: Comparing locality-sensitive hashes (LSH)
/// - **Error detection**: Counting bit errors in transmitted data
/// - **DNA/Genome**: Comparing nucleotide sequences (encoded as bits)
/// - **Near-duplicate detection**: Comparing SimHash/MinHash fingerprints
///
/// # Bit Packing
///
/// Input bytes are treated as packed binary vectors:
/// - Each `u8` contains 8 binary features
/// - Vector of 128 bytes = 1024 binary dimensions
///
/// # Performance
///
/// Uses hardware `POPCNT` instruction on modern CPUs for optimal speed.
///
/// # Returns
///
/// `None` if the slices differ in length.
pub trait Hamming: Sized {
    type Output;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Hamming for u8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_hamming_b8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Hamming

// region: Jaccard

/// Computes the **Jaccard distance** between two binary vectors.
///
/// # Formula
///
/// For binary vectors (packed as bytes):
/// ```text
/// jaccard(a, b) = 1 - |a ∩ b| / |a ∪ b|
///              = 1 - popcount(a & b) / popcount(a | b)
/// ```
///
/// Also known as the **Tanimoto distance** in cheminformatics.
///
/// # Properties
///
/// - **Symmetric**: `jaccard(a, b) = jaccard(b, a)`
/// - **Range**: `[0, 1]` — 0 means identical, 1 means completely different
/// - **True metric**: Satisfies triangle inequality
/// - **Set-theoretic**: Based on set intersection and union
///
/// # Use Cases
///
/// - **Molecular similarity**: Comparing chemical fingerprints (RDKit, ECFP)
/// - **Document similarity**: Comparing shingle/n-gram sets
/// - **Recommendation**: Comparing user preference sets
/// - **Plagiarism detection**: Comparing token sets
///
/// # Jaccard vs Hamming
///
/// - **Jaccard** normalizes by union size → scale-invariant
/// - **Hamming** counts raw bit differences → sensitive to vector density
/// - Use Jaccard for sparse binary vectors (most bits are 0)
/// - Use Hamming for dense binary vectors
///
/// # Returns
///
/// `None` if the slices differ in length.
pub trait Jaccard: Sized {
    type Output;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Jaccard for u8 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_b8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Jaccard

// region: KullbackLeibler

/// Computes the **Kullback-Leibler divergence** between two probability distributions.
///
/// # Formula
///
/// ```text
/// KL(P || Q) = Σᵢ pᵢ · log(pᵢ / qᵢ)
/// ```
///
/// Measures how much distribution **P** differs from reference distribution **Q**.
///
/// # Properties
///
/// - **Asymmetric**: `KL(P || Q) ≠ KL(Q || P)` in general!
/// - **Range**: `[0, +∞)` — 0 only when P = Q
/// - **Not a metric**: Violates symmetry and triangle inequality
/// - **Requires**: Q(x) > 0 wherever P(x) > 0 (otherwise undefined)
///
/// # Use Cases
///
/// - **Machine learning**: Loss function for classification (cross-entropy)
/// - **Information theory**: Measuring information gain
/// - **Variational inference**: ELBO optimization in VAEs
/// - **Model comparison**: How well Q approximates P
///
/// # Interpretation
///
/// - KL(P || Q) = expected extra bits needed to encode P using code optimized for Q
/// - Forward KL: mode-seeking (Q tries to cover P's modes)
/// - Reverse KL: mean-seeking (Q avoids P's low-probability regions)
///
/// # Important Notes
///
/// - Inputs must be valid probability distributions (sum to 1, non-negative)
/// - Returns `+∞` if Q has zeros where P is non-zero
/// - For a symmetric alternative, use `JensenShannon`
///
/// # Returns
///
/// `None` if the slices differ in length.
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
/// # Formula
///
/// ```text
/// JS(P, Q) = ½·KL(P || M) + ½·KL(Q || M)
/// ```
///
/// Where `M = ½(P + Q)` is the mixture distribution.
///
/// # Properties
///
/// - **Symmetric**: `JS(P, Q) = JS(Q, P)` ✓
/// - **Range**: `[0, log(2)]` ≈ `[0, 0.693]` (using natural log)
/// - **True metric**: `√JS` satisfies triangle inequality
/// - **Bounded**: Always finite, even when distributions have different support
///
/// # Use Cases
///
/// - **Topic modeling**: Comparing document topic distributions (LDA)
/// - **Image segmentation**: Comparing pixel intensity distributions
/// - **Clustering**: Grouping similar probability distributions
/// - **GAN training**: Measuring generator vs real data distribution
///
/// # JS vs KL Divergence
///
/// | Property | KL | JS |
/// |----------|----|----|
/// | Symmetric | ✗ | ✓ |
/// | Bounded | ✗ | ✓ |
/// | Metric (sqrt) | ✗ | ✓ |
/// | Handles zero probs | ✗ | ✓ |
///
/// Use JS when you need a symmetric, bounded measure that handles
/// distributions with different support.
///
/// # Returns
///
/// `None` if the slices differ in length.
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
/// # Formula
///
/// For complex vectors **a** and **b** with elements `aₖ = aᵣₖ + i·aᵢₖ`:
///
/// ```text
/// dot(a, b) = Σₖ aₖ · bₖ = Σₖ (aᵣₖ + i·aᵢₖ)(bᵣₖ + i·bᵢₖ)
///           = Σₖ (aᵣₖ·bᵣₖ - aᵢₖ·bᵢₖ) + i·Σₖ (aᵣₖ·bᵢₖ + aᵢₖ·bᵣₖ)
/// ```
///
/// # Memory Layout
///
/// Complex numbers are stored as **interleaved** real/imaginary pairs:
/// ```text
/// [re₀, im₀, re₁, im₁, re₂, im₂, ...]
/// ```
///
/// A slice of length 6 represents 3 complex numbers.
///
/// # Properties
///
/// - **Not conjugate**: This is `Σ aₖ·bₖ`, not the Hermitian inner product
/// - **Bilinear**: Linear in both arguments (over complex field)
/// - **Returns**: `(real_part, imaginary_part)` as tuple
///
/// # Use Cases
///
/// - **Signal processing**: Correlation of complex signals
/// - **Quantum computing**: Inner products in Hilbert space (use `ComplexVDot` for proper QM)
/// - **Fourier analysis**: Working with frequency-domain data
///
/// # See Also
///
/// - [`ComplexVDot`]: Conjugate dot product (Hermitian inner product) for proper norms
///
/// # Returns
///
/// `None` if slices differ in length or have odd length (not valid complex vectors).
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
/// # Formula
///
/// For complex vectors **a** and **b**:
///
/// ```text
/// vdot(a, b) = Σₖ conj(aₖ) · bₖ = Σₖ (aᵣₖ - i·aᵢₖ)(bᵣₖ + i·bᵢₖ)
///            = Σₖ (aᵣₖ·bᵣₖ + aᵢₖ·bᵢₖ) + i·Σₖ (aᵣₖ·bᵢₖ - aᵢₖ·bᵣₖ)
/// ```
///
/// This conjugates the **first** argument, matching NumPy's `numpy.vdot` convention.
///
/// # Memory Layout
///
/// Same as `ComplexDot`: interleaved `[re₀, im₀, re₁, im₁, ...]`
///
/// # Properties
///
/// - **Sesquilinear**: Conjugate-linear in first argument, linear in second
/// - **Proper norm**: `vdot(a, a)` gives real, non-negative result = `||a||²`
/// - **Hermitian**: `vdot(a, b) = conj(vdot(b, a))`
///
/// # Use Cases
///
/// - **Quantum mechanics**: Inner product `⟨ψ|φ⟩` in Hilbert space
/// - **Signal processing**: Computing power spectral density
/// - **Complex norms**: `||a||² = real(vdot(a, a))`
///
/// # ComplexDot vs ComplexVDot
///
/// | Operation | ComplexDot | ComplexVDot |
/// |-----------|------------|-------------|
/// | Formula | `Σ aₖ·bₖ` | `Σ conj(aₖ)·bₖ` |
/// | Self-product | Complex | Real (≥0) |
/// | Use for | Correlation | Norms, QM |
///
/// # Returns
///
/// `None` if slices differ in length or have odd length.
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

// region: Sparse

/// Computes the **intersection size** between two sorted sparse vectors.
///
/// # Formula
///
/// ```text
/// intersect(a, b) = |{x : x ∈ a ∧ x ∈ b}|
/// ```
///
/// Returns the count of elements present in both vectors.
///
/// # Input Requirements
///
/// - Both vectors must be **sorted in ascending order**
/// - Elements are typically feature indices (u16 or u32)
/// - Vectors can have different lengths
///
/// # Algorithm
///
/// Uses an optimized merge-style intersection with SIMD acceleration,
/// running in O(min(|a|, |b|)) time for typical cases.
///
/// # Use Cases
///
/// - **Sparse feature matching**: Comparing bags-of-words, n-gram sets
/// - **Inverted index search**: Finding documents with matching terms
/// - **Set similarity**: Computing Jaccard = intersection / union
/// - **Recommendation**: Finding common items between users
///
/// # Sparse vs Dense Representations
///
/// | Representation | Memory | Best For |
/// |----------------|--------|----------|
/// | Dense (f32/f64) | O(dimensions) | < 1000 dims, most non-zero |
/// | Sparse (indices) | O(nnz) | High-dim, mostly zero |
///
/// Use sparse when less than ~10% of features are non-zero.
///
/// # Example
///
/// ```text
/// a = [1, 5, 10, 20]     // sorted feature indices
/// b = [2, 5, 15, 20, 30]
/// intersect(a, b) = 2    // {5, 20} are common
/// ```
///
/// # Returns
///
/// Count of common elements. Always succeeds (returns `Some`).
pub trait Sparse: Sized {
    type Output;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Sparse for u16 {
    type Output = u32;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        let mut result: Self::Output = 0;
        unsafe {
            nk_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Sparse for u32 {
    type Output = u32;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        let mut result: Self::Output = 0;
        unsafe {
            nk_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Sparse

// region: Sin

/// Computes **element-wise sine** of a vector.
///
/// # Formula
///
/// ```text
/// outputs[i] = sin(inputs[i])
/// ```
///
/// Input values are in **radians**.
///
/// # Properties
///
/// - **Range**: `[-1, +1]`
/// - **Periodic**: `sin(x + 2π) = sin(x)`
/// - **Odd function**: `sin(-x) = -sin(x)`
///
/// # Implementation
///
/// Uses SIMD-accelerated polynomial approximation (typically 7th-order)
/// with accuracy of ~1e-6 for f32 and ~1e-12 for f64.
///
/// # Use Cases
///
/// - **Signal processing**: Generating waveforms, modulation
/// - **Graphics**: Rotation transforms, animation curves
/// - **Physics**: Oscillatory motion, wave equations
/// - **Audio**: Synthesizer oscillators
///
/// # Performance
///
/// 5-10x faster than scalar `libm` by using SIMD and avoiding
/// expensive range reduction for typical input ranges.
///
/// # Returns
///
/// `None` if input and output lengths differ.
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

// endregion: Sin

// region: Cos

/// Computes **element-wise cosine** of a vector.
///
/// # Formula
///
/// ```text
/// outputs[i] = cos(inputs[i])
/// ```
///
/// Input values are in **radians**.
///
/// # Properties
///
/// - **Range**: `[-1, +1]`
/// - **Periodic**: `cos(x + 2π) = cos(x)`
/// - **Even function**: `cos(-x) = cos(x)`
/// - **Phase shift**: `cos(x) = sin(x + π/2)`
///
/// # Implementation
///
/// Uses SIMD-accelerated polynomial approximation. Shares implementation
/// with `Sin` (cos computed as phase-shifted sin).
///
/// # Use Cases
///
/// - **Signal processing**: Quadrature signals, I/Q demodulation
/// - **Graphics**: Rotation matrices, circular motion
/// - **Physics**: Wave interference, standing waves
/// - **Machine learning**: Positional encodings (Transformers)
///
/// # Returns
///
/// `None` if input and output lengths differ.
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

// endregion: Cos

// region: ATan

/// Computes **element-wise arctangent** (inverse tangent) of a vector.
///
/// # Formula
///
/// ```text
/// outputs[i] = atan(inputs[i])
/// ```
///
/// Output values are in **radians**.
///
/// # Properties
///
/// - **Domain**: `(-∞, +∞)`
/// - **Range**: `(-π/2, +π/2)` ≈ `(-1.571, +1.571)`
/// - **Odd function**: `atan(-x) = -atan(x)`
/// - **Limits**: `atan(±∞) = ±π/2`
///
/// # Implementation
///
/// Uses SIMD-accelerated rational polynomial approximation,
/// optimized for the full input range.
///
/// # Use Cases
///
/// - **Computer graphics**: Converting slopes to angles
/// - **Robotics**: Inverse kinematics, angle calculations
/// - **Signal processing**: Phase extraction from I/Q samples
/// - **Navigation**: Bearing calculations
///
/// # Related Functions
///
/// For computing `atan2(y, x)` (angle from coordinates), compute
/// `atan(y/x)` with quadrant correction, or use standard library.
///
/// # Returns
///
/// `None` if input and output lengths differ.
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

// endregion: ATan

// region: Scale

/// Computes **element-wise affine transform** (scale and shift).
///
/// # Formula
///
/// ```text
/// result[i] = α · a[i] + β
/// ```
///
/// # Parameters
///
/// - `alpha`: Multiplicative scaling factor
/// - `beta`: Additive bias/offset
///
/// # Use Cases
///
/// - **Normalization**: `scale(x, 1/std, -mean/std)` for z-score
/// - **Denormalization**: `scale(x, std, mean)` to restore original scale
/// - **Unit conversion**: `scale(celsius, 1.8, 32)` → Fahrenheit
/// - **Contrast adjustment**: `scale(pixels, contrast, brightness)`
///
/// # Common Patterns
///
/// | Operation | Alpha | Beta |
/// |-----------|-------|------|
/// | Negate | -1 | 0 |
/// | Double | 2 | 0 |
/// | Shift by k | 1 | k |
/// | Normalize [0,1]→[-1,1] | 2 | -1 |
///
/// # Performance
///
/// Single fused SIMD operation. More efficient than separate
/// multiply and add operations.
///
/// # Returns
///
/// `None` if input and output lengths differ.
pub trait Scale: Sized {
    /// The scalar type used for alpha and beta parameters.
    type Scalar;
    fn scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl Scale for f64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f64(
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

impl Scale for f32 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f32(
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

impl Scale for f16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f16(
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

impl Scale for bf16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_bf16(
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

impl Scale for i8 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i8(
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

impl Scale for u8 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u8(
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

impl Scale for i16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i16(
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

impl Scale for u16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u16(
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

impl Scale for i32 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i32(
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

impl Scale for u32 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u32(
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

impl Scale for i64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i64(
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

impl Scale for u64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u64(
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

// endregion: Scale

// region: Sum

/// Computes **element-wise addition** of two vectors.
///
/// # Formula
///
/// ```text
/// result[i] = a[i] + b[i]
/// ```
///
/// # Use Cases
///
/// - **Vector addition**: Basic linear algebra operation
/// - **Residual connections**: `output = layer(x) + x` in neural networks
/// - **Blending**: Combining two signals or images
/// - **Gradient accumulation**: Summing gradients across batches
///
/// # Comparison with WSum
///
/// | Operation | Use |
/// |-----------|-----|
/// | `sum(a, b)` | Simple addition, no scaling |
/// | `wsum(a, b, α, β)` | Weighted: `α·a + β·b` |
///
/// Use `sum` when you don't need weights (slightly faster).
///
/// # Returns
///
/// `None` if slice lengths are incompatible.
pub trait Sum: Sized {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()>;
}

impl Sum for f64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for f32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for f16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Sum for bf16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Sum for i8 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u8 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Sum

// region: WSum

/// Computes **element-wise weighted sum** of two vectors.
///
/// # Formula
///
/// ```text
/// result[i] = α · a[i] + β · b[i]
/// ```
///
/// # Parameters
///
/// - `alpha`: Weight for first vector
/// - `beta`: Weight for second vector
///
/// # Use Cases
///
/// - **Linear interpolation**: `wsum(a, b, 1-t, t)` blends a→b as t goes 0→1
/// - **Exponential moving average**: `wsum(old, new, 0.9, 0.1)`
/// - **Gradient descent**: `wsum(weights, gradients, 1, -lr)`
/// - **Convex combination**: When α + β = 1, result lies "between" a and b
///
/// # Common Patterns
///
/// | Operation | Alpha | Beta |
/// |-----------|-------|------|
/// | Lerp at t | 1-t | t |
/// | Average | 0.5 | 0.5 |
/// | Difference | 1 | -1 |
/// | Momentum update | 0.9 | 0.1 |
///
/// # Performance
///
/// Single fused SIMD operation. ~2x faster than naive
/// `for i { result[i] = alpha*a[i] + beta*b[i] }`.
///
/// # Returns
///
/// `None` if slice lengths are incompatible.
pub trait WSum: Sized {
    /// The scalar type used for alpha and beta parameters.
    type Scalar;
    fn wsum(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl WSum for f64 {
    type Scalar = f64;
    fn wsum(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f64(
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

impl WSum for f32 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f32(
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

impl WSum for f16 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f16(
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

impl WSum for bf16 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_bf16(
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

impl WSum for i8 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_i8(
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

impl WSum for u8 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_u8(
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

// endregion: WSum

// region: FMA

/// Computes **fused multiply-add** across three vectors.
///
/// # Formula
///
/// ```text
/// result[i] = α · a[i] · b[i] + β · c[i]
/// ```
///
/// # Parameters
///
/// - `a`, `b`: Vectors to multiply element-wise
/// - `c`: Vector to add (scaled by beta)
/// - `alpha`: Scale factor for product
/// - `beta`: Scale factor for addend
///
/// # Use Cases
///
/// - **Attention mechanism**: `fma(Q, K, bias, scale, 1)` in transformers
/// - **Polynomial evaluation**: Horner's method chains
/// - **Physics**: Force = mass × acceleration + drag
/// - **Neural networks**: Gated activations like `x * sigmoid(x) + residual`
///
/// # Precision Advantage
///
/// True FMA computes `a*b+c` with only one rounding error (not two),
/// giving better numerical accuracy than separate multiply and add.
///
/// # Performance
///
/// Uses hardware FMA instructions (FMA3 on x86, single instruction on ARM).
/// ~1.5x faster than separate multiply + add.
///
/// # Common Patterns
///
/// | Operation | Alpha | Beta |
/// |-----------|-------|------|
/// | a*b + c | 1 | 1 |
/// | a*b - c | 1 | -1 |
/// | -a*b + c | -1 | 1 |
///
/// # Returns
///
/// `None` if slice lengths are incompatible.
pub trait FMA: Sized {
    /// The scalar type used for alpha and beta parameters.
    type Scalar;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl FMA for f64 {
    type Scalar = f64;
    fn fma(
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
            nk_fma_f64(
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

impl FMA for f32 {
    type Scalar = f32;
    fn fma(
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
            nk_fma_f32(
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

impl FMA for f16 {
    type Scalar = f32;
    fn fma(
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
            nk_fma_f16(
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

impl FMA for bf16 {
    type Scalar = f32;
    fn fma(
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
            nk_fma_bf16(
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

impl FMA for i8 {
    type Scalar = f32;
    fn fma(
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
            nk_fma_i8(
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

impl FMA for u8 {
    type Scalar = f32;
    fn fma(
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
            nk_fma_u8(
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

/// `Elementwise` bundles element-wise operations: Scale, Sum, WSum, and FMA.
///
/// Use `numkong::prelude::*` to import all traits at once.
pub trait Elementwise: Scale + Sum + WSum + FMA {}
impl<T: Scale + Sum + WSum + FMA> Elementwise for T {}

/// `Trigonometry` bundles trigonometric functions: Sin, Cos, and ATan.
///
/// Use `numkong::prelude::*` to import all traits at once.
pub trait Trigonometry: Sin + Cos + ATan {}
impl<T: Sin + Cos + ATan> Trigonometry for T {}

// endregion: Convenience Trait Aliases

// region: Prelude

/// The prelude module re-exports all traits for convenient wildcard import.
///
/// # Example
/// ```
/// use numkong::prelude::*;
///
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// let mut result = vec![0.0_f32; 3];
///
/// // All trait methods are now in scope - no turbofish needed!
/// f32::wsum(&a, &b, 0.5, 0.5, &mut result);
/// f32::scale(&a, 2.0, 1.0, &mut result);
/// let angular = f32::angular(&a, &b);
/// ```
pub mod prelude {
    pub use crate::{
        bf16,
        e4m3,
        e5m2,
        f16,
        ATan,
        Angular,
        BinarySimilarity,
        // Complex products
        ComplexDot,
        ComplexProductF32,
        ComplexProductF64,
        ComplexProducts,
        ComplexVDot,
        Cos,
        // Spatial similarity
        Dot,
        Elementwise,
        Euclidean,
        // Binary similarity
        Hamming,
        Jaccard,
        JensenShannon,
        // Probability divergence
        KullbackLeibler,
        // Mesh alignment
        MeshAlignment,
        MeshAlignmentResult,
        ProbabilitySimilarity,
        // Elementwise
        Scale,
        // Trigonometry
        Sin,
        // Sparse
        Sparse,
        SpatialSimilarity,
        Sum,
        Trigonometry,
        WSum,
        FMA,
    };
}

// endregion: Prelude

// region: MeshAlignment

/// Result of mesh alignment operations (RMSD, Kabsch, Umeyama).
///
/// Contains the computed transformation that aligns point cloud A to point cloud B.
/// The transformation follows the convention:
///
/// ```text
/// a'_i = scale * R * (a_i - a_centroid) + b_centroid
/// ```
///
/// # Fields
///
/// * `rotation_matrix` - 3x3 rotation matrix in row-major order
/// * `scale` - Uniform scaling factor (1.0 for RMSD and Kabsch, computed for Umeyama)
/// * `rmsd` - Root mean square deviation after alignment
/// * `a_centroid` - Centroid of the source point cloud A
/// * `b_centroid` - Centroid of the target point cloud B
///
/// # Example
///
/// ```rust
/// use numkong::MeshAlignment;
///
/// let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
///
/// // Kabsch alignment (rigid transformation, scale = 1.0)
/// let result = f64::kabsch(a, b).unwrap();
/// assert!((result.scale - 1.0).abs() < 1e-6);
/// assert!(result.rmsd < 1e-6);
///
/// // Umeyama alignment (similarity transformation with scale)
/// let result = f64::umeyama(a, b).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAlignmentResult<T> {
    /// 3x3 rotation matrix in row-major order.
    pub rotation_matrix: [T; 9],
    /// Uniform scaling factor (1.0 for RMSD/Kabsch, computed for Umeyama).
    pub scale: T,
    /// Root mean square deviation after alignment.
    pub rmsd: T,
    /// Centroid of source point cloud A.
    pub a_centroid: [T; 3],
    /// Centroid of target point cloud B.
    pub b_centroid: [T; 3],
}

impl MeshAlignmentResult<f64> {
    /// Transform a single 3D point using this alignment.
    ///
    /// Applies: `a'_i = scale * R * (a_i - a_centroid) + b_centroid`
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

    /// Transform multiple 3D points using this alignment.
    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

impl MeshAlignmentResult<f32> {
    /// Transform a single 3D point using this alignment.
    ///
    /// Applies: `a'_i = scale * R * (a_i - a_centroid) + b_centroid`
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

    /// Transform multiple 3D points using this alignment.
    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Mesh alignment operations for 3D point clouds.
///
/// This trait provides three alignment methods with increasing generality:
///
/// * [`rmsd`](MeshAlignment::rmsd) - Computes RMSD without finding optimal alignment
/// * [`kabsch`](MeshAlignment::kabsch) - Finds optimal rigid transformation (rotation only, scale = 1.0)
/// * [`umeyama`](MeshAlignment::umeyama) - Finds optimal similarity transformation (rotation + uniform scale)
///
/// All methods accept point clouds as slices of `[T; 3]` arrays (N×3 layout) and return
/// a [`MeshAlignmentResult`] containing the transformation parameters and RMSD.
///
/// # Transformation Convention
///
/// The computed transformation aligns point cloud A to point cloud B:
///
/// ```text
/// a'_i = scale * R * (a_i - a_centroid) + b_centroid
/// ```
///
/// Where:
/// - `R` is the rotation matrix (3×3, row-major)
/// - `scale` is the uniform scaling factor
/// - `a_centroid` and `b_centroid` are the centroids of the two point clouds
///
/// # Example
///
/// ```rust
/// use numkong::MeshAlignment;
///
/// // Two identical point clouds
/// let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
///
/// let result = f64::kabsch(a, b).unwrap();
/// assert!(result.rmsd < 1e-6);
/// ```
pub trait MeshAlignment: Sized {
    /// Compute RMSD between two point clouds without alignment.
    ///
    /// Returns the root mean square deviation between corresponding points
    /// after centering both clouds at the origin. The rotation matrix output
    /// will be the identity matrix and scale will be 1.0.
    ///
    /// # Arguments
    ///
    /// * `a` - Source point cloud (N×3)
    /// * `b` - Target point cloud (N×3)
    ///
    /// # Returns
    ///
    /// `None` if point clouds have different sizes or fewer than 3 points.
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;

    /// Find optimal rigid transformation using the Kabsch algorithm.
    ///
    /// Computes the optimal rotation matrix that minimizes RMSD between
    /// the two point clouds. Scale is fixed at 1.0.
    ///
    /// # Algorithm
    ///
    /// Uses SVD decomposition of the cross-covariance matrix to find the
    /// optimal rotation. Handles reflections by checking the determinant.
    ///
    /// # Arguments
    ///
    /// * `a` - Source point cloud (N×3)
    /// * `b` - Target point cloud (N×3)
    ///
    /// # Returns
    ///
    /// `None` if point clouds have different sizes or fewer than 3 points.
    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;

    /// Find optimal similarity transformation using the Umeyama algorithm.
    ///
    /// Computes the optimal rotation matrix and uniform scaling factor that
    /// minimize RMSD between the two point clouds.
    ///
    /// # Use Cases
    ///
    /// * Protein structure comparison with different scales
    /// * 3D shape matching
    /// * Point cloud registration with scaling
    ///
    /// # Arguments
    ///
    /// * `a` - Source point cloud (N×3)
    /// * `b` - Target point cloud (N×3)
    ///
    /// # Returns
    ///
    /// `None` if point clouds have different sizes or fewer than 3 points.
    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;
}

impl MeshAlignment for f64 {
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }
}

impl MeshAlignment for f32 {
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
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
            );
        }
        Some(result)
    }
}

// endregion: MeshAlignment

// region: NDArray and GEMM

use core::marker::PhantomData;
use core::ptr::NonNull;

/// Maximum number of dimensions supported by NDArray.
pub const MAX_NDIM: usize = 8;

/// Alignment for SIMD-friendly allocations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Memory allocator trait for custom allocation strategies.
///
/// Implement this trait to use custom allocators (arena, pool, etc.) with
/// [`NDArray`] and [`PackedMatrix`].
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

/// Fixed-size shape descriptor for error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeDescriptor {
    dims: [usize; MAX_NDIM],
    ndim: usize,
}

impl ShapeDescriptor {
    /// Create from a slice (truncates if > MAX_NDIM).
    pub fn from_slice(shape: &[usize]) -> Self {
        let mut dims = [0usize; MAX_NDIM];
        let ndim = shape.len().min(MAX_NDIM);
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

/// Error type for NDArray operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NDArrayError {
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
    /// Too many dimensions (exceeds MAX_NDIM).
    TooManyDimensions { got: usize },
}

#[cfg(feature = "std")]
impl std::error::Error for NDArrayError {}

impl core::fmt::Display for NDArrayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NDArrayError::AllocationFailed => write!(f, "memory allocation failed"),
            NDArrayError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, got)
            }
            NDArrayError::InvalidShape { shape, reason } => {
                write!(f, "invalid shape {}: {}", shape, reason)
            }
            NDArrayError::NonContiguousRows => {
                write!(f, "operation requires contiguous rows")
            }
            NDArrayError::DimensionMismatch { expected, got } => {
                write!(f, "expected {} dimensions, got {}", expected, got)
            }
            NDArrayError::IndexOutOfBounds { index, size } => {
                write!(f, "index {} out of bounds for size {}", index, size)
            }
            NDArrayError::TooManyDimensions { got } => {
                write!(f, "too many dimensions: {} (max {})", got, MAX_NDIM)
            }
        }
    }
}

/// Low-level trait for packed GEMM operations.
///
/// Computes C = A × Bᵀ where B is pre-packed for optimal memory access.
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
    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );
}

impl Dots for f32 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_f32f32f32_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_f32f32f32_pack(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_f32f32f32(
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
}

impl Dots for f64 {
    type Accumulator = f64;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_f64f64f64_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_f64f64f64_pack(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_f64f64f64(
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
}

impl Dots for f16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_f16f16f32_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_f16f16f32_pack(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_f16f16f32(
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
}

impl Dots for bf16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_bf16bf16f32_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_bf16bf16f32_pack(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_bf16bf16f32(
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
}

impl Dots for i8 {
    type Accumulator = i32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_i8i8i32_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_i8i8i32_pack(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_i8i8i32(
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
}

impl Dots for u8 {
    type Accumulator = u32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_u8u8u32_packed_size(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_u8u8u32_pack(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_u8u8u32(
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
}

/// N-dimensional array with NumKong-accelerated operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// Supports:
/// - Slicing and subviews (zero-copy)
/// - Matrix multiplication with [`PackedMatrix`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, wsum, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// use numkong::{NDArray, PackedMatrix};
///
/// let a = NDArray::<f32>::try_new(&[1024, 512], 1.0).unwrap();
/// let b = NDArray::<f32>::try_new(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let packed_b = PackedMatrix::try_pack(&b).unwrap();
/// let c = a.matmul(&packed_b);  // Returns (1024 × 256)
/// ```
pub struct NDArray<T, A: Allocator = Global> {
    /// Raw pointer to data buffer.
    data: NonNull<T>,
    /// Total number of elements.
    len: usize,
    /// Shape dimensions.
    shape: [usize; MAX_NDIM],
    /// Strides in bytes.
    strides: [usize; MAX_NDIM],
    /// Number of dimensions.
    ndim: usize,
    /// Allocator instance.
    alloc: A,
}

// Safety: NDArray owns its data and T: Send implies the array is Send
unsafe impl<T: Send, A: Allocator + Send> Send for NDArray<T, A> {}
// Safety: NDArray has no interior mutability, &NDArray<T> is safe to share if T: Sync
unsafe impl<T: Sync, A: Allocator + Sync> Sync for NDArray<T, A> {}

impl<T, A: Allocator> Drop for NDArray<T, A> {
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

impl<T: Clone, A: Allocator + Clone> Clone for NDArray<T, A> {
    fn clone(&self) -> Self {
        Self::try_from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
            .expect("clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<T: Clone, A: Allocator> NDArray<T, A> {
    /// Creates a new NDArray filled with a value using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_new_in(shape: &[usize], value: T, alloc: A) -> Result<Self, NDArrayError> {
        if shape.len() > MAX_NDIM {
            return Err(NDArrayError::TooManyDimensions { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() && shape.iter().any(|&d| d == 0) {
            return Err(NDArrayError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "zero-sized dimension",
            });
        }

        // Allocate raw buffer using our allocator
        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
                .map_err(|_| NDArrayError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(NDArrayError::AllocationFailed)?;
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
        let mut shape_arr = [0usize; MAX_NDIM];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0usize; MAX_NDIM];
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

    /// Creates an NDArray from existing slice data using a custom allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice_in(data: &[T], shape: &[usize], alloc: A) -> Result<Self, NDArrayError> {
        if shape.len() > MAX_NDIM {
            return Err(NDArrayError::TooManyDimensions { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(shape),
                got: ShapeDescriptor::from_slice(&[data.len()]),
            });
        }

        // Allocate and copy using our allocator
        let ptr = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
                .map_err(|_| NDArrayError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(NDArrayError::AllocationFailed)?;
            // Clone all elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..total {
                    core::ptr::write(ptr.add(i), data[i].clone());
                }
                NonNull::new_unchecked(ptr)
            }
        };

        let mut shape_arr = [0usize; MAX_NDIM];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0usize; MAX_NDIM];
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

    fn compute_strides_into(shape: &[usize], strides: &mut [usize; MAX_NDIM]) {
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
impl<T, A: Allocator> NDArray<T, A> {
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
impl<T: Clone> NDArray<T, Global> {
    /// Creates a new NDArray filled with a value using the global allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_new(shape: &[usize], value: T) -> Result<Self, NDArrayError> {
        Self::try_new_in(shape, value, Global)
    }

    /// Creates an NDArray from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[T], shape: &[usize]) -> Result<Self, NDArrayError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn new(shape: &[usize], value: T) -> Self {
        Self::try_new(shape, value).expect("NDArray::new failed")
    }

    /// Convenience constructor that panics on error.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self {
        Self::try_from_slice(data, shape).expect("NDArray::from_slice failed")
    }
}

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

/// A read-only view into an NDArray (doesn't own data).
///
/// Views provide zero-copy access to array subregions with potentially
/// different strides than the original array.
pub struct NDArrayView<'a, T> {
    /// Pointer to first element of view.
    data: *const T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_NDIM],
    /// Strides in bytes.
    strides: [usize; MAX_NDIM],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a T>,
}

impl<'a, T> NDArrayView<'a, T> {
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

impl<'a, T: Clone> NDArrayView<'a, T> {
    /// Copy the view contents to a new owned NDArray.
    pub fn to_owned(&self) -> Result<NDArray<T>, NDArrayError> {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.len) };
            NDArray::try_from_slice(slice, self.shape())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = NDArray::try_new(self.shape(), unsafe { (*self.data).clone() })?;
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
            let mut indices = [0usize; MAX_NDIM];
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

/// A mutable view into an NDArray.
pub struct NDArrayViewMut<'a, T> {
    /// Pointer to first element of view.
    data: *mut T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_NDIM],
    /// Strides in bytes.
    strides: [usize; MAX_NDIM],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> NDArrayViewMut<'a, T> {
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
    pub fn as_view(&self) -> NDArrayView<'_, T> {
        NDArrayView {
            data: self.data,
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone> NDArray<T> {
    /// Create a view of the entire array.
    pub fn view(&self) -> NDArrayView<'_, T> {
        NDArrayView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Create a mutable view of the entire array.
    pub fn view_mut(&mut self) -> NDArrayViewMut<'_, T> {
        NDArrayViewMut {
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
    /// use numkong::{NDArray, SliceRange};
    ///
    /// let arr = NDArray::<f32>::try_new(&[4, 5], 1.0).unwrap();
    ///
    /// // Get rows 0..2, all columns
    /// let view = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    ///
    /// // Get row 1 (reduces to 1D)
    /// let row = arr.slice(&[SliceRange::index(1), SliceRange::full()]).unwrap();
    /// ```
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<NDArrayView<'_, T>, NDArrayError> {
        if ranges.len() != self.ndim {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_NDIM];
        let mut new_strides = [0usize; MAX_NDIM];
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
                        return Err(NDArrayError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    // Single index reduces dimension (doesn't add to new shape)
                    offset += i * dim_stride;
                }
                SliceRange::Range { start, end } => {
                    if start > end || end > dim_size {
                        return Err(NDArrayError::IndexOutOfBounds {
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
                        return Err(NDArrayError::IndexOutOfBounds {
                            index: if start >= dim_size { start } else { end },
                            size: dim_size,
                        });
                    }
                    if step == 0 {
                        return Err(NDArrayError::InvalidShape {
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

        Ok(NDArrayView {
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
    ) -> Result<NDArrayViewMut<'_, T>, NDArrayError> {
        if ranges.len() != self.ndim {
            return Err(NDArrayError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_NDIM];
        let mut new_strides = [0usize; MAX_NDIM];
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
                        return Err(NDArrayError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    offset += i * dim_stride;
                }
                SliceRange::Range { start, end } => {
                    if start > end || end > dim_size {
                        return Err(NDArrayError::IndexOutOfBounds {
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
                        return Err(NDArrayError::IndexOutOfBounds {
                            index: if start >= dim_size { start } else { end },
                            size: dim_size,
                        });
                    }
                    if step == 0 {
                        return Err(NDArrayError::InvalidShape {
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

        Ok(NDArrayViewMut {
            data: new_ptr,
            len: new_len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        })
    }

    /// Transpose a 2D array (swaps strides, no data copy).
    pub fn t(&self) -> Result<NDArrayView<'_, T>, NDArrayError> {
        if self.ndim != 2 {
            return Err(NDArrayError::DimensionMismatch {
                expected: 2,
                got: self.ndim,
            });
        }

        let mut new_shape = [0usize; MAX_NDIM];
        let mut new_strides = [0usize; MAX_NDIM];
        new_shape[0] = self.shape[1];
        new_shape[1] = self.shape[0];
        new_strides[0] = self.strides[1];
        new_strides[1] = self.strides[0];

        Ok(NDArrayView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: 2,
            _marker: PhantomData,
        })
    }

    /// Reshape the array (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<NDArrayView<'_, T>, NDArrayError> {
        if new_shape.len() > MAX_NDIM {
            return Err(NDArrayError::TooManyDimensions {
                got: new_shape.len(),
            });
        }

        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(new_shape),
                got: ShapeDescriptor::from_slice(self.shape()),
            });
        }

        // Check if currently contiguous
        if !self.view().is_contiguous() {
            return Err(NDArrayError::NonContiguousRows);
        }

        let mut shape_arr = [0usize; MAX_NDIM];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);

        let mut strides_arr = [0usize; MAX_NDIM];
        NDArray::<T>::compute_strides_into(new_shape, &mut strides_arr);

        Ok(NDArrayView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }
}

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
/// let packed_b = PackedMatrix::try_pack(&b_array).unwrap();
/// let c = a_array.matmul(&packed_b);
/// ```
///
/// For C = A × B where B is (k × n) (standard GEMM layout):
/// ```rust,ignore
/// let packed_b = PackedMatrix::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.matmul(&packed_b);
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
    pub fn try_pack_in<BA: Allocator>(b: &NDArray<T, BA>, alloc: A) -> Result<Self, NDArrayError> {
        if b.ndim() != 2 {
            return Err(NDArrayError::DimensionMismatch {
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
                .map_err(|_| NDArrayError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(NDArrayError::AllocationFailed)?;
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
    pub fn try_pack_transposed_in<BA: Allocator>(
        b: &NDArray<T, BA>,
        alloc: A,
    ) -> Result<Self, NDArrayError> {
        if b.ndim() != 2 {
            return Err(NDArrayError::DimensionMismatch {
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
                .map_err(|_| NDArrayError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(NDArrayError::AllocationFailed)?;
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
    pub fn try_pack<BA: Allocator>(b: &NDArray<T, BA>) -> Result<Self, NDArrayError> {
        Self::try_pack_in(b, Global)
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using the global allocator.
    ///
    /// Result computes: C = A × B
    pub fn try_pack_transposed<BA: Allocator>(b: &NDArray<T, BA>) -> Result<Self, NDArrayError> {
        Self::try_pack_transposed_in(b, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<BA: Allocator>(b: &NDArray<T, BA>) -> Self {
        Self::try_pack(b).expect("PackedMatrix::pack failed")
    }

    /// Convenience constructor that panics on error.
    pub fn pack_transposed<BA: Allocator>(b: &NDArray<T, BA>) -> Self {
        Self::try_pack_transposed(b).expect("PackedMatrix::pack_transposed failed")
    }
}

impl<T: Dots, A: Allocator + Clone> NDArray<T, A>
where
    T::Accumulator: Clone + Default,
{
    /// Matrix multiply: C = self × packed_bᵀ
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
    pub fn try_matmul<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> Result<NDArray<T::Accumulator, A>, NDArrayError> {
        if self.ndim() != 2 {
            return Err(NDArrayError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(NDArrayError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }

        let mut c = NDArray::try_new_in(&[m, n], T::Accumulator::default(), self.alloc.clone())?;
        unsafe {
            T::dots(
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
    pub fn matmul<BA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
    ) -> NDArray<T::Accumulator, A> {
        self.try_matmul(packed_b).expect("matmul failed")
    }
}

impl<T: Dots, A: Allocator> NDArray<T, A> {
    /// Matrix multiply into existing output (avoids allocation).
    pub fn try_matmul_into<BA: Allocator, CA: Allocator>(
        &self,
        packed_b: &PackedMatrix<T, BA>,
        c: &mut NDArray<T::Accumulator, CA>,
    ) -> Result<(), NDArrayError> {
        if self.ndim() != 2 {
            return Err(NDArrayError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(NDArrayError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(NDArrayError::NonContiguousRows);
        }

        unsafe {
            T::dots(
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

// region: NDArray Elementwise Operations

impl<T: Clone + Scale> NDArray<T> {
    /// Apply element-wise scale: result[i] = alpha * self[i] + beta
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(&self, alpha: T::Scalar, beta: T::Scalar) -> Result<NDArray<T>, NDArrayError> {
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::scale(self.as_slice(), alpha, beta, result.as_mut_slice());
        Ok(result)
    }

    /// Apply element-wise scale in-place: self[i] = alpha * self[i] + beta
    pub fn scale_inplace(&mut self, alpha: T::Scalar, beta: T::Scalar) {
        // Need a temporary for in-place operation since input and output overlap
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::scale(slice, alpha, beta, self.as_mut_slice());
        }
    }
}

impl<T: Clone + Sum> NDArray<T> {
    /// Element-wise sum: result[i] = self[i] + other[i]
    ///
    /// Returns a new array with the summed values.
    pub fn add(&self, other: &NDArray<T>) -> Result<NDArray<T>, NDArrayError> {
        if self.shape() != other.shape() {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::sum(self.as_slice(), other.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sum in-place: self[i] = self[i] + other[i]
    pub fn add_inplace(&mut self, other: &NDArray<T>) -> Result<(), NDArrayError> {
        if self.shape() != other.shape() {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::sum(slice, other.as_slice(), self.as_mut_slice());
        }
        Ok(())
    }
}

impl<T: Clone + WSum> NDArray<T> {
    /// Weighted sum: result[i] = alpha * self[i] + beta * other[i]
    ///
    /// Returns a new array with the weighted sum.
    pub fn wsum(
        &self,
        other: &NDArray<T>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<NDArray<T>, NDArrayError> {
        if self.shape() != other.shape() {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::wsum(
            self.as_slice(),
            other.as_slice(),
            alpha,
            beta,
            result.as_mut_slice(),
        );
        Ok(result)
    }
}

impl<T: Clone + FMA> NDArray<T> {
    /// Fused multiply-add: result[i] = alpha * self[i] * b[i] + beta * c[i]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma(
        &self,
        b: &NDArray<T>,
        c: &NDArray<T>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<NDArray<T>, NDArrayError> {
        if self.shape() != b.shape() || self.shape() != c.shape() {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(b.shape()),
            });
        }
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::fma(
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

// endregion: NDArray Elementwise Operations

// region: NDArray Trigonometry

impl<T: Clone + Sin> NDArray<T> {
    /// Element-wise sine: result[i] = sin(self[i])
    ///
    /// Input values are in radians.
    pub fn sin(&self) -> Result<NDArray<T>, NDArrayError> {
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::sin(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sine in-place: self[i] = sin(self[i])
    pub fn sin_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::sin(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + Cos> NDArray<T> {
    /// Element-wise cosine: result[i] = cos(self[i])
    ///
    /// Input values are in radians.
    pub fn cos(&self) -> Result<NDArray<T>, NDArrayError> {
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::cos(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise cosine in-place: self[i] = cos(self[i])
    pub fn cos_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::cos(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + ATan> NDArray<T> {
    /// Element-wise arctangent: result[i] = atan(self[i])
    ///
    /// Output values are in radians in the range (-π/2, π/2).
    pub fn atan(&self) -> Result<NDArray<T>, NDArrayError> {
        let mut result = NDArray::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::atan(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise arctangent in-place: self[i] = atan(self[i])
    pub fn atan_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::atan(slice, self.as_mut_slice());
        }
    }
}

// endregion: NDArray Trigonometry

// region: NDArray Reductions

impl<T: Clone + Dot> NDArray<T> {
    /// Compute the dot product of this array with another.
    ///
    /// Both arrays must be 1D with the same length.
    pub fn dot_product(&self, other: &NDArray<T>) -> Result<T::Output, NDArrayError> {
        if self.ndim != 1 || other.ndim != 1 {
            return Err(NDArrayError::DimensionMismatch {
                expected: 1,
                got: if self.ndim != 1 {
                    self.ndim
                } else {
                    other.ndim
                },
            });
        }
        if self.len != other.len {
            return Err(NDArrayError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        // Dot::dot returns Option, unwrap since we verified lengths match
        Ok(T::dot(self.as_slice(), other.as_slice()).expect("dot product failed"))
    }
}

impl NDArray<f32> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f32 {
        let ones = NDArray::try_new(self.shape(), 1.0f32).expect("allocation failed");
        <f32 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

impl NDArray<f64> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f64 {
        let ones = NDArray::try_new(self.shape(), 1.0f64).expect("allocation failed");
        <f64 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

// endregion: NDArray Reductions

// endregion: NDArray and GEMM

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16 as HalfBF16;
    use half::f16 as HalfF16;

    trait IntoF64 {
        fn into_f64(self) -> f64;
    }
    impl IntoF64 for f64 {
        fn into_f64(self) -> f64 {
            self
        }
    }
    impl IntoF64 for f32 {
        fn into_f64(self) -> f64 {
            self as f64
        }
    }
    impl IntoF64 for i32 {
        fn into_f64(self) -> f64 {
            self as f64
        }
    }
    impl IntoF64 for u32 {
        fn into_f64(self) -> f64 {
            self as f64
        }
    }

    fn assert_almost_equal<T: IntoF64>(left: f64, right: T, tolerance: f64) {
        let right = right.into_f64();
        let lower = right - tolerance;
        let upper = right + tolerance;
        assert!(left >= lower && left <= upper);
    }

    fn assert_vec_almost_equal_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tolerance,
                "Element {}: expected {} but got {}, diff {} > tolerance {}",
                i,
                e,
                a,
                (a - e).abs(),
                tolerance
            );
        }
    }

    fn assert_vec_almost_equal_f64(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tolerance,
                "Element {}: expected {} but got {}, diff {} > tolerance {}",
                i,
                e,
                a,
                (a - e).abs(),
                tolerance
            );
        }
    }

    // Hardware detection test
    #[test]
    fn hardware_features_detection() {
        let uses_arm = capabilities::uses_neon() || capabilities::uses_sve();
        let uses_x86 = capabilities::uses_haswell()
            || capabilities::uses_skylake()
            || capabilities::uses_ice()
            || capabilities::uses_genoa()
            || capabilities::uses_sapphire()
            || capabilities::uses_turin();

        if uses_arm {
            assert!(!uses_x86);
        }
        if uses_x86 {
            assert!(!uses_arm);
        }

        println!("- uses_neon: {}", capabilities::uses_neon());
        println!("- uses_neonhalf: {}", capabilities::uses_neonhalf());
        println!("- uses_neonbfdot: {}", capabilities::uses_neonbfdot());
        println!("- uses_neonsdot: {}", capabilities::uses_neonsdot());
        println!("- uses_sve: {}", capabilities::uses_sve());
        println!("- uses_svehalf: {}", capabilities::uses_svehalf());
        println!("- uses_svebfdot: {}", capabilities::uses_svebfdot());
        println!("- uses_svesdot: {}", capabilities::uses_svesdot());
        println!("- uses_haswell: {}", capabilities::uses_haswell());
        println!("- uses_skylake: {}", capabilities::uses_skylake());
        println!("- uses_ice: {}", capabilities::uses_ice());
        println!("- uses_genoa: {}", capabilities::uses_genoa());
        println!("- uses_sapphire: {}", capabilities::uses_sapphire());
        println!("- uses_turin: {}", capabilities::uses_turin());
        println!("- uses_sierra: {}", capabilities::uses_sierra());
    }

    // Dot product tests
    #[test]
    fn dot_i8() {
        let a = &[1_i8, 2, 3];
        let b = &[4_i8, 5, 6];
        if let Some(result) = i8::dot(a, b) {
            assert_almost_equal(32.0, result, 0.01);
        }
    }

    #[test]
    fn dot_f32() {
        let a = &[1.0_f32, 2.0, 3.0];
        let b = &[4.0_f32, 5.0, 6.0];
        if let Some(result) = <f32 as Dot>::dot(a, b) {
            assert_almost_equal(32.0, result, 0.01);
        }
    }

    // Angular distance tests
    #[test]
    fn cos_i8() {
        let a = &[3_i8, 97, 127];
        let b = &[3_i8, 97, 127];
        if let Some(result) = i8::angular(a, b) {
            assert_almost_equal(0.00012027938, result, 0.01);
        }
    }

    #[test]
    fn cos_f32() {
        let a = &[1.0_f32, 2.0, 3.0];
        let b = &[4.0_f32, 5.0, 6.0];
        if let Some(result) = f32::angular(a, b) {
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn cos_f16_same() {
        let a_u16: &[u16] = &[15360, 16384, 17408];
        let b_u16: &[u16] = &[15360, 16384, 17408];
        let a_f16: &[f16] =
            unsafe { core::slice::from_raw_parts(a_u16.as_ptr() as *const f16, a_u16.len()) };
        let b_f16: &[f16] =
            unsafe { core::slice::from_raw_parts(b_u16.as_ptr() as *const f16, b_u16.len()) };
        if let Some(result) = f16::angular(a_f16, b_f16) {
            assert_almost_equal(0.0, result, 0.01);
        }
    }

    #[test]
    fn cos_bf16_same() {
        let a_u16: &[u16] = &[15360, 16384, 17408];
        let b_u16: &[u16] = &[15360, 16384, 17408];
        let a_bf16: &[bf16] =
            unsafe { core::slice::from_raw_parts(a_u16.as_ptr() as *const bf16, a_u16.len()) };
        let b_bf16: &[bf16] =
            unsafe { core::slice::from_raw_parts(b_u16.as_ptr() as *const bf16, b_u16.len()) };
        if let Some(result) = bf16::angular(a_bf16, b_bf16) {
            assert_almost_equal(0.0, result, 0.01);
        }
    }

    #[test]
    fn cos_f16_interop() {
        let a_half: Vec<HalfF16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();
        let b_half: Vec<HalfF16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();
        let a_numkong: &[f16] =
            unsafe { core::slice::from_raw_parts(a_half.as_ptr() as *const f16, a_half.len()) };
        let b_numkong: &[f16] =
            unsafe { core::slice::from_raw_parts(b_half.as_ptr() as *const f16, b_half.len()) };
        if let Some(result) = f16::angular(a_numkong, b_numkong) {
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn cos_bf16_interop() {
        let a_half: Vec<HalfBF16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| HalfBF16::from_f32(x))
            .collect();
        let b_half: Vec<HalfBF16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| HalfBF16::from_f32(x))
            .collect();
        let a_numkong: &[bf16] =
            unsafe { core::slice::from_raw_parts(a_half.as_ptr() as *const bf16, a_half.len()) };
        let b_numkong: &[bf16] =
            unsafe { core::slice::from_raw_parts(b_half.as_ptr() as *const bf16, b_half.len()) };
        if let Some(result) = bf16::angular(a_numkong, b_numkong) {
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    // Euclidean distance tests
    #[test]
    fn l2sq_i8() {
        let a = &[1_i8, 2, 3];
        let b = &[4_i8, 5, 6];
        if let Some(result) = i8::l2sq(a, b) {
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    #[test]
    fn l2sq_f32() {
        let a = &[1.0_f32, 2.0, 3.0];
        let b = &[4.0_f32, 5.0, 6.0];
        if let Some(result) = f32::l2sq(a, b) {
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    #[test]
    fn l2_f32() {
        let a: &[f32; 3] = &[1.0, 2.0, 3.0];
        let b: &[f32; 3] = &[4.0, 5.0, 6.0];
        if let Some(result) = f32::l2(a, b) {
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    #[test]
    fn l2_f64() {
        let a: &[f64; 3] = &[1.0, 2.0, 3.0];
        let b: &[f64; 3] = &[4.0, 5.0, 6.0];
        if let Some(result) = f64::l2(a, b) {
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    #[test]
    fn l2_f16() {
        let a_half: Vec<HalfF16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();
        let b_half: Vec<HalfF16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();
        let a_numkong: &[f16] =
            unsafe { core::slice::from_raw_parts(a_half.as_ptr() as *const f16, a_half.len()) };
        let b_numkong: &[f16] =
            unsafe { core::slice::from_raw_parts(b_half.as_ptr() as *const f16, b_half.len()) };
        if let Some(result) = f16::l2(a_numkong, b_numkong) {
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    #[test]
    fn l2_i8() {
        let a = &[1_i8, 2, 3];
        let b = &[4_i8, 5, 6];
        if let Some(result) = i8::l2(a, b) {
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    // Binary similarity tests
    #[test]
    fn hamming_u8() {
        let a = &[0b01010101_u8, 0b11110000, 0b10101010];
        let b = &[0b01010101_u8, 0b11110000, 0b10101010];
        if let Some(result) = u8::hamming(a, b) {
            assert_almost_equal(0.0, result, 0.01);
        }
    }

    #[test]
    fn jaccard_u8() {
        let a = &[0b11110000_u8, 0b00001111, 0b10101010];
        let b = &[0b11110000_u8, 0b00001111, 0b01010101];
        if let Some(result) = u8::jaccard(a, b) {
            assert_almost_equal(0.5, result, 0.01);
        }
    }

    // Probability divergence tests
    #[test]
    fn js_f32() {
        let a: &[f32; 3] = &[0.1, 0.9, 0.0];
        let b: &[f32; 3] = &[0.2, 0.8, 0.0];
        if let Some(result) = f32::jensenshannon(a, b) {
            assert_almost_equal(0.099, result, 0.01);
        }
    }

    #[test]
    fn kl_f32() {
        let a: &[f32; 3] = &[0.1, 0.9, 0.0];
        let b: &[f32; 3] = &[0.2, 0.8, 0.0];
        if let Some(result) = f32::kullbackleibler(a, b) {
            assert_almost_equal(0.036, result, 0.01);
        }
    }

    // Complex product tests
    #[test]
    fn dot_f32_complex() {
        let a: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0];
        let b: &[f32; 4] = &[5.0, 6.0, 7.0, 8.0];
        if let Some((real, imag)) = <f32 as ComplexDot>::dot(a, b) {
            assert_almost_equal(-18.0, real, 0.01);
            assert_almost_equal(68.0, imag, 0.01);
        }
    }

    #[test]
    fn vdot_f32_complex() {
        let a: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0];
        let b: &[f32; 4] = &[5.0, 6.0, 7.0, 8.0];
        if let Some((real, imag)) = f32::vdot(a, b) {
            assert_almost_equal(70.0, real, 0.01);
            assert_almost_equal(-8.0, imag, 0.01);
        }
    }

    // Sparse intersection tests
    #[test]
    fn intersect_u16() {
        {
            let a_u16: &[u16] = &[153, 16384, 17408];
            let b_u16: &[u16] = &[7408, 15360, 16384];
            if let Some(result) = u16::intersect(a_u16, b_u16) {
                assert_almost_equal(1.0, result, 0.0001);
            }
        }
        {
            let a_u16: &[u16] = &[8, 153, 11638];
            let b_u16: &[u16] = &[7408, 15360, 16384];
            if let Some(result) = u16::intersect(a_u16, b_u16) {
                assert_almost_equal(0.0, result, 0.0001);
            }
        }
    }

    #[test]
    fn intersect_u32() {
        {
            let a_u32: &[u32] = &[11, 153];
            let b_u32: &[u32] = &[11, 153, 7408, 16384];
            if let Some(result) = u32::intersect(a_u32, b_u32) {
                assert_almost_equal(2.0, result, 0.0001);
            }
        }
        {
            let a_u32: &[u32] = &[153, 7408, 11638];
            let b_u32: &[u32] = &[153, 7408, 11638];
            if let Some(result) = u32::intersect(a_u32, b_u32) {
                assert_almost_equal(3.0, result, 0.0001);
            }
        }
    }

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
                let result =
                    u32::intersect(array_a.as_slice(), array_b.as_slice()).unwrap() as usize;
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
                let result =
                    u16::intersect(array_a.as_slice(), array_b.as_slice()).unwrap() as usize;
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
        assert_eq!(u32::intersect(empty, empty), Some(0u32));
        assert_eq!(u32::intersect(empty, non_empty), Some(0u32));
        assert_eq!(u32::intersect(non_empty, empty), Some(0u32));

        assert_eq!(u32::intersect(&[42u32], &[42u32]), Some(1u32));
        assert_eq!(u32::intersect(&[42u32], &[43u32]), Some(0u32));

        let a: &[u32] = &[1, 2, 3, 4, 5];
        let b: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::intersect(a, b), Some(0u32));

        let c: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::intersect(c, c), Some(5u32));

        let boundary_16: Vec<u32> = (0..16).collect();
        let boundary_32: Vec<u32> = (0..32).collect();
        let boundary_64: Vec<u32> = (0..64).collect();
        assert_eq!(u32::intersect(&boundary_16, &boundary_16), Some(16u32));
        assert_eq!(u32::intersect(&boundary_32, &boundary_32), Some(32u32));
        assert_eq!(u32::intersect(&boundary_64, &boundary_64), Some(64u32));

        let first_half: Vec<u32> = (0..32).collect();
        let second_half: Vec<u32> = (16..48).collect();
        assert_eq!(u32::intersect(&first_half, &second_half), Some(16u32));
    }

    // Numeric type tests
    #[test]
    fn f16_arithmetic() {
        let a = f16::from_f32(3.5);
        let b = f16::from_f32(2.0);

        assert!((a + b).to_f32() - 5.5 < 0.01);
        assert!((a - b).to_f32() - 1.5 < 0.01);
        assert!((a * b).to_f32() - 7.0 < 0.01);
        assert!((a / b).to_f32() - 1.75 < 0.01);
        assert!((-a).to_f32() + 3.5 < 0.01);

        assert!(f16::ZERO.to_f32() == 0.0);
        assert!((f16::ONE.to_f32() - 1.0).abs() < 0.01);
        assert!((f16::NEG_ONE.to_f32() + 1.0).abs() < 0.01);

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert!((-a).abs().to_f32() - 3.5 < 0.01);
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn bf16_arithmetic() {
        let a = bf16::from_f32(3.5);
        let b = bf16::from_f32(2.0);

        assert!((a + b).to_f32() - 5.5 < 0.1);
        assert!((a - b).to_f32() - 1.5 < 0.1);
        assert!((a * b).to_f32() - 7.0 < 0.1);
        assert!((a / b).to_f32() - 1.75 < 0.1);
        assert!((-a).to_f32() + 3.5 < 0.1);

        assert!(bf16::ZERO.to_f32() == 0.0);
        assert!((bf16::ONE.to_f32() - 1.0).abs() < 0.01);
        assert!((bf16::NEG_ONE.to_f32() + 1.0).abs() < 0.01);

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert!((-a).abs().to_f32() - 3.5 < 0.1);
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
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
    fn e4m3_arithmetic() {
        let a = e4m3::from_f32(2.0);
        let b = e4m3::from_f32(1.5);

        assert!((a + b).to_f32() - 3.5 < 0.5);
        assert!((a - b).to_f32() - 0.5 < 0.5);
        assert!((a * b).to_f32() - 3.0 < 0.5);
        assert!((a / b).to_f32() - 1.333 < 0.5);
        assert!((-a).to_f32() + 2.0 < 0.1);

        assert!(e4m3::ZERO.to_f32() == 0.0);
        assert!((e4m3::ONE.to_f32() - 1.0).abs() < 0.1);
        assert!((e4m3::NEG_ONE.to_f32() + 1.0).abs() < 0.1);

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert!((-a).abs().to_f32() - 2.0 < 0.1);
        assert!(a.is_finite());
        assert!(!a.is_nan());
    }

    #[test]
    fn e5m2_arithmetic() {
        let a = e5m2::from_f32(2.0);
        let b = e5m2::from_f32(1.5);

        assert!((a + b).to_f32() - 3.5 < 0.5);
        assert!((a - b).to_f32() - 0.5 < 0.5);
        assert!((a * b).to_f32() - 3.0 < 0.5);
        assert!((a / b).to_f32() - 1.333 < 0.5);
        assert!((-a).to_f32() + 2.0 < 0.1);

        assert!(e5m2::ZERO.to_f32() == 0.0);
        assert!((e5m2::ONE.to_f32() - 1.0).abs() < 0.1);
        assert!((e5m2::NEG_ONE.to_f32() + 1.0).abs() < 0.1);

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert!((-a).abs().to_f32() - 2.0 < 0.1);
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn e4m3_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 128.0, 224.0,
        ];
        for &val in &test_values {
            let fp8 = e4m3::from_f32(val);
            let roundtrip = fp8.to_f32();
            if val != 0.0 {
                let rel_error = ((roundtrip - val) / val).abs();
                assert!(
                    rel_error < 0.5,
                    "e4m3 roundtrip failed for {}: got {}",
                    val,
                    roundtrip
                );
            } else {
                assert_eq!(roundtrip, 0.0);
            }
        }
    }

    #[test]
    fn e5m2_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0,
        ];
        for &val in &test_values {
            let fp8 = e5m2::from_f32(val);
            let roundtrip = fp8.to_f32();
            if val != 0.0 {
                let rel_error = ((roundtrip - val) / val).abs();
                assert!(
                    rel_error < 0.5,
                    "e5m2 roundtrip failed for {}: got {}",
                    val,
                    roundtrip
                );
            } else {
                assert_eq!(roundtrip, 0.0);
            }
        }
    }

    // Trigonometry tests
    #[test]
    fn sin_f32_small() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..11).map(|i| (i as f32) * PI / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.sin()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Sin>::sin(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn sin_f32_medium() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..97).map(|i| (i as f32) * 2.0 * PI / 97.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.sin()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Sin>::sin(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn sin_f64_test() {
        use core::f64::consts::PI;
        let inputs: Vec<f64> = (0..97).map(|i| (i as f64) * 2.0 * PI / 97.0).collect();
        let expected: Vec<f64> = inputs.iter().map(|x| x.sin()).collect();
        let mut result = vec![0.0f64; inputs.len()];
        <f64 as Sin>::sin(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    #[test]
    fn cos_f32_test() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..97).map(|i| (i as f32) * 2.0 * PI / 97.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.cos()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Cos>::cos(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn cos_f64_test() {
        use core::f64::consts::PI;
        let inputs: Vec<f64> = (0..97).map(|i| (i as f64) * 2.0 * PI / 97.0).collect();
        let expected: Vec<f64> = inputs.iter().map(|x| x.cos()).collect();
        let mut result = vec![0.0f64; inputs.len()];
        <f64 as Cos>::cos(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    #[test]
    fn atan_f32_test() {
        let inputs: Vec<f32> = (-50..50).map(|i| (i as f32) / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.atan()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as ATan>::atan(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn atan_f64_test() {
        let inputs: Vec<f64> = (-50..50).map(|i| (i as f64) / 10.0).collect();
        let expected: Vec<f64> = inputs.iter().map(|x| x.atan()).collect();
        let mut result = vec![0.0f64; inputs.len()];
        <f64 as ATan>::atan(&inputs, &mut result).unwrap();
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    // Scale tests
    #[test]
    fn scale_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 2.0_f32;
        let beta = 1.0_f32;
        let mut result = vec![0.0f32; a.len()];
        f32::scale(&a, alpha, beta, &mut result).unwrap();
        let expected: Vec<f32> = a.iter().map(|x| alpha * x + beta).collect();
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn scale_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 2.0_f64;
        let beta = 1.0_f64;
        let mut result = vec![0.0f64; a.len()];
        f64::scale(&a, alpha, beta, &mut result).unwrap();
        let expected: Vec<f64> = a.iter().map(|x| alpha * x + beta).collect();
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    #[test]
    fn scale_i32() {
        let a: Vec<i32> = vec![1, 2, 3, 4, 5];
        let alpha = 2.0_f64;
        let beta = 1.0_f64;
        let mut result = vec![0i32; a.len()];
        i32::scale(&a, alpha, beta, &mut result).unwrap();
        for (i, &r) in result.iter().enumerate() {
            let expected = (alpha * a[i] as f64 + beta).round() as i32;
            assert!(
                (r - expected).abs() <= 1,
                "Element {}: expected {} but got {}",
                i,
                expected,
                r
            );
        }
    }

    #[test]
    fn scale_f16_test() {
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let alpha = 2.0_f32;
        let beta = 1.0_f32;
        let mut result = vec![f16::ZERO; a.len()];
        f16::scale(&a, alpha, beta, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha * (i + 1) as f32 + beta;
            assert!(
                (r.to_f32() - expected).abs() < 0.2,
                "Element {}: expected {} but got {}",
                i,
                expected,
                r.to_f32()
            );
        }
    }

    // Sum tests
    #[test]
    fn sum_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        let mut result = vec![0.0f32; a.len()];
        f32::sum(&a, &b, &mut result).unwrap();
        let expected: Vec<f32> = vec![5.0, 7.0, 9.0];
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn sum_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let mut result = vec![0.0f64; a.len()];
        f64::sum(&a, &b, &mut result).unwrap();
        let expected: Vec<f64> = vec![5.0, 7.0, 9.0];
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    #[test]
    fn sum_length_mismatch() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0];
        let mut result = vec![0.0f32; a.len()];
        assert!(f32::sum(&a, &b, &mut result).is_none());
    }

    // WSum tests
    #[test]
    fn wsum_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        let alpha = 0.5;
        let beta = 0.5;
        let mut result = vec![0.0f32; a.len()];
        f32::wsum(&a, &b, alpha, beta, &mut result).unwrap();
        let expected: Vec<f32> = vec![2.5, 3.5, 4.5];
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn wsum_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let alpha = 0.5;
        let beta = 0.5;
        let mut result = vec![0.0f64; a.len()];
        f64::wsum(&a, &b, alpha, beta, &mut result).unwrap();
        let expected: Vec<f64> = vec![2.5, 3.5, 4.5];
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    // FMA tests
    #[test]
    fn fma_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![2.0, 3.0, 4.0];
        let c: Vec<f32> = vec![1.0, 1.0, 1.0];
        let alpha = 1.0;
        let beta = 1.0;
        let mut result = vec![0.0f32; a.len()];
        f32::fma(&a, &b, &c, alpha, beta, &mut result).unwrap();
        let expected: Vec<f32> = vec![3.0, 7.0, 13.0];
        assert_vec_almost_equal_f32(&result, &expected, 0.1);
    }

    #[test]
    fn fma_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![2.0, 3.0, 4.0];
        let c: Vec<f64> = vec![1.0, 1.0, 1.0];
        let alpha = 1.0;
        let beta = 1.0;
        let mut result = vec![0.0f64; a.len()];
        f64::fma(&a, &b, &c, alpha, beta, &mut result).unwrap();
        let expected: Vec<f64> = vec![3.0, 7.0, 13.0];
        assert_vec_almost_equal_f64(&result, &expected, 0.1);
    }

    // Large vector tests
    #[test]
    fn large_vector_scale() {
        let a: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let alpha = 2.0;
        let beta = 0.5;
        let mut result = vec![0.0f32; a.len()];
        f32::scale(&a, alpha, beta, &mut result).unwrap();
        assert_eq!(result.len(), 1536);
        for i in 0..1536 {
            let expected = alpha as f32 * a[i] + beta as f32;
            assert!(
                (result[i] - expected).abs() < 0.1,
                "Element {}: expected {} but got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn large_vector_sum() {
        let a: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32) * 2.0).collect();
        let mut result = vec![0.0f32; a.len()];
        f32::sum(&a, &b, &mut result).unwrap();
        assert_eq!(result.len(), 1536);
        for i in 0..1536 {
            let expected = a[i] + b[i];
            assert!(
                (result[i] - expected).abs() < 0.1,
                "Element {}: expected {} but got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    // MeshAlignment tests

    #[test]
    fn kabsch_f64_identical_points() {
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = f64::kabsch(a, b).unwrap();

        // Scale should be 1.0 for Kabsch
        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD should be ~0 for identical points
        assert!(result.rmsd < 1e-6, "Expected RMSD ~0, got {}", result.rmsd);
    }

    #[test]
    fn kabsch_f32_identical_points() {
        let a: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = f32::kabsch(a, b).unwrap();

        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD should be ~0 for identical points
        assert!(result.rmsd < 1e-4, "Expected RMSD ~0, got {}", result.rmsd);
    }

    #[test]
    fn umeyama_f64_scaled_points() {
        // B is 2x scaled version of A
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f64; 3]] = &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let result = f64::umeyama(a, b).unwrap();

        // Scale should be ~2.0 (transforming A to B)
        assert!(
            (result.scale - 2.0).abs() < 0.1,
            "Expected scale ~2.0, got {}",
            result.scale
        );
    }

    #[test]
    fn rmsd_f64_basic() {
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = f64::rmsd(a, b).unwrap();

        // Scale should be 1.0 for RMSD
        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD should be 0 for identical points
        assert!(result.rmsd < 1e-6, "Expected RMSD ~0, got {}", result.rmsd);
    }

    #[test]
    fn mesh_alignment_length_mismatch() {
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; // Different length

        assert!(f64::kabsch(a, b).is_none());
        assert!(f64::rmsd(a, b).is_none());
        assert!(f64::umeyama(a, b).is_none());
    }

    #[test]
    fn mesh_alignment_too_few_points() {
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; // Only 2 points
        let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        assert!(f64::kabsch(a, b).is_none());
    }

    #[test]
    fn mesh_alignment_transform_point() {
        let a: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = f64::kabsch(a, b).unwrap();

        // Transform a point - should stay approximately the same for identical clouds
        let transformed = result.transform_point([1.0, 0.0, 0.0]);
        assert!(
            (transformed[0] - 1.0).abs() < 0.1,
            "Expected x ~1.0, got {}",
            transformed[0]
        );
        assert!(
            transformed[1].abs() < 0.1,
            "Expected y ~0.0, got {}",
            transformed[1]
        );
        assert!(
            transformed[2].abs() < 0.1,
            "Expected z ~0.0, got {}",
            transformed[2]
        );
    }

    #[test]
    fn mesh_alignment_rotation_determinant() {
        let a: &[[f64; 3]] = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let b: &[[f64; 3]] = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];

        let result = f64::kabsch(a, b).unwrap();
        let r = &result.rotation_matrix;

        // Compute determinant of 3x3 rotation matrix
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);

        // Determinant should be +1 (proper rotation) or -1 (improper/reflection)
        assert!(
            (det.abs() - 1.0).abs() < 0.01,
            "Expected det(R) ~±1.0, got {}",
            det
        );
    }
}
