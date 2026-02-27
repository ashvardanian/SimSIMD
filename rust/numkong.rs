//! # NumKong - Hardware-Accelerated Numerics
//!
//! Provides SIMD-accelerated distance metrics, elementwise operations, and tensor algebra
//! targeting ARM NEON/SVE/SME and x86 AVX2/AVX-512 backends.
//!
//! ## Modules
//!
//! - [`scalars`]: Mixed-precision scalar types (`f16`, `bf16`, FP8, packed integers) and [`FloatLike`] trait
//! - [`numerics`]: Distance functions, elementwise operations, trigonometry, reductions, and geospatial
//! - [`tensor`]: N-dimensional tensors, GEMM, and packed spatial distance operations
//!
//! ## Implemented operations include:
//!
//! * Euclidean (L2), inner product, and angular (cosine) spatial distances.
//! * Hamming and Jaccard binary distances.
//! * Kullback-Leibler and Jensen-Shannon divergences.
//! * Elementwise scale, sum, blend, and FMA operations.
//! * Trigonometric functions (sin, cos, atan).
//! * Type casting between all scalar formats.
//! * Matrix multiplication with pre-packing (GEMM).
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
//! let l2sq_dist = f32::sqeuclidean(a, b);
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
//! - `angular(a, b)` / `cosine(a, b)`: Computes angular distance (1 − cosine similarity).
//! - `sqeuclidean(a, b)`: Computes squared Euclidean distance.
//! - `euclidean(a, b)`: Computes Euclidean distance.
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
//! The `Elementwise` trait (combining `EachScale`, `EachSum`, `EachBlend`, `EachFMA`) covers:
//!
//! - `scale(a, alpha, beta, result)`: Element-wise `result[i] = α × a[i] + β`.
//! - `sum(a, b, result)`: Element-wise `result[i] = a[i] + b[i]`.
//! - `wsum(a, b, alpha, beta, result)`: Weighted sum `result[i] = α × a[i] + β × b[i]`.
//! - `fma(a, b, c, alpha, beta, result)`: Fused multiply-add `result[i] = α × a[i] × b[i] + β × c[i]`.
//!
//! The `Trigonometry` trait (combining `EachSin`, `EachCos`, `EachATan`) covers:
//!
//! - `sin(input, result)`: Element-wise sine.
//! - `cos(input, result)`: Element-wise cosine.
//! - `atan(input, result)`: Element-wise arctangent.
//!
//! Additional traits: `ComplexDot`, `ComplexVDot`, `SparseIntersect`, `SparseDot`.
//!
#![allow(non_camel_case_types)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

// Module declarations
pub mod numerics;
pub mod scalars;
pub mod tensor;

// Re-export scalar types at crate root
pub use scalars::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, FloatLike};

// Re-export complex product types
pub use numerics::{ComplexProductF32, ComplexProductF64};

// Re-export all numeric traits
pub use numerics::{
    Angular, Bilinear, BinarySimilarity, ComplexBilinear, ComplexDot, ComplexProducts, ComplexVDot,
    Dot, EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum, Elementwise,
    Euclidean, Hamming, Haversine, Jaccard, JensenShannon, KullbackLeibler, Mahalanobis,
    MeshAlignment, MeshAlignmentResult, ProbabilitySimilarity, ReduceMinMax, ReduceMoments,
    Reductions, SparseDot, SparseIntersect, SpatialSimilarity, Trigonometry, Vincenty,
};

// Re-export cast operations
pub use numerics::{cast, CastDtype};

// Re-export capabilities module
pub use numerics::cap;
pub use numerics::capabilities;

// Re-export tensor types
pub use tensor::{
    Allocator, Angulars, Dots, Euclideans, Global, Hammings, Jaccards, Matrix, MatrixView,
    MatrixViewMut, ShapeDescriptor, SliceRange, Tensor, TensorError, TensorView, TensorViewMut,
    TransposedMatrixMultiplier, DEFAULT_MAX_RANK, SIMD_ALIGNMENT,
};

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalars::{FloatLike, TestableType};

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
            let actual = r.to_f64();
            let tol = T::atol() + T::rtol() * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "scale element {}: expected {} but got {} (tol={})",
                i,
                expected,
                actual,
                tol
            );
        }
    }

    fn check_each_sum<T>(values_a: &[f32], values_b: &[f32])
    where
        T: FloatLike + TestableType + EachSum,
    {
        let a: Vec<T> = values_a.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = values_b.iter().map(|&v| T::from_f32(v)).collect();
        let mut result = vec![T::zero(); a.len()];
        T::each_sum(&a, &b, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = values_a[i] as f64 + values_b[i] as f64;
            let actual = r.to_f64();
            let tol = T::atol() + T::rtol() * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "sum element {}: expected {} but got {} (tol={})",
                i,
                expected,
                actual,
                tol
            );
        }
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
            let actual = r.to_f64();
            let tol = T::atol() + T::rtol() * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "blend element {}: expected {} but got {} (tol={})",
                i,
                expected,
                actual,
                tol
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
            let actual = r.to_f64();
            let tol = T::atol() + T::rtol() * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "fma element {}: expected {} but got {} (tol={})",
                i,
                expected,
                actual,
                tol
            );
        }
    }

    fn check_each_sin<T>(count: usize)
    where
        T: FloatLike + TestableType + EachSin,
    {
        use core::f64::consts::PI;
        let values: Vec<f64> = (0..count)
            .map(|i| (i as f64) * 2.0 * PI / (count as f64))
            .collect();
        let a: Vec<T> = values.iter().map(|&v| T::from_f32(v as f32)).collect();
        let mut result = vec![T::zero(); count];
        T::sin(&a, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = values[i].sin();
            let actual = r.to_f64();
            let tol = T::atol() * 10000.0 + T::rtol() * 10000.0 * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "sin({}): expected {} but got {} (tol={})",
                values[i],
                expected,
                actual,
                tol
            );
        }
    }

    fn check_each_cos<T>(count: usize)
    where
        T: FloatLike + TestableType + EachCos,
    {
        use core::f64::consts::PI;
        let values: Vec<f64> = (0..count)
            .map(|i| (i as f64) * 2.0 * PI / (count as f64))
            .collect();
        let a: Vec<T> = values.iter().map(|&v| T::from_f32(v as f32)).collect();
        let mut result = vec![T::zero(); count];
        T::cos(&a, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = values[i].cos();
            let actual = r.to_f64();
            let tol = T::atol() * 10000.0 + T::rtol() * 10000.0 * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "cos({}): expected {} but got {} (tol={})",
                values[i],
                expected,
                actual,
                tol
            );
        }
    }

    fn check_each_atan<T>(count: usize)
    where
        T: FloatLike + TestableType + EachATan,
    {
        let values: Vec<f64> = (0..count)
            .map(|i| -5.0 + 10.0 * (i as f64) / (count as f64))
            .collect();
        let a: Vec<T> = values.iter().map(|&v| T::from_f32(v as f32)).collect();
        let mut result = vec![T::zero(); count];
        T::atan(&a, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = values[i].atan();
            let actual = r.to_f64();
            let tol = T::atol() * 10000.0 + T::rtol() * 10000.0 * expected.abs();
            assert!(
                (actual - expected).abs() <= tol,
                "atan({}): expected {} but got {} (tol={})",
                values[i],
                expected,
                actual,
                tol
            );
        }
    }

    fn check_dot<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Dot,
        T::Output: FloatLike,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result: f64 = T::dot(&a, &b).unwrap().to_f64();
        let tol = T::atol() + T::rtol() * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "dot: expected {} but got {} (tol={})",
            expected,
            result,
            tol
        );
    }

    fn check_angular<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Angular,
        T::Output: FloatLike,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result: f64 = T::angular(&a, &b).unwrap().to_f64();
        let tol = T::atol() + T::rtol() * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "angular: expected {} but got {} (tol={})",
            expected,
            result,
            tol
        );
    }

    fn check_sqeuclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::SqEuclideanOutput: FloatLike,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result: f64 = T::sqeuclidean(&a, &b).unwrap().to_f64();
        let tol = T::atol() + T::rtol() * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "sqeuclidean: expected {} but got {} (tol={})",
            expected,
            result,
            tol
        );
    }

    fn check_euclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::EuclideanOutput: FloatLike,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result: f64 = T::euclidean(&a, &b).unwrap().to_f64();
        let tol = T::atol() + T::rtol() * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "euclidean: expected {} but got {} (tol={})",
            expected,
            result,
            tol
        );
    }

    fn check_cast_roundtrip<T: FloatLike + TestableType + CastDtype>(values: &[f32]) {
        let src: Vec<T> = values.iter().map(|&v| T::from_f32(v)).collect();
        let mut dst = vec![0.0f32; src.len()];
        cast(&src, &mut dst).unwrap();
        for (i, (&expected, &actual)) in values.iter().zip(dst.iter()).enumerate() {
            let tol = T::atol() + T::rtol() * (expected as f64).abs();
            assert!(
                (actual as f64 - expected as f64).abs() <= tol,
                "cast roundtrip element {}: expected {} but got {} (tol={})",
                i,
                expected,
                actual,
                tol
            );
        }
    }

    // region: Dot Products

    #[test]
    fn dot() {
        check_dot::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
    }

    // endregion

    // region: Angular Distances

    #[test]
    fn angular() {
        // angular([1,2,3],[4,5,6]) = 1 - 32/sqrt(14*77) ≈ 0.025368
        let expected = 1.0 - 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        check_angular::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
    }

    // endregion

    // region: Euclidean Distances

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
        check_euclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
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
        let result: f64 = T::kullbackleibler(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        let tol = T::atol().max(1e-6) + T::rtol().max(1e-6) * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "kld<{}>: expected {} but got {} (tol={})",
            core::any::type_name::<T>(),
            expected,
            result,
            tol
        );
    }

    fn check_jsd<T>(a: &[f32], b: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + JensenShannon,
        T::Output: FloatLike,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result: f64 = T::jensenshannon(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        let tol = T::atol().max(1e-6) + T::rtol().max(1e-6) * expected.abs();
        assert!(
            (result - expected).abs() <= tol,
            "jsd<{}>: expected {} but got {} (tol={})",
            core::any::type_name::<T>(),
            expected,
            result,
            tol
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
        fn re_f64(&self) -> f64;
        fn im_f64(&self) -> f64;
    }
    impl ComplexOutput for (f32, f32) {
        fn re_f64(&self) -> f64 {
            self.0 as f64
        }
        fn im_f64(&self) -> f64 {
            self.1 as f64
        }
    }
    impl ComplexOutput for (f64, f64) {
        fn re_f64(&self) -> f64 {
            self.0
        }
        fn im_f64(&self) -> f64 {
            self.1
        }
    }

    fn check_complex_dot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: FloatLike + TestableType + ComplexDot,
        T::Output: ComplexOutput,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = <T as ComplexDot>::dot(&a_t, &b_t).unwrap();
        let tol = T::atol() + T::rtol() * expected_re.abs().max(expected_im.abs());
        assert!(
            (result.re_f64() - expected_re).abs() <= tol,
            "complex_dot<{}> real: expected {} got {} (tol={})",
            core::any::type_name::<T>(),
            expected_re,
            result.re_f64(),
            tol
        );
        assert!(
            (result.im_f64() - expected_im).abs() <= tol,
            "complex_dot<{}> imag: expected {} got {} (tol={})",
            core::any::type_name::<T>(),
            expected_im,
            result.im_f64(),
            tol
        );
    }

    fn check_complex_vdot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: FloatLike + TestableType + ComplexVDot,
        T::Output: ComplexOutput,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = T::vdot(&a_t, &b_t).unwrap();
        let tol = T::atol() + T::rtol() * expected_re.abs().max(expected_im.abs());
        assert!(
            (result.re_f64() - expected_re).abs() <= tol,
            "complex_vdot<{}> real: expected {} got {} (tol={})",
            core::any::type_name::<T>(),
            expected_re,
            result.re_f64(),
            tol
        );
        assert!(
            (result.im_f64() - expected_im).abs() <= tol,
            "complex_vdot<{}> imag: expected {} got {} (tol={})",
            core::any::type_name::<T>(),
            expected_im,
            result.im_f64(),
            tol
        );
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
        assert!(
            (result.re_f64() - 1.0).abs() <= tol,
            "complex_bilinear<{}> real: expected ~1.0, got {} (tol={})",
            core::any::type_name::<T>(),
            result.re_f64(),
            tol
        );
        assert!(
            result.im_f64().abs() <= tol,
            "complex_bilinear<{}> imag: expected ~0.0, got {} (tol={})",
            core::any::type_name::<T>(),
            result.im_f64(),
            tol
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

    // region: Sparse Intersections

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
        let cloud_t: Vec<[T; 3]> = cloud
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect();
        let result = T::kabsch(&cloud_t, &cloud_t).unwrap();
        let scale = FloatLike::to_f64(result.scale);
        let tol = T::atol() + T::rtol();
        assert!(
            (scale - 1.0).abs() < tol,
            "kabsch<{}> scale: expected ~1.0, got {} (tol={})",
            core::any::type_name::<T>(),
            scale,
            tol
        );
        let rmsd = FloatLike::to_f64(result.rmsd);
        assert!(
            rmsd < tol,
            "kabsch<{}> rmsd: expected ~0, got {} (tol={})",
            core::any::type_name::<T>(),
            rmsd,
            tol
        );
    }

    fn check_umeyama_scaled<T>(cloud: &[[f32; 3]], scaled: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Output: FloatLike,
    {
        let cloud_t: Vec<[T; 3]> = cloud
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect();
        let scaled_t: Vec<[T; 3]> = scaled
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect();
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
        let cloud_t: Vec<[T; 3]> = cloud
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect();
        let result = T::rmsd(&cloud_t, &cloud_t).unwrap();
        let scale = FloatLike::to_f64(result.scale);
        let tol = T::atol() + T::rtol();
        assert!(
            (scale - 1.0).abs() < tol,
            "rmsd<{}> scale: expected ~1.0, got {} (tol={})",
            core::any::type_name::<T>(),
            scale,
            tol
        );
        let rmsd = FloatLike::to_f64(result.rmsd);
        assert!(
            rmsd < tol,
            "rmsd<{}> rmsd: expected ~0, got {} (tol={})",
            core::any::type_name::<T>(),
            rmsd,
            tol
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

    // endregion
}

// endregion: Tests

// region: WASM Runtime Tests

/// WASM runtime integration tests using Wasmtime
/// These tests validate that WASI builds work correctly with standalone runtimes
#[cfg(all(test, feature = "wasm-runtime"))]
mod wasm_runtime_tests {
    use std::fs;
    use wasmtime::*;
    use wasmtime_wasi::WasiCtx;

    /// Test that WASI WASM module can be loaded and executed with Wasmtime
    /// This validates the dual-path capability detection (EM_ASM vs WASI imports)
    #[test]
    fn wasi_with_wasmtime() -> wasmtime::Result<()> {
        // Check if WASI build exists
        let wasm_path = "build-wasi/test.wasm";
        if !std::path::Path::new(wasm_path).exists() {
            eprintln!("WASI build not found at {}. Run:", wasm_path);
            eprintln!("  export WASI_SDK_PATH=~/wasi-sdk");
            eprintln!("  cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_WASM_WASI=ON");
            eprintln!("  cmake --build build-wasi");
            return Ok(()); // Skip test if build doesn't exist
        }

        println!("Loading WASI module from {}", wasm_path);

        // Create Wasmtime engine and linker
        let engine = Engine::default();
        let mut linker = Linker::new(&engine);

        // Create WASI context (Wasmtime 41+ API)
        let wasi = WasiCtx::builder().inherit_stdio().inherit_args().build_p1();
        let mut store = Store::new(&engine, wasi);

        // Add WASI support (Wasmtime 41+ requires p1 module)
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s)?;

        // Provide capability detection imports (required for WASI build)
        // These functions are called from nk_capabilities_v128relaxed_() in C code
        linker.func_wrap("env", "nk_has_v128", || -> i32 {
            // Return 1 (true) - assume SIMD128 is available in Wasmtime
            println!("  nk_has_v128() called from WASM -> returning 1");
            1
        })?;

        linker.func_wrap("env", "nk_has_relaxed", || -> i32 {
            // Return 1 (true) - assume Relaxed SIMD is available in Wasmtime
            println!("  nk_has_relaxed() called from WASM -> returning 1");
            1
        })?;

        // Load WASM module
        let wasm_bytes = fs::read(wasm_path)?;
        let module = Module::new(&engine, wasm_bytes)?;

        // Instantiate module
        println!("Instantiating WASM module...");
        let instance = linker.instantiate(&mut store, &module)?;

        // Get main function
        let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;

        // Run tests
        println!("Running WASM tests...");
        let exit_code = main.call(&mut store, ())?;

        println!("WASM tests completed with exit code: {}", exit_code);

        // Assert tests passed
        assert_eq!(
            exit_code, 0,
            "WASI tests failed with exit code {}",
            exit_code
        );

        Ok(())
    }

    /// Test capability detection mechanism works in WASI environment
    #[test]
    fn capability_imports() -> wasmtime::Result<()> {
        println!("Testing capability import mechanism...");

        // Create minimal engine for import testing
        let engine = Engine::default();
        let mut linker = Linker::<()>::new(&engine);

        // Test that we can define the required imports
        linker.func_wrap("env", "nk_has_v128", || -> i32 { 1 })?;
        linker.func_wrap("env", "nk_has_relaxed", || -> i32 { 0 })?;

        println!("  ✓ Capability imports defined successfully");

        Ok(())
    }
}

// endregion: WASM Runtime Tests
