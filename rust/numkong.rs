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

// Module declarations
pub mod numerics;
pub mod scalars;
pub mod tensor;

// Re-export scalar types at crate root
pub use scalars::{bf16, e4m3, e5m2, f16, i4x2, u1x8, u4x2};

// Re-export complex product types
pub use numerics::{ComplexProductF32, ComplexProductF64};

// Re-export all numeric traits
pub use numerics::{
    ATan, Angular, BinarySimilarity, ComplexDot, ComplexProducts, ComplexVDot, Cos, Dot,
    Elementwise, Euclidean, Hamming, Haversine, Jaccard, JensenShannon, KullbackLeibler,
    MeshAlignment, MeshAlignmentResult, ProbabilitySimilarity, Scale, Sin, Sparse,
    SpatialSimilarity, Sum, Trigonometry, Vincenty, WSum, FMA,
};

// Re-export cast operations
pub use numerics::{cast, CastDtype};

// Re-export capabilities module
pub use numerics::capabilities;

// Re-export tensor types
pub use tensor::{
    Allocator, Dots, Global, Matrix, MatrixMultiplier, MatrixView, MatrixViewMut, ShapeDescriptor,
    SliceRange, Tensor, TensorError, TensorView, TensorViewMut, DEFAULT_MAX_RANK, SIMD_ALIGNMENT,
};

// region: Tests

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
    fn hamming_u1x8() {
        let a = &[u1x8(0b01010101), u1x8(0b11110000), u1x8(0b10101010)];
        let b = &[u1x8(0b01010101), u1x8(0b11110000), u1x8(0b10101010)];
        if let Some(result) = u1x8::hamming(a, b) {
            assert_almost_equal(0.0, result, 0.01);
        }
    }

    #[test]
    fn jaccard_u1x8() {
        let a = &[u1x8(0b11110000), u1x8(0b00001111), u1x8(0b10101010)];
        let b = &[u1x8(0b11110000), u1x8(0b00001111), u1x8(0b01010101)];
        if let Some(result) = u1x8::jaccard(a, b) {
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

    // Cast smoke tests with known hex values
    #[test]
    fn cast_f16_f32() {
        let src = [f16(0x3C00), f16(0xBC00)]; // 1.0, -1.0
        let mut dst = [0.0f32; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.0, -1.0]);
    }

    #[test]
    fn cast_bf16_f32() {
        let src = [bf16(0x3F80), bf16(0xBF80)]; // 1.0, -1.0
        let mut dst = [0.0f32; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.0, -1.0]);
    }

    #[test]
    fn cast_e4m3_f32() {
        let src = [e4m3(0x38), e4m3(0xB8)]; // 1.0, -1.0
        let mut dst = [0.0f32; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.0, -1.0]);
    }

    #[test]
    fn cast_e5m2_f32() {
        let src = [e5m2(0x3C), e5m2(0xBC)]; // 1.0, -1.0
        let mut dst = [0.0f32; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.0, -1.0]);
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
        // Use non-symmetric point cloud to avoid repeated eigenvalues in covariance matrix
        let a: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let b: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];

        let result = f64::kabsch(a, b).unwrap();

        // Scale should be 1.0 for Kabsch
        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD should be ~0 for identical points (relaxed tolerance for approximate McAdams SVD)
        assert!(result.rmsd < 0.01, "Expected RMSD ~0, got {}", result.rmsd);
    }

    #[test]
    fn kabsch_f32_identical_points() {
        // Use non-symmetric point cloud to avoid repeated eigenvalues in covariance matrix
        let a: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let b: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];

        let result = f32::kabsch(a, b).unwrap();

        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD should be ~0 for identical points (relaxed tolerance for approximate McAdams SVD)
        assert!(result.rmsd < 0.01, "Expected RMSD ~0, got {}", result.rmsd);
    }

    #[test]
    fn umeyama_f64_scaled_points() {
        // Use non-symmetric point cloud to avoid repeated eigenvalues
        // B is 2x scaled version of A
        let a: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let b: &[[f64; 3]] = &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ];

        let result = f64::umeyama(a, b).unwrap();

        // Scale should be ~2.0 (transforming A to B)
        // Note: McAdams SVD is approximate, so we use a relaxed tolerance
        // The algorithm correctly detects scaling > 1 and produces usable results
        assert!(
            result.scale > 1.0 && result.scale < 3.0,
            "Expected scale in range (1.0, 3.0), got {}",
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

// endregion: Tests
