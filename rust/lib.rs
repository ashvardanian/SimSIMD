//! # SimSIMD - Hardware-Accelerated Similarity Metrics and Distance Functions
//!
//! * Zero-dependency header-only C 99 library with bindings for Python, JavaScript and Rust.
//! * Targets ARM NEON, SVE, x86 AVX2, AVX-512 (VNNI, FP16) hardware backends.
//! * Zero-copy compatible with NumPy, PyTorch, TensorFlow, and other tensors.
//! * Handles f64 double-, f32 single-, and f16 half-precision, i8 integral, and binary vectors.
//! * Up to 200x faster than scipy.spatial.distance and numpy.inner.
//! * Used in USearch and several DBMS products.
//!
//! ## Implemented distance functions include:
//!
//! * Euclidean (L2), Inner Distance, and Cosine (Angular) spatial distances.
//! * Hamming (~ Manhattan) and Jaccard (~ Tanimoto) binary distances.
//! * Kullback-Leibler and Jensen–Shannon divergences for probability distributions.
//!
//! The functions in this module are exposed through a trait `SimSIMD`, which is
//! implemented for types with SIMD support (e.g., i8 and f32). The trait provides
//! methods for computing cosine similarity, inner product, and squared Euclidean
//! distance between slices of these types.
//!
//! # Example
//!
//! ```rust
//! use simsimd::SimSIMD;
//!
//! let a = &[1, 2, 3];
//! let b = &[4, 5, 6];
//!
//! // Compute cosine similarity
//! let cosine_sim = i8::cosine(a, b);
//!
//! // Compute inner product
//! let inner_product = i8::inner(a, b);
//!
//! // Compute squared Euclidean distance
//! let sqeuclidean_dist = i8::sqeuclidean(a, b);
//! ```
//!
//! # Safety
//!
//! The functions declared in the `extern "C"` block are low-level bindings to
//! SimSIMD implementation. It is crucial to ensure that the input slices have the
//! same length (`c` parameter) to avoid undefined behavior.
//!
//!
//! # Trait Implementation
//!
//! The `SimSIMD` trait is implemented for types with SIMD support and provides
//! three associated methods:
//!
//! - `cosine(a: &[Self], b: &[Self]) -> Option<f32>`: Computes cosine similarity between two slices.
//! - `inner(a: &[Self], b: &[Self]) -> Option<f32>`: Computes inner product between two slices.
//! - `sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>`: Computes squared Euclidean distance between two slices.
//!
#![allow(non_camel_case_types)]

extern "C" {
    /// Computes cosine similarity for i8 types.
    fn cosine_i8(a: *const i8, b: *const i8, c: usize) -> f32;

    /// Computes cosine similarity for f32 types.
    fn cosine_f32(a: *const f32, b: *const f32, c: usize) -> f32;

    /// Computes inner product for i8 types.
    fn inner_i8(a: *const i8, b: *const i8, c: usize) -> f32;

    /// Computes inner product for f32 types.
    fn inner_f32(a: *const f32, b: *const f32, c: usize) -> f32;

    /// Computes squared Euclidean distance for i8 types.
    fn sqeuclidean_i8(a: *const i8, b: *const i8, c: usize) -> f32;

    /// Computes squared Euclidean distance for f32 types.
    fn sqeuclidean_f32(a: *const f32, b: *const f32, c: usize) -> f32;
}

/// A trait for SIMD similarity functions.
pub trait SimSIMD
where
    Self: Sized,
{
    /// Computes cosine similarity between two slices.
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes inner product between two slices.
    fn inner(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes squared Euclidean distance between two slices.
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>;
}

impl SimSIMD for i8 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { cosine_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { inner_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { sqeuclidean_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }
}

impl SimSIMD for f32 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { cosine_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { inner_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //
    fn assert_almost_equal(left: f32, right: f32, tolerance: f32) {
        let lower = right - tolerance;
        let upper = right + tolerance;

        assert!(left >= lower && left <= upper);
    }

    #[test]
    fn test_cosine_i8() {
        let a = &[3, 97, 127];
        let b = &[3, 97, 127];

        if let Some(result) = SimSIMD::cosine(a, b) {
            assert_almost_equal(0.00012027938, result, 0.01);
            println!("The result of cosine_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_cosine_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[1.0, 2.0, 3.0];

        if let Some(result) = SimSIMD::cosine(a, b) {
            assert_almost_equal(0.004930496, result, 0.01);
            println!("The result of cosine_f32 is {:.8}", result);
        }
    }

    #[test]
    fn test_inner_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::inner(a, b) {
            assert_almost_equal(0.029403687, result, 0.01);
            println!("The result of inner_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_inner_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::inner(a, b) {
            assert_almost_equal(-31.0, result, 0.01);
            println!("The result of inner_f32 is {:.8}", result);
        }
    }

    #[test]
    fn test_sqeuclidean_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            assert_almost_equal(27.0, result, 0.01);
            println!("The result of sqeuclidean_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_sqeuclidean_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            assert_almost_equal(27.0, result, 0.01);
            println!("The result of sqeuclidean_f32 is {:.8}", result);
        }
    }
}
