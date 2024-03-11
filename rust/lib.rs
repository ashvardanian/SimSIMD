//! # SimSIMD - Hardware-Accelerated Similarity Metrics and Distance Functions
//!
//! * Targets ARM NEON, SVE, x86 AVX2, AVX-512 (VNNI, FP16) hardware backends.
//! * Handles `f64` double-, `f32` single-, and `f16` half-precision, `i8` integral, and binary vectors.
//! * Zero-dependency header-only C 99 library with bindings for Rust and other langauges.
//!
//! ## Implemented distance functions include:
//!
//! * Euclidean (L2), Inner Distance, and Cosine (Angular) spatial distances.
//! * Hamming (~ Manhattan) and Jaccard (~ Tanimoto) binary distances.
//! * Kullback-Leibler and Jensen-Shannon divergences for probability distributions.
//!
//! ## Example
//!
//! ```rust
//! use simsimd::SpatialSimilarity;
//!
//! let a = &[1, 2, 3];
//! let b = &[4, 5, 6];
//!
//! // Compute cosine similarity
//! let cosine_sim = i8::cosine(a, b);
//!
//! // Compute inner product distance
//! let inner_product = i8::inner(a, b);
//!
//! // Compute squared Euclidean distance
//! let sqeuclidean_dist = i8::sqeuclidean(a, b);
//! ```
//!
//! ## Traits
//!
//! The `SpatialSimilarity` trait covers following methods:
//!
//! - `cosine(a: &[Self], b: &[Self]) -> Option<f32>`: Computes cosine similarity between two slices.
//! - `inner(a: &[Self], b: &[Self]) -> Option<f32>`: Computes inner product distance between two slices.
//! - `sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>`: Computes squared Euclidean distance between two slices.
//!
//! The `BinarySimilarity` trait covers following methods:
//!
//! - `hamming(a: &[Self], b: &[Self]) -> Option<f32>`: Computes Hamming distance between two slices.
//! - `jaccard(a: &[Self], b: &[Self]) -> Option<f32>`: Computes Jaccard index between two slices.
//!
//! The `ProbabilitySimilarity` trait covers following methods:
//!
//! - `jensenshannon(a: &[Self], b: &[Self]) -> Option<f32>`: Computes Jensen-Shannon divergence between two slices.
//! - `kullbackleibler(a: &[Self], b: &[Self]) -> Option<f32>`: Computes Kullback-Leibler divergence between two slices.
//!
#![allow(non_camel_case_types)]

extern "C" {
    fn cosine_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn cosine_f16(a: *const u16, b: *const u16, c: usize) -> f32;
    fn cosine_f32(a: *const f32, b: *const f32, c: usize) -> f32;
    fn cosine_f64(a: *const f64, b: *const f64, c: usize) -> f32;

    fn inner_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn inner_f16(a: *const u16, b: *const u16, c: usize) -> f32;
    fn inner_f32(a: *const f32, b: *const f32, c: usize) -> f32;
    fn inner_f64(a: *const f64, b: *const f64, c: usize) -> f32;

    fn sqeuclidean_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn sqeuclidean_f16(a: *const u16, b: *const u16, c: usize) -> f32;
    fn sqeuclidean_f32(a: *const f32, b: *const f32, c: usize) -> f32;
    fn sqeuclidean_f64(a: *const f64, b: *const f64, c: usize) -> f32;

    fn hamming_b8(a: *const u8, b: *const u8, c: usize) -> f32;
    fn jaccard_b8(a: *const u8, b: *const u8, c: usize) -> f32;

    fn js_f16(a: *const u16, b: *const u16, c: usize) -> f32;
    fn js_f32(a: *const f32, b: *const f32, c: usize) -> f32;
    fn js_f64(a: *const f64, b: *const f64, c: usize) -> f32;

    fn kl_f16(a: *const u16, b: *const u16, c: usize) -> f32;
    fn kl_f32(a: *const f32, b: *const f32, c: usize) -> f32;
    fn kl_f64(a: *const f64, b: *const f64, c: usize) -> f32;
}

/// A half-precision floating point number.
#[repr(transparent)]
pub struct f16(u16);

impl f16 {}

/// `SpatialSimilarity` provides a set of trait methods for computing similarity
/// or distance between spatial data vectors in SIMD (Single Instruction, Multiple Data) context.
/// These methods can be used to calculate metrics like cosine similarity, inner product,
/// and squared Euclidean distance between two slices of data.
///
/// Each method takes two slices of data (a and b) and returns an Option<f32>.
/// The result is `None` if the slices are not of the same length, as these operations
/// require one-to-one correspondence between the elements of the slices.
/// Otherwise, it returns the computed similarity or distance as `Some(f32)`.
pub trait SpatialSimilarity
where
    Self: Sized,
{
    /// Computes the cosine similarity between two slices.
    /// The cosine similarity is a measure of similarity between two non-zero vectors
    /// of an inner product space that measures the cosine of the angle between them.
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes the inner product (also known as dot product) between two slices.
    /// The inner product is the sum of the products of the corresponding entries
    /// of the two sequences of numbers.
    fn inner(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes the squared Euclidean distance between two slices.
    /// The squared Euclidean distance is the sum of the squared differences
    /// between corresponding elements of the two slices.
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>;
}

/// `BinarySimilarity` provides trait methods for computing similarity metrics
/// that are commonly used with binary data vectors, such as Hamming distance
/// and Jaccard index.
///
/// The methods accept two slices of binary data and return an Option<f32>
/// indicating the computed similarity or distance, with `None` returned if the
/// slices differ in length.
pub trait BinarySimilarity
where
    Self: Sized,
{
    /// Computes the Hamming distance between two binary data slices.
    /// The Hamming distance between two strings of equal length is the number of
    /// bits at which the corresponding values are different.
    fn hamming(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes the Jaccard index between two bitsets represented by binary data slices.
    /// The Jaccard index, also known as the Jaccard similarity coefficient, is a statistic
    /// used for gauging the similarity and diversity of sample sets.
    fn jaccard(a: &[Self], b: &[Self]) -> Option<f32>;
}

/// `ProbabilitySimilarity` provides trait methods for computing similarity or divergence
/// measures between probability distributions, such as the Jensen-Shannon divergence
/// and the Kullback-Leibler divergence.
///
/// These methods are particularly useful in contexts such as information theory and
/// machine learning, where one often needs to measure how one probability distribution
/// differs from a second, reference probability distribution.
pub trait ProbabilitySimilarity
where
    Self: Sized,
{
    /// Computes the Jensen-Shannon divergence between two probability distributions.
    /// The Jensen-Shannon divergence is a method of measuring the similarity between
    /// two probability distributions. It is based on the Kullback-Leibler divergence,
    /// but is symmetric and always has a finite value.
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<f32>;

    /// Computes the Kullback-Leibler divergence between two probability distributions.
    /// The Kullback-Leibler divergence is a measure of how one probability distribution
    /// diverges from a second, expected probability distribution.
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<f32>;
}

impl BinarySimilarity for u8 {
    fn hamming(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { hamming_b8(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }

    fn jaccard(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { jaccard_b8(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }
}

impl SpatialSimilarity for i8 {
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

impl SpatialSimilarity for f16 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let operation = unsafe { cosine_f16(a_ptr, b_ptr, a.len()) };
        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let operation = unsafe { inner_f16(a_ptr, b_ptr, a.len()) };
        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let operation = unsafe { sqeuclidean_f16(a_ptr, b_ptr, a.len()) };
        Some(operation)
    }
}

impl SpatialSimilarity for f32 {
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

impl SpatialSimilarity for f64 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { cosine_f64(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { inner_f64(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { sqeuclidean_f64(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }
}

impl ProbabilitySimilarity for f16 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let operation = unsafe { js_f16(a_ptr, b_ptr, a.len()) };
        Some(operation)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let operation = unsafe { kl_f16(a_ptr, b_ptr, a.len()) };
        Some(operation)
    }
}

impl ProbabilitySimilarity for f32 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { js_f32(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { kl_f32(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }
}

impl ProbabilitySimilarity for f64 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { js_f64(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let operation = unsafe { kl_f64(a.as_ptr(), b.as_ptr(), a.len()) };
        Some(operation)
    }
}

// In the older revisions of the library, the `SpatialSimilarity`
// trait was called `SimSIMD`. The following trait provides a compatibility layer.
pub trait SimSIMD
where
    Self: Sized,
{
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32>;
    fn inner(a: &[Self], b: &[Self]) -> Option<f32>;
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>;
}

impl<T: SpatialSimilarity> SimSIMD for T {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        SpatialSimilarity::cosine(a, b)
    }
    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        SpatialSimilarity::inner(a, b)
    }
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        SpatialSimilarity::sqeuclidean(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16 as HalfF16;

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
            println!("The result of cosine_i8 is {:.8}", result);
            assert_almost_equal(0.00012027938, result, 0.01);
        }
    }

    #[test]
    fn test_cosine_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::cosine(a, b) {
            println!("The result of cosine_f32 is {:.8}", result);
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn test_inner_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::inner(a, b) {
            println!("The result of inner_i8 is {:.8}", result);
            assert_almost_equal(0.029403687, result, 0.01);
        }
    }

    #[test]
    fn test_inner_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::inner(a, b) {
            println!("The result of inner_f32 is {:.8}", result);
            assert_almost_equal(-31.0, result, 0.01);
        }
    }

    #[test]
    fn test_sqeuclidean_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            println!("The result of sqeuclidean_i8 is {:.8}", result);
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    #[test]
    fn test_sqeuclidean_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            println!("The result of sqeuclidean_f32 is {:.8}", result);
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    // Adding new tests for bit-level distances
    #[test]
    fn test_hamming_u8() {
        let a = &[0b01010101, 0b11110000, 0b10101010]; // Binary representations for clarity
        let b = &[0b01010101, 0b11110000, 0b10101010];

        if let Some(result) = BinarySimilarity::hamming(a, b) {
            println!("The result of hamming_u8 is {:.8}", result);
            assert_almost_equal(0.0, result, 0.01); // Perfect match
        }
    }

    #[test]
    fn test_jaccard_u8() {
        // For binary data, treat each byte as a set of bits
        let a = &[0b11110000, 0b00001111, 0b10101010];
        let b = &[0b11110000, 0b00001111, 0b01010101];

        if let Some(result) = BinarySimilarity::jaccard(a, b) {
            println!("The result of jaccard_u8 is {:.8}", result);
            assert_almost_equal(0.5, result, 0.01); // Example value
        }
    }

    // Adding new tests for probability similarities
    #[test]
    fn test_js_f32() {
        let a = &[0.1, 0.9, 0.0];
        let b = &[0.2, 0.8, 0.0];

        if let Some(result) = ProbabilitySimilarity::jensenshannon(a, b) {
            println!("The result of js_f32 is {:.8}", result);
            assert_almost_equal(0.01, result, 0.01); // Example value
        }
    }

    #[test]
    fn test_kl_f32() {
        let a = &[0.1, 0.9, 0.0];
        let b = &[0.2, 0.8, 0.0];

        if let Some(result) = ProbabilitySimilarity::kullbackleibler(a, b) {
            println!("The result of kl_f32 is {:.8}", result);
            assert_almost_equal(0.036, result, 0.01); // Example value
        }
    }

    #[test]
    fn test_cosine_f16_same() {
        // Assuming these u16 values represent f16 bit patterns, and they are identical
        let a_u16: &[u16] = &[15360, 16384, 17408]; // Corresponding to some f16 values
        let b_u16: &[u16] = &[15360, 16384, 17408]; // Same as above for simplicity

        // Reinterpret cast from &[u16] to &[f16]
        // SAFETY: This is safe as long as the representations are guaranteed to be identical,
        // which they are for transparent structs wrapping the same type.
        let a_f16: &[f16] =
            unsafe { std::slice::from_raw_parts(a_u16.as_ptr() as *const f16, a_u16.len()) };
        let b_f16: &[f16] =
            unsafe { std::slice::from_raw_parts(b_u16.as_ptr() as *const f16, b_u16.len()) };

        if let Some(result) = SimSIMD::cosine(a_f16, b_f16) {
            println!("The result of cosine_f16 is {:.8}", result);
            assert_almost_equal(0.0, result, 0.01); // Example value, adjust according to actual expected value
        }
    }

    #[test]
    fn test_cosine_f16_interop() {
        let a_half: Vec<HalfF16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();
        let b_half: Vec<HalfF16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| HalfF16::from_f32(x))
            .collect();

        // SAFETY: This is safe as long as the memory representations are guaranteed to be identical,
        // which they are due to both being #[repr(transparent)] wrappers around u16.
        let a_simsimd: &[f16] =
            unsafe { std::slice::from_raw_parts(a_half.as_ptr() as *const f16, a_half.len()) };
        let b_simsimd: &[f16] =
            unsafe { std::slice::from_raw_parts(b_half.as_ptr() as *const f16, b_half.len()) };

        // Use the reinterpret-casted slices with your SpatialSimilarity implementation
        if let Some(result) = SpatialSimilarity::cosine(a_simsimd, b_simsimd) {
            // Expected value might need adjustment depending on actual cosine functionality
            // Assuming identical vectors yield cosine similarity of 1.0
            println!("The result of cosine_f16 (interop) is {:.8}", result);
            assert_almost_equal(0.025, result, 0.01);
        }
    }
}
