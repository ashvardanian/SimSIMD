//! # SpatialSimilarity - Hardware-Accelerated Similarity Metrics and Distance Functions
//!
//! * Targets ARM NEON, SVE, x86 AVX2, AVX-512 (VNNI, FP16) hardware backends.
//! * Handles `f64` double- and `f32` single-precision, integral, and binary vectors.
//! * Exposes half-precision (`f16`) and brain floating point (`bf16`) types.
//! * Zero-dependency header-only C 99 library with bindings for Rust and other languages.
//!
//! ## Implemented distance functions include:
//!
//! * Euclidean (L2 & L2 squared), Inner Distance, and Angular (Cosine) spatial distances.
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
//! // Compute dot product distance
//! let dot_product = i8::dot(a, b);
//!
//! // Compute angular distance
//! let angular_dist = i8::angular(a, b);
//!
//! // Compute squared Euclidean distance
//! let l2sq_dist = i8::l2sq(a, b);
//!
//! // Optimize performance by flushing denormals
//! simsimd::capabilities::flush_denormals();
//! ```
//!
//! ## Mixed Precision Support
//!
//! ```rust
//! use simsimd::{SpatialSimilarity, f16, bf16};
//!
//! // Work with half-precision floats
//! let half_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_cos_dist = f16::cos(&half_a, &half_b);
//!
//! // Work with brain floats
//! let brain_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_cos_dist = bf16::cos(&brain_a, &brain_b);
//!
//! // Direct bit manipulation
//! let half = f16::from_f32(3.14);
//! let bits = half.0; // Access raw u16 representation
//! let reconstructed = f16(bits);
//! ```
//!
//! ## Traits
//!
//! The `SpatialSimilarity` trait covers following methods:
//!
//! - `angular(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes angular distance between two slices.
//! - `dot(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes dot product between two slices.
//! - `euclidean(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes Euclidean distance between two slices.
//! - `sqeuclidean(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes squared Euclidean distance between two slices.
//!
//! The `BinarySimilarity` trait covers following methods:
//!
//! - `hamming(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes Hamming distance between two slices.
//! - `jaccard(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes Jaccard distance between two slices.
//!
//! The `ProbabilitySimilarity` trait covers following methods:
//!
//! - `jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes Jensen-Shannon divergence between two slices.
//! - `kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance>`: Computes Kullback-Leibler divergence between two slices.
//!
#![allow(non_camel_case_types)]
#![cfg_attr(not(test), no_std)]

pub type Distance = f64;
pub type ComplexProduct = (f64, f64);

/// Compatibility function for pre 1.85 Rust versions lacking `f32::abs`.
#[inline(always)]
fn f32_abs_compat(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[link(name = "simsimd")]
extern "C" {

    fn simsimd_dot_i8(a: *const i8, b: *const i8, c: usize, d: *mut Distance);
    fn simsimd_dot_u8(a: *const u8, b: *const u8, c: usize, d: *mut Distance);
    fn simsimd_dot_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_dot_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_dot_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_dot_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_dot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_dot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_dot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_dot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_vdot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_vdot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_vdot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_vdot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_angular_i8(a: *const i8, b: *const i8, c: usize, d: *mut Distance);
    fn simsimd_angular_u8(a: *const u8, b: *const u8, c: usize, d: *mut Distance);
    fn simsimd_angular_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_angular_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_angular_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_angular_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_l2sq_i8(a: *const i8, b: *const i8, c: usize, d: *mut Distance);
    fn simsimd_l2sq_u8(a: *const u8, b: *const u8, c: usize, d: *mut Distance);
    fn simsimd_l2sq_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_l2sq_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_l2sq_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_l2sq_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_l2_i8(a: *const i8, b: *const i8, c: usize, d: *mut Distance);
    fn simsimd_l2_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_l2_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_l2_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_l2_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_hamming_b8(a: *const u8, b: *const u8, c: usize, d: *mut Distance);
    fn simsimd_jaccard_b8(a: *const u8, b: *const u8, c: usize, d: *mut Distance);

    fn simsimd_js_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_js_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_js_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_js_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_kl_f16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_kl_bf16(a: *const u16, b: *const u16, c: usize, d: *mut Distance);
    fn simsimd_kl_f32(a: *const f32, b: *const f32, c: usize, d: *mut Distance);
    fn simsimd_kl_f64(a: *const f64, b: *const f64, c: usize, d: *mut Distance);

    fn simsimd_intersect_u16(
        a: *const u16,
        b: *const u16,
        a_length: usize,
        b_length: usize,
        d: *mut Distance,
    );
    fn simsimd_intersect_u32(
        a: *const u32,
        b: *const u32,
        a_length: usize,
        b_length: usize,
        d: *mut Distance,
    );

    fn simsimd_uses_neon() -> i32;
    fn simsimd_uses_neon_f16() -> i32;
    fn simsimd_uses_neon_bf16() -> i32;
    fn simsimd_uses_neon_i8() -> i32;
    fn simsimd_uses_sve() -> i32;
    fn simsimd_uses_sve_f16() -> i32;
    fn simsimd_uses_sve_bf16() -> i32;
    fn simsimd_uses_sve_i8() -> i32;
    fn simsimd_uses_haswell() -> i32;
    fn simsimd_uses_skylake() -> i32;
    fn simsimd_uses_ice() -> i32;
    fn simsimd_uses_genoa() -> i32;
    fn simsimd_uses_sapphire() -> i32;
    fn simsimd_uses_turin() -> i32;
    fn simsimd_uses_sierra() -> i32;

    fn simsimd_flush_denormals() -> i32;
    fn simsimd_uses_dynamic_dispatch() -> i32;

    fn simsimd_f32_to_f16(f32_value: f32, result_ptr: *mut u16);
    fn simsimd_f16_to_f32(f16_ptr: *const u16) -> f32;
    fn simsimd_f32_to_bf16(f32_value: f32, result_ptr: *mut u16);
    fn simsimd_bf16_to_f32(bf16_ptr: *const u16) -> f32;
}

/// A half-precision (16-bit) floating point number.
///
/// This type represents IEEE 754 half-precision binary floating-point format.
/// It provides conversion methods to and from f32, and the underlying u16
/// representation is publicly accessible for direct bit manipulation.
///
/// # Examples
///
/// ```
/// use simsimd::f16;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct f16(pub u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = f16(0);

    /// Positive one.
    pub const ONE: Self = f16(0x3C00);

    /// Negative one.
    pub const NEG_ONE: Self = f16(0xBC00);

    /// Converts an f32 to f16 representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use simsimd::f16;
    /// let half = f16::from_f32(3.14159);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { simsimd_f32_to_f16(value, &mut result) };
        f16(result)
    }

    /// Converts the f16 to an f32.
    ///
    /// # Examples
    ///
    /// ```
    /// use simsimd::f16;
    /// let half = f16::from_f32(3.14159);
    /// let float = half.to_f32();
    /// ```
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        unsafe { simsimd_f16_to_f32(&self.0) }
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
impl std::fmt::Display for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// A brain floating point (bfloat16) number.
///
/// This type represents Google's bfloat16 format, which truncates IEEE 754
/// single-precision to 16 bits by keeping the exponent bits but reducing
/// the mantissa. This provides a wider range than f16 but lower precision.
///
/// # Examples
///
/// ```
/// use simsimd::bf16;
///
/// // Create from f32
/// let brain_half = bf16::from_f32(3.14);
///
/// // Convert back to f32
/// let float = brain_half.to_f32();
///
/// // Direct access to bits
/// let bits = brain_half.0;
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct bf16(pub u16);

impl bf16 {
    /// Positive zero.
    pub const ZERO: Self = bf16(0);

    /// Positive one.
    pub const ONE: Self = bf16(0x3F80);

    /// Negative one.
    pub const NEG_ONE: Self = bf16(0xBF80);

    /// Converts an f32 to bf16 representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use simsimd::bf16;
    /// let brain_half = bf16::from_f32(3.14159);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { simsimd_f32_to_bf16(value, &mut result) };
        bf16(result)
    }

    /// Converts the bf16 to an f32.
    ///
    /// # Examples
    ///
    /// ```
    /// use simsimd::bf16;
    /// let brain_half = bf16::from_f32(3.14159);
    /// let float = brain_half.to_f32();
    /// ```
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        unsafe { simsimd_bf16_to_f32(&self.0) }
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
impl std::fmt::Display for bf16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// The `capabilities` module provides functions for detecting the hardware features
/// available on the current system.
pub mod capabilities {

    pub fn uses_neon() -> bool {
        unsafe { crate::simsimd_uses_neon() != 0 }
    }

    pub fn uses_neon_f16() -> bool {
        unsafe { crate::simsimd_uses_neon_f16() != 0 }
    }

    pub fn uses_neon_bf16() -> bool {
        unsafe { crate::simsimd_uses_neon_bf16() != 0 }
    }

    pub fn uses_neon_i8() -> bool {
        unsafe { crate::simsimd_uses_neon_i8() != 0 }
    }

    pub fn uses_sve() -> bool {
        unsafe { crate::simsimd_uses_sve() != 0 }
    }

    pub fn uses_sve_f16() -> bool {
        unsafe { crate::simsimd_uses_sve_f16() != 0 }
    }

    pub fn uses_sve_bf16() -> bool {
        unsafe { crate::simsimd_uses_sve_bf16() != 0 }
    }

    pub fn uses_sve_i8() -> bool {
        unsafe { crate::simsimd_uses_sve_i8() != 0 }
    }

    pub fn uses_haswell() -> bool {
        unsafe { crate::simsimd_uses_haswell() != 0 }
    }

    pub fn uses_skylake() -> bool {
        unsafe { crate::simsimd_uses_skylake() != 0 }
    }

    pub fn uses_ice() -> bool {
        unsafe { crate::simsimd_uses_ice() != 0 }
    }

    pub fn uses_genoa() -> bool {
        unsafe { crate::simsimd_uses_genoa() != 0 }
    }

    pub fn uses_sapphire() -> bool {
        unsafe { crate::simsimd_uses_sapphire() != 0 }
    }

    pub fn uses_turin() -> bool {
        unsafe { crate::simsimd_uses_turin() != 0 }
    }

    pub fn uses_sierra() -> bool {
        unsafe { crate::simsimd_uses_sierra() != 0 }
    }

    /// Flushes denormalized numbers to zero on the current CPU architecture.
    ///
    /// This function should be called on each thread before any SIMD operations
    /// to avoid performance penalties. When facing denormalized values,
    /// Fused-Multiply-Add (FMA) operations can be up to 30x slower.
    ///
    /// # Returns
    ///
    /// Returns `true` if the operation was successful, `false` otherwise.
    pub fn flush_denormals() -> bool {
        unsafe { crate::simsimd_flush_denormals() != 0 }
    }

    /// Checks if the library is using dynamic dispatch for function selection.
    ///
    /// # Returns
    ///
    /// Returns `true` when the C backend is compiled with dynamic dispatch
    /// (default for this crate via `build.rs`), otherwise `false`.
    pub fn uses_dynamic_dispatch() -> bool {
        unsafe { crate::simsimd_uses_dynamic_dispatch() != 0 }
    }
}

/// `SpatialSimilarity` provides a set of trait methods for computing similarity
/// or distance between spatial data vectors in SIMD (Single Instruction, Multiple Data) context.
/// These methods can be used to calculate metrics like angular distance, dot product,
/// and squared Euclidean distance between two slices of data.
///
/// Each method takes two slices of data (a and b) and returns an Option<Distance>.
/// The result is `None` if the slices are not of the same length, as these operations
/// require one-to-one correspondence between the elements of the slices.
/// Otherwise, it returns the computed similarity or distance as `Some(f64)`.
/// Convenience methods like `cosine`/`sqeuclidean` delegate to the core methods
/// `cos`/`l2sq` implemented by this trait.
pub trait SpatialSimilarity
where
    Self: Sized,
{
    /// Computes the angular distance between two slices.
    /// The angular distance is a measure of similarity between two non-zero vectors
    /// of an dot product space that measures the cosine of the angle between them.
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the inner product (also known as dot product) between two slices.
    /// The dot product is the sum of the products of the corresponding entries
    /// of the two sequences of numbers.
    fn dot(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the squared Euclidean distance between two slices.
    /// The squared Euclidean distance is the sum of the squared differences
    /// between corresponding elements of the two slices.
    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the Euclidean distance between two slices.
    /// The Euclidean distance is the square root of
    //  sum of the squared differences between corresponding
    /// elements of the two slices.
    fn l2(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the squared Euclidean distance between two slices.
    /// The squared Euclidean distance is the sum of the squared differences
    /// between corresponding elements of the two slices.
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Distance> {
        SpatialSimilarity::l2sq(a, b)
    }

    /// Computes the Euclidean distance between two slices.
    /// The Euclidean distance is the square root of the
    /// sum of the squared differences between corresponding
    /// elements of the two slices.
    fn euclidean(a: &[Self], b: &[Self]) -> Option<Distance> {
        SpatialSimilarity::l2(a, b)
    }

    /// Computes the squared Euclidean distance between two slices.
    /// The squared Euclidean distance is the sum of the squared differences
    /// between corresponding elements of the two slices.
    fn inner(a: &[Self], b: &[Self]) -> Option<Distance> {
        SpatialSimilarity::dot(a, b)
    }
}

/// `BinarySimilarity` provides trait methods for computing similarity metrics
/// that are commonly used with binary data vectors, such as Hamming distance
/// and Jaccard index.
///
/// The methods accept two slices of binary data and return an Option<Distance>
/// indicating the computed similarity or distance, with `None` returned if the
/// slices differ in length.
pub trait BinarySimilarity
where
    Self: Sized,
{
    /// Computes the Hamming distance between two binary data slices.
    /// The Hamming distance between two strings of equal length is the number of
    /// bits at which the corresponding values are different.
    fn hamming(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the Jaccard index between two bitsets represented by binary data slices.
    /// The Jaccard index, also known as the Jaccard similarity coefficient, is a statistic
    /// used for gauging the similarity and diversity of sample sets.
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Distance>;
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
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance>;

    /// Computes the Kullback-Leibler divergence between two probability distributions.
    /// The Kullback-Leibler divergence is a measure of how one probability distribution
    /// diverges from a second, expected probability distribution.
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance>;
}

/// `ComplexProducts` provides trait methods for computing products between
/// complex number vectors. This includes standard and Hermitian dot products.
pub trait ComplexProducts
where
    Self: Sized,
{
    /// Computes the dot product between two complex number vectors.
    fn dot(a: &[Self], b: &[Self]) -> Option<ComplexProduct>;

    /// Computes the Hermitian dot product (conjugate dot product) between two complex number vectors.
    fn vdot(a: &[Self], b: &[Self]) -> Option<ComplexProduct>;
}

/// `Sparse` provides trait methods for sparse vectors.
pub trait Sparse
where
    Self: Sized,
{
    /// Computes the number of common elements between two sparse vectors.
    /// both vectors must be sorted in ascending order.
    fn intersect(a: &[Self], b: &[Self]) -> Option<Distance>;
}

impl BinarySimilarity for u8 {
    fn hamming(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_hamming_b8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn jaccard(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_jaccard_b8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for i8 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_i8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_i8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_i8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2_i8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl Sparse for u16 {
    fn intersect(a: &[Self], b: &[Self]) -> Option<Distance> {
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_intersect_u16(a.as_ptr(), b.as_ptr(), a.len(), b.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl Sparse for u32 {
    fn intersect(a: &[Self], b: &[Self]) -> Option<Distance> {
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_intersect_u32(a.as_ptr(), b.as_ptr(), a.len(), b.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for u8 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_u8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_u8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_u8(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for f16 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for bf16 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for f32 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl SpatialSimilarity for f64 {
    fn angular(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_angular_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn dot(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_dot_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2sq_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_l2_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl ProbabilitySimilarity for f16 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_js_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_kl_f16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl ProbabilitySimilarity for bf16 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_js_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }

        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_kl_bf16(a_ptr, b_ptr, a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl ProbabilitySimilarity for f32 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_js_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_kl_f32(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl ProbabilitySimilarity for f64 {
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_js_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }

    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Distance> {
        if a.len() != b.len() {
            return None;
        }
        let mut distance_value: Distance = 0.0;
        let distance_ptr: *mut Distance = &mut distance_value as *mut Distance;
        unsafe { simsimd_kl_f64(a.as_ptr(), b.as_ptr(), a.len(), distance_ptr) };
        Some(distance_value)
    }
}

impl ComplexProducts for f16 {
    fn dot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        // Prepare the output array where the real and imaginary parts will be stored
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        // Explicitly cast `*const f16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        unsafe { simsimd_dot_f16c(a_ptr, b_ptr, a.len(), product_ptr) };
        Some((product[0], product[1]))
    }

    fn vdot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        unsafe { simsimd_vdot_f16c(a_ptr, b_ptr, a.len(), product_ptr) };
        Some((product[0], product[1]))
    }
}

impl ComplexProducts for bf16 {
    fn dot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        // Prepare the output array where the real and imaginary parts will be stored
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        unsafe { simsimd_dot_bf16c(a_ptr, b_ptr, a.len(), product_ptr) };
        Some((product[0], product[1]))
    }

    fn vdot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        // Prepare the output array where the real and imaginary parts will be stored
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        // Explicitly cast `*const bf16` to `*const u16`
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;
        unsafe { simsimd_vdot_bf16c(a_ptr, b_ptr, a.len(), product_ptr) };
        Some((product[0], product[1]))
    }
}

impl ComplexProducts for f32 {
    fn dot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        unsafe { simsimd_dot_f32c(a.as_ptr(), b.as_ptr(), a.len(), product_ptr) };
        Some((product[0], product[1]))
    }

    fn vdot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        unsafe { simsimd_vdot_f32c(a.as_ptr(), b.as_ptr(), a.len(), product_ptr) };
        Some((product[0], product[1]))
    }
}

impl ComplexProducts for f64 {
    fn dot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        unsafe { simsimd_dot_f64c(a.as_ptr(), b.as_ptr(), a.len(), product_ptr) };
        Some((product[0], product[1]))
    }

    fn vdot(a: &[Self], b: &[Self]) -> Option<ComplexProduct> {
        if a.len() != b.len() {
            return None;
        }
        let mut product: [Distance; 2] = [0.0, 0.0];
        let product_ptr: *mut Distance = &mut product[0] as *mut _;
        unsafe { simsimd_vdot_f64c(a.as_ptr(), b.as_ptr(), a.len(), product_ptr) };
        Some((product[0], product[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16 as HalfBF16;
    use half::f16 as HalfF16;

    #[test]
    fn hardware_features_detection() {
        let uses_arm = capabilities::uses_neon() || capabilities::uses_sve();
        let uses_x86 = capabilities::uses_haswell()
            || capabilities::uses_skylake()
            || capabilities::uses_ice()
            || capabilities::uses_genoa()
            || capabilities::uses_sapphire()
            || capabilities::uses_turin();

        // The CPU can't simultaneously support ARM and x86 SIMD extensions
        if uses_arm {
            assert!(!uses_x86);
        }
        if uses_x86 {
            assert!(!uses_arm);
        }

        println!("- uses_neon: {}", capabilities::uses_neon());
        println!("- uses_sve: {}", capabilities::uses_sve());
        println!("- uses_haswell: {}", capabilities::uses_haswell());
        println!("- uses_skylake: {}", capabilities::uses_skylake());
        println!("- uses_ice: {}", capabilities::uses_ice());
        println!("- uses_genoa: {}", capabilities::uses_genoa());
        println!("- uses_sapphire: {}", capabilities::uses_sapphire());
        println!("- uses_turin: {}", capabilities::uses_turin());
        println!("- uses_sierra: {}", capabilities::uses_sierra());
    }

    //
    fn assert_almost_equal(left: Distance, right: Distance, tolerance: Distance) {
        let lower = right - tolerance;
        let upper = right + tolerance;

        assert!(left >= lower && left <= upper);
    }

    #[test]
    fn angular_i8() {
        let a: &[i8; 3] = &[3, 97, 127];
        let b: &[i8; 3] = &[3, 97, 127];

        if let Some(result) = SpatialSimilarity::angular(a, b) {
            println!("The result of cos_i8 is {:.8}", result);
            assert_almost_equal(0.00012027938, result, 0.01);
        }
    }

    #[test]
    fn angular_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SpatialSimilarity::angular(a, b) {
            println!("The result of cos_f32 is {:.8}", result);
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn dot_i8() {
        let a: &[i8; 3] = &[1, 2, 3];
        let b: &[i8; 3] = &[4, 5, 6];

        if let Some(result) = SpatialSimilarity::dot(a, b) {
            println!("The result of dot_i8 is {:.8}", result);
            assert_almost_equal(32.0, result, 0.01);
        }
    }

    #[test]
    fn dot_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SpatialSimilarity::dot(a, b) {
            println!("The result of dot_f32 is {:.8}", result);
            assert_almost_equal(32.0, result, 0.01);
        }
    }

    #[test]
    fn dot_f32_complex() {
        // Let's consider these as complex numbers where every pair is (real, imaginary)
        let a: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0]; // Represents two complex numbers: 1+2i, 3+4i
        let b: &[f32; 4] = &[5.0, 6.0, 7.0, 8.0]; // Represents two complex numbers: 5+6i, 7+8i

        if let Some((real, imag)) = ComplexProducts::dot(a, b) {
            println!(
                "The result of dot_f32_complex is real: {:.8}, imag: {:.8}",
                real, imag
            );
            // These values should be replaced with the expected real and imaginary parts of the result
            assert_almost_equal(-18.0, real, 0.01); // Corrected expected real part
            assert_almost_equal(68.0, imag, 0.01); // Corrected expected imaginary part
        }
    }

    #[test]
    fn vdot_f32_complex() {
        // Here we're assuming a similar setup to the previous test, but for the Hermitian (conjugate) dot product
        let a: &[f32; 4] = &[1.0, 2.0, 3.0, 4.0]; // Represents two complex numbers: 1+2i, 3+4i
        let b: &[f32; 4] = &[5.0, 6.0, 7.0, 8.0]; // Represents two complex numbers: 5+6i, 7+8i

        if let Some((real, imag)) = ComplexProducts::vdot(a, b) {
            println!(
                "The result of vdot_f32_complex is real: {:.8}, imag: {:.8}",
                real, imag
            );
            // Replace these with the actual expected values
            assert_almost_equal(70.0, real, 0.01); // Example expected real part
            assert_almost_equal(-8.0, imag, 0.01); // Example expected imaginary part
        }
    }

    #[test]
    fn l2sq_i8() {
        let a: &[i8; 3] = &[1, 2, 3];
        let b: &[i8; 3] = &[4, 5, 6];

        if let Some(result) = SpatialSimilarity::sqeuclidean(a, b) {
            println!("The result of l2sq_i8 is {:.8}", result);
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    #[test]
    fn l2sq_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SpatialSimilarity::sqeuclidean(a, b) {
            println!("The result of l2sq_f32 is {:.8}", result);
            assert_almost_equal(27.0, result, 0.01);
        }
    }

    #[test]
    fn l2_f32() {
        let a: &[f32; 3] = &[1.0, 2.0, 3.0];
        let b: &[f32; 3] = &[4.0, 5.0, 6.0];
        if let Some(result) = SpatialSimilarity::euclidean(a, b) {
            println!("The result of l2_f32 is {:.8}", result);
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    #[test]
    fn l2_f64() {
        let a: &[f64; 3] = &[1.0, 2.0, 3.0];
        let b: &[f64; 3] = &[4.0, 5.0, 6.0];
        if let Some(result) = SpatialSimilarity::euclidean(a, b) {
            println!("The result of l2_f64 is {:.8}", result);
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

        let a_simsimd: &[f16] =
            unsafe { std::slice::from_raw_parts(a_half.as_ptr() as *const f16, a_half.len()) };
        let b_simsimd: &[f16] =
            unsafe { std::slice::from_raw_parts(b_half.as_ptr() as *const f16, b_half.len()) };

        if let Some(result) = SpatialSimilarity::euclidean(&a_simsimd, &b_simsimd) {
            println!("The result of l2_f16 is {:.8}", result);
            assert_almost_equal(5.2, result, 0.01);
        }
    }

    #[test]
    fn l2_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SpatialSimilarity::euclidean(a, b) {
            println!("The result of l2_i8 is {:.8}", result);
            assert_almost_equal(5.2, result, 0.01);
        }
    }
    // Adding new tests for bit-level distances
    #[test]
    fn hamming_u8() {
        let a = &[0b01010101, 0b11110000, 0b10101010]; // Binary representations for clarity
        let b = &[0b01010101, 0b11110000, 0b10101010];

        if let Some(result) = BinarySimilarity::hamming(a, b) {
            println!("The result of hamming_u8 is {:.8}", result);
            assert_almost_equal(0.0, result, 0.01); // Perfect match
        }
    }

    #[test]
    fn jaccard_u8() {
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
    fn js_f32() {
        let a: &[f32; 3] = &[0.1, 0.9, 0.0];
        let b: &[f32; 3] = &[0.2, 0.8, 0.0];

        if let Some(result) = ProbabilitySimilarity::jensenshannon(a, b) {
            println!("The result of js_f32 is {:.8}", result);
            assert_almost_equal(0.099, result, 0.01); // Example value
        }
    }

    #[test]
    fn kl_f32() {
        let a: &[f32; 3] = &[0.1, 0.9, 0.0];
        let b: &[f32; 3] = &[0.2, 0.8, 0.0];

        if let Some(result) = ProbabilitySimilarity::kullbackleibler(a, b) {
            println!("The result of kl_f32 is {:.8}", result);
            assert_almost_equal(0.036, result, 0.01); // Example value
        }
    }

    #[test]
    fn angular_f16_same() {
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

        if let Some(result) = SpatialSimilarity::angular(a_f16, b_f16) {
            println!("The result of cos_f16 is {:.8}", result);
            assert_almost_equal(0.0, result, 0.01); // Example value, adjust according to actual expected value
        }
    }

    #[test]
    fn angular_bf16_same() {
        // Assuming these u16 values represent bf16 bit patterns, and they are identical
        let a_u16: &[u16] = &[15360, 16384, 17408]; // Corresponding to some bf16 values
        let b_u16: &[u16] = &[15360, 16384, 17408]; // Same as above for simplicity

        // Reinterpret cast from &[u16] to &[bf16]
        // SAFETY: This is safe as long as the representations are guaranteed to be identical,
        // which they are for transparent structs wrapping the same type.
        let a_bf16: &[bf16] =
            unsafe { std::slice::from_raw_parts(a_u16.as_ptr() as *const bf16, a_u16.len()) };
        let b_bf16: &[bf16] =
            unsafe { std::slice::from_raw_parts(b_u16.as_ptr() as *const bf16, b_u16.len()) };

        if let Some(result) = SpatialSimilarity::angular(a_bf16, b_bf16) {
            println!("The result of cos_bf16 is {:.8}", result);
            assert_almost_equal(0.0, result, 0.01); // Example value, adjust according to actual expected value
        }
    }

    #[test]
    fn angular_f16_interop() {
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
        if let Some(result) = SpatialSimilarity::angular(a_simsimd, b_simsimd) {
            // Expected value might need adjustment depending on actual cosine functionality
            // Assuming identical vectors yield angular distance of 1.0
            println!("The result of cos_f16 (interop) is {:.8}", result);
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn angular_bf16_interop() {
        let a_half: Vec<HalfBF16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| HalfBF16::from_f32(x))
            .collect();
        let b_half: Vec<HalfBF16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| HalfBF16::from_f32(x))
            .collect();

        // SAFETY: This is safe as long as the memory representations are guaranteed to be identical,
        // which they are due to both being #[repr(transparent)] wrappers around u16.
        let a_simsimd: &[bf16] =
            unsafe { std::slice::from_raw_parts(a_half.as_ptr() as *const bf16, a_half.len()) };
        let b_simsimd: &[bf16] =
            unsafe { std::slice::from_raw_parts(b_half.as_ptr() as *const bf16, b_half.len()) };

        // Use the reinterpret-casted slices with your SpatialSimilarity implementation
        if let Some(result) = SpatialSimilarity::angular(a_simsimd, b_simsimd) {
            // Expected value might need adjustment depending on actual cosine functionality
            // Assuming identical vectors yield angular distance of 1.0
            println!("The result of cos_bf16 (interop) is {:.8}", result);
            assert_almost_equal(0.025, result, 0.01);
        }
    }

    #[test]
    fn intersect_u16() {
        {
            let a_u16: &[u16] = &[153, 16384, 17408];
            let b_u16: &[u16] = &[15360, 16384, 7408];

            if let Some(result) = Sparse::intersect(a_u16, b_u16) {
                println!("The result of intersect_u16 is {:.8}", result);
                assert_almost_equal(1.0, result, 0.0001);
            }
        }

        {
            let a_u16: &[u16] = &[153, 11638, 08];
            let b_u16: &[u16] = &[15360, 16384, 7408];

            if let Some(result) = Sparse::intersect(a_u16, b_u16) {
                println!("The result of intersect_u16 is {:.8}", result);
                assert_almost_equal(0.0, result, 0.0001);
            }
        }
    }

    #[test]
    fn intersect_u32() {
        {
            let a_u32: &[u32] = &[11, 153];
            let b_u32: &[u32] = &[11, 153, 7408, 16384];

            if let Some(result) = Sparse::intersect(a_u32, b_u32) {
                println!("The result of intersect_u32 is {:.8}", result);
                assert_almost_equal(2.0, result, 0.0001);
            }
        }

        {
            let a_u32: &[u32] = &[153, 7408, 11638];
            let b_u32: &[u32] = &[153, 7408, 11638];

            if let Some(result) = Sparse::intersect(a_u32, b_u32) {
                println!("The result of intersect_u32 is {:.8}", result);
                assert_almost_equal(3.0, result, 0.0001);
            }
        }
    }

    #[test]
    fn f16_arithmetic() {
        let a = f16::from_f32(3.5);
        let b = f16::from_f32(2.0);

        // Test basic arithmetic
        assert!((a + b).to_f32() - 5.5 < 0.01);
        assert!((a - b).to_f32() - 1.5 < 0.01);
        assert!((a * b).to_f32() - 7.0 < 0.01);
        assert!((a / b).to_f32() - 1.75 < 0.01);
        assert!((-a).to_f32() + 3.5 < 0.01);

        // Test constants
        assert!(f16::ZERO.to_f32() == 0.0);
        assert!((f16::ONE.to_f32() - 1.0).abs() < 0.01);
        assert!((f16::NEG_ONE.to_f32() + 1.0).abs() < 0.01);

        // Test comparisons
        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        // Test utility methods
        assert!((-a).abs().to_f32() - 3.5 < 0.01);
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn bf16_arithmetic() {
        let a = bf16::from_f32(3.5);
        let b = bf16::from_f32(2.0);

        // Test basic arithmetic
        assert!((a + b).to_f32() - 5.5 < 0.1);
        assert!((a - b).to_f32() - 1.5 < 0.1);
        assert!((a * b).to_f32() - 7.0 < 0.1);
        assert!((a / b).to_f32() - 1.75 < 0.1);
        assert!((-a).to_f32() + 3.5 < 0.1);

        // Test constants
        assert!(bf16::ZERO.to_f32() == 0.0);
        assert!((bf16::ONE.to_f32() - 1.0).abs() < 0.01);
        assert!((bf16::NEG_ONE.to_f32() + 1.0).abs() < 0.01);

        // Test comparisons
        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        // Test utility methods
        assert!((-a).abs().to_f32() - 3.5 < 0.1);
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }
}
