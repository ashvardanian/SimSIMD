//! Spatial similarity: dot products, angular (cosine), and Euclidean distances.
//!
//! This module provides:
//!
//! - [`Dot`]: Inner product of two vectors
//! - [`Angular`]: Cosine distance (1 - cosine similarity)
//! - [`Euclidean`]: Squared L2 distance
//! - [`VDot`]: Complex-valued dot product
//! - [`Roots`]: Scalar square root and reciprocal square root
//! - [`SpatialSimilarity`]: Blanket trait combining `Dot + Angular + Euclidean`

use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, u4x2};

#[link(name = "numkong")]
extern "C" {
    // Scalar roots
    fn nk_f32_sqrt(x: f32) -> f32;
    fn nk_f32_rsqrt(x: f32) -> f32;
    fn nk_f64_sqrt(x: f64) -> f64;
    fn nk_f64_rsqrt(x: f64) -> f64;
    fn nk_f16_sqrt(x: u16) -> u16;
    fn nk_f16_rsqrt(x: u16) -> u16;

    // Vector dot products
    fn nk_dot_i8(a: *const i8, b: *const i8, c: usize, d: *mut i32);
    fn nk_dot_u8(a: *const u8, b: *const u8, c: usize, d: *mut u32);
    fn nk_dot_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_dot_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_dot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_dot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_vdot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_vdot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_vdot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_vdot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    // Spatial similarity/distance functions
    fn nk_angular_i8(a: *const i8, b: *const i8, c: usize, d: *mut f32);
    fn nk_angular_u8(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_angular_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_angular_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_angular_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_sqeuclidean_i8(a: *const i8, b: *const i8, c: usize, d: *mut u32);
    fn nk_sqeuclidean_u8(a: *const u8, b: *const u8, c: usize, d: *mut u32);
    fn nk_sqeuclidean_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_sqeuclidean_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_sqeuclidean_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_euclidean_i8(a: *const i8, b: *const i8, c: usize, d: *mut f32);
    fn nk_euclidean_u8(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_euclidean_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_euclidean_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_euclidean_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    // 4-bit integer kernels
    fn nk_dot_i4(a: *const u8, b: *const u8, n: usize, result: *mut i32);
    fn nk_dot_u4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_sqeuclidean_i4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_sqeuclidean_u4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_euclidean_i4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_euclidean_u4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_angular_i4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_angular_u4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
}

// region: Scalar Roots

/// Scalar square-root and reciprocal-square-root operations backed by NumKong's exported kernels.
pub trait Roots: Sized {
    fn sqrt(self) -> Self;
    fn rsqrt(self) -> Self;
}

impl Roots for f32 {
    fn sqrt(self) -> Self { unsafe { nk_f32_sqrt(self) } }

    fn rsqrt(self) -> Self { unsafe { nk_f32_rsqrt(self) } }
}

impl Roots for f64 {
    fn sqrt(self) -> Self { unsafe { nk_f64_sqrt(self) } }

    fn rsqrt(self) -> Self { unsafe { nk_f64_rsqrt(self) } }
}

impl Roots for f16 {
    fn sqrt(self) -> Self { f16(unsafe { nk_f16_sqrt(self.0) }) }

    fn rsqrt(self) -> Self { f16(unsafe { nk_f16_rsqrt(self.0) }) }
}

// endregion: Scalar Roots

// region: Dot

/// Computes the **dot product** (inner product) between two vectors.
///
/// d = ∑ᵢ aᵢ × bᵢ
///
/// Range: unbounded. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
///
/// # Example
/// ```
/// use numkong::Dot;
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// let result = f32::dot(&a, &b).unwrap();
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
pub trait Dot: Sized {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `dot`.
    fn inner(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::dot(a, b) }
}

impl Dot for f64 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for f32 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
        unsafe { nk_dot_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
        unsafe { nk_dot_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
        let n = a.len() * 2; // Each i4x2 contains 2 elements
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
        let n = a.len() * 2; // Each u4x2 contains 2 elements
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

impl Dot for f16c {
    type Output = f32c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_dot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for bf16c {
    type Output = f32c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_dot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for f32c {
    type Output = f64c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_dot_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for f64c {
    type Output = f64c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_dot_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

// endregion: Dot

// region: Angular

/// Computes the **angular distance** (cosine distance) between two vectors.
///
/// d = 1 − (a · b) / (‖a‖ × ‖b‖)
///
/// Range: \[0, 2\]. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
pub trait Angular: Sized {
    type Output;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `angular`.
    fn cosine(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::angular(a, b) }
}

impl Angular for f64 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for f32 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
        unsafe { nk_angular_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
        unsafe { nk_angular_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
        let n = a.len() * 2; // Each i4x2 contains 2 elements
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
        let n = a.len() * 2; // Each u4x2 contains 2 elements
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
///
/// d = √(∑ᵢ (aᵢ − bᵢ)²)
///
/// Range: \[0, ∞). Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
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
        unsafe { nk_sqeuclidean_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for f32 {
    type SqEuclideanOutput = f64;
    type EuclideanOutput = f64;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
        unsafe { nk_sqeuclidean_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
        unsafe { nk_sqeuclidean_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
                a.len(),
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
        let n = a.len() * 2; // Each i4x2 contains 2 elements
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
        let n = a.len() * 2; // Each i4x2 contains 2 elements
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
        let n = a.len() * 2; // Each u4x2 contains 2 elements
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
        let n = a.len() * 2; // Each u4x2 contains 2 elements
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

// region: VDot

/// Computes the conjugating dot product.
///
/// For real-valued types this is identical to [`Dot::dot`]. For complex-valued types this
/// computes the Hermitian inner product `∑ᵢ conj(aᵢ) × bᵢ`.
pub trait VDot: Dot {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::dot(a, b) }
}

impl VDot for f64 {}
impl VDot for f32 {}
impl VDot for f16 {}
impl VDot for bf16 {}
impl VDot for i8 {}
impl VDot for u8 {}
impl VDot for e4m3 {}
impl VDot for e5m2 {}
impl VDot for e2m3 {}
impl VDot for e3m2 {}
impl VDot for i4x2 {}
impl VDot for u4x2 {}

impl VDot for f16c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_vdot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for bf16c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_vdot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for f32c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_vdot_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for f64c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_vdot_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

// endregion: VDot

/// `SpatialSimilarity` bundles spatial distance metrics: Dot, Angular, and Euclidean.
pub trait SpatialSimilarity: Dot + Angular + Euclidean {}
impl<T: Dot + Angular + Euclidean> SpatialSimilarity for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curved::Bilinear;
    use crate::types::{
        assert_close, bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, u4x2,
        FloatLike, StorageElement, TestableType,
    };

    /// Test a two-input metric: convert f32 inputs to T, call `op`, compare to `expected`.
    pub(crate) fn check_binary<T, R, F>(
        a_vals: &[f32],
        b_vals: &[f32],
        op: F,
        expected: f64,
        label: &str,
    ) where
        T: FloatLike + TestableType,
        R: FloatLike,
        F: FnOnce(&[T], &[T]) -> Option<R>,
    {
        let a: Vec<T> = a_vals.iter().map(|&v| T::from_f32(v)).collect();
        let b: Vec<T> = b_vals.iter().map(|&v| T::from_f32(v)).collect();
        let result = op(&a, &b).unwrap().to_f64();
        assert_close(
            result,
            expected,
            T::atol(),
            T::rtol(),
            &format!("{}<{}>", label, core::any::type_name::<T>()),
        );
    }

    // region: Dot Products

    fn check_dot<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Dot,
        T::Output: FloatLike,
    {
        check_binary::<T, T::Output, _>(a_vals, b_vals, T::dot, expected, "dot");
    }

    #[test]
    fn dot() {
        check_dot::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 6.0);
        check_dot::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
        check_dot::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
    }

    // endregion

    // region: Angular Distances

    fn check_angular<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Angular,
        T::Output: FloatLike,
    {
        check_binary::<T, T::Output, _>(a_vals, b_vals, T::angular, expected, "angular");
    }

    #[test]
    fn angular() {
        // angular([1,2,3],[4,5,6]) = 1 - 32/sqrt(14*77) ≈ 0.025368
        let expected = 1.0 - 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        check_angular::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        // e2m3 max is 7.5, use values in range
        let expected_e2m3 = 1.0 - 6.0 / (14.0_f64.sqrt() * 3.0_f64.sqrt());
        check_angular::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
    }

    // endregion

    // region: Euclidean Distances

    fn check_sqeuclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::SqEuclideanOutput: FloatLike,
    {
        check_binary::<T, T::SqEuclideanOutput, _>(
            a_vals,
            b_vals,
            T::sqeuclidean,
            expected,
            "sqeuclidean",
        );
    }

    fn check_euclidean<T>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + Euclidean,
        T::EuclideanOutput: FloatLike,
    {
        check_binary::<T, T::EuclideanOutput, _>(
            a_vals,
            b_vals,
            T::euclidean,
            expected,
            "euclidean",
        );
    }

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
        check_euclidean::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e2m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        let expected_packed = 54.0_f64.sqrt(); // i4x2 duplicates each value into both nibbles
        check_euclidean::<i4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
        check_euclidean::<u4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
    }

    // endregion

    // region: Complex Products

    trait ComplexValue {
        fn real(&self) -> f64;
        fn imag(&self) -> f64;
    }
    impl ComplexValue for f32c {
        fn real(&self) -> f64 { self.re as f64 }
        fn imag(&self) -> f64 { self.im as f64 }
    }
    impl ComplexValue for f64c {
        fn real(&self) -> f64 { self.re }
        fn imag(&self) -> f64 { self.im }
    }

    trait ComplexSample: Copy + StorageElement + Dot + VDot + Bilinear {
        fn from_real_imag(re: f32, im: f32) -> Self;
        fn atol() -> f64;
        fn rtol() -> f64;
    }

    impl ComplexSample for f16c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: f16::from_f32(re),
                im: f16::from_f32(im),
            }
        }
        fn atol() -> f64 { 5e-2 }
        fn rtol() -> f64 { 5e-2 }
    }

    impl ComplexSample for bf16c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: bf16::from_f32(re),
                im: bf16::from_f32(im),
            }
        }
        fn atol() -> f64 { 5e-2 }
        fn rtol() -> f64 { 5e-2 }
    }

    impl ComplexSample for f32c {
        fn from_real_imag(re: f32, im: f32) -> Self { Self { re, im } }
        fn atol() -> f64 { 1e-6 }
        fn rtol() -> f64 { 1e-6 }
    }

    impl ComplexSample for f64c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: re as f64,
                im: im as f64,
            }
        }
        fn atol() -> f64 { 1e-12 }
        fn rtol() -> f64 { 1e-12 }
    }

    /// Test a complex two-input operation with real + imaginary expected outputs.
    fn check_complex<T, R, F>(
        a: &[(f32, f32)],
        b: &[(f32, f32)],
        op: F,
        expected_re: f64,
        expected_im: f64,
        label: &str,
    ) where
        T: ComplexSample,
        R: ComplexValue,
        F: FnOnce(&[T], &[T]) -> Option<R>,
    {
        let a_t: Vec<T> = a
            .iter()
            .map(|&(re, im)| T::from_real_imag(re, im))
            .collect();
        let b_t: Vec<T> = b
            .iter()
            .map(|&(re, im)| T::from_real_imag(re, im))
            .collect();
        let result = op(&a_t, &b_t).unwrap();
        let tol = T::atol() + T::rtol() * expected_re.abs().max(expected_im.abs());
        assert_close(
            result.real(),
            expected_re,
            tol,
            0.0,
            &format!("{}<{}> real", label, core::any::type_name::<T>()),
        );
        assert_close(
            result.imag(),
            expected_im,
            tol,
            0.0,
            &format!("{}<{}> imag", label, core::any::type_name::<T>()),
        );
    }

    fn check_complex_dot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: ComplexSample,
        <T as Dot>::Output: ComplexValue,
    {
        let a_pairs: Vec<(f32, f32)> = a
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        let b_pairs: Vec<(f32, f32)> = b
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        check_complex::<T, <T as Dot>::Output, _>(
            &a_pairs,
            &b_pairs,
            <T as Dot>::dot,
            expected_re,
            expected_im,
            "complex_dot",
        );
    }

    fn check_complex_vdot<T>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        T: ComplexSample,
        <T as Dot>::Output: ComplexValue,
    {
        let a_pairs: Vec<(f32, f32)> = a
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        let b_pairs: Vec<(f32, f32)> = b
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        check_complex::<T, <T as Dot>::Output, _>(
            &a_pairs,
            &b_pairs,
            T::vdot,
            expected_re,
            expected_im,
            "complex_vdot",
        );
    }

    fn check_complex_bilinear_identity<T>(n: usize)
    where
        T: ComplexSample,
        <T as Bilinear>::Output: ComplexValue,
    {
        let mut a = vec![T::zero(); n];
        let mut b = vec![T::zero(); n];
        a[0] = T::one();
        b[0] = T::one();
        let mut c = vec![T::zero(); n * n];
        for i in 0..n {
            c[i * n + i] = T::one();
        }
        let result = T::bilinear(&a, &b, &c).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            result.real(),
            1.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> real", core::any::type_name::<T>()),
        );
        assert_close(
            result.imag(),
            0.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> imag", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn complex_dot_vdot_bilinear() {
        // [1+2i, 3+4i] · [5+6i, 7+8i]
        let a = &[1.0_f32, 2.0, 3.0, 4.0];
        let b = &[5.0_f32, 6.0, 7.0, 8.0];

        // dot: (-18, 68)
        check_complex_dot::<f64c>(a, b, -18.0, 68.0);
        check_complex_dot::<f32c>(a, b, -18.0, 68.0);
        check_complex_dot::<f16c>(a, b, -18.0, 68.0);
        check_complex_dot::<bf16c>(a, b, -18.0, 68.0);

        // vdot (conjugate): (70, -8)
        check_complex_vdot::<f64c>(a, b, 70.0, -8.0);
        check_complex_vdot::<f32c>(a, b, 70.0, -8.0);
        check_complex_vdot::<f16c>(a, b, 70.0, -8.0);
        check_complex_vdot::<bf16c>(a, b, 70.0, -8.0);

        // bilinear: identity matrix, unit vector → (1, 0)
        check_complex_bilinear_identity::<f64c>(4);
        check_complex_bilinear_identity::<f32c>(4);
        check_complex_bilinear_identity::<f16c>(4);
        check_complex_bilinear_identity::<bf16c>(4);
    }

    // endregion
}
