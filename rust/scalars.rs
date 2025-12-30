//! Half-precision and 8-bit floating point scalar types.
//!
//! This module provides portable scalar types for mixed-precision computing:
//!
//! - [`f16`]: IEEE 754 half-precision (16-bit) floating point
//! - [`bf16`]: Brain floating point (bfloat16) - truncated single precision
//! - [`e4m3`]: 8-bit floating point with 4 exponent, 3 mantissa bits (OCP FP8)
//! - [`e5m2`]: 8-bit floating point with 5 exponent, 2 mantissa bits (OCP FP8)
//!
//! All types support standard arithmetic operations via trait implementations
//! and conversion to/from `f32`.

#![allow(non_camel_case_types)]

#[link(name = "numkong")]
extern "C" {
    fn nk_f32_to_f16(src: *const f32, dest: *mut u16);
    fn nk_f16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_bf16(src: *const f32, dest: *mut u16);
    fn nk_bf16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_e4m3(src: *const f32, dest: *mut u8);
    fn nk_e4m3_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e5m2(src: *const f32, dest: *mut u8);
    fn nk_e5m2_to_f32(src: *const u8, dest: *mut f32);
}

/// Compatibility function for pre 1.85 Rust versions lacking `f32::abs`.
#[inline(always)]
pub(crate) fn f32_abs_compat(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
