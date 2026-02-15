//! Half-precision and 8-bit floating point scalar types.
//!
//! This module provides portable scalar types for mixed-precision computing:
//!
//! - [`f16`]: IEEE 754 half-precision (16-bit) floating point
//! - [`bf16`]: Brain floating point (bfloat16) - truncated single precision
//! - [`e4m3`]: 8-bit floating point with 4 exponent, 3 mantissa bits (OCP FP8)
//! - [`e5m2`]: 8-bit floating point with 5 exponent, 2 mantissa bits (OCP FP8)
//! - [`e2m3`]: 6-bit floating point with 2 exponent, 3 mantissa bits (padded to 8-bit)
//! - [`e3m2`]: 6-bit floating point with 3 exponent, 2 mantissa bits (padded to 8-bit)
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
    fn nk_f32_to_e2m3(src: *const f32, dest: *mut u8);
    fn nk_e2m3_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e3m2(src: *const f32, dest: *mut u8);
    fn nk_e3m2_to_f32(src: *const u8, dest: *mut f32);
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

// region: e2m3 Type

/// A 6-bit floating point number in E2M3FN format (padded to 8-bit).
///
/// E2M3FN uses 1 sign bit, 2 exponent bits, and 3 mantissa bits. It provides
/// higher precision than E3M2 but narrower dynamic range. Note: E2M3FN has no
/// infinities, using those bit patterns for NaN instead.
///
/// # Examples
///
/// ```
/// use numkong::e2m3;
///
/// let fp6 = e2m3::from_f32(2.5);
/// let float = fp6.to_f32();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct e2m3(pub u8);

impl e2m3 {
    /// Positive zero.
    pub const ZERO: Self = e2m3(0x00);

    /// Positive one.
    pub const ONE: Self = e2m3(0x08);

    /// Negative one.
    pub const NEG_ONE: Self = e2m3(0x28);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e2m3(&value, &mut result) };
        e2m3(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e2m3_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    /// E2M3FN has no infinities - all special values are NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        false // E2M3FN has no NaN representation
    }

    /// Returns true if this value is positive or negative infinity.
    /// E2M3FN format has no infinities.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        false
    }

    /// Returns true if this number is neither infinite nor NaN.
    /// Note: E2M3FN format has no infinities or NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        true
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        e2m3(self.0 & 0x1F) // Clear sign bit
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
impl core::fmt::Display for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for e2m3 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: e2m3 Type

// region: e3m2 Type

/// A 6-bit floating point number in E3M2FN format (padded to 8-bit).
///
/// E3M2FN uses 1 sign bit, 3 exponent bits, and 2 mantissa bits. It provides
/// wider dynamic range than E2M3 but lower precision. Unlike E2M3FN, E3M2FN
/// supports infinities.
///
/// # Examples
///
/// ```
/// use numkong::e3m2;
///
/// let fp6 = e3m2::from_f32(2.5);
/// let float = fp6.to_f32();
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct e3m2(pub u8);

impl e3m2 {
    /// Positive zero.
    pub const ZERO: Self = e3m2(0x00);

    /// Positive one.
    pub const ONE: Self = e3m2(0x0C);

    /// Negative one.
    pub const NEG_ONE: Self = e3m2(0x2C);

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e3m2(&value, &mut result) };
        e3m2(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e3m2_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        let mant = self.0 & 0x03;
        exp == 0x07 && mant != 0
    }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        let mant = self.0 & 0x03;
        exp == 0x07 && mant == 0
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        exp != 0x07
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        e3m2(self.0 & 0x1F) // Clear sign bit
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
impl core::fmt::Display for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::ops::Add for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl core::ops::Sub for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl core::ops::Mul for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl core::ops::Div for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl core::ops::Neg for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::from_f32(-self.to_f32())
    }
}

impl core::cmp::PartialOrd for e3m2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// endregion: e3m2 Type

// region: u1x8 Type

/// A packed 8-bit vector representing 8 binary (1-bit) values.
///
/// Used for Hamming and Jaccard distance on binary vectors.
/// Each `u1x8` holds 8 bits packed into a single byte (b0 = LSB, b7 = MSB).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct u1x8(pub u8);

impl u1x8 {
    /// Create from raw packed bits.
    #[inline(always)]
    pub const fn new(bits: u8) -> Self {
        u1x8(bits)
    }

    /// Get the raw packed bits.
    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Construct from 8 booleans (b0 = LSB, b7 = MSB).
    #[inline(always)]
    pub const fn from_bools(
        b0: bool,
        b1: bool,
        b2: bool,
        b3: bool,
        b4: bool,
        b5: bool,
        b6: bool,
        b7: bool,
    ) -> Self {
        u1x8(
            (b0 as u8)
                | ((b1 as u8) << 1)
                | ((b2 as u8) << 2)
                | ((b3 as u8) << 3)
                | ((b4 as u8) << 4)
                | ((b5 as u8) << 5)
                | ((b6 as u8) << 6)
                | ((b7 as u8) << 7),
        )
    }

    /// Extract to 8 booleans (b0 = LSB, b7 = MSB).
    #[inline(always)]
    pub const fn to_bools(self) -> (bool, bool, bool, bool, bool, bool, bool, bool) {
        (
            (self.0 & 1) != 0,
            (self.0 & 2) != 0,
            (self.0 & 4) != 0,
            (self.0 & 8) != 0,
            (self.0 & 16) != 0,
            (self.0 & 32) != 0,
            (self.0 & 64) != 0,
            (self.0 & 128) != 0,
        )
    }
}

impl From<(bool, bool, bool, bool, bool, bool, bool, bool)> for u1x8 {
    #[inline(always)]
    fn from(b: (bool, bool, bool, bool, bool, bool, bool, bool)) -> Self {
        u1x8::from_bools(b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7)
    }
}

impl From<u1x8> for (bool, bool, bool, bool, bool, bool, bool, bool) {
    #[inline(always)]
    fn from(v: u1x8) -> Self {
        v.to_bools()
    }
}

// endregion: u1x8 Type

// region: u4x2 Type

/// A packed byte containing two unsigned 4-bit values (0..15).
///
/// Used for dot products and spatial distances on 4-bit quantized vectors.
/// Layout: low nibble = first element, high nibble = second element.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct u4x2(pub u8);

impl u4x2 {
    /// Create from raw packed byte.
    #[inline(always)]
    pub const fn new(packed: u8) -> Self {
        u4x2(packed)
    }

    /// Get the raw packed byte.
    #[inline(always)]
    pub const fn packed(self) -> u8 {
        self.0
    }

    /// Construct from two u8 values with saturation to 0..15.
    #[inline(always)]
    pub const fn from_u8s(lo: u8, hi: u8) -> Self {
        let lo_sat = if lo > 15 { 15 } else { lo };
        let hi_sat = if hi > 15 { 15 } else { hi };
        u4x2(lo_sat | (hi_sat << 4))
    }

    /// Extract to two u8 values (0..15 each).
    #[inline(always)]
    pub const fn to_u8s(self) -> (u8, u8) {
        (self.0 & 0x0F, self.0 >> 4)
    }
}

impl From<(u8, u8)> for u4x2 {
    #[inline(always)]
    fn from(v: (u8, u8)) -> Self {
        u4x2::from_u8s(v.0, v.1)
    }
}

impl From<u4x2> for (u8, u8) {
    #[inline(always)]
    fn from(v: u4x2) -> Self {
        v.to_u8s()
    }
}

// endregion: u4x2 Type

// region: i4x2 Type

/// A packed byte containing two signed 4-bit values (-8..7).
///
/// Used for dot products and spatial distances on 4-bit quantized vectors.
/// Layout: low nibble = first element, high nibble = second element (two's complement).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct i4x2(pub u8);

impl i4x2 {
    /// Create from raw packed byte.
    #[inline(always)]
    pub const fn new(packed: u8) -> Self {
        i4x2(packed)
    }

    /// Get the raw packed byte.
    #[inline(always)]
    pub const fn packed(self) -> u8 {
        self.0
    }

    /// Construct from two i8 values with saturation to -8..7.
    #[inline(always)]
    pub const fn from_i8s(lo: i8, hi: i8) -> Self {
        let lo_sat = if lo < -8 {
            -8
        } else if lo > 7 {
            7
        } else {
            lo
        };
        let hi_sat = if hi < -8 {
            -8
        } else if hi > 7 {
            7
        } else {
            hi
        };
        i4x2(((lo_sat as u8) & 0x0F) | (((hi_sat as u8) & 0x0F) << 4))
    }

    /// Extract to two i8 values (sign-extended from 4 bits).
    #[inline(always)]
    pub const fn to_i8s(self) -> (i8, i8) {
        let lo = (self.0 & 0x0F) as i8;
        let hi = ((self.0 >> 4) & 0x0F) as i8;
        // Sign extend from 4 bits: if bit 3 is set, fill upper bits with 1s
        let lo = if lo & 0x08 != 0 { lo | (!0x0Fi8) } else { lo };
        let hi = if hi & 0x08 != 0 { hi | (!0x0Fi8) } else { hi };
        (lo, hi)
    }
}

impl From<(i8, i8)> for i4x2 {
    #[inline(always)]
    fn from(v: (i8, i8)) -> Self {
        i4x2::from_i8s(v.0, v.1)
    }
}

impl From<i4x2> for (i8, i8) {
    #[inline(always)]
    fn from(v: i4x2) -> Self {
        v.to_i8s()
    }
}

// endregion: i4x2 Type

#[cfg(test)]
mod tests {
    use super::*;

    // Test-only trait for type-specific tolerance values
    #[allow(dead_code)]
    trait TestableScalar: Sized {
        const ABSOLUTE_TOLERANCE: f32;
        const RELATIVE_TOLERANCE: f32;
        const MAX_VALUE: f32;
        const MIN_VALUE: f32;
        const HAS_INFINITY: bool;
        const HAS_NAN: bool;
        const HAS_SUBNORMALS: bool;

        fn from_f32(v: f32) -> Self;
        fn to_f32(self) -> f32;
    }

    // f16: 10 mantissa bits, epsilon = 2^-10 = 0.000977
    impl TestableScalar for f16 {
        const ABSOLUTE_TOLERANCE: f32 = 0.002; // 2^(1-10) * 2 = 0.001953
        const RELATIVE_TOLERANCE: f32 = 0.001; // 2^(-10+1) * 2 = 0.000977 * 2
        const MAX_VALUE: f32 = 65504.0;
        const MIN_VALUE: f32 = 6.1e-5;
        const HAS_INFINITY: bool = true;
        const HAS_NAN: bool = true;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            f16::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // bf16: 7 mantissa bits, epsilon = 2^-7 = 0.0078125
    impl TestableScalar for bf16 {
        const ABSOLUTE_TOLERANCE: f32 = 0.016; // 2^(1-7) * 2 = 0.015625
        const RELATIVE_TOLERANCE: f32 = 0.008; // 2^(-7+1) * 2 = 0.015625
        const MAX_VALUE: f32 = 3.4e38;
        const MIN_VALUE: f32 = 1.2e-38;
        const HAS_INFINITY: bool = true;
        const HAS_NAN: bool = true;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            bf16::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // e4m3: 3 mantissa bits, epsilon = 2^-3 = 0.125
    impl TestableScalar for e4m3 {
        const ABSOLUTE_TOLERANCE: f32 = 0.25; // 2^(1-3) * 2 = 0.25
        const RELATIVE_TOLERANCE: f32 = 0.125; // 2^(-3+1) * 2 = 0.25
        const MAX_VALUE: f32 = 448.0;
        const MIN_VALUE: f32 = 0.001953125; // 2^-9
        const HAS_INFINITY: bool = false;
        const HAS_NAN: bool = true;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            e4m3::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // e5m2: 2 mantissa bits, epsilon = 2^-2 = 0.25
    impl TestableScalar for e5m2 {
        const ABSOLUTE_TOLERANCE: f32 = 0.5; // 2^(1-2) * 2 = 0.5
        const RELATIVE_TOLERANCE: f32 = 0.25; // 2^(-2+1) * 2 = 0.5
        const MAX_VALUE: f32 = 57344.0;
        const MIN_VALUE: f32 = 0.00006103515625; // 2^-14
        const HAS_INFINITY: bool = true;
        const HAS_NAN: bool = true;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            e5m2::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // e2m3: 3 mantissa bits, epsilon = 2^-3 = 0.125
    impl TestableScalar for e2m3 {
        const ABSOLUTE_TOLERANCE: f32 = 0.25; // 2^(1-3) * 2 = 0.25
        const RELATIVE_TOLERANCE: f32 = 0.125; // 2^(-3+1) * 2 = 0.25
        const MAX_VALUE: f32 = 7.5;
        const MIN_VALUE: f32 = 0.0625; // 2^-4
        const HAS_INFINITY: bool = false;
        const HAS_NAN: bool = false;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            e2m3::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // e3m2: 2 mantissa bits, epsilon = 2^-2 = 0.25
    impl TestableScalar for e3m2 {
        const ABSOLUTE_TOLERANCE: f32 = 0.5; // 2^(1-2) * 2 = 0.5
        const RELATIVE_TOLERANCE: f32 = 0.25; // 2^(-2+1) * 2 = 0.5
        const MAX_VALUE: f32 = 28.0;
        const MIN_VALUE: f32 = 0.125; // 2^-3
        const HAS_INFINITY: bool = false;
        const HAS_NAN: bool = false;
        const HAS_SUBNORMALS: bool = true;

        fn from_f32(v: f32) -> Self {
            e3m2::from_f32(v)
        }
        fn to_f32(self) -> f32 {
            self.to_f32()
        }
    }

    // Generic helper function for roundtrip testing
    fn assert_scalar_roundtrip<T: TestableScalar>(original: f32) {
        let converted = T::from_f32(original);
        let roundtrip = converted.to_f32();

        if original == 0.0 {
            assert_eq!(roundtrip, 0.0, "Zero should roundtrip exactly");
            return;
        }

        let abs_error = (roundtrip - original).abs();
        let rel_error = abs_error / original.abs();

        assert!(
            abs_error <= T::ABSOLUTE_TOLERANCE || rel_error <= T::RELATIVE_TOLERANCE,
            "Roundtrip failed for {}: got {} (abs_err={:.6}, rel_err={:.6}, abs_tol={:.6}, rel_tol={:.6})",
            original,
            roundtrip,
            abs_error,
            rel_error,
            T::ABSOLUTE_TOLERANCE,
            T::RELATIVE_TOLERANCE
        );
    }

    // Generic helper function for approximate equality testing
    fn assert_scalar_almost_equal<T: TestableScalar>(actual: f32, expected: f32, context: &str) {
        let abs_error = (actual - expected).abs();
        let rel_error = if expected != 0.0 {
            abs_error / expected.abs()
        } else {
            abs_error
        };

        assert!(
            abs_error <= T::ABSOLUTE_TOLERANCE || rel_error <= T::RELATIVE_TOLERANCE,
            "{}: expected {} but got {} (abs_err={:.6}, rel_err={:.6}, abs_tol={:.6}, rel_tol={:.6})",
            context,
            expected,
            actual,
            abs_error,
            rel_error,
            T::ABSOLUTE_TOLERANCE,
            T::RELATIVE_TOLERANCE
        );
    }

    #[test]
    fn f16_arithmetic() {
        let a = f16::from_f32(3.5);
        let b = f16::from_f32(2.0);

        assert_scalar_almost_equal::<f16>((a + b).to_f32(), 5.5, "addition");
        assert_scalar_almost_equal::<f16>((a - b).to_f32(), 1.5, "subtraction");
        assert_scalar_almost_equal::<f16>((a * b).to_f32(), 7.0, "multiplication");
        assert_scalar_almost_equal::<f16>((a / b).to_f32(), 1.75, "division");
        assert_scalar_almost_equal::<f16>((-a).to_f32(), -3.5, "negation");

        assert_eq!(f16::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<f16>(f16::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<f16>(f16::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<f16>((-a).abs().to_f32(), 3.5, "abs");
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn f16_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 100.0, 1000.0, 10000.0, 0.001, 0.0001,
            0.00001, -100.0, -1000.0,
        ];
        for &val in &test_values {
            assert_scalar_roundtrip::<f16>(val);
        }
    }

    #[test]
    fn bf16_arithmetic() {
        let a = bf16::from_f32(3.5);
        let b = bf16::from_f32(2.0);

        assert_scalar_almost_equal::<bf16>((a + b).to_f32(), 5.5, "addition");
        assert_scalar_almost_equal::<bf16>((a - b).to_f32(), 1.5, "subtraction");
        assert_scalar_almost_equal::<bf16>((a * b).to_f32(), 7.0, "multiplication");
        assert_scalar_almost_equal::<bf16>((a / b).to_f32(), 1.75, "division");
        assert_scalar_almost_equal::<bf16>((-a).to_f32(), -3.5, "negation");

        assert_eq!(bf16::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<bf16>(bf16::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<bf16>(bf16::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<bf16>((-a).abs().to_f32(), 3.5, "abs");
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn bf16_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6, 0.001, 1e-6, -100.0, -1000.0,
        ];
        for &val in &test_values {
            assert_scalar_roundtrip::<bf16>(val);
        }
    }

    #[test]
    fn e4m3_arithmetic() {
        let a = e4m3::from_f32(2.0);
        let b = e4m3::from_f32(1.5);

        assert_scalar_almost_equal::<e4m3>((a + b).to_f32(), 3.5, "addition");
        assert_scalar_almost_equal::<e4m3>((a - b).to_f32(), 0.5, "subtraction");
        assert_scalar_almost_equal::<e4m3>((a * b).to_f32(), 3.0, "multiplication");
        assert_scalar_almost_equal::<e4m3>((a / b).to_f32(), 1.333, "division");
        assert_scalar_almost_equal::<e4m3>((-a).to_f32(), -2.0, "negation");

        assert_eq!(e4m3::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<e4m3>(e4m3::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<e4m3>(e4m3::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<e4m3>((-a).abs().to_f32(), 2.0, "abs");
        assert!(a.is_finite());
        assert!(!a.is_nan());
        // e4m3 has no infinities (no is_infinite() method available)
    }

    #[test]
    fn e5m2_arithmetic() {
        let a = e5m2::from_f32(2.0);
        let b = e5m2::from_f32(1.5);

        assert_scalar_almost_equal::<e5m2>((a + b).to_f32(), 3.5, "addition");
        assert_scalar_almost_equal::<e5m2>((a - b).to_f32(), 0.5, "subtraction");
        assert_scalar_almost_equal::<e5m2>((a * b).to_f32(), 3.0, "multiplication");
        assert_scalar_almost_equal::<e5m2>((a / b).to_f32(), 1.333, "division");
        assert_scalar_almost_equal::<e5m2>((-a).to_f32(), -2.0, "negation");

        assert_eq!(e5m2::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<e5m2>(e5m2::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<e5m2>(e5m2::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<e5m2>((-a).abs().to_f32(), 2.0, "abs");
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
            assert_scalar_roundtrip::<e4m3>(val);
        }
    }

    #[test]
    fn e5m2_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0,
        ];
        for &val in &test_values {
            assert_scalar_roundtrip::<e5m2>(val);
        }
    }

    #[test]
    fn e2m3_arithmetic() {
        let a = e2m3::from_f32(2.0);
        let b = e2m3::from_f32(1.5);

        assert_scalar_almost_equal::<e2m3>((a + b).to_f32(), 3.5, "addition");
        assert_scalar_almost_equal::<e2m3>((a - b).to_f32(), 0.5, "subtraction");
        assert_scalar_almost_equal::<e2m3>((a * b).to_f32(), 3.0, "multiplication");
        assert_scalar_almost_equal::<e2m3>((a / b).to_f32(), 1.333, "division");
        assert_scalar_almost_equal::<e2m3>((-a).to_f32(), -2.0, "negation");

        assert_eq!(e2m3::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<e2m3>(e2m3::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<e2m3>(e2m3::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<e2m3>((-a).abs().to_f32(), 2.0, "abs");
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn e3m2_arithmetic() {
        let a = e3m2::from_f32(4.0);
        let b = e3m2::from_f32(2.0);

        assert_scalar_almost_equal::<e3m2>((a + b).to_f32(), 6.0, "addition");
        assert_scalar_almost_equal::<e3m2>((a - b).to_f32(), 2.0, "subtraction");
        assert_scalar_almost_equal::<e3m2>((a * b).to_f32(), 8.0, "multiplication");
        assert_scalar_almost_equal::<e3m2>((a / b).to_f32(), 2.0, "division");
        assert_scalar_almost_equal::<e3m2>((-a).to_f32(), -4.0, "negation");

        assert_eq!(e3m2::ZERO.to_f32(), 0.0);
        assert_scalar_almost_equal::<e3m2>(e3m2::ONE.to_f32(), 1.0, "ONE constant");
        assert_scalar_almost_equal::<e3m2>(e3m2::NEG_ONE.to_f32(), -1.0, "NEG_ONE constant");

        assert!(a > b);
        assert!(!(a < b));
        assert!(a == a);

        assert_scalar_almost_equal::<e3m2>((-a).abs().to_f32(), 4.0, "abs");
        assert!(a.is_finite());
        assert!(!a.is_nan());
        assert!(!a.is_infinite());
    }

    #[test]
    fn e2m3_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 0.25, -0.25, 0.125, -0.125,
            -4.0, -6.0, -7.5,
        ];
        for &val in &test_values {
            assert_scalar_roundtrip::<e2m3>(val);
        }
    }

    #[test]
    fn e3m2_roundtrip() {
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, 28.0, 0.25, -0.25, -20.0,
            -28.0,
        ];
        for &val in &test_values {
            assert_scalar_roundtrip::<e3m2>(val);
        }
    }

    #[test]
    fn f16_edge_cases() {
        // Zero
        assert_eq!(f16::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(f16::from_f32(-0.0).to_f32(), 0.0);

        // Infinity
        assert!(f16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(f16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());

        // NaN
        assert!(f16::from_f32(f32::NAN).to_f32().is_nan());

        // Overflow to infinity
        let overflow = f16::from_f32(100000.0);
        assert!(overflow.to_f32().is_infinite() || overflow.to_f32() >= 65504.0);
    }

    #[test]
    fn bf16_edge_cases() {
        // Zero
        assert_eq!(bf16::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(bf16::from_f32(-0.0).to_f32(), 0.0);

        // Infinity
        assert!(bf16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(bf16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());

        // NaN
        assert!(bf16::from_f32(f32::NAN).to_f32().is_nan());

        // Overflow to infinity
        let overflow = bf16::from_f32(f32::MAX);
        assert!(overflow.to_f32().is_infinite());
    }

    #[test]
    fn e4m3_edge_cases() {
        // Zero
        assert_eq!(e4m3::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(e4m3::from_f32(-0.0).to_f32(), 0.0);

        // NaN (e4m3 has NaN but no infinities)
        assert!(e4m3::from_f32(f32::NAN).to_f32().is_nan());

        // Overflow saturates to max value, not infinity
        let overflow = e4m3::from_f32(10000.0);
        let overflow_val = overflow.to_f32();
        assert!(!overflow_val.is_infinite());
        assert!(overflow_val <= 448.0);

        // Infinity input should not produce infinity output
        assert!(!e4m3::from_f32(f32::INFINITY).to_f32().is_infinite());
    }

    #[test]
    fn e5m2_edge_cases() {
        // Zero
        assert_eq!(e5m2::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(e5m2::from_f32(-0.0).to_f32(), 0.0);

        // Infinity (e5m2 supports infinity)
        assert!(e5m2::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(e5m2::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());

        // NaN
        assert!(e5m2::from_f32(f32::NAN).to_f32().is_nan());

        // Overflow to infinity
        let overflow = e5m2::from_f32(1e10);
        assert!(overflow.to_f32().is_infinite() || overflow.to_f32() >= 57344.0);
    }

    #[test]
    fn e2m3_edge_cases() {
        // Zero
        assert_eq!(e2m3::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(e2m3::from_f32(-0.0).to_f32(), 0.0);

        // e2m3 has no NaN or infinities, should saturate
        let overflow_pos = e2m3::from_f32(100.0);
        let overflow_val = overflow_pos.to_f32();
        assert!(!overflow_val.is_infinite());
        assert!(!overflow_val.is_nan());
        assert!(overflow_val <= 7.5);

        let overflow_neg = e2m3::from_f32(-100.0);
        let overflow_val_neg = overflow_neg.to_f32();
        assert!(!overflow_val_neg.is_infinite());
        assert!(!overflow_val_neg.is_nan());
        assert!(overflow_val_neg >= -7.5);

        // Infinity and NaN inputs should not produce infinity/NaN outputs
        assert!(!e2m3::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(!e2m3::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn e3m2_edge_cases() {
        // Zero
        assert_eq!(e3m2::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(e3m2::from_f32(-0.0).to_f32(), 0.0);

        // e3m2 has no NaN or infinities, should saturate
        let overflow_pos = e3m2::from_f32(1000.0);
        let overflow_val = overflow_pos.to_f32();
        assert!(!overflow_val.is_infinite());
        assert!(!overflow_val.is_nan());
        assert!(overflow_val <= 28.0);

        let overflow_neg = e3m2::from_f32(-1000.0);
        let overflow_val_neg = overflow_neg.to_f32();
        assert!(!overflow_val_neg.is_infinite());
        assert!(!overflow_val_neg.is_nan());
        assert!(overflow_val_neg >= -28.0);

        // Infinity and NaN inputs should not produce infinity/NaN outputs
        assert!(!e3m2::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(!e3m2::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn f16_subnormals() {
        // f16 normal minimum: ~6.1e-5
        // f16 subnormal minimum: ~6.0e-8
        // Test values in subnormal range
        let subnormal_values = [1e-5f32, 1e-6, 1e-7, 5e-6, 5e-7];

        for &val in &subnormal_values {
            let converted = f16::from_f32(val);
            let roundtrip = converted.to_f32();
            // Should preserve small value or round to zero
            // Subnormals should be non-negative and very small
            assert!(
                roundtrip >= 0.0 && roundtrip < 1e-4,
                "f16 subnormal test failed for {}: got {}",
                val,
                roundtrip
            );
        }
    }

    #[test]
    fn bf16_subnormals() {
        // bf16 normal minimum: ~1.2e-38
        // bf16 subnormal minimum: ~1.4e-45
        // Test values in subnormal range
        let subnormal_values = [1e-39f32, 1e-40, 1e-42];

        for &val in &subnormal_values {
            let converted = bf16::from_f32(val);
            let roundtrip = converted.to_f32();
            // Should preserve small value or round to zero
            // Subnormals should be non-negative and very small
            assert!(
                roundtrip >= 0.0 && roundtrip < 1e-37,
                "bf16 subnormal test failed for {}: got {}",
                val,
                roundtrip
            );
        }
    }
}
