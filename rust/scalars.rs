//! Scalar types and conversion trait for mixed-precision computing.
//!
//! This module provides portable scalar types:
//!
//! - [`struct@f16`]: IEEE 754 half-precision (16-bit) floating point
//! - [`bf16`]: Brain floating point (bfloat16) - truncated single precision
//! - [`e4m3`]: 8-bit floating point with 4 exponent, 3 mantissa bits (OCP FP8)
//! - [`e5m2`]: 8-bit floating point with 5 exponent, 2 mantissa bits (OCP FP8)
//! - [`e2m3`]: 6-bit floating point with 2 exponent, 3 mantissa bits (padded to 8-bit)
//! - [`e3m2`]: 6-bit floating point with 3 exponent, 2 mantissa bits (padded to 8-bit)
//! - [`i4x2`]: Packed pair of signed 4-bit integers
//! - [`u4x2`]: Packed pair of unsigned 4-bit integers
//! - [`u1x8`]: Packed 8 binary values in a single byte
//!
//! All types support standard arithmetic operations and conversion to/from `f32`
//! via the [`FloatLike`] trait.

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

/// Compatibility function for pre 1.85 Rust versions lacking `f32::round`.
#[inline(always)]
pub(crate) fn f32_round_compat(x: f32) -> f32 {
    let t = x as i32 as f32;
    let d = x - t;
    if d >= 0.5 {
        t + 1.0
    } else if d <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// Compatibility function for pre 1.85 Rust versions lacking `f64::round`.
#[inline(always)]
pub(crate) fn f64_round_compat(x: f64) -> f64 {
    let t = x as i64 as f64;
    let d = x - t;
    if d >= 0.5 {
        t + 1.0
    } else if d <= -0.5 {
        t - 1.0
    } else {
        t
    }
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

    /// Converts an `f32` value to `f16`. Out-of-range values saturate.
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_f16(&value, &mut result) };
        f16(result)
    }

    /// Converts this value back to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_f16_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.to_f32().is_nan()
    }

    /// Returns true if this value is ±∞.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        self.to_f32().is_infinite()
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.to_f32().is_finite()
    }

    /// Returns |self|.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }

    /// Returns ⌊self⌋. Requires `std`.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Returns ⌈self⌉. Requires `std`.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Rounds to the nearest integer; half-way cases go away from zero. Requires `std`.
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

    /// Converts an `f32` value to `bf16`. Out-of-range values saturate.
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_bf16(&value, &mut result) };
        bf16(result)
    }

    /// Converts this value back to `f32`.
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

    /// Converts an `f32` value to `e4m3`. Out-of-range values saturate.
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e4m3(&value, &mut result) };
        e4m3(result)
    }

    /// Converts this value back to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e4m3_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7F) == 0x7F
    }

    /// Returns true if this value is ±∞. Always false for E4M3 (no infinities).
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        false
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

// region: FloatLike Trait

/// Trait for types that support conversion to/from f32 with classification and constants.
///
/// Provides a unified interface for all numeric types used in NumKong,
/// including half-precision floats, mini-floats, integers, and packed types.
pub trait FloatLike: Sized + Copy + Clone {
    /// Convert from f32 to this type.
    fn from_f32(v: f32) -> Self;
    /// Convert from this type to f32.
    fn to_f32(self) -> f32;
    /// Convert from f64 to this type (default: via f32 roundtrip).
    fn from_f64(v: f64) -> Self {
        Self::from_f32(v as f32)
    }
    /// Convert from this type to f64 (default: via f32 roundtrip).
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    fn abs(self) -> Self {
        Self::from_f32(f32_abs_compat(self.to_f32()))
    }
    fn is_nan(self) -> bool {
        self.to_f32().is_nan()
    }
    fn is_finite(self) -> bool {
        self.to_f32().is_finite()
    }
    fn is_infinite(self) -> bool {
        self.to_f32().is_infinite()
    }
    fn zero() -> Self {
        Self::from_f32(0.0)
    }
    fn one() -> Self {
        Self::from_f32(1.0)
    }

    fn has_infinity() -> bool {
        false
    }
    fn has_nan() -> bool {
        false
    }
    fn has_subnormals() -> bool {
        false
    }
    fn max_value() -> f32 {
        f32::MAX
    }
    fn min_positive() -> f32 {
        f32::MIN_POSITIVE
    }
}

impl FloatLike for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn abs(self) -> Self {
        f32_abs_compat(self)
    }
    fn is_nan(self) -> bool {
        self != self
    }
    fn is_finite(self) -> bool {
        (self != f32::INFINITY) && (self != f32::NEG_INFINITY) && !self.is_nan()
    }
    fn is_infinite(self) -> bool {
        self == f32::INFINITY || self == f32::NEG_INFINITY
    }
    fn has_infinity() -> bool {
        true
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        f32::MAX
    }
    fn min_positive() -> f32 {
        f32::MIN_POSITIVE
    }
}

impl FloatLike for f64 {
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn abs(self) -> Self {
        f64::from_bits(self.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }
    fn is_nan(self) -> bool {
        self != self
    }
    fn is_finite(self) -> bool {
        (self != f64::INFINITY) && (self != f64::NEG_INFINITY) && !self.is_nan()
    }
    fn is_infinite(self) -> bool {
        self == f64::INFINITY || self == f64::NEG_INFINITY
    }
    fn has_infinity() -> bool {
        true
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        f32::MAX
    }
    fn min_positive() -> f32 {
        f32::MIN_POSITIVE
    }
}

impl FloatLike for f16 {
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_infinity() -> bool {
        true
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        65504.0
    }
    fn min_positive() -> f32 {
        6.1e-5
    }
}

impl FloatLike for bf16 {
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_infinity() -> bool {
        true
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        3.4e38
    }
    fn min_positive() -> f32 {
        1.2e-38
    }
}

impl FloatLike for e4m3 {
    fn from_f32(v: f32) -> Self {
        e4m3::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        448.0
    }
    fn min_positive() -> f32 {
        0.001953125
    }
}

impl FloatLike for e5m2 {
    fn from_f32(v: f32) -> Self {
        e5m2::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_infinity() -> bool {
        true
    }
    fn has_nan() -> bool {
        true
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        57344.0
    }
    fn min_positive() -> f32 {
        0.00006103515625
    }
}

impl FloatLike for e2m3 {
    fn from_f32(v: f32) -> Self {
        e2m3::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        7.5
    }
    fn min_positive() -> f32 {
        0.0625
    }
}

impl FloatLike for e3m2 {
    fn from_f32(v: f32) -> Self {
        e3m2::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn has_subnormals() -> bool {
        true
    }
    fn max_value() -> f32 {
        28.0
    }
    fn min_positive() -> f32 {
        0.125
    }
}

impl FloatLike for i8 {
    fn from_f32(v: f32) -> Self {
        f32_round_compat(v) as i8
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        f64_round_compat(v) as i8
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for u8 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u8
        }
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u8
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for i32 {
    fn from_f32(v: f32) -> Self {
        f32_round_compat(v) as i32
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        f64_round_compat(v) as i32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for u32 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u32
        }
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u32
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for i16 {
    fn from_f32(v: f32) -> Self {
        f32_round_compat(v) as i16
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        f64_round_compat(v) as i16
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for u16 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u16
        }
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u16
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for i64 {
    fn from_f32(v: f32) -> Self {
        f32_round_compat(v) as i64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        f64_round_compat(v) as i64
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for u64 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u64
        }
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u64
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatLike for i4x2 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v) as i8;
        i4x2::from((r, r))
    }
    fn to_f32(self) -> f32 {
        let (a, _) = self.into();
        a as f32
    }
    fn one() -> Self {
        i4x2::from((1i8, 1i8))
    }
    fn zero() -> Self {
        i4x2::from((0i8, 0i8))
    }
}

impl FloatLike for u4x2 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        let r = if r < 0.0 { 0u8 } else { r as u8 };
        u4x2::from((r, r))
    }
    fn to_f32(self) -> f32 {
        let (a, _) = self.into();
        a as f32
    }
    fn one() -> Self {
        u4x2::from((1u8, 1u8))
    }
    fn zero() -> Self {
        u4x2::from((0u8, 0u8))
    }
}

impl FloatLike for u1x8 {
    fn from_f32(v: f32) -> Self {
        if v > 0.0 {
            u1x8(0xFF)
        } else {
            u1x8(0x00)
        }
    }
    fn to_f32(self) -> f32 {
        self.0.count_ones() as f32
    }
    fn one() -> Self {
        u1x8(0xFF)
    }
    fn zero() -> Self {
        u1x8(0x00)
    }
}

// endregion: FloatLike Trait

// region: TestableType Trait (test-only)

#[cfg(test)]
pub(crate) trait TestableType: FloatLike {
    /// Base absolute tolerance for this type.
    fn atol() -> f64;
    /// Base relative tolerance for this type.
    fn rtol() -> f64;
    fn dimensions_per_value() -> usize {
        1
    }
}

#[cfg(test)]
pub(crate) fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, msg: &str) {
    let tol = atol + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{}: expected {} but got {} (atol={}, rtol={}, tol={})",
        msg,
        expected,
        actual,
        atol,
        rtol,
        tol
    );
}

#[cfg(test)]
impl TestableType for f32 {
    fn atol() -> f64 {
        1e-4
    }
    fn rtol() -> f64 {
        1e-4
    }
}
#[cfg(test)]
impl TestableType for f64 {
    fn atol() -> f64 {
        1e-9
    }
    fn rtol() -> f64 {
        1e-9
    }
}
#[cfg(test)]
impl TestableType for f16 {
    fn atol() -> f64 {
        0.05
    }
    fn rtol() -> f64 {
        0.05
    }
}
#[cfg(test)]
impl TestableType for bf16 {
    fn atol() -> f64 {
        0.1
    }
    fn rtol() -> f64 {
        0.1
    }
}
#[cfg(test)]
impl TestableType for e4m3 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
#[cfg(test)]
impl TestableType for e5m2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.1
    }
}
#[cfg(test)]
impl TestableType for e2m3 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
#[cfg(test)]
impl TestableType for e3m2 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
#[cfg(test)]
impl TestableType for i8 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for u8 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for i32 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for u32 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for i16 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for u16 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for i64 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for u64 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
#[cfg(test)]
impl TestableType for i4x2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
    fn dimensions_per_value() -> usize {
        2
    }
}
#[cfg(test)]
impl TestableType for u4x2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
    fn dimensions_per_value() -> usize {
        2
    }
}
#[cfg(test)]
impl TestableType for u1x8 {
    fn atol() -> f64 {
        0.0
    }
    fn rtol() -> f64 {
        0.0
    }
    fn dimensions_per_value() -> usize {
        8
    }
}

// endregion: TestableType Trait

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_scalar_roundtrip<T: FloatLike>(original: f32, abs_tol: f32, rel_tol: f32) {
        let converted = T::from_f32(original);
        let roundtrip = FloatLike::to_f32(converted);
        if original == 0.0 {
            assert_eq!(roundtrip, 0.0, "Zero should roundtrip exactly");
            return;
        }
        let abs_error = f32_abs_compat(roundtrip - original);
        let rel_error = abs_error / f32_abs_compat(original);
        assert!(
            abs_error <= abs_tol || rel_error <= rel_tol,
            "Roundtrip failed for {}: got {} (abs_err={:.6}, rel_err={:.6})",
            original,
            roundtrip,
            abs_error,
            rel_error
        );
    }

    fn assert_scalar_almost_equal<T: FloatLike>(
        actual: f32,
        expected: f32,
        abs_tol: f32,
        rel_tol: f32,
        context: &str,
    ) {
        let abs_error = f32_abs_compat(actual - expected);
        let rel_error = if expected != 0.0 {
            abs_error / f32_abs_compat(expected)
        } else {
            abs_error
        };
        assert!(
            abs_error <= abs_tol || rel_error <= rel_tol,
            "{}: expected {} but got {} (abs_err={:.6}, rel_err={:.6})",
            context,
            expected,
            actual,
            abs_error,
            rel_error
        );
    }

    fn check_arithmetic<T>(a_val: f32, b_val: f32, abs_tol: f32, rel_tol: f32)
    where
        T: FloatLike
            + PartialOrd
            + PartialEq
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::Div<Output = T>
            + core::ops::Neg<Output = T>,
    {
        let a = T::from_f32(a_val);
        let b = T::from_f32(b_val);
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(a + b),
            a_val + b_val,
            abs_tol,
            rel_tol,
            "add",
        );
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(a - b),
            a_val - b_val,
            abs_tol,
            rel_tol,
            "sub",
        );
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(a * b),
            a_val * b_val,
            abs_tol,
            rel_tol,
            "mul",
        );
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(a / b),
            a_val / b_val,
            abs_tol,
            rel_tol,
            "div",
        );
        assert_scalar_almost_equal::<T>(FloatLike::to_f32(-a), -a_val, abs_tol, rel_tol, "neg");
        assert_eq!(FloatLike::to_f32(T::zero()), 0.0);
        assert_scalar_almost_equal::<T>(FloatLike::to_f32(T::one()), 1.0, abs_tol, rel_tol, "ONE");
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(T::from_f32(-1.0)),
            -1.0,
            abs_tol,
            rel_tol,
            "NEG_ONE",
        );
        assert!(a > b);
        assert!(a == a);
        assert_scalar_almost_equal::<T>(
            FloatLike::to_f32(FloatLike::abs(-a)),
            a_val,
            abs_tol,
            rel_tol,
            "abs",
        );
        assert!(FloatLike::is_finite(a));
        assert!(!FloatLike::is_nan(a));
        if T::has_infinity() {
            assert!(!FloatLike::is_infinite(a));
        }
    }

    fn check_roundtrip<T: FloatLike>(values: &[f32], abs_tol: f32, rel_tol: f32) {
        for &v in values {
            assert_scalar_roundtrip::<T>(v, abs_tol, rel_tol);
        }
    }

    fn check_edge_cases<T: FloatLike>() {
        assert_eq!(FloatLike::to_f32(T::from_f32(0.0)), 0.0);
        assert_eq!(FloatLike::to_f32(T::from_f32(-0.0)), 0.0);
        if T::has_infinity() {
            assert!(FloatLike::to_f32(T::from_f32(f32::INFINITY)).is_infinite());
            assert!(FloatLike::to_f32(T::from_f32(f32::NEG_INFINITY)).is_infinite());
        } else {
            assert!(!FloatLike::to_f32(T::from_f32(f32::INFINITY)).is_infinite());
        }
        if T::has_nan() {
            assert!(FloatLike::to_f32(T::from_f32(f32::NAN)).is_nan());
        } else {
            assert!(!FloatLike::to_f32(T::from_f32(f32::NAN)).is_nan());
        }
        let big = T::max_value() * 10.0;
        let overflow = T::from_f32(big);
        if T::has_infinity() {
            assert!(
                FloatLike::to_f32(overflow).is_infinite()
                    || FloatLike::to_f32(overflow) >= T::max_value()
            );
        } else {
            let v = FloatLike::to_f32(overflow);
            assert!(!v.is_infinite() && !v.is_nan());
            assert!(v <= T::max_value());
            let neg = FloatLike::to_f32(T::from_f32(-big));
            assert!(!neg.is_infinite() && !neg.is_nan());
            assert!(neg >= -T::max_value());
        }
    }

    fn check_subnormals<T: FloatLike>(values: &[f32], upper_bound: f32) {
        for &val in values {
            let roundtrip = FloatLike::to_f32(T::from_f32(val));
            assert!(
                roundtrip >= 0.0 && roundtrip < upper_bound,
                "{} subnormal test failed for {}: got {}",
                core::any::type_name::<T>(),
                val,
                roundtrip
            );
        }
    }

    #[test]
    fn arithmetic_ieee_halfs() {
        check_arithmetic::<f16>(3.5, 2.0, 0.002, 0.001);
        check_arithmetic::<bf16>(3.5, 2.0, 0.016, 0.008);
    }

    #[test]
    fn arithmetic_minifloats() {
        check_arithmetic::<e4m3>(2.0, 1.5, 0.25, 0.125);
        check_arithmetic::<e5m2>(2.0, 1.5, 0.5, 0.25);
        check_arithmetic::<e2m3>(2.0, 1.5, 0.25, 0.125);
        check_arithmetic::<e3m2>(4.0, 2.0, 0.5, 0.25);
    }

    #[test]
    fn roundtrip() {
        check_roundtrip::<f16>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 100.0, 1000.0, 10000.0, 0.001, 0.0001,
                0.00001, -100.0, -1000.0,
            ],
            0.002,
            0.001,
        );
        check_roundtrip::<bf16>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6, 0.001, 1e-6, -100.0, -1000.0,
            ],
            0.016,
            0.008,
        );
        check_roundtrip::<e4m3>(
            &[0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 128.0, 224.0],
            0.25,
            0.125,
        );
        check_roundtrip::<e5m2>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0,
            ],
            0.5,
            0.25,
        );
        check_roundtrip::<e2m3>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 0.25, -0.25, 0.125, -0.125,
                -4.0, -6.0, -7.5,
            ],
            0.25,
            0.125,
        );
        check_roundtrip::<e3m2>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, 28.0, 0.25, -0.25, -20.0,
                -28.0,
            ],
            0.5,
            0.25,
        );
    }

    #[test]
    fn edge_cases() {
        check_edge_cases::<f16>();
        check_edge_cases::<bf16>();
        check_edge_cases::<e4m3>();
        check_edge_cases::<e5m2>();
        check_edge_cases::<e2m3>();
        check_edge_cases::<e3m2>();
    }

    #[test]
    fn subnormals() {
        check_subnormals::<f16>(&[1e-5, 1e-6, 1e-7, 5e-6, 5e-7], 1e-4);
        check_subnormals::<bf16>(&[1e-39, 1e-40, 1e-42], 1e-37);
        check_subnormals::<e4m3>(&[0.001, 0.0005], 0.002);
        check_subnormals::<e5m2>(&[0.00005, 0.00003, 0.00001], 0.0001);
        check_subnormals::<e2m3>(&[0.03, 0.015], 0.07);
        check_subnormals::<e3m2>(&[0.0625, 0.03], 0.15);
    }

    #[test]
    fn half_crate_interop() {
        use half::bf16 as HalfBF16;
        use half::f16 as HalfF16;

        // f16: all 65536 bit patterns
        for bits in 0u16..=u16::MAX {
            let half_val = HalfF16::from_bits(bits).to_f32();
            let nk_val = f16(bits).to_f32();
            assert!(
                half_val.to_bits() == nk_val.to_bits() || (half_val.is_nan() && nk_val.is_nan()),
                "f16 mismatch at bits 0x{bits:04X}: half={half_val}, numkong={nk_val}"
            );
        }

        // bf16: all 65536 bit patterns
        for bits in 0u16..=u16::MAX {
            let half_val = HalfBF16::from_bits(bits).to_f32();
            let nk_val = bf16(bits).to_f32();
            assert!(
                half_val.to_bits() == nk_val.to_bits() || (half_val.is_nan() && nk_val.is_nan()),
                "bf16 mismatch at bits 0x{bits:04X}: half={half_val}, numkong={nk_val}"
            );
        }
    }
}
