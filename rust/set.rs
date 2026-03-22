//! Binary set similarity: Hamming and Jaccard distances.
//!
//! This module provides:
//!
//! - [`Hamming`]: Bit-level or byte-level Hamming distance
//! - [`Jaccard`]: Jaccard distance (1 - intersection/union)
//! - [`BinarySimilarity`]: Blanket trait combining `Hamming + Jaccard`

use crate::types::{u1x8, StorageElement};

#[link(name = "numkong")]
extern "C" {
    fn nk_hamming_u1(a: *const u8, b: *const u8, c: usize, d: *mut u32);
    fn nk_jaccard_u1(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_hamming_u8(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_jaccard_u16(a: *const u16, b: *const u16, n: usize, result: *mut f32);
    fn nk_jaccard_u32(a: *const u32, b: *const u32, n: usize, result: *mut f32);
}

// region: Hamming

/// Computes the **Hamming distance** between two binary vectors.
///
/// Counts differing bits (for `u1x8`) or differing bytes (for `u8`).
///
/// Range: \[0, n\]. Returns `None` if lengths differ.
///
/// Implemented for: `u1x8`, `u8`.
pub trait Hamming: StorageElement {
    type Output;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Hamming for u1x8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let n_bits = a.len() * 8; // Each u1x8 contains 8 bits
        unsafe {
            nk_hamming_u1(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Hamming for u8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_hamming_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

// endregion: Hamming

// region: Jaccard

/// Computes the **Jaccard distance** between two sets represented as bit/integer vectors.
///
/// d = 1 − |A ∩ B| / |A ∪ B|
///
/// Range: \[0, 1\]. Returns `None` if lengths differ.
///
/// Implemented for: `u1x8`, `u16`, `u32`.
pub trait Jaccard: StorageElement {
    type Output;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Jaccard for u1x8 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let n_bits = a.len() * 8; // Each u1x8 contains 8 bits
        unsafe {
            nk_jaccard_u1(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Jaccard for u16 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_u16(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Jaccard for u32 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_u32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

// endregion: Jaccard

/// `BinarySimilarity` bundles binary distance metrics: Hamming and Jaccard.
pub trait BinarySimilarity: Hamming + Jaccard {}
impl<T: Hamming + Jaccard> BinarySimilarity for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, u1x8};

    #[test]
    fn hamming() {
        // u1x8
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b00001111), u1x8(0b01010101)];
        assert_eq!(u1x8::hamming(&a, &b).unwrap(), 16);

        // u8
        let a: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let b: Vec<u8> = vec![0, 1, 2, 3, 0, 0, 0, 0];
        assert_eq!(u8::hamming(&a, &b).unwrap(), 4);
    }

    #[test]
    fn jaccard() {
        // u1x8 — identical
        let a = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let b = vec![u1x8(0b11110000), u1x8(0b10101010)];
        assert_close(
            u1x8::jaccard(&a, &b).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u1x8",
        );

        // u16 — identical
        let a: Vec<u16> = vec![1, 2, 3, 4];
        let b: Vec<u16> = vec![1, 2, 3, 4];
        assert_close(
            u16::jaccard(&a, &b).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u16 identical",
        );
        // u16 — disjoint
        let c: Vec<u16> = vec![5, 6, 7, 8];
        assert_close(
            u16::jaccard(&a, &c).unwrap() as f64,
            1.0,
            0.01,
            0.0,
            "jaccard_u16 disjoint",
        );

        // u32 — partial overlap
        let a: Vec<u32> = vec![1, 2, 3, 4];
        let b: Vec<u32> = vec![1, 2, 5, 6];
        assert_close(
            u32::jaccard(&a, &b).unwrap() as f64,
            0.5,
            0.01,
            0.0,
            "jaccard_u32",
        );
    }
}
