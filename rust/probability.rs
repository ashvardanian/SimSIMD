//! Probability measures: Kullback-Leibler divergence and Jensen-Shannon distance.
//!
//! This module provides:
//!
//! - [`KullbackLeibler`]: KL divergence between two distributions
//! - [`JensenShannon`]: Jensen-Shannon distance (symmetric metric)
//! - [`ProbabilitySimilarity`]: Blanket trait combining `KullbackLeibler + JensenShannon`

use crate::types::{bf16, f16};

#[link(name = "numkong")]
extern "C" {
    fn nk_jsd_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_jsd_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_jsd_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_jsd_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_kld_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_kld_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_kld_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_kld_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);
}

// region: KullbackLeibler

/// Computes the **Kullback-Leibler divergence** between two probability distributions.
///
/// D_KL(P‖Q) = ∑ᵢ pᵢ × ln(pᵢ / qᵢ)
///
/// Range: \[0, ∞). Not symmetric. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait KullbackLeibler: Sized {
    type Output;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `kullbackleibler`.
    fn kl(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::kullbackleibler(a, b) }
}

impl KullbackLeibler for f64 {
    type Output = f64;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl KullbackLeibler for f32 {
    type Output = f64;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl KullbackLeibler for f16 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_kld_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &mut result,
            )
        };
        Some(result)
    }
}

impl KullbackLeibler for bf16 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_kld_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: KullbackLeibler

// region: JensenShannon

/// Computes the **Jensen-Shannon distance** between two probability distributions.
///
/// d_JS(P, Q) = √(½(D_KL(P‖M) + D_KL(Q‖M))), where M = (P + Q) / 2
///
/// Range: \[0, √ln2\]. Symmetric. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`.
pub trait JensenShannon: Sized {
    type Output;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `jensenshannon`.
    fn js(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::jensenshannon(a, b) }
}

impl JensenShannon for f64 {
    type Output = f64;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl JensenShannon for f32 {
    type Output = f64;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl JensenShannon for f16 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jsd_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &mut result,
            )
        };
        Some(result)
    }
}

impl JensenShannon for bf16 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jsd_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: JensenShannon

/// `ProbabilitySimilarity` bundles probability divergence metrics: KullbackLeibler and JensenShannon.
pub trait ProbabilitySimilarity: KullbackLeibler + JensenShannon {}
impl<T: KullbackLeibler + JensenShannon> ProbabilitySimilarity for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, bf16, f16, FloatLike, NumberLike, TestableType};

    fn check_kld<T>(a: &[f32], b: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + KullbackLeibler,
        T::Output: FloatLike,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = T::kullbackleibler(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        assert_close(
            result,
            expected,
            T::atol().max(1e-6),
            T::rtol().max(1e-6),
            &format!("kld<{}>", core::any::type_name::<T>()),
        );
    }

    fn check_jsd<T>(a: &[f32], b: &[f32], expected: f64)
    where
        T: FloatLike + TestableType + JensenShannon,
        T::Output: FloatLike,
    {
        let a_t: Vec<T> = a.iter().map(|&v| T::from_f32(v)).collect();
        let b_t: Vec<T> = b.iter().map(|&v| T::from_f32(v)).collect();
        let result = T::jensenshannon(&a_t, &b_t).unwrap().to_f64();
        // Divergences involve ln() so need wider tolerance than simple dot products
        assert_close(
            result,
            expected,
            T::atol().max(1e-6),
            T::rtol().max(1e-6),
            &format!("jsd<{}>", core::any::type_name::<T>()),
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
}
