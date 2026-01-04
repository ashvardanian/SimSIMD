//! Numeric traits and implementations for vector operations.
//!
//! This module provides hardware-accelerated implementations of:
//!
//! - **Spatial similarity**: [`Dot`], [`Angular`], [`Euclidean`]
//! - **Binary similarity**: [`Hamming`], [`Jaccard`]
//! - **Probability divergence**: [`KullbackLeibler`], [`JensenShannon`]
//! - **Complex products**: [`ComplexDot`], [`ComplexVDot`]
//! - **Elementwise operations**: [`Scale`], [`Sum`], [`WSum`], [`FMA`]
//! - **Trigonometry**: [`Sin`], [`Cos`], [`ATan`]
//! - **Geospatial**: [`Haversine`], [`Vincenty`]
//! - **Mesh alignment**: [`MeshAlignment`]
//! - **Sparse sets**: [`Sparse`]

use crate::scalars::{bf16, e4m3, e5m2, f16};

pub type ComplexProductF32 = (f32, f32);
pub type ComplexProductF64 = (f64, f64);

/// Size type used in C FFI to match `nk_size_t` which is always `uint64_t`.
type u64size = u64;

#[link(name = "numkong")]
extern "C" {
    // Capability detection
    fn nk_uses_neon() -> i32;
    fn nk_uses_neonhalf() -> i32;
    fn nk_uses_neonbfdot() -> i32;
    fn nk_uses_neonsdot() -> i32;
    fn nk_uses_sve() -> i32;
    fn nk_uses_svehalf() -> i32;
    fn nk_uses_svebfdot() -> i32;
    fn nk_uses_svesdot() -> i32;
    fn nk_uses_sve2() -> i32;
    fn nk_uses_sve2p1() -> i32;
    fn nk_uses_neonfhm() -> i32;
    fn nk_uses_sme() -> i32;
    fn nk_uses_sme2() -> i32;
    fn nk_uses_sme2p1() -> i32;
    fn nk_uses_smef64() -> i32;
    fn nk_uses_smehalf() -> i32;
    fn nk_uses_smebf16() -> i32;
    fn nk_uses_smelut2() -> i32;
    fn nk_uses_smefa64() -> i32;
    fn nk_uses_haswell() -> i32;
    fn nk_uses_skylake() -> i32;
    fn nk_uses_ice() -> i32;
    fn nk_uses_genoa() -> i32;
    fn nk_uses_sapphire() -> i32;
    fn nk_uses_turin() -> i32;
    fn nk_uses_sierra() -> i32;
    fn nk_uses_sapphire_amx() -> i32;
    fn nk_uses_granite_amx() -> i32;
    fn nk_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_dynamic_dispatch() -> i32;

    // Vector dot products
    fn nk_dot_i8(a: *const i8, b: *const i8, c: u64size, d: *mut i32);
    fn nk_dot_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_e4m3(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_e5m2(a: *const u8, b: *const u8, c: u64size, d: *mut f32);
    fn nk_dot_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_dot_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_dot_f16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_bf16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_dot_f32c(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_dot_f64c(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_vdot_f16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_vdot_bf16c(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_vdot_f32c(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_vdot_f64c(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    // Spatial similarity/distance functions
    fn nk_angular_i8(a: *const i8, b: *const i8, c: u64size, d: *mut f32);
    fn nk_angular_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_angular_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_angular_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_l2sq_i8(a: *const i8, b: *const i8, c: u64size, d: *mut u32);
    fn nk_l2sq_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2sq_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2sq_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_l2sq_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_l2_i8(a: *const i8, b: *const i8, c: u64size, d: *mut f32);
    fn nk_l2_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_l2_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_l2_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_hamming_u1(a: *const u8, b: *const u8, c: u64size, d: *mut u32);
    fn nk_jaccard_u1(a: *const u8, b: *const u8, c: u64size, d: *mut f32);

    // Probability distribution distances/divergences
    fn nk_jsd_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_jsd_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_jsd_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_jsd_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    fn nk_kld_f16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_kld_bf16(a: *const u16, b: *const u16, c: u64size, d: *mut f32);
    fn nk_kld_f32(a: *const f32, b: *const f32, c: u64size, d: *mut f32);
    fn nk_kld_f64(a: *const f64, b: *const f64, c: u64size, d: *mut f64);

    // Sparse sets
    fn nk_intersect_u16(
        a: *const u16,
        b: *const u16,
        a_length: u64size,
        b_length: u64size,
        d: *mut u32,
    );
    fn nk_intersect_u32(
        a: *const u32,
        b: *const u32,
        a_length: u64size,
        b_length: u64size,
        d: *mut u32,
    );

    // Trigonometry functions
    fn nk_sin_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_sin_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_cos_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_cos_f64(inputs: *const f64, n: u64size, outputs: *mut f64);
    fn nk_atan_f32(inputs: *const f32, n: u64size, outputs: *mut f32);
    fn nk_atan_f64(inputs: *const f64, n: u64size, outputs: *mut f64);

    // Elementwise operations
    fn nk_scale_f64(
        a: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_scale_f32(
        a: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_scale_f16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_bf16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_i8(a: *const i8, n: u64size, alpha: *const f32, beta: *const f32, result: *mut i8);
    fn nk_scale_u8(a: *const u8, n: u64size, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_scale_i16(
        a: *const i16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i16,
    );
    fn nk_scale_u16(
        a: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_scale_i32(
        a: *const i32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i32,
    );
    fn nk_scale_u32(
        a: *const u32,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u32,
    );
    fn nk_scale_i64(
        a: *const i64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i64,
    );
    fn nk_scale_u64(
        a: *const u64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u64,
    );

    fn nk_sum_f64(a: *const f64, b: *const f64, n: u64size, result: *mut f64);
    fn nk_sum_f32(a: *const f32, b: *const f32, n: u64size, result: *mut f32);
    fn nk_sum_f16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_bf16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_i8(a: *const i8, b: *const i8, n: u64size, result: *mut i8);
    fn nk_sum_u8(a: *const u8, b: *const u8, n: u64size, result: *mut u8);
    fn nk_sum_i16(a: *const i16, b: *const i16, n: u64size, result: *mut i16);
    fn nk_sum_u16(a: *const u16, b: *const u16, n: u64size, result: *mut u16);
    fn nk_sum_i32(a: *const i32, b: *const i32, n: u64size, result: *mut i32);
    fn nk_sum_u32(a: *const u32, b: *const u32, n: u64size, result: *mut u32);
    fn nk_sum_i64(a: *const i64, b: *const i64, n: u64size, result: *mut i64);
    fn nk_sum_u64(a: *const u64, b: *const u64, n: u64size, result: *mut u64);

    fn nk_wsum_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_wsum_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_wsum_f16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_wsum_bf16(
        a: *const u16,
        b: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_wsum_i8(
        a: *const i8,
        b: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_wsum_u8(
        a: *const u8,
        b: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    fn nk_fma_f64(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: u64size,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_fma_f32(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_fma_f16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_fma_bf16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_fma_i8(
        a: *const i8,
        b: *const i8,
        c: *const i8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_fma_u8(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: u64size,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

    // Mesh superposition metrics
    fn nk_rmsd_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_rmsd_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_kabsch_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_umeyama_f32(
        a: *const f32,
        b: *const f32,
        n: u64size,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_f64(
        a: *const f64,
        b: *const f64,
        n: u64size,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );

    // Geospatial distance functions
    fn nk_haversine_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: u64size,
        results: *mut f32,
    );
    fn nk_haversine_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: u64size,
        results: *mut f64,
    );
    fn nk_vincenty_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: u64size,
        results: *mut f32,
    );
    fn nk_vincenty_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: u64size,
        results: *mut f64,
    );
}

// region: Capabilities

/// Hardware capability detection functions.
pub mod capabilities {
    pub fn uses_neon() -> bool {
        unsafe { super::nk_uses_neon() != 0 }
    }
    pub fn uses_neonhalf() -> bool {
        unsafe { super::nk_uses_neonhalf() != 0 }
    }
    pub fn uses_neonbfdot() -> bool {
        unsafe { super::nk_uses_neonbfdot() != 0 }
    }
    pub fn uses_neonsdot() -> bool {
        unsafe { super::nk_uses_neonsdot() != 0 }
    }
    pub fn uses_sve() -> bool {
        unsafe { super::nk_uses_sve() != 0 }
    }
    pub fn uses_svehalf() -> bool {
        unsafe { super::nk_uses_svehalf() != 0 }
    }
    pub fn uses_svebfdot() -> bool {
        unsafe { super::nk_uses_svebfdot() != 0 }
    }
    pub fn uses_svesdot() -> bool {
        unsafe { super::nk_uses_svesdot() != 0 }
    }
    pub fn uses_sve2() -> bool {
        unsafe { super::nk_uses_sve2() != 0 }
    }
    pub fn uses_sve2p1() -> bool {
        unsafe { super::nk_uses_sve2p1() != 0 }
    }
    pub fn uses_neonfhm() -> bool {
        unsafe { super::nk_uses_neonfhm() != 0 }
    }
    pub fn uses_sme() -> bool {
        unsafe { super::nk_uses_sme() != 0 }
    }
    pub fn uses_sme2() -> bool {
        unsafe { super::nk_uses_sme2() != 0 }
    }
    pub fn uses_sme2p1() -> bool {
        unsafe { super::nk_uses_sme2p1() != 0 }
    }
    pub fn uses_smef64() -> bool {
        unsafe { super::nk_uses_smef64() != 0 }
    }
    pub fn uses_smehalf() -> bool {
        unsafe { super::nk_uses_smehalf() != 0 }
    }
    pub fn uses_smebf16() -> bool {
        unsafe { super::nk_uses_smebf16() != 0 }
    }
    pub fn uses_smelut2() -> bool {
        unsafe { super::nk_uses_smelut2() != 0 }
    }
    pub fn uses_smefa64() -> bool {
        unsafe { super::nk_uses_smefa64() != 0 }
    }
    pub fn uses_haswell() -> bool {
        unsafe { super::nk_uses_haswell() != 0 }
    }
    pub fn uses_skylake() -> bool {
        unsafe { super::nk_uses_skylake() != 0 }
    }
    pub fn uses_ice() -> bool {
        unsafe { super::nk_uses_ice() != 0 }
    }
    pub fn uses_genoa() -> bool {
        unsafe { super::nk_uses_genoa() != 0 }
    }
    pub fn uses_sapphire() -> bool {
        unsafe { super::nk_uses_sapphire() != 0 }
    }
    pub fn uses_turin() -> bool {
        unsafe { super::nk_uses_turin() != 0 }
    }
    pub fn uses_sierra() -> bool {
        unsafe { super::nk_uses_sierra() != 0 }
    }
    pub fn uses_sapphire_amx() -> bool {
        unsafe { super::nk_uses_sapphire_amx() != 0 }
    }
    pub fn uses_granite_amx() -> bool {
        unsafe { super::nk_uses_granite_amx() != 0 }
    }

    /// Flushes denormalized numbers to zero on the current CPU.
    pub fn configure_thread() -> bool {
        unsafe { super::nk_configure_thread(0) != 0 }
    }

    /// Returns `true` if the library uses dynamic dispatch for function selection.
    pub fn uses_dynamic_dispatch() -> bool {
        unsafe { super::nk_uses_dynamic_dispatch() != 0 }
    }
}

// endregion: Capabilities

// region: Dot

/// Computes the **dot product** (inner product) between two vectors.
pub trait Dot: Sized {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `dot`.
    fn inner(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::dot(a, b)
    }
}

impl Dot for f64 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Dot for f32 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
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
                a.len() as u64size,
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
                a.len() as u64size,
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
        unsafe { nk_dot_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
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
                a.len() as u64size,
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
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Dot

// region: Angular

/// Computes the **angular distance** (cosine distance) between two vectors.
pub trait Angular: Sized {
    type Output;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `angular`.
    fn cosine(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::angular(a, b)
    }
}

impl Angular for f64 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Angular for f32 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
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
                a.len() as u64size,
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
                a.len() as u64size,
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
        unsafe { nk_angular_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Angular

// region: Euclidean

/// Computes the **Euclidean distance** (L2) between two vectors.
pub trait Euclidean: Sized {
    type L2sqOutput;
    type L2Output;

    /// Squared Euclidean distance (L2²). Faster than `l2` for comparisons.
    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput>;

    /// Euclidean distance (L2). True metric distance.
    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output>;

    /// Alias for `l2sq` (SciPy compatibility).
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        Self::l2sq(a, b)
    }
}

impl Euclidean for f64 {
    type L2sqOutput = f64;
    type L2Output = f64;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe { nk_l2sq_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f32 {
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe { nk_l2sq_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl Euclidean for f16 {
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe {
            nk_l2sq_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe {
            nk_l2_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for bf16 {
    type L2sqOutput = f32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0.0;
        unsafe {
            nk_l2sq_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe {
            nk_l2_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for i8 {
    type L2sqOutput = u32;
    type L2Output = f32;

    fn l2sq(a: &[Self], b: &[Self]) -> Option<Self::L2sqOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2sqOutput = 0;
        unsafe { nk_l2sq_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }

    fn l2(a: &[Self], b: &[Self]) -> Option<Self::L2Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::L2Output = 0.0;
        unsafe { nk_l2_i8(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Euclidean

// region: Geospatial

/// Computes **great-circle distances** between geographic coordinates on Earth.
pub trait Haversine: Sized {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Computes **Vincenty geodesic distances** on the WGS84 ellipsoid.
pub trait Vincenty: Sized {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Combined trait for all geospatial distance computations.
pub trait Geospatial: Haversine + Vincenty {}

impl Haversine for f64 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f64 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f64 {}

impl Haversine for f32 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f32 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f32 {}

// endregion: Geospatial

// region: Hamming

/// Computes the **Hamming distance** between two binary vectors.
pub trait Hamming: Sized {
    type Output;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Hamming for u8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_hamming_u1(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Hamming

// region: Jaccard

/// Computes the **Jaccard distance** between two binary vectors.
pub trait Jaccard: Sized {
    type Output;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Jaccard for u8 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jaccard_u1(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

// endregion: Jaccard

// region: KullbackLeibler

/// Computes the **Kullback-Leibler divergence** between two probability distributions.
pub trait KullbackLeibler: Sized {
    type Output;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `kullbackleibler`.
    fn kl(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::kullbackleibler(a, b)
    }
}

impl KullbackLeibler for f64 {
    type Output = f64;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl KullbackLeibler for f32 {
    type Output = f32;
    fn kullbackleibler(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_kld_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
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
                a.len() as u64size,
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
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: KullbackLeibler

// region: JensenShannon

/// Computes the **Jensen-Shannon divergence** between two probability distributions.
pub trait JensenShannon: Sized {
    type Output;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `jensenshannon`.
    fn js(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        Self::jensenshannon(a, b)
    }
}

impl JensenShannon for f64 {
    type Output = f64;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f64(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
        Some(result)
    }
}

impl JensenShannon for f32 {
    type Output = f32;
    fn jensenshannon(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_jsd_f32(a.as_ptr(), b.as_ptr(), a.len() as u64size, &mut result) };
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
                a.len() as u64size,
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
                a.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: JensenShannon

// region: ComplexDot

/// Computes the **complex dot product** between two complex vectors.
pub trait ComplexDot: Sized {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl ComplexDot for f64 {
    type Output = ComplexProductF64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f64; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for f32 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for f16 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexDot for bf16 {
    type Output = ComplexProductF32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_dot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

// endregion: ComplexDot

// region: ComplexVDot

/// Computes the **conjugate dot product** (Hermitian inner product) between complex vectors.
pub trait ComplexVDot: Sized {
    type Output;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl ComplexVDot for f64 {
    type Output = ComplexProductF64;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f64; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f64c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for f32 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f32c(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for f16 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

impl ComplexVDot for bf16 {
    type Output = ComplexProductF32;
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return None;
        }
        let mut result: [f32; 2] = [0.0, 0.0];
        unsafe {
            nk_vdot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size / 2,
                result.as_mut_ptr(),
            )
        };
        Some((result[0], result[1]))
    }
}

// endregion: ComplexVDot

// region: Sparse

/// Computes the **intersection size** between two sorted sparse vectors.
pub trait Sparse: Sized {
    type Output;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output>;
}

impl Sparse for u16 {
    type Output = u32;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        let mut result: Self::Output = 0;
        unsafe {
            nk_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Sparse for u32 {
    type Output = u32;
    fn intersect(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        let mut result: Self::Output = 0;
        unsafe {
            nk_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                b.len() as u64size,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Sparse

// region: Sin

/// Computes **element-wise sine** of a vector.
pub trait Sin: Sized {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl Sin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_sin_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_sin_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Sin

// region: Cos

/// Computes **element-wise cosine** of a vector.
pub trait Cos: Sized {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl Cos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_cos_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Cos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_cos_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Cos

// region: ATan

/// Computes **element-wise arctangent** (inverse tangent) of a vector.
pub trait ATan: Sized {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl ATan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_atan_f64(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl ATan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_atan_f32(
                inputs.as_ptr(),
                inputs.len() as u64size,
                outputs.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: ATan

// region: Scale

/// Computes **element-wise affine transform** (scale and shift).
pub trait Scale: Sized {
    type Scalar;
    fn scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl Scale for f64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for f32 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for f16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_f16(
                a.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Scale for bf16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_bf16(
                a.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Scale for i8 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i8(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for u8 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u8(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for i16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i16(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for u16 {
    type Scalar = f32;
    fn scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u16(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for i32 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for u32 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u32(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for i64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_i64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Scale for u64 {
    type Scalar = f64;
    fn scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_scale_u64(
                a.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Scale

// region: Sum

/// Computes **element-wise addition** of two vectors.
pub trait Sum: Sized {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()>;
}

impl Sum for f64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for f32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for f16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Sum for bf16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl Sum for i8 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u8 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u16 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u32 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for i64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_i64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Sum for u64 {
    fn sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_sum_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: Sum

// region: WSum

/// Computes **element-wise weighted sum** of two vectors.
pub trait WSum: Sized {
    type Scalar;
    fn wsum(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl WSum for f64 {
    type Scalar = f64;
    fn wsum(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl WSum for f32 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl WSum for f16 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl WSum for bf16 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl WSum for i8 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl WSum for u8 {
    type Scalar = f32;
    fn wsum(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_wsum_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: WSum

// region: FMA

/// Computes **fused multiply-add** across three vectors.
pub trait FMA: Sized {
    type Scalar;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl FMA for f64 {
    type Scalar = f64;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_f64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl FMA for f32 {
    type Scalar = f32;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl FMA for f16 {
    type Scalar = f32;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl FMA for bf16 {
    type Scalar = f32;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

impl FMA for i8 {
    type Scalar = f32;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_i8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl FMA for u8 {
    type Scalar = f32;
    fn fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_fma_u8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len() as u64size,
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

// endregion: FMA

// region: MeshAlignment

/// Result of mesh alignment operations (RMSD, Kabsch, Umeyama).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAlignmentResult<T> {
    pub rotation_matrix: [T; 9],
    pub scale: T,
    pub rmsd: T,
    pub a_centroid: [T; 3],
    pub b_centroid: [T; 3],
}

impl MeshAlignmentResult<f64> {
    #[inline]
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

impl MeshAlignmentResult<f32> {
    #[inline]
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Mesh alignment operations for 3D point clouds.
pub trait MeshAlignment: Sized {
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;
    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;
    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>>;
}

impl MeshAlignment for f64 {
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for f32 {
    fn rmsd(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(a: &[[Self; 3]], b: &[[Self; 3]]) -> Option<MeshAlignmentResult<Self>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len() as u64size,
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

// endregion: MeshAlignment

// region: Convenience Trait Aliases

/// `SpatialSimilarity` bundles spatial distance metrics: Dot, Angular, and Euclidean.
pub trait SpatialSimilarity: Dot + Angular + Euclidean {}
impl<T: Dot + Angular + Euclidean> SpatialSimilarity for T {}

/// `BinarySimilarity` bundles binary distance metrics: Hamming and Jaccard.
pub trait BinarySimilarity: Hamming + Jaccard {}
impl<T: Hamming + Jaccard> BinarySimilarity for T {}

/// `ProbabilitySimilarity` bundles probability divergence metrics: KullbackLeibler and JensenShannon.
pub trait ProbabilitySimilarity: KullbackLeibler + JensenShannon {}
impl<T: KullbackLeibler + JensenShannon> ProbabilitySimilarity for T {}

/// `ComplexProducts` bundles complex number products: ComplexDot and ComplexVDot.
pub trait ComplexProducts: ComplexDot + ComplexVDot {}
impl<T: ComplexDot + ComplexVDot> ComplexProducts for T {}

/// `Elementwise` bundles element-wise operations: Scale, Sum, WSum, and FMA.
pub trait Elementwise: Scale + Sum + WSum + FMA {}
impl<T: Scale + Sum + WSum + FMA> Elementwise for T {}

/// `Trigonometry` bundles trigonometric functions: Sin, Cos, and ATan.
pub trait Trigonometry: Sin + Cos + ATan {}
impl<T: Sin + Cos + ATan> Trigonometry for T {}

// endregion: Convenience Trait Aliases

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = <f32 as Dot>::dot(&a, &b).unwrap();
        assert!((result - 32.0).abs() < 0.01);
    }

    #[test]
    fn dot_i8() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let result = i8::dot(&a, &b).unwrap();
        assert_eq!(result, 32);
    }

    #[test]
    fn l2_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = f32::l2sq(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 0.01);
    }

    #[test]
    fn l2_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let result = f64::l2sq(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 0.01);
    }

    #[test]
    fn l2_f16() {
        let a: Vec<f16> = vec![1.0, 2.0, 3.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let b: Vec<f16> = vec![4.0, 5.0, 6.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let result = f16::l2sq(&a, &b).unwrap();
        assert!((result - 27.0).abs() < 1.0);
    }

    #[test]
    fn hamming_u8() {
        let a = vec![0b11110000u8, 0b10101010];
        let b = vec![0b00001111u8, 0b01010101];
        let result = u8::hamming(&a, &b).unwrap();
        assert_eq!(result, 16);
    }

    #[test]
    fn jaccard_u8() {
        let a = vec![0b11110000u8, 0b10101010];
        let b = vec![0b11110000u8, 0b10101010];
        let result = u8::jaccard(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn js_f32() {
        let a = vec![0.25f32, 0.25, 0.25, 0.25];
        let b = vec![0.25f32, 0.25, 0.25, 0.25];
        let result = f32::jensenshannon(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn kl_f32() {
        let a = vec![0.25f32, 0.25, 0.25, 0.25];
        let b = vec![0.25f32, 0.25, 0.25, 0.25];
        let result = f32::kullbackleibler(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 0.01);
    }

    #[test]
    fn scale_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut result = vec![0.0f32; 3];
        f32::scale(&a, 2.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn scale_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let mut result = vec![0.0f64; 3];
        f64::scale(&a, 2.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn wsum_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut result = vec![0.0f32; 3];
        f32::wsum(&a, &b, 0.5, 0.5, &mut result).unwrap();
        assert!((result[0] - 2.5).abs() < 0.01);
        assert!((result[1] - 3.5).abs() < 0.01);
        assert!((result[2] - 4.5).abs() < 0.01);
    }

    #[test]
    fn fma_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![2.0f32, 2.0, 2.0];
        let c = vec![1.0f32, 1.0, 1.0];
        let mut result = vec![0.0f32; 3];
        f32::fma(&a, &b, &c, 1.0, 1.0, &mut result).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn sin_f32_small() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..11).map(|i| (i as f32) * PI / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.sin()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Sin>::sin(&inputs, &mut result).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 0.1, "sin mismatch: {} vs {}", r, e);
        }
    }

    #[test]
    fn cos_f32_test() {
        use core::f32::consts::PI;
        let inputs: Vec<f32> = (0..11).map(|i| (i as f32) * PI / 10.0).collect();
        let expected: Vec<f32> = inputs.iter().map(|x| x.cos()).collect();
        let mut result = vec![0.0f32; inputs.len()];
        <f32 as Cos>::cos(&inputs, &mut result).unwrap();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 0.1, "cos mismatch: {} vs {}", r, e);
        }
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
    fn mesh_alignment_length_mismatch() {
        let a: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(f64::kabsch(a, b).is_none());
    }

    #[test]
    fn mesh_alignment_too_few_points() {
        let a: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b: &[[f64; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(f64::kabsch(a, b).is_none());
    }
}
