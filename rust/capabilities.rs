//! Runtime CPU capability detection.
//!
//! This module provides:
//!
//! - [`available`]: Query the bitmask of supported SIMD instruction sets
//! - [`configure_thread`]: Enable optimal SIMD settings for the current thread
//! - [`uses_dynamic_dispatch`]: Check if the library selects kernels at runtime
//! - [`cap`]: Constants for individual capability bits (NEON, SKYLAKE, etc.)

#[link(name = "numkong")]
extern "C" {
    fn nk_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_dynamic_dispatch() -> i32;
    fn nk_capabilities() -> u64;
}

/// Returns the bitmask of available CPU capabilities.
/// Use with `cap::*` constants to check for specific features.
///
/// # Example
/// ```
/// use numkong::{capabilities, cap};
///
/// let caps = capabilities::available();
/// if caps & cap::NEON != 0 {
///     println!("NEON is available");
/// }
/// if caps & cap::SKYLAKE != 0 {
///     println!("AVX-512 (Skylake) is available");
/// }
/// ```
pub fn available() -> u64 { unsafe { nk_capabilities() } }

/// Configures the current thread for optimal SIMD performance.
/// This includes flushing denormalized numbers to zero and enabling AMX on supported CPUs.
/// Must be called once per thread before using AMX (Advanced Matrix Extensions) operations.
pub fn configure_thread() -> bool {
    // Pass !0 to enable all capabilities including AMX
    unsafe { nk_configure_thread(!0) != 0 }
}

/// Returns `true` if the library uses dynamic dispatch for function selection.
pub fn uses_dynamic_dispatch() -> bool { unsafe { nk_uses_dynamic_dispatch() != 0 } }

/// Capability bit masks in chronological order (by first commercial silicon).
pub mod cap {
    pub const SERIAL: u64 = 1 << 0; // Always: Fallback
    pub const NEON: u64 = 1 << 1; // 2013: ARM NEON
    pub const HASWELL: u64 = 1 << 2; // 2013: Intel AVX2
    pub const SKYLAKE: u64 = 1 << 3; // 2017: Intel AVX-512
    pub const NEONHALF: u64 = 1 << 4; // 2017: ARM NEON FP16
    pub const NEONSDOT: u64 = 1 << 5; // 2017: ARM NEON i8 dot
    pub const NEONFHM: u64 = 1 << 6; // 2018: ARM NEON FP16 FML
    pub const ICELAKE: u64 = 1 << 7; // 2019: Intel AVX-512 VNNI
    pub const GENOA: u64 = 1 << 8; // 2020: Intel/AMD AVX-512 BF16
    pub const NEONBFDOT: u64 = 1 << 9; // 2020: ARM NEON BF16
    pub const SVE: u64 = 1 << 10; // 2020: ARM SVE
    pub const SVEHALF: u64 = 1 << 11; // 2020: ARM SVE FP16
    pub const SVESDOT: u64 = 1 << 12; // 2020: ARM SVE i8 dot
    pub const SIERRA: u64 = 1 << 13; // 2021: Intel AVX2+VNNI
    pub const SVEBFDOT: u64 = 1 << 14; // 2021: ARM SVE BF16
    pub const SVE2: u64 = 1 << 15; // 2022: ARM SVE2
    pub const V128RELAXED: u64 = 1 << 16; // 2022: WASM Relaxed SIMD
    pub const SAPPHIRE: u64 = 1 << 17; // 2023: Intel AVX-512 FP16
    pub const SAPPHIREAMX: u64 = 1 << 18; // 2023: Intel Sapphire AMX
    pub const RVV: u64 = 1 << 19; // 2023: RISC-V Vector
    pub const RVVHALF: u64 = 1 << 20; // 2023: RISC-V Zvfh
    pub const RVVBF16: u64 = 1 << 21; // 2023: RISC-V Zvfbfwma
    pub const RVVBB: u64 = 1 << 33; // 2024: RISC-V Zvbb
    pub const GRANITEAMX: u64 = 1 << 22; // 2024: Intel Granite AMX FP16
    pub const TURIN: u64 = 1 << 23; // 2024: AMD Turin AVX-512 CD
    pub const SME: u64 = 1 << 24; // 2024: ARM SME
    pub const SME2: u64 = 1 << 25; // 2024: ARM SME2
    pub const SMEF64: u64 = 1 << 26; // 2024: ARM SME F64
    pub const SMEFA64: u64 = 1 << 27; // 2024: ARM SME FA64
    pub const SVE2P1: u64 = 1 << 28; // 2025+: ARM SVE2.1
    pub const SME2P1: u64 = 1 << 29; // 2025+: ARM SME2.1
    pub const SMEHALF: u64 = 1 << 30; // 2025+: ARM SME F16F16
    pub const SMEBF16: u64 = 1 << 31; // 2025+: ARM SME B16B16
    pub const SMELUT2: u64 = 1 << 32; // 2025+: ARM SME LUTv2
}
