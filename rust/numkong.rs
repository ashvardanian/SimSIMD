//! # NumKong - Hardware-Accelerated Numerics
//!
//! Provides SIMD-accelerated distance metrics, elementwise operations, and tensor algebra
//! targeting ARM NEON/SVE/SME and x86 AVX2/AVX-512 backends.
//!
//! ## Modules
//!
//! - [`scalars`]: Mixed-precision scalar types (`f16`, `bf16`, FP8, packed integers) and [`FloatLike`] trait
//! - [`numerics`]: Distance functions, elementwise operations, trigonometry, reductions, and geospatial
//! - [`tensor`]: N-dimensional tensors, GEMM, and packed spatial distance operations
//!
//! ## Implemented operations include:
//!
//! * Euclidean (L2), inner product, and angular (cosine) spatial distances.
//! * Hamming and Jaccard binary distances.
//! * Kullback-Leibler and Jensen-Shannon divergences.
//! * Elementwise scale, sum, blend, and FMA operations.
//! * Trigonometric functions (sin, cos, atan).
//! * Type casting between all scalar formats.
//! * Matrix multiplication with pre-packing (GEMM).
//!
//! ## Example
//!
//! ```rust
//! use numkong::{Dot, Angular, Euclidean};
//!
//! let a = &[1.0_f32, 2.0, 3.0];
//! let b = &[4.0_f32, 5.0, 6.0];
//!
//! let dot_product = f32::dot(a, b);
//! let angular_dist = f32::angular(a, b);
//! let l2sq_dist = f32::sqeuclidean(a, b);
//!
//! // Optimize performance by flushing denormals
//! numkong::capabilities::configure_thread();
//! ```
//!
//! ## Mixed Precision Support
//!
//! ```rust
//! use numkong::{Angular, f16, bf16};
//!
//! // Work with half-precision floats
//! let half_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_angular_dist = f16::angular(&half_a, &half_b);
//!
//! // Work with brain floats
//! let brain_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_angular_dist = bf16::angular(&brain_a, &brain_b);
//!
//! // Direct bit manipulation
//! let half = f16::from_f32(3.14);
//! let bits = half.0; // Access raw u16 representation
//! let reconstructed = f16(bits);
//! ```
//!
//! ## Traits
//!
//! The `SpatialSimilarity` trait (combining `Dot`, `Angular`, `Euclidean`) covers:
//!
//! - `dot(a, b)`: Computes dot product between two slices.
//! - `angular(a, b)` / `cosine(a, b)`: Computes angular distance (1 − cosine similarity).
//! - `sqeuclidean(a, b)`: Computes squared Euclidean distance.
//! - `euclidean(a, b)`: Computes Euclidean distance.
//!
//! The `BinarySimilarity` trait (combining `Hamming`, `Jaccard`) covers:
//!
//! - `hamming(a, b)`: Computes Hamming distance between two slices.
//! - `jaccard(a, b)`: Computes Jaccard distance between two slices.
//!
//! The `ProbabilitySimilarity` trait (combining `KullbackLeibler`, `JensenShannon`) covers:
//!
//! - `jensenshannon(a, b)`: Computes Jensen-Shannon divergence.
//! - `kullbackleibler(a, b)`: Computes Kullback-Leibler divergence.
//!
//! The elementwise traits (including `EachScale`, `EachSum`, `EachBlend`, `EachFMA`) covers:
//!
//! - `scale(a, alpha, beta, result)`: Element-wise `result[i] = α × a[i] + β`.
//! - `sum(a, b, result)`: Element-wise `result[i] = a[i] + b[i]`.
//! - `wsum(a, b, alpha, beta, result)`: Weighted sum `result[i] = α × a[i] + β × b[i]`.
//! - `fma(a, b, c, alpha, beta, result)`: Fused multiply-add `result[i] = α × a[i] × b[i] + β × c[i]`.
//!
//! The `Trigonometry` trait (combining `EachSin`, `EachCos`, `EachATan`) covers:
//!
//! - `sin(input, result)`: Element-wise sine.
//! - `cos(input, result)`: Element-wise cosine.
//! - `atan(input, result)`: Element-wise arctangent.
//!
//! Additional traits: `ComplexDot`, `ComplexVDot`, `SparseIntersect`, `SparseDot`.
//!
#![allow(non_camel_case_types)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

// Module declarations
pub mod numerics;
pub mod scalar;
pub mod tensor;

// Re-export scalar types at crate root
pub use scalar::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, FloatLike};

// Re-export complex product types
pub use numerics::{ComplexProductF32, ComplexProductF64};

// Re-export all numeric traits
pub use numerics::{
    Angular, Bilinear, BinarySimilarity, ComplexBilinear, ComplexDot, ComplexEachBlend,
    ComplexEachFMA, ComplexEachScale, ComplexEachSum, ComplexProducts, ComplexVDot, Dot, EachATan,
    EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum, Euclidean, Hamming, Haversine,
    Jaccard, JensenShannon, KullbackLeibler, Mahalanobis, MeshAlignment, MeshAlignmentResult,
    ProbabilitySimilarity, ReduceMinMax, ReduceMoments, Reductions, SparseDot, SparseIntersect,
    SpatialSimilarity, Trigonometry, Vincenty,
};

// Re-export cast operations
pub use numerics::{cast, CastDtype};

// Re-export capabilities module
pub use numerics::cap;
pub use numerics::capabilities;

// Re-export tensor types
pub use tensor::{
    Allocator, Angulars, Dots, Euclideans, Global, Hammings, Jaccards, Matrix, MatrixView,
    MatrixViewMut, ShapeDescriptor, SliceRange, Tensor, TensorError, TensorView, TensorViewMut,
    TransposedMatrixMultiplier, DEFAULT_MAX_RANK, SIMD_ALIGNMENT,
};

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_dot() {
        let first = [1.0_f32, 2.0, 3.0];
        let second = [4.0_f32, 5.0, 6.0];
        assert!((<f32 as Dot>::dot(&first, &second).unwrap() - 32.0).abs() < 0.01);
    }

    #[test]
    fn smoke_angular() {
        let first = [1.0_f32, 0.0];
        let second = [0.0_f32, 1.0];
        // Orthogonal vectors → angular distance = 1.0
        assert!((f32::angular(&first, &second).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn smoke_euclidean() {
        let first = [0.0_f32, 0.0, 0.0];
        let second = [3.0_f32, 4.0, 0.0];
        assert!((f32::euclidean(&first, &second).unwrap() - 5.0).abs() < 0.1);
    }

    #[test]
    fn smoke_tensor_dots() {
        capabilities::configure_thread();
        let queries = Tensor::<f32>::try_new(&[2, 4], 1.0).unwrap();
        let targets = Tensor::<f32>::try_new(&[3, 4], 1.0).unwrap();
        let packed_targets = TransposedMatrixMultiplier::try_pack(&targets).unwrap();
        let products = queries.dots_packed(&packed_targets);
        assert_eq!(products.shape(), &[2, 3]);
        assert!((products.as_slice()[0] - 4.0).abs() < 0.01);
    }
}

// endregion: Tests

// region: WASM Runtime Tests

/// WASM runtime integration tests using Wasmtime
/// These tests validate that WASI builds work correctly with standalone runtimes
#[cfg(all(test, feature = "wasm-runtime"))]
mod wasm_runtime_tests {
    use std::fs;
    use wasmtime::*;
    use wasmtime_wasi::WasiCtx;

    /// Test that WASI WASM module can be loaded and executed with Wasmtime
    /// This validates the dual-path capability detection (EM_ASM vs WASI imports)
    #[test]
    fn wasi_with_wasmtime() -> wasmtime::Result<()> {
        // Check if WASI build exists
        let wasm_path = "build-wasi/test.wasm";
        if !std::path::Path::new(wasm_path).exists() {
            eprintln!("WASI build not found at {}. Run:", wasm_path);
            eprintln!("  export WASI_SDK_PATH=~/wasi-sdk");
            eprintln!("  cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_WASM_WASI=ON");
            eprintln!("  cmake --build build-wasi");
            return Ok(()); // Skip test if build doesn't exist
        }

        println!("Loading WASI module from {}", wasm_path);

        // Create Wasmtime engine and linker
        let engine = Engine::default();
        let mut linker = Linker::new(&engine);

        // Create WASI context (Wasmtime 41+ API)
        let wasi = WasiCtx::builder().inherit_stdio().inherit_args().build_p1();
        let mut store = Store::new(&engine, wasi);

        // Add WASI support (Wasmtime 41+ requires p1 module)
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s)?;

        // Provide capability detection imports (required for WASI build)
        // These functions are called from nk_capabilities_v128relaxed_() in C code
        linker.func_wrap("env", "nk_has_v128", || -> i32 {
            // Return 1 (true) - assume SIMD128 is available in Wasmtime
            println!("  nk_has_v128() called from WASM -> returning 1");
            1
        })?;

        linker.func_wrap("env", "nk_has_relaxed", || -> i32 {
            // Return 1 (true) - assume Relaxed SIMD is available in Wasmtime
            println!("  nk_has_relaxed() called from WASM -> returning 1");
            1
        })?;

        // Load WASM module
        let wasm_bytes = fs::read(wasm_path)?;
        let module = Module::new(&engine, wasm_bytes)?;

        // Instantiate module
        println!("Instantiating WASM module...");
        let instance = linker.instantiate(&mut store, &module)?;

        // Get main function
        let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;

        // Run tests
        println!("Running WASM tests...");
        let exit_code = main.call(&mut store, ())?;

        println!("WASM tests completed with exit code: {}", exit_code);

        // Assert tests passed
        assert_eq!(
            exit_code, 0,
            "WASI tests failed with exit code {}",
            exit_code
        );

        Ok(())
    }

    /// Test capability detection mechanism works in WASI environment
    #[test]
    fn capability_imports() -> wasmtime::Result<()> {
        println!("Testing capability import mechanism...");

        // Create minimal engine for import testing
        let engine = Engine::default();
        let mut linker = Linker::<()>::new(&engine);

        // Test that we can define the required imports
        linker.func_wrap("env", "nk_has_v128", || -> i32 { 1 })?;
        linker.func_wrap("env", "nk_has_relaxed", || -> i32 { 0 })?;

        println!("  ✓ Capability imports defined successfully");

        Ok(())
    }
}

// endregion: WASM Runtime Tests
