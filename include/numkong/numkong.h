/**
 *  @file       numkong.h
 *  @brief      SIMD-accelerated Similarity Measures and Distance Functions.
 *  @author     Ash Vardanian
 *  @date       March 14, 2023
 *
 *  @section x86_targets Choosing x86 Target Generations
 *
 *  It's important to provide fine-grained controls over AVX512 families, as they are very fragmented:
 *
 *  - Intel Skylake servers: F, CD, VL, DQ, BW
 *  - Intel Cascade Lake workstations: F, CD, VL, DQ, BW, VNNI
 *       > In other words, it extends Skylake with VNNI support
 *  - Intel Sunny Cove (Ice Lake) servers:
 *         F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ
 *  - AMD Zen4 (Genoa):
 *         F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, BF16
 *       > In other words, it extends Sunny Cove with BF16 support
 *  - Intel Golden Cove (Sapphire Rapids): extends Zen4 and Sunny Cove with FP16 support
 *  - AMD Zen5 (Turin): makes VP2INTERSECT cool again
 *
 *  Intel Palm Cove was an irrelevant intermediate release extending Skylake with IFMA and VBMI.
 *  Intel Willow Cove was an irrelevant intermediate release extending Sunny Cove with VP2INTERSECT,
 *  which are not supported by other CPUs to date and are only available in Tiger Lake laptops.
 *  Intel Cooper Lake was the only intermediary platform, that supported BF16, but not FP16.
 *  It's mostly used in 4-socket and 8-socket high-memory configurations.
 *
 *  For us, it makes sense to differentiate only these AVX512 generations:
 *
 *  1. Intel Skylake (pre 2019): supports single-precision dot-products.
 *  2. Intel Ice Lake (2019-2021): advanced integer algorithms.
 *  3. AMD Genoa (2023+): brain-floating point support.
 *  4. Intel Sapphire Rapids (2023+): advanced mixed-precision float processing.
 *  5. AMD Turin (2024+): advanced sparse algorithms.
 *
 *  Beyond those, we support AVX2 for old Haswell generation CPUs, and AVX2+VNNI for modern Sierra generation.
 *
 *  To list all available macros for x86, take a recent compiler, like GCC 12 and run:
 *       gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
 *  On Arm machines you may want to check for other flags:
 *       gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
 *
 *  @section arm_targets Choosing Arm Target Generations
 *
 *  Arm CPUs share design IP, but are produced by different vendors, potentially making the platform
 *  even more fragmented than x86. There are 2 important families of SIMD extensions - NEON and SVE.
 *
 *  - Armv8-A: +fp, +simd
 *  - Armv8.1-A: armv8-a, +crc, +lse, +rdma
 *  - Armv8.2-A: armv8.1-a
 *  - Armv8.3-A: armv8.2-a, +pauth
 *  - Armv8.4-A: armv8.3-a, +flagm, +fp16fml, +dotprod
 *  - Armv8.5-A: armv8.4-a, +sb, +ssbs, +predres
 *  - Armv8.6-A: armv8.5-a, +bf16, +i8mm
 *  - Armv8.7-A: armv8.6-a, +ls64
 *  - Armv8.8-A: armv8.7-a, +mops
 *  - Armv8.9-A: armv8.8-a
 *  - Armv9-A: armv8.5-a, +sve, +sve2
 *  - Armv9.1-A: armv9-a, +bf16, +i8mm
 *  - Armv9.2-A: armv9.1-a, +ls64
 *  - Armv9.3-A: armv9.2-a, +mops
 *  - Armv9.4-A: armv9.3-a
 *
 *  SVE has been optional since Armv8.2-A, but it's a requirement for Armv9.0-A.
 *  A 512-bit SVE variant has already been implemented on the Fugaku supercomputer.
 *  A more flexible version, 2x256 SVE, was implemented by the AWS Graviton3 ARM processor.
 *  Here are the most important recent families of CPU cores designed by Arm:
 *
 *  - Neoverse N1: armv8.2-a, extended with Armv8.4 "dotprod" instructions.
 *    Used in AWS @b Graviton2 and Ampere @b Altra.
 *    https://developer.arm.com/Processors/Neoverse%20N1
 *  - Neoverse V1: armv8.4-a, extended with Armv8.6 bfloat/int8 "matmul" instructions.
 *    Used in AWS @b Graviton3, which also enables `sve`, `svebf16`, and `svei8mm`.
 *    https://developer.arm.com/Processors/Neoverse%20V1
 *  - Neoverse V2: armv9.0 with SVE2 and SVE bit-permutes
 *    Used in AWS @b Graviton4, NVIDIA @b Grace, Google @b Axion.
 *    https://developer.arm.com/Processors/Neoverse%20V2
 *    The N2 core is very similar to V2 and is used by Microsoft @b Cobalt.
 *    https://developer.arm.com/Processors/Neoverse%20N2
 *
 *  On the consumer side, Apple is the biggest player with mobile @b A chips and desktop @b M chips.
 *  The M1 implements Armv8.5-A, both M2 and M3 implement Armv8.6-A, and M4 is expected to have Armv9.1-A.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics
 *  - Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#ifndef NK_H
#define NK_H

#define NK_VERSION_MAJOR 6
#define NK_VERSION_MINOR 5
#define NK_VERSION_PATCH 12

/**
 *  @brief  Removes compile-time dispatching, and replaces it with runtime dispatching.
 *          So the `nk_dot_f32` function will invoke the most advanced backend supported by the CPU,
 *          that runs the program, rather than the most advanced backend supported by the CPU
 *          used to compile the library or the downstream application.
 */
#if !defined(NK_DYNAMIC_DISPATCH)
#define NK_DYNAMIC_DISPATCH (0) // true or false
#endif

#include "binary.h"       // Hamming, Jaccard
#include "curved.h"       // Mahalanobis, Bilinear Forms
#include "dot.h"          // Inner (dot) product, and its conjugate
#include "dots.h"         // GEMM-style MxN batched dot-products
#include "elementwise.h"  // Weighted Sum, Fused-Multiply-Add
#include "geospatial.h"   // Haversine and Vincenty
#include "mesh.h"         // RMSD, Kabsch, Umeyama
#include "probability.h"  // Kullback-Leibler, Jensen–Shannon
#include "reduce.h"       // Horizontal reductions: sum, min, max
#include "sparse.h"       // Intersect
#include "spatial.h"      // L2, Angular
#include "trigonometry.h" // Sin, Cos, Atan

// On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
#if defined(NK_DEFINED_APPLE_)
#include <fenv.h>       // `fesetenv` - part of C 99 standard
#include <sys/sysctl.h> // `sysctlbyname`
#endif

// Detect POSIX extensions availability for signal handling.
// POSIX extensions provide `sigaction`, `sigjmp_buf`, and `sigsetjmp` for safe signal handling.
// These are needed on Linux ARM for safely testing `mrs` instruction availability.
#if defined(NK_DEFINED_LINUX_) && defined(_POSIX_VERSION)
#include <setjmp.h> // `sigjmp_buf`, `sigsetjmp`, `siglongjmp`
#include <signal.h> // `sigaction`, `SIGILL`
#define NK_HAS_POSIX_EXTENSIONS_ 1
#else
#define NK_HAS_POSIX_EXTENSIONS_ 0
#endif

// On Linux x86, we need syscall for AMX permission request
#if defined(NK_DEFINED_LINUX_) && NK_TARGET_X86_
#include <sys/syscall.h> // `syscall`, `SYS_arch_prctl`
#include <unistd.h>      // `syscall`
#endif

// On Windows ARM, we use IsProcessorFeaturePresent API for capability detection
#if defined(NK_DEFINED_WINDOWS_) && NK_TARGET_ARM_
#include <processthreadsapi.h> // `IsProcessorFeaturePresent`
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Enumeration of supported metric kinds.
 *          Some have aliases for convenience/discoverability.
 */
typedef enum {
    nk_metric_unknown_k = 0, ///< Unknown metric kind

    // Classics:
    nk_metric_dot_k = 'i',   ///< Inner product
    nk_metric_inner_k = 'i', ///< Inner product alias

    nk_metric_vdot_k = 'v', ///< Complex inner product

    nk_metric_angular_k = 'a', ///< Angular (cosine) distance

    nk_metric_l2_k = 'e',          ///< Euclidean distance alias
    nk_metric_euclidean_k = 'e',   ///< Euclidean distance alias
    nk_metric_l2sq_k = '2',        ///< Squared Euclidean distance
    nk_metric_sqeuclidean_k = '2', ///< Squared Euclidean distance alias

    // Binary:
    nk_metric_hamming_k = 'h',   ///< Hamming distance
    nk_metric_manhattan_k = 'h', ///< Manhattan distance is same as Hamming

    nk_metric_jaccard_k = 'j',  ///< Jaccard coefficient
    nk_metric_tanimoto_k = 'j', ///< Tanimoto coefficient is same as Jaccard

    // Sets:
    nk_metric_intersect_k = 'x',  ///< Equivalent to unnormalized Jaccard
    nk_metric_sparse_dot_k = 'd', ///< Sparse dot product with weighted indices

    // Curved Spaces:
    nk_metric_bilinear_k = 'b',    ///< Bilinear form
    nk_metric_mahalanobis_k = 'm', ///< Mahalanobis distance
    nk_metric_haversine_k = 'o',   ///< Haversine distance
    nk_metric_vincenty_k = 'O',    ///< Vincenty distance (ellipsoidal geodesic)

    // Probability:
    nk_metric_kld_k = 'k', ///< Kullback-Leibler divergence
    nk_metric_jsd_k = 's', ///< Jensen-Shannon divergence

    // BLAS-like operations:
    nk_metric_scale_k = '*', ///< Scale
    nk_metric_sum_k = '+',   ///< Sum
    nk_metric_wsum_k = 'w',  ///< Weighted Sum
    nk_metric_fma_k = 'f',   ///< Fused Multiply-Add

    // Element-wise trigonometric functions:
    nk_metric_sin_k = 'S',  ///< Element-wise trigonometric sine
    nk_metric_cos_k = 'C',  ///< Element-wise trigonometric cosine
    nk_metric_atan_k = 'A', ///< Element-wise trigonometric arctangent

    // Mesh superposition metrics:
    nk_metric_rmsd_k = 'r',    ///< RMSD without optimal superposition
    nk_metric_kabsch_k = 'K',  ///< Kabsch RMSD with optimal rotation
    nk_metric_umeyama_k = 'U', ///< Umeyama RMSD with optimal rotation and scale

    // Horizontal reductions:
    nk_metric_reduce_add_k = 'R', ///< Horizontal sum reduction
    nk_metric_reduce_min_k = '<', ///< Horizontal min reduction with argmin
    nk_metric_reduce_max_k = '>', ///< Horizontal max reduction with argmax

} nk_metric_kind_t;

/**
 *  @brief  Enumeration of SIMD capabilities of the target architecture.
 */
typedef enum {
    nk_cap_serial_k = 1,       ///< Serial (non-SIMD) capability
    nk_cap_any_k = 0x7FFFFFFF, ///< Mask representing any capability with `INT_MAX`

    nk_cap_haswell_k = 1 << 10,      ///< x86 AVX2 capability with FMA and F16C extensions
    nk_cap_skylake_k = 1 << 11,      ///< x86 AVX512 baseline capability
    nk_cap_ice_k = 1 << 12,          ///< x86 AVX512 capability with advanced integer algos
    nk_cap_genoa_k = 1 << 13,        ///< x86 AVX512 capability with `bf16` support
    nk_cap_sapphire_k = 1 << 14,     ///< x86 AVX512 capability with `f16` support
    nk_cap_turin_k = 1 << 15,        ///< x86 AVX512 capability with conflict detection
    nk_cap_sierra_k = 1 << 16,       ///< x86 AVX2+VNNI capability with `i8` dot-products
    nk_cap_sapphire_amx_k = 1 << 17, ///< x86 AMX capability with `i8` and `bf16` support
    nk_cap_granite_amx_k = 1 << 18,  ///< x86 AMX capability with `f16` support

    nk_cap_neon_k = 1 << 20,      ///< ARM NEON baseline capability
    nk_cap_neon_f16_k = 1 << 21,  ///< ARM NEON `f16` capability
    nk_cap_neon_bf16_k = 1 << 22, ///< ARM NEON `bf16` capability
    nk_cap_neon_i8_k = 1 << 23,   ///< ARM NEON `i8` capability
    nk_cap_sve_k = 1 << 24,       ///< ARM SVE baseline capability
    nk_cap_sve_f16_k = 1 << 25,   ///< ARM SVE `f16` capability
    nk_cap_sve_bf16_k = 1 << 26,  ///< ARM SVE `bf16` capability
    nk_cap_sve_i8_k = 1 << 27,    ///< ARM SVE `i8` capability
    nk_cap_sve2_k = 1 << 28,      ///< ARM SVE2 capability
    nk_cap_sve2p1_k = 1 << 29,    ///< ARM SVE2p1 capability

} nk_capability_t;

/**
 *  @brief  Enumeration of supported data types.
 *
 *  Includes complex type descriptors which in C code would use the real counterparts,
 *  but the independent flags contain metadata to be passed between programming language
 *  interfaces.
 */
typedef enum {
    nk_datatype_unknown_k = 0, ///< Unknown data type
    nk_b8_k = 1 << 1,          ///< Single-bit values packed into 8-bit words
    nk_b1x8_k = nk_b8_k,       ///< Single-bit values packed into 8-bit words
    nk_i4x2_k = 1 << 19,       ///< 4-bit signed integers packed into 8-bit words

    nk_i8_k = 1 << 2,  ///< 8-bit signed integer
    nk_i16_k = 1 << 3, ///< 16-bit signed integer
    nk_i32_k = 1 << 4, ///< 32-bit signed integer
    nk_i64_k = 1 << 5, ///< 64-bit signed integer

    nk_u8_k = 1 << 6,  ///< 8-bit unsigned integer
    nk_u16_k = 1 << 7, ///< 16-bit unsigned integer
    nk_u32_k = 1 << 8, ///< 32-bit unsigned integer
    nk_u64_k = 1 << 9, ///< 64-bit unsigned integer

    nk_f64_k = 1 << 10,  ///< Double precision floating point
    nk_f32_k = 1 << 11,  ///< Single precision floating point
    nk_f16_k = 1 << 12,  ///< Half precision floating point
    nk_bf16_k = 1 << 13, ///< Brain floating point

    nk_e4m3_k = 1 << 14, ///< FP8 E4M3 floating point
    nk_e5m2_k = 1 << 15, ///< FP8 E5M2 floating point

    nk_f64c_k = 1 << 20,  ///< Complex double precision floating point
    nk_f32c_k = 1 << 21,  ///< Complex single precision floating point
    nk_f16c_k = 1 << 22,  ///< Complex half precision floating point
    nk_bf16c_k = 1 << 23, ///< Complex brain floating point
} nk_datatype_t;

typedef enum {
    nk_datatype_unknown_family_k = 0,
    nk_datatype_binary_family_k,
    nk_datatype_float_family_k,
    nk_datatype_complex_float_family_k,
    nk_datatype_int_family_k,
    nk_datatype_uint_family_k,
} nk_datatype_family_k;

/**
 *  @brief  Classifies the family of the datatype.
 *  @return The family of the datatype.
 */
NK_PUBLIC nk_datatype_family_k nk_datatype_family(nk_datatype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_datatype_float_family_k;
    case nk_f32_k: return nk_datatype_float_family_k;
    case nk_f16_k: return nk_datatype_float_family_k;
    case nk_bf16_k: return nk_datatype_float_family_k;
    case nk_e4m3_k: return nk_datatype_float_family_k;
    case nk_e5m2_k: return nk_datatype_float_family_k;
    case nk_f64c_k: return nk_datatype_complex_float_family_k;
    case nk_f32c_k: return nk_datatype_complex_float_family_k;
    case nk_f16c_k: return nk_datatype_complex_float_family_k;
    case nk_bf16c_k: return nk_datatype_complex_float_family_k;
    case nk_b8_k: return nk_datatype_binary_family_k;
    case nk_u8_k: return nk_datatype_uint_family_k;
    case nk_u16_k: return nk_datatype_uint_family_k;
    case nk_u32_k: return nk_datatype_uint_family_k;
    case nk_u64_k: return nk_datatype_uint_family_k;
    case nk_i8_k: return nk_datatype_int_family_k;
    case nk_i16_k: return nk_datatype_int_family_k;
    case nk_i32_k: return nk_datatype_int_family_k;
    case nk_i64_k: return nk_datatype_int_family_k;
    case nk_i4x2_k: return nk_datatype_int_family_k;
    default: return nk_datatype_unknown_family_k;
    }
}

/**
 *  @brief  Type-punned function pointer for dense vector representations and simplest similarity measures.
 *
 *  @param[in] a    Pointer to the first data array.
 *  @param[in] b    Pointer to the second data array.
 *  @param[in] n    Number of scalar words in the input arrays.
 *                  When dealing with sub-byte types, the number of scalar words is the number of bytes.
 *                  When dealing with complex types, the number of scalar words is the sum of real and imaginary parts.
 *  @param[out] d   Output value as a double-precision float.
 *                  In complex dot-products @b two scalars are exported for the real and imaginary parts.
 */
typedef void (*nk_metric_dense_punned_t)(void const *a, void const *b, nk_size_t n, void *d);

/**
 *  @brief  Type-punned function pointer for sparse vector representations and similarity measures.
 *
 *  @param[in] a          Pointer to the first data array, generally a sorted array of integers.
 *  @param[in] b          Pointer to the second data array, generally a sorted array of integers.
 *  @param[in] a_length   Number of scalar words in the first input array.
 *  @param[in] b_length   Number of scalar words in the second input array.
 *  @param[out] d         Output value as a double-precision float, generally without decimals.
 */
typedef void (*nk_metric_sparse_punned_t)(void const *a, void const *b,           //
                                          nk_size_t a_length, nk_size_t b_length, //
                                          void *d);

/**
 *  @brief  Type-punned function pointer for sparse dot products with weights.
 *
 *  @param[in] a          First sorted array of indices.
 *  @param[in] b          Second sorted array of indices.
 *  @param[in] a_weights  Weights for first array.
 *  @param[in] b_weights  Weights for second array.
 *  @param[in] a_length   Number of elements in first array.
 *  @param[in] b_length   Number of elements in second array.
 *  @param[out] product   Output dot product (void* for type punning).
 */
typedef void (*nk_metric_sparse_dot_punned_t)(void const *a, void const *b,                 //
                                              void const *a_weights, void const *b_weights, //
                                              nk_size_t a_length, nk_size_t b_length,       //
                                              void *product);

/**
 *  @brief  Type-punned function pointer for curved vector spaces and similarity measures.
 *
 *  @param[in] a    Pointer to the first data array.
 *  @param[in] b    Pointer to the second data array.
 *  @param[in] c    Pointer to the metric tensor array or some covariance matrix.
 *  @param[in] n    Number of scalar words in the input arrays.
 *  @param[out] d   Output value as a double-precision float.
 */
typedef void (*nk_metric_curved_punned_t)(void const *a, void const *b, void const *c, //
                                          nk_size_t n, void *d);

/**
 *  @brief  Type-punned function pointer for geospatial distance functions.
 *
 *  @param[in] a_lats   Latitudes of first point set (in radians).
 *  @param[in] a_lons   Longitudes of first point set (in radians).
 *  @param[in] b_lats   Latitudes of second point set (in radians).
 *  @param[in] b_lons   Longitudes of second point set (in radians).
 *  @param[in] n        Number of point pairs.
 *  @param[out] results Output distances in meters (void* for type punning).
 */
typedef void (*nk_metric_geospatial_punned_t)(void const *a_lats, void const *a_lons, //
                                              void const *b_lats, void const *b_lons, //
                                              nk_size_t n, void *results);

/**
 *  @brief  Type-punned function pointer for Scaling & Shifting operations on dense vector representations.
 *          Implements the `y = alpha * a + beta` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor (type depends on input precision).
 *  @param[in] beta     Pointer to offset/bias term (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_kernel_scale_punned_t)(void const *a, nk_size_t n, void const *alpha, void const *beta, void *y);

/**
 *  @brief  Type-punned function pointer for element-wise Sum operations on dense vector representations.
 *          Implements the `y = a + b` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_kernel_sum_punned_t)(void const *a, void const *b, nk_size_t n, void *y);

/**
 *  @brief  Type-punned function pointer for Weighted Sum operations on dense vector representations.
 *          Implements the `y = alpha * a + beta * b` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor for the first array (type depends on input precision).
 *  @param[in] beta     Pointer to scaling factor for the second array (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_kernel_wsum_punned_t)(void const *a, void const *b, nk_size_t n, void const *alpha, void const *beta,
                                        void *y);

/**
 *  @brief  Type-punned function pointer for FMA operations on dense vector representations.
 *          Implements the `y = alpha * a * b + beta * c` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] c        Pointer to the third data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor for a*b product (type depends on input precision).
 *  @param[in] beta     Pointer to scaling factor for c array (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_kernel_fma_punned_t)(void const *a, void const *b, void const *c, nk_size_t n, void const *alpha,
                                       void const *beta, void *y);

/**
 *  @brief  Type-punned function pointer for element-wise trigonometric functions.
 *          Implements operations like `y[i] = sin(x[i])`.
 *
 *  @param[in] x        Pointer to the input data array.
 *  @param[in] n        Number of scalar words in the input array.
 *  @param[out] y       Output value in the same precision as the input array.
 */
typedef void (*nk_kernel_trigonometry_punned_t)(void const *x, nk_size_t n, void *y);

/**
 *  @brief  Type-punned function pointer for mesh superposition metrics (RMSD, Kabsch, Umeyama).
 *          All mesh functions share a unified signature with rotation matrix and scale factor.
 *
 *  The transformation aligns point cloud A to B: a'_i = scale * R * (a_i - a_centroid) + b_centroid
 *
 *  @param[in] a            Pointer to first point cloud (source, interleaved xyz coordinates).
 *  @param[in] b            Pointer to second point cloud (target, interleaved xyz coordinates).
 *  @param[in] n            Number of 3D points in each cloud.
 *  @param[out] a_centroid  Output centroid of first cloud (3 values), or NULL.
 *  @param[out] b_centroid  Output centroid of second cloud (3 values), or NULL.
 *  @param[out] rotation    Output 3x3 rotation matrix (9 values, row-major), or NULL.
 *  @param[out] scale       Output scale factor (1.0 for RMSD/Kabsch), or NULL.
 *  @param[out] d           Output RMSD value as a double-precision float.
 */
typedef void (*nk_metric_mesh_punned_t)(void const *a, void const *b, nk_size_t n, //
                                        void *a_centroid, void *b_centroid,        //
                                        void *rotation, void *scale,               //
                                        void *d);

/**
 *  @brief  Type-punned function pointer for horizontal sum reduction.
 *          Implements `result = sum(data[i])` over strided elements.
 *
 *  @param[in] data         Pointer to the input data array.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes.
 *  @param[out] result      Output sum (type may be widened for precision).
 */
typedef void (*nk_kernel_reduce_add_punned_t)(void const *data, nk_size_t count, nk_size_t stride_bytes, void *result);

/**
 *  @brief  Type-punned function pointer for horizontal min/max reduction with argmin/argmax.
 *          Implements `value = min/max(data[i]), index = argmin/argmax(data[i])`.
 *
 *  @param[in] data         Pointer to the input data array.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes.
 *  @param[out] value       Output min/max value (same type as input).
 *  @param[out] index       Output index of the min/max value.
 */
typedef void (*nk_kernel_reduce_minmax_punned_t)(void const *data, nk_size_t count, nk_size_t stride_bytes, void *value,
                                                 nk_size_t *index);

/**
 *  @brief  Type-punned function pointer for a NumKong public interface.
 *
 *  Can be a `nk_metric_dense_punned_t`, `nk_metric_sparse_punned_t`, `nk_metric_curved_punned_t`,
 *  `nk_metric_mesh_punned_t`, `nk_kernel_fma_punned_t`, `nk_kernel_wsum_punned_t`,
 *  `nk_kernel_scale_punned_t`, `nk_kernel_sum_punned_t`, `nk_kernel_trigonometry_punned_t`,
 *  `nk_kernel_reduce_add_punned_t`, or `nk_kernel_reduce_minmax_punned_t`.
 */
typedef void (*nk_kernel_punned_t)(void *);

#if NK_DYNAMIC_DISPATCH
NK_DYNAMIC nk_capability_t nk_capabilities(void);
NK_DYNAMIC void nk_find_kernel_punned( //
    nk_metric_kind_t kind,             //
    nk_datatype_t datatype,            //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output);
NK_DYNAMIC int nk_configure_thread(nk_capability_t);
#else
NK_PUBLIC nk_capability_t nk_capabilities(void);
NK_PUBLIC void nk_find_kernel_punned(  //
    nk_metric_kind_t kind,             //
    nk_datatype_t datatype,            //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output);
NK_PUBLIC int nk_configure_thread(nk_capability_t);
#endif

#if NK_TARGET_X86_

/**
 *  @brief  Function to flush denormalized numbers to zero on x86 CPUs.
 *  @param  capabilities A bitmask of capabilities. If `nk_cap_sapphire_amx_k` is set,
 *          also requests OS permission for AMX tile data on Linux.
 *  @note   This should be called on each thread before any SIMD operations to avoid performance penalties.
 *  @return 1 if the operation was successful, 0 otherwise.
 */
NK_PUBLIC int nk_configure_thread_x86_(nk_capability_t capabilities) {
#if defined(_MSC_VER)
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= 1 << 15; // bit 15 = Flush-To-Zero (FTZ)
    mxcsr |= 1 << 6;  // bit 6  = Denormals-Are-Zero (DAZ)
    _mm_setcsr(mxcsr);
#else // GCC, Clang, ICC
    unsigned int mxcsr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 1 << 15; // bit 15 = Flush-To-Zero (FTZ)
    mxcsr |= 1 << 6;  // bit 6  = Denormals-Are-Zero (DAZ)
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
#endif

    // Intel AMX (Advanced Matrix Extensions) requires explicit permission from the OS before use.
    // On Linux, this is done via the `arch_prctl` system call with ARCH_REQ_XCOMP_PERM.
#if defined(NK_DEFINED_LINUX_) && NK_TARGET_SAPPHIRE
    if (capabilities & nk_cap_sapphire_amx_k) {
        int const ARCH_REQ_XCOMP_PERM = 0x1023;
        unsigned long const XFEATURE_XTILEDATA = 18;
        syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    }
#else
    (void)capabilities;
#endif
    return 1;
}

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit x86 machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `nk_capability_t` enum value.
 */
NK_PUBLIC nk_capability_t nk_capabilities_x86_(void) {

    /// The states of 4 registers populated for a specific "cpuid" assembly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7, info7sub1;

#if defined(_MSC_VER)
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
    __cpuidex(info7sub1.array, 7, 1);
#else // GCC, Clang, ICC
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
        : "a"(1), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
        : "a"(7), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info7sub1.named.eax), "=b"(info7sub1.named.ebx), "=c"(info7sub1.named.ecx), "=d"(info7sub1.named.edx)
        : "a"(7), "c"(1));
#endif

    // Check for AVX2 (Function ID 7, EBX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L148
    unsigned supports_avx2 = (info7.named.ebx & 0x00000020) != 0;
    // Check for F16C (Function ID 1, ECX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L107
    unsigned supports_f16c = (info1.named.ecx & 0x20000000) != 0;
    unsigned supports_fma = (info1.named.ecx & 0x00001000) != 0;
    // Check for AVX512F (Function ID 7, EBX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L155
    unsigned supports_avx512f = (info7.named.ebx & 0x00010000) != 0;
    // Check for AVX512FP16 (Function ID 7, EDX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L198C9-L198C23
    unsigned supports_avx512fp16 = (info7.named.edx & 0x00800000) != 0;
    // Check for AVX512VNNI (Function ID 7, ECX register)
    unsigned supports_avx512vnni = (info7.named.ecx & 0x00000800) != 0;
    // Check for AVX512IFMA (Function ID 7, EBX register)
    unsigned supports_avx512ifma = (info7.named.ebx & 0x00200000) != 0;
    // Check for AVX512BITALG (Function ID 7, ECX register)
    unsigned supports_avx512bitalg = (info7.named.ecx & 0x00001000) != 0;
    // Check for AVX512VBMI2 (Function ID 7, ECX register)
    unsigned supports_avx512vbmi2 = (info7.named.ecx & 0x00000040) != 0;
    // Check for AVX512VPOPCNTDQ (Function ID 7, ECX register)
    unsigned supports_avx512vpopcntdq = (info7.named.ecx & 0x00004000) != 0;
    // Check for AVX512BF16 (Function ID 7, Sub-leaf 1, EAX register)
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L205
    unsigned supports_avx512bf16 = (info7sub1.named.eax & 0x00000020) != 0;
    // Clang doesn't show the VP2INTERSECT flag, but we can get it from QEMU
    // https://stackoverflow.com/a/68289220/2766161
    unsigned supports_avx512vp2intersect = (info7.named.edx & 0x00000100) != 0;
    // Check for AMX-TILE (Function ID 7, EDX bit 24) - base AMX support
    unsigned supports_amx_tile = (info7.named.edx & 0x01000000) != 0;
    // Check for AMX-BF16 (Function ID 7, EDX bit 22)
    unsigned supports_amx_bf16 = (info7.named.edx & 0x00400000) != 0;
    // Check for AMX-INT8 (Function ID 7, EDX bit 25)
    unsigned supports_amx_int8 = (info7.named.edx & 0x02000000) != 0;
    // Check for AMX-FP16 (Function ID 7, Sub-leaf 1, EAX bit 21)
    unsigned supports_amx_fp16 = (info7sub1.named.eax & 0x00200000) != 0;

    // Convert specific features into CPU generations
    unsigned supports_haswell = supports_avx2 && supports_f16c && supports_fma;
    unsigned supports_skylake = supports_avx512f;
    unsigned supports_ice = supports_avx512vnni && supports_avx512ifma && supports_avx512bitalg &&
                            supports_avx512vbmi2 && supports_avx512vpopcntdq;
    unsigned supports_genoa = supports_avx512bf16;
    unsigned supports_sapphire = supports_avx512fp16;
    // We don't want to accidentally enable AVX512VP2INTERSECT on Intel Tiger Lake CPUs
    unsigned supports_turin = supports_avx512vp2intersect && supports_avx512bf16;
    unsigned supports_sierra = 0;
    // Sapphire Rapids AMX: requires AMX-TILE, AMX-BF16, and AMX-INT8
    unsigned supports_sapphire_amx = supports_amx_tile && supports_amx_bf16 && supports_amx_int8;
    // Granite Rapids AMX: requires Sapphire AMX plus AMX-FP16
    unsigned supports_granite_amx = supports_sapphire_amx && supports_amx_fp16;

    return (nk_capability_t)(                             //
        (nk_cap_haswell_k * supports_haswell) |           //
        (nk_cap_skylake_k * supports_skylake) |           //
        (nk_cap_ice_k * supports_ice) |                   //
        (nk_cap_genoa_k * supports_genoa) |               //
        (nk_cap_sapphire_k * supports_sapphire) |         //
        (nk_cap_turin_k * supports_turin) |               //
        (nk_cap_sierra_k * supports_sierra) |             //
        (nk_cap_sapphire_amx_k * supports_sapphire_amx) | //
        (nk_cap_granite_amx_k * supports_granite_amx) |   //
        (nk_cap_serial_k));
}

#endif // NK_TARGET_X86_

#if NK_TARGET_ARM_

/*  Compiling the next section one may get: selected processor does not support system register name
 * 'id_aa64zfr0_el1'. Suppressing assembler errors is very complicated, so when dealing with older ARM CPUs it's
 * simpler to compile this function targeting newer ones.
 */
#pragma GCC push_options
#pragma GCC target("arch=armv8.5-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.5-a+sve"))), apply_to = function)

#if NK_HAS_POSIX_EXTENSIONS_
/** @brief SIGILL handler for `mrs` instruction testing on Linux ARM */
static sigjmp_buf nk_mrs_test_jump_buffer_;
static void nk_mrs_test_sigill_handler_(int sig) {
    (void)sig; // Unused parameter
    siglongjmp(nk_mrs_test_jump_buffer_, 1);
}
#endif

/**
 *  @brief  Function to flush denormalized numbers to zero on Arm CPUs.
 *  @param  capabilities A bitmask of capabilities (unused on ARM, for API consistency).
 *  @note   This should be called on each thread before any SIMD operations to avoid performance penalties.
 *  @note   On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
 *  @return 1 if the operation was successful, 0 otherwise.
 */
NK_PUBLIC int nk_configure_thread_arm_(nk_capability_t capabilities) {
    (void)capabilities;
#if defined(NK_DEFINED_APPLE_)
    // https://stackoverflow.com/a/19904907/2766161
    // https://stackoverflow.com/a/78252076/2766161
    int is_success = fesetenv(FE_DFL_DISABLE_DENORMS_ENV) == 0;
    return is_success;
#elif defined(NK_DEFINED_LINUX_)
    // For Linux, we can toggle bits in the Floating-point Control Register (FPCR)
    // https://developer.arm.com/documentation/ddi0601/2024-12/AArch64-Registers/FPCR--Floating-point-Control-Register
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1 << 19); // bit 19 = FZ16 (Flush half-precision to zero)
    fpcr |= (1 << 24); // bit 24 = FZ (Flush subnormals to zero)
    fpcr |= (1 << 25); // bit 25 = DN (Force Default NaN instead of preserving payload)
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    return 1;
#else
    return 0;
#endif
}

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit Arm machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `nk_capability_t` enum value.
 */
NK_PUBLIC nk_capability_t nk_capabilities_arm_(void) {
#if defined(NK_DEFINED_APPLE_)
    // On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
    unsigned supports_neon = 0, supports_fp16 = 0, supports_bf16 = 0, supports_i8mm = 0;
    size_t size = sizeof(supports_neon);
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &supports_fp16, &size, NULL, 0) != 0) supports_fp16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &supports_bf16, &size, NULL, 0) != 0) supports_bf16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &supports_i8mm, &size, NULL, 0) != 0) supports_i8mm = 0;

    return (nk_capability_t)(                                     //
        (nk_cap_neon_k * (supports_neon)) |                       //
        (nk_cap_neon_f16_k * (supports_neon && supports_fp16)) |  //
        (nk_cap_neon_bf16_k * (supports_neon && supports_bf16)) | //
        (nk_cap_neon_i8_k * (supports_neon && supports_i8mm)) |   //
        (nk_cap_serial_k));

#elif defined(NK_DEFINED_LINUX_)

    // Depending on the environment, reading system registers may cause SIGILL.
    // One option to avoid the crash is to use `getauxval(AT_HWCAP)` and `getauxval(AT_HWCAP2)`,
    // Linux APIs, but those aren't as informative as reading the registers directly.
    // So before reading the ID registers, we set up a signal handler to catch SIGILL
    // and probe one of the registers, reverting back to the old signal handler afterwards.
    //
    // This issue was originally observed in: https://github.com/ashvardanian/NumKong/issues/279
#if NK_HAS_POSIX_EXTENSIONS_
    struct sigaction action_new, action_old;
    action_new.sa_handler = nk_mrs_test_sigill_handler_;
    sigemptyset(&action_new.sa_mask);
    action_new.sa_flags = 0;

    int mrs_works = 0;
    if (sigaction(SIGILL, &action_new, &action_old) == 0) {
        if (sigsetjmp(nk_mrs_test_jump_buffer_, 1) == 0) {
            unsigned long midr_value;
            __asm__ __volatile__("mrs %0, MIDR_EL1" : "=r"(midr_value));
            mrs_works = 1;
        }
        sigaction(SIGILL, &action_old, NULL);
    }

    // Early exit if `mrs` doesn't work - return conservative NEON-only capabilities
    if (!mrs_works) return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);
#else  // NK_HAS_POSIX_EXTENSIONS_
    // Without POSIX signal handlers, fall back to conservative NEON capabilities.
    return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);
#endif // NK_HAS_POSIX_EXTENSIONS_

    // Read CPUID registers directly
    unsigned long id_aa64isar0_el1 = 0, id_aa64isar1_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;

    // Now let's unpack the status flags from ID_AA64ISAR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR0-EL1--AArch64-Instruction-Set-Attribute-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR0_EL1" : "=r"(id_aa64isar0_el1));
    // DP, bits [47:44] of ID_AA64ISAR0_EL1
    unsigned supports_integer_dot_products = ((id_aa64isar0_el1 >> 44) & 0xF) >= 1;
    // Now let's unpack the status flags from ID_AA64ISAR1_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ISAR1-EL1--AArch64-Instruction-Set-Attribute-Register-1?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64ISAR1_EL1" : "=r"(id_aa64isar1_el1));
    // I8MM, bits [55:52] of ID_AA64ISAR1_EL1
    unsigned supports_i8mm = ((id_aa64isar1_el1 >> 52) & 0xF) >= 1;
    // BF16, bits [47:44] of ID_AA64ISAR1_EL1
    unsigned supports_bf16 = ((id_aa64isar1_el1 >> 44) & 0xF) >= 1;

    // Now let's unpack the status flags from ID_AA64PFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64PFR0-EL1--AArch64-Processor-Feature-Register-0?lang=en
    __asm__ __volatile__("mrs %0, ID_AA64PFR0_EL1" : "=r"(id_aa64pfr0_el1));
    // SVE, bits [35:32] of ID_AA64PFR0_EL1
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    // AdvSIMD, bits [23:20] of ID_AA64PFR0_EL1 can be used to check for `fp16` support
    //
    //  - 0b0000: integers, single, double precision arithmetic
    //  - 0b0001: includes support for half-precision floating-point arithmetic
    //  - 0b1111: NEON is not supported?!
    //
    // That's a really weird way to encode lack of NEON support, but it's important to
    // check in case we are running on R-profile CPUs.
    unsigned supports_fp16 = ((id_aa64pfr0_el1 >> 20) & 0xF) == 0x1;
    unsigned supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;

    // Now let's unpack the status flags from ID_AA64ZFR0_EL1
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64ZFR0-EL1--SVE-Feature-ID-Register-0?lang=en
    if (supports_sve) __asm__ __volatile__("mrs %0, ID_AA64ZFR0_EL1" : "=r"(id_aa64zfr0_el1));
    // I8MM, bits [47:44] of ID_AA64ZFR0_EL1
    unsigned supports_sve_i8mm = ((id_aa64zfr0_el1 >> 44) & 0xF) >= 1;
    // BF16, bits [23:20] of ID_AA64ZFR0_EL1
    unsigned supports_sve_bf16 = ((id_aa64zfr0_el1 >> 20) & 0xF) >= 1;
    // SVEver, bits [3:0] can be used to check for capability levels:
    //
    //  - 0b0000: SVE is implemented
    //  - 0b0001: SVE2 is implemented
    //  - 0b0010: SVE2.1 is implemented
    //
    // This value must match the existing indicator obtained from ID_AA64PFR0_EL1:
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_sve2p1 = ((id_aa64zfr0_el1) & 0xF) >= 2;

    return (nk_capability_t)(                                                                    //
        (nk_cap_neon_k * (supports_neon)) |                                                      //
        (nk_cap_neon_f16_k * (supports_neon && supports_fp16)) |                                 //
        (nk_cap_neon_bf16_k * (supports_neon && supports_bf16)) |                                //
        (nk_cap_neon_i8_k * (supports_neon && supports_i8mm && supports_integer_dot_products)) | //
        (nk_cap_sve_k * (supports_sve)) |                                                        //
        (nk_cap_sve_f16_k * (supports_sve && supports_fp16)) |                                   //
        (nk_cap_sve_bf16_k * (supports_sve && supports_sve_bf16)) |                              //
        (nk_cap_sve_i8_k * (supports_sve && supports_sve_i8mm)) |                                //
        (nk_cap_sve2_k * (supports_sve2)) |                                                      //
        (nk_cap_sve2p1_k * (supports_sve2p1)) |                                                  //
        (nk_cap_serial_k));
#elif defined(NK_DEFINED_WINDOWS_)

    unsigned supports_neon = 0, supports_dp = 0;

    // On Windows ARM, use the `IsProcessorFeaturePresent` API for capability detection.
    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
#if defined(PF_ARM_V8_INSTRUCTIONS_AVAILABLE)
    supports_neon = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    supports_dp = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE);
#endif

    // Windows API doesn't provide reliable detection for FP16, BF16.
    return (nk_capability_t)(                                 //
        (nk_cap_neon_k * (supports_neon)) |                   //
        (nk_cap_neon_i8_k * (supports_neon && supports_dp)) | //
        (nk_cap_serial_k));

#else // Unknown platform

    // Conservative fallback for unknown platforms: NEON is mandatory in ARMv8-A (ARM64)
    return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);

#endif
}

#pragma clang attribute pop
#pragma GCC pop_options

#endif

/**
 *  @brief  Function to flush @b denormalized numbers to zero to avoid performance penalties.
 *  @param  capabilities A bitmask of capabilities. If `nk_cap_sapphire_amx_k` is set,
 *          also requests OS permission for AMX tile data on Linux x86.
 *  @return 1 if the operation was successful, 0 otherwise.
 *
 *  When facing denormalized values Fused-Multiply-Add (FMA) operations can be up to 30x slower,
 *  as measured on Intel Sapphire Rapids: https://github.com/ashvardanian/ParallelReductionsBenchmark
 */
NK_PUBLIC int nk_configure_thread_(nk_capability_t capabilities) {
#if NK_TARGET_X86_
    return nk_configure_thread_x86_(capabilities);
#endif // NK_TARGET_X86_
#if NK_TARGET_ARM_
    return nk_configure_thread_arm_(capabilities);
#endif // NK_TARGET_ARM_
    (void)capabilities;
    return 0;
}

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit x86 machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `nk_capability_t` enum value.
 */
NK_PUBLIC nk_capability_t nk_capabilities_(void) {
#if NK_TARGET_X86_
    return nk_capabilities_x86_();
#endif // NK_TARGET_X86_
#if NK_TARGET_ARM_
    return nk_capabilities_arm_();
#endif // NK_TARGET_ARM_
    return nk_cap_serial_k;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type"

#ifdef __cplusplus //! option "-Wvolatile" is valid for C++/ObjC++ but not for C
#pragma GCC diagnostic ignored "-Wvolatile"
#pragma clang diagnostic ignored "-Wvolatile"
#endif

NK_INTERNAL void nk_find_kernel_punned_f64_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64_sve, *c = nk_cap_sve_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f64_sve, *c = nk_cap_sve_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f64_sve, *c = nk_cap_sve_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f64_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f32_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32_sve, *c = nk_cap_sve_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f32_sve, *c = nk_cap_sve_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f32_sve, *c = nk_cap_sve_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f32_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_haversine_k: *m = (m_t)&nk_haversine_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vincenty_k: *m = (m_t)&nk_vincenty_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sin_k: *m = (m_t)&nk_sin_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_cos_k: *m = (m_t)&nk_cos_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_atan_k: *m = (m_t)&nk_atan_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f16_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE_F16
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16_sve, *c = nk_cap_sve_f16_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f16_sve, *c = nk_cap_sve_f16_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f16_sve, *c = nk_cap_sve_f16_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f16_sve, *c = nk_cap_sve_f16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON_F16
    if (v & nk_cap_neon_f16_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f16_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f16_neon, *c = nk_cap_neon_f16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f16_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_f16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_f16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_bf16_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE_BF16
    if (v & nk_cap_sve_bf16_k) switch (k) {
        case nk_metric_angular_k: *m = (m_t)&nk_angular_bf16_sve, *c = nk_cap_sve_bf16_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_bf16_sve, *c = nk_cap_sve_bf16_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_bf16_sve, *c = nk_cap_sve_bf16_k; return;
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_sve2, *c = nk_cap_sve_bf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON_BF16
    if (v & nk_cap_neon_bf16_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_bf16_neon, *c = nk_cap_neon_bf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_bf16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_bf16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_jsd_k: *m = (m_t)&nk_jsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kld_k: *m = (m_t)&nk_kld_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_rmsd_k: *m = (m_t)&nk_rmsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_kabsch_k: *m = (m_t)&nk_kabsch_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_umeyama_k: *m = (m_t)&nk_umeyama_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i8_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON_I8
    if (v & nk_cap_neon_i8_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_i8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_i8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_i8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_i8_neon, *c = nk_cap_neon_i8_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON_F16 //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_neon_f16_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i8_neon, *c = nk_cap_neon_f16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i8_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_i8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_i8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_i8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_i8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i8_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i8_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i8_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}
NK_INTERNAL void nk_find_kernel_punned_u8_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON_I8
    if (v & nk_cap_neon_i8_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_u8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_u8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_u8_neon, *c = nk_cap_neon_i8_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_u8_neon, *c = nk_cap_neon_i8_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON_F16 //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_neon_f16_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u8_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u8_neon, *c = nk_cap_neon_f16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u8_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u8_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_u8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_u8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_u8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_u8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u8_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u8_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_angular_k: *m = (m_t)&nk_angular_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2sq_k: *m = (m_t)&nk_l2sq_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_l2_k: *m = (m_t)&nk_l2_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e4m3_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e4m3_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e4m3_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e4m3_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e4m3_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e4m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e5m2_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e5m2_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e5m2_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e5m2_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e5m2_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_e5m2_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_b8_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_hamming_k: *m = (m_t)&nk_hamming_b8_sve, *c = nk_cap_sve_k; return;
        case nk_metric_jaccard_k: *m = (m_t)&nk_jaccard_b8_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_hamming_k: *m = (m_t)&nk_hamming_b8_neon, *c = nk_cap_neon_k; return;
        case nk_metric_jaccard_k: *m = (m_t)&nk_jaccard_b8_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_hamming_k: *m = (m_t)&nk_hamming_b8_ice, *c = nk_cap_ice_k; return;
        case nk_metric_jaccard_k: *m = (m_t)&nk_jaccard_b8_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_hamming_k: *m = (m_t)&nk_hamming_b8_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_jaccard_k: *m = (m_t)&nk_jaccard_b8_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_hamming_k: *m = (m_t)&nk_hamming_b8_serial, *c = nk_cap_serial_k; return;
        case nk_metric_jaccard_k: *m = (m_t)&nk_jaccard_b8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f64c_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64c_sve, *c = nk_cap_sve_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f64c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f64c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f64c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f32c_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32c_sve, *c = nk_cap_sve_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f32c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32c_neon, *c = nk_cap_neon_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f32c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32c_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f32c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32c_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f32c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f32c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f32c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f16c_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE_F16
    if (v & nk_cap_sve_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16c_sve, *c = nk_cap_sve_f16_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f16c_sve, *c = nk_cap_sve_f16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON_F16
    if (v & nk_cap_neon_f16_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16c_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f16c_neon, *c = nk_cap_neon_f16_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16c_neon, *c = nk_cap_neon_bf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16c_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f16c_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16c_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16c_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f16c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_f16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_bf16c_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                              nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON_BF16
    if (v & nk_cap_neon_bf16_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16c_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_bf16c_neon, *c = nk_cap_neon_bf16_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_neon, *c = nk_cap_neon_bf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_dot_k: *m = (m_t)&nk_dot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_vdot_k: *m = (m_t)&nk_vdot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_metric_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u16_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u16_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u16_neon, *c = nk_cap_neon_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u16_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u16_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u16_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u16_ice, *c = nk_cap_skylake_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u16_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u16_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i16_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i16_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i16_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i16_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i16_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i16_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u32_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u32_turin, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u32_ice, *c = nk_cap_skylake_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_intersect_k: *m = (m_t)&nk_intersect_u32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i32_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i32_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i32_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i32_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i64_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i64_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_i64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_i64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_i64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_i64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u64_(nk_capability_t v, nk_metric_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u64_neon, *c = nk_cap_neon_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u64_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_metric_fma_k: *m = (m_t)&nk_fma_u64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_scale_k: *m = (m_t)&nk_scale_u64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_sum_k: *m = (m_t)&nk_sum_u64_serial, *c = nk_cap_serial_k; return;
        case nk_metric_wsum_k: *m = (m_t)&nk_wsum_u64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

/**
 *  @brief  Determines the best suited metric implementation based on the given datatype,
 *          supported and allowed by hardware capabilities.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param datatype The data type for which the metric needs to be evaluated.
 *  @param supported The hardware capabilities supported by the CPU.
 *  @param allowed The hardware capabilities allowed for use.
 *  @param kernel_output Output variable for the selected similarity function.
 *  @param capability_output Output variable for the utilized hardware capabilities.
 */
NK_INTERNAL void nk_find_kernel_punned_( //
    nk_metric_kind_t kind,               //
    nk_datatype_t datatype,              //
    nk_capability_t supported,           //
    nk_capability_t allowed,             //
    nk_kernel_punned_t *kernel_output,   //
    nk_capability_t *capability_output) {

    // Modern compilers abso-freaking-lutely love optimizing-out my logic!
    // Just marking the variables as `volatile` is not enough, so we have
    // to add inline assembly to further discourage them!
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif

    nk_kernel_punned_t *m = kernel_output;
    nk_capability_t *c = capability_output;
    nk_capability_t viable = (nk_capability_t)(supported & allowed);

    switch (datatype) {

    case nk_f64_k: nk_find_kernel_punned_f64_(viable, kind, m, c); return;
    case nk_f32_k: nk_find_kernel_punned_f32_(viable, kind, m, c); return;
    case nk_f16_k: nk_find_kernel_punned_f16_(viable, kind, m, c); return;
    case nk_bf16_k: nk_find_kernel_punned_bf16_(viable, kind, m, c); return;
    case nk_e4m3_k: nk_find_kernel_punned_e4m3_(viable, kind, m, c); return;
    case nk_e5m2_k: nk_find_kernel_punned_e5m2_(viable, kind, m, c); return;

    case nk_f32c_k: nk_find_kernel_punned_f32c_(viable, kind, m, c); return;
    case nk_f64c_k: nk_find_kernel_punned_f64c_(viable, kind, m, c); return;
    case nk_f16c_k: nk_find_kernel_punned_f16c_(viable, kind, m, c); return;
    case nk_bf16c_k: nk_find_kernel_punned_bf16c_(viable, kind, m, c); return;

    case nk_i8_k: nk_find_kernel_punned_i8_(viable, kind, m, c); return;
    case nk_i16_k: nk_find_kernel_punned_i16_(viable, kind, m, c); return;
    case nk_i32_k: nk_find_kernel_punned_i32_(viable, kind, m, c); return;
    case nk_i64_k: nk_find_kernel_punned_i64_(viable, kind, m, c); return;

    case nk_u8_k: nk_find_kernel_punned_u8_(viable, kind, m, c); return;
    case nk_u16_k: nk_find_kernel_punned_u16_(viable, kind, m, c); return;
    case nk_u32_k: nk_find_kernel_punned_u32_(viable, kind, m, c); return;
    case nk_u64_k: nk_find_kernel_punned_u64_(viable, kind, m, c); return;

    case nk_b8_k: nk_find_kernel_punned_b8_(viable, kind, m, c); return;

    // These data-types are not supported yet
    case nk_i4x2_k: break;
    case nk_datatype_unknown_k: break;
    default: break;
    }

    // Replace with zeros if no suitable implementation was found
    *m = (nk_kernel_punned_t)0;
    *c = (nk_capability_t)0;

    // Modern compilers abso-freaking-lutely love optimizing-out my logic!
    // Just marking the variables as `volatile` is not enough, so we have
    // to add inline assembly to further discourage them!
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop

/**
 *  @brief  Selects the most suitable metric implementation based on the given metric kind, datatype,
 *          and allowed capabilities. @b Don't call too often and prefer caching the `nk_capabilities()`.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param datatype The data type for which the metric needs to be evaluated.
 *  @param allowed The hardware capabilities allowed for use.
 *  @return A function pointer to the selected metric implementation.
 */
NK_PUBLIC nk_kernel_punned_t nk_metric_punned( //
    nk_metric_kind_t kind,                     //
    nk_datatype_t datatype,                    //
    nk_capability_t allowed) {

    nk_kernel_punned_t result = 0;
    nk_capability_t c = nk_cap_serial_k;
    nk_capability_t supported = nk_capabilities();
    nk_find_kernel_punned(kind, datatype, supported, allowed, &result, &c);
    return result;
}

#if NK_DYNAMIC_DISPATCH

/*  Run-time feature-testing functions
 *
 *  - Check if the CPU supports NEON or SVE extensions on Arm
 *  - Check if the CPU supports AVX2 and F16C extensions on Haswell x86 CPUs and newer
 *  - Check if the CPU supports AVX512F and AVX512BW extensions on Skylake x86 CPUs and newer
 *  - Check if the CPU supports AVX512VNNI, AVX512IFMA, AVX512BITALG, AVX512VBMI2, and AVX512VPOPCNTDQ
 *    extensions on Ice Lake x86 CPUs and newer
 *  - Check if the CPU supports AVX512BF16 extensions on Genoa x86 CPUs and newer
 *  - Check if the CPU supports AVX512FP16 extensions on Sapphire Rapids x86 CPUs and newer
 *  - Check if the CPU supports AVX2VP2INTERSECT extensions on Turin x86 CPUs and newer
 *
 *  @return 1 if the CPU supports the SIMD instruction set, 0 otherwise.
 */
NK_DYNAMIC nk_capability_t nk_capabilities(void);
NK_DYNAMIC int nk_configure_thread(nk_capability_t);
NK_DYNAMIC int nk_uses_dynamic_dispatch(void);
NK_DYNAMIC int nk_uses_neon(void);
NK_DYNAMIC int nk_uses_neon_f16(void);
NK_DYNAMIC int nk_uses_neon_bf16(void);
NK_DYNAMIC int nk_uses_neon_i8(void);
NK_DYNAMIC int nk_uses_sve(void);
NK_DYNAMIC int nk_uses_sve_f16(void);
NK_DYNAMIC int nk_uses_sve_bf16(void);
NK_DYNAMIC int nk_uses_sve_i8(void);
NK_DYNAMIC int nk_uses_sve2(void);
NK_DYNAMIC int nk_uses_haswell(void);
NK_DYNAMIC int nk_uses_skylake(void);
NK_DYNAMIC int nk_uses_ice(void);
NK_DYNAMIC int nk_uses_genoa(void);
NK_DYNAMIC int nk_uses_sapphire(void);
NK_DYNAMIC int nk_uses_turin(void);
NK_DYNAMIC int nk_uses_sierra(void);

#else

/*  Compile-time feature-testing functions
 *
 *  - Check if the CPU supports NEON or SVE extensions on Arm
 *  - Check if the CPU supports AVX2 and F16C extensions on Haswell x86 CPUs and newer
 *  - Check if the CPU supports AVX512F and AVX512BW extensions on Skylake x86 CPUs and newer
 *  - Check if the CPU supports AVX512VNNI, AVX512IFMA, AVX512BITALG, AVX512VBMI2, and AVX512VPOPCNTDQ
 *    extensions on Ice Lake x86 CPUs and newer
 *  - Check if the CPU supports AVX512BF16 extensions on Genoa x86 CPUs and newer
 *  - Check if the CPU supports AVX512FP16 extensions on Sapphire Rapids x86 CPUs and newer
 *
 *  @return 1 if the CPU supports the SIMD instruction set, 0 otherwise.
 */

NK_PUBLIC int nk_uses_neon(void) { return NK_TARGET_ARM_ && NK_TARGET_NEON; }
NK_PUBLIC int nk_uses_neon_f16(void) { return NK_TARGET_ARM_ && NK_TARGET_NEON_F16; }
NK_PUBLIC int nk_uses_neon_bf16(void) { return NK_TARGET_ARM_ && NK_TARGET_NEON_BF16; }
NK_PUBLIC int nk_uses_neon_i8(void) { return NK_TARGET_ARM_ && NK_TARGET_NEON_I8; }
NK_PUBLIC int nk_uses_sve(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE; }
NK_PUBLIC int nk_uses_sve_f16(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE_F16; }
NK_PUBLIC int nk_uses_sve_bf16(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE_BF16; }
NK_PUBLIC int nk_uses_sve_i8(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE_I8; }
NK_PUBLIC int nk_uses_sve2(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE2; }
NK_PUBLIC int nk_uses_haswell(void) { return NK_TARGET_X86_ && NK_TARGET_HASWELL; }
NK_PUBLIC int nk_uses_skylake(void) { return NK_TARGET_X86_ && NK_TARGET_SKYLAKE; }
NK_PUBLIC int nk_uses_ice(void) { return NK_TARGET_X86_ && NK_TARGET_ICE; }
NK_PUBLIC int nk_uses_genoa(void) { return NK_TARGET_X86_ && NK_TARGET_GENOA; }
NK_PUBLIC int nk_uses_sapphire(void) { return NK_TARGET_X86_ && NK_TARGET_SAPPHIRE; }
NK_PUBLIC int nk_uses_turin(void) { return NK_TARGET_X86_ && NK_TARGET_TURIN; }
NK_PUBLIC int nk_uses_sierra(void) { return NK_TARGET_X86_ && NK_TARGET_SIERRA; }
NK_PUBLIC int nk_uses_dynamic_dispatch(void) { return 0; }
NK_PUBLIC int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }
NK_PUBLIC nk_capability_t nk_capabilities(void) { return nk_capabilities_(); }
NK_PUBLIC void nk_find_kernel_punned(  //
    nk_metric_kind_t kind,             //
    nk_datatype_t datatype,            //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output) {
    nk_find_kernel_punned_(kind, datatype, supported, allowed, kernel_output, capability_output);
}

#endif

#ifdef __cplusplus
}
#endif

#endif // NK_H
