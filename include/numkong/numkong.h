/**
 *  @brief SIMD-accelerated Similarity Measures and Distance Functions.
 *  @file include/numkong.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
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

#ifndef NK_NUMKONG_H
#define NK_NUMKONG_H

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

#include "numkong/cast.h"         // Type Conversions
#include "numkong/set.h"          // Hamming, Jaccard
#include "numkong/curved.h"       // Mahalanobis, Bilinear Forms
#include "numkong/dot.h"          // Inner (dot) product, and its conjugate
#include "numkong/dots.h"         // GEMM-style MxN batched dot-products
#include "numkong/each.h"         // Weighted Sum, Fused-Multiply-Add
#include "numkong/geospatial.h"   // Haversine and Vincenty
#include "numkong/mesh.h"         // RMSD, Kabsch, Umeyama
#include "numkong/probability.h"  // Kullback-Leibler, Jensen–Shannon
#include "numkong/reduce.h"       // Horizontal reductions: sum, min, max
#include "numkong/sparse.h"       // Intersect
#include "numkong/spatial.h"      // L2, Angular
#include "numkong/trigonometry.h" // Sin, Cos, Atan

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

// On Linux RISC-V, we use getauxval and hwprobe syscall for capability detection
#if defined(NK_DEFINED_LINUX_) && NK_TARGET_RISCV_
#include <sys/auxv.h>    // `getauxval`, `AT_HWCAP`
#include <sys/syscall.h> // `syscall`
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
    nk_kernel_unknown_k = 0, ///< Unknown kernel kind

    // Classics:
    nk_kernel_dot_k = 'i',         ///< Inner product
    nk_kernel_vdot_k = 'v',        ///< Complex inner product
    nk_kernel_angular_k = 'a',     ///< Angular (cosine) distance
    nk_kernel_euclidean_k = 'e',   ///< Euclidean distance
    nk_kernel_sqeuclidean_k = '2', ///< Squared Euclidean distance

    // Binary:
    nk_kernel_hamming_k = 'h', ///< Hamming (or Manhattan) distance
    nk_kernel_jaccard_k = 'j', ///< Jaccard (or Tanimoto) coefficient

    // Curved Spaces:
    nk_kernel_bilinear_k = 'b',    ///< Bilinear form
    nk_kernel_mahalanobis_k = 'm', ///< Mahalanobis distance

    // Geospatial:
    nk_kernel_haversine_k = 'o', ///< Haversine distance
    nk_kernel_vincenty_k = 'O',  ///< Vincenty distance (ellipsoidal geodesic)

    // Probability:
    nk_kernel_kld_k = 'k', ///< Kullback-Leibler divergence
    nk_kernel_jsd_k = 's', ///< Jensen-Shannon divergence

    // Mesh superposition:
    nk_kernel_rmsd_k = 'r',    ///< RMSD without optimal superposition
    nk_kernel_kabsch_k = 'K',  ///< Kabsch RMSD with optimal rotation
    nk_kernel_umeyama_k = 'U', ///< Umeyama RMSD with optimal rotation and scale

    // Sparse Sets:
    nk_kernel_sparse_dot_k = 'd',       ///< Sparse dot product with weighted indices
    nk_kernel_sparse_intersect_k = 'x', ///< Equivalent to unnormalized Jaccard

    // BLAS-like operations:
    nk_kernel_each_scale_k = '*', ///< Scale
    nk_kernel_each_sum_k = '+',   ///< Sum
    nk_kernel_each_blend_k = 'w', ///< Weighted Sum
    nk_kernel_each_fma_k = 'f',   ///< Fused Multiply-Add

    // Trigonometric functions:
    nk_kernel_sin_k = 'S',  ///< Element-wise sine
    nk_kernel_cos_k = 'C',  ///< Element-wise cosine
    nk_kernel_atan_k = 'A', ///< Element-wise arctangent

    // Horizontal reductions:
    nk_kernel_reduce_add_k = 'R', ///< Horizontal sum reduction
    nk_kernel_reduce_min_k = '<', ///< Horizontal min reduction with argmin
    nk_kernel_reduce_max_k = '>', ///< Horizontal max reduction with argmax

    // Matrix multiplication (GEMM):
    nk_kernel_dots_packed_size_k = 'P', ///< GEMM packed buffer size
    nk_kernel_dots_pack_k = 'Q',        ///< GEMM B matrix packing
    nk_kernel_dots_k = 'G',             ///< GEMM computation
    nk_kernel_dots_compacting_k = 'g',  ///< GEMM computation with following renormalization
    nk_kernel_dots_symmetric_k = 'y',   ///< Symmetric Gram matrix (A × Aᵀ)

    nk_kernel_cast_k = '-', ///< Type casting from one type to another

} nk_kernel_kind_t;

/**
 *  @brief  64-bit bitmask representing SIMD capabilities of the target architecture.
 *
 *  Each bit represents a specific hardware capability. Multiple capabilities can be
 *  combined using bitwise OR. Use nk_capabilities() to detect available capabilities
 *  at runtime.
 */
typedef nk_u64_t nk_capability_t;

/**
 *  @brief  Serial (non-SIMD) fallback capability.
 *
 *  Always available. Used when no SIMD acceleration is possible or desired.
 *  All kernel dispatch functions fall back to serial implementations when
 *  no other capability matches.
 */
#define nk_cap_serial_k ((nk_capability_t)1)

/**
 *  @brief  Mask representing any capability.
 *
 *  Use this to accept any available SIMD implementation in kernel dispatch.
 */
#define nk_cap_any_k ((nk_capability_t)NK_U64_MAX)

/**
 *  @brief  Intel Haswell (2013) / AMD Excavator (2015) - AVX2 + FMA + F16C
 *
 *  Instructions: VFMADD*, VCVTPH2PS, VCVTPS2PH, 256-bit integer SIMD
 *  Used for: f32/f64 dot products, f16 conversion, integer operations
 *
 *  Detection: CPUID AVX2 + FMA + F16C flags
 */
#define nk_cap_haswell_k ((nk_capability_t)1 << 8)

/**
 *  @brief  Intel Skylake-X (2017) / AMD Zen 4 (2022) - AVX-512 Foundation
 *
 *  Instructions: 512-bit SIMD, masked operations, VFMADD512*
 *  Used for: All f32/f64 operations with 2x throughput vs AVX2
 *
 *  Detection: CPUID AVX512F + AVX512VL + AVX512BW + AVX512DQ
 */
#define nk_cap_skylake_k ((nk_capability_t)1 << 9)

/**
 *  @brief  Intel Ice Lake (2019) / AMD Zen 4 (2022) - AVX-512 VNNI + VPOPCNTDQ
 *
 *  Instructions: VPDPBUSD (i8 dot product), VPOPCNT (popcount)
 *  Used for: i8/u8 dot products, binary Hamming/Jaccard distance
 *
 *  Detection: CPUID AVX512VNNI + AVX512VPOPCNTDQ
 */
#define nk_cap_ice_k ((nk_capability_t)1 << 10)

/**
 *  @brief  AMD Genoa (2022) / Intel Cooper Lake (2020) - AVX-512 BF16
 *
 *  Instructions: VDPBF16PS (bf16 dot product to f32)
 *  Used for: bf16 dot products, angular similarity, L2 distance
 *
 *  Detection: CPUID AVX512BF16
 */
#define nk_cap_genoa_k ((nk_capability_t)1 << 11)

/**
 *  @brief  Intel Sapphire Rapids (2023) - AVX-512 FP16
 *
 *  Instructions: Native f16 arithmetic (VADDPH, VMULPH, VFMADDPH)
 *  Used for: f16 dot products, each ops without f32 conversion
 *
 *  Detection: CPUID AVX512FP16
 */
#define nk_cap_sapphire_k ((nk_capability_t)1 << 12)

/**
 *  @brief  AMD Turin (2024) - AVX-512 CD (Conflict Detection)
 *
 *  Instructions: VPCONFLICT (conflict detection for scatter/gather)
 *  Used for: Sparse vector operations, set intersection
 *
 *  Detection: CPUID AVX512CD
 */
#define nk_cap_turin_k ((nk_capability_t)1 << 13)

/**
 *  @brief  Intel Alder Lake (2021) - AVX2 + VNNI (no AVX-512)
 *
 *  Instructions: VPDPBUSD (i8 dot product) in 256-bit mode
 *  Used for: i8/u8 dot products on hybrid CPUs without AVX-512
 *
 *  Detection: CPUID AVX2 + AVXVNNI (different from AVX512VNNI)
 */
#define nk_cap_sierra_k ((nk_capability_t)1 << 14)

/**
 *  @brief  Intel Sapphire Rapids (2023) - AMX with INT8 and BF16
 *
 *  Instructions: TDPBSSD (i8 matmul), TDPBF16PS (bf16 matmul)
 *  Used for: Batch matrix multiplication (dots.h), ML inference
 *  Tile config: 16x64 bytes per tile, 8 tiles, 1KB each
 *
 *  Detection: CPUID AMX-INT8 + AMX-BF16, requires OS permission (XSAVE)
 */
#define nk_cap_sapphire_amx_k ((nk_capability_t)1 << 15)

/**
 *  @brief  Intel Granite Rapids (2024) - AMX with FP16
 *
 *  Instructions: TDPFP16PS (f16 matmul to f32 accumulator)
 *  Used for: f16 batch matrix multiplication
 *
 *  Detection: CPUID AMX-FP16
 */
#define nk_cap_granite_amx_k ((nk_capability_t)1 << 16)

/* Bits 17-31: Reserved for future x86 (AVX10, APX, etc.) */

/**
 *  @brief  ARM NEON baseline (ARMv8-A) - ASIMD
 *
 *  Instructions: 128-bit SIMD, FMLA, LD1/ST1
 *  Used for: f32/f64 operations, baseline ARM acceleration
 *  Devices: All 64-bit ARM (iPhone 5S+, RPi 3+, AWS Graviton)
 *
 *  Detection: Always available on AArch64
 */
#define nk_cap_neon_k ((nk_capability_t)1 << 32)

/**
 *  @brief  ARM NEON with FP16 (ARMv8.2-A) - FEAT_FP16
 *
 *  Instructions: Native f16 arithmetic (FADD, FMUL, FMLA for float16x8_t)
 *  Used for: f16 dot products, spatial metrics without f32 conversion
 *  Devices: Cortex-A75+ (2017), Apple A11+ (2017), Graviton 2+ (2020)
 *
 *  Detection: ID_AA64PFR0_EL1.FP == 1, or sysctl hw.optional.arm.FEAT_FP16
 */
#define nk_cap_neonhalf_k ((nk_capability_t)1 << 33)

/**
 *  @brief  ARM NEON with FP16 FML (ARMv8.2-A) - FEAT_FHM
 *
 *  Instructions: FMLAL/FMLSL (f16 ⨯ f16 → f32 widening multiply-accumulate)
 *  Used for: f16 dot products with f32 accumulator (20-48% faster than convert+FMA)
 *  Devices: Cortex-A76+ (2018), Apple A12+ (2018), Neoverse N1+ (2019)
 *
 *  Detection: ID_AA64ISAR0_EL1.FHM == 1, or sysctl hw.optional.arm.FEAT_FHM
 */
#define nk_cap_neonfhm_k ((nk_capability_t)1 << 34)

/**
 *  @brief  ARM NEON with BF16 (ARMv8.6-A) - FEAT_BF16
 *
 *  Instructions: BFDOT (bf16 ⨯ bf16 → f32 dot product), BFCVT (f32 → bf16)
 *  Used for: bf16 dot products, ML inference with bf16 weights
 *  Devices: Cortex-A78+ (2020), Apple M1+ (2020), Graviton 3+ (2021)
 *
 *  Detection: ID_AA64ISAR1_EL1.BF16 == 1, or sysctl hw.optional.arm.FEAT_BF16
 */
#define nk_cap_neonbfdot_k ((nk_capability_t)1 << 35)

/**
 *  @brief  ARM NEON with integer dot products (ARMv8.2-A) - FEAT_DotProd
 *
 *  Instructions: SDOT/UDOT (i8 ⨯ i8 → i32 dot product)
 *  Used for: i8/u8 dot products, quantized ML inference
 *  Devices: Cortex-A75+ (2017), Apple A11+ (2017), Graviton 2+ (2020)
 *
 *  Detection: ID_AA64ISAR0_EL1.DP == 1, or sysctl hw.optional.arm.FEAT_DotProd
 */
#define nk_cap_neonsdot_k ((nk_capability_t)1 << 36)

/**
 *  @brief  ARM SVE baseline (ARMv8.2-A) - Scalable Vector Extension
 *
 *  Instructions: Predicated ops (SVMLA, SVLD1), scalable 128-2048 bit vectors
 *  Used for: f32/f64 operations with hardware-defined vector length
 *  Devices: Fujitsu A64FX (512b), Graviton 3 (256b), Neoverse V1 (256b)
 *
 *  Detection: ID_AA64PFR0_EL1.SVE == 1, or sysctl hw.optional.arm.FEAT_SVE
 */
#define nk_cap_sve_k ((nk_capability_t)1 << 40)

/**
 *  @brief  ARM SVE with FP16 - FEAT_SVE + FEAT_FP16
 *
 *  Instructions: SVMLA_F16, SVLD1_F16 with predication
 *  Used for: f16 operations with scalable vectors
 *
 *  Detection: SVE + FP16 both present
 */
#define nk_cap_svehalf_k ((nk_capability_t)1 << 41)

/**
 *  @brief  ARM SVE with BF16 - FEAT_SVE_BF16
 *
 *  Instructions: SVBFDOT (bf16 dot product with SVE predication)
 *  Used for: bf16 angular similarity, L2 distance with SVE
 *  Devices: Neoverse V1+ (2021), Graviton 3+ (2021)
 *
 *  Detection: ID_AA64ZFR0_EL1.BF16 == 1
 */
#define nk_cap_svebfdot_k ((nk_capability_t)1 << 42)

/**
 *  @brief  ARM SVE with integer dot products - FEAT_SVE + FEAT_DotProd
 *
 *  Instructions: SVSDOT/SVUDOT (i8 dot products with predication)
 *  Used for: i8/u8 operations with SVE
 *
 *  Detection: SVE + DotProd both present
 */
#define nk_cap_svesdot_k ((nk_capability_t)1 << 43)

/**
 *  @brief  ARM SVE2 (ARMv9-A) - Scalable Vector Extension 2
 *
 *  Instructions: SVMATCH, SVHISTCNT (histogram), extended integer ops
 *  Used for: Set intersection, sparse operations
 *  Devices: Neoverse V2 (2023), Cortex-X3+ (2022), Graviton 4 (2024)
 *
 *  Detection: ID_AA64PFR0_EL1.SVE == 1 && ID_AA64ZFR0_EL1.SVEver ≥ 1
 */
#define nk_cap_sve2_k ((nk_capability_t)1 << 44)

/**
 *  @brief  ARM SVE2.1 (ARMv9.4-A) - SVE2 with additional instructions
 *
 *  Instructions: Extended predication, new gather/scatter modes
 *  Devices: Future ARM cores (2025+)
 *
 *  Detection: ID_AA64ZFR0_EL1.SVEver ≥ 2
 */
#define nk_cap_sve2p1_k ((nk_capability_t)1 << 45)

/**
 *  @brief  ARM SME baseline (ARMv9.2-A) - Scalable Matrix Extension
 *
 *  Instructions: FMOPA (outer product), ZA tiles, streaming mode
 *  Data types: I8I32, I16I32, F16F32, BF16F32, F32F32
 *  Used for: Batch matmul (dots.h), ML inference
 *  Devices: Apple M4 (2024), Cortex-X925 (2024)
 *
 *  Detection: ID_AA64PFR1_EL1.SME == 1
 */
#define nk_cap_sme_k ((nk_capability_t)1 << 48)

/**
 *  @brief  ARM SME2 (ARMv9.2-A) - Multi-vector operations, ZT0 LUT
 *
 *  Instructions: Multi-vector FMOPA, LUTI2/LUTI4 (weight decompression)
 *  Used for: 2-4x outer product throughput, INT4 weight decompression
 *  Devices: Apple M4 (2024), Cortex-X925 (2024)
 *
 *  Detection: ID_AA64SMFR0_EL1.SMEver ≥ 1
 */
#define nk_cap_sme2_k ((nk_capability_t)1 << 49)

/**
 *  @brief  ARM SME2.1 - Non-widening FP16/BF16, LUTv2
 *
 *  Instructions: FMOPA.H (f16 → f16), BFMOPA non-widening
 *  Used for: Native f16/bf16 accumulation without f32 conversion
 *  Devices: Apple M5 (2025), future Cortex cores
 *
 *  Detection: ID_AA64SMFR0_EL1.SMEver ≥ 2
 */
#define nk_cap_sme2p1_k ((nk_capability_t)1 << 50)

/**
 *  @brief  ARM SME F64 - Double precision outer products (FEAT_SME_F64F64)
 *
 *  Instructions: FMOPA.D (f64*f64 → f64 outer product)
 *  Used for: High-precision matmul
 *  Devices: Apple M4 (2024)
 *
 *  Detection: ID_AA64SMFR0_EL1.F64F64 == 1
 */
#define nk_cap_smef64_k ((nk_capability_t)1 << 51)

/**
 *  @brief  ARM SME F16F16 - Native f16 outer products (FEAT_SME_F16F16)
 *
 *  Instructions: FMOPA.H (f16 ⨯ f16 → f16 non-widening)
 *  Used for: f16 matmul without f32 conversion overhead
 *  Devices: Apple M5 (2025)
 *
 *  Detection: ID_AA64SMFR0_EL1.F16F16 == 1
 */
#define nk_cap_smehalf_k ((nk_capability_t)1 << 52)

/**
 *  @brief  ARM SME B16B16 - Native bf16 outer products (FEAT_SME_B16B16)
 *
 *  Instructions: BFMOPA non-widening (bf16 ⨯ bf16 → bf16)
 *  Used for: bf16 matmul without f32 conversion
 *  Devices: Apple M5 (2025)
 *
 *  Detection: ID_AA64SMFR0_EL1.B16B16 == 1
 */
#define nk_cap_smebf16_k ((nk_capability_t)1 << 53)

/**
 *  @brief  ARM SME LUTv2 - Extended lookup table (FEAT_SME_LUTv2)
 *
 *  Instructions: LUTI4 with 4-bit indices, 8-bit elements
 *  Used for: INT4 weight decompression for LLM inference
 *  Devices: Apple M5 (2025)
 *
 *  Detection: ID_AA64SMFR0_EL1.LUTv2 == 1
 */
#define nk_cap_smelut2_k ((nk_capability_t)1 << 54)

/**
 *  @brief  ARM SME FA64 - Full A64 in streaming mode (FEAT_SME_FA64)
 *
 *  Instructions: Enables all A64 instructions while streaming
 *  Used for: Mixed SME + NEON/SVE code without mode switching
 *
 *  Detection: ID_AA64SMFR0_EL1.FA64 == 1
 */
#define nk_cap_smefa64_k ((nk_capability_t)1 << 55)

/**
 *  @brief Capability flag for RISC-V Vector Extension (RVV 1.0).
 *
 *  Instructions: vsetvl, vle*, vse*, vwmul, vfwmul, vredsum, etc.
 *  Used for: Vector length agnostic SIMD with automatic tail handling.
 *
 *  Detection: getauxval(AT_HWCAP) & COMPAT_HWCAP_ISA_V
 */
#define nk_cap_spacemit_k ((nk_capability_t)1 << 56)

/**
 *  @brief  Capability of the RISC-V Vector Zvfh extension (vector half-precision f16).
 *
 *  Instructions: vle16 (f16), vfwmul.vv (f16 → f32), vfredusum, etc.
 *  Used for: Half-precision floating point vector operations.
 *
 *  Detection: Compile-time via __riscv_zvfh. Runtime detection requires parsing /proc/cpuinfo or HWCAP2.
 */
#define nk_cap_sifive_k ((nk_capability_t)1 << 57)

/**
 *  @brief  Capability of the RISC-V Vector Zvfbfwma extension (bf16 widening FMA).
 *
 *  Instructions: vfwmaccbf16.vv (f32 += bf16 × bf16), etc.
 *  Used for: BFloat16 AI/ML workloads with widening FMA for dot products.
 *
 *  Detection: Compile-time via __riscv_zvfbfwma. Runtime detection requires parsing /proc/cpuinfo or HWCAP2.
 */
#define nk_cap_xuantie_k ((nk_capability_t)1 << 58)

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
 *  @param[out] result    Nullable output buffer.
 *  @param[out] count     Always written.
 */
typedef void (*nk_sparse_intersect_punned_t)(void const *a, void const *b, nk_size_t a_length, nk_size_t b_length,
                                             void *result,      // nullable output buffer
                                             nk_size_t *count); // always written

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
typedef void (*nk_sparse_dot_punned_t)(void const *a, void const *b,                 //
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
 *          Implements the `y = α  × a + β` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor (type depends on input precision).
 *  @param[in] beta     Pointer to offset/bias term (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_each_scale_punned_t)(void const *a, nk_size_t n, void const *alpha, void const *beta, void *y);

/**
 *  @brief  Type-punned function pointer for element-wise Sum operations on dense vector representations.
 *          Implements the `y = a + b` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_each_sum_punned_t)(void const *a, void const *b, nk_size_t n, void *y);

/**
 *  @brief  Type-punned function pointer for Weighted Sum operations on dense vector representations.
 *          Implements the `y = α  × a + β  × b` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor for the first array (type depends on input precision).
 *  @param[in] beta     Pointer to scaling factor for the second array (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_each_blend_punned_t)(void const *a, void const *b, nk_size_t n, void const *alpha, void const *beta,
                                       void *y);

/**
 *  @brief  Type-punned function pointer for FMA operations on dense vector representations.
 *          Implements the `y = α  × a  × b + β  × c` operation.
 *
 *  @param[in] a        Pointer to the first data array.
 *  @param[in] b        Pointer to the second data array.
 *  @param[in] c        Pointer to the third data array.
 *  @param[in] n        Number of scalar words in the input arrays.
 *  @param[in] alpha    Pointer to scaling factor for a × b product (type depends on input precision).
 *  @param[in] beta     Pointer to scaling factor for c array (type depends on input precision).
 *  @param[out] y       Output value in the same precision as the input arrays.
 */
typedef void (*nk_each_fma_punned_t)(void const *a, void const *b, void const *c, nk_size_t n, void const *alpha,
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
 *  The transformation aligns point cloud A to B: a′ᵢ = scale × R × (aᵢ - ā) + b̄
 *
 *  @param[in] a            Pointer to first point cloud (source, interleaved xyz coordinates).
 *  @param[in] b            Pointer to second point cloud (target, interleaved xyz coordinates).
 *  @param[in] n            Number of 3D points in each cloud.
 *  @param[out] a_centroid  Output centroid of first cloud (3 values), or NULL.
 *  @param[out] b_centroid  Output centroid of second cloud (3 values), or NULL.
 *  @param[out] rotation    Output 3 × 3 rotation matrix (9 values, row-major), or NULL.
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
 *  @brief  Type-punned function pointer for GEMM packed buffer size computation.
 */
typedef nk_size_t (*nk_dots_packed_size_punned_t)(nk_size_t n, nk_size_t k);

/**
 *  @brief  Type-punned function pointer for GEMM B matrix packing.
 */
typedef void (*nk_dots_pack_punned_t)(void const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed);

/**
 *  @brief  Type-punned function pointer for GEMM computation.
 */
typedef void (*nk_dots_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t m, nk_size_t n, nk_size_t k,
                                 nk_size_t a_stride, nk_size_t c_stride);

/**
 *  @brief  Type-punned function pointer for symmetric Gram matrix computation (C = A × Aᵀ).
 */
typedef void (*nk_dots_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                           void *result, nk_size_t result_stride, nk_size_t row_start,
                                           nk_size_t row_count);

/**
 *  @brief  Type-punned function pointer for type casting operations.
 */
typedef void (*nk_kernel_cast_punned_t)(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                        nk_dtype_t to_type);

/**
 *  @brief  Type-punned function pointer for a NumKong public interface.
 *
 *  Can be a `nk_metric_dense_punned_t`, `nk_sparse_intersect_punned_t`, `nk_metric_curved_punned_t`,
 *  `nk_metric_mesh_punned_t`, `nk_each_fma_punned_t`, `nk_each_blend_punned_t`,
 *  `nk_each_scale_punned_t`, `nk_each_sum_punned_t`, `nk_kernel_trigonometry_punned_t`,
 *  `nk_kernel_reduce_add_punned_t`, `nk_kernel_reduce_minmax_punned_t`,
 *  `nk_dots_packed_size_punned_t`, `nk_dots_pack_punned_t`, `nk_dots_punned_t`, or
 *  `nk_dots_symmetric_punned_t`.
 */
typedef void (*nk_kernel_punned_t)(void *);

#if NK_DYNAMIC_DISPATCH
NK_DYNAMIC nk_capability_t nk_capabilities(void);
NK_DYNAMIC void nk_find_kernel_punned( //
    nk_kernel_kind_t kind,             //
    nk_dtype_t dtype,                  //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output);
NK_DYNAMIC int nk_configure_thread(nk_capability_t);
#else
NK_PUBLIC nk_capability_t nk_capabilities(void);
NK_PUBLIC void nk_find_kernel_punned(  //
    nk_kernel_kind_t kind,             //
    nk_dtype_t dtype,                  //
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
    // Check for AVX-VNNI (Function ID 7, Sub-leaf 1, EAX bit 4)
    // https://github.com/llvm/llvm-project/blob/ab450597dad2bc8f92376da2c230d9604fc8afc1/clang/lib/Headers/cpuid.h#L208
    unsigned supports_avxvnni = (info7sub1.named.eax & 0x00000010) != 0;

    // Convert specific features into CPU generations
    unsigned supports_haswell = supports_avx2 && supports_f16c && supports_fma;
    unsigned supports_skylake = supports_avx512f;
    unsigned supports_ice = supports_avx512vnni && supports_avx512ifma && supports_avx512bitalg &&
                            supports_avx512vbmi2 && supports_avx512vpopcntdq;
    unsigned supports_genoa = supports_avx512bf16;
    unsigned supports_sapphire = supports_avx512fp16;
    // We don't want to accidentally enable AVX512VP2INTERSECT on Intel Tiger Lake CPUs
    unsigned supports_turin = supports_avx512vp2intersect && supports_avx512bf16;
    unsigned supports_sierra = supports_haswell && supports_avxvnni && !supports_avx512f;
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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.5-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.5-a+sve")
#endif

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
    unsigned supports_neon = 0, supports_fp16 = 0, supports_fhm = 0, supports_bf16 = 0, supports_i8mm = 0;
    unsigned supports_sme = 0, supports_sme2 = 0;
    size_t size = sizeof(supports_neon);
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &supports_fp16, &size, NULL, 0) != 0) supports_fp16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FHM", &supports_fhm, &size, NULL, 0) != 0) supports_fhm = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &supports_bf16, &size, NULL, 0) != 0) supports_bf16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &supports_i8mm, &size, NULL, 0) != 0) supports_i8mm = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &supports_sme, &size, NULL, 0) != 0) supports_sme = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &supports_sme2, &size, NULL, 0) != 0) supports_sme2 = 0;

    return (nk_capability_t)(                                     //
        (nk_cap_neon_k * (supports_neon)) |                       //
        (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |  //
        (nk_cap_neonfhm_k * (supports_neon && supports_fhm)) |    //
        (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) | //
        (nk_cap_neonsdot_k * (supports_neon && supports_i8mm)) |  //
        (nk_cap_sme_k * (supports_sme)) |                         //
        (nk_cap_sme2_k * (supports_sme2)) |                       //
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
    // FHM (FEAT_FHM), bits [51:48] of ID_AA64ISAR0_EL1 - FP16 multiply-accumulate
    unsigned supports_fhm = ((id_aa64isar0_el1 >> 48) & 0xF) >= 1;
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
    unsigned supports_svesdotmm = ((id_aa64zfr0_el1 >> 44) & 0xF) >= 1;
    // BF16, bits [23:20] of ID_AA64ZFR0_EL1
    unsigned supports_svebfdot = ((id_aa64zfr0_el1 >> 20) & 0xF) >= 1;
    // SVEver, bits [3:0] can be used to check for capability levels:
    //
    //  - 0b0000: SVE is implemented
    //  - 0b0001: SVE2 is implemented
    //  - 0b0010: SVE2.1 is implemented
    //
    // This value must match the existing indicator obtained from ID_AA64PFR0_EL1:
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_sve2p1 = ((id_aa64zfr0_el1) & 0xF) >= 2;

    // Now let's unpack the status flags from ID_AA64PFR1_EL1 for SME support
    // https://developer.arm.com/documentation/ddi0601/2024-03/AArch64-Registers/ID-AA64PFR1-EL1
    unsigned long id_aa64pfr1_el1 = 0, id_aa64smfr0_el1 = 0;
    __asm__ __volatile__("mrs %0, ID_AA64PFR1_EL1" : "=r"(id_aa64pfr1_el1));
    // SME, bits [27:24] of ID_AA64PFR1_EL1
    unsigned supports_sme = ((id_aa64pfr1_el1 >> 24) & 0xF) >= 1;

    // SME feature flags from ID_AA64SMFR0_EL1
    unsigned supports_sme2 = 0, supports_sme2p1 = 0;
    unsigned supports_smef64 = 0, supports_smehalf = 0, supports_smebf16 = 0;
    unsigned supports_smelut2 = 0, supports_smefa64 = 0;
    if (supports_sme) {
        __asm__ __volatile__("mrs %0, ID_AA64SMFR0_EL1" : "=r"(id_aa64smfr0_el1));
        // SMEver, bits [59:56] of ID_AA64SMFR0_EL1
        unsigned sme_version = (id_aa64smfr0_el1 >> 56) & 0xF;
        supports_sme2 = sme_version >= 1;
        supports_sme2p1 = sme_version >= 2;
        // F64F64, bit [48] of ID_AA64SMFR0_EL1
        supports_smef64 = (id_aa64smfr0_el1 >> 48) & 0x1;
        // F16F16, bit [42] of ID_AA64SMFR0_EL1
        supports_smehalf = (id_aa64smfr0_el1 >> 42) & 0x1;
        // B16B16, bit [44] of ID_AA64SMFR0_EL1
        supports_smebf16 = (id_aa64smfr0_el1 >> 44) & 0x1;
        // LUTv2, bit [56] - already masked with sme_version, check bit [56-56] separately
        // Actually LUTv2 is at different position, let's use bit [56] for SME2 LUT
        // FA64, bit [63] of ID_AA64SMFR0_EL1
        supports_smefa64 = (id_aa64smfr0_el1 >> 63) & 0x1;
    }

    return (nk_capability_t)(                                                                     //
        (nk_cap_neon_k * (supports_neon)) |                                                       //
        (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |                                  //
        (nk_cap_neonfhm_k * (supports_neon && supports_fhm)) |                                    //
        (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |                                 //
        (nk_cap_neonsdot_k * (supports_neon && supports_i8mm && supports_integer_dot_products)) | //
        (nk_cap_sve_k * (supports_sve)) |                                                         //
        (nk_cap_svehalf_k * (supports_sve && supports_fp16)) |                                    //
        (nk_cap_svebfdot_k * (supports_sve && supports_svebfdot)) |                               //
        (nk_cap_svesdot_k * (supports_sve && supports_svesdotmm)) |                               //
        (nk_cap_sve2_k * (supports_sve2)) |                                                       //
        (nk_cap_sve2p1_k * (supports_sve2p1)) |                                                   //
        (nk_cap_sme_k * (supports_sme)) |                                                         //
        (nk_cap_sme2_k * (supports_sme2)) |                                                       //
        (nk_cap_sme2p1_k * (supports_sme2p1)) |                                                   //
        (nk_cap_smef64_k * (supports_smef64)) |                                                   //
        (nk_cap_smehalf_k * (supports_smehalf)) |                                                 //
        (nk_cap_smebf16_k * (supports_smebf16)) |                                                 //
        (nk_cap_smefa64_k * (supports_smefa64)) |                                                 //
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
    return (nk_capability_t)(                                  //
        (nk_cap_neon_k * (supports_neon)) |                    //
        (nk_cap_neonsdot_k * (supports_neon && supports_dp)) | //
        (nk_cap_serial_k));

#else // Unknown platform

    // Conservative fallback for unknown platforms: NEON is mandatory in ARMv8-A (ARM64)
    return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);

#endif
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_ARM_

#if NK_TARGET_RISCV_

/**
 *  @brief  Function to determine the SIMD capabilities of the current 64-bit RISC-V machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `nk_capability_t` enum value.
 */
NK_PUBLIC nk_capability_t nk_capabilities_riscv_(void) {
#if defined(NK_DEFINED_LINUX_)
    // Base V extension from AT_HWCAP (bit 21 = 'V' - 'A')
    unsigned long hwcap = getauxval(AT_HWCAP);
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & (1UL << 21)) {
        caps |= nk_cap_spacemit_k;
        // hwprobe() syscall for Zvfh/Zvfbfwma (Linux 6.4+)
        // syscall 258, key 4 = IMA_EXT_0, bit 30 = ZVFH, bit 54 = ZVFBFWMA
        struct {
            long key;
            unsigned long value;
        } pairs[1] = {{4, 0}};
        if (syscall(258, pairs, 1, 0, (void *)0, 0) == 0) {
            if (pairs[0].value & (1ULL << 30)) caps |= nk_cap_sifive_k;
            if (pairs[0].value & (1ULL << 54)) caps |= nk_cap_xuantie_k;
        }
    }
    return caps;
#else
    return nk_cap_serial_k;
#endif
}

#endif // NK_TARGET_RISCV_

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
 *  @brief  Function to determine the SIMD capabilities of the current machine at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `nk_capability_t` enum value.
 */
NK_PUBLIC nk_capability_t nk_capabilities_(void) {
#if NK_TARGET_X86_
    return nk_capabilities_x86_();
#endif // NK_TARGET_X86_
#if NK_TARGET_ARM_
    return nk_capabilities_arm_();
#endif // NK_TARGET_ARM_
#if NK_TARGET_RISCV_
    return nk_capabilities_riscv_();
#endif // NK_TARGET_RISCV_
    return nk_cap_serial_k;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type"

#ifdef __cplusplus //! option "-Wvolatile" is valid for C++/ObjC++ but not for C/Clang
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wvolatile"
#endif
#endif

NK_INTERNAL void nk_find_kernel_punned_f64_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SMEF64
    if (v & nk_cap_smef64_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_smef64, *c = nk_cap_smef64_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f32_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SMEF64
    if (v & nk_cap_smef64_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_smef64, *c = nk_cap_smef64_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sin_k: *m = (m_t)&nk_sin_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_cos_k: *m = (m_t)&nk_cos_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_atan_k: *m = (m_t)&nk_atan_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f16_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIFIVE
    if (v & nk_cap_sifive_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_sifive, *c = nk_cap_sifive_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_sifive, *c = nk_cap_sifive_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_sifive, *c = nk_cap_sifive_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_sifive, *c = nk_cap_sifive_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVEHALF
    if (v & nk_cap_svehalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_svehalf, *c = nk_cap_svehalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_bf16_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_XUANTIE
    if (v & nk_cap_xuantie_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_xuantie, *c = nk_cap_xuantie_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_xuantie, *c = nk_cap_xuantie_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_xuantie, *c = nk_cap_xuantie_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_xuantie, *c = nk_cap_xuantie_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE_AMX
    if (v & nk_cap_sapphire_amx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_bf16_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVEBFDOT
    if (v & nk_cap_svebfdot_k) switch (k) {
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_neonbfdot, *c = nk_cap_neonbfdot_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i8_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE_AMX
    if (v & nk_cap_sapphire_amx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_i8_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_i8_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i8_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIERRA
    if (v & nk_cap_sierra_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_sierra, *c = nk_cap_sierra_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i8_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}
NK_INTERNAL void nk_find_kernel_punned_u8_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u8_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE_AMX
    if (v & nk_cap_sapphire_amx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_u8_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u8_neonhalf, *c = nk_cap_neonhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u8_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE //! Scaling of 8-bit integers is performed using 16-bit floats.
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u8_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u8_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIERRA
    if (v & nk_cap_sierra_k) switch (k) {
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_sierra, *c = nk_cap_sierra_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIFIVE
    if (v & nk_cap_sifive_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_spacemit, *c = nk_cap_sifive_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u8_spacemit, *c = nk_cap_sifive_k; return;
        default: break;
        }
#endif
#if NK_TARGET_XUANTIE
    if (v & nk_cap_xuantie_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_spacemit, *c = nk_cap_xuantie_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u8_spacemit, *c = nk_cap_xuantie_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i4_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u4_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u4_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e4m3_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE_AMX
    if (v & nk_cap_sapphire_amx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e4m3_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e5m2_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE_AMX
    if (v & nk_cap_sapphire_amx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e5m2_sapphire_amx, *c = nk_cap_sapphire_amx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_sapphire_amx, *c = nk_cap_sapphire_amx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e2m3_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_e3m2_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u1_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                           nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIFIVE
    if (v & nk_cap_sifive_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_spacemit, *c = nk_cap_sifive_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_spacemit, *c = nk_cap_sifive_k; return;
        default: break;
        }
#endif
#if NK_TARGET_XUANTIE
    if (v & nk_cap_xuantie_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_spacemit, *c = nk_cap_xuantie_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_spacemit, *c = nk_cap_xuantie_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_spacemit, *c = nk_cap_spacemit_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f64c_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f32c_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_f16c_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                             nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVEHALF
    if (v & nk_cap_svehalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_svehalf, *c = nk_cap_svehalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16c_neonhalf, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16c_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_bf16c_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                              nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u16_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u16_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u16_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u16_ice, *c = nk_cap_skylake_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u16_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIFIVE
    if (v & nk_cap_sifive_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u16_spacemit, *c = nk_cap_sifive_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_spacemit, *c = nk_cap_sifive_k; return;
        default: break;
        }
#endif
#if NK_TARGET_XUANTIE
    if (v & nk_cap_xuantie_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u16_spacemit, *c = nk_cap_xuantie_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_spacemit, *c = nk_cap_xuantie_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i16_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i16_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u32_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_turin, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_ice, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIFIVE
    if (v & nk_cap_sifive_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u32_spacemit, *c = nk_cap_sifive_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_spacemit, *c = nk_cap_sifive_k; return;
        default: break;
        }
#endif
#if NK_TARGET_XUANTIE
    if (v & nk_cap_xuantie_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u32_spacemit, *c = nk_cap_xuantie_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_spacemit, *c = nk_cap_xuantie_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SPACEMIT
    if (v & nk_cap_spacemit_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_spacemit, *c = nk_cap_spacemit_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i32_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i32_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_i64_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i64_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i64_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_i64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_i64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_u64_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                            nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_ice, *c = nk_cap_ice_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

NK_INTERNAL void nk_find_kernel_punned_unkown_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m,
                                               nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICE
    if (v & nk_cap_ice_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_ice, *c = nk_cap_ice_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_serial, *c = nk_cap_serial_k; return;
        default: break;
        }
}

/**
 *  @brief  Determines the best suited metric implementation based on the given dtype,
 *          supported and allowed by hardware capabilities.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param dtype The data type for which the metric needs to be evaluated.
 *  @param supported The hardware capabilities supported by the CPU.
 *  @param allowed The hardware capabilities allowed for use.
 *  @param kernel_output Output variable for the selected similarity function.
 *  @param capability_output Output variable for the utilized hardware capabilities.
 */
NK_INTERNAL void nk_find_kernel_punned_( //
    nk_kernel_kind_t kind,               //
    nk_dtype_t dtype,                    //
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

    switch (dtype) {

    case nk_f64_k: nk_find_kernel_punned_f64_(viable, kind, m, c); return;
    case nk_f32_k: nk_find_kernel_punned_f32_(viable, kind, m, c); return;
    case nk_f16_k: nk_find_kernel_punned_f16_(viable, kind, m, c); return;
    case nk_bf16_k: nk_find_kernel_punned_bf16_(viable, kind, m, c); return;
    case nk_e4m3_k: nk_find_kernel_punned_e4m3_(viable, kind, m, c); return;
    case nk_e5m2_k: nk_find_kernel_punned_e5m2_(viable, kind, m, c); return;
    case nk_e2m3_k: nk_find_kernel_punned_e2m3_(viable, kind, m, c); return;
    case nk_e3m2_k: nk_find_kernel_punned_e3m2_(viable, kind, m, c); return;

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

    case nk_u1_k: nk_find_kernel_punned_u1_(viable, kind, m, c); return;
    case nk_i4_k: nk_find_kernel_punned_i4_(viable, kind, m, c); return;
    case nk_u4_k: nk_find_kernel_punned_u4_(viable, kind, m, c); return;

    case nk_dtype_unknown_k: nk_find_kernel_punned_unkown_(viable, kind, m, c); break;
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
 *  @brief  Selects the most suitable metric implementation based on the given metric kind, dtype,
 *          and allowed capabilities. @b Don't call too often and prefer caching the `nk_capabilities()`.
 *
 *  @param kind The kind of metric to be evaluated.
 *  @param dtype The data type for which the metric needs to be evaluated.
 *  @param allowed The hardware capabilities allowed for use.
 *  @return A function pointer to the selected metric implementation.
 */
NK_PUBLIC nk_kernel_punned_t nk_metric_punned( //
    nk_kernel_kind_t kind,                     //
    nk_dtype_t dtype,                          //
    nk_capability_t allowed) {

    nk_kernel_punned_t result = 0;
    nk_capability_t c = nk_cap_serial_k;
    nk_capability_t supported = nk_capabilities();
    nk_find_kernel_punned(kind, dtype, supported, allowed, &result, &c);
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
NK_DYNAMIC int nk_uses_neonhalf(void);
NK_DYNAMIC int nk_uses_neonbfdot(void);
NK_DYNAMIC int nk_uses_neonsdot(void);
NK_DYNAMIC int nk_uses_sve(void);
NK_DYNAMIC int nk_uses_svehalf(void);
NK_DYNAMIC int nk_uses_svebfdot(void);
NK_DYNAMIC int nk_uses_svesdot(void);
NK_DYNAMIC int nk_uses_sve2(void);
NK_DYNAMIC int nk_uses_sve2p1(void);
NK_DYNAMIC int nk_uses_neonfhm(void);
NK_DYNAMIC int nk_uses_sme(void);
NK_DYNAMIC int nk_uses_sme2(void);
NK_DYNAMIC int nk_uses_sme2p1(void);
NK_DYNAMIC int nk_uses_smef64(void);
NK_DYNAMIC int nk_uses_smehalf(void);
NK_DYNAMIC int nk_uses_smebf16(void);
NK_DYNAMIC int nk_uses_smelut2(void);
NK_DYNAMIC int nk_uses_smefa64(void);
NK_DYNAMIC int nk_uses_smebi32(void);
NK_DYNAMIC int nk_uses_haswell(void);
NK_DYNAMIC int nk_uses_skylake(void);
NK_DYNAMIC int nk_uses_ice(void);
NK_DYNAMIC int nk_uses_genoa(void);
NK_DYNAMIC int nk_uses_sapphire(void);
NK_DYNAMIC int nk_uses_turin(void);
NK_DYNAMIC int nk_uses_sierra(void);
NK_DYNAMIC int nk_uses_sapphire_amx(void);
NK_DYNAMIC int nk_uses_granite_amx(void);

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
NK_PUBLIC int nk_uses_neonhalf(void) { return NK_TARGET_ARM_ && NK_TARGET_NEONHALF; }
NK_PUBLIC int nk_uses_neonbfdot(void) { return NK_TARGET_ARM_ && NK_TARGET_NEONBFDOT; }
NK_PUBLIC int nk_uses_neonsdot(void) { return NK_TARGET_ARM_ && NK_TARGET_NEONSDOT; }
NK_PUBLIC int nk_uses_sve(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE; }
NK_PUBLIC int nk_uses_svehalf(void) { return NK_TARGET_ARM_ && NK_TARGET_SVEHALF; }
NK_PUBLIC int nk_uses_svebfdot(void) { return NK_TARGET_ARM_ && NK_TARGET_SVEBFDOT; }
NK_PUBLIC int nk_uses_svesdot(void) { return NK_TARGET_ARM_ && NK_TARGET_SVESDOT; }
NK_PUBLIC int nk_uses_sve2(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE2; }
NK_PUBLIC int nk_uses_sve2p1(void) { return NK_TARGET_ARM_ && NK_TARGET_SVE2P1; }
NK_PUBLIC int nk_uses_neonfhm(void) { return NK_TARGET_ARM_ && NK_TARGET_NEONFHM; }
NK_PUBLIC int nk_uses_sme(void) { return NK_TARGET_ARM_ && NK_TARGET_SME; }
NK_PUBLIC int nk_uses_sme2(void) { return NK_TARGET_ARM_ && NK_TARGET_SME2; }
NK_PUBLIC int nk_uses_sme2p1(void) { return NK_TARGET_ARM_ && NK_TARGET_SME2P1; }
NK_PUBLIC int nk_uses_smef64(void) { return NK_TARGET_ARM_ && NK_TARGET_SMEF64; }
NK_PUBLIC int nk_uses_smehalf(void) { return NK_TARGET_ARM_ && NK_TARGET_SMEHALF; }
NK_PUBLIC int nk_uses_smebf16(void) { return NK_TARGET_ARM_ && NK_TARGET_SMEBF16; }
NK_PUBLIC int nk_uses_smelut2(void) { return NK_TARGET_ARM_ && NK_TARGET_SMELUT2; }
NK_PUBLIC int nk_uses_smefa64(void) { return NK_TARGET_ARM_ && NK_TARGET_SMEFA64; }
NK_PUBLIC int nk_uses_smebi32(void) { return NK_TARGET_ARM_ && NK_TARGET_SMEBI32; }
NK_PUBLIC int nk_uses_haswell(void) { return NK_TARGET_X86_ && NK_TARGET_HASWELL; }
NK_PUBLIC int nk_uses_skylake(void) { return NK_TARGET_X86_ && NK_TARGET_SKYLAKE; }
NK_PUBLIC int nk_uses_ice(void) { return NK_TARGET_X86_ && NK_TARGET_ICE; }
NK_PUBLIC int nk_uses_genoa(void) { return NK_TARGET_X86_ && NK_TARGET_GENOA; }
NK_PUBLIC int nk_uses_sapphire(void) { return NK_TARGET_X86_ && NK_TARGET_SAPPHIRE; }
NK_PUBLIC int nk_uses_turin(void) { return NK_TARGET_X86_ && NK_TARGET_TURIN; }
NK_PUBLIC int nk_uses_sierra(void) { return NK_TARGET_X86_ && NK_TARGET_SIERRA; }
NK_PUBLIC int nk_uses_sapphire_amx(void) { return NK_TARGET_X86_ && NK_TARGET_SAPPHIRE_AMX; }
NK_PUBLIC int nk_uses_granite_amx(void) { return NK_TARGET_X86_ && NK_TARGET_GRANITE_AMX; }
NK_PUBLIC int nk_uses_spacemit(void) { return NK_TARGET_RISCV_ && NK_TARGET_SPACEMIT; }
NK_PUBLIC int nk_uses_sifive(void) { return NK_TARGET_RISCV_ && NK_TARGET_SIFIVE; }
NK_PUBLIC int nk_uses_xuantie(void) { return NK_TARGET_RISCV_ && NK_TARGET_XUANTIE; }
NK_PUBLIC int nk_uses_dynamic_dispatch(void) { return 0; }
NK_PUBLIC int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }
NK_PUBLIC nk_capability_t nk_capabilities(void) { return nk_capabilities_(); }
NK_PUBLIC void nk_find_kernel_punned(  //
    nk_kernel_kind_t kind,             //
    nk_dtype_t dtype,                  //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output) {
    nk_find_kernel_punned_(kind, dtype, supported, allowed, kernel_output, capability_output);
}

#endif

#ifdef __cplusplus
}
#endif

#endif // NK_NUMKONG_H
