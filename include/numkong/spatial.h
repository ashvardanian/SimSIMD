/**
 *  @brief SIMD-accelerated Spatial Similarity Measures.
 *  @file include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Contains following similarity measures:
 *
 *  - L2 (Euclidean) regular and squared distance
 *  - Cosine (Angular) distance - @b not similarity!
 *
 *  For dtypes:
 *
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integral numbers
 *  - 8-bit signed integral numbers
 *  - 4-bit signed integral numbers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire
 *
 *  @section streaming_api Streaming API
 *
 *  Angular and L2 distances can be computed from a single dot-product stream and precomputed magnitudes.
 *  The streaming helpers operate on 512-bit blocks (`nk_b512_vec_t`) and only accumulate $A*B$.
 *  Finalization takes the magnitudes of the full vectors (L2 norms) and computes the distance.
 *  Let the following be computed over the full vectors:
 *
 *      ab   = Σᵢ (aᵢ · bᵢ)
 *      ‖a‖ = √(Σᵢ aᵢ²)
 *      ‖b‖ = √(Σᵢ bᵢ²)
 *
 *  Finalization formulas:
 *
 *      angular(a, b) = 1 − ab / (‖a‖ · ‖b‖)
 *      l2(a, b)      = √( ‖a‖² + ‖b‖² − 2·ab )
 *
 *  The angular distance is clamped to ≥ 0, with a 0 result when both norms are zero and a 1 result when $ab$ is zero.
 *  L2 clamps the argument of the square root at 0 to avoid negative values from rounding.
 *
 *  @code{.c}
 *  nk_b512_vec_t a_block, b_block;
 *  nk_f32_t a_norm = ..., b_norm = ...; // Precomputed L2 norms of full vectors
 *  nk_angular_f32x8_state_haswell_t state; // Often equivalent to dot-product state
 *  nk_angular_f32x8_init_haswell(&state);
 *  nk_angular_f32x8_update_haswell(&state, a_block, b_block);
 *  nk_angular_f32x8_finalize_haswell(&state, a_norm, b_norm, &distance);
 *  @endcode
 *
 *  @section rsqrt_notes Reciprocal Square Root and Newton-Raphson Notes
 *
 *  Angular distance normalization uses reciprocal square roots to avoid the
 *  latency of full sqrt/div pipelines. We refine the rsqrt estimate with one
 *  (x86) or two (Arm NEON) Newton-Raphson iterations to reduce error.
 *
 *  Relevant instructions and caveats:
 *
 *      Intrinsic                   Instruction        Notes
 *      _mm_rsqrt_ps                VRSQRTPS           fast approx; refine with NR
 *      _mm_maskz_rsqrt14_pd        VRSQRT14PD         higher-precision approx; MSVC masked-only
 *      _mm_sqrt_ps/_mm_sqrt_pd     VSQRTPS/VSQRTPD    higher latency, sqrt/div unit
 *
 *  Latency/port notes (rule of thumb):
 *  - On Intel client cores, sqrt/rsqrt execute on the divide/sqrt unit (often
 *    port 0) and can bottleneck tight loops.
 *  - NR refinement uses mul/FMA ports and amortizes well when `ab` is reduced
 *    to a scalar and reused for finalization.
 *  - Arm NEON `rsqrt` accuracy is coarse; we apply two refinement steps to keep
 *    angular distance error bounded.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  AVX2 lacks signed 8-bit dot products, so Haswell widens to i16 and uses VPMADDWD.
 *  AVX-512 VNNI replaces that with VPDPWSSD. BF16 uses VDPBF16PS where available to avoid
 *  convert+FMA sequences; if the ISA lacks it, we fall back to f32 FMA in the AVX2/serial:
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_madd_epi16       VPMADDWD (YMM, YMM, YMM)        5c @ p01    3c @ p01
 *      _mm512_dpwssd_epi32     VPDPWSSD (ZMM, K, ZMM, ZMM)     5c @ p05    4c @ p01
 *      _mm512_dpbf16_ps        VDPBF16PS (ZMM, K, ZMM, ZMM)    n/a         6c @ p01
 *      _mm_rsqrt_ps            VRSQRTPS (XMM, XMM)             5c @ p0     4c @ p01
 *      _mm_maskz_rsqrt14_pd    VRSQRT14PD (XMM, K, XMM)        4c @ p0     5c @ p01
 *      _mm_sqrt_ps             VSQRTPS (XMM, XMM)              12c @ p0    15c @ p01
 *
 *  @section arm_instructions Relevant Arm Instructions
 *
 *  The NEON/SVE kernels in this header are structured around FMLA/SDOT/BFDOT loops,
 *  which is why we avoid mul+add splits and keep reductions to scalars before square roots.
 *  Dot-product kernels for i8/u8 are only built when the "dotprod+i8mm" target is enabled;
 *  otherwise we rely on the serial backends. BF16 kernels are enabled only with BF16 dot
 *  instructions skipping `vbfmlal` and `vbfmlalt` alternatives to limit shuffle overhead
 *  and code complexity.
 *
 *      Intrinsic               Instruction         M1 Firestorm
 *      vfmaq_f32               FMLA.S (vec)        4c / 4c
 *      vfmaq_f64               FMLA.D (vec)        4c / 4c
 *      vdotq_s32               SDOT.B (vec)        3c / 4c
 *      vbfdotq_f32             BFDOT (vec)         n/a
 *      vrsqrteq_f32            FRSQRTE.S (vec)     3c / 1c
 *      vrsqrtsq_f32            FRSQRTS.S (vec)     4c / 4c
 *      vsqrtq_f32              FSQRT.S (vec)       10c / 0.5c
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_SPATIAL_H
#define NK_SPATIAL_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief L2 (Euclidean) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of elements in each vector.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_l2_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_DYNAMIC void nk_l2_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Squared L2 (Euclidean) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of elements in each vector.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_l2sq_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_DYNAMIC void nk_l2sq_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Angular (cosine) distance between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of elements in each vector.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_angular_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_DYNAMIC void nk_angular_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 */
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
#if NK_TARGET_NEON
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONSDOT

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
#if NK_TARGET_SVE
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
#endif // NK_TARGET_SVE

#if NK_TARGET_SVEHALF
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SVEHALF

#if NK_TARGET_SVEBFDOT
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SVEBFDOT

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
#endif // NK_TARGET_HASWELL

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer, using 32-bit arithmetic over 512-bit words.
 *  Skylake was launched in 2015, and discontinued in 2019. Skylake had support for F, CD, VL, DQ, and BW extensions,
 *  as well as masked operations. This is enough to supersede auto-vectorization on `f32` and `f64` types.
 *
 *  Sadly, we can't effectively interleave different kinds of arithmetic instructions to utilize more ports:
 *
 *  > Like Intel server architectures since Skylake-X, SPR cores feature two 512-bit FMA units, and organize them in a
 *    similar fashion. > One 512-bit FMA unit is created by fusing two 256-bit ones on port 0 and port 1. The other is
 *    added to port 5, as a server-specific > core extension. The FMA units on port 0 and 1 are configured into
 *    2×256-bit or 1×512-bit mode depending on whether 512-bit FMA > instructions are present in the scheduler. That
 *    means a mix of 256-bit and 512-bit FMA instructions will not achieve higher IPC > than executing 512-bit
 *    instructions alone.
 *
 *  Source: https://chipsandcheese.com/p/a-peek-at-sapphire-rapids
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX512 CPUs of Ice Lake generation and newer, using mixed arithmetic over 512-bit words.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral
 *  operations. Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA
 *  instructions.
 */
#if NK_TARGET_ICE
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_ICE

#if NK_TARGET_GENOA
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e4m3_genoa(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e4m3_genoa(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e4m3_genoa(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2_f64 */
NK_PUBLIC void nk_l2_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_f64 */
NK_PUBLIC void nk_l2sq_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_l2_e4m3 */
NK_PUBLIC void nk_l2_e4m3_sapphire(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_l2sq_e4m3 */
NK_PUBLIC void nk_l2sq_e4m3_sapphire(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SAPPHIRE

/*  SIMD-powered backends for AVX-INT8-VNNI extensions on Xeon 6 CPUs, including Sierra Forest and Granite Rapids.
 *  The packs many "efficiency" cores into a single socket, avoiding heavy 512-bit operations, and focusing on
 *  256-bit ones.
 */
#if NK_TARGET_SIERRA
/** @copydoc nk_angular_f64 */
NK_PUBLIC void nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SIERRA

/**
 *  @brief  Returns the output dtype for L2 (Euclidean) distance.
 */
NK_INTERNAL nk_dtype_t nk_l2_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k;
    case nk_u8_k: return nk_f32_k;
    case nk_i4_k: return nk_f32_k;
    case nk_u4_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for L2 squared distance.
 */
NK_INTERNAL nk_dtype_t nk_l2sq_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_i8_k: return nk_u32_k;
    case nk_u8_k: return nk_u32_k;
    case nk_i4_k: return nk_u32_k;
    case nk_u4_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for angular/cosine distance.
 */
NK_INTERNAL nk_dtype_t nk_angular_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k;
    case nk_u8_k: return nk_f32_k;
    case nk_i4_k: return nk_f32_k;
    case nk_u4_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#include "numkong/spatial/serial.h"
#include "numkong/spatial/neon.h"
#include "numkong/spatial/neonhalf.h"
#include "numkong/spatial/neonbfdot.h"
#include "numkong/spatial/neonsdot.h"
#include "numkong/spatial/sve.h"
#include "numkong/spatial/svehalf.h"
#include "numkong/spatial/svebfdot.h"
#include "numkong/spatial/haswell.h"
#include "numkong/spatial/skylake.h"
#include "numkong/spatial/genoa.h"
#include "numkong/spatial/sapphire.h"
#include "numkong/spatial/ice.h"
#include "numkong/spatial/sierra.h"
#include "numkong/spatial/spacemit.h"
#include "numkong/spatial/sifive.h"
#include "numkong/spatial/xuantie.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_l2_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2_f64_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_l2_f64_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_l2_f64_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_f64_haswell(a, b, n, result);
#else
    nk_l2_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2sq_f64_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_l2sq_f64_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_l2sq_f64_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2sq_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_f64_haswell(a, b, n, result);
#else
    nk_l2sq_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SPACEMIT
    nk_angular_f64_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_angular_f64_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_angular_f64_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_angular_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_f64_haswell(a, b, n, result);
#else
    nk_angular_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2_f32_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_l2_f32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_l2_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2_f32_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_f32_haswell(a, b, n, result);
#else
    nk_l2_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2sq_f32_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_l2sq_f32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_l2sq_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2sq_f32_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_f32_haswell(a, b, n, result);
#else
    nk_l2sq_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_angular_f32_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_angular_f32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_angular_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_angular_f32_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_f32_haswell(a, b, n, result);
#else
    nk_angular_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SIFIVE
    nk_l2_f16_sifive(a, b, n, result);
#elif NK_TARGET_SVEHALF
    nk_l2_f16_svehalf(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_l2_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_f16_haswell(a, b, n, result);
#else
    nk_l2_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SIFIVE
    nk_l2sq_f16_sifive(a, b, n, result);
#elif NK_TARGET_SVEHALF
    nk_l2sq_f16_svehalf(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_l2sq_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_f16_haswell(a, b, n, result);
#else
    nk_l2sq_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SIFIVE
    nk_angular_f16_sifive(a, b, n, result);
#elif NK_TARGET_SVEHALF
    nk_angular_f16_svehalf(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_angular_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_f16_haswell(a, b, n, result);
#else
    nk_angular_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_XUANTIE
    nk_l2_bf16_xuantie(a, b, n, result);
#elif NK_TARGET_SVEBFDOT
    nk_l2_bf16_svebfdot(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_l2_bf16_neonbfdot(a, b, n, result);
#elif NK_TARGET_GENOA
    nk_l2_bf16_genoa(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_bf16_haswell(a, b, n, result);
#else
    nk_l2_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_XUANTIE
    nk_l2sq_bf16_xuantie(a, b, n, result);
#elif NK_TARGET_SVEBFDOT
    nk_l2sq_bf16_svebfdot(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_l2sq_bf16_neonbfdot(a, b, n, result);
#elif NK_TARGET_GENOA
    nk_l2sq_bf16_genoa(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_bf16_haswell(a, b, n, result);
#else
    nk_l2sq_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_XUANTIE
    nk_angular_bf16_xuantie(a, b, n, result);
#elif NK_TARGET_SVEBFDOT
    nk_angular_bf16_svebfdot(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_angular_bf16_neonbfdot(a, b, n, result);
#elif NK_TARGET_GENOA
    nk_angular_bf16_genoa(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_bf16_haswell(a, b, n, result);
#else
    nk_angular_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_l2_e4m3_genoa(a, b, n, result);
#elif NK_TARGET_SAPPHIRE
    nk_l2_e4m3_sapphire(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2_e4m3_skylake(a, b, n, result);
#else
    nk_l2_e4m3_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_l2sq_e4m3_genoa(a, b, n, result);
#elif NK_TARGET_SAPPHIRE
    nk_l2sq_e4m3_sapphire(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2sq_e4m3_skylake(a, b, n, result);
#else
    nk_l2sq_e4m3_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_angular_e4m3_genoa(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_angular_e4m3_skylake(a, b, n, result);
#else
    nk_angular_e4m3_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_l2_e5m2_genoa(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2_e5m2_skylake(a, b, n, result);
#else
    nk_l2_e5m2_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_l2sq_e5m2_genoa(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_l2sq_e5m2_skylake(a, b, n, result);
#else
    nk_l2sq_e5m2_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_angular_e5m2_genoa(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_angular_e5m2_skylake(a, b, n, result);
#else
    nk_angular_e5m2_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2_i8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_l2_i8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_l2_i8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_i8_haswell(a, b, n, result);
#else
    nk_l2_i8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2sq_i8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_l2sq_i8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_l2sq_i8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_i8_haswell(a, b, n, result);
#else
    nk_l2sq_i8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_angular_i8_spacemit(a, b, n, result);
#elif NK_TARGET_SIERRA
    nk_angular_i8_sierra(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_angular_i8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_angular_i8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_i8_haswell(a, b, n, result);
#else
    nk_angular_i8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2_u8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_l2_u8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_l2_u8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2_u8_haswell(a, b, n, result);
#else
    nk_l2_u8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_l2sq_u8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_l2sq_u8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_l2sq_u8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_l2sq_u8_haswell(a, b, n, result);
#else
    nk_l2sq_u8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_angular_u8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_angular_u8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_angular_u8_ice(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_angular_u8_haswell(a, b, n, result);
#else
    nk_angular_u8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICE
    nk_l2_i4_ice(a, b, n, result);
#else
    nk_l2_i4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_ICE
    nk_l2sq_i4_ice(a, b, n, result);
#else
    nk_l2sq_i4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICE
    nk_angular_i4_ice(a, b, n, result);
#else
    nk_angular_i4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICE
    nk_l2_u4_ice(a, b, n, result);
#else
    nk_l2_u4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_l2sq_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_ICE
    nk_l2sq_u4_ice(a, b, n, result);
#else
    nk_l2sq_u4_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_angular_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_ICE
    nk_angular_u4_ice(a, b, n, result);
#else
    nk_angular_u4_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
