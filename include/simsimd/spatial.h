/**
 *  @brief SIMD-accelerated Spatial Similarity Measures.
 *  @file include/simsimd/spatial.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Contains following similarity measures:
 *
 *  - L2 (Euclidean) regular and squared distance
 *  - Cosine (Angular) distance - @b not similarity!
 *
 *  For datatypes:
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
 *  The streaming helpers operate on 512-bit blocks (`simsimd_b512_vec_t`) and only accumulate $A*B$.
 *  Finalization takes the magnitudes of the full vectors (L2 norms) and computes the distance.
 *  Let the following be computed over the full vectors:
 *
 *      ab   = Σ_i (a_i · b_i)
 *      ||a|| = √(Σ_i a_i²)
 *      ||b|| = √(Σ_i b_i²)
 *
 *  Finalization formulas:
 *
 *      angular(a, b) = 1 − ab / (||a|| · ||b||)
 *      l2(a, b)      = √( ||a||² + ||b||² − 2·ab )
 *
 *  The angular distance is clamped to ≥ 0, with a 0 result when both norms are zero and a 1 result when $ab$ is zero.
 *  L2 clamps the argument of the square root at 0 to avoid negative values from rounding.
 *
 *  @code{.c}
 *  simsimd_b512_vec_t a_block, b_block;
 *  simsimd_distance_t a_norm = ..., b_norm = ...; // Precomputed L2 norms of full vectors
 *  simsimd_angular_f32x8_state_haswell_t state; // Often equivalent to dot-product state
 *  simsimd_angular_f32x8_init_haswell(&state);
 *  simsimd_angular_f32x8_update_haswell(&state, a_block, b_block);
 *  simsimd_angular_f32x8_finalize_haswell(&state, a_norm, b_norm, &distance);
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
#ifndef SIMSIMD_SPATIAL_H
#define SIMSIMD_SPATIAL_H

#include "types.h"

#include "reduce.h" // For horizontal reduction helpers

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
SIMSIMD_DYNAMIC void simsimd_l2_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                    simsimd_f64_t *result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_DYNAMIC void simsimd_l2_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                    simsimd_f32_t *result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_DYNAMIC void simsimd_l2_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                    simsimd_f32_t *result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_DYNAMIC void simsimd_l2_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_DYNAMIC void simsimd_l2_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                   simsimd_f32_t *result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_DYNAMIC void simsimd_l2_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                   simsimd_f32_t *result);

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
SIMSIMD_DYNAMIC void simsimd_l2sq_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                      simsimd_f64_t *result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_DYNAMIC void simsimd_l2sq_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_DYNAMIC void simsimd_l2sq_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_DYNAMIC void simsimd_l2sq_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_DYNAMIC void simsimd_l2sq_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                     simsimd_u32_t *result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_DYNAMIC void simsimd_l2sq_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                     simsimd_u32_t *result);

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
SIMSIMD_DYNAMIC void simsimd_angular_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                         simsimd_f64_t *result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_DYNAMIC void simsimd_angular_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_DYNAMIC void simsimd_angular_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_DYNAMIC void simsimd_angular_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_DYNAMIC void simsimd_angular_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_DYNAMIC void simsimd_angular_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result);

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                          simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                            simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                               simsimd_f64_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                               simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                               simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                                simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                           simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                           simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);


/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                            simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                              simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                                 simsimd_f64_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                            simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                              simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                                 simsimd_f64_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                             simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                               simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                                  simsimd_f64_t* result);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                        simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                          simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f64_neon(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                             simsimd_f64_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                        simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                        simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                       simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                       simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
#endif // SIMSIMD_TARGET_NEON

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
#if SIMSIMD_TARGET_SVE
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                       simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                       simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                         simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                        simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_sve(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                       simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                         simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                            simsimd_f64_t* result);
#endif // SIMSIMD_TARGET_SVE

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                            simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                               simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                            simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                               simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                                simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                                 simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                                simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                           simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                             simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f64_haswell(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                                simsimd_f64_t* result);
#endif // SIMSIMD_TARGET_HASWELL

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer, using 32-bit arithmetic over 512-bit words.
 *  Skylake was launched in 2015, and discontinued in 2019. Skylake had support for F, CD, VL, DQ, and BW extensions,
 *  as well as masked operations. This is enough to supersede auto-vectorization on `f32` and `f64` types.
 *
 *  Sadly, we can't effectively interleave different kinds of arithmetic instructions to utilize more ports:
 *
 *  > Like Intel server architectures since Skylake-X, SPR cores feature two 512-bit FMA units, and organize them in a similar fashion.
 *  > One 512-bit FMA unit is created by fusing two 256-bit ones on port 0 and port 1. The other is added to port 5, as a server-specific
 *  > core extension. The FMA units on port 0 and 1 are configured into 2×256-bit or 1×512-bit mode depending on whether 512-bit FMA
 *  > instructions are present in the scheduler. That means a mix of 256-bit and 512-bit FMA instructions will not achieve higher IPC
 *  > than executing 512-bit instructions alone.
 *
 *  Source: https://chipsandcheese.com/p/a-peek-at-sapphire-rapids
 */
#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n,
                                                simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                           simsimd_f64_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                             simsimd_f64_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n,
                                                simsimd_f64_t* result);
#endif // SIMSIMD_TARGET_SKYLAKE

/*  SIMD-powered backends for AVX512 CPUs of Ice Lake generation and newer, using mixed arithmetic over 512-bit words.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
#if SIMSIMD_TARGET_ICE
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n,
                                        simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n,
                                          simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i4x2_ice(simsimd_i4x2_t const* a, simsimd_i4x2_t const* b, simsimd_size_t n,
                                             simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                      simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                        simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                      simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                        simsimd_u32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n,
                                           simsimd_f32_t* result);
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_GENOA
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                          simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n,
                                               simsimd_f32_t* result);
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
/** @copydoc simsimd_l2_f64 */
SIMSIMD_PUBLIC void simsimd_l2_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                            simsimd_f32_t* result);
/** @copydoc simsimd_l2sq_f64 */
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n,
                                                 simsimd_f32_t* result);
#endif // SIMSIMD_TARGET_SAPPHIRE

/*  SIMD-powered backends for AVX-INT8-VNNI extensions on Xeon 6 CPUs, including Sierra Forest and Granite Rapids.
 *  The packs many "efficiency" cores into a single socket, avoiding heavy 512-bit operations, and focusing on 256-bit ones.
 */
#if SIMSIMD_TARGET_SIERRA
/** @copydoc simsimd_angular_f64 */
SIMSIMD_PUBLIC void simsimd_angular_i8_sierra(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n,
                                              simsimd_f32_t* result);
#endif // SIMSIMD_TARGET_SIERRA

// clang-format on

#define SIMSIMD_MAKE_L2SQ(name, input_type, accumulator_type, output_type, load_and_convert)                    \
    SIMSIMD_PUBLIC void simsimd_l2sq_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                           simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                           simsimd_##output_type##_t *result) {                 \
        simsimd_##accumulator_type##_t distance_sq = 0, a_element, b_element;                                   \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                               \
            load_and_convert(a + i, &a_element);                                                                \
            load_and_convert(b + i, &b_element);                                                                \
            distance_sq += (a_element - b_element) * (a_element - b_element);                                   \
        }                                                                                                       \
        *result = (simsimd_##output_type##_t)distance_sq;                                                       \
    }

#define SIMSIMD_MAKE_L2(name, input_type, accumulator_type, l2sq_output_type, output_type, load_and_convert,  \
                        compute_sqrt)                                                                         \
    SIMSIMD_PUBLIC void simsimd_l2_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                         simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                         simsimd_##output_type##_t *result) {                 \
        simsimd_##l2sq_output_type##_t distance_sq;                                                           \
        simsimd_l2sq_##input_type##_##name(a, b, n, &distance_sq);                                            \
        *result = compute_sqrt((simsimd_##output_type##_t)distance_sq);                                       \
    }

#define SIMSIMD_MAKE_COS(name, input_type, accumulator_type, output_type, load_and_convert, compute_rsqrt)         \
    SIMSIMD_PUBLIC void simsimd_angular_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                              simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                              simsimd_##output_type##_t *result) {                 \
        simsimd_##accumulator_type##_t dot_product = 0, a_norm_sq = 0, b_norm_sq = 0, a_element, b_element;        \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                  \
            load_and_convert(a + i, &a_element);                                                                   \
            load_and_convert(b + i, &b_element);                                                                   \
            dot_product += a_element * b_element;                                                                  \
            a_norm_sq += a_element * a_element;                                                                    \
            b_norm_sq += b_element * b_element;                                                                    \
        }                                                                                                          \
        if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }                                                     \
        else if (dot_product == 0) { *result = 1; }                                                                \
        else {                                                                                                     \
            simsimd_##output_type##_t unclipped_distance = 1 - dot_product * compute_rsqrt(a_norm_sq) *            \
                                                                   compute_rsqrt(b_norm_sq);                       \
            *result = unclipped_distance > 0 ? unclipped_distance : 0;                                             \
        }                                                                                                          \
    }

SIMSIMD_MAKE_COS(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F64_RSQRT)    // simsimd_angular_f64_serial
SIMSIMD_MAKE_L2SQ(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO)                      // simsimd_l2sq_f64_serial
SIMSIMD_MAKE_L2(serial, f64, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F64_SQRT) // simsimd_l2_f64_serial

SIMSIMD_MAKE_COS(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_RSQRT)    // simsimd_angular_f32_serial
SIMSIMD_MAKE_L2SQ(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO)                      // simsimd_l2sq_f32_serial
SIMSIMD_MAKE_L2(serial, f32, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_SQRT) // simsimd_l2_f32_serial

SIMSIMD_MAKE_COS(serial, f16, f32, f32, simsimd_f16_to_f32, SIMSIMD_F32_RSQRT)    // simsimd_angular_f16_serial
SIMSIMD_MAKE_L2SQ(serial, f16, f32, f32, simsimd_f16_to_f32)                      // simsimd_l2sq_f16_serial
SIMSIMD_MAKE_L2(serial, f16, f32, f32, f32, simsimd_f16_to_f32, SIMSIMD_F32_SQRT) // simsimd_l2_f16_serial

SIMSIMD_MAKE_COS(serial, bf16, f32, f32, simsimd_bf16_to_f32, SIMSIMD_F32_RSQRT)    // simsimd_angular_bf16_serial
SIMSIMD_MAKE_L2SQ(serial, bf16, f32, f32, simsimd_bf16_to_f32)                      // simsimd_l2sq_bf16_serial
SIMSIMD_MAKE_L2(serial, bf16, f32, f32, f32, simsimd_bf16_to_f32, SIMSIMD_F32_SQRT) // simsimd_l2_bf16_serial

SIMSIMD_MAKE_COS(serial, i8, i32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_RSQRT)    // simsimd_angular_i8_serial
SIMSIMD_MAKE_L2SQ(serial, i8, i32, u32, SIMSIMD_ASSIGN_FROM_TO)                      // simsimd_l2sq_i8_serial
SIMSIMD_MAKE_L2(serial, i8, i32, u32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_SQRT) // simsimd_l2_i8_serial

SIMSIMD_MAKE_COS(serial, u8, i32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_RSQRT)    // simsimd_angular_u8_serial
SIMSIMD_MAKE_L2SQ(serial, u8, i32, u32, SIMSIMD_ASSIGN_FROM_TO)                      // simsimd_l2sq_u8_serial
SIMSIMD_MAKE_L2(serial, u8, i32, u32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_SQRT) // simsimd_l2_u8_serial

SIMSIMD_MAKE_COS(accurate, f32, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F64_RSQRT)    // simsimd_angular_f32_accurate
SIMSIMD_MAKE_L2SQ(accurate, f32, f64, f64, SIMSIMD_ASSIGN_FROM_TO)                      // simsimd_l2sq_f32_accurate
SIMSIMD_MAKE_L2(accurate, f32, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F64_SQRT) // simsimd_l2_f32_accurate

SIMSIMD_MAKE_COS(accurate, f16, f64, f64, simsimd_f16_to_f64, SIMSIMD_F64_RSQRT)    // simsimd_angular_f16_accurate
SIMSIMD_MAKE_L2SQ(accurate, f16, f64, f64, simsimd_f16_to_f64)                      // simsimd_l2sq_f16_accurate
SIMSIMD_MAKE_L2(accurate, f16, f64, f64, f64, simsimd_f16_to_f64, SIMSIMD_F64_SQRT) // simsimd_l2_f16_accurate

SIMSIMD_MAKE_COS(accurate, bf16, f64, f64, simsimd_bf16_to_f64, SIMSIMD_F64_RSQRT)    // simsimd_angular_bf16_accurate
SIMSIMD_MAKE_L2SQ(accurate, bf16, f64, f64, simsimd_bf16_to_f64)                      // simsimd_l2sq_bf16_accurate
SIMSIMD_MAKE_L2(accurate, bf16, f64, f64, f64, simsimd_bf16_to_f64, SIMSIMD_F64_SQRT) // simsimd_l2_bf16_accurate

typedef simsimd_dot_f64x2_state_serial_t simsimd_angular_f64x2_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_f64x2_init_serial(simsimd_angular_f64x2_state_serial_t *state) {
    simsimd_dot_f64x2_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f64x2_update_serial(simsimd_angular_f64x2_state_serial_t *state,
                                                          simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_f64x2_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f64x2_finalize_serial(simsimd_angular_f64x2_state_serial_t const *state_a,
                                                            simsimd_angular_f64x2_state_serial_t const *state_b,
                                                            simsimd_angular_f64x2_state_serial_t const *state_c,
                                                            simsimd_angular_f64x2_state_serial_t const *state_d,
                                                            simsimd_f64_t query_norm, simsimd_f64_t target_norm_a,
                                                            simsimd_f64_t target_norm_b, simsimd_f64_t target_norm_c,
                                                            simsimd_f64_t target_norm_d, simsimd_f64_t *results) {
    simsimd_f64_t dot_product_a = state_a->sums[0] + state_a->sums[1];
    simsimd_f64_t dot_product_b = state_b->sums[0] + state_b->sums[1];
    simsimd_f64_t dot_product_c = state_c->sums[0] + state_c->sums[1];
    simsimd_f64_t dot_product_d = state_d->sums[0] + state_d->sums[1];

    simsimd_f64_t query_norm_sq = query_norm * query_norm;
    simsimd_f64_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f64_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f64_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f64_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Precompute query rsqrt once (was computed 4x before)
    simsimd_f64_t query_rsqrt = query_norm_sq > 0 ? SIMSIMD_F64_RSQRT(query_norm_sq) : 0;
    simsimd_f64_t target_rsqrt_a = target_norm_sq_a > 0 ? SIMSIMD_F64_RSQRT(target_norm_sq_a) : 0;
    simsimd_f64_t target_rsqrt_b = target_norm_sq_b > 0 ? SIMSIMD_F64_RSQRT(target_norm_sq_b) : 0;
    simsimd_f64_t target_rsqrt_c = target_norm_sq_c > 0 ? SIMSIMD_F64_RSQRT(target_norm_sq_c) : 0;
    simsimd_f64_t target_rsqrt_d = target_norm_sq_d > 0 ? SIMSIMD_F64_RSQRT(target_norm_sq_d) : 0;

    simsimd_f64_t unclipped_distance_a = 1 - dot_product_a * query_rsqrt * target_rsqrt_a;
    simsimd_f64_t unclipped_distance_b = 1 - dot_product_b * query_rsqrt * target_rsqrt_b;
    simsimd_f64_t unclipped_distance_c = 1 - dot_product_c * query_rsqrt * target_rsqrt_c;
    simsimd_f64_t unclipped_distance_d = 1 - dot_product_d * query_rsqrt * target_rsqrt_d;

    results[0] = (query_norm_sq == 0 && target_norm_sq_a == 0) ? 0
                                                               : (unclipped_distance_a > 0 ? unclipped_distance_a : 0);
    results[1] = (query_norm_sq == 0 && target_norm_sq_b == 0) ? 0
                                                               : (unclipped_distance_b > 0 ? unclipped_distance_b : 0);
    results[2] = (query_norm_sq == 0 && target_norm_sq_c == 0) ? 0
                                                               : (unclipped_distance_c > 0 ? unclipped_distance_c : 0);
    results[3] = (query_norm_sq == 0 && target_norm_sq_d == 0) ? 0
                                                               : (unclipped_distance_d > 0 ? unclipped_distance_d : 0);
}

typedef simsimd_dot_f64x2_state_serial_t simsimd_l2_f64x2_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_f64x2_init_serial(simsimd_l2_f64x2_state_serial_t *state) {
    simsimd_dot_f64x2_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f64x2_update_serial(simsimd_l2_f64x2_state_serial_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    simsimd_dot_f64x2_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f64x2_finalize_serial(simsimd_l2_f64x2_state_serial_t const *state_a,
                                                       simsimd_l2_f64x2_state_serial_t const *state_b,
                                                       simsimd_l2_f64x2_state_serial_t const *state_c,
                                                       simsimd_l2_f64x2_state_serial_t const *state_d,
                                                       simsimd_f64_t query_norm, simsimd_f64_t target_norm_a,
                                                       simsimd_f64_t target_norm_b, simsimd_f64_t target_norm_c,
                                                       simsimd_f64_t target_norm_d, simsimd_f64_t *results) {
    simsimd_f64_t dot_product_a = state_a->sums[0] + state_a->sums[1];
    simsimd_f64_t dot_product_b = state_b->sums[0] + state_b->sums[1];
    simsimd_f64_t dot_product_c = state_c->sums[0] + state_c->sums[1];
    simsimd_f64_t dot_product_d = state_d->sums[0] + state_d->sums[1];

    simsimd_f64_t query_norm_sq = query_norm * query_norm;
    simsimd_f64_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f64_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f64_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f64_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f64_t distance_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_product_a;
    simsimd_f64_t distance_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_product_b;
    simsimd_f64_t distance_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_product_c;
    simsimd_f64_t distance_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_product_d;

    results[0] = distance_sq_a > 0 ? SIMSIMD_F64_SQRT(distance_sq_a) : 0;
    results[1] = distance_sq_b > 0 ? SIMSIMD_F64_SQRT(distance_sq_b) : 0;
    results[2] = distance_sq_c > 0 ? SIMSIMD_F64_SQRT(distance_sq_c) : 0;
    results[3] = distance_sq_d > 0 ? SIMSIMD_F64_SQRT(distance_sq_d) : 0;
}

typedef simsimd_dot_f32x4_state_serial_t simsimd_angular_f32x4_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_f32x4_init_serial(simsimd_angular_f32x4_state_serial_t *state) {
    simsimd_dot_f32x4_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x4_update_serial(simsimd_angular_f32x4_state_serial_t *state,
                                                          simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_f32x4_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x4_finalize_serial(simsimd_angular_f32x4_state_serial_t const *state_a,
                                                            simsimd_angular_f32x4_state_serial_t const *state_b,
                                                            simsimd_angular_f32x4_state_serial_t const *state_c,
                                                            simsimd_angular_f32x4_state_serial_t const *state_d,
                                                            simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                            simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                            simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    simsimd_f32_t dots[4];
    simsimd_dot_f32x4_finalize_serial(state_a, state_b, state_c, state_d, dots);

    simsimd_f32_t dot_product_a = dots[0], dot_product_b = dots[1];
    simsimd_f32_t dot_product_c = dots[2], dot_product_d = dots[3];

    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Precompute query rsqrt once (was computed 4x before)
    simsimd_f32_t query_rsqrt = query_norm_sq > 0 ? SIMSIMD_F32_RSQRT(query_norm_sq) : 0;
    simsimd_f32_t target_rsqrt_a = target_norm_sq_a > 0 ? SIMSIMD_F32_RSQRT(target_norm_sq_a) : 0;
    simsimd_f32_t target_rsqrt_b = target_norm_sq_b > 0 ? SIMSIMD_F32_RSQRT(target_norm_sq_b) : 0;
    simsimd_f32_t target_rsqrt_c = target_norm_sq_c > 0 ? SIMSIMD_F32_RSQRT(target_norm_sq_c) : 0;
    simsimd_f32_t target_rsqrt_d = target_norm_sq_d > 0 ? SIMSIMD_F32_RSQRT(target_norm_sq_d) : 0;

    simsimd_f32_t unclipped_distance_a = 1 - dot_product_a * query_rsqrt * target_rsqrt_a;
    simsimd_f32_t unclipped_distance_b = 1 - dot_product_b * query_rsqrt * target_rsqrt_b;
    simsimd_f32_t unclipped_distance_c = 1 - dot_product_c * query_rsqrt * target_rsqrt_c;
    simsimd_f32_t unclipped_distance_d = 1 - dot_product_d * query_rsqrt * target_rsqrt_d;

    results[0] = (query_norm_sq == 0 && target_norm_sq_a == 0) ? 0
                                                               : (unclipped_distance_a > 0 ? unclipped_distance_a : 0);
    results[1] = (query_norm_sq == 0 && target_norm_sq_b == 0) ? 0
                                                               : (unclipped_distance_b > 0 ? unclipped_distance_b : 0);
    results[2] = (query_norm_sq == 0 && target_norm_sq_c == 0) ? 0
                                                               : (unclipped_distance_c > 0 ? unclipped_distance_c : 0);
    results[3] = (query_norm_sq == 0 && target_norm_sq_d == 0) ? 0
                                                               : (unclipped_distance_d > 0 ? unclipped_distance_d : 0);
}

typedef simsimd_dot_f32x4_state_serial_t simsimd_l2_f32x4_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_f32x4_init_serial(simsimd_l2_f32x4_state_serial_t *state) {
    simsimd_dot_f32x4_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x4_update_serial(simsimd_l2_f32x4_state_serial_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    simsimd_dot_f32x4_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x4_finalize_serial(simsimd_l2_f32x4_state_serial_t const *state_a,
                                                       simsimd_l2_f32x4_state_serial_t const *state_b,
                                                       simsimd_l2_f32x4_state_serial_t const *state_c,
                                                       simsimd_l2_f32x4_state_serial_t const *state_d,
                                                       simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                       simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                       simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    simsimd_f32_t dots[4];
    simsimd_dot_f32x4_finalize_serial(state_a, state_b, state_c, state_d, dots);

    simsimd_f32_t dot_product_a = dots[0], dot_product_b = dots[1];
    simsimd_f32_t dot_product_c = dots[2], dot_product_d = dots[3];

    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f32_t distance_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_product_a;
    simsimd_f32_t distance_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_product_b;
    simsimd_f32_t distance_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_product_c;
    simsimd_f32_t distance_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_product_d;

    results[0] = distance_sq_a > 0 ? SIMSIMD_F32_SQRT(distance_sq_a) : 0;
    results[1] = distance_sq_b > 0 ? SIMSIMD_F32_SQRT(distance_sq_b) : 0;
    results[2] = distance_sq_c > 0 ? SIMSIMD_F32_SQRT(distance_sq_c) : 0;
    results[3] = distance_sq_d > 0 ? SIMSIMD_F32_SQRT(distance_sq_d) : 0;
}

typedef simsimd_dot_f16x8_state_serial_t simsimd_angular_f16x8_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_f16x8_init_serial(simsimd_angular_f16x8_state_serial_t *state) {
    simsimd_dot_f16x8_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x8_update_serial(simsimd_angular_f16x8_state_serial_t *state,
                                                          simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_f16x8_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x8_finalize_serial(simsimd_angular_f16x8_state_serial_t const *state_a,
                                                            simsimd_angular_f16x8_state_serial_t const *state_b,
                                                            simsimd_angular_f16x8_state_serial_t const *state_c,
                                                            simsimd_angular_f16x8_state_serial_t const *state_d,
                                                            simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                            simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                            simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    simsimd_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    simsimd_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    simsimd_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    simsimd_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    simsimd_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0)
        results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0)
        results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0)
        results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0)
        results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef simsimd_dot_f16x8_state_serial_t simsimd_l2_f16x8_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_f16x8_init_serial(simsimd_l2_f16x8_state_serial_t *state) {
    simsimd_dot_f16x8_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x8_update_serial(simsimd_l2_f16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    simsimd_dot_f16x8_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x8_finalize_serial(simsimd_l2_f16x8_state_serial_t const *state_a,
                                                       simsimd_l2_f16x8_state_serial_t const *state_b,
                                                       simsimd_l2_f16x8_state_serial_t const *state_c,
                                                       simsimd_l2_f16x8_state_serial_t const *state_d,
                                                       simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                       simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                       simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    simsimd_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    simsimd_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    simsimd_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    simsimd_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared distances (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    simsimd_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    simsimd_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    simsimd_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? SIMSIMD_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? SIMSIMD_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? SIMSIMD_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? SIMSIMD_F32_SQRT(dist_sq_d) : 0;
}

typedef simsimd_dot_bf16x8_state_serial_t simsimd_angular_bf16x8_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_init_serial(simsimd_angular_bf16x8_state_serial_t *state) {
    simsimd_dot_bf16x8_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_update_serial(simsimd_angular_bf16x8_state_serial_t *state,
                                                           simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_bf16x8_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_finalize_serial(simsimd_angular_bf16x8_state_serial_t const *state_a,
                                                             simsimd_angular_bf16x8_state_serial_t const *state_b,
                                                             simsimd_angular_bf16x8_state_serial_t const *state_c,
                                                             simsimd_angular_bf16x8_state_serial_t const *state_d,
                                                             simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                             simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                             simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    simsimd_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    simsimd_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    simsimd_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    simsimd_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    simsimd_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0)
        results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0)
        results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0)
        results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0)
        results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef simsimd_dot_bf16x8_state_serial_t simsimd_l2_bf16x8_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_init_serial(simsimd_l2_bf16x8_state_serial_t *state) {
    simsimd_dot_bf16x8_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_update_serial(simsimd_l2_bf16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_dot_bf16x8_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_finalize_serial(simsimd_l2_bf16x8_state_serial_t const *state_a,
                                                        simsimd_l2_bf16x8_state_serial_t const *state_b,
                                                        simsimd_l2_bf16x8_state_serial_t const *state_c,
                                                        simsimd_l2_bf16x8_state_serial_t const *state_d,
                                                        simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                        simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                        simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    simsimd_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    simsimd_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    simsimd_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    simsimd_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared distances (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    simsimd_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    simsimd_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    simsimd_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? SIMSIMD_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? SIMSIMD_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? SIMSIMD_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? SIMSIMD_F32_SQRT(dist_sq_d) : 0;
}

typedef simsimd_dot_i8x16_state_serial_t simsimd_angular_i8x16_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_i8x16_init_serial(simsimd_angular_i8x16_state_serial_t *state) {
    simsimd_dot_i8x16_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x16_update_serial(simsimd_angular_i8x16_state_serial_t *state,
                                                          simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_i8x16_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x16_finalize_serial(simsimd_angular_i8x16_state_serial_t const *state_a,
                                                            simsimd_angular_i8x16_state_serial_t const *state_b,
                                                            simsimd_angular_i8x16_state_serial_t const *state_c,
                                                            simsimd_angular_i8x16_state_serial_t const *state_d,
                                                            simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                            simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                            simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    simsimd_i64_t dot_a = state_a->sums[0] + state_a->sums[1];
    simsimd_i64_t dot_b = state_b->sums[0] + state_b->sums[1];
    simsimd_i64_t dot_c = state_c->sums[0] + state_c->sums[1];
    simsimd_i64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    simsimd_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0)
        results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0)
        results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0)
        results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0)
        results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef simsimd_dot_i8x16_state_serial_t simsimd_l2_i8x16_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_i8x16_init_serial(simsimd_l2_i8x16_state_serial_t *state) {
    simsimd_dot_i8x16_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x16_update_serial(simsimd_l2_i8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    simsimd_dot_i8x16_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x16_finalize_serial(simsimd_l2_i8x16_state_serial_t const *state_a,
                                                       simsimd_l2_i8x16_state_serial_t const *state_b,
                                                       simsimd_l2_i8x16_state_serial_t const *state_c,
                                                       simsimd_l2_i8x16_state_serial_t const *state_d,
                                                       simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                       simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                       simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    simsimd_i64_t dot_a = state_a->sums[0] + state_a->sums[1];
    simsimd_i64_t dot_b = state_b->sums[0] + state_b->sums[1];
    simsimd_i64_t dot_c = state_c->sums[0] + state_c->sums[1];
    simsimd_i64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared distances (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    simsimd_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    simsimd_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    simsimd_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? SIMSIMD_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? SIMSIMD_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? SIMSIMD_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? SIMSIMD_F32_SQRT(dist_sq_d) : 0;
}

typedef simsimd_dot_u8x16_state_serial_t simsimd_angular_u8x16_state_serial_t;
SIMSIMD_INTERNAL void simsimd_angular_u8x16_init_serial(simsimd_angular_u8x16_state_serial_t *state) {
    simsimd_dot_u8x16_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x16_update_serial(simsimd_angular_u8x16_state_serial_t *state,
                                                          simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_u8x16_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x16_finalize_serial(simsimd_angular_u8x16_state_serial_t const *state_a,
                                                            simsimd_angular_u8x16_state_serial_t const *state_b,
                                                            simsimd_angular_u8x16_state_serial_t const *state_c,
                                                            simsimd_angular_u8x16_state_serial_t const *state_d,
                                                            simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                            simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                            simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    simsimd_u64_t dot_a = state_a->sums[0] + state_a->sums[1];
    simsimd_u64_t dot_b = state_b->sums[0] + state_b->sums[1];
    simsimd_u64_t dot_c = state_c->sums[0] + state_c->sums[1];
    simsimd_u64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    simsimd_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0)
        results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0)
        results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0)
        results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0)
        results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * SIMSIMD_F32_RSQRT(query_norm_sq) * SIMSIMD_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef simsimd_dot_u8x16_state_serial_t simsimd_l2_u8x16_state_serial_t;
SIMSIMD_INTERNAL void simsimd_l2_u8x16_init_serial(simsimd_l2_u8x16_state_serial_t *state) {
    simsimd_dot_u8x16_init_serial(state);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x16_update_serial(simsimd_l2_u8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    simsimd_dot_u8x16_update_serial(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x16_finalize_serial(simsimd_l2_u8x16_state_serial_t const *state_a,
                                                       simsimd_l2_u8x16_state_serial_t const *state_b,
                                                       simsimd_l2_u8x16_state_serial_t const *state_c,
                                                       simsimd_l2_u8x16_state_serial_t const *state_d,
                                                       simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                       simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                       simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    simsimd_u64_t dot_a = state_a->sums[0] + state_a->sums[1];
    simsimd_u64_t dot_b = state_b->sums[0] + state_b->sums[1];
    simsimd_u64_t dot_c = state_c->sums[0] + state_c->sums[1];
    simsimd_u64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared distances (loop-unrolled)
    simsimd_f32_t query_norm_sq = query_norm * query_norm;
    simsimd_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    simsimd_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    simsimd_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    simsimd_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    simsimd_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    simsimd_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    simsimd_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    simsimd_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? SIMSIMD_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? SIMSIMD_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? SIMSIMD_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? SIMSIMD_F32_SQRT(dist_sq_d) : 0;
}

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_f32_t _simsimd_sqrt_f32_neon(simsimd_f32_t x) {
    return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
}
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_sqrt_f64_neon(simsimd_f64_t x) {
    return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0);
}
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_angular_normalize_f32_neon(simsimd_f32_t ab, simsimd_f32_t a2,
                                                                   simsimd_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    simsimd_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts = vrsqrte_f32(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    vst1_f32(squares_arr, rsqrts);
    simsimd_f32_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_angular_normalize_f64_neon(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                   simsimd_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    simsimd_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts = vrsqrteq_f64(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    vst1q_f64(squares_arr, rsqrts);
    simsimd_f64_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

SIMSIMD_PUBLIC void simsimd_l2_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
    simsimd_l2sq_f32_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        simsimd_f32_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_angular_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    }
    simsimd_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = (simsimd_f32_t)_simsimd_angular_normalize_f64_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_f64_t *result) {
    simsimd_l2sq_f64_neon(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                          simsimd_f64_t *result) {
    float64x2_t sum_vec = vdupq_n_f64(0);
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        float64x2_t diff_vec = vsubq_f64(a_vec, b_vec);
        sum_vec = vfmaq_f64(sum_vec, diff_vec, diff_vec);
    }
    simsimd_f64_t sum = vaddvq_f64(sum_vec);
    for (; i < n; ++i) {
        simsimd_f64_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_angular_f64_neon(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *result) {
    float64x2_t ab_vec = vdupq_n_f64(0), a2_vec = vdupq_n_f64(0), b2_vec = vdupq_n_f64(0);
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        ab_vec = vfmaq_f64(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f64(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f64(b2_vec, b_vec, b_vec);
    }
    simsimd_f64_t ab = vaddvq_f64(ab_vec), a2 = vaddvq_f64(a2_vec), b2 = vaddvq_f64(b2_vec);
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_angular_normalize_f64_neon(ab, a2, b2);
}

typedef simsimd_dot_f32x4_state_neon_t simsimd_angular_f32x4_state_neon_t;
SIMSIMD_INTERNAL void simsimd_angular_f32x4_init_neon(simsimd_angular_f32x4_state_neon_t *state) {
    simsimd_dot_f32x4_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x4_update_neon(simsimd_angular_f32x4_state_neon_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_dot_f32x4_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x4_finalize_neon(simsimd_angular_f32x4_state_neon_t const *state_a,
                                                          simsimd_angular_f32x4_state_neon_t const *state_b,
                                                          simsimd_angular_f32x4_state_neon_t const *state_c,
                                                          simsimd_angular_f32x4_state_neon_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single vectorized call
    simsimd_f32_t dots[4];
    simsimd_dot_f32x4_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F64 vectors for parallel processing (2x float64x2_t for precision)
    float64x2_t dots_ab = {(simsimd_f64_t)dots[0], (simsimd_f64_t)dots[1]};
    float64x2_t dots_cd = {(simsimd_f64_t)dots[2], (simsimd_f64_t)dots[3]};

    simsimd_f64_t query_norm_sq = (simsimd_f64_t)query_norm * (simsimd_f64_t)query_norm;
    float64x2_t query_sq = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab = {(simsimd_f64_t)target_norm_a, (simsimd_f64_t)target_norm_b};
    float64x2_t target_norms_cd = {(simsimd_f64_t)target_norm_c, (simsimd_f64_t)target_norm_d};
    float64x2_t target_sq_ab = vmulq_f64(target_norms_ab, target_norms_ab);
    float64x2_t target_sq_cd = vmulq_f64(target_norms_cd, target_norms_cd);

    // Compute products for normalization: query_sq * target_sq
    float64x2_t products_ab = vmulq_f64(query_sq, target_sq_ab);
    float64x2_t products_cd = vmulq_f64(query_sq, target_sq_cd);

    // Vectorized rsqrt with Newton-Raphson (2 iterations for ~48-bit precision)
    float64x2_t rsqrt_ab = vrsqrteq_f64(products_ab);
    float64x2_t rsqrt_cd = vrsqrteq_f64(products_cd);
    rsqrt_ab = vmulq_f64(rsqrt_ab, vrsqrtsq_f64(vmulq_f64(products_ab, rsqrt_ab), rsqrt_ab));
    rsqrt_cd = vmulq_f64(rsqrt_cd, vrsqrtsq_f64(vmulq_f64(products_cd, rsqrt_cd), rsqrt_cd));
    rsqrt_ab = vmulq_f64(rsqrt_ab, vrsqrtsq_f64(vmulq_f64(products_ab, rsqrt_ab), rsqrt_ab));
    rsqrt_cd = vmulq_f64(rsqrt_cd, vrsqrtsq_f64(vmulq_f64(products_cd, rsqrt_cd), rsqrt_cd));

    // Compute angular distance = 1 - dot * rsqrt(product)
    float64x2_t ones = vdupq_n_f64(1.0);
    float64x2_t zeros = vdupq_n_f64(0.0);
    float64x2_t result_ab = vsubq_f64(ones, vmulq_f64(dots_ab, rsqrt_ab));
    float64x2_t result_cd = vsubq_f64(ones, vmulq_f64(dots_cd, rsqrt_cd));

    // Clamp to [0, inf)
    result_ab = vmaxq_f64(result_ab, zeros);
    result_cd = vmaxq_f64(result_cd, zeros);

    // Handle edge cases with vectorized selects
    uint64x2_t products_zero_ab = vceqq_f64(products_ab, zeros);
    uint64x2_t products_zero_cd = vceqq_f64(products_cd, zeros);
    uint64x2_t dots_zero_ab = vceqq_f64(dots_ab, zeros);
    uint64x2_t dots_zero_cd = vceqq_f64(dots_cd, zeros);

    // Both zero -> result = 0; products zero but dots nonzero -> result = 1
    uint64x2_t both_zero_ab = vandq_u64(products_zero_ab, dots_zero_ab);
    uint64x2_t both_zero_cd = vandq_u64(products_zero_cd, dots_zero_cd);
    result_ab = vbslq_f64(both_zero_ab, zeros, result_ab);
    result_cd = vbslq_f64(both_zero_cd, zeros, result_cd);

    uint64x2_t prod_zero_dot_nonzero_ab = vandq_u64(products_zero_ab, vmvnq_u64(dots_zero_ab));
    uint64x2_t prod_zero_dot_nonzero_cd = vandq_u64(products_zero_cd, vmvnq_u64(dots_zero_cd));
    result_ab = vbslq_f64(prod_zero_dot_nonzero_ab, ones, result_ab);
    result_cd = vbslq_f64(prod_zero_dot_nonzero_cd, ones, result_cd);

    // Convert to F32 and store
    float32x2_t result_ab_f32 = vcvt_f32_f64(result_ab);
    float32x2_t result_cd_f32 = vcvt_f32_f64(result_cd);
    float32x4_t result_vec = vcombine_f32(result_ab_f32, result_cd_f32);
    vst1q_f32(results, result_vec);
}

typedef simsimd_dot_f32x4_state_neon_t simsimd_l2_f32x4_state_neon_t;
SIMSIMD_INTERNAL void simsimd_l2_f32x4_init_neon(simsimd_l2_f32x4_state_neon_t *state) {
    simsimd_dot_f32x4_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x4_update_neon(simsimd_l2_f32x4_state_neon_t *state, simsimd_b128_vec_t a,
                                                   simsimd_b128_vec_t b) {
    simsimd_dot_f32x4_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x4_finalize_neon(simsimd_l2_f32x4_state_neon_t const *state_a,
                                                     simsimd_l2_f32x4_state_neon_t const *state_b,
                                                     simsimd_l2_f32x4_state_neon_t const *state_c,
                                                     simsimd_l2_f32x4_state_neon_t const *state_d,
                                                     simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                     simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                     simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products
    simsimd_f32_t dots[4];
    simsimd_dot_f32x4_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F64 vectors (for precision as in original)
    float64x2_t dots_ab = {(simsimd_f64_t)dots[0], (simsimd_f64_t)dots[1]};
    float64x2_t dots_cd = {(simsimd_f64_t)dots[2], (simsimd_f64_t)dots[3]};

    simsimd_f64_t query_norm_sq = (simsimd_f64_t)query_norm * (simsimd_f64_t)query_norm;
    float64x2_t query_sq = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab = {(simsimd_f64_t)target_norm_a, (simsimd_f64_t)target_norm_b};
    float64x2_t target_norms_cd = {(simsimd_f64_t)target_norm_c, (simsimd_f64_t)target_norm_d};
    float64x2_t target_sq_ab = vmulq_f64(target_norms_ab, target_norms_ab);
    float64x2_t target_sq_cd = vmulq_f64(target_norms_cd, target_norms_cd);

    // Compute dist_sq = query_sq + target_sq - 2*dot using FMA
    float64x2_t neg_two = vdupq_n_f64(-2.0);
    float64x2_t sum_sq_ab = vaddq_f64(query_sq, target_sq_ab);
    float64x2_t sum_sq_cd = vaddq_f64(query_sq, target_sq_cd);
    float64x2_t dist_sq_ab = vfmaq_f64(sum_sq_ab, neg_two, dots_ab);
    float64x2_t dist_sq_cd = vfmaq_f64(sum_sq_cd, neg_two, dots_cd);

    // Clamp negative values to zero (numerical stability)
    float64x2_t zeros = vdupq_n_f64(0.0);
    dist_sq_ab = vmaxq_f64(dist_sq_ab, zeros);
    dist_sq_cd = vmaxq_f64(dist_sq_cd, zeros);

    // Compute sqrt using hardware vsqrtq_f64
    float64x2_t dist_ab = vsqrtq_f64(dist_sq_ab);
    float64x2_t dist_cd = vsqrtq_f64(dist_sq_cd);

    // Convert to F32 and store
    float32x2_t dist_ab_f32 = vcvt_f32_f64(dist_ab);
    float32x2_t dist_cd_f32 = vcvt_f32_f64(dist_cd);
    float32x4_t dist_vec = vcombine_f32(dist_ab_f32, dist_cd_f32);
    vst1q_f32(results, dist_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
    simsimd_l2sq_f16_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t distance_sq_f32x4 = vdupq_n_f32(0);

simsimd_l2sq_f16_neon_cycle:
    if (n < 4) {
        a_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    distance_sq_f32x4 = vfmaq_f32(distance_sq_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto simsimd_l2sq_f16_neon_cycle;

    *result = vaddvq_f32(distance_sq_f32x4);
}

SIMSIMD_PUBLIC void simsimd_angular_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {
    float32x4_t dot_product_f32x4 = vdupq_n_f32(0), a_norm_sq_f32x4 = vdupq_n_f32(0), b_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t a_f32x4, b_f32x4;

simsimd_angular_f16_neon_cycle:
    if (n < 4) {
        a_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    dot_product_f32x4 = vfmaq_f32(dot_product_f32x4, a_f32x4, b_f32x4);
    a_norm_sq_f32x4 = vfmaq_f32(a_norm_sq_f32x4, a_f32x4, a_f32x4);
    b_norm_sq_f32x4 = vfmaq_f32(b_norm_sq_f32x4, b_f32x4, b_f32x4);
    if (n) goto simsimd_angular_f16_neon_cycle;

    simsimd_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    simsimd_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    simsimd_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = _simsimd_angular_normalize_f32_neon(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef simsimd_dot_f16x8_state_neon_t simsimd_angular_f16x8_state_neon_t;
SIMSIMD_INTERNAL void simsimd_angular_f16x8_init_neon(simsimd_angular_f16x8_state_neon_t *state) {
    simsimd_dot_f16x8_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x8_update_neon(simsimd_angular_f16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_dot_f16x8_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x8_finalize_neon(simsimd_angular_f16x8_state_neon_t const *state_a,
                                                          simsimd_angular_f16x8_state_neon_t const *state_b,
                                                          simsimd_angular_f16x8_state_neon_t const *state_c,
                                                          simsimd_angular_f16x8_state_neon_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single call
    simsimd_f32_t dots[4];
    simsimd_dot_f16x8_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors for parallel processing
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);

    // Compute products for normalization: query_sq * target_sq
    float32x4_t products = vmulq_f32(query_sq, target_sq);

    // Vectorized rsqrt with Newton-Raphson (2 iterations)
    float32x4_t rsqrt_vec = vrsqrteq_f32(products);
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));

    // Compute angular distance = 1 - dot * rsqrt(product)
    float32x4_t ones = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t result_vec = vsubq_f32(ones, vmulq_f32(dots_vec, rsqrt_vec));

    // Clamp to [0, inf) and handle edge cases
    result_vec = vmaxq_f32(result_vec, zeros);
    uint32x4_t products_zero = vceqq_f32(products, zeros);
    uint32x4_t dots_zero = vceqq_f32(dots_vec, zeros);
    uint32x4_t both_zero = vandq_u32(products_zero, dots_zero);
    result_vec = vbslq_f32(both_zero, zeros, result_vec);
    uint32x4_t prod_zero_dot_nonzero = vandq_u32(products_zero, vmvnq_u32(dots_zero));
    result_vec = vbslq_f32(prod_zero_dot_nonzero, ones, result_vec);

    vst1q_f32(results, result_vec);
}

typedef simsimd_dot_f16x8_state_neon_t simsimd_l2_f16x8_state_neon_t;
SIMSIMD_INTERNAL void simsimd_l2_f16x8_init_neon(simsimd_l2_f16x8_state_neon_t *state) {
    simsimd_dot_f16x8_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x8_update_neon(simsimd_l2_f16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                   simsimd_b128_vec_t b) {
    simsimd_dot_f16x8_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x8_finalize_neon(simsimd_l2_f16x8_state_neon_t const *state_a,
                                                     simsimd_l2_f16x8_state_neon_t const *state_b,
                                                     simsimd_l2_f16x8_state_neon_t const *state_c,
                                                     simsimd_l2_f16x8_state_neon_t const *state_d,
                                                     simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                     simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                     simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products
    simsimd_f32_t dots[4];
    simsimd_dot_f16x8_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);

    // Compute dist_sq = query_sq + target_sq - 2*dot using FMA
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    float32x4_t sum_sq = vaddq_f32(query_sq, target_sq);
    float32x4_t dist_sq = vfmaq_f32(sum_sq, neg_two, dots_vec);

    // Clamp and sqrt
    float32x4_t zeros = vdupq_n_f32(0.0f);
    dist_sq = vmaxq_f32(dist_sq, zeros);
    float32x4_t dist_vec = vsqrtq_f32(dist_sq);

    vst1q_f32(results, dist_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_angular_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result) {

    // Similar to `simsimd_angular_i8_neon`, we can use the `BFMMLA` instruction through
    // the `vbfmmlaq_f32` intrinsic to compute matrix products and later drop 1/4 of values.
    // The only difference is that `zip` isn't provided for `bf16` and we need to reinterpret back
    // and forth before zipping. Same as with integers, on modern Arm CPUs, this "smart"
    // approach is actually slower by around 25%.
    //
    //   float32x4_t products_low_vec = vdupq_n_f32(0.0f);
    //   float32x4_t products_high_vec = vdupq_n_f32(0.0f);
    //   for (; i + 8 <= n; i += 8) {
    //       bfloat16x8_t a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)a + i);
    //       bfloat16x8_t b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const*)b + i);
    //       int16x8_t a_vec_s16 = vreinterpretq_s16_bf16(a_vec);
    //       int16x8_t b_vec_s16 = vreinterpretq_s16_bf16(b_vec);
    //       int16x8x2_t y_w_vecs_s16 = vzipq_s16(a_vec_s16, b_vec_s16);
    //       bfloat16x8_t y_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[0]);
    //       bfloat16x8_t w_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[1]);
    //       bfloat16x4_t a_low = vget_low_bf16(a_vec);
    //       bfloat16x4_t b_low = vget_low_bf16(b_vec);
    //       bfloat16x4_t a_high = vget_high_bf16(a_vec);
    //       bfloat16x4_t b_high = vget_high_bf16(b_vec);
    //       bfloat16x8_t x_vec = vcombine_bf16(a_low, b_low);
    //       bfloat16x8_t v_vec = vcombine_bf16(a_high, b_high);
    //       products_low_vec = vbfmmlaq_f32(products_low_vec, x_vec, y_vec);
    //       products_high_vec = vbfmmlaq_f32(products_high_vec, v_vec, w_vec);
    //   }
    //   float32x4_t products_vec = vaddq_f32(products_high_vec, products_low_vec);
    //   simsimd_f32_t a2 = products_vec[0], ab = products_vec[1], b2 = products_vec[3];
    //
    // Another way of accomplishing the same thing is to process the odd and even elements separately,
    // using special `vbfmlaltq_f32` and `vbfmlalbq_f32` intrinsics:
    //
    //      ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_vec, b_vec);
    //      ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_vec, b_vec);
    //      a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_vec, a_vec);
    //      a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_vec, a_vec);
    //      b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_vec, b_vec);
    //      b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_vec, b_vec);
    //

    float32x4_t dot_product_f32x4 = vdupq_n_f32(0);
    float32x4_t a_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t b_norm_sq_f32x4 = vdupq_n_f32(0);
    bfloat16x8_t a_bf16x8, b_bf16x8;

simsimd_angular_bf16_neon_cycle:
    if (n < 8) {
        a_bf16x8 = _simsimd_partial_load_bf16x8_neon(a, n);
        b_bf16x8 = _simsimd_partial_load_bf16x8_neon(b, n);
        n = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a);
        b_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b);
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x4 = vbfdotq_f32(dot_product_f32x4, a_bf16x8, b_bf16x8);
    a_norm_sq_f32x4 = vbfdotq_f32(a_norm_sq_f32x4, a_bf16x8, a_bf16x8);
    b_norm_sq_f32x4 = vbfdotq_f32(b_norm_sq_f32x4, b_bf16x8, b_bf16x8);
    if (n) goto simsimd_angular_bf16_neon_cycle;

    // Avoid `simsimd_f32_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    simsimd_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    simsimd_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = _simsimd_angular_normalize_f32_neon(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

SIMSIMD_PUBLIC void simsimd_l2_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result) {
    simsimd_l2sq_bf16_neon(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {
    float32x4_t diff_high_f32x4, diff_low_f32x4;
    float32x4_t distance_sq_high_f32x4 = vdupq_n_f32(0), distance_sq_low_f32x4 = vdupq_n_f32(0);

simsimd_l2sq_bf16_neon_cycle:
    if (n < 8) {
        bfloat16x8_t a_bf16x8 = _simsimd_partial_load_bf16x8_neon(a, n);
        bfloat16x8_t b_bf16x8 = _simsimd_partial_load_bf16x8_neon(b, n);
        diff_high_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_bf16x8)), vcvt_f32_bf16(vget_high_bf16(b_bf16x8)));
        diff_low_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_bf16x8)), vcvt_f32_bf16(vget_low_bf16(b_bf16x8)));
        n = 0;
    }
    else {
        bfloat16x8_t a_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a);
        bfloat16x8_t b_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b);
        diff_high_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_bf16x8)), vcvt_f32_bf16(vget_high_bf16(b_bf16x8)));
        diff_low_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_bf16x8)), vcvt_f32_bf16(vget_low_bf16(b_bf16x8)));
        n -= 8, a += 8, b += 8;
    }
    distance_sq_high_f32x4 = vfmaq_f32(distance_sq_high_f32x4, diff_high_f32x4, diff_high_f32x4);
    distance_sq_low_f32x4 = vfmaq_f32(distance_sq_low_f32x4, diff_low_f32x4, diff_low_f32x4);
    if (n) goto simsimd_l2sq_bf16_neon_cycle;

    *result = vaddvq_f32(vaddq_f32(distance_sq_high_f32x4, distance_sq_low_f32x4));
}

typedef simsimd_dot_bf16x8_state_neon_t simsimd_angular_bf16x8_state_neon_t;
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_init_neon(simsimd_angular_bf16x8_state_neon_t *state) {
    simsimd_dot_bf16x8_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_update_neon(simsimd_angular_bf16x8_state_neon_t *state,
                                                         simsimd_b128_vec_t a, simsimd_b128_vec_t b) {
    simsimd_dot_bf16x8_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x8_finalize_neon(simsimd_angular_bf16x8_state_neon_t const *state_a,
                                                           simsimd_angular_bf16x8_state_neon_t const *state_b,
                                                           simsimd_angular_bf16x8_state_neon_t const *state_c,
                                                           simsimd_angular_bf16x8_state_neon_t const *state_d,
                                                           simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                           simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                           simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single call
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x8_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors for parallel processing
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);
    float32x4_t products = vmulq_f32(query_sq, target_sq);

    // Vectorized rsqrt with Newton-Raphson
    float32x4_t rsqrt_vec = vrsqrteq_f32(products);
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));

    // Compute angular distance and handle edge cases
    float32x4_t ones = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t result_vec = vsubq_f32(ones, vmulq_f32(dots_vec, rsqrt_vec));
    result_vec = vmaxq_f32(result_vec, zeros);

    uint32x4_t products_zero = vceqq_f32(products, zeros);
    uint32x4_t dots_zero = vceqq_f32(dots_vec, zeros);
    uint32x4_t both_zero = vandq_u32(products_zero, dots_zero);
    result_vec = vbslq_f32(both_zero, zeros, result_vec);
    uint32x4_t prod_zero_dot_nonzero = vandq_u32(products_zero, vmvnq_u32(dots_zero));
    result_vec = vbslq_f32(prod_zero_dot_nonzero, ones, result_vec);

    vst1q_f32(results, result_vec);
}

typedef simsimd_dot_bf16x8_state_neon_t simsimd_l2_bf16x8_state_neon_t;
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_init_neon(simsimd_l2_bf16x8_state_neon_t *state) {
    simsimd_dot_bf16x8_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_update_neon(simsimd_l2_bf16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b) {
    simsimd_dot_bf16x8_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x8_finalize_neon(simsimd_l2_bf16x8_state_neon_t const *state_a,
                                                      simsimd_l2_bf16x8_state_neon_t const *state_b,
                                                      simsimd_l2_bf16x8_state_neon_t const *state_c,
                                                      simsimd_l2_bf16x8_state_neon_t const *state_d,
                                                      simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                      simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                      simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x8_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);

    // Compute dist_sq = query_sq + target_sq - 2*dot using FMA
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    float32x4_t sum_sq = vaddq_f32(query_sq, target_sq);
    float32x4_t dist_sq = vfmaq_f32(sum_sq, neg_two, dots_vec);

    // Clamp and sqrt
    float32x4_t zeros = vdupq_n_f32(0.0f);
    dist_sq = vmaxq_f32(dist_sq, zeros);
    float32x4_t dist_vec = vsqrtq_f32(dist_sq);

    vst1q_f32(results, dist_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
    simsimd_u32_t distance_sq_u32;
    simsimd_l2sq_i8_neon(a, b, n, &distance_sq_u32);
    *result = _simsimd_sqrt_f32_neon((simsimd_f32_t)distance_sq_u32);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                         simsimd_u32_t *result) {

    // The naive approach is to upcast 8-bit signed integers into 16-bit signed integers
    // for subtraction, then multiply within 16-bit integers and accumulate the results
    // into 32-bit integers. This approach is slow on modern Arm CPUs. On Graviton 4,
    // that approach results in 17 GB/s of throughput, compared to 39 GB/s for `i8`
    // dot-products.
    //
    // Luckily we can use the `vabdq_s8` which technically returns `i8` values, but it's a
    // matter of reinterpret-casting! That approach boosts us to 33 GB/s of throughput.
    uint32x4_t distance_sq_u32x4 = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a + i);
        int8x16_t b_i8x16 = vld1q_s8(b + i);
        uint8x16_t diff_u8x16 = vreinterpretq_u8_s8(vabdq_s8(a_i8x16, b_i8x16));
        distance_sq_u32x4 = vdotq_u32(distance_sq_u32x4, diff_u8x16, diff_u8x16);
    }
    simsimd_u32_t distance_sq_u32 = vaddvq_u32(distance_sq_u32x4);
    for (; i < n; ++i) {
        simsimd_i32_t diff_i32 = (simsimd_i32_t)a[i] - b[i];
        distance_sq_u32 += (simsimd_u32_t)(diff_i32 * diff_i32);
    }
    *result = distance_sq_u32;
}

SIMSIMD_PUBLIC void simsimd_angular_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {

    simsimd_size_t i = 0;

    // Variant 1.
    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    //
    //  int32x4_t ab_vec = vdupq_n_s32(0);
    //  int32x4_t a2_vec = vdupq_n_s32(0);
    //  int32x4_t b2_vec = vdupq_n_s32(0);
    //  for (simsimd_size_t i = 0; i != n; i += 8) {
    //      int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //      int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //      int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //      int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //      int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //      ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(ab_part_vec))));
    //      a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(a2_part_vec))));
    //      b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(b2_part_vec))));
    //  }
    //
    // Variant 2.
    // With the 128-bit `vdotq_s32` intrinsic, we can use the following code:
    //
    //  for (; i + 16 <= n; i += 16) {
    //      int8x16_t a_vec = vld1q_s8(a + i);
    //      int8x16_t b_vec = vld1q_s8(b + i);
    //      ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    //      a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
    //      b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    //  }
    //
    // Variant 3.
    // To use MMLA instructions, we need to reorganize the contents of the vectors.
    // On input we have `a_vec` and `b_vec`:
    //
    //   a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]
    //   b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //
    // We will be multiplying matrices of size 2x8 and 8x2. So we need to perform a few shuffles:
    //
    //   X =
    //      a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    //      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    //   Y =
    //      a[0], b[0],
    //      a[1], b[1],
    //      a[2], b[2],
    //      a[3], b[3],
    //      a[4], b[4],
    //      a[5], b[5],
    //      a[6], b[6],
    //      a[7], b[7]
    //
    //   V =
    //      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
    //      b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //   W =
    //      a[8],   b[8],
    //      a[9],   b[9],
    //      a[10],  b[10],
    //      a[11],  b[11],
    //      a[12],  b[12],
    //      a[13],  b[13],
    //      a[14],  b[14],
    //      a[15],  b[15]
    //
    // Performing matrix multiplications we can aggregate into a matrix `products_low_vec` and `products_high_vec`:
    //
    //      X * X, X * Y                V * W, V * V
    //      Y * X, Y * Y                W * W, W * V
    //
    // Of those values we need only 3/4, as the (X * Y) and (Y * X) are the same.
    //
    //      int32x4_t products_low_vec = vdupq_n_s32(0), products_high_vec = vdupq_n_s32(0);
    //      int8x16_t a_low_b_low_vec, a_high_b_high_vec;
    //      for (; i + 16 <= n; i += 16) {
    //          int8x16_t a_vec = vld1q_s8(a + i);
    //          int8x16_t b_vec = vld1q_s8(b + i);
    //          int8x16x2_t y_w_vecs = vzipq_s8(a_vec, b_vec);
    //          int8x16_t x_vec = vcombine_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
    //          int8x16_t v_vec = vcombine_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
    //          products_low_vec = vmmlaq_s32(products_low_vec, x_vec, y_w_vecs.val[0]);
    //          products_high_vec = vmmlaq_s32(products_high_vec, v_vec, y_w_vecs.val[1]);
    //      }
    //      int32x4_t products_vec = vaddq_s32(products_high_vec, products_low_vec);
    //      simsimd_i32_t a2 = products_vec[0];
    //      simsimd_i32_t ab = products_vec[1];
    //      simsimd_i32_t b2 = products_vec[3];
    //
    // That solution is elegant, but it requires the additional `+i8mm` extension and is currently slower,
    // at least on AWS Graviton 3.
    int32x4_t dot_product_i32x4 = vdupq_n_s32(0);
    int32x4_t a_norm_sq_i32x4 = vdupq_n_s32(0);
    int32x4_t b_norm_sq_i32x4 = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a + i);
        int8x16_t b_i8x16 = vld1q_s8(b + i);
        dot_product_i32x4 = vdotq_s32(dot_product_i32x4, a_i8x16, b_i8x16);
        a_norm_sq_i32x4 = vdotq_s32(a_norm_sq_i32x4, a_i8x16, a_i8x16);
        b_norm_sq_i32x4 = vdotq_s32(b_norm_sq_i32x4, b_i8x16, b_i8x16);
    }
    simsimd_i32_t dot_product_i32 = vaddvq_s32(dot_product_i32x4);
    simsimd_i32_t a_norm_sq_i32 = vaddvq_s32(a_norm_sq_i32x4);
    simsimd_i32_t b_norm_sq_i32 = vaddvq_s32(b_norm_sq_i32x4);

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = _simsimd_angular_normalize_f32_neon(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

SIMSIMD_PUBLIC void simsimd_l2_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
    simsimd_u32_t d2;
    simsimd_l2sq_u8_neon(a, b, n, &d2);
    *result = _simsimd_sqrt_f32_neon((simsimd_f32_t)d2);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                         simsimd_u32_t *result) {
    uint32x4_t distance_sq_u32x4 = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a + i);
        uint8x16_t b_u8x16 = vld1q_u8(b + i);
        uint8x16_t diff_u8x16 = vabdq_u8(a_u8x16, b_u8x16);
        distance_sq_u32x4 = vdotq_u32(distance_sq_u32x4, diff_u8x16, diff_u8x16);
    }
    simsimd_u32_t distance_sq_u32 = vaddvq_u32(distance_sq_u32x4);
    for (; i < n; ++i) {
        simsimd_i32_t diff_i32 = (simsimd_i32_t)a[i] - b[i];
        distance_sq_u32 += (simsimd_u32_t)(diff_i32 * diff_i32);
    }
    *result = distance_sq_u32;
}

SIMSIMD_PUBLIC void simsimd_angular_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {

    simsimd_size_t i = 0;
    uint32x4_t ab_vec = vdupq_n_u32(0);
    uint32x4_t a2_vec = vdupq_n_u32(0);
    uint32x4_t b2_vec = vdupq_n_u32(0);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        ab_vec = vdotq_u32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_u32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_u32(b2_vec, b_vec, b_vec);
    }
    simsimd_u32_t ab = vaddvq_u32(ab_vec);
    simsimd_u32_t a2 = vaddvq_u32(a2_vec);
    simsimd_u32_t b2 = vaddvq_u32(b2_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_u32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = _simsimd_angular_normalize_f32_neon(ab, a2, b2);
}

typedef simsimd_dot_i8x16_state_neon_t simsimd_angular_i8x16_state_neon_t;
SIMSIMD_INTERNAL void simsimd_angular_i8x16_init_neon(simsimd_angular_i8x16_state_neon_t *state) {
    simsimd_dot_i8x16_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x16_update_neon(simsimd_angular_i8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_dot_i8x16_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x16_finalize_neon(simsimd_angular_i8x16_state_neon_t const *state_a,
                                                          simsimd_angular_i8x16_state_neon_t const *state_b,
                                                          simsimd_angular_i8x16_state_neon_t const *state_c,
                                                          simsimd_angular_i8x16_state_neon_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x16_finalize_neon(state_a, state_b, state_c, state_d, dots_i32);

    int32x4_t dots_i32x4 = vld1q_s32(dots_i32);
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots_i32x4);
    float32x4_t query_norm_sq_f32x4 = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms_sq_f32x4 = {target_norm_a * target_norm_a, target_norm_b * target_norm_b,
                                         target_norm_c * target_norm_c, target_norm_d * target_norm_d};

    float32x4_t products_f32x4 = vmulq_f32(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with 2 Newton-Raphson iterations for better precision
    float32x4_t rsqrt_f32x4 = vrsqrteq_f32(products_f32x4);
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));

    float32x4_t ones_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t zeros_f32x4 = vdupq_n_f32(0.0f);
    float32x4_t result_f32x4 = vsubq_f32(ones_f32x4, vmulq_f32(dots_f32x4, rsqrt_f32x4));
    result_f32x4 = vmaxq_f32(result_f32x4, zeros_f32x4);

    // Edge case handling with vectorized selects
    uint32x4_t products_zero_u32x4 = vceqq_f32(products_f32x4, zeros_f32x4);
    uint32x4_t dots_zero_u32x4 = vceqq_f32(dots_f32x4, zeros_f32x4);
    uint32x4_t both_zero_u32x4 = vandq_u32(products_zero_u32x4, dots_zero_u32x4);
    result_f32x4 = vbslq_f32(both_zero_u32x4, zeros_f32x4, result_f32x4);
    uint32x4_t prod_zero_dot_nonzero_u32x4 = vandq_u32(products_zero_u32x4, vmvnq_u32(dots_zero_u32x4));
    result_f32x4 = vbslq_f32(prod_zero_dot_nonzero_u32x4, ones_f32x4, result_f32x4);

    vst1q_f32(results, result_f32x4);
}

typedef simsimd_dot_i8x16_state_neon_t simsimd_l2_i8x16_state_neon_t;
SIMSIMD_INTERNAL void simsimd_l2_i8x16_init_neon(simsimd_l2_i8x16_state_neon_t *state) {
    simsimd_dot_i8x16_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x16_update_neon(simsimd_l2_i8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                   simsimd_b128_vec_t b) {
    simsimd_dot_i8x16_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x16_finalize_neon(simsimd_l2_i8x16_state_neon_t const *state_a,
                                                     simsimd_l2_i8x16_state_neon_t const *state_b,
                                                     simsimd_l2_i8x16_state_neon_t const *state_c,
                                                     simsimd_l2_i8x16_state_neon_t const *state_d,
                                                     simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                     simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                     simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x16_finalize_neon(state_a, state_b, state_c, state_d, dots_i32);

    int32x4_t dots_i32x4 = vld1q_s32(dots_i32);
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots_i32x4);
    float32x4_t query_norm_sq_f32x4 = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms_sq_f32x4 = {target_norm_a * target_norm_a, target_norm_b * target_norm_b,
                                         target_norm_c * target_norm_c, target_norm_d * target_norm_d};

    // L2 distance: sqrt(query_sq + target_sq - 2*dot) using FMA
    float32x4_t neg_two_f32x4 = vdupq_n_f32(-2.0f);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_norm_sq_f32x4, target_norms_sq_f32x4);
    float32x4_t dist_sq_f32x4 = vfmaq_f32(sum_sq_f32x4, neg_two_f32x4, dots_f32x4);

    float32x4_t zeros_f32x4 = vdupq_n_f32(0.0f);
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, zeros_f32x4);
    float32x4_t dist_f32x4 = vsqrtq_f32(dist_sq_f32x4);

    vst1q_f32(results, dist_f32x4);
}

typedef simsimd_dot_u8x16_state_neon_t simsimd_angular_u8x16_state_neon_t;
SIMSIMD_INTERNAL void simsimd_angular_u8x16_init_neon(simsimd_angular_u8x16_state_neon_t *state) {
    simsimd_dot_u8x16_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x16_update_neon(simsimd_angular_u8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_dot_u8x16_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x16_finalize_neon(simsimd_angular_u8x16_state_neon_t const *state_a,
                                                          simsimd_angular_u8x16_state_neon_t const *state_b,
                                                          simsimd_angular_u8x16_state_neon_t const *state_c,
                                                          simsimd_angular_u8x16_state_neon_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x16_finalize_neon(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 and build vectors for parallel processing
    uint32x4_t dots_u32x4 = vld1q_u32(dots_u32);
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots_u32x4);
    float32x4_t query_norm_sq_f32x4 = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms_sq_f32x4 = {target_norm_a * target_norm_a, target_norm_b * target_norm_b,
                                         target_norm_c * target_norm_c, target_norm_d * target_norm_d};

    // products = query_norm_sq * target_norms_sq
    float32x4_t products_f32x4 = vmulq_f32(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement
    float32x4_t rsqrt_f32x4 = vrsqrteq_f32(products_f32x4);
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));

    // normalized = dots * rsqrt(products)
    float32x4_t normalized_f32x4 = vmulq_f32(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    float32x4_t ones_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t angular_f32x4 = vsubq_f32(ones_f32x4, normalized_f32x4);

    // Store results
    vst1q_f32(results, angular_f32x4);
}

typedef simsimd_dot_u8x16_state_neon_t simsimd_l2_u8x16_state_neon_t;
SIMSIMD_INTERNAL void simsimd_l2_u8x16_init_neon(simsimd_l2_u8x16_state_neon_t *state) {
    simsimd_dot_u8x16_init_neon(state);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x16_update_neon(simsimd_l2_u8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                   simsimd_b128_vec_t b) {
    simsimd_dot_u8x16_update_neon(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x16_finalize_neon(simsimd_l2_u8x16_state_neon_t const *state_a,
                                                     simsimd_l2_u8x16_state_neon_t const *state_b,
                                                     simsimd_l2_u8x16_state_neon_t const *state_c,
                                                     simsimd_l2_u8x16_state_neon_t const *state_d,
                                                     simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                     simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                     simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x16_finalize_neon(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 and build vectors for parallel processing
    uint32x4_t dots_u32x4 = vld1q_u32(dots_u32);
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots_u32x4);
    float32x4_t query_norm_sq_f32x4 = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms_sq_f32x4 = {target_norm_a * target_norm_a, target_norm_b * target_norm_b,
                                         target_norm_c * target_norm_c, target_norm_d * target_norm_d};

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    float32x4_t two_f32x4 = vdupq_n_f32(2.0f);
    float32x4_t two_dots_f32x4 = vmulq_f32(two_f32x4, dots_f32x4);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_norm_sq_f32x4, target_norms_sq_f32x4);
    float32x4_t dist_sq_f32x4 = vsubq_f32(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    float32x4_t zeros_f32x4 = vdupq_n_f32(0.0f);
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, zeros_f32x4);
    float32x4_t dist_f32x4 = vsqrtq_f32(dist_sq_f32x4);

    // Store results
    vst1q_f32(results, dist_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_I8

#if SIMSIMD_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
    simsimd_l2sq_f32_sve(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        svfloat32_t a_minus_b_vec = svsub_f32_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntw();
    } while (i < n);
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_angular_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntw();
    } while (i < n);

    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = (simsimd_f32_t)_simsimd_angular_normalize_f64_neon(ab, a2, b2);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                       simsimd_f64_t *result) {
    simsimd_l2sq_f64_sve(a, b, n, result);
    *result = _simsimd_sqrt_f64_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                         simsimd_f64_t *result) {
    simsimd_size_t i = 0;
    svfloat64_t d2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        svfloat64_t a_minus_b_vec = svsub_f64_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f64_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntd();
    } while (i < n);
    simsimd_f64_t d2 = svaddv_f64(svptrue_b32(), d2_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_angular_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_f64_t *result) {
    simsimd_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f64_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f64_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntd();
    } while (i < n);

    simsimd_f64_t ab = svaddv_f64(svptrue_b32(), ab_vec);
    simsimd_f64_t a2 = svaddv_f64(svptrue_b32(), a2_vec);
    simsimd_f64_t b2 = svaddv_f64(svptrue_b32(), b2_vec);
    *result = _simsimd_angular_normalize_f64_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_SVE_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_sve(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
    simsimd_l2sq_f16_sve(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sve(simsimd_f16_t const *a_enum, simsimd_f16_t const *b_enum, simsimd_size_t n,
                                         simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat16_t d2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const *a = (simsimd_f16_for_arm_simd_t const *)(a_enum);
    simsimd_f16_for_arm_simd_t const *b = (simsimd_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        svfloat16_t a_minus_b_vec = svsub_f16_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f16_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f16_for_arm_simd_t d2_f16 = svaddv_f16(svptrue_b16(), d2_vec);
    *result = d2_f16;
}

SIMSIMD_PUBLIC void simsimd_angular_f16_sve(simsimd_f16_t const *a_enum, simsimd_f16_t const *b_enum, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t a2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t b2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    simsimd_f16_for_arm_simd_t const *a = (simsimd_f16_for_arm_simd_t const *)(a_enum);
    simsimd_f16_for_arm_simd_t const *b = (simsimd_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f16_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f16_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    simsimd_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    simsimd_f16_for_arm_simd_t a2 = svaddv_f16(svptrue_b16(), a2_vec);
    simsimd_f16_for_arm_simd_t b2 = svaddv_f16(svptrue_b16(), b2_vec);
    *result = _simsimd_angular_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE_F16

#if SIMSIMD_TARGET_SVE_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_bf16_sve(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
    simsimd_l2sq_bf16_sve(a, b, n, result);
    *result = _simsimd_sqrt_f32_neon(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_sve(simsimd_bf16_t const *a_enum, simsimd_bf16_t const *b_enum, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t d2_low_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t d2_high_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    simsimd_u16_t const *a = (simsimd_u16_t const *)(a_enum);
    simsimd_u16_t const *b = (simsimd_u16_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svuint16_t a_vec = svld1_u16(pg_vec, a + i);
        svuint16_t b_vec = svld1_u16(pg_vec, b + i);

        // There is no `bf16` subtraction in SVE, so we need to convert to `u32` and shift.
        svbool_t pg_low_vec = svwhilelt_b32((unsigned int)(i), (unsigned int)n);
        svbool_t pg_high_vec = svwhilelt_b32((unsigned int)(i + svcnth() / 2), (unsigned int)n);
        svfloat32_t a_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(a_vec), 16));
        svfloat32_t a_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(a_vec), 16));
        svfloat32_t b_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(b_vec), 16));
        svfloat32_t b_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(b_vec), 16));

        svfloat32_t a_minus_b_low_vec = svsub_f32_x(pg_low_vec, a_low_vec, b_low_vec);
        svfloat32_t a_minus_b_high_vec = svsub_f32_x(pg_high_vec, a_high_vec, b_high_vec);
        d2_low_vec = svmla_f32_x(pg_vec, d2_low_vec, a_minus_b_low_vec, a_minus_b_low_vec);
        d2_high_vec = svmla_f32_x(pg_vec, d2_high_vec, a_minus_b_high_vec, a_minus_b_high_vec);
        i += svcnth();
    } while (i < n);
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_low_vec) + svaddv_f32(svptrue_b32(), d2_high_vec);
    *result = d2;
}

SIMSIMD_PUBLIC void simsimd_angular_bf16_sve(simsimd_bf16_t const *a_enum, simsimd_bf16_t const *b_enum,
                                             simsimd_size_t n, simsimd_f32_t *result) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    simsimd_bf16_for_arm_simd_t const *a = (simsimd_bf16_for_arm_simd_t const *)(a_enum);
    simsimd_bf16_for_arm_simd_t const *b = (simsimd_bf16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svbfloat16_t a_vec = svld1_bf16(pg_vec, a + i);
        svbfloat16_t b_vec = svld1_bf16(pg_vec, b + i);
        ab_vec = svbfdot_f32(ab_vec, a_vec, b_vec);
        a2_vec = svbfdot_f32(a2_vec, a_vec, a_vec);
        b2_vec = svbfdot_f32(b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = _simsimd_angular_normalize_f32_neon(ab, a2, b2);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE_BF16
#endif // _SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_f32_t _simsimd_sqrt_f32_haswell(simsimd_f32_t x) {
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x)));
}
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_sqrt_f64_haswell(simsimd_f64_t x) {
    return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x)));
}

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_angular_normalize_f64_haswell(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                      simsimd_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0)
        return 1;
    // We want to avoid the `simsimd_f32_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The latency of the native instruction is 4 cycles and it's broadly supported.
    // For single-precision floats it has a maximum relative error of 1.5*2^-12.
    // Higher precision isn't implemented on older CPUs. See `_simsimd_angular_normalize_f64_skylake` for that.
    __m128d squares = _mm_set_pd(a2, b2);
    __m128d rsqrts = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(squares)));
    // Newton-Raphson iteration for reciprocal square root:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    simsimd_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    simsimd_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    simsimd_f64_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

SIMSIMD_INTERNAL simsimd_f32_t _simsimd_angular_normalize_f32_haswell(simsimd_f32_t ab, simsimd_f32_t a2,
                                                                      simsimd_f32_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0.0f)
        return 1.0f;

    // Load the squares into an __m128 register for single-precision floating-point operations
    __m128 squares = _mm_set_ps(a2, b2, a2, b2); // We replicate to make use of full register

    // Compute the reciprocal square root of the squares using `_mm_rsqrt_ps` (single-precision)
    __m128 rsqrts = _mm_rsqrt_ps(squares);

    // Perform one iteration of Newton-Raphson refinement to improve the precision of rsqrt:
    // Formula: y' = y * (1.5 - 0.5 * x * y * y)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_halves = _mm_set1_ps(1.5f);
    rsqrts = _mm_mul_ps(rsqrts,
                        _mm_sub_ps(three_halves, _mm_mul_ps(half, _mm_mul_ps(squares, _mm_mul_ps(rsqrts, rsqrts)))));

    // Extract the reciprocal square roots of a2 and b2 from the __m128 register
    simsimd_f32_t a2_reciprocal = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    simsimd_f32_t b2_reciprocal = _mm_cvtss_f32(rsqrts);

    // Calculate the angular distance: 1 - dot_product * a2_reciprocal * b2_reciprocal
    simsimd_f32_t result = 1.0f - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL
#endif // _SIMSIMD_TARGET_X86

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {
    simsimd_l2sq_f16_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

simsimd_l2sq_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = _simsimd_partial_load_f16x8_haswell(a, n);
        b_f32x8 = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    if (n) goto simsimd_l2sq_f16_haswell_cycle;

    *result = _simsimd_reduce_add_f32x8_haswell(distance_sq_f32x8);
}

SIMSIMD_PUBLIC void simsimd_angular_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();

simsimd_angular_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = _simsimd_partial_load_f16x8_haswell(a, n);
        b_f32x8 = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    if (n) goto simsimd_angular_f16_haswell_cycle;

    simsimd_f32_t dot_product_f32 = _simsimd_reduce_add_f32x8_haswell(dot_product_f32x8);
    simsimd_f32_t a_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(a_norm_sq_f32x8);
    simsimd_f32_t b_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(b_norm_sq_f32x8);
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

SIMSIMD_PUBLIC void simsimd_l2_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_l2sq_bf16_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

simsimd_l2sq_bf16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(a, n));
        b_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(b, n));
        n = 0;
    }
    else {
        a_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    if (n) goto simsimd_l2sq_bf16_haswell_cycle;

    *result = _simsimd_reduce_add_f32x8_haswell(distance_sq_f32x8);
}

SIMSIMD_PUBLIC void simsimd_angular_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                                 simsimd_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();

simsimd_angular_bf16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(a, n));
        b_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_simsimd_partial_load_bf16x8_haswell(b, n));
        n = 0;
    }
    else {
        a_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    if (n) goto simsimd_angular_bf16_haswell_cycle;

    simsimd_f32_t dot_product_f32 = _simsimd_reduce_add_f32x8_haswell(dot_product_f32x8);
    simsimd_f32_t a_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(a_norm_sq_f32x8);
    simsimd_f32_t b_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(b_norm_sq_f32x8);
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

SIMSIMD_PUBLIC void simsimd_l2_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    simsimd_u32_t distance_sq_u32;
    simsimd_l2sq_i8_haswell(a, b, n, &distance_sq_u32);
    *result = _simsimd_sqrt_f32_haswell((simsimd_f32_t)distance_sq_u32);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                            simsimd_u32_t *result) {

    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Sign extend `i8` to `i16`
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8x32));
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8x32));
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));

        // Subtract
        // After this we will be squaring the values. The sign will be dropped
        // and each difference will be in the range [0, 255].
        __m256i diff_low_i16x16 = _mm256_sub_epi16(a_low_i16x16, b_low_i16x16);
        __m256i diff_high_i16x16 = _mm256_sub_epi16(a_high_i16x16, b_high_i16x16);

        // Accumulate into `i32` vectors
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Accumulate the 32-bit integers from `distance_sq_high_i32x8` and `distance_sq_low_i32x8`
    simsimd_i32_t distance_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t diff_i32 = (simsimd_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (simsimd_u32_t)distance_sq_i32;
}

SIMSIMD_PUBLIC void simsimd_angular_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *result) {

    __m256i dot_product_low_i32x8 = _mm256_setzero_si256();
    __m256i dot_product_high_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_high_i32x8 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Unpack `int8` to `int16`
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 0));
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 0));
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));

        // Multiply and accumulate as `int16`, accumulate products as `int32`:
        dot_product_low_i32x8 = _mm256_add_epi32(dot_product_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        dot_product_high_i32x8 = _mm256_add_epi32(dot_product_high_i32x8,
                                                  _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
        a_norm_sq_low_i32x8 = _mm256_add_epi32(a_norm_sq_low_i32x8, _mm256_madd_epi16(a_low_i16x16, a_low_i16x16));
        a_norm_sq_high_i32x8 = _mm256_add_epi32(a_norm_sq_high_i32x8, _mm256_madd_epi16(a_high_i16x16, a_high_i16x16));
        b_norm_sq_low_i32x8 = _mm256_add_epi32(b_norm_sq_low_i32x8, _mm256_madd_epi16(b_low_i16x16, b_low_i16x16));
        b_norm_sq_high_i32x8 = _mm256_add_epi32(b_norm_sq_high_i32x8, _mm256_madd_epi16(b_high_i16x16, b_high_i16x16));
    }

    // Further reduce to a single sum for each vector
    simsimd_i32_t dot_product_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(dot_product_low_i32x8, dot_product_high_i32x8));
    simsimd_i32_t a_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(a_norm_sq_low_i32x8, a_norm_sq_high_i32x8));
    simsimd_i32_t b_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(b_norm_sq_low_i32x8, b_norm_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = _simsimd_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

SIMSIMD_PUBLIC void simsimd_l2_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    simsimd_u32_t distance_sq_u32;
    simsimd_l2sq_u8_haswell(a, b, n, &distance_sq_u32);
    *result = _simsimd_sqrt_f32_haswell((simsimd_f32_t)distance_sq_u32);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                            simsimd_u32_t *result) {

    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Substracting unsigned vectors in AVX2 is done by saturating subtraction:
        __m256i diff_u8x32 = _mm256_or_si256(_mm256_subs_epu8(a_u8x32, b_u8x32), _mm256_subs_epu8(b_u8x32, a_u8x32));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i diff_low_i16x16 = _mm256_unpacklo_epi8(diff_u8x32, zeros_i8x32);
        __m256i diff_high_i16x16 = _mm256_unpackhi_epi8(diff_u8x32, zeros_i8x32);

        // Multiply and accumulate at `int16` level, accumulate at `int32` level:
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Accumulate the 32-bit integers from `distance_sq_high_i32x8` and `distance_sq_low_i32x8`
    simsimd_i32_t distance_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t diff_i32 = (simsimd_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (simsimd_u32_t)distance_sq_i32;
}

SIMSIMD_PUBLIC void simsimd_angular_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *result) {

    __m256i dot_product_low_i32x8 = _mm256_setzero_si256();
    __m256i dot_product_high_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);

        // Multiply and accumulate as `int16`, accumulate products as `int32`
        dot_product_low_i32x8 = _mm256_add_epi32(dot_product_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        dot_product_high_i32x8 = _mm256_add_epi32(dot_product_high_i32x8,
                                                  _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
        a_norm_sq_low_i32x8 = _mm256_add_epi32(a_norm_sq_low_i32x8, _mm256_madd_epi16(a_low_i16x16, a_low_i16x16));
        a_norm_sq_high_i32x8 = _mm256_add_epi32(a_norm_sq_high_i32x8, _mm256_madd_epi16(a_high_i16x16, a_high_i16x16));
        b_norm_sq_low_i32x8 = _mm256_add_epi32(b_norm_sq_low_i32x8, _mm256_madd_epi16(b_low_i16x16, b_low_i16x16));
        b_norm_sq_high_i32x8 = _mm256_add_epi32(b_norm_sq_high_i32x8, _mm256_madd_epi16(b_high_i16x16, b_high_i16x16));
    }

    // Further reduce to a single sum for each vector
    simsimd_i32_t dot_product_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(dot_product_low_i32x8, dot_product_high_i32x8));
    simsimd_i32_t a_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(a_norm_sq_low_i32x8, a_norm_sq_high_i32x8));
    simsimd_i32_t b_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(b_norm_sq_low_i32x8, b_norm_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = _simsimd_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

SIMSIMD_PUBLIC void simsimd_l2_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {
    simsimd_l2sq_f32_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {

    __m256 distance_sq_f32x8 = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    }

    simsimd_f64_t distance_sq_f64 = _simsimd_reduce_add_f32x8_haswell(distance_sq_f32x8);
    for (; i < n; ++i) {
        simsimd_f32_t diff_f32 = a[i] - b[i];
        distance_sq_f64 += diff_f32 * diff_f32;
    }

    *result = distance_sq_f64;
}

SIMSIMD_PUBLIC void simsimd_angular_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *result) {

    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    }

    simsimd_f32_t dot_product_f32 = _simsimd_reduce_add_f32x8_haswell(dot_product_f32x8);
    simsimd_f32_t a_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(a_norm_sq_f32x8);
    simsimd_f32_t b_norm_sq_f32 = _simsimd_reduce_add_f32x8_haswell(b_norm_sq_f32x8);
    for (; i < n; ++i) {
        simsimd_f32_t a_element_f32 = a[i], b_element_f32 = b[i];
        dot_product_f32 += a_element_f32 * b_element_f32;
        a_norm_sq_f32 += a_element_f32 * a_element_f32;
        b_norm_sq_f32 += b_element_f32 * b_element_f32;
    }
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                           simsimd_f64_t *result) {
    simsimd_l2sq_f64_haswell(a, b, n, result);
    *result = _simsimd_sqrt_f64_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *result) {

    __m256d distance_sq_f64x4 = _mm256_setzero_pd();
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d diff_f64x4 = _mm256_sub_pd(a_f64x4, b_f64x4);
        distance_sq_f64x4 = _mm256_fmadd_pd(diff_f64x4, diff_f64x4, distance_sq_f64x4);
    }

    simsimd_f64_t distance_sq_f64 = _simsimd_reduce_add_f64x4_haswell(distance_sq_f64x4);
    for (; i < n; ++i) {
        simsimd_f64_t diff_f64 = a[i] - b[i];
        distance_sq_f64 += diff_f64 * diff_f64;
    }

    *result = distance_sq_f64;
}

SIMSIMD_PUBLIC void simsimd_angular_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                                simsimd_f64_t *result) {

    __m256d dot_product_f64x4 = _mm256_setzero_pd();
    __m256d a_norm_sq_f64x4 = _mm256_setzero_pd();
    __m256d b_norm_sq_f64x4 = _mm256_setzero_pd();
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        dot_product_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, dot_product_f64x4);
        a_norm_sq_f64x4 = _mm256_fmadd_pd(a_f64x4, a_f64x4, a_norm_sq_f64x4);
        b_norm_sq_f64x4 = _mm256_fmadd_pd(b_f64x4, b_f64x4, b_norm_sq_f64x4);
    }

    simsimd_f64_t dot_product_f64 = _simsimd_reduce_add_f64x4_haswell(dot_product_f64x4);
    simsimd_f64_t a_norm_sq_f64 = _simsimd_reduce_add_f64x4_haswell(a_norm_sq_f64x4);
    simsimd_f64_t b_norm_sq_f64 = _simsimd_reduce_add_f64x4_haswell(b_norm_sq_f64x4);
    for (; i < n; ++i) {
        simsimd_f64_t a_element_f64 = a[i], b_element_f64 = b[i];
        dot_product_f64 += a_element_f64 * b_element_f64;
        a_norm_sq_f64 += a_element_f64 * a_element_f64;
        b_norm_sq_f64 += b_element_f64 * b_element_f64;
    }
    *result = _simsimd_angular_normalize_f64_haswell(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

typedef simsimd_dot_f32x8_state_haswell_t simsimd_angular_f32x8_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_angular_f32x8_init_haswell(simsimd_angular_f32x8_state_haswell_t *state) {
    simsimd_dot_f32x8_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x8_update_haswell(simsimd_angular_f32x8_state_haswell_t *state,
                                                           simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    simsimd_dot_f32x8_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x8_finalize_haswell(simsimd_angular_f32x8_state_haswell_t const *state_a,
                                                             simsimd_angular_f32x8_state_haswell_t const *state_b,
                                                             simsimd_angular_f32x8_state_haswell_t const *state_c,
                                                             simsimd_angular_f32x8_state_haswell_t const *state_d,
                                                             simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                             simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                             simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f32x8_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel processing
    __m256d dots_f64x4 = _mm256_set_pd((simsimd_f64_t)dots[3], (simsimd_f64_t)dots[2], //
                                       (simsimd_f64_t)dots[1], (simsimd_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((simsimd_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((simsimd_f64_t)target_norm_d, (simsimd_f64_t)target_norm_c,
                                               (simsimd_f64_t)target_norm_b, (simsimd_f64_t)target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Vectorized rsqrt with Newton-Raphson (approximate rsqrt not available for f64)
    // Use full sqrt: 1.0 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results
    simsimd_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, angular_f64x4);
    results[0] = (simsimd_f32_t)u.f64s[0];
    results[1] = (simsimd_f32_t)u.f64s[1];
    results[2] = (simsimd_f32_t)u.f64s[2];
    results[3] = (simsimd_f32_t)u.f64s[3];
}

typedef simsimd_dot_f32x8_state_haswell_t simsimd_l2_f32x8_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_l2_f32x8_init_haswell(simsimd_l2_f32x8_state_haswell_t *state) {
    simsimd_dot_f32x8_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x8_update_haswell(simsimd_l2_f32x8_state_haswell_t *state, simsimd_b256_vec_t a,
                                                      simsimd_b256_vec_t b) {
    simsimd_dot_f32x8_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x8_finalize_haswell(simsimd_l2_f32x8_state_haswell_t const *state_a,
                                                        simsimd_l2_f32x8_state_haswell_t const *state_b,
                                                        simsimd_l2_f32x8_state_haswell_t const *state_c,
                                                        simsimd_l2_f32x8_state_haswell_t const *state_d,
                                                        simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                        simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                        simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f32x8_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_set_pd((simsimd_f64_t)dots[3], (simsimd_f64_t)dots[2], //
                                       (simsimd_f64_t)dots[1], (simsimd_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((simsimd_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((simsimd_f64_t)target_norm_d, (simsimd_f64_t)target_norm_c,
                                               (simsimd_f64_t)target_norm_b, (simsimd_f64_t)target_norm_a);

    // Compute squared norms in parallel: q² and t² vectors
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results (convert f64 → f32)
    simsimd_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, dist_f64x4);
    results[0] = (simsimd_f32_t)u.f64s[0];
    results[1] = (simsimd_f32_t)u.f64s[1];
    results[2] = (simsimd_f32_t)u.f64s[2];
    results[3] = (simsimd_f32_t)u.f64s[3];
}

typedef simsimd_dot_f16x16_state_haswell_t simsimd_angular_f16x16_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_angular_f16x16_init_haswell(simsimd_angular_f16x16_state_haswell_t *state) {
    simsimd_dot_f16x16_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x16_update_haswell(simsimd_angular_f16x16_state_haswell_t *state,
                                                            simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    simsimd_dot_f16x16_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x16_finalize_haswell(simsimd_angular_f16x16_state_haswell_t const *state_a,
                                                              simsimd_angular_f16x16_state_haswell_t const *state_b,
                                                              simsimd_angular_f16x16_state_haswell_t const *state_c,
                                                              simsimd_angular_f16x16_state_haswell_t const *state_d,
                                                              simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                              simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                              simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing (F16 uses F32 accumulation)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement for F32
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_f16x16_state_haswell_t simsimd_l2_f16x16_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_l2_f16x16_init_haswell(simsimd_l2_f16x16_state_haswell_t *state) {
    simsimd_dot_f16x16_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x16_update_haswell(simsimd_l2_f16x16_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b) {
    simsimd_dot_f16x16_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x16_finalize_haswell(simsimd_l2_f16x16_state_haswell_t const *state_a,
                                                         simsimd_l2_f16x16_state_haswell_t const *state_b,
                                                         simsimd_l2_f16x16_state_haswell_t const *state_c,
                                                         simsimd_l2_f16x16_state_haswell_t const *state_d,
                                                         simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                         simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                         simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef simsimd_dot_bf16x16_state_haswell_t simsimd_angular_bf16x16_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_angular_bf16x16_init_haswell(simsimd_angular_bf16x16_state_haswell_t *state) {
    simsimd_dot_bf16x16_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x16_update_haswell(simsimd_angular_bf16x16_state_haswell_t *state,
                                                             simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    simsimd_dot_bf16x16_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x16_finalize_haswell(simsimd_angular_bf16x16_state_haswell_t const *state_a,
                                                               simsimd_angular_bf16x16_state_haswell_t const *state_b,
                                                               simsimd_angular_bf16x16_state_haswell_t const *state_c,
                                                               simsimd_angular_bf16x16_state_haswell_t const *state_d,
                                                               simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                               simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                               simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing (BF16 uses F32 accumulation)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement for F32
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_bf16x16_state_haswell_t simsimd_l2_bf16x16_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_l2_bf16x16_init_haswell(simsimd_l2_bf16x16_state_haswell_t *state) {
    simsimd_dot_bf16x16_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x16_update_haswell(simsimd_l2_bf16x16_state_haswell_t *state, simsimd_b256_vec_t a,
                                                        simsimd_b256_vec_t b) {
    simsimd_dot_bf16x16_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x16_finalize_haswell(simsimd_l2_bf16x16_state_haswell_t const *state_a,
                                                          simsimd_l2_bf16x16_state_haswell_t const *state_b,
                                                          simsimd_l2_bf16x16_state_haswell_t const *state_c,
                                                          simsimd_l2_bf16x16_state_haswell_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef simsimd_dot_i8x32_state_haswell_t simsimd_angular_i8x32_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_angular_i8x32_init_haswell(simsimd_angular_i8x32_state_haswell_t *state) {
    simsimd_dot_i8x32_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x32_update_haswell(simsimd_angular_i8x32_state_haswell_t *state,
                                                           simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    simsimd_dot_i8x32_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x32_finalize_haswell(simsimd_angular_i8x32_state_haswell_t const *state_a,
                                                             simsimd_angular_i8x32_state_haswell_t const *state_b,
                                                             simsimd_angular_i8x32_state_haswell_t const *state_c,
                                                             simsimd_angular_i8x32_state_haswell_t const *state_d,
                                                             simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                             simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                             simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (i32 output)
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_i32);

    // Build 128-bit F32 vectors for parallel processing (convert i32 → f32)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_i32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_i8x32_state_haswell_t simsimd_l2_i8x32_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_l2_i8x32_init_haswell(simsimd_l2_i8x32_state_haswell_t *state) {
    simsimd_dot_i8x32_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x32_update_haswell(simsimd_l2_i8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                      simsimd_b256_vec_t b) {
    simsimd_dot_i8x32_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x32_finalize_haswell(simsimd_l2_i8x32_state_haswell_t const *state_a,
                                                        simsimd_l2_i8x32_state_haswell_t const *state_b,
                                                        simsimd_l2_i8x32_state_haswell_t const *state_c,
                                                        simsimd_l2_i8x32_state_haswell_t const *state_d,
                                                        simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                        simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                        simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (i32 output)
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_i32);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_i32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef simsimd_dot_u8x32_state_haswell_t simsimd_angular_u8x32_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_angular_u8x32_init_haswell(simsimd_angular_u8x32_state_haswell_t *state) {
    simsimd_dot_u8x32_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x32_update_haswell(simsimd_angular_u8x32_state_haswell_t *state,
                                                           simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    simsimd_dot_u8x32_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x32_finalize_haswell(simsimd_angular_u8x32_state_haswell_t const *state_a,
                                                             simsimd_angular_u8x32_state_haswell_t const *state_b,
                                                             simsimd_angular_u8x32_state_haswell_t const *state_c,
                                                             simsimd_angular_u8x32_state_haswell_t const *state_d,
                                                             simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                             simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                             simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (u32 output)
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_u32);

    // Build 128-bit F32 vectors for parallel processing (convert u32 → f32 via i32)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_u32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_u8x32_state_haswell_t simsimd_l2_u8x32_state_haswell_t;
SIMSIMD_INTERNAL void simsimd_l2_u8x32_init_haswell(simsimd_l2_u8x32_state_haswell_t *state) {
    simsimd_dot_u8x32_init_haswell(state);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x32_update_haswell(simsimd_l2_u8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                      simsimd_b256_vec_t b) {
    simsimd_dot_u8x32_update_haswell(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x32_finalize_haswell(simsimd_l2_u8x32_state_haswell_t const *state_a,
                                                        simsimd_l2_u8x32_state_haswell_t const *state_b,
                                                        simsimd_l2_u8x32_state_haswell_t const *state_c,
                                                        simsimd_l2_u8x32_state_haswell_t const *state_d,
                                                        simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                        simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                        simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (u32 output)
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_u32);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_u32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512bw", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512bw,avx512vl,bmi2"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {
    simsimd_l2sq_f32_skylake(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result) {
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

simsimd_l2sq_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
    d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n) goto simsimd_l2sq_f32_skylake_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(d2_vec);
}

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_angular_normalize_f64_skylake(simsimd_f64_t ab, simsimd_f64_t a2,
                                                                      simsimd_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0)
        return 1;

    // We want to avoid the `simsimd_f32_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The maximum relative error for this approximation is less than 2^-14, which is 6x lower than
    // for single-precision floats in the `_simsimd_angular_normalize_f64_haswell` implementation.
    // Mysteriously, MSVC has no `_mm_rsqrt14_pd` intrinsic, but has its masked variants,
    // so let's use `_mm_maskz_rsqrt14_pd(0xFF, ...)` instead.
    __m128d squares = _mm_set_pd(a2, b2);
    __m128d rsqrts = _mm_maskz_rsqrt14_pd(0xFF, squares);

    // Let's implement a single Newton-Raphson iteration to refine the result.
    // This is how it affects downstream applications for 1536-dimensional vectors:
    //
    //      DType     Baseline Error       Old SimSIMD Error    New SimSIMD Error
    //      bfloat16  1.89e-08 ± 1.59e-08  3.07e-07 ± 3.09e-07  3.53e-09 ± 2.70e-09
    //      float16   1.67e-02 ± 1.44e-02  2.68e-05 ± 1.95e-05  2.02e-05 ± 1.39e-05
    //      float32   2.21e-08 ± 1.65e-08  3.47e-07 ± 3.49e-07  3.77e-09 ± 2.84e-09
    //      float64   0.00e+00 ± 0.00e+00  3.80e-07 ± 4.50e-07  1.35e-11 ± 1.85e-11
    //      int8      0.00e+00 ± 0.00e+00  4.60e-04 ± 3.36e-04  4.20e-04 ± 4.88e-04
    //
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    simsimd_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    simsimd_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    simsimd_f64_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

SIMSIMD_PUBLIC void simsimd_angular_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                                simsimd_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero();
    __m512 a_norm_sq_f32x16 = _mm512_setzero();
    __m512 b_norm_sq_f32x16 = _mm512_setzero();
    __m512 a_f32x16, b_f32x16;

simsimd_angular_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    dot_product_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_product_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto simsimd_angular_f32_skylake_cycle;

    simsimd_f64_t dot_product_f64 = _simsimd_reduce_add_f32x16_skylake(dot_product_f32x16);
    simsimd_f64_t a_norm_sq_f64 = _simsimd_reduce_add_f32x16_skylake(a_norm_sq_f32x16);
    simsimd_f64_t b_norm_sq_f64 = _simsimd_reduce_add_f32x16_skylake(b_norm_sq_f32x16);
    *result = _simsimd_angular_normalize_f64_skylake(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

SIMSIMD_PUBLIC void simsimd_l2_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                           simsimd_f64_t *result) {
    simsimd_l2sq_f64_skylake(a, b, n, result);
    *result = _simsimd_sqrt_f64_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *result) {
    __m512d distance_sq_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

simsimd_l2sq_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    distance_sq_f64x8 = _mm512_fmadd_pd(diff_f64x8, diff_f64x8, distance_sq_f64x8);
    if (n) goto simsimd_l2sq_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(distance_sq_f64x8);
}

SIMSIMD_PUBLIC void simsimd_angular_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                                simsimd_f64_t *result) {
    __m512d dot_product_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

simsimd_angular_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    dot_product_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, dot_product_f64x8);
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto simsimd_angular_f64_skylake_cycle;

    simsimd_f64_t dot_product_f64 = _mm512_reduce_add_pd(dot_product_f64x8);
    simsimd_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    simsimd_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = _simsimd_angular_normalize_f64_skylake(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

typedef simsimd_dot_f64x8_state_skylake_t simsimd_angular_f64x8_state_skylake_t;
SIMSIMD_INTERNAL void simsimd_angular_f64x8_init_skylake(simsimd_angular_f64x8_state_skylake_t *state) {
    simsimd_dot_f64x8_init_skylake(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f64x8_update_skylake(simsimd_angular_f64x8_state_skylake_t *state,
                                                           simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    simsimd_dot_f64x8_update_skylake(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f64x8_finalize_skylake(simsimd_angular_f64x8_state_skylake_t const *state_a,
                                                             simsimd_angular_f64x8_state_skylake_t const *state_b,
                                                             simsimd_angular_f64x8_state_skylake_t const *state_c,
                                                             simsimd_angular_f64x8_state_skylake_t const *state_d,
                                                             simsimd_f64_t query_norm, simsimd_f64_t target_norm_a,
                                                             simsimd_f64_t target_norm_b, simsimd_f64_t target_norm_c,
                                                             simsimd_f64_t target_norm_d, simsimd_f64_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f64_t dots[4];
    simsimd_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel angular computation (use 4 F64 values = 256-bit)
    __m256d dots_f64x4 = _mm256_loadu_pd(dots);
    __m256d query_norm_f64x4 = _mm256_set1_pd(query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Compute sqrt and normalize: 1 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results
    _mm256_storeu_pd(results, angular_f64x4);
}

typedef simsimd_dot_f64x8_state_skylake_t simsimd_l2_f64x8_state_skylake_t;
SIMSIMD_INTERNAL void simsimd_l2_f64x8_init_skylake(simsimd_l2_f64x8_state_skylake_t *state) {
    simsimd_dot_f64x8_init_skylake(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f64x8_update_skylake(simsimd_l2_f64x8_state_skylake_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    simsimd_dot_f64x8_update_skylake(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f64x8_finalize_skylake(simsimd_l2_f64x8_state_skylake_t const *state_a,
                                                        simsimd_l2_f64x8_state_skylake_t const *state_b,
                                                        simsimd_l2_f64x8_state_skylake_t const *state_c,
                                                        simsimd_l2_f64x8_state_skylake_t const *state_d,
                                                        simsimd_f64_t query_norm, simsimd_f64_t target_norm_a,
                                                        simsimd_f64_t target_norm_b, simsimd_f64_t target_norm_c,
                                                        simsimd_f64_t target_norm_d, simsimd_f64_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f64_t dots[4];
    simsimd_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_loadu_pd(dots);
    __m256d query_norm_f64x4 = _mm256_set1_pd(query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results
    _mm256_storeu_pd(results, dist_f64x4);
}

typedef simsimd_dot_f32x16_state_skylake_t simsimd_angular_f32x16_state_skylake_t;
SIMSIMD_INTERNAL void simsimd_angular_f32x16_init_skylake(simsimd_angular_f32x16_state_skylake_t *state) {
    simsimd_dot_f32x16_init_skylake(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x16_update_skylake(simsimd_angular_f32x16_state_skylake_t *state,
                                                            simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    simsimd_dot_f32x16_update_skylake(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f32x16_finalize_skylake(simsimd_angular_f32x16_state_skylake_t const *state_a,
                                                              simsimd_angular_f32x16_state_skylake_t const *state_b,
                                                              simsimd_angular_f32x16_state_skylake_t const *state_c,
                                                              simsimd_angular_f32x16_state_skylake_t const *state_d,
                                                              simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                              simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                              simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f32x16_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for higher precision angular computation
    __m256d dots_f64x4 = _mm256_set_pd((simsimd_f64_t)dots[3], (simsimd_f64_t)dots[2], //
                                       (simsimd_f64_t)dots[1], (simsimd_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((simsimd_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((simsimd_f64_t)target_norm_d, (simsimd_f64_t)target_norm_c,
                                               (simsimd_f64_t)target_norm_b, (simsimd_f64_t)target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Compute sqrt and normalize: 1 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results (convert f64 → f32)
    simsimd_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, angular_f64x4);
    results[0] = (simsimd_f32_t)u.f64s[0];
    results[1] = (simsimd_f32_t)u.f64s[1];
    results[2] = (simsimd_f32_t)u.f64s[2];
    results[3] = (simsimd_f32_t)u.f64s[3];
}

typedef simsimd_dot_f32x16_state_skylake_t simsimd_l2_f32x16_state_skylake_t;
SIMSIMD_INTERNAL void simsimd_l2_f32x16_init_skylake(simsimd_l2_f32x16_state_skylake_t *state) {
    simsimd_dot_f32x16_init_skylake(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x16_update_skylake(simsimd_l2_f32x16_state_skylake_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    simsimd_dot_f32x16_update_skylake(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f32x16_finalize_skylake(simsimd_l2_f32x16_state_skylake_t const *state_a,
                                                         simsimd_l2_f32x16_state_skylake_t const *state_b,
                                                         simsimd_l2_f32x16_state_skylake_t const *state_c,
                                                         simsimd_l2_f32x16_state_skylake_t const *state_d,
                                                         simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                         simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                         simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f32x16_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_set_pd((simsimd_f64_t)dots[3], (simsimd_f64_t)dots[2], //
                                       (simsimd_f64_t)dots[1], (simsimd_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((simsimd_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((simsimd_f64_t)target_norm_d, (simsimd_f64_t)target_norm_c,
                                               (simsimd_f64_t)target_norm_b, (simsimd_f64_t)target_norm_a);

    // Compute squared norms in parallel
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results (convert f64 → f32)
    simsimd_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, dist_f64x4);
    results[0] = (simsimd_f32_t)u.f64s[0];
    results[1] = (simsimd_f32_t)u.f64s[1];
    results[2] = (simsimd_f32_t)u.f64s[2];
    results[3] = (simsimd_f32_t)u.f64s[3];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

SIMSIMD_INTERNAL __m512i _simsimd_substract_bf16x32_genoa(__m512i a_i16, __m512i b_i16) {

    simsimd_b512_vec_t d_odd, d_even, d, a_f32_even, b_f32_even, d_f32_even, a_f32_odd, b_f32_odd, d_f32_odd, a, b;
    a.zmm = a_i16;
    b.zmm = b_i16;

    // There are several approaches to perform subtraction in `bf16`. The first one is:
    //
    //      Perform a couple of casts - each is a bitshift. To convert `bf16` to `f32`,
    //      expand it to 32-bit integers, then shift the bits by 16 to the left.
    //      Then subtract as floats, and shift back. During expansion, we will double the space,
    //      and should use separate registers for top and bottom halves.
    //      Some compilers don't have `_mm512_extracti32x8_epi32`, so we use `_mm512_extracti64x4_epi64`:
    //
    //          a_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_i16)), 16));
    //          b_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_i16)), 16));
    //          a_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_i16, 1)), 16));
    //          b_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_i16, 1)), 16));
    //          d_f32_top.fvec = _mm512_sub_ps(a_f32_top.fvec, b_f32_top.fvec);
    //          d_f32_bot.fvec = _mm512_sub_ps(a_f32_bot.fvec, b_f32_bot.fvec);
    //          d.ivec = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_bot.fvec), 16)));
    //          d.ivec = _mm512_inserti64x4(d.ivec, _mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_top.fvec), 16)), 1);
    //
    // Instead of using multple shifts and an insertion, we can achieve similar result with fewer expensive
    // calls to `_mm512_permutex2var_epi16`, or a cheap `_mm512_mask_shuffle_epi8` and blend:
    //
    a_f32_odd.zmm = _mm512_and_si512(a_i16, _mm512_set1_epi32(0xFFFF0000));
    a_f32_even.zmm = _mm512_slli_epi32(a_i16, 16);
    b_f32_odd.zmm = _mm512_and_si512(b_i16, _mm512_set1_epi32(0xFFFF0000));
    b_f32_even.zmm = _mm512_slli_epi32(b_i16, 16);

    d_f32_odd.zmm_ps = _mm512_sub_ps(a_f32_odd.zmm_ps, b_f32_odd.zmm_ps);
    d_f32_even.zmm_ps = _mm512_sub_ps(a_f32_even.zmm_ps, b_f32_even.zmm_ps);

    d_f32_even.zmm = _mm512_srli_epi32(d_f32_even.zmm, 16);
    d.zmm = _mm512_mask_blend_epi16(0x55555555, d_f32_odd.zmm, d_f32_even.zmm);

    return d.zmm;
}

SIMSIMD_PUBLIC void simsimd_l2_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result) {
    simsimd_l2sq_bf16_genoa(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    __m512 distance_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32, diff_bf16x32;

simsimd_l2sq_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    diff_bf16x32 = _simsimd_substract_bf16x32_genoa(a_bf16x32, b_bf16x32);
    distance_sq_f32x16 = _mm512_dpbf16_ps(distance_sq_f32x16, (__m512bh)(diff_bf16x32), (__m512bh)(diff_bf16x32));
    if (n) goto simsimd_l2sq_bf16_genoa_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(distance_sq_f32x16);
}

SIMSIMD_PUBLIC void simsimd_angular_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

simsimd_angular_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    dot_product_f32x16 = _mm512_dpbf16_ps(dot_product_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_norm_sq_f32x16 = _mm512_dpbf16_ps(a_norm_sq_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(a_bf16x32));
    b_norm_sq_f32x16 = _mm512_dpbf16_ps(b_norm_sq_f32x16, (__m512bh)(b_bf16x32), (__m512bh)(b_bf16x32));
    if (n) goto simsimd_angular_bf16_genoa_cycle;

    simsimd_f32_t dot_product_f32 = _simsimd_reduce_add_f32x16_skylake(dot_product_f32x16);
    simsimd_f32_t a_norm_sq_f32 = _simsimd_reduce_add_f32x16_skylake(a_norm_sq_f32x16);
    simsimd_f32_t b_norm_sq_f32 = _simsimd_reduce_add_f32x16_skylake(b_norm_sq_f32x16);
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef simsimd_dot_bf16x32_state_genoa_t simsimd_angular_bf16x32_state_genoa_t;
SIMSIMD_INTERNAL void simsimd_angular_bf16x32_init_genoa(simsimd_angular_bf16x32_state_genoa_t *state) {
    simsimd_dot_bf16x32_init_genoa(state);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x32_update_genoa(simsimd_angular_bf16x32_state_genoa_t *state,
                                                           simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    simsimd_dot_bf16x32_update_genoa(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_bf16x32_finalize_genoa(simsimd_angular_bf16x32_state_genoa_t const *state_a,
                                                             simsimd_angular_bf16x32_state_genoa_t const *state_b,
                                                             simsimd_angular_bf16x32_state_genoa_t const *state_c,
                                                             simsimd_angular_bf16x32_state_genoa_t const *state_d,
                                                             simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                             simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                             simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x32_finalize_genoa(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_bf16x32_state_genoa_t simsimd_l2_bf16x32_state_genoa_t;
SIMSIMD_INTERNAL void simsimd_l2_bf16x32_init_genoa(simsimd_l2_bf16x32_state_genoa_t *state) {
    simsimd_dot_bf16x32_init_genoa(state);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x32_update_genoa(simsimd_l2_bf16x32_state_genoa_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    simsimd_dot_bf16x32_update_genoa(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_bf16x32_finalize_genoa(simsimd_l2_bf16x32_state_genoa_t const *state_a,
                                                        simsimd_l2_bf16x32_state_genoa_t const *state_b,
                                                        simsimd_l2_bf16x32_state_genoa_t const *state_c,
                                                        simsimd_l2_bf16x32_state_genoa_t const *state_d,
                                                        simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                        simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                        simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_bf16x32_finalize_genoa(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_l2sq_f16_sapphire(a, b, n, result);
    *result = _simsimd_sqrt_f32_haswell(*result);
}
SIMSIMD_PUBLIC void simsimd_l2sq_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result) {
    __m512h distance_sq_f16x32 = _mm512_setzero_ph();
    __m512i a_f16x32, b_f16x32;

simsimd_l2sq_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a);
        b_f16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h diff_f16x32 = _mm512_sub_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32));
    distance_sq_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, distance_sq_f16x32);
    if (n) goto simsimd_l2sq_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(distance_sq_f16x32);
}

SIMSIMD_PUBLIC void simsimd_angular_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                                 simsimd_f32_t *result) {
    __m512h dot_product_f16x32 = _mm512_setzero_ph();
    __m512h a_norm_sq_f16x32 = _mm512_setzero_ph();
    __m512h b_norm_sq_f16x32 = _mm512_setzero_ph();
    __m512i a_f16x32, b_f16x32;

simsimd_angular_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a);
        b_f16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    dot_product_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32),
                                         dot_product_f16x32);
    a_norm_sq_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(a_f16x32), a_norm_sq_f16x32);
    b_norm_sq_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(b_f16x32), _mm512_castsi512_ph(b_f16x32), b_norm_sq_f16x32);
    if (n) goto simsimd_angular_f16_sapphire_cycle;

    simsimd_f32_t dot_product_f32 = _mm512_reduce_add_ph(dot_product_f16x32);
    simsimd_f32_t a_norm_sq_f32 = _mm512_reduce_add_ph(a_norm_sq_f16x32);
    simsimd_f32_t b_norm_sq_f32 = _mm512_reduce_add_ph(b_norm_sq_f16x32);
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef simsimd_dot_f16x32_state_sapphire_t simsimd_angular_f16x32_state_sapphire_t;
SIMSIMD_INTERNAL void simsimd_angular_f16x32_init_sapphire(simsimd_angular_f16x32_state_sapphire_t *state) {
    simsimd_dot_f16x32_init_sapphire(state);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x32_update_sapphire(simsimd_angular_f16x32_state_sapphire_t *state,
                                                             simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    simsimd_dot_f16x32_update_sapphire(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_f16x32_finalize_sapphire(simsimd_angular_f16x32_state_sapphire_t const *state_a,
                                                               simsimd_angular_f16x32_state_sapphire_t const *state_b,
                                                               simsimd_angular_f16x32_state_sapphire_t const *state_c,
                                                               simsimd_angular_f16x32_state_sapphire_t const *state_d,
                                                               simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                               simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                               simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f16x32_finalize_sapphire(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_f16x32_state_sapphire_t simsimd_l2_f16x32_state_sapphire_t;
SIMSIMD_INTERNAL void simsimd_l2_f16x32_init_sapphire(simsimd_l2_f16x32_state_sapphire_t *state) {
    simsimd_dot_f16x32_init_sapphire(state);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x32_update_sapphire(simsimd_l2_f16x32_state_sapphire_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    simsimd_dot_f16x32_update_sapphire(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_f16x32_finalize_sapphire(simsimd_l2_f16x32_state_sapphire_t const *state_a,
                                                          simsimd_l2_f16x32_state_sapphire_t const *state_b,
                                                          simsimd_l2_f16x32_state_sapphire_t const *state_c,
                                                          simsimd_l2_f16x32_state_sapphire_t const *state_d,
                                                          simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                          simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                          simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_f32_t dots[4];
    simsimd_dot_f16x32_finalize_sapphire(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_l2_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result) {
    simsimd_u32_t d2;
    simsimd_l2sq_i8_ice(a, b, n, &d2);
    *result = _simsimd_sqrt_f32_haswell((simsimd_f32_t)d2);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_u32_t *result) {
    __m512i distance_sq_i32x16 = _mm512_setzero_si512();
    __m512i a_i16x32, b_i16x32, diff_i16x32;

simsimd_l2sq_i8_ice_cycle:
    if (n < 32) { // TODO: Avoid early i16 upcast to step through 64 values at a time
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }
    diff_i16x32 = _mm512_sub_epi16(a_i16x32, b_i16x32);
    distance_sq_i32x16 = _mm512_dpwssd_epi32(distance_sq_i32x16, diff_i16x32, diff_i16x32);
    if (n) goto simsimd_l2sq_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(distance_sq_i32x16);
}

SIMSIMD_PUBLIC void simsimd_angular_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {

    __m512i dot_product_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_i32x16 = _mm512_setzero_si512();
    __m512i a_i16x32, b_i16x32;
simsimd_angular_i8_ice_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }

    // We can't directly use the `_mm512_dpbusd_epi32` intrinsic everywhere,
    // as it's asymmetric with respect to the sign of the input arguments:
    //
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    //
    // To compute the squares, we could just drop the sign bit of the second argument.
    // But this would lead to big-big problems on values like `-128`!
    // For dot-products we don't have the luxury of optimizing the sign bit away.
    // Assuming this is an approximate kernel (with reciprocal square root approximations)
    // in the end, we can allow clamping the value to [-127, 127] range.
    //
    // On Ice Lake:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can only execute on port 0, with 5 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also only execute on port 0, with 5 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 5, with 5 cycle latency.
    //
    // On Zen4 Genoa:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 4 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also execute on ports 0 and 1, with 4 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 3 cycle latency.
    //
    // The old solution was complex replied on 1. and 2.:
    //
    //    a_i8_abs_vec = _mm512_abs_epi8(a_i8_vec);
    //    b_i8_abs_vec = _mm512_abs_epi8(b_i8_vec);
    //    a2_i32_vec = _mm512_dpbusds_epi32(a2_i32_vec, a_i8_abs_vec, a_i8_abs_vec);
    //    b2_i32_vec = _mm512_dpbusds_epi32(b2_i32_vec, b_i8_abs_vec, b_i8_abs_vec);
    //    ab_i32_low_vec = _mm512_dpwssds_epi32(                      //
    //        ab_i32_low_vec,                                         //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)), //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_vec)));
    //    ab_i32_high_vec = _mm512_dpwssds_epi32(                           //
    //        ab_i32_high_vec,                                              //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)), //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_vec, 1)));
    //
    // The new solution is simpler and relies on 3.:
    dot_product_i32x16 = _mm512_add_epi32(dot_product_i32x16, _mm512_madd_epi16(a_i16x32, b_i16x32));
    a_norm_sq_i32x16 = _mm512_add_epi32(a_norm_sq_i32x16, _mm512_madd_epi16(a_i16x32, a_i16x32));
    b_norm_sq_i32x16 = _mm512_add_epi32(b_norm_sq_i32x16, _mm512_madd_epi16(b_i16x32, b_i16x32));
    if (n) goto simsimd_angular_i8_ice_cycle;

    simsimd_i32_t dot_product_i32 = _mm512_reduce_add_epi32(dot_product_i32x16);
    simsimd_i32_t a_norm_sq_i32 = _mm512_reduce_add_epi32(a_norm_sq_i32x16);
    simsimd_i32_t b_norm_sq_i32 = _mm512_reduce_add_epi32(b_norm_sq_i32x16);
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}
SIMSIMD_PUBLIC void simsimd_l2_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result) {
    simsimd_u32_t d2;
    simsimd_l2sq_u8_ice(a, b, n, &d2);
    *result = _simsimd_sqrt_f32_haswell((simsimd_f32_t)d2);
}
SIMSIMD_PUBLIC void simsimd_l2sq_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_u32_t *result) {
    __m512i distance_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i distance_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();
    __m512i diff_low_i16x32, diff_high_i16x32;
    __m512i a_u8x64, b_u8x64, diff_u8x64;

simsimd_l2sq_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a);
        b_u8x64 = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
    diff_u8x64 = _mm512_or_si512(_mm512_subs_epu8(a_u8x64, b_u8x64), _mm512_subs_epu8(b_u8x64, a_u8x64));
    diff_low_i16x32 = _mm512_unpacklo_epi8(diff_u8x64, zeros_i8x64);
    diff_high_i16x32 = _mm512_unpackhi_epi8(diff_u8x64, zeros_i8x64);

    // Multiply and accumulate at `int16` level, accumulate at `int32` level:
    distance_sq_low_i32x16 = _mm512_dpwssd_epi32(distance_sq_low_i32x16, diff_low_i16x32, diff_low_i16x32);
    distance_sq_high_i32x16 = _mm512_dpwssd_epi32(distance_sq_high_i32x16, diff_high_i16x32, diff_high_i16x32);
    if (n) goto simsimd_l2sq_u8_ice_cycle;

    *result = _mm512_reduce_add_epi32(_mm512_add_epi32(distance_sq_low_i32x16, distance_sq_high_i32x16));
}

SIMSIMD_PUBLIC void simsimd_angular_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result) {

    __m512i dot_product_low_i32x16 = _mm512_setzero_si512();
    __m512i dot_product_high_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();
    __m512i a_low_i16x32, a_high_i16x32, b_low_i16x32, b_high_i16x32;
    __m512i a_u8x64, b_u8x64;

simsimd_angular_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a);
        b_u8x64 = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);

    // Multiply and accumulate as `int16`, accumulate products as `int32`:
    dot_product_low_i32x16 = _mm512_dpwssds_epi32(dot_product_low_i32x16, a_low_i16x32, b_low_i16x32);
    dot_product_high_i32x16 = _mm512_dpwssds_epi32(dot_product_high_i32x16, a_high_i16x32, b_high_i16x32);
    a_norm_sq_low_i32x16 = _mm512_dpwssds_epi32(a_norm_sq_low_i32x16, a_low_i16x32, a_low_i16x32);
    a_norm_sq_high_i32x16 = _mm512_dpwssds_epi32(a_norm_sq_high_i32x16, a_high_i16x32, a_high_i16x32);
    b_norm_sq_low_i32x16 = _mm512_dpwssds_epi32(b_norm_sq_low_i32x16, b_low_i16x32, b_low_i16x32);
    b_norm_sq_high_i32x16 = _mm512_dpwssds_epi32(b_norm_sq_high_i32x16, b_high_i16x32, b_high_i16x32);
    if (n) goto simsimd_angular_u8_ice_cycle;

    simsimd_i32_t dot_product_i32 = _mm512_reduce_add_epi32(
        _mm512_add_epi32(dot_product_low_i32x16, dot_product_high_i32x16));
    simsimd_i32_t a_norm_sq_i32 = _mm512_reduce_add_epi32(
        _mm512_add_epi32(a_norm_sq_low_i32x16, a_norm_sq_high_i32x16));
    simsimd_i32_t b_norm_sq_i32 = _mm512_reduce_add_epi32(
        _mm512_add_epi32(b_norm_sq_low_i32x16, b_norm_sq_high_i32x16));
    *result = _simsimd_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

SIMSIMD_PUBLIC void simsimd_l2_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                        simsimd_f32_t *result) {
    simsimd_u32_t d2;
    simsimd_l2sq_i4x2_ice(a, b, n_words, &d2);
    *result = _simsimd_sqrt_f32_haswell((simsimd_f32_t)d2);
}
SIMSIMD_PUBLIC void simsimd_l2sq_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                          simsimd_u32_t *result) {

    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // The absolute difference between two 4-bit integers is at most 15 and it is always a `uint4_t` value!
    // Moreover, it's square is at most 225, which fits into `uint8_t` and can be computed with a single
    // lookup table. Accumulating those values is similar to checksumming, a piece of cake for SIMD!
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const u4_squares_lookup_vec = _mm512_set_epi8(                                        //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i d_u8_low_vec, d_u8_high_vec; //! Only the low 4 bits are actually used
    __m512i d2_u8_low_vec, d2_u8_high_vec;
    __m512i d2_u16_low_vec, d2_u16_high_vec;

    // Accumulators:
    __m512i d2_u32_vec = _mm512_setzero_si512();

simsimd_l2sq_i4x2_ice_cycle:
    if (n_words < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec);
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec);
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec);
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // We can implement subtraction with a lookup table, or using `_mm512_sub_epi8`.
    d_u8_low_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_low_vec, b_i8_low_vec));
    d_u8_high_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_high_vec, b_i8_high_vec));

    // Now we can use the lookup table to compute the squares of the 4-bit unsigned integers
    // in the low nibbles of the `d_u8_low_vec` and `d_u8_high_vec` vectors.
    d2_u8_low_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_low_vec);
    d2_u8_high_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_high_vec);

    // Aggregating into 16-bit integers, we need to first upcast our 8-bit values to 16 bits.
    // After that, we will perform one more operation, upcasting further into 32-bit integers.
    d2_u16_low_vec =      //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_low_vec, _mm512_setzero_si512()),
            _mm512_unpackhi_epi8(d2_u8_low_vec, _mm512_setzero_si512()));
    d2_u16_high_vec =     //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_high_vec, _mm512_setzero_si512()),
            _mm512_unpackhi_epi8(d2_u8_high_vec, _mm512_setzero_si512()));
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_low_vec, _mm512_setzero_si512()));
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_high_vec, _mm512_setzero_si512()));
    if (n_words) goto simsimd_l2sq_i4x2_ice_cycle;

    // Finally, we can reduce the 16-bit integers to 32-bit integers and sum them up.
    int d2 = _mm512_reduce_add_epi32(d2_u32_vec);
    *result = d2;
}
SIMSIMD_PUBLIC void simsimd_angular_i4x2_ice(simsimd_i4x2_t const *a, simsimd_i4x2_t const *b, simsimd_size_t n_words,
                                             simsimd_f32_t *result) {

    // We need to compose a lookup table for all the scalar products of 4-bit integers.
    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // Practically speaking, the product of two 4-bit signed integers is a 7-bit integer,
    // as the maximum absolute value of the product is `abs(-8 * -8) == 64`.
    //
    // To store 128 possible values of 2^7 bits we only need 128 single-byte scalars,
    // or just 2x ZMM registers. In that case our lookup will only take `vpermi2b` instruction,
    // easily inokable with `_mm512_permutex2var_epi8` intrinsic with latency of 6 on Sapphire Rapids.
    // The problem is converting 2d indices of our symmetric matrix into 1d offsets in the dense array.
    //
    // Alternatively, we can take the entire symmetric (16 x 16) matrix of products,
    // put into 4x ZMM registers, and use it with `_mm512_shuffle_epi8`, remembering
    // that it can only lookup with 128-bit lanes (16x 8-bit values).
    // That intrinsic has latency 1, but will need to be repeated and combined with
    // multiple iterations of `_mm512_shuffle_i64x2` that has latency 3.
    //
    // Altenatively, we can get down to 3 cycles per lookup with `vpermb` and `_mm512_permutexvar_epi8` intrinsics.
    // For that we can split our (16 x 16) matrix into 4x (8 x 8) submatrices, and use 4x ZMM registers.
    //
    // Still, all of those solutions are quite heavy compared to two parallel calls to `_mm512_dpbusds_epi32`
    // for the dot product. But we can still use the `_mm512_permutexvar_epi8` to compute the squares of the
    // 16 possible `int4_t` values faster.
    //
    // Here is how our `int4_t` range looks:
    //
    //      dec:     0   1   2   3   4   5   6   7  -8  -7  -6  -5  -4  -3  -2  -1
    //      hex:     0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    //
    // Squared:
    //
    //      dec2:    0   1   4   9  16  25  36  49  64  49  36  25  16   9   4   1
    //      hex2:    0   1   4   9  10  19  24  31  40  31  24  19  10   9   4   1
    //
    // Broadcast it to every lane, so that: `square(x) == _mm512_shuffle_epi8(i4_squares_lookup_vec, x)`.
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const i4_squares_lookup_vec = _mm512_set_epi8(       //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i a2_u8_vec, b2_u8_vec;

    // Accumulators:
    __m512i a2_u16_low_vec = _mm512_setzero_si512();
    __m512i a2_u16_high_vec = _mm512_setzero_si512();
    __m512i b2_u16_low_vec = _mm512_setzero_si512();
    __m512i b2_u16_high_vec = _mm512_setzero_si512();
    __m512i ab_i32_low_vec = _mm512_setzero_si512();
    __m512i ab_i32_high_vec = _mm512_setzero_si512();

simsimd_angular_i4x2_ice_cycle:
    if (n_words < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    // For now, they are not really 8-bit integers, as they are not sign-extended.
    // That part will come later, using the `i4_to_i8_lookup_vec` lookup.
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec);
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec);
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);

    // Compute the squares of the 4-bit integers.
    // For symmetry we could have used 4 registers, aka "a2_i8_low_vec", "a2_i8_high_vec", "b2_i8_low_vec",
    // "b2_i8_high_vec". But the largest square value is just 64, so we can safely aggregate into 8-bit unsigned values.
    a2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_high_vec));
    b2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_high_vec));

    // We can safely aggregate into just 16-bit sums without overflow, if the vectors have less than:
    //      (2 scalars / byte) * (64 bytes / register) * (256 non-overflowing 8-bit additions in 16-bit intesgers)
    //      = 32'768 dimensions.
    //
    // We use saturated addition here to clearly inform in case of overflow.
    a2_u16_low_vec = _mm512_adds_epu16(a2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    a2_u16_high_vec = _mm512_adds_epu16(a2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));
    b2_u16_low_vec = _mm512_adds_epu16(b2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    b2_u16_high_vec = _mm512_adds_epu16(b2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));

    // Time to perform the proper sign extension of the 4-bit integers to 8-bit integers.
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec);
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // The same trick won't work for the primary dot-product, as the signs vector
    // components may differ significantly. So we have to use two `_mm512_dpwssds_epi32`
    // intrinsics instead, upcasting four chunks to 16-bit integers beforehand!
    // Alternatively, we can flip the signs of the second argument and use `_mm512_dpbusds_epi32`,
    // but it ends up taking more instructions.
    ab_i32_low_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_low_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_low_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_low_vec)));
    ab_i32_low_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_low_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_low_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_low_vec, 1)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_high_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_high_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_high_vec)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_high_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_high_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_high_vec, 1)));
    if (n_words) goto simsimd_angular_i4x2_ice_cycle;

    int ab = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    unsigned short a2_u16[32], b2_u16[32];
    _mm512_storeu_si512(a2_u16, _mm512_add_epi16(a2_u16_low_vec, a2_u16_high_vec));
    _mm512_storeu_si512(b2_u16, _mm512_add_epi16(b2_u16_low_vec, b2_u16_high_vec));
    unsigned int a2 = 0, b2 = 0;
    for (int i = 0; i < 32; ++i) a2 += a2_u16[i], b2 += b2_u16[i];
    *result = _simsimd_angular_normalize_f32_haswell(ab, a2, b2);
}

typedef simsimd_dot_i8x64_state_ice_t simsimd_angular_i8x64_state_ice_t;
SIMSIMD_INTERNAL void simsimd_angular_i8x64_init_ice(simsimd_angular_i8x64_state_ice_t *state) {
    simsimd_dot_i8x64_init_ice(state);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x64_update_ice(simsimd_angular_i8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    simsimd_dot_i8x64_update_ice(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x64_finalize_ice(simsimd_angular_i8x64_state_ice_t const *state_a,
                                                         simsimd_angular_i8x64_state_ice_t const *state_b,
                                                         simsimd_angular_i8x64_state_ice_t const *state_c,
                                                         simsimd_angular_i8x64_state_ice_t const *state_d,
                                                         simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                         simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                         simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // products = query_norm_sq * target_norms_sq
    __m128 products_f32x4 = _mm_mul_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement: x' = x * (1.5 - 0.5 * val * x * x)
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq_f32x4 = _mm_mul_ps(rsqrt_f32x4, rsqrt_f32x4);
    __m128 half_prod_f32x4 = _mm_mul_ps(half_f32x4, products_f32x4);
    __m128 muls_f32x4 = _mm_mul_ps(half_prod_f32x4, rsqrt_sq_f32x4);
    __m128 refinement_f32x4 = _mm_sub_ps(three_halves_f32x4, muls_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(rsqrt_f32x4, refinement_f32x4);

    // normalized = dots * rsqrt(products)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_i8x64_state_ice_t simsimd_l2_i8x64_state_ice_t;
SIMSIMD_INTERNAL void simsimd_l2_i8x64_init_ice(simsimd_l2_i8x64_state_ice_t *state) {
    simsimd_dot_i8x64_init_ice(state);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x64_update_ice(simsimd_l2_i8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                  simsimd_b512_vec_t b) {
    simsimd_dot_i8x64_update_ice(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x64_finalize_ice(simsimd_l2_i8x64_state_ice_t const *state_a,
                                                    simsimd_l2_i8x64_state_ice_t const *state_b,
                                                    simsimd_l2_i8x64_state_ice_t const *state_c,
                                                    simsimd_l2_i8x64_state_ice_t const *state_d,
                                                    simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                    simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                    simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_i32_t dots_i32[4];
    simsimd_dot_i8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    // dist_sq = query_norm_sq + target_norms_sq - 2*dots
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 two_dots_f32x4 = _mm_mul_ps(two_f32x4, dots_f32x4);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_sub_ps(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    dist_sq_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(dist_sq_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef simsimd_dot_u8x64_state_ice_t simsimd_angular_u8x64_state_ice_t;
SIMSIMD_INTERNAL void simsimd_angular_u8x64_init_ice(simsimd_angular_u8x64_state_ice_t *state) {
    simsimd_dot_u8x64_init_ice(state);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x64_update_ice(simsimd_angular_u8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    simsimd_dot_u8x64_update_ice(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_u8x64_finalize_ice(simsimd_angular_u8x64_state_ice_t const *state_a,
                                                         simsimd_angular_u8x64_state_ice_t const *state_b,
                                                         simsimd_angular_u8x64_state_ice_t const *state_c,
                                                         simsimd_angular_u8x64_state_ice_t const *state_d,
                                                         simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                         simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                         simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 (u32 values fit in f32 mantissa for typical vector lengths)
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_u32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // products = query_norm_sq * target_norms_sq
    __m128 products_f32x4 = _mm_mul_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement: x' = x * (1.5 - 0.5 * val * x * x)
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq_f32x4 = _mm_mul_ps(rsqrt_f32x4, rsqrt_f32x4);
    __m128 half_prod_f32x4 = _mm_mul_ps(half_f32x4, products_f32x4);
    __m128 muls_f32x4 = _mm_mul_ps(half_prod_f32x4, rsqrt_sq_f32x4);
    __m128 refinement_f32x4 = _mm_sub_ps(three_halves_f32x4, muls_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(rsqrt_f32x4, refinement_f32x4);

    // normalized = dots * rsqrt(products)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef simsimd_dot_u8x64_state_ice_t simsimd_l2_u8x64_state_ice_t;
SIMSIMD_INTERNAL void simsimd_l2_u8x64_init_ice(simsimd_l2_u8x64_state_ice_t *state) {
    simsimd_dot_u8x64_init_ice(state);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x64_update_ice(simsimd_l2_u8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                  simsimd_b512_vec_t b) {
    simsimd_dot_u8x64_update_ice(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_u8x64_finalize_ice(simsimd_l2_u8x64_state_ice_t const *state_a,
                                                    simsimd_l2_u8x64_state_ice_t const *state_b,
                                                    simsimd_l2_u8x64_state_ice_t const *state_c,
                                                    simsimd_l2_u8x64_state_ice_t const *state_d,
                                                    simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                    simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                    simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    simsimd_u32_t dots_u32[4];
    simsimd_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 (u32 values fit in f32 mantissa for typical vector lengths)
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_u32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    // dist_sq = query_norm_sq + target_norms_sq - 2*dots
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 two_dots_f32x4 = _mm_mul_ps(two_f32x4, dots_f32x4);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_sub_ps(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    dist_sq_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(dist_sq_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_angular_i8_sierra(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));
        dot_product_i32x8 = _mm256_dpbssds_epi32(dot_product_i32x8, a_i8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbssds_epi32(a_norm_sq_i32x8, a_i8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbssds_epi32(b_norm_sq_i32x8, b_i8x32, b_i8x32);
    }

    // Further reduce to a single sum for each vector
    simsimd_i32_t dot_product_i32 = _simsimd_reduce_add_i32x8_haswell(dot_product_i32x8);
    simsimd_i32_t a_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(a_norm_sq_i32x8);
    simsimd_i32_t b_norm_sq_i32 = _simsimd_reduce_add_i32x8_haswell(b_norm_sq_i32x8);

    // Take care of the tail:
    for (; i < n; ++i) {
        simsimd_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = _simsimd_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

typedef simsimd_dot_i8x64_state_sierra_t simsimd_angular_i8x64_state_sierra_t;
SIMSIMD_INTERNAL void simsimd_angular_i8x64_init_sierra(simsimd_angular_i8x64_state_sierra_t *state) {
    simsimd_dot_i8x64_init_sierra(state);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x64_update_sierra(simsimd_angular_i8x64_state_sierra_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    simsimd_dot_i8x64_update_sierra(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_angular_i8x64_finalize_sierra(simsimd_angular_i8x64_state_sierra_t const *state_a,
                                                            simsimd_angular_i8x64_state_sierra_t const *state_b,
                                                            simsimd_angular_i8x64_state_sierra_t const *state_c,
                                                            simsimd_angular_i8x64_state_sierra_t const *state_d,
                                                            simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                            simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                            simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states
    simsimd_distance_t dot_product_a, dot_product_b, dot_product_c, dot_product_d;
    simsimd_dot_i8x64_finalize_sierra(state_a, &dot_product_a);
    simsimd_dot_i8x64_finalize_sierra(state_b, &dot_product_b);
    simsimd_dot_i8x64_finalize_sierra(state_c, &dot_product_c);
    simsimd_dot_i8x64_finalize_sierra(state_d, &dot_product_d);

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = (simsimd_f32_t)query_norm * (simsimd_f32_t)query_norm;
    simsimd_f32_t target_norm_sq_a = (simsimd_f32_t)target_norm_a * (simsimd_f32_t)target_norm_a;
    simsimd_f32_t target_norm_sq_b = (simsimd_f32_t)target_norm_b * (simsimd_f32_t)target_norm_b;
    simsimd_f32_t target_norm_sq_c = (simsimd_f32_t)target_norm_c * (simsimd_f32_t)target_norm_c;
    simsimd_f32_t target_norm_sq_d = (simsimd_f32_t)target_norm_d * (simsimd_f32_t)target_norm_d;

    // Compute angular distances (loop-unrolled)
    results[0] = _simsimd_angular_normalize_f32_haswell((simsimd_f32_t)dot_product_a, query_norm_sq, target_norm_sq_a);
    results[1] = _simsimd_angular_normalize_f32_haswell((simsimd_f32_t)dot_product_b, query_norm_sq, target_norm_sq_b);
    results[2] = _simsimd_angular_normalize_f32_haswell((simsimd_f32_t)dot_product_c, query_norm_sq, target_norm_sq_c);
    results[3] = _simsimd_angular_normalize_f32_haswell((simsimd_f32_t)dot_product_d, query_norm_sq, target_norm_sq_d);
}

typedef simsimd_dot_i8x64_state_sierra_t simsimd_l2_i8x64_state_sierra_t;
SIMSIMD_INTERNAL void simsimd_l2_i8x64_init_sierra(simsimd_l2_i8x64_state_sierra_t *state) {
    simsimd_dot_i8x64_init_sierra(state);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x64_update_sierra(simsimd_l2_i8x64_state_sierra_t *state, simsimd_b512_vec_t a,
                                                     simsimd_b512_vec_t b) {
    simsimd_dot_i8x64_update_sierra(state, a, b);
}
SIMSIMD_INTERNAL void simsimd_l2_i8x64_finalize_sierra(simsimd_l2_i8x64_state_sierra_t const *state_a,
                                                       simsimd_l2_i8x64_state_sierra_t const *state_b,
                                                       simsimd_l2_i8x64_state_sierra_t const *state_c,
                                                       simsimd_l2_i8x64_state_sierra_t const *state_d,
                                                       simsimd_f32_t query_norm, simsimd_f32_t target_norm_a,
                                                       simsimd_f32_t target_norm_b, simsimd_f32_t target_norm_c,
                                                       simsimd_f32_t target_norm_d, simsimd_f32_t *results) {
    // Extract dots from states
    simsimd_distance_t dot_product_a, dot_product_b, dot_product_c, dot_product_d;
    simsimd_dot_i8x64_finalize_sierra(state_a, &dot_product_a);
    simsimd_dot_i8x64_finalize_sierra(state_b, &dot_product_b);
    simsimd_dot_i8x64_finalize_sierra(state_c, &dot_product_c);
    simsimd_dot_i8x64_finalize_sierra(state_d, &dot_product_d);

    // Compute squared norms (loop-unrolled)
    simsimd_f32_t query_norm_sq = (simsimd_f32_t)query_norm * (simsimd_f32_t)query_norm;
    simsimd_f32_t target_norm_sq_a = (simsimd_f32_t)target_norm_a * (simsimd_f32_t)target_norm_a;
    simsimd_f32_t target_norm_sq_b = (simsimd_f32_t)target_norm_b * (simsimd_f32_t)target_norm_b;
    simsimd_f32_t target_norm_sq_c = (simsimd_f32_t)target_norm_c * (simsimd_f32_t)target_norm_c;
    simsimd_f32_t target_norm_sq_d = (simsimd_f32_t)target_norm_d * (simsimd_f32_t)target_norm_d;

    // Compute squared distances (loop-unrolled)
    simsimd_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - (simsimd_f32_t)2 * (simsimd_f32_t)dot_product_a;
    simsimd_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - (simsimd_f32_t)2 * (simsimd_f32_t)dot_product_b;
    simsimd_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - (simsimd_f32_t)2 * (simsimd_f32_t)dot_product_c;
    simsimd_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - (simsimd_f32_t)2 * (simsimd_f32_t)dot_product_d;

    // Use 4-way SSE sqrt (128-bit)
    __m128 dist_sq_vec = _mm_set_ps((float)(dist_sq_d < 0 ? 0 : dist_sq_d), (float)(dist_sq_c < 0 ? 0 : dist_sq_c),
                                    (float)(dist_sq_b < 0 ? 0 : dist_sq_b), (float)(dist_sq_a < 0 ? 0 : dist_sq_a));
    __m128 dist_vec = _mm_sqrt_ps(dist_sq_vec);

    // Store results using simsimd_b512_vec_t
    simsimd_b512_vec_t storage;
    _mm_storeu_ps(storage.f32s, dist_vec);
    results[0] = storage.f32s[0];
    results[1] = storage.f32s[1];
    results[2] = storage.f32s[2];
    results[3] = storage.f32s[3];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SIERRA
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_l2_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                   simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2_f64_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2_f64_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_l2_f64_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_f64_haswell(a, b, n, result);
#else
    simsimd_l2_f64_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2sq_f64_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2sq_f64_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_l2sq_f64_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_f64_haswell(a, b, n, result);
#else
    simsimd_l2sq_f64_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_angular_f64_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_angular_f64_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_angular_f64_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_f64_haswell(a, b, n, result);
#else
    simsimd_angular_f64_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                   simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2_f32_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_l2_f32_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_f32_haswell(a, b, n, result);
#else
    simsimd_l2_f32_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2sq_f32_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2sq_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_l2sq_f32_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_f32_haswell(a, b, n, result);
#else
    simsimd_l2sq_f32_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_angular_f32_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_angular_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_angular_f32_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_f32_haswell(a, b, n, result);
#else
    simsimd_angular_f32_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                   simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2_f16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_l2_f16_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_f16_haswell(a, b, n, result);
#else
    simsimd_l2_f16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2sq_f16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2sq_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_l2sq_f16_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_f16_haswell(a, b, n, result);
#else
    simsimd_l2sq_f16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_angular_f16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_angular_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_angular_f16_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_f16_haswell(a, b, n, result);
#else
    simsimd_angular_f16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                    simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2_bf16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2_bf16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_GENOA
    simsimd_l2_bf16_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_bf16_haswell(a, b, n, result);
#else
    simsimd_l2_bf16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_l2sq_bf16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_l2sq_bf16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_GENOA
    simsimd_l2sq_bf16_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_bf16_haswell(a, b, n, result);
#else
    simsimd_l2sq_bf16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_angular_bf16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_angular_bf16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_GENOA
    simsimd_angular_bf16_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_bf16_haswell(a, b, n, result);
#else
    simsimd_angular_bf16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                  simsimd_f32_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_l2_i8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_l2_i8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_i8_haswell(a, b, n, result);
#else
    simsimd_l2_i8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                    simsimd_u32_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_l2sq_i8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_l2sq_i8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_i8_haswell(a, b, n, result);
#else
    simsimd_l2sq_i8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SIERRA
    simsimd_angular_i8_sierra(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_angular_i8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_angular_i8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_i8_haswell(a, b, n, result);
#else
    simsimd_angular_i8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                  simsimd_f32_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_l2_u8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_l2_u8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2_u8_haswell(a, b, n, result);
#else
    simsimd_l2_u8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_l2sq_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                    simsimd_u32_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_l2sq_u8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_l2sq_u8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_l2sq_u8_haswell(a, b, n, result);
#else
    simsimd_l2sq_u8_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_angular_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                       simsimd_f32_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_angular_u8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_angular_u8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_angular_u8_haswell(a, b, n, result);
#else
    simsimd_angular_u8_serial(a, b, n, result);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
