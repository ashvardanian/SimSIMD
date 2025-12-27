/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers.
 *  @file include/simsimd/dot.h
 *  @author Ash Vardanian
 *  @date February 24, 2024
 *
 *  Contains:
 *
 *  - Dot Product for Real and Complex vectors
 *  - Conjugate Dot Product for Complex vectors
 *
 *  For datatypes:
 *
 *  - 64-bit IEEE floating point numbers → 64-bit floats
 *  - 32-bit IEEE floating point numbers → 32-bit floats
 *  - 16-bit IEEE floating point numbers → 32-bit floats
 *  - 16-bit brain floating point numbers → 32-bit floats
 *  - 8-bit e4m3 floating point numbers → 32-bit floats
 *  - 8-bit e5m2 floating point numbers → 32-bit floats
 *  - 8-bit unsigned integers → 32-bit unsigned integers
 *  - 8-bit signed integers → 32-bit signed integers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire, Sierra Forest
 *
 *  @section streaming_api Streaming API
 *
 *  For compile-time dispatch and vector-at-a-time accumulation, we provide streaming helpers
 *  that accept two `simsimd_b512_vec_t` blocks and update a running sum for non-complex dot
 *  products. The `<count>` suffix reflects how many scalars of that type fit in a 512-bit block.
 *  The helpers are exposed per scalar type as:
 *
 *  - simsimd_dot_<type>x<count>_state_<isa>_t
 *  - simsimd_dot_<type>x<count>_init_<isa>
 *  - simsimd_dot_<type>x<count>_update_<isa>
 *  - simsimd_dot_<type>x<count>_finalize_<isa>
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Floating-point dot products use FMA (VFMADD231PS/PD) for sum += a[i]*b[i] accumulation.
 *  Integer i8 dot products use VPMADDUBSW (u8 × i8 → i16) + VPMADDWD (i16 × 1 → i32) on Haswell,
 *  or the newer VNNI instructions VPDPBUSD/VPDPWSSD on Ice Lake+ for direct u8 × i8 → i32.
 *  BF16 dot products (VDPBF16PS) are Genoa-only, accumulating bf16 pairs directly to f32.
 *  Genoa shows 40% faster integer multiply-add (3c vs 5c) than Ice Lake.
 *
 *      Intrinsic               Instruction                     Haswell     Ice         Genoa
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     5c @ p01    4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     5c @ p01    4c @ p01    4c @ p01
 *      _mm256_maddubs_epi16    VPMADDUBSW (YMM, YMM, YMM)      5c @ p0     5c @ p01    3c @ p01
 *      _mm256_madd_epi16       VPMADDWD (YMM, YMM, YMM)        5c @ p0     5c @ p01    3c @ p01
 *      _mm256_dpbusd_epi32     VPDPBUSD (YMM, YMM, YMM)        N/A         5c @ p01    4c @ p01
 *      _mm512_dpwssd_epi32     VPDPWSSD (ZMM, ZMM, ZMM)        N/A         5c @ p0     4c @ p01
 *      _mm512_dpbf16_ps        VDPBF16PS (ZMM, ZMM, ZMM)       N/A         N/A         6c @ p01
 *
 *  @section arm_neon_instructions Relevant ARM NEON Instructions
 *
 *  NEON integer dot products use SDOT/UDOT (ARMv8.2 dotprod) for direct i8 × i8 → i32 or u8 × u8 → u32
 *  accumulation - 4x faster than the multiply-add sequence on older cores. BFDOT (ARMv8.6 bf16)
 *  provides native bf16 dot products on Graviton 3+. Complex dot products use LD2 for deinterleaved
 *  loads of real/imag pairs, though its L01+V throughput can bottleneck on memory-bound workloads.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vfmaq_f64               FMLA.D (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vdotq_s32               SDOT (vec)      3c @ V0123      3c @ V0123      3c @ V0123
 *      vdotq_u32               UDOT (vec)      3c @ V0123      3c @ V0123      3c @ V0123
 *      vbfdotq_f32             BFDOT (vec)     N/A             4c @ V0123      5c @ V0123
 *      vld2q_f32               LD2 (Q-form)    5c @ L01+V      8c @ L01+V      8c @ L01+V
 *
 *  @section arm_sve_instructions Relevant ARM SVE Instructions
 *
 *  SVE implementations use predicated FMA (svmla_f32_x) with WHILELT for tail masking, avoiding
 *  scalar cleanup loops. FADDV performs horizontal reduction; notably 45% faster on Graviton 4
 *  (6c) than Graviton 3 (11c). SVE complex dot products use svld2 for structure loads.
 *
 *      Intrinsic               Instruction     Graviton 3      Graviton 4
 *      svmla_f32_x             FMLA (pred)     4c @ V0123      4c @ V0123
 *      svmls_f32_x             FMLS (pred)     4c @ V0123      4c @ V0123
 *      svwhilelt_b32           WHILELT         3c @ M0         3c @ M0
 *      svld2_f32               LD2 (SVE)       8c @ L01+V      8c @ L01+V
 *      svaddv_f32              FADDV           11c @ V0123     6c @ V0123
 *
 *  @section complex_instructions Complex Number Optimizations
 *
 *  Standard complex multiplication involves subtraction for the real part.
 *  Instead of using subtracting variants of FMA for every element, we accumulate real
 *  and imaginary products positively and apply a single bitwise XOR to flip the sign
 *  bits before the final horizontal reduction. This delayed application of the sign
 *  flip doubles the throughput on older x86 architectures like Haswell by maximizing
 *  FMA unit utilization and reducing execution dependency chains.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef SIMSIMD_DOT_H
#define SIMSIMD_DOT_H

#include "types.h"

#include "reduce.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Dot product computing the sum of elementwise products between two vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[out] result The output dot product value.
 *
 *  @note The output value can be negative.
 *  @note The output value is zero if and only if the two vectors are orthogonal.
 *  @note Defined only for floating-point and integer data types.
 */
SIMSIMD_DYNAMIC void simsimd_dot_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_f64_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                    simsimd_i32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                    simsimd_u32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_e4m3(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_e5m2(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                      simsimd_f32_t *result);

/**
 *  @brief Complex dot product computing the sum of elementwise products between two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] results The output complex value as {real, imag}.
 *
 *  @note The output value can be negative.
 */
SIMSIMD_DYNAMIC void simsimd_dot_f32c(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                      simsimd_size_t count_pairs, simsimd_f32c_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_f64c(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                      simsimd_size_t count_pairs, simsimd_f64c_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_f16c(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                      simsimd_size_t count_pairs, simsimd_f32c_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_bf16c(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_f32c_t *result);

/**
 *  @brief Complex conjugate dot product between two complex vectors.
 *
 *  @param[in] a_pairs The first complex vector.
 *  @param[in] b_pairs The second complex vector.
 *  @param[in] count_pairs The number of complex pairs in the vectors.
 *  @param[out] results The output complex value as {real, imag}.
 *
 *  @note The output value can be negative.
 */
SIMSIMD_DYNAMIC void simsimd_vdot_f32c(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_f64c(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_f16c(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_bf16c(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                        simsimd_size_t count_pairs, simsimd_f32c_t *result);

/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_serial(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                           simsimd_f64_t *result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_serial(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                            simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_serial(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                             simsimd_f64c_t *result);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_serial(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_serial(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                            simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_serial(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_serial(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_serial(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                            simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_serial(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_serial(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_serial(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_serial(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                              simsimd_f32c_t *result);

/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_serial(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                          simsimd_i32_t *result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_serial(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                          simsimd_u32_t *result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_serial(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_serial(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_accurate(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_accurate(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                              simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_accurate(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                               simsimd_f64c_t *result);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_accurate(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_accurate(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                              simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_accurate(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                               simsimd_f64c_t *result);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_accurate(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                              simsimd_f64_t *result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_accurate(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                               simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_accurate(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                                simsimd_f64c_t *result);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_neon(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                          simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_neon(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                           simsimd_f32c_t *result);

typedef struct simsimd_dot_f32x4_state_neon_t simsimd_dot_f32x4_state_neon_t;
/** @copydoc simsimd_dot_f32x4_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_init_neon(simsimd_dot_f32x4_state_neon_t *state);
/** @copydoc simsimd_dot_f32x4_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_update_neon(simsimd_dot_f32x4_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_f32x4_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_finalize_neon(simsimd_dot_f32x4_state_neon_t const *state_a,
                                                      simsimd_dot_f32x4_state_neon_t const *state_b,
                                                      simsimd_dot_f32x4_state_neon_t const *state_c,
                                                      simsimd_dot_f32x4_state_neon_t const *state_d,
                                                      simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_neon(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                          simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_neon(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                           simsimd_f32c_t *result);

typedef struct simsimd_dot_f16x8_state_neon_t simsimd_dot_f16x8_state_neon_t;
/** @copydoc simsimd_dot_f16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_init_neon(simsimd_dot_f16x8_state_neon_t *state);
/** @copydoc simsimd_dot_f16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_update_neon(simsimd_dot_f16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_f16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_finalize_neon(simsimd_dot_f16x8_state_neon_t const *state_a,
                                                      simsimd_dot_f16x8_state_neon_t const *state_b,
                                                      simsimd_dot_f16x8_state_neon_t const *state_c,
                                                      simsimd_dot_f16x8_state_neon_t const *state_d,
                                                      simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_I8
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_i32_t *result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_u32_t *result);

typedef struct simsimd_dot_i8x16_state_neon_t simsimd_dot_i8x16_state_neon_t;
/** @copydoc simsimd_dot_i8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_init_neon(simsimd_dot_i8x16_state_neon_t *state);
/** @copydoc simsimd_dot_i8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_update_neon(simsimd_dot_i8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_i8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_finalize_neon(simsimd_dot_i8x16_state_neon_t const *state_a,
                                                      simsimd_dot_i8x16_state_neon_t const *state_b,
                                                      simsimd_dot_i8x16_state_neon_t const *state_c,
                                                      simsimd_dot_i8x16_state_neon_t const *state_d,
                                                      simsimd_i32_t *results);

typedef struct simsimd_dot_u8x16_state_neon_t simsimd_dot_u8x16_state_neon_t;
/** @copydoc simsimd_dot_u8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_init_neon(simsimd_dot_u8x16_state_neon_t *state);
/** @copydoc simsimd_dot_u8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_update_neon(simsimd_dot_u8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_u8x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_finalize_neon(simsimd_dot_u8x16_state_neon_t const *state_a,
                                                      simsimd_dot_u8x16_state_neon_t const *state_b,
                                                      simsimd_dot_u8x16_state_neon_t const *state_c,
                                                      simsimd_dot_u8x16_state_neon_t const *state_d,
                                                      simsimd_u32_t *results);
#endif // SIMSIMD_TARGET_NEON_I8

#if SIMSIMD_TARGET_NEON_BF16
/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_f32_t *result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_neon(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                           simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_neon(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                            simsimd_f32c_t *result);

typedef struct simsimd_dot_bf16x8_state_neon_t simsimd_dot_bf16x8_state_neon_t;
/** @copydoc simsimd_dot_bf16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_init_neon(simsimd_dot_bf16x8_state_neon_t *state);
/** @copydoc simsimd_dot_bf16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_update_neon(simsimd_dot_bf16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_bf16x8_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_finalize_neon(simsimd_dot_bf16x8_state_neon_t const *state_a,
                                                       simsimd_dot_bf16x8_state_neon_t const *state_b,
                                                       simsimd_dot_bf16x8_state_neon_t const *state_c,
                                                       simsimd_dot_bf16x8_state_neon_t const *state_d,
                                                       simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_SVE
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_sve(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_sve(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                         simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_sve(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                          simsimd_f32c_t *result);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_sve(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                        simsimd_f32_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_sve(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                         simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_sve(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                          simsimd_f32c_t *result);

/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_sve(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                        simsimd_f64_t *result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_sve(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                         simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_sve(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                          simsimd_f64c_t *result);
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_haswell(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_haswell(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                              simsimd_f32c_t *result);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_haswell(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_haswell(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                              simsimd_f32c_t *result);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_haswell(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_haswell(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);

/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                           simsimd_i32_t *result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                           simsimd_u32_t *result);

typedef struct simsimd_dot_f32x8_state_haswell_t simsimd_dot_f32x8_state_haswell_t;
/** @copydoc simsimd_dot_f32x8_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x8_init_haswell(simsimd_dot_f32x8_state_haswell_t *state);
/** @copydoc simsimd_dot_f32x8_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x8_update_haswell(simsimd_dot_f32x8_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_f32x8_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x8_finalize_haswell(simsimd_dot_f32x8_state_haswell_t const *state_a,
                                                         simsimd_dot_f32x8_state_haswell_t const *state_b,
                                                         simsimd_dot_f32x8_state_haswell_t const *state_c,
                                                         simsimd_dot_f32x8_state_haswell_t const *state_d,
                                                         simsimd_f32_t *results);

typedef struct simsimd_dot_f16x16_state_haswell_t simsimd_dot_f16x16_state_haswell_t;
/** @copydoc simsimd_dot_f16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x16_init_haswell(simsimd_dot_f16x16_state_haswell_t *state);
/** @copydoc simsimd_dot_f16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x16_update_haswell(simsimd_dot_f16x16_state_haswell_t *state, simsimd_b256_vec_t a,
                                                        simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_f16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x16_finalize_haswell(simsimd_dot_f16x16_state_haswell_t const *state_a,
                                                          simsimd_dot_f16x16_state_haswell_t const *state_b,
                                                          simsimd_dot_f16x16_state_haswell_t const *state_c,
                                                          simsimd_dot_f16x16_state_haswell_t const *state_d,
                                                          simsimd_f32_t *results);

typedef struct simsimd_dot_bf16x16_state_haswell_t simsimd_dot_bf16x16_state_haswell_t;
/** @copydoc simsimd_dot_bf16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x16_init_haswell(simsimd_dot_bf16x16_state_haswell_t *state);
/** @copydoc simsimd_dot_bf16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x16_update_haswell(simsimd_dot_bf16x16_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_bf16x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x16_finalize_haswell(simsimd_dot_bf16x16_state_haswell_t const *state_a,
                                                           simsimd_dot_bf16x16_state_haswell_t const *state_b,
                                                           simsimd_dot_bf16x16_state_haswell_t const *state_c,
                                                           simsimd_dot_bf16x16_state_haswell_t const *state_d,
                                                           simsimd_f32_t *results);

typedef struct simsimd_dot_e4m3x32_state_haswell_t simsimd_dot_e4m3x32_state_haswell_t;
/** @copydoc simsimd_dot_e4m3x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_init_haswell(simsimd_dot_e4m3x32_state_haswell_t *state);
/** @copydoc simsimd_dot_e4m3x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_update_haswell(simsimd_dot_e4m3x32_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_e4m3x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_finalize_haswell(simsimd_dot_e4m3x32_state_haswell_t const *state_a,
                                                           simsimd_dot_e4m3x32_state_haswell_t const *state_b,
                                                           simsimd_dot_e4m3x32_state_haswell_t const *state_c,
                                                           simsimd_dot_e4m3x32_state_haswell_t const *state_d,
                                                           simsimd_f32_t *results);

typedef struct simsimd_dot_e5m2x32_state_haswell_t simsimd_dot_e5m2x32_state_haswell_t;
/** @copydoc simsimd_dot_e5m2x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_init_haswell(simsimd_dot_e5m2x32_state_haswell_t *state);
/** @copydoc simsimd_dot_e5m2x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_update_haswell(simsimd_dot_e5m2x32_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_e5m2x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_finalize_haswell(simsimd_dot_e5m2x32_state_haswell_t const *state_a,
                                                           simsimd_dot_e5m2x32_state_haswell_t const *state_b,
                                                           simsimd_dot_e5m2x32_state_haswell_t const *state_c,
                                                           simsimd_dot_e5m2x32_state_haswell_t const *state_d,
                                                           simsimd_f32_t *results);

typedef struct simsimd_dot_i8x32_state_haswell_t simsimd_dot_i8x32_state_haswell_t;
/** @copydoc simsimd_dot_i8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_init_haswell(simsimd_dot_i8x32_state_haswell_t *state);
/** @copydoc simsimd_dot_i8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_update_haswell(simsimd_dot_i8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_i8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_finalize_haswell(simsimd_dot_i8x32_state_haswell_t const *state_a,
                                                         simsimd_dot_i8x32_state_haswell_t const *state_b,
                                                         simsimd_dot_i8x32_state_haswell_t const *state_c,
                                                         simsimd_dot_i8x32_state_haswell_t const *state_d,
                                                         simsimd_i32_t *results);

typedef struct simsimd_dot_u8x32_state_haswell_t simsimd_dot_u8x32_state_haswell_t;
/** @copydoc simsimd_dot_u8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x32_init_haswell(simsimd_dot_u8x32_state_haswell_t *state);
/** @copydoc simsimd_dot_u8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x32_update_haswell(simsimd_dot_u8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_u8x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x32_finalize_haswell(simsimd_dot_u8x32_state_haswell_t const *state_a,
                                                         simsimd_dot_u8x32_state_haswell_t const *state_b,
                                                         simsimd_dot_u8x32_state_haswell_t const *state_c,
                                                         simsimd_dot_u8x32_state_haswell_t const *state_d,
                                                         simsimd_u32_t *results);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_f64_t *result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_skylake(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                             simsimd_f64c_t *result);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_skylake(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                              simsimd_f64c_t *result);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_skylake(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_skylake(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                              simsimd_f32c_t *result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_skylake(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_skylake(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);

typedef struct simsimd_dot_f64x8_state_skylake_t simsimd_dot_f64x8_state_skylake_t;
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_skylake(simsimd_dot_f64x8_state_skylake_t *state);
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_skylake(simsimd_dot_f64x8_state_skylake_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_skylake(simsimd_dot_f64x8_state_skylake_t const *state_a,
                                                         simsimd_dot_f64x8_state_skylake_t const *state_b,
                                                         simsimd_dot_f64x8_state_skylake_t const *state_c,
                                                         simsimd_dot_f64x8_state_skylake_t const *state_d,
                                                         simsimd_f64_t *results);

typedef struct simsimd_dot_f32x16_state_skylake_t simsimd_dot_f32x16_state_skylake_t;
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_skylake(simsimd_dot_f32x16_state_skylake_t *state);
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_skylake(simsimd_dot_f32x16_state_skylake_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_skylake(simsimd_dot_f32x16_state_skylake_t const *state_a,
                                                          simsimd_dot_f32x16_state_skylake_t const *state_b,
                                                          simsimd_dot_f32x16_state_skylake_t const *state_c,
                                                          simsimd_dot_f32x16_state_skylake_t const *state_d,
                                                          simsimd_f32_t *results);

typedef struct simsimd_dot_e4m3x64_state_skylake_t simsimd_dot_e4m3x64_state_skylake_t;
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_skylake(simsimd_dot_e4m3x64_state_skylake_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_skylake(simsimd_dot_e4m3x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_skylake(simsimd_dot_e4m3x64_state_skylake_t const *state_a,
                                                           simsimd_dot_e4m3x64_state_skylake_t const *state_b,
                                                           simsimd_dot_e4m3x64_state_skylake_t const *state_c,
                                                           simsimd_dot_e4m3x64_state_skylake_t const *state_d,
                                                           simsimd_f32_t *results);

typedef struct simsimd_dot_e5m2x64_state_skylake_t simsimd_dot_e5m2x64_state_skylake_t;
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_skylake(simsimd_dot_e5m2x64_state_skylake_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_skylake(simsimd_dot_e5m2x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_skylake(simsimd_dot_e5m2x64_state_skylake_t const *state_a,
                                                           simsimd_dot_e5m2x64_state_skylake_t const *state_b,
                                                           simsimd_dot_e5m2x64_state_skylake_t const *state_c,
                                                           simsimd_dot_e5m2x64_state_skylake_t const *state_d,
                                                           simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_ICE
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                       simsimd_i32_t *result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                       simsimd_u32_t *result);

typedef struct simsimd_dot_i8x64_state_ice_t simsimd_dot_i8x64_state_ice_t;
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_ice(simsimd_dot_i8x64_state_ice_t *state);
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_ice(simsimd_dot_i8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_ice(simsimd_dot_i8x64_state_ice_t const *state_a,
                                                     simsimd_dot_i8x64_state_ice_t const *state_b,
                                                     simsimd_dot_i8x64_state_ice_t const *state_c,
                                                     simsimd_dot_i8x64_state_ice_t const *state_d,
                                                     simsimd_i32_t *results);

typedef struct simsimd_dot_u8x64_state_ice_t simsimd_dot_u8x64_state_ice_t;
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_ice(simsimd_dot_u8x64_state_ice_t *state);
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_ice(simsimd_dot_u8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_ice(simsimd_dot_u8x64_state_ice_t const *state_a,
                                                     simsimd_dot_u8x64_state_ice_t const *state_b,
                                                     simsimd_dot_u8x64_state_ice_t const *state_c,
                                                     simsimd_dot_u8x64_state_ice_t const *state_d,
                                                     simsimd_u32_t *results);
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_GENOA
/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_genoa(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_genoa(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                            simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_genoa(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                             simsimd_f32c_t *result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_genoa(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_genoa(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                           simsimd_f32_t *result);

typedef struct simsimd_dot_bf16x32_state_genoa_t simsimd_dot_bf16x32_state_genoa_t;
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_genoa(simsimd_dot_bf16x32_state_genoa_t *state);
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_genoa(simsimd_dot_bf16x32_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_genoa(simsimd_dot_bf16x32_state_genoa_t const *state_a,
                                                         simsimd_dot_bf16x32_state_genoa_t const *state_b,
                                                         simsimd_dot_bf16x32_state_genoa_t const *state_c,
                                                         simsimd_dot_bf16x32_state_genoa_t const *state_d,
                                                         simsimd_f32_t *results);

typedef struct simsimd_dot_e4m3x64_state_genoa_t simsimd_dot_e4m3x64_state_genoa_t;
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_genoa(simsimd_dot_e4m3x64_state_genoa_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_genoa(simsimd_dot_e4m3x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_genoa(simsimd_dot_e4m3x64_state_genoa_t const *state_a,
                                                         simsimd_dot_e4m3x64_state_genoa_t const *state_b,
                                                         simsimd_dot_e4m3x64_state_genoa_t const *state_c,
                                                         simsimd_dot_e4m3x64_state_genoa_t const *state_d,
                                                         simsimd_f32_t *results);

typedef struct simsimd_dot_e5m2x64_state_genoa_t simsimd_dot_e5m2x64_state_genoa_t;
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_genoa(simsimd_dot_e5m2x64_state_genoa_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_genoa(simsimd_dot_e5m2x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_genoa(simsimd_dot_e5m2x64_state_genoa_t const *state_a,
                                                         simsimd_dot_e5m2x64_state_genoa_t const *state_b,
                                                         simsimd_dot_e5m2x64_state_genoa_t const *state_c,
                                                         simsimd_dot_e5m2x64_state_genoa_t const *state_d,
                                                         simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_sapphire(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                              simsimd_f32c_t *result);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_sapphire(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                               simsimd_f32c_t *result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_sapphire(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                              simsimd_f32_t *result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire_lut(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                                  simsimd_f32_t *result);

typedef struct simsimd_dot_f16x32_state_sapphire_t simsimd_dot_f16x32_state_sapphire_t;
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_sapphire(simsimd_dot_f16x32_state_sapphire_t *state);
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_sapphire(simsimd_dot_f16x32_state_sapphire_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_sapphire(simsimd_dot_f16x32_state_sapphire_t const *state_a,
                                                           simsimd_dot_f16x32_state_sapphire_t const *state_b,
                                                           simsimd_dot_f16x32_state_sapphire_t const *state_c,
                                                           simsimd_dot_f16x32_state_sapphire_t const *state_d,
                                                           simsimd_f32_t *results);

typedef struct simsimd_dot_e4m3x64_state_sapphire_t simsimd_dot_e4m3x64_state_sapphire_t;
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_sapphire(simsimd_dot_e4m3x64_state_sapphire_t const *state_a,
                                                            simsimd_dot_e4m3x64_state_sapphire_t const *state_b,
                                                            simsimd_dot_e4m3x64_state_sapphire_t const *state_c,
                                                            simsimd_dot_e4m3x64_state_sapphire_t const *state_d,
                                                            simsimd_f32_t *results);

typedef struct simsimd_dot_e5m2x64_state_sapphire_t simsimd_dot_e5m2x64_state_sapphire_t;
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_sapphire(simsimd_dot_e5m2x64_state_sapphire_t const *state_a,
                                                            simsimd_dot_e5m2x64_state_sapphire_t const *state_b,
                                                            simsimd_dot_e5m2x64_state_sapphire_t const *state_c,
                                                            simsimd_dot_e5m2x64_state_sapphire_t const *state_d,
                                                            simsimd_f32_t *results);
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_SIERRA
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_sierra(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                          simsimd_i32_t *result);

typedef struct simsimd_dot_i8x32_state_sierra_t simsimd_dot_i8x32_state_sierra_t;
/** @copydoc simsimd_dot_i8x32_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_init_sierra(simsimd_dot_i8x32_state_sierra_t *state);
/** @copydoc simsimd_dot_i8x32_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_update_sierra(simsimd_dot_i8x32_state_sierra_t *state, simsimd_b256_vec_t a,
                                                      simsimd_b256_vec_t b);
/** @copydoc simsimd_dot_i8x32_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x32_finalize_sierra(simsimd_dot_i8x32_state_sierra_t const *state_a,
                                                        simsimd_dot_i8x32_state_sierra_t const *state_b,
                                                        simsimd_dot_i8x32_state_sierra_t const *state_c,
                                                        simsimd_dot_i8x32_state_sierra_t const *state_d,
                                                        simsimd_i32_t *results);
#endif // SIMSIMD_TARGET_SIERRA

typedef struct simsimd_dot_f64x2_state_serial_t simsimd_dot_f64x2_state_serial_t;
/** @copydoc simsimd_dot_f64x2_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x2_init_serial(simsimd_dot_f64x2_state_serial_t *state);
/** @copydoc simsimd_dot_f64x2_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x2_update_serial(simsimd_dot_f64x2_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_f64x2_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x2_finalize_serial(simsimd_dot_f64x2_state_serial_t const *state_a,
                                                        simsimd_dot_f64x2_state_serial_t const *state_b,
                                                        simsimd_dot_f64x2_state_serial_t const *state_c,
                                                        simsimd_dot_f64x2_state_serial_t const *state_d,
                                                        simsimd_f64_t *results);

typedef struct simsimd_dot_f32x4_state_serial_t simsimd_dot_f32x4_state_serial_t;
/** @copydoc simsimd_dot_f32x4_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_init_serial(simsimd_dot_f32x4_state_serial_t *state);
/** @copydoc simsimd_dot_f32x4_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_update_serial(simsimd_dot_f32x4_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_f32x4_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x4_finalize_serial(simsimd_dot_f32x4_state_serial_t const *state_a,
                                                        simsimd_dot_f32x4_state_serial_t const *state_b,
                                                        simsimd_dot_f32x4_state_serial_t const *state_c,
                                                        simsimd_dot_f32x4_state_serial_t const *state_d,
                                                        simsimd_f32_t *results);

typedef struct simsimd_dot_f16x8_state_serial_t simsimd_dot_f16x8_state_serial_t;
/** @copydoc simsimd_dot_f16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_init_serial(simsimd_dot_f16x8_state_serial_t *state);
/** @copydoc simsimd_dot_f16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_update_serial(simsimd_dot_f16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_f16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x8_finalize_serial(simsimd_dot_f16x8_state_serial_t const *state_a,
                                                        simsimd_dot_f16x8_state_serial_t const *state_b,
                                                        simsimd_dot_f16x8_state_serial_t const *state_c,
                                                        simsimd_dot_f16x8_state_serial_t const *state_d,
                                                        simsimd_f32_t *results);

typedef struct simsimd_dot_bf16x8_state_serial_t simsimd_dot_bf16x8_state_serial_t;
/** @copydoc simsimd_dot_bf16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_init_serial(simsimd_dot_bf16x8_state_serial_t *state);
/** @copydoc simsimd_dot_bf16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_update_serial(simsimd_dot_bf16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                       simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_bf16x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x8_finalize_serial(simsimd_dot_bf16x8_state_serial_t const *state_a,
                                                         simsimd_dot_bf16x8_state_serial_t const *state_b,
                                                         simsimd_dot_bf16x8_state_serial_t const *state_c,
                                                         simsimd_dot_bf16x8_state_serial_t const *state_d,
                                                         simsimd_f32_t *results);

typedef struct simsimd_dot_i8x16_state_serial_t simsimd_dot_i8x16_state_serial_t;
/** @copydoc simsimd_dot_i8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_init_serial(simsimd_dot_i8x16_state_serial_t *state);
/** @copydoc simsimd_dot_i8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_update_serial(simsimd_dot_i8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_i8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x16_finalize_serial(simsimd_dot_i8x16_state_serial_t const *state_a,
                                                        simsimd_dot_i8x16_state_serial_t const *state_b,
                                                        simsimd_dot_i8x16_state_serial_t const *state_c,
                                                        simsimd_dot_i8x16_state_serial_t const *state_d,
                                                        simsimd_i32_t *results);

typedef struct simsimd_dot_u8x16_state_serial_t simsimd_dot_u8x16_state_serial_t;
/** @copydoc simsimd_dot_u8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_init_serial(simsimd_dot_u8x16_state_serial_t *state);
/** @copydoc simsimd_dot_u8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_update_serial(simsimd_dot_u8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_u8x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x16_finalize_serial(simsimd_dot_u8x16_state_serial_t const *state_a,
                                                        simsimd_dot_u8x16_state_serial_t const *state_b,
                                                        simsimd_dot_u8x16_state_serial_t const *state_c,
                                                        simsimd_dot_u8x16_state_serial_t const *state_d,
                                                        simsimd_u32_t *results);

typedef struct simsimd_dot_e4m3x16_state_serial_t simsimd_dot_e4m3x16_state_serial_t;
/** @copydoc simsimd_dot_e4m3x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_init_serial(simsimd_dot_e4m3x16_state_serial_t *state);
/** @copydoc simsimd_dot_e4m3x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_update_serial(simsimd_dot_e4m3x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_e4m3x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_finalize_serial(simsimd_dot_e4m3x16_state_serial_t const *state_a,
                                                          simsimd_dot_e4m3x16_state_serial_t const *state_b,
                                                          simsimd_dot_e4m3x16_state_serial_t const *state_c,
                                                          simsimd_dot_e4m3x16_state_serial_t const *state_d,
                                                          simsimd_f32_t *results);

typedef struct simsimd_dot_e5m2x16_state_serial_t simsimd_dot_e5m2x16_state_serial_t;
/** @copydoc simsimd_dot_e5m2x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_init_serial(simsimd_dot_e5m2x16_state_serial_t *state);
/** @copydoc simsimd_dot_e5m2x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_update_serial(simsimd_dot_e5m2x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b);
/** @copydoc simsimd_dot_e5m2x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_finalize_serial(simsimd_dot_e5m2x16_state_serial_t const *state_a,
                                                          simsimd_dot_e5m2x16_state_serial_t const *state_b,
                                                          simsimd_dot_e5m2x16_state_serial_t const *state_c,
                                                          simsimd_dot_e5m2x16_state_serial_t const *state_d,
                                                          simsimd_f32_t *results);

#define SIMSIMD_MAKE_DOT(name, input_type, accumulator_type, output_type, load_and_convert)                    \
    SIMSIMD_PUBLIC void simsimd_dot_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_##output_type##_t *result) {                 \
        simsimd_##accumulator_type##_t ab = 0, ai, bi;                                                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            load_and_convert(a + i, &ai);                                                                      \
            load_and_convert(b + i, &bi);                                                                      \
            ab += ai * bi;                                                                                     \
        }                                                                                                      \
        *result = (simsimd_##output_type##_t)ab;                                                               \
    }

#define SIMSIMD_MAKE_COMPLEX_DOT(name, input_type, accumulator_type, output_complex_type, load_and_convert)           \
    SIMSIMD_PUBLIC void simsimd_dot_##input_type##_##name(                                                            \
        simsimd_##input_type##_t const *a_pairs, simsimd_##input_type##_t const *b_pairs, simsimd_size_t count_pairs, \
        simsimd_##output_complex_type##_t *result) {                                                                  \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                      \
        for (simsimd_size_t i = 0; i != count_pairs; ++i) {                                                           \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                              \
            load_and_convert(&(b_pairs + i)->real, &br);                                                              \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                              \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                              \
            ab_real += ar * br - ai * bi;                                                                             \
            ab_imag += ar * bi + ai * br;                                                                             \
        }                                                                                                             \
        result->real = ab_real;                                                                                       \
        result->imag = ab_imag;                                                                                       \
    }

#define SIMSIMD_MAKE_COMPLEX_VDOT(name, input_type, accumulator_type, output_complex_type, load_and_convert)          \
    SIMSIMD_PUBLIC void simsimd_vdot_##input_type##_##name(                                                           \
        simsimd_##input_type##_t const *a_pairs, simsimd_##input_type##_t const *b_pairs, simsimd_size_t count_pairs, \
        simsimd_##output_complex_type##_t *result) {                                                                  \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                      \
        for (simsimd_size_t i = 0; i != count_pairs; ++i) {                                                           \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                              \
            load_and_convert(&(b_pairs + i)->real, &br);                                                              \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                              \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                              \
            ab_real += ar * br + ai * bi;                                                                             \
            ab_imag += ar * bi - ai * br;                                                                             \
        }                                                                                                             \
        result->real = ab_real;                                                                                       \
        result->imag = ab_imag;                                                                                       \
    }

SIMSIMD_MAKE_DOT(serial, f64, f64, f64, SIMSIMD_ASSIGN_FROM_TO)            // simsimd_dot_f64_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f64c, f64, f64c, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f64c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f64c, f64, f64c, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f64c_serial

SIMSIMD_MAKE_DOT(serial, f32, f32, f32, SIMSIMD_ASSIGN_FROM_TO)            // simsimd_dot_f32_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f32c, f32, f32c, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f32c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f32c, f32, f32c, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f32c_serial

SIMSIMD_MAKE_DOT(serial, f16, f32, f32, simsimd_f16_to_f32)            // simsimd_dot_f16_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f16c, f32, f32c, simsimd_f16_to_f32)  // simsimd_dot_f16c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f16c, f32, f32c, simsimd_f16_to_f32) // simsimd_vdot_f16c_serial

SIMSIMD_MAKE_DOT(serial, bf16, f32, f32, simsimd_bf16_to_f32)            // simsimd_dot_bf16_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, bf16c, f32, f32c, simsimd_bf16_to_f32)  // simsimd_dot_bf16c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, bf16c, f32, f32c, simsimd_bf16_to_f32) // simsimd_vdot_bf16c_serial

SIMSIMD_MAKE_DOT(serial, i8, i64, i32, SIMSIMD_ASSIGN_FROM_TO) // simsimd_dot_i8_serial
SIMSIMD_MAKE_DOT(serial, u8, u64, u32, SIMSIMD_ASSIGN_FROM_TO) // simsimd_dot_u8_serial

SIMSIMD_PUBLIC void simsimd_dot_e4m3_serial(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_f32_t ab = 0, ai, bi;
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_e4m3_to_f32(a + i, &ai);
        simsimd_e4m3_to_f32(b + i, &bi);
        ab += ai * bi;
    }
    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_serial(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    simsimd_f32_t ab = 0, ai, bi;
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_e5m2_to_f32(a + i, &ai);
        simsimd_e5m2_to_f32(b + i, &bi);
        ab += ai * bi;
    }
    *result = ab;
}

SIMSIMD_MAKE_DOT(accurate, f32, f64, f64, SIMSIMD_ASSIGN_FROM_TO)            // simsimd_dot_f32_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f32c, f64, f64c, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f32c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f32c, f64, f64c, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f32c_accurate

SIMSIMD_MAKE_DOT(accurate, f16, f64, f64, simsimd_f16_to_f64)            // simsimd_dot_f16_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f16c, f64, f64c, simsimd_f16_to_f64)  // simsimd_dot_f16c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f16c, f64, f64c, simsimd_f16_to_f64) // simsimd_vdot_f16c_accurate

SIMSIMD_MAKE_DOT(accurate, bf16, f64, f64, simsimd_bf16_to_f64)            // simsimd_dot_bf16_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, bf16c, f64, f64c, simsimd_bf16_to_f64)  // simsimd_dot_bf16c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, bf16c, f64, f64c, simsimd_bf16_to_f64) // simsimd_vdot_bf16c_accurate

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars.
 */
typedef struct simsimd_dot_f64x2_state_serial_t {
    simsimd_f64_t sums[2];
} simsimd_dot_f64x2_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f64x2_init_serial(simsimd_dot_f64x2_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f64x2_update_serial(simsimd_dot_f64x2_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_f64_t sum0 = state->sums[0];
    simsimd_f64_t sum1 = state->sums[1];

    sum0 += a.f64s[0] * b.f64s[0], sum1 += a.f64s[1] * b.f64s[1];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_f64x2_finalize_serial(                                              //
    simsimd_dot_f64x2_state_serial_t const *state_a, simsimd_dot_f64x2_state_serial_t const *state_b, //
    simsimd_dot_f64x2_state_serial_t const *state_c, simsimd_dot_f64x2_state_serial_t const *state_d, //
    simsimd_f64_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1];
    results[1] = state_b->sums[0] + state_b->sums[1];
    results[2] = state_c->sums[0] + state_c->sums[1];
    results[3] = state_d->sums[0] + state_d->sums[1];
}

/**
 *  @brief Running state for 128-bit dot accumulation over f32 scalars.
 */
typedef struct simsimd_dot_f32x4_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_f32x4_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x4_init_serial(simsimd_dot_f32x4_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x4_update_serial(simsimd_dot_f32x4_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];

    sum0 += a.f32s[0] * b.f32s[0], sum1 += a.f32s[1] * b.f32s[1];
    sum2 += a.f32s[2] * b.f32s[2], sum3 += a.f32s[3] * b.f32s[3];

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x4_finalize_serial(                                              //
    simsimd_dot_f32x4_state_serial_t const *state_a, simsimd_dot_f32x4_state_serial_t const *state_b, //
    simsimd_dot_f32x4_state_serial_t const *state_c, simsimd_dot_f32x4_state_serial_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars.
 */
typedef struct simsimd_dot_f16x8_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_f16x8_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x8_init_serial(simsimd_dot_f16x8_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x8_update_serial(simsimd_dot_f16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (simsimd_size_t i = 0; i < 8; i += 4) {
        simsimd_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        simsimd_f16_to_f32(a.f16s + i + 0, &a0);
        simsimd_f16_to_f32(a.f16s + i + 1, &a1);
        simsimd_f16_to_f32(a.f16s + i + 2, &a2);
        simsimd_f16_to_f32(a.f16s + i + 3, &a3);
        simsimd_f16_to_f32(b.f16s + i + 0, &b0);
        simsimd_f16_to_f32(b.f16s + i + 1, &b1);
        simsimd_f16_to_f32(b.f16s + i + 2, &b2);
        simsimd_f16_to_f32(b.f16s + i + 3, &b3);
        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x8_finalize_serial(                                              //
    simsimd_dot_f16x8_state_serial_t const *state_a, simsimd_dot_f16x8_state_serial_t const *state_b, //
    simsimd_dot_f16x8_state_serial_t const *state_c, simsimd_dot_f16x8_state_serial_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars.
 */
typedef struct simsimd_dot_bf16x8_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_bf16x8_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_init_serial(simsimd_dot_bf16x8_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_update_serial(simsimd_dot_bf16x8_state_serial_t *state, simsimd_b128_vec_t a,
                                                       simsimd_b128_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (simsimd_size_t i = 0; i < 8; i += 4) {
        simsimd_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        simsimd_bf16_to_f32(a.bf16s + i + 0, &a0);
        simsimd_bf16_to_f32(a.bf16s + i + 1, &a1);
        simsimd_bf16_to_f32(a.bf16s + i + 2, &a2);
        simsimd_bf16_to_f32(a.bf16s + i + 3, &a3);
        simsimd_bf16_to_f32(b.bf16s + i + 0, &b0);
        simsimd_bf16_to_f32(b.bf16s + i + 1, &b1);
        simsimd_bf16_to_f32(b.bf16s + i + 2, &b2);
        simsimd_bf16_to_f32(b.bf16s + i + 3, &b3);
        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_finalize_serial(                                               //
    simsimd_dot_bf16x8_state_serial_t const *state_a, simsimd_dot_bf16x8_state_serial_t const *state_b, //
    simsimd_dot_bf16x8_state_serial_t const *state_c, simsimd_dot_bf16x8_state_serial_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars.
 */
typedef struct simsimd_dot_i8x16_state_serial_t {
    simsimd_i64_t sums[2];
} simsimd_dot_i8x16_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x16_init_serial(simsimd_dot_i8x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x16_update_serial(simsimd_dot_i8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_i64_t sum0 = state->sums[0];
    simsimd_i64_t sum1 = state->sums[1];

    sum0 += (simsimd_i16_t)a.i8s[0] * (simsimd_i16_t)b.i8s[0];
    sum1 += (simsimd_i16_t)a.i8s[1] * (simsimd_i16_t)b.i8s[1];
    sum0 += (simsimd_i16_t)a.i8s[2] * (simsimd_i16_t)b.i8s[2];
    sum1 += (simsimd_i16_t)a.i8s[3] * (simsimd_i16_t)b.i8s[3];
    sum0 += (simsimd_i16_t)a.i8s[4] * (simsimd_i16_t)b.i8s[4];
    sum1 += (simsimd_i16_t)a.i8s[5] * (simsimd_i16_t)b.i8s[5];
    sum0 += (simsimd_i16_t)a.i8s[6] * (simsimd_i16_t)b.i8s[6];
    sum1 += (simsimd_i16_t)a.i8s[7] * (simsimd_i16_t)b.i8s[7];
    sum0 += (simsimd_i16_t)a.i8s[8] * (simsimd_i16_t)b.i8s[8];
    sum1 += (simsimd_i16_t)a.i8s[9] * (simsimd_i16_t)b.i8s[9];
    sum0 += (simsimd_i16_t)a.i8s[10] * (simsimd_i16_t)b.i8s[10];
    sum1 += (simsimd_i16_t)a.i8s[11] * (simsimd_i16_t)b.i8s[11];
    sum0 += (simsimd_i16_t)a.i8s[12] * (simsimd_i16_t)b.i8s[12];
    sum1 += (simsimd_i16_t)a.i8s[13] * (simsimd_i16_t)b.i8s[13];
    sum0 += (simsimd_i16_t)a.i8s[14] * (simsimd_i16_t)b.i8s[14];
    sum1 += (simsimd_i16_t)a.i8s[15] * (simsimd_i16_t)b.i8s[15];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x16_finalize_serial(                                              //
    simsimd_dot_i8x16_state_serial_t const *state_a, simsimd_dot_i8x16_state_serial_t const *state_b, //
    simsimd_dot_i8x16_state_serial_t const *state_c, simsimd_dot_i8x16_state_serial_t const *state_d, //
    simsimd_i32_t *results) {
    results[0] = (simsimd_i32_t)(state_a->sums[0] + state_a->sums[1]);
    results[1] = (simsimd_i32_t)(state_b->sums[0] + state_b->sums[1]);
    results[2] = (simsimd_i32_t)(state_c->sums[0] + state_c->sums[1]);
    results[3] = (simsimd_i32_t)(state_d->sums[0] + state_d->sums[1]);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars.
 */
typedef struct simsimd_dot_u8x16_state_serial_t {
    simsimd_u64_t sums[2];
} simsimd_dot_u8x16_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x16_init_serial(simsimd_dot_u8x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x16_update_serial(simsimd_dot_u8x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                      simsimd_b128_vec_t b) {
    simsimd_u64_t sum0 = state->sums[0];
    simsimd_u64_t sum1 = state->sums[1];

    sum0 += (simsimd_u16_t)a.u8s[0] * (simsimd_u16_t)b.u8s[0];
    sum1 += (simsimd_u16_t)a.u8s[1] * (simsimd_u16_t)b.u8s[1];
    sum0 += (simsimd_u16_t)a.u8s[2] * (simsimd_u16_t)b.u8s[2];
    sum1 += (simsimd_u16_t)a.u8s[3] * (simsimd_u16_t)b.u8s[3];
    sum0 += (simsimd_u16_t)a.u8s[4] * (simsimd_u16_t)b.u8s[4];
    sum1 += (simsimd_u16_t)a.u8s[5] * (simsimd_u16_t)b.u8s[5];
    sum0 += (simsimd_u16_t)a.u8s[6] * (simsimd_u16_t)b.u8s[6];
    sum1 += (simsimd_u16_t)a.u8s[7] * (simsimd_u16_t)b.u8s[7];
    sum0 += (simsimd_u16_t)a.u8s[8] * (simsimd_u16_t)b.u8s[8];
    sum1 += (simsimd_u16_t)a.u8s[9] * (simsimd_u16_t)b.u8s[9];
    sum0 += (simsimd_u16_t)a.u8s[10] * (simsimd_u16_t)b.u8s[10];
    sum1 += (simsimd_u16_t)a.u8s[11] * (simsimd_u16_t)b.u8s[11];
    sum0 += (simsimd_u16_t)a.u8s[12] * (simsimd_u16_t)b.u8s[12];
    sum1 += (simsimd_u16_t)a.u8s[13] * (simsimd_u16_t)b.u8s[13];
    sum0 += (simsimd_u16_t)a.u8s[14] * (simsimd_u16_t)b.u8s[14];
    sum1 += (simsimd_u16_t)a.u8s[15] * (simsimd_u16_t)b.u8s[15];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x16_finalize_serial(                                              //
    simsimd_dot_u8x16_state_serial_t const *state_a, simsimd_dot_u8x16_state_serial_t const *state_b, //
    simsimd_dot_u8x16_state_serial_t const *state_c, simsimd_dot_u8x16_state_serial_t const *state_d, //
    simsimd_u32_t *results) {
    results[0] = (simsimd_u32_t)(state_a->sums[0] + state_a->sums[1]);
    results[1] = (simsimd_u32_t)(state_b->sums[0] + state_b->sums[1]);
    results[2] = (simsimd_u32_t)(state_c->sums[0] + state_c->sums[1]);
    results[3] = (simsimd_u32_t)(state_d->sums[0] + state_d->sums[1]);
}

/**
 *  @brief Running state for 128-bit dot accumulation over e4m3 scalars.
 */
typedef struct simsimd_dot_e4m3x16_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_e4m3x16_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_init_serial(simsimd_dot_e4m3x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_update_serial(simsimd_dot_e4m3x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];
    simsimd_f32_t ai0, ai1, ai2, ai3;
    simsimd_f32_t bi0, bi1, bi2, bi3;
    for (simsimd_size_t i = 0; i != 16; i += 4) {
        simsimd_e4m3_to_f32(a.e4m3s + i, &ai0);
        simsimd_e4m3_to_f32(b.e4m3s + i, &bi0);
        simsimd_e4m3_to_f32(a.e4m3s + i + 1, &ai1);
        simsimd_e4m3_to_f32(b.e4m3s + i + 1, &bi1);
        simsimd_e4m3_to_f32(a.e4m3s + i + 2, &ai2);
        simsimd_e4m3_to_f32(b.e4m3s + i + 2, &bi2);
        simsimd_e4m3_to_f32(a.e4m3s + i + 3, &ai3);
        simsimd_e4m3_to_f32(b.e4m3s + i + 3, &bi3);
        sum0 += ai0 * bi0;
        sum1 += ai1 * bi1;
        sum2 += ai2 * bi2;
        sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x16_finalize_serial(                                                //
    simsimd_dot_e4m3x16_state_serial_t const *state_a, simsimd_dot_e4m3x16_state_serial_t const *state_b, //
    simsimd_dot_e4m3x16_state_serial_t const *state_c, simsimd_dot_e4m3x16_state_serial_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

/**
 *  @brief Running state for 128-bit dot accumulation over e5m2 scalars.
 */
typedef struct simsimd_dot_e5m2x16_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_e5m2x16_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_init_serial(simsimd_dot_e5m2x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_update_serial(simsimd_dot_e5m2x16_state_serial_t *state, simsimd_b128_vec_t a,
                                                        simsimd_b128_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];
    simsimd_f32_t ai0, ai1, ai2, ai3;
    simsimd_f32_t bi0, bi1, bi2, bi3;
    for (simsimd_size_t i = 0; i != 16; i += 4) {
        simsimd_e5m2_to_f32(a.e5m2s + i, &ai0);
        simsimd_e5m2_to_f32(b.e5m2s + i, &bi0);
        simsimd_e5m2_to_f32(a.e5m2s + i + 1, &ai1);
        simsimd_e5m2_to_f32(b.e5m2s + i + 1, &bi1);
        simsimd_e5m2_to_f32(a.e5m2s + i + 2, &ai2);
        simsimd_e5m2_to_f32(b.e5m2s + i + 2, &bi2);
        simsimd_e5m2_to_f32(a.e5m2s + i + 3, &ai3);
        simsimd_e5m2_to_f32(b.e5m2s + i + 3, &bi3);
        sum0 += ai0 * bi0;
        sum1 += ai1 * bi1;
        sum2 += ai2 * bi2;
        sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x16_finalize_serial(                                                //
    simsimd_dot_e5m2x16_state_serial_t const *state_a, simsimd_dot_e5m2x16_state_serial_t const *state_b, //
    simsimd_dot_e5m2x16_state_serial_t const *state_c, simsimd_dot_e5m2x16_state_serial_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

SIMSIMD_INTERNAL float32x4_t _simsimd_partial_load_f32x4_neon(simsimd_f32_t const *x, simsimd_size_t n) {
    simsimd_b512_vec_t result;
    result.u32x4s[0] = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.f32s[i] = x[i];
    return vreinterpretq_f32_u32(result.u32x4s[0]);
}

SIMSIMD_INTERNAL void _simsimd_partial_store_f32x4_neon(float32x4_t vec, simsimd_f32_t *x, simsimd_size_t n) {
    simsimd_b512_vec_t u;
    u.u32x4s[0] = vreinterpretq_u32_f32(vec);
    if (n > 0) x[0] = u.f32s[0];
    if (n > 1) x[1] = u.f32s[1];
    if (n > 2) x[2] = u.f32s[2];
    if (n > 3) x[3] = u.f32s[3];
}

SIMSIMD_INTERNAL void _simsimd_partial_store_i32x4_neon(int32x4_t vec, simsimd_i32_t *x, simsimd_size_t n) {
    simsimd_b512_vec_t u;
    u.u32x4s[0] = vreinterpretq_u32_s32(vec);
    if (n > 0) x[0] = u.i32s[0];
    if (n > 1) x[1] = u.i32s[1];
    if (n > 2) x[2] = u.i32s[2];
    if (n > 3) x[3] = u.i32s[3];
}

SIMSIMD_PUBLIC void simsimd_dot_f32_neon(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                         simsimd_size_t count_scalars, simsimd_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a_scalars + idx_scalars);
        float32x4_t b_f32x4 = vld1q_f32(b_scalars + idx_scalars);
        sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_f32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_neon(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_f32x4x2 = vld2q_f32((simsimd_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_f32x4x2 = vld2q_f32((simsimd_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_f32x4 = a_f32x4x2.val[0];
        float32x4_t a_imag_f32x4 = a_f32x4x2.val[1];
        float32x4_t b_real_f32x4 = b_f32x4x2.val[0];
        float32x4_t b_imag_f32x4 = b_f32x4x2.val[1];

        // Compute the dot product:
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
    }

    // Reduce horizontal sums:
    simsimd_f32_t sum_real = vaddvq_f32(sum_real_f32x4);
    simsimd_f32_t sum_imag = vaddvq_f32(sum_imag_f32x4);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br - ai * bi;
        sum_imag += ar * bi + ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_neon(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_f32x4x2 = vld2q_f32((simsimd_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_f32x4x2 = vld2q_f32((simsimd_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_f32x4 = a_f32x4x2.val[0];
        float32x4_t a_imag_f32x4 = a_f32x4x2.val[1];
        float32x4_t b_real_f32x4 = b_f32x4x2.val[0];
        float32x4_t b_imag_f32x4 = b_f32x4x2.val[1];

        // Compute the dot product:
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
    }

    // Reduce horizontal sums:
    simsimd_f32_t sum_real = vaddvq_f32(sum_real_f32x4);
    simsimd_f32_t sum_imag = vaddvq_f32(sum_imag_f32x4);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br + ai * bi;
        sum_imag += ar * bi - ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

/**
 *  @brief Running state for 128-bit dot accumulation over f32 scalars on NEON.
 */
typedef struct simsimd_dot_f32x4_state_neon_t {
    float32x4_t sum_f32x4;
} simsimd_dot_f32x4_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x4_init_neon(simsimd_dot_f32x4_state_neon_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_f32x4_update_neon(simsimd_dot_f32x4_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b) {
    float32x4_t sum_f32x4 = state->sum_f32x4;
    sum_f32x4 = vfmaq_f32(sum_f32x4, vreinterpretq_f32_u32(a.u32x4s[0]), vreinterpretq_f32_u32(b.u32x4s[0]));
    state->sum_f32x4 = sum_f32x4;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x4_finalize_neon(                                            //
    simsimd_dot_f32x4_state_neon_t const *state_a, simsimd_dot_f32x4_state_neon_t const *state_b, //
    simsimd_dot_f32x4_state_neon_t const *state_c, simsimd_dot_f32x4_state_neon_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = vaddvq_f32(state_a->sum_f32x4);
    results[1] = vaddvq_f32(state_b->sum_f32x4);
    results[2] = vaddvq_f32(state_c->sum_f32x4);
    results[3] = vaddvq_f32(state_d->sum_f32x4);
}

/** @brief Type-agnostic 128-bit full load (NEON). */
SIMSIMD_INTERNAL void _simsimd_load_b128_neon(void const *src, simsimd_b128_vec_t *dst) {
    dst->u8x16 = vld1q_u8((simsimd_u8_t const *)src);
}

/** @brief Type-agnostic partial load for 32-bit elements (4 elements max) into 128-bit vector (NEON). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b32x4_neon(void const *src, simsimd_size_t n, simsimd_b128_vec_t *dst) {
    simsimd_u32_t const *s = (simsimd_u32_t const *)src;
    dst->u32x4 = vdupq_n_u32(0);
    for (simsimd_size_t i = 0; i < n && i < 4; ++i) dst->u32s[i] = s[i];
}

/** @brief Type-agnostic partial load for 16-bit elements (8 elements max) into 128-bit vector (NEON). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b16x8_neon(void const *src, simsimd_size_t n, simsimd_b128_vec_t *dst) {
    simsimd_u16_t const *s = (simsimd_u16_t const *)src;
    dst->u16x8 = vdupq_n_u16(0);
    for (simsimd_size_t i = 0; i < n && i < 8; ++i) dst->u16s[i] = s[i];
}

/** @brief Type-agnostic partial load for 8-bit elements (16 elements max) into 128-bit vector (NEON). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b8x16_neon(void const *src, simsimd_size_t n, simsimd_b128_vec_t *dst) {
    simsimd_u8_t const *s = (simsimd_u8_t const *)src;
    dst->u8x16 = vdupq_n_u8(0);
    for (simsimd_size_t i = 0; i < n && i < 16; ++i) dst->u8s[i] = s[i];
}

/** @brief Type-agnostic partial store for 32-bit elements (4 elements max) from 128-bit vector (NEON). */
SIMSIMD_INTERNAL void _simsimd_partial_store_b32x4_neon(simsimd_b128_vec_t const *src, void *dst, simsimd_size_t n) {
    simsimd_u32_t *d = (simsimd_u32_t *)dst;
    for (simsimd_size_t i = 0; i < n && i < 4; ++i) d[i] = src->u32s[i];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_i8_neon(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_i32_t *result) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a_scalars + idx_scalars);
        int8x16_t b_i8x16 = vld1q_s8(b_scalars + idx_scalars);
        sum_i32x4 = vdotq_s32(sum_i32x4, a_i8x16, b_i8x16);
    }
    simsimd_i32_t sum = vaddvq_s32(sum_i32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum += (simsimd_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_dot_u8_neon(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_u32_t *result) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a_scalars + idx_scalars);
        uint8x16_t b_u8x16 = vld1q_u8(b_scalars + idx_scalars);
        sum_u32x4 = vdotq_u32(sum_u32x4, a_u8x16, b_u8x16);
    }
    simsimd_u32_t sum = vaddvq_u32(sum_u32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum += (simsimd_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on NEON.
 */
typedef struct simsimd_dot_i8x16_state_neon_t {
    int32x4_t sum_i32x4;
} simsimd_dot_i8x16_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x16_init_neon(simsimd_dot_i8x16_state_neon_t *state) {
    state->sum_i32x4 = vdupq_n_s32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_i8x16_update_neon(simsimd_dot_i8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b) {
    int32x4_t sum_i32x4 = state->sum_i32x4;
    sum_i32x4 = vdotq_s32(sum_i32x4, vreinterpretq_s8_u32(a.u32x4s[0]), vreinterpretq_s8_u32(b.u32x4s[0]));
    state->sum_i32x4 = sum_i32x4;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x16_finalize_neon(                                            //
    simsimd_dot_i8x16_state_neon_t const *state_a, simsimd_dot_i8x16_state_neon_t const *state_b, //
    simsimd_dot_i8x16_state_neon_t const *state_c, simsimd_dot_i8x16_state_neon_t const *state_d, //
    simsimd_i32_t *results) {
    results[0] = (simsimd_i32_t)vaddvq_s32(state_a->sum_i32x4);
    results[1] = (simsimd_i32_t)vaddvq_s32(state_b->sum_i32x4);
    results[2] = (simsimd_i32_t)vaddvq_s32(state_c->sum_i32x4);
    results[3] = (simsimd_i32_t)vaddvq_s32(state_d->sum_i32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on NEON.
 */
typedef struct simsimd_dot_u8x16_state_neon_t {
    uint32x4_t sum_u32x4;
} simsimd_dot_u8x16_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x16_init_neon(simsimd_dot_u8x16_state_neon_t *state) {
    state->sum_u32x4 = vdupq_n_u32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_u8x16_update_neon(simsimd_dot_u8x16_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b) {
    uint32x4_t sum_u32x4 = state->sum_u32x4;
    sum_u32x4 = vdotq_u32(sum_u32x4, vreinterpretq_u8_u32(a.u32x4s[0]), vreinterpretq_u8_u32(b.u32x4s[0]));
    state->sum_u32x4 = sum_u32x4;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x16_finalize_neon(                                            //
    simsimd_dot_u8x16_state_neon_t const *state_a, simsimd_dot_u8x16_state_neon_t const *state_b, //
    simsimd_dot_u8x16_state_neon_t const *state_c, simsimd_dot_u8x16_state_neon_t const *state_d, //
    simsimd_u32_t *results) {
    results[0] = (simsimd_u32_t)vaddvq_u32(state_a->sum_u32x4);
    results[1] = (simsimd_u32_t)vaddvq_u32(state_b->sum_u32x4);
    results[2] = (simsimd_u32_t)vaddvq_u32(state_c->sum_u32x4);
    results[3] = (simsimd_u32_t)vaddvq_u32(state_d->sum_u32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_I8

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_INTERNAL float16x4_t _simsimd_partial_load_f16x4_neon(simsimd_f16_t const *x, simsimd_size_t n) {
    // In case the software emulation for `f16` scalars is enabled, the `simsimd_f16_to_f32`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    simsimd_b512_vec_t result;
    result.u64s[0] = 0;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.f16s[i] = x[i];
    return vreinterpret_f16_u16(vget_low_u16(vreinterpretq_u16_u32(result.u32x4s[0])));
}

SIMSIMD_PUBLIC void simsimd_dot_f16_neon(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                         simsimd_size_t count_scalars, simsimd_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
simsimd_dot_f16_neon_cycle:
    if (count_scalars < 4) {
        a_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a_scalars, count_scalars));
        b_f32x4 = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b_scalars, count_scalars));
        count_scalars = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a_scalars));
        b_f32x4 = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b_scalars));
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    if (count_scalars) goto simsimd_dot_f16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_neon(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    simsimd_f32c_t tail_result;
    simsimd_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_neon(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    simsimd_f32c_t tail_result;
    simsimd_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on NEON.
 */
typedef struct simsimd_dot_f16x8_state_neon_t {
    float32x4_t sum_f32x4;
} simsimd_dot_f16x8_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x8_init_neon(simsimd_dot_f16x8_state_neon_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_f16x8_update_neon(simsimd_dot_f16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                    simsimd_b128_vec_t b) {
    float32x4_t sum_f32x4 = state->sum_f32x4;
    simsimd_f16_t const *a_scalars = a.f16s;
    simsimd_f16_t const *b_scalars = b.f16s;
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 0))),
                          vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 0))));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 4))),
                          vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 4))));
    state->sum_f32x4 = sum_f32x4;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x8_finalize_neon(                                            //
    simsimd_dot_f16x8_state_neon_t const *state_a, simsimd_dot_f16x8_state_neon_t const *state_b, //
    simsimd_dot_f16x8_state_neon_t const *state_c, simsimd_dot_f16x8_state_neon_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = vaddvq_f32(state_a->sum_f32x4);
    results[1] = vaddvq_f32(state_b->sum_f32x4);
    results[2] = vaddvq_f32(state_c->sum_f32x4);
    results[3] = vaddvq_f32(state_d->sum_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_INTERNAL bfloat16x8_t _simsimd_partial_load_bf16x8_neon(simsimd_bf16_t const *x, simsimd_size_t n) {
    simsimd_b512_vec_t result;
    result.u32x4s[0] = vdupq_n_u32(0);
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.bf16s[i] = x[i];
    return vreinterpretq_bf16_u32(result.u32x4s[0]);
}

SIMSIMD_PUBLIC void simsimd_dot_bf16_neon(simsimd_bf16_t const *a_scalars, simsimd_bf16_t const *b_scalars,
                                          simsimd_size_t count_scalars, simsimd_f32_t *result) {
    bfloat16x8_t a_bf16x8, b_bf16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
simsimd_dot_bf16_neon_cycle:
    if (count_scalars < 8) {
        a_bf16x8 = _simsimd_partial_load_bf16x8_neon(a_scalars, count_scalars);
        b_bf16x8 = _simsimd_partial_load_bf16x8_neon(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a_scalars);
        b_bf16x8 = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x4 = vbfdotq_f32(sum_f32x4, a_bf16x8, b_bf16x8);
    if (count_scalars) goto simsimd_dot_bf16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

SIMSIMD_PUBLIC void simsimd_dot_bf16c_neon(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    simsimd_f32c_t tail_result;
    simsimd_dot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

SIMSIMD_PUBLIC void simsimd_vdot_bf16c_neon(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                            simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    simsimd_f32c_t tail_result;
    simsimd_vdot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on NEON.
 */
typedef struct simsimd_dot_bf16x8_state_neon_t {
    float32x4_t sum_f32x4;
} simsimd_dot_bf16x8_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_init_neon(simsimd_dot_bf16x8_state_neon_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_update_neon(simsimd_dot_bf16x8_state_neon_t *state, simsimd_b128_vec_t a,
                                                     simsimd_b128_vec_t b) {
    float32x4_t sum_f32x4 = state->sum_f32x4;
    simsimd_bf16_t const *a_scalars = a.bf16s;
    simsimd_bf16_t const *b_scalars = b.bf16s;
    sum_f32x4 = vbfdotq_f32(sum_f32x4, vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(a_scalars + 0)),
                            vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(b_scalars + 0)));
    state->sum_f32x4 = sum_f32x4;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x8_finalize_neon(                                             //
    simsimd_dot_bf16x8_state_neon_t const *state_a, simsimd_dot_bf16x8_state_neon_t const *state_b, //
    simsimd_dot_bf16x8_state_neon_t const *state_c, simsimd_dot_bf16x8_state_neon_t const *state_d, //
    simsimd_f32_t *results) {
    results[0] = vaddvq_f32(state_a->sum_f32x4);
    results[1] = vaddvq_f32(state_b->sum_f32x4);
    results[2] = vaddvq_f32(state_c->sum_f32x4);
    results[3] = vaddvq_f32(state_d->sum_f32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_SVE

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_f32_sve(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_distance_t *result) {
    simsimd_size_t idx_scalars = 0;
    svfloat32_t ab_vec = svdup_f32(0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat32_t a_vec = svld1_f32(pg_vec, a_scalars + idx_scalars);
        svfloat32_t b_vec = svld1_f32(pg_vec, b_scalars + idx_scalars);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        idx_scalars += svcntw();
    } while (idx_scalars < count_scalars);
    *result = svaddv_f32(svptrue_b32(), ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_sve(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                         simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat32_t ab_real_vec = svdup_f32(0.f);
    svfloat32_t ab_imag_vec = svdup_f32(0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat32x2_t a_vec = svld2_f32(pg_vec, (simsimd_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_vec = svld2_f32(pg_vec, (simsimd_f32_t const *)(b_pairs + idx_pairs));
        svfloat32_t a_real_vec = svget2_f32(a_vec, 0);
        svfloat32_t a_imag_vec = svget2_f32(a_vec, 1);
        svfloat32_t b_real_vec = svget2_f32(b_vec, 0);
        svfloat32_t b_imag_vec = svget2_f32(b_vec, 1);
        ab_real_vec = svmla_f32_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmls_f32_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f32_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmla_f32_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcntw();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f32(svptrue_b32(), ab_real_vec);
    results[1] = svaddv_f32(svptrue_b32(), ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_sve(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat32_t ab_real_vec = svdup_f32(0.f);
    svfloat32_t ab_imag_vec = svdup_f32(0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat32x2_t a_vec = svld2_f32(pg_vec, (simsimd_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_vec = svld2_f32(pg_vec, (simsimd_f32_t const *)(b_pairs + idx_pairs));
        svfloat32_t a_real_vec = svget2_f32(a_vec, 0);
        svfloat32_t a_imag_vec = svget2_f32(a_vec, 1);
        svfloat32_t b_real_vec = svget2_f32(b_vec, 0);
        svfloat32_t b_imag_vec = svget2_f32(b_vec, 1);
        ab_real_vec = svmla_f32_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmla_f32_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f32_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmls_f32_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcntw();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f32(svptrue_b32(), ab_real_vec);
    results[1] = svaddv_f32(svptrue_b32(), ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f64_sve(simsimd_f64_t const *a_scalars, simsimd_f64_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_distance_t *result) {
    simsimd_size_t idx_scalars = 0;
    svfloat64_t ab_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat64_t a_vec = svld1_f64(pg_vec, a_scalars + idx_scalars);
        svfloat64_t b_vec = svld1_f64(pg_vec, b_scalars + idx_scalars);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        idx_scalars += svcntd();
    } while (idx_scalars < count_scalars);
    *result = svaddv_f64(svptrue_b32(), ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f64c_sve(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                         simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat64_t ab_real_vec = svdup_f64(0.);
    svfloat64_t ab_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (simsimd_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (simsimd_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_vec = svget2_f64(a_vec, 0);
        svfloat64_t a_imag_vec = svget2_f64(a_vec, 1);
        svfloat64_t b_real_vec = svget2_f64(b_vec, 0);
        svfloat64_t b_imag_vec = svget2_f64(b_vec, 1);
        ab_real_vec = svmla_f64_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmls_f64_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f64_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmla_f64_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f64(svptrue_b64(), ab_real_vec);
    results[1] = svaddv_f64(svptrue_b64(), ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f64c_sve(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat64_t ab_real_vec = svdup_f64(0.);
    svfloat64_t ab_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (simsimd_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (simsimd_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_vec = svget2_f64(a_vec, 0);
        svfloat64_t a_imag_vec = svget2_f64(a_vec, 1);
        svfloat64_t b_real_vec = svget2_f64(b_vec, 0);
        svfloat64_t b_imag_vec = svget2_f64(b_vec, 1);
        ab_real_vec = svmla_f64_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmla_f64_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f64_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmls_f64_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f64(svptrue_b64(), ab_real_vec);
    results[1] = svaddv_f64(svptrue_b64(), ab_imag_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_f16_sve(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_distance_t *result) {
    simsimd_size_t idx_scalars = 0;
    svfloat16_t ab_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat16_t a_vec = svld1_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(a_scalars + idx_scalars));
        svfloat16_t b_vec = svld1_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(b_scalars + idx_scalars));
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        idx_scalars += svcnth();
    } while (idx_scalars < count_scalars);
    simsimd_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_sve(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                         simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat16_t ab_real_vec = svdup_f16(0);
    svfloat16_t ab_imag_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat16x2_t a_vec = svld2_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(a_pairs + idx_pairs));
        svfloat16x2_t b_vec = svld2_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(b_pairs + idx_pairs));
        svfloat16_t a_real_vec = svget2_f16(a_vec, 0);
        svfloat16_t a_imag_vec = svget2_f16(a_vec, 1);
        svfloat16_t b_real_vec = svget2_f16(b_vec, 0);
        svfloat16_t b_imag_vec = svget2_f16(b_vec, 1);
        ab_real_vec = svmla_f16_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmls_f16_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f16_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmla_f16_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcnth();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f16(svptrue_b16(), ab_real_vec);
    results[1] = svaddv_f16(svptrue_b16(), ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_sve(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_distance_t *results) {
    simsimd_size_t idx_pairs = 0;
    svfloat16_t ab_real_vec = svdup_f16(0);
    svfloat16_t ab_imag_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat16x2_t a_vec = svld2_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(a_pairs + idx_pairs));
        svfloat16x2_t b_vec = svld2_f16(pg_vec, (simsimd_f16_for_arm_simd_t const *)(b_pairs + idx_pairs));
        svfloat16_t a_real_vec = svget2_f16(a_vec, 0);
        svfloat16_t a_imag_vec = svget2_f16(a_vec, 1);
        svfloat16_t b_real_vec = svget2_f16(b_vec, 0);
        svfloat16_t b_imag_vec = svget2_f16(b_vec, 1);
        ab_real_vec = svmla_f16_x(pg_vec, ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = svmla_f16_x(pg_vec, ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = svmla_f16_x(pg_vec, ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = svmls_f16_x(pg_vec, ab_imag_vec, a_imag_vec, b_real_vec);
        idx_pairs += svcnth();
    } while (idx_pairs < count_pairs);
    results[0] = svaddv_f16(svptrue_b16(), ab_real_vec);
    results[1] = svaddv_f16(svptrue_b16(), ab_imag_vec);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE
#endif // _SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_f32_haswell(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count_scalars; idx_scalars += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a_scalars + idx_scalars);
        __m256 b_f32x8 = _mm256_loadu_ps(b_scalars + idx_scalars);
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    }
    simsimd_f32_t sum = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_f32x8);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_haswell(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    // Using XOR to flip sign bits is cheaper than separate FMA/FMS. Throughput doubles from 2.5 GB/s to 5 GB/s.
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps((simsimd_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_f32x8 = _mm256_loadu_ps((simsimd_f32_t const *)(b_pairs + idx_pairs));
        __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
            _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));
    simsimd_f32_t sum_real = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_real_f32x8);
    simsimd_f32_t sum_imag = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_imag_f32x8);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real - a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag + a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_haswell(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps((simsimd_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_f32x8 = _mm256_loadu_ps((simsimd_f32_t const *)(b_pairs + idx_pairs));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        b_f32x8 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_imag_f32x8);
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));
    simsimd_f32_t sum_real = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_real_f32x8);
    simsimd_f32_t sum_imag = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_imag_f32x8);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real + a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag - a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

SIMSIMD_INTERNAL __m256 _simsimd_partial_load_f16x8_haswell(simsimd_f16_t const *a, simsimd_size_t n) {
    // In case the software emulation for `f16` scalars is enabled, the `simsimd_f16_to_f32`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    simsimd_b512_vec_t result;
    result.xmms[0] = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.f16s[i] = a[i];
    return _mm256_cvtph_ps(result.xmms[0]);
}

SIMSIMD_INTERNAL void _simsimd_partial_store_f32x8_haswell(__m256 vec, simsimd_f32_t *x, simsimd_size_t n) {
    simsimd_b512_vec_t u;
    u.ymms_ps[0] = vec;
    if (n > 0) x[0] = u.f32s[0];
    if (n > 1) x[1] = u.f32s[1];
    if (n > 2) x[2] = u.f32s[2];
    if (n > 3) x[3] = u.f32s[3];
    if (n > 4) x[4] = u.f32s[4];
    if (n > 5) x[5] = u.f32s[5];
    if (n > 6) x[6] = u.f32s[6];
    if (n > 7) x[7] = u.f32s[7];
}

SIMSIMD_INTERNAL void _simsimd_partial_store_i32x8_haswell(__m256i vec, simsimd_i32_t *x, simsimd_size_t n) {
    simsimd_b512_vec_t u;
    u.ymms[0] = vec;
    if (n > 0) x[0] = u.i32s[0];
    if (n > 1) x[1] = u.i32s[1];
    if (n > 2) x[2] = u.i32s[2];
    if (n > 3) x[3] = u.i32s[3];
    if (n > 4) x[4] = u.i32s[4];
    if (n > 5) x[5] = u.i32s[5];
    if (n > 6) x[6] = u.i32s[6];
    if (n > 7) x[7] = u.i32s[7];
}

SIMSIMD_PUBLIC void simsimd_dot_f16_haswell(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
simsimd_dot_f16_haswell_cycle:
    if (count_scalars < 8) {
        a_f32x8 = _simsimd_partial_load_f16x8_haswell(a_scalars, count_scalars);
        b_f32x8 = _simsimd_partial_load_f16x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_scalars));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_scalars));
        count_scalars -= 8, a_scalars += 8, b_scalars += 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto simsimd_dot_f16_haswell_cycle;
    *result = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_f32x8);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_haswell(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
            _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));
    simsimd_f32c_t tail_result;
    simsimd_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_real_f32x8);
    result->imag = tail_result.imag + (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_imag_f32x8);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_haswell(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        b_f32x8 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));
    simsimd_f32c_t tail_result;
    simsimd_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_real_f32x8);
    result->imag = tail_result.imag + (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_imag_f32x8);
}

SIMSIMD_PUBLIC void simsimd_dot_i8_haswell(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_i32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        // Upcast `int8` to `int16`
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 0));
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 0));
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    simsimd_i32_t sum = _simsimd_reduce_add_i32x8_haswell(_mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum += (simsimd_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_dot_u8_haswell(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_u32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        // Upcast `uint8` to `int16`. Unpacking is faster than extracts.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    simsimd_u32_t sum = (simsimd_u32_t)_simsimd_reduce_add_i32x8_haswell(
        _mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum += (simsimd_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

SIMSIMD_INTERNAL __m256 _simsimd_bf16x8_to_f32x8_haswell(__m128i x) {
    // Upcasting from `bf16` to `f32` is done by shifting the `bf16` values by 16 bits to the left, like:
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x), 16));
}

SIMSIMD_INTERNAL __m128i _simsimd_f32x8_to_bf16x8_haswell(__m256 x) {
    // Pack the 32-bit integers into 16-bit integers.
    // This is less trivial than unpacking: https://stackoverflow.com/a/77781241/2766161
    // The best approach is to shuffle within lanes first: https://stackoverflow.com/a/49723746/2766161
    // Our shuffling mask will drop the low 2-bytes from every 4-byte word.
    __m256i trunc_elements = _mm256_shuffle_epi8(                       //
        _mm256_castps_si256(x),                                         //
        _mm256_set_epi8(                                                //
            -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 11, 10, 7, 6, 3, 2, //
            -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 11, 10, 7, 6, 3, 2  //
            ));
    __m256i ordered = _mm256_permute4x64_epi64(trunc_elements, 0x58);
    __m128i result = _mm256_castsi256_si128(ordered);
    return result;
}

SIMSIMD_INTERNAL __m128i _simsimd_partial_load_bf16x8_haswell(simsimd_bf16_t const *a, simsimd_size_t n) {
    // In case the software emulation for `bf16` scalars is enabled, the `simsimd_bf16_to_f32`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    simsimd_b512_vec_t result;
    result.xmms[0] = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.bf16s[i] = a[i];
    return result.xmms[0];
}

SIMSIMD_PUBLIC void simsimd_dot_bf16_haswell(simsimd_bf16_t const *a_scalars, simsimd_bf16_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m128i a_bf16x8, b_bf16x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
simsimd_dot_bf16_haswell_cycle:
    if (count_scalars < 8) {
        a_bf16x8 = _simsimd_partial_load_bf16x8_haswell(a_scalars, count_scalars);
        b_bf16x8 = _simsimd_partial_load_bf16x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x8 = _mm_lddqu_si128((__m128i const *)a_scalars);
        b_bf16x8 = _mm_lddqu_si128((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(a_bf16x8), _simsimd_bf16x8_to_f32x8_haswell(b_bf16x8),
                                sum_f32x8);
    if (count_scalars) goto simsimd_dot_bf16_haswell_cycle;
    *result = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_f32x8);
}

/*  Convert 8x E4M3 values to 8x F32 values using AVX2 bit manipulation.
 *
 *  E4M3 format: S EEEE MMM (bias=7, range: 2^-6 to 448)
 *  F32 format:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (bias=127)
 *  Conversion:  sign<<31, (exp+120)<<23, mant<<20
 */
SIMSIMD_INTERNAL __m256 _simsimd_e4m3x8_to_f32x8_haswell(__m128i fp8) {
    // Only use lower 64 bits (8 bytes)
    __m256i v = _mm256_cvtepu8_epi32(fp8);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(v, 7), _mm256_set1_epi32(1)), 31);
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(v, 3), _mm256_set1_epi32(0x0F));
    __m256i mant = _mm256_and_si256(v, _mm256_set1_epi32(0x07));
    // Build F32: (exp + 120) << 23, mant << 20
    __m256i f32_exp = _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23);
    __m256i f32_mant = _mm256_slli_epi32(mant, 20);
    __m256i f32_bits = _mm256_or_si256(sign, _mm256_or_si256(f32_exp, f32_mant));
    // Handle exp=0: zero out the entire value (flush denormals to zero)
    // AVX2 doesn't have masked move, so use blend with comparison
    __m256i zero_mask = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    f32_bits = _mm256_andnot_si256(zero_mask, f32_bits);
    return _mm256_castsi256_ps(f32_bits);
}

/*  Convert 8x E5M2 values to 8x F32 values using AVX2 bit manipulation.
 *
 *  E5M2 format: S EEEEE MM (bias=15, range: 2^-14 to 57344)
 *  F32 format:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (bias=127)
 *  Conversion:  sign<<31, (exp+112)<<23, mant<<21
 */
SIMSIMD_INTERNAL __m256 _simsimd_e5m2x8_to_f32x8_haswell(__m128i fp8) {
    // Only use lower 64 bits (8 bytes)
    __m256i v = _mm256_cvtepu8_epi32(fp8);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(v, 7), _mm256_set1_epi32(1)), 31);
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(v, 2), _mm256_set1_epi32(0x1F));
    __m256i mant = _mm256_and_si256(v, _mm256_set1_epi32(0x03));
    // Build F32: (exp + 112) << 23, mant << 21
    __m256i f32_exp = _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(112)), 23);
    __m256i f32_mant = _mm256_slli_epi32(mant, 21);
    __m256i f32_bits = _mm256_or_si256(sign, _mm256_or_si256(f32_exp, f32_mant));
    // Handle exp=0: zero out the entire value (flush denormals to zero)
    __m256i zero_mask = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    f32_bits = _mm256_andnot_si256(zero_mask, f32_bits);
    return _mm256_castsi256_ps(f32_bits);
}

SIMSIMD_INTERNAL __m128i _simsimd_partial_load_e4m3x8_haswell(simsimd_e4m3_t const *a, simsimd_size_t n) {
    simsimd_b512_vec_t result;
    result.xmms[0] = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.e4m3s[i] = a[i];
    return result.xmms[0];
}

SIMSIMD_INTERNAL __m128i _simsimd_partial_load_e5m2x8_haswell(simsimd_e5m2_t const *a, simsimd_size_t n) {
    simsimd_b512_vec_t result;
    result.xmms[0] = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.e5m2s[i] = a[i];
    return result.xmms[0];
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_haswell(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m128i a_e4m3x8, b_e4m3x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
simsimd_dot_e4m3_haswell_cycle:
    if (count_scalars < 8) {
        a_e4m3x8 = _simsimd_partial_load_e4m3x8_haswell(a_scalars, count_scalars);
        b_e4m3x8 = _simsimd_partial_load_e4m3x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x8 = _mm_loadl_epi64((__m128i const *)a_scalars);
        b_e4m3x8 = _mm_loadl_epi64((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(a_e4m3x8), _simsimd_e4m3x8_to_f32x8_haswell(b_e4m3x8),
                                sum_f32x8);
    if (count_scalars) goto simsimd_dot_e4m3_haswell_cycle;
    *result = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_f32x8);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_haswell(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m128i a_e5m2x8, b_e5m2x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
simsimd_dot_e5m2_haswell_cycle:
    if (count_scalars < 8) {
        a_e5m2x8 = _simsimd_partial_load_e5m2x8_haswell(a_scalars, count_scalars);
        b_e5m2x8 = _simsimd_partial_load_e5m2x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x8 = _mm_loadl_epi64((__m128i const *)a_scalars);
        b_e5m2x8 = _mm_loadl_epi64((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(a_e5m2x8), _simsimd_e5m2x8_to_f32x8_haswell(b_e5m2x8),
                                sum_f32x8);
    if (count_scalars) goto simsimd_dot_e5m2_haswell_cycle;
    *result = (simsimd_f32_t)_simsimd_reduce_add_f32x8_haswell(sum_f32x8);
}

/**
 *  @brief Running state for 256-bit dot accumulation over f32 scalars on Haswell.
 */
typedef struct simsimd_dot_f32x8_state_haswell_t {
    __m256 sum_f32x8;
} simsimd_dot_f32x8_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x8_init_haswell(simsimd_dot_f32x8_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_f32x8_update_haswell(simsimd_dot_f32x8_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(a.ymm_ps, b.ymm_ps, sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x8_finalize_haswell(                                               //
    simsimd_dot_f32x8_state_haswell_t const *state_a, simsimd_dot_f32x8_state_haswell_t const *state_b, //
    simsimd_dot_f32x8_state_haswell_t const *state_c, simsimd_dot_f32x8_state_haswell_t const *state_d, //
    simsimd_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction for f32 in AVX2
    __m128 sum_f32x4_a = _mm_add_ps(_mm256_castps256_ps128(state_a->sum_f32x8),
                                    _mm256_extractf128_ps(state_a->sum_f32x8, 1));
    __m128 sum_f32x4_b = _mm_add_ps(_mm256_castps256_ps128(state_b->sum_f32x8),
                                    _mm256_extractf128_ps(state_b->sum_f32x8, 1));
    __m128 sum_f32x4_c = _mm_add_ps(_mm256_castps256_ps128(state_c->sum_f32x8),
                                    _mm256_extractf128_ps(state_c->sum_f32x8, 1));
    __m128 sum_f32x4_d = _mm_add_ps(_mm256_castps256_ps128(state_d->sum_f32x8),
                                    _mm256_extractf128_ps(state_d->sum_f32x8, 1));
    __m128 transpose_ab_low_f32x4 = _mm_unpacklo_ps(sum_f32x4_a, sum_f32x4_b);
    __m128 transpose_cd_low_f32x4 = _mm_unpacklo_ps(sum_f32x4_c, sum_f32x4_d);
    __m128 transpose_ab_high_f32x4 = _mm_unpackhi_ps(sum_f32x4_a, sum_f32x4_b);
    __m128 transpose_cd_high_f32x4 = _mm_unpackhi_ps(sum_f32x4_c, sum_f32x4_d);
    __m128 sum_lane0_f32x4 = _mm_movelh_ps(transpose_ab_low_f32x4, transpose_cd_low_f32x4);
    __m128 sum_lane1_f32x4 = _mm_movehl_ps(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m128 sum_lane2_f32x4 = _mm_movelh_ps(transpose_ab_high_f32x4, transpose_cd_high_f32x4);
    __m128 sum_lane3_f32x4 = _mm_movehl_ps(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    __m128 final_sum_f32x4 = _mm_add_ps(_mm_add_ps(sum_lane0_f32x4, sum_lane1_f32x4),
                                        _mm_add_ps(sum_lane2_f32x4, sum_lane3_f32x4));
    _mm_storeu_ps(results, final_sum_f32x4);
}

/**
 *  @brief Running state for 256-bit dot accumulation over f16 scalars on Haswell.
 */
typedef struct simsimd_dot_f16x16_state_haswell_t {
    __m256 sum_f32x8;
} simsimd_dot_f16x16_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x16_init_haswell(simsimd_dot_f16x16_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_f16x16_update_haswell(simsimd_dot_f16x16_state_haswell_t *state, simsimd_b256_vec_t a,
                                                        simsimd_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 0))),
                                _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 0))), sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 8))),
                                _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 8))), sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x16_finalize_haswell(                                                //
    simsimd_dot_f16x16_state_haswell_t const *state_a, simsimd_dot_f16x16_state_haswell_t const *state_b, //
    simsimd_dot_f16x16_state_haswell_t const *state_c, simsimd_dot_f16x16_state_haswell_t const *state_d, //
    simsimd_f32_t *results) {
    simsimd_dot_f32x8_finalize_haswell(                                                                         //
        (simsimd_dot_f32x8_state_haswell_t const *)state_a, (simsimd_dot_f32x8_state_haswell_t const *)state_b, //
        (simsimd_dot_f32x8_state_haswell_t const *)state_c, (simsimd_dot_f32x8_state_haswell_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 256-bit dot accumulation over bf16 scalars on Haswell.
 */
typedef struct simsimd_dot_bf16x16_state_haswell_t {
    __m256 sum_f32x8;
} simsimd_dot_bf16x16_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x16_init_haswell(simsimd_dot_bf16x16_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x16_update_haswell(simsimd_dot_bf16x16_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 0))),
                                _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 8))),
                                _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 8))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x16_finalize_haswell(                                                 //
    simsimd_dot_bf16x16_state_haswell_t const *state_a, simsimd_dot_bf16x16_state_haswell_t const *state_b, //
    simsimd_dot_bf16x16_state_haswell_t const *state_c, simsimd_dot_bf16x16_state_haswell_t const *state_d, //
    simsimd_f32_t *results) {
    simsimd_dot_f32x8_finalize_haswell(                                                                         //
        (simsimd_dot_f32x8_state_haswell_t const *)state_a, (simsimd_dot_f32x8_state_haswell_t const *)state_b, //
        (simsimd_dot_f32x8_state_haswell_t const *)state_c, (simsimd_dot_f32x8_state_haswell_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 256-bit dot accumulation over e4m3 scalars on Haswell.
 */
typedef struct simsimd_dot_e4m3x32_state_haswell_t {
    __m256 sum_f32x8;
} simsimd_dot_e4m3x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_init_haswell(simsimd_dot_e4m3x32_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_update_haswell(simsimd_dot_e4m3x32_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 0))),
                                _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 8))),
                                _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 8))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 16))),
                                _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 16))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 24))),
                                _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 24))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x32_finalize_haswell(                                                 //
    simsimd_dot_e4m3x32_state_haswell_t const *state_a, simsimd_dot_e4m3x32_state_haswell_t const *state_b, //
    simsimd_dot_e4m3x32_state_haswell_t const *state_c, simsimd_dot_e4m3x32_state_haswell_t const *state_d, //
    simsimd_f32_t *results) {
    simsimd_dot_f32x8_finalize_haswell(                                                                         //
        (simsimd_dot_f32x8_state_haswell_t const *)state_a, (simsimd_dot_f32x8_state_haswell_t const *)state_b, //
        (simsimd_dot_f32x8_state_haswell_t const *)state_c, (simsimd_dot_f32x8_state_haswell_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 256-bit dot accumulation over e5m2 scalars on Haswell.
 */
typedef struct simsimd_dot_e5m2x32_state_haswell_t {
    __m256 sum_f32x8;
} simsimd_dot_e5m2x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_init_haswell(simsimd_dot_e5m2x32_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_update_haswell(simsimd_dot_e5m2x32_state_haswell_t *state,
                                                         simsimd_b256_vec_t a, simsimd_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 0))),
                                _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 8))),
                                _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 8))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 16))),
                                _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 16))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 24))),
                                _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 24))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x32_finalize_haswell(                                                 //
    simsimd_dot_e5m2x32_state_haswell_t const *state_a, simsimd_dot_e5m2x32_state_haswell_t const *state_b, //
    simsimd_dot_e5m2x32_state_haswell_t const *state_c, simsimd_dot_e5m2x32_state_haswell_t const *state_d, //
    simsimd_f32_t *results) {
    simsimd_dot_f32x8_finalize_haswell(                                                                         //
        (simsimd_dot_f32x8_state_haswell_t const *)state_a, (simsimd_dot_f32x8_state_haswell_t const *)state_b, //
        (simsimd_dot_f32x8_state_haswell_t const *)state_c, (simsimd_dot_f32x8_state_haswell_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 256-bit dot accumulation over i8 scalars on Haswell.
 */
typedef struct simsimd_dot_i8x32_state_haswell_t {
    __m256i sum_i32x8_low;
    __m256i sum_i32x8_high;
} simsimd_dot_i8x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x32_init_haswell(simsimd_dot_i8x32_state_haswell_t *state) {
    state->sum_i32x8_low = _mm256_setzero_si256();
    state->sum_i32x8_high = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x32_update_haswell(simsimd_dot_i8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b) {
    __m256i sum_i32x8_low = state->sum_i32x8_low;
    __m256i sum_i32x8_high = state->sum_i32x8_high;

    __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a.i8s + 0));
    __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b.i8s + 0));
    __m256i a_i16x16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 0));
    __m256i a_i16x16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
    __m256i b_i16x16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 0));
    __m256i b_i16x16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));
    sum_i32x8_low = _mm256_add_epi32(sum_i32x8_low, _mm256_madd_epi16(a_i16x16_low, b_i16x16_low));
    sum_i32x8_high = _mm256_add_epi32(sum_i32x8_high, _mm256_madd_epi16(a_i16x16_high, b_i16x16_high));

    state->sum_i32x8_low = sum_i32x8_low;
    state->sum_i32x8_high = sum_i32x8_high;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x32_finalize_haswell(                                               //
    simsimd_dot_i8x32_state_haswell_t const *state_a, simsimd_dot_i8x32_state_haswell_t const *state_b, //
    simsimd_dot_i8x32_state_haswell_t const *state_c, simsimd_dot_i8x32_state_haswell_t const *state_d, //
    simsimd_i32_t *results) {
    // First, combine the low and high accumulators for each state
    __m256i sum_i32x8_a = _mm256_add_epi32(state_a->sum_i32x8_low, state_a->sum_i32x8_high);
    __m256i sum_i32x8_b = _mm256_add_epi32(state_b->sum_i32x8_low, state_b->sum_i32x8_high);
    __m256i sum_i32x8_c = _mm256_add_epi32(state_c->sum_i32x8_low, state_c->sum_i32x8_high);
    __m256i sum_i32x8_d = _mm256_add_epi32(state_d->sum_i32x8_low, state_d->sum_i32x8_high);
    // ILP-optimized 4-way horizontal reduction for i32 in AVX2
    // Step 1: 8->4 for all 4 states
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_a), _mm256_extracti128_si256(sum_i32x8_a, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_b), _mm256_extracti128_si256(sum_i32x8_b, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_c), _mm256_extracti128_si256(sum_i32x8_c, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_d), _mm256_extracti128_si256(sum_i32x8_d, 1));
    // Step 2: Transpose 4x4 matrix
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 3: Vertical sum and store as i32
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

/**
 *  @brief Running state for 256-bit dot accumulation over u8 scalars on Haswell.
 */
typedef struct simsimd_dot_u8x32_state_haswell_t {
    __m256i sum_u32x8_low;
    __m256i sum_u32x8_high;
} simsimd_dot_u8x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x32_init_haswell(simsimd_dot_u8x32_state_haswell_t *state) {
    state->sum_u32x8_low = _mm256_setzero_si256();
    state->sum_u32x8_high = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_u8x32_update_haswell(simsimd_dot_u8x32_state_haswell_t *state, simsimd_b256_vec_t a,
                                                       simsimd_b256_vec_t b) {
    __m256i sum_u32x8_low = state->sum_u32x8_low;
    __m256i sum_u32x8_high = state->sum_u32x8_high;
    __m256i const zeros_u8x32 = _mm256_setzero_si256();

    __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a.u8s + 0));
    __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b.u8s + 0));
    __m256i a_u16x16_low = _mm256_unpacklo_epi8(a_u8x32, zeros_u8x32);
    __m256i a_u16x16_high = _mm256_unpackhi_epi8(a_u8x32, zeros_u8x32);
    __m256i b_u16x16_low = _mm256_unpacklo_epi8(b_u8x32, zeros_u8x32);
    __m256i b_u16x16_high = _mm256_unpackhi_epi8(b_u8x32, zeros_u8x32);
    sum_u32x8_low = _mm256_add_epi32(sum_u32x8_low, _mm256_madd_epi16(a_u16x16_low, b_u16x16_low));
    sum_u32x8_high = _mm256_add_epi32(sum_u32x8_high, _mm256_madd_epi16(a_u16x16_high, b_u16x16_high));

    state->sum_u32x8_low = sum_u32x8_low;
    state->sum_u32x8_high = sum_u32x8_high;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x32_finalize_haswell(                                               //
    simsimd_dot_u8x32_state_haswell_t const *state_a, simsimd_dot_u8x32_state_haswell_t const *state_b, //
    simsimd_dot_u8x32_state_haswell_t const *state_c, simsimd_dot_u8x32_state_haswell_t const *state_d, //
    simsimd_u32_t *results) {
    // State is layout-compatible with i8x32 (both contain sum_*_low and sum_*_high)
    // Result storage is also compatible (same bit pattern, different signedness interpretation)
    simsimd_dot_i8x32_finalize_haswell(                                                                         //
        (simsimd_dot_i8x32_state_haswell_t const *)state_a, (simsimd_dot_i8x32_state_haswell_t const *)state_b, //
        (simsimd_dot_i8x32_state_haswell_t const *)state_c, (simsimd_dot_i8x32_state_haswell_t const *)state_d,
        (simsimd_i32_t *)results);
}

/** @brief Type-agnostic 256-bit full load (Haswell AVX2). */
SIMSIMD_INTERNAL void _simsimd_load_b256_haswell(void const *src, simsimd_b256_vec_t *dst) {
    dst->ymm = _mm256_loadu_si256((const __m256i *)src);
}

/** @brief Type-agnostic partial load for 32-bit elements (8 elements max) into 256-bit vector (Haswell AVX2). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b32x8_haswell(void const *src, simsimd_size_t n, simsimd_b256_vec_t *dst) {
    simsimd_u32_t const *s = (simsimd_u32_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (simsimd_size_t i = 0; i < n && i < 8; ++i) dst->u32s[i] = s[i];
}

/** @brief Type-agnostic partial load for 16-bit elements (16 elements max) into 256-bit vector (Haswell AVX2). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b16x16_haswell(void const *src, simsimd_size_t n, simsimd_b256_vec_t *dst) {
    simsimd_u16_t const *s = (simsimd_u16_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (simsimd_size_t i = 0; i < n && i < 16; ++i) dst->u16s[i] = s[i];
}

/** @brief Type-agnostic partial load for 8-bit elements (32 elements max) into 256-bit vector (Haswell AVX2). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b8x32_haswell(void const *src, simsimd_size_t n, simsimd_b256_vec_t *dst) {
    simsimd_u8_t const *s = (simsimd_u8_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (simsimd_size_t i = 0; i < n && i < 32; ++i) dst->u8s[i] = s[i];
}

/** @brief Type-agnostic partial store for 32-bit elements (8 elements max) from 256-bit vector (Haswell AVX2). */
SIMSIMD_INTERNAL void _simsimd_partial_store_b32x8_haswell(simsimd_b256_vec_t const *src, void *dst, simsimd_size_t n) {
    simsimd_u32_t *d = (simsimd_u32_t *)dst;
    for (simsimd_size_t i = 0; i < n && i < 8; ++i) d[i] = src->u32s[i];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

SIMSIMD_INTERNAL __m512 _simsimd_bf16x16_to_f32x16_skylake(__m256i a) {
    // Upcasting from `bf16` to `f32` is done by shifting the `bf16` values by 16 bits to the left, like:
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
}

SIMSIMD_INTERNAL __m256i _simsimd_f32x16_to_bf16x16_skylake(__m512 a) {
    // Add 2^15 and right shift 16 to do round-nearest
    __m512i x = _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(a), _mm512_set1_epi32(1 << 15)), 16);
    return _mm512_cvtepi32_epi16(x);
}

SIMSIMD_PUBLIC void simsimd_dot_f32_skylake(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_f32x16 = _mm512_setzero();

simsimd_dot_f32_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a_scalars);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a_scalars);
        b_f32x16 = _mm512_loadu_ps(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto simsimd_dot_f32_skylake_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_f64_skylake(simsimd_f64_t const *a_scalars, simsimd_f64_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_f64_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();

simsimd_dot_f64_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_scalars);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_scalars);
        b_f64x8 = _mm512_loadu_pd(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_f64x8);
    if (count_scalars) goto simsimd_dot_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_skylake(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_real_f32x16 = _mm512_setzero();
    __m512 sum_imag_f32x16 = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f32x16 = _mm512_set1_epi64(0x8000000000000000);
simsimd_dot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a_pairs);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a_pairs);
        b_f32x16 = _mm512_loadu_ps(b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    sum_real_f32x16 = _mm512_fmadd_ps(b_f32x16, a_f32x16, sum_real_f32x16);
    b_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1); //? Swap adjacent entries within each pair
    sum_imag_f32x16 = _mm512_fmadd_ps(b_f32x16, a_f32x16, sum_imag_f32x16);
    if (count_pairs) goto simsimd_dot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x16 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(sum_real_f32x16), sign_flip_f32x16));

    // Reduce horizontal sums:
    result->real = _simsimd_reduce_add_f32x16_skylake(sum_real_f32x16);
    result->imag = _simsimd_reduce_add_f32x16_skylake(sum_imag_f32x16);
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_skylake(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_real_f32x16 = _mm512_setzero();
    __m512 sum_imag_f32x16 = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f32x16 = _mm512_set1_epi64(0x8000000000000000);
simsimd_vdot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, (simsimd_f32_t const *)a_pairs);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, (simsimd_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps((simsimd_f32_t const *)a_pairs);
        b_f32x16 = _mm512_loadu_ps((simsimd_f32_t const *)b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    sum_real_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_real_f32x16);
    b_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1); //? Swap adjacent entries within each pair
    sum_imag_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_imag_f32x16);
    if (count_pairs) goto simsimd_vdot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x16 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(sum_imag_f32x16), sign_flip_f32x16));

    // Reduce horizontal sums:
    result->real = _simsimd_reduce_add_f32x16_skylake(sum_real_f32x16);
    result->imag = _simsimd_reduce_add_f32x16_skylake(sum_imag_f32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_f64c_skylake(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_f64c_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
simsimd_dot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_pairs);
        b_f64x8 = _mm512_loadu_pd(b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    sum_real_f64x8 = _mm512_fmadd_pd(b_f64x8, a_f64x8, sum_real_f64x8);
    b_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.
    sum_imag_f64x8 = _mm512_fmadd_pd(b_f64x8, a_f64x8, sum_imag_f64x8);
    if (count_pairs) goto simsimd_dot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

SIMSIMD_PUBLIC void simsimd_vdot_f64c_skylake(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_f64c_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
simsimd_vdot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, (simsimd_f64_t const *)a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, (simsimd_f64_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd((simsimd_f64_t const *)a_pairs);
        b_f64x8 = _mm512_loadu_pd((simsimd_f64_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    b_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_imag_f64x8);
    if (count_pairs) goto simsimd_vdot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

/*  Convert 16x E4M3 values to 16x F32 values using bit manipulation.
 *  This works on Skylake-X and later (AVX-512F only, no BF16/FP16 required).
 *
 *  E4M3 format: S EEEE MMM (bias=7, range: 2^-6 to 448)
 *  F32 format:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (bias=127)
 *  Conversion:  sign<<31, (exp+120)<<23, mant<<20
 */
SIMSIMD_INTERNAL __m512 _simsimd_e4m3x16_to_f32x16_skylake(__m128i fp8) {
    __m512i v = _mm512_cvtepu8_epi32(fp8);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(v, 7), _mm512_set1_epi32(1)), 31);
    __m512i exp = _mm512_and_si512(_mm512_srli_epi32(v, 3), _mm512_set1_epi32(0x0F));
    __m512i mant = _mm512_and_si512(v, _mm512_set1_epi32(0x07));
    // Build F32: (exp + 120) << 23, mant << 20
    __m512i f32_exp = _mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(120)), 23);
    __m512i f32_mant = _mm512_slli_epi32(mant, 20);
    __m512i f32_bits = _mm512_or_si512(sign, _mm512_or_si512(f32_exp, f32_mant));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero - single instruction!
    __mmask16 has_exp = _mm512_test_epi32_mask(v, _mm512_set1_epi32(0x78));
    f32_bits = _mm512_maskz_mov_epi32(has_exp, f32_bits);
    return _mm512_castsi512_ps(f32_bits);
}

/*  Convert 16x E5M2 values to 16x F32 values using bit manipulation.
 *  This works on Skylake-X and later (AVX-512F only, no BF16/FP16 required).
 *
 *  E5M2 format: S EEEEE MM (bias=15, range: 2^-14 to 57344)
 *  F32 format:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (bias=127)
 *  Conversion:  sign<<31, (exp+112)<<23, mant<<21
 */
SIMSIMD_INTERNAL __m512 _simsimd_e5m2x16_to_f32x16_skylake(__m128i fp8) {
    __m512i v = _mm512_cvtepu8_epi32(fp8);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(v, 7), _mm512_set1_epi32(1)), 31);
    __m512i exp = _mm512_and_si512(_mm512_srli_epi32(v, 2), _mm512_set1_epi32(0x1F));
    __m512i mant = _mm512_and_si512(v, _mm512_set1_epi32(0x03));
    // Build F32: (exp + 112) << 23, mant << 21
    __m512i f32_exp = _mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(112)), 23);
    __m512i f32_mant = _mm512_slli_epi32(mant, 21);
    __m512i f32_bits = _mm512_or_si512(sign, _mm512_or_si512(f32_exp, f32_mant));
    // DAZ: use TEST to check if exp bits (bits 6-2) are nonzero - single instruction!
    __mmask16 has_exp = _mm512_test_epi32_mask(v, _mm512_set1_epi32(0x7C));
    f32_bits = _mm512_maskz_mov_epi32(has_exp, f32_bits);
    return _mm512_castsi512_ps(f32_bits);
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_skylake(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m128i a_e4m3x16, b_e4m3x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

simsimd_dot_e4m3_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = _simsimd_e4m3x16_to_f32x16_skylake(a_e4m3x16);
    __m512 b_f32x16 = _simsimd_e4m3x16_to_f32x16_skylake(b_e4m3x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto simsimd_dot_e4m3_skylake_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_skylake(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m128i a_e5m2x16, b_e5m2x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

simsimd_dot_e5m2_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = _simsimd_e5m2x16_to_f32x16_skylake(a_e5m2x16);
    __m512 b_f32x16 = _simsimd_e5m2x16_to_f32x16_skylake(b_e5m2x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto simsimd_dot_e5m2_skylake_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

/**
 *  @brief Running state for 512-bit dot accumulation over f64 scalars on Skylake.
 */
typedef struct simsimd_dot_f64x8_state_skylake_t {
    __m512d sum_f64x8;
} simsimd_dot_f64x8_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_skylake(simsimd_dot_f64x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_skylake(simsimd_dot_f64x8_state_skylake_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    state->sum_f64x8 = _mm512_fmadd_pd(a.zmm_pd, b.zmm_pd, state->sum_f64x8);
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_skylake(                                               //
    simsimd_dot_f64x8_state_skylake_t const *state_a, simsimd_dot_f64x8_state_skylake_t const *state_b, //
    simsimd_dot_f64x8_state_skylake_t const *state_c, simsimd_dot_f64x8_state_skylake_t const *state_d, //
    simsimd_f64_t *results) {
    // ILP-optimized 4-way horizontal reduction for f64
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256d reduced_a = _mm256_add_pd(_mm512_castpd512_pd256(state_a->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_a->sum_f64x8, 1));
    __m256d reduced_b = _mm256_add_pd(_mm512_castpd512_pd256(state_b->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_b->sum_f64x8, 1));
    __m256d reduced_c = _mm256_add_pd(_mm512_castpd512_pd256(state_c->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_c->sum_f64x8, 1));
    __m256d reduced_d = _mm256_add_pd(_mm512_castpd512_pd256(state_d->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_d->sum_f64x8, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128d partial_a = _mm_add_pd(_mm256_castpd256_pd128(reduced_a), _mm256_extractf128_pd(reduced_a, 1));
    __m128d partial_b = _mm_add_pd(_mm256_castpd256_pd128(reduced_b), _mm256_extractf128_pd(reduced_b, 1));
    __m128d partial_c = _mm_add_pd(_mm256_castpd256_pd128(reduced_c), _mm256_extractf128_pd(reduced_c, 1));
    __m128d partial_d = _mm_add_pd(_mm256_castpd256_pd128(reduced_d), _mm256_extractf128_pd(reduced_d, 1));
    // Step 3: 2->1 for each state and combine into 4-element result
    // Each __m128d has [low, high], need to add them to get final scalar
    __m128d sum_ab = _mm_add_pd(_mm_unpacklo_pd(partial_a, partial_b), _mm_unpackhi_pd(partial_a, partial_b));
    __m128d sum_cd = _mm_add_pd(_mm_unpacklo_pd(partial_c, partial_d), _mm_unpackhi_pd(partial_c, partial_d));
    // Store as f64
    _mm_storeu_pd(results, sum_ab);
    _mm_storeu_pd(results + 2, sum_cd);
}

/**
 *  @brief Running state for 512-bit dot accumulation over f32 scalars on Skylake.
 */
typedef struct simsimd_dot_f32x16_state_skylake_t {
    __m512 sum_f32x16;
} simsimd_dot_f32x16_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_skylake(simsimd_dot_f32x16_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_skylake(simsimd_dot_f32x16_state_skylake_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    // Use pre-loaded zmm_ps directly (avoids redundant loads when used with GEMM macro)
    state->sum_f32x16 = _mm512_fmadd_ps(a.zmm_ps, b.zmm_ps, state->sum_f32x16);
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_skylake(                                                //
    simsimd_dot_f32x16_state_skylake_t const *state_a, simsimd_dot_f32x16_state_skylake_t const *state_b, //
    simsimd_dot_f32x16_state_skylake_t const *state_c, simsimd_dot_f32x16_state_skylake_t const *state_d, //
    simsimd_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction
    // Step 1: 16->8 for all 4 states (extract high 256-bit half and add to low half)
    __m256 reduced_a = _mm256_add_ps(_mm512_castps512_ps256(state_a->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_a->sum_f32x16, 1));
    __m256 reduced_b = _mm256_add_ps(_mm512_castps512_ps256(state_b->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_b->sum_f32x16, 1));
    __m256 reduced_c = _mm256_add_ps(_mm512_castps512_ps256(state_c->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_c->sum_f32x16, 1));
    __m256 reduced_d = _mm256_add_ps(_mm512_castps512_ps256(state_d->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_d->sum_f32x16, 1));
    // Step 2: 8->4 for all 4 states (extract high 128-bit half and add to low half)
    __m128 partial_a = _mm_add_ps(_mm256_castps256_ps128(reduced_a), _mm256_extractf128_ps(reduced_a, 1));
    __m128 partial_b = _mm_add_ps(_mm256_castps256_ps128(reduced_b), _mm256_extractf128_ps(reduced_b, 1));
    __m128 partial_c = _mm_add_ps(_mm256_castps256_ps128(reduced_c), _mm256_extractf128_ps(reduced_c, 1));
    __m128 partial_d = _mm_add_ps(_mm256_castps256_ps128(reduced_d), _mm256_extractf128_ps(reduced_d, 1));
    // Step 3: Transpose 4x4 matrix of partial sums - now each row has one element from each state
    __m128 transpose_ab_lo = _mm_unpacklo_ps(partial_a, partial_b);
    __m128 transpose_cd_lo = _mm_unpacklo_ps(partial_c, partial_d);
    __m128 transpose_ab_hi = _mm_unpackhi_ps(partial_a, partial_b);
    __m128 transpose_cd_hi = _mm_unpackhi_ps(partial_c, partial_d);
    __m128 sum_lane_0 = _mm_movelh_ps(transpose_ab_lo, transpose_cd_lo);
    __m128 sum_lane_1 = _mm_movehl_ps(transpose_cd_lo, transpose_ab_lo);
    __m128 sum_lane_2 = _mm_movelh_ps(transpose_ab_hi, transpose_cd_hi);
    __m128 sum_lane_3 = _mm_movehl_ps(transpose_cd_hi, transpose_ab_hi);
    // Step 4: Vertical sum - each lane becomes the final result for one state
    __m128 final_sum = _mm_add_ps(_mm_add_ps(sum_lane_0, sum_lane_1), _mm_add_ps(sum_lane_2, sum_lane_3));
    _mm_storeu_ps(results, final_sum);
}

/**
 *  @brief Running state for 512-bit dot accumulation over e4m3 scalars on Skylake.
 */
typedef struct simsimd_dot_e4m3x64_state_skylake_t {
    __m512 sum_f32x16;
} simsimd_dot_e4m3x64_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_skylake(simsimd_dot_e4m3x64_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_skylake(simsimd_dot_e4m3x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m128i a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 0));
    __m128i b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 0));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_e4m3x16),
                                 _simsimd_e4m3x16_to_f32x16_skylake(b_e4m3x16), sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 16));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 16));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_e4m3x16),
                                 _simsimd_e4m3x16_to_f32x16_skylake(b_e4m3x16), sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 32));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 32));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_e4m3x16),
                                 _simsimd_e4m3x16_to_f32x16_skylake(b_e4m3x16), sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 48));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 48));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_e4m3x16),
                                 _simsimd_e4m3x16_to_f32x16_skylake(b_e4m3x16), sum_f32x16);
    state->sum_f32x16 = sum_f32x16;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_skylake(                                                 //
    simsimd_dot_e4m3x64_state_skylake_t const *state_a, simsimd_dot_e4m3x64_state_skylake_t const *state_b, //
    simsimd_dot_e4m3x64_state_skylake_t const *state_c, simsimd_dot_e4m3x64_state_skylake_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    simsimd_dot_f32x16_finalize_skylake(                                                                          //
        (simsimd_dot_f32x16_state_skylake_t const *)state_a, (simsimd_dot_f32x16_state_skylake_t const *)state_b, //
        (simsimd_dot_f32x16_state_skylake_t const *)state_c, (simsimd_dot_f32x16_state_skylake_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 512-bit dot accumulation over e5m2 scalars on Skylake.
 */
typedef struct simsimd_dot_e5m2x64_state_skylake_t {
    __m512 sum_f32x16;
} simsimd_dot_e5m2x64_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_skylake(simsimd_dot_e5m2x64_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_skylake(simsimd_dot_e5m2x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m128i a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 0));
    __m128i b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 0));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_e5m2x16),
                                 _simsimd_e5m2x16_to_f32x16_skylake(b_e5m2x16), sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 16));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 16));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_e5m2x16),
                                 _simsimd_e5m2x16_to_f32x16_skylake(b_e5m2x16), sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 32));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 32));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_e5m2x16),
                                 _simsimd_e5m2x16_to_f32x16_skylake(b_e5m2x16), sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 48));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 48));
    sum_f32x16 = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_e5m2x16),
                                 _simsimd_e5m2x16_to_f32x16_skylake(b_e5m2x16), sum_f32x16);
    state->sum_f32x16 = sum_f32x16;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_skylake(                                                 //
    simsimd_dot_e5m2x64_state_skylake_t const *state_a, simsimd_dot_e5m2x64_state_skylake_t const *state_b, //
    simsimd_dot_e5m2x64_state_skylake_t const *state_c, simsimd_dot_e5m2x64_state_skylake_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    simsimd_dot_f32x16_finalize_skylake(                                                                          //
        (simsimd_dot_f32x16_state_skylake_t const *)state_a, (simsimd_dot_f32x16_state_skylake_t const *)state_b, //
        (simsimd_dot_f32x16_state_skylake_t const *)state_c, (simsimd_dot_f32x16_state_skylake_t const *)state_d,
        results);
}

/** @brief Type-agnostic 512-bit full load (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_load_b512_skylake(void const *src, simsimd_b512_vec_t *dst) {
    dst->zmm = _mm512_loadu_si512(src);
}

/** @brief Type-agnostic partial load for 64-bit elements (8 elements max) into 512-bit vector (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b64x8_skylake(void const *src, simsimd_size_t n, simsimd_b512_vec_t *dst) {
    simsimd_u64_t const *s = (simsimd_u64_t const *)src;
    dst->zmm = _mm512_setzero_si512();
    for (simsimd_size_t i = 0; i < n && i < 8; ++i) dst->u64s[i] = s[i];
}

/** @brief Type-agnostic partial load for 32-bit elements (16 elements max) into 512-bit vector (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b32x16_skylake(void const *src, simsimd_size_t n, simsimd_b512_vec_t *dst) {
    simsimd_u32_t const *s = (simsimd_u32_t const *)src;
    dst->zmm = _mm512_setzero_si512();
    for (simsimd_size_t i = 0; i < n && i < 16; ++i) dst->u32s[i] = s[i];
}

/** @brief Type-agnostic partial load for 16-bit elements (32 elements max) into 512-bit vector (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b16x32_skylake(void const *src, simsimd_size_t n, simsimd_b512_vec_t *dst) {
    simsimd_u16_t const *s = (simsimd_u16_t const *)src;
    dst->zmm = _mm512_setzero_si512();
    for (simsimd_size_t i = 0; i < n && i < 32; ++i) dst->u16s[i] = s[i];
}

/** @brief Type-agnostic partial load for 8-bit elements (64 elements max) into 512-bit vector (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_partial_load_b8x64_skylake(void const *src, simsimd_size_t n, simsimd_b512_vec_t *dst) {
    simsimd_u8_t const *s = (simsimd_u8_t const *)src;
    dst->zmm = _mm512_setzero_si512();
    for (simsimd_size_t i = 0; i < n && i < 64; ++i) dst->u8s[i] = s[i];
}

/** @brief Type-agnostic partial store for 32-bit elements (16 elements max) from 512-bit vector (Skylake AVX-512). */
SIMSIMD_INTERNAL void _simsimd_partial_store_b32x16_skylake(simsimd_b512_vec_t const *src, void *dst,
                                                            simsimd_size_t n) {
    simsimd_u32_t *d = (simsimd_u32_t *)dst;
    for (simsimd_size_t i = 0; i < n && i < 16; ++i) d[i] = src->u32s[i];
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_bf16_genoa(simsimd_bf16_t const *a_scalars, simsimd_bf16_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

simsimd_dot_bf16_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a_scalars);
        b_bf16x32 = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto simsimd_dot_bf16_genoa_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_bf16c_genoa(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                            simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_real_f32x16 = _mm512_setzero_ps();
    __m512 sum_imag_f32x16 = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_bf16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_bf16x32 = _mm512_set_epi8(              //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16((simsimd_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_loadu_epi16((simsimd_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    sum_real_f32x16 = _mm512_dpbf16_ps(sum_real_f32x16, (__m512bh)(_mm512_xor_si512(b_bf16x32, sign_flip_bf16x32)),
                                       (__m512bh)(a_bf16x32));
    sum_imag_f32x16 = _mm512_dpbf16_ps(
        sum_imag_f32x16, (__m512bh)(_mm512_shuffle_epi8(b_bf16x32, swap_adjacent_bf16x32)), (__m512bh)(a_bf16x32));
    if (count_pairs) goto simsimd_dot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    result->real = _simsimd_reduce_add_f32x16_skylake(sum_real_f32x16);
    result->imag = _simsimd_reduce_add_f32x16_skylake(sum_imag_f32x16);
}

SIMSIMD_PUBLIC void simsimd_vdot_bf16c_genoa(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_real_f32x16 = _mm512_setzero_ps();
    __m512 sum_imag_f32x16 = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_bf16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_bf16x32 = _mm512_set_epi8(              //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_vdot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16((simsimd_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_loadu_epi16((simsimd_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    sum_real_f32x16 = _mm512_dpbf16_ps(sum_real_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_bf16x32 = _mm512_xor_si512(a_bf16x32, sign_flip_bf16x32);
    b_bf16x32 = _mm512_shuffle_epi8(b_bf16x32, swap_adjacent_bf16x32);
    sum_imag_f32x16 = _mm512_dpbf16_ps(sum_imag_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_pairs) goto simsimd_vdot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    result->real = _simsimd_reduce_add_f32x16_skylake(sum_real_f32x16);
    result->imag = _simsimd_reduce_add_f32x16_skylake(sum_imag_f32x16);
}

/**
 *  @brief Convert 32x E4M3 values to 32x BF16 values.
 *
 *  Uses optimized path with fused exp+mant extraction.
 *  Denormals (exp=0, mant!=0) are flushed to zero (DAZ behavior).
 *
 *  E4M3 format: S EEEE MMM (bias=7, range: 2^-6 to 448)
 *  BF16 format: S EEEEEEEE MMMMMMM (bias=127)
 *  Conversion: sign<<8, (exp+120)<<7, mant<<4
 */
SIMSIMD_INTERNAL __m512i _simsimd_e4m3_to_bf16_genoa(__m256i fp8) {
    __m512i v = _mm512_cvtepu8_epi16(fp8);
    // Sign: shift bit 7 to bit 15
    __m512i sign = _mm512_and_si512(_mm512_slli_epi16(v, 8), _mm512_set1_epi16((short)0x8000));
    // Lower 7 bits contain exp (4) and mant (3): shift left 4 and add bias
    __m512i low7 = _mm512_and_si512(v, _mm512_set1_epi16(0x7F));
    __m512i exp_mant = _mm512_add_epi16(_mm512_slli_epi16(low7, 4), _mm512_set1_epi16(0x3C00)); // 120<<7 = 0x3C00
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero - single instruction!
    __mmask32 has_exp = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x78));
    __m512i masked_exp_mant = _mm512_maskz_mov_epi16(has_exp, exp_mant);
    return _mm512_or_si512(sign, masked_exp_mant);
}

/**
 *  @brief Convert 32x E5M2 values to 32x BF16 values.
 *
 *  Uses optimized path with fused exp+mant extraction.
 *  Denormals (exp=0, mant!=0) are flushed to zero (DAZ behavior).
 *
 *  E5M2 format: S EEEEE MM (bias=15, range: 2^-14 to 57344)
 *  BF16 format: S EEEEEEEE MMMMMMM (bias=127)
 *  Conversion: sign<<8, (exp+112)<<7, mant<<5
 */
SIMSIMD_INTERNAL __m512i _simsimd_e5m2_to_bf16_genoa(__m256i fp8) {
    __m512i v = _mm512_cvtepu8_epi16(fp8);
    __m512i sign = _mm512_and_si512(_mm512_slli_epi16(v, 8), _mm512_set1_epi16((short)0x8000));
    // Lower 7 bits: exp(5) + mant(2), shift left 5 and add bias
    __m512i low7 = _mm512_and_si512(v, _mm512_set1_epi16(0x7F));
    __m512i exp_mant = _mm512_add_epi16(_mm512_slli_epi16(low7, 5), _mm512_set1_epi16(0x3800)); // 112<<7 = 0x3800
    // DAZ: use TEST to check if exp bits (bits 6-2) are nonzero - single instruction!
    __mmask32 has_exp = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7C));
    __m512i masked_exp_mant = _mm512_maskz_mov_epi16(has_exp, exp_mant);
    return _mm512_or_si512(sign, masked_exp_mant);
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_genoa(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

simsimd_dot_e4m3_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a_scalars);
        b_e4m3x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E4M3 to BF16 and compute dot product
    __m512i a_bf16x32 = _simsimd_e4m3_to_bf16_genoa(a_e4m3x32);
    __m512i b_bf16x32 = _simsimd_e4m3_to_bf16_genoa(b_e4m3x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto simsimd_dot_e4m3_genoa_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_genoa(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256i a_e5m2x32, b_e5m2x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

simsimd_dot_e5m2_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a_scalars);
        b_e5m2x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E5M2 to BF16 and compute dot product
    __m512i a_bf16x32 = _simsimd_e5m2_to_bf16_genoa(a_e5m2x32);
    __m512i b_bf16x32 = _simsimd_e5m2_to_bf16_genoa(b_e5m2x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto simsimd_dot_e5m2_genoa_cycle;

    *result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
}

/**
 *  @brief Running state for 512-bit dot accumulation over bf16 scalars on Genoa.
 */
typedef struct simsimd_dot_bf16x32_state_genoa_t {
    __m512 sum_f32x16;
} simsimd_dot_bf16x32_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_genoa(simsimd_dot_bf16x32_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_genoa(simsimd_dot_bf16x32_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    // Use pre-loaded zmm directly (avoids redundant loads when used with GEMM macro)
    state->sum_f32x16 = _mm512_dpbf16_ps(state->sum_f32x16, (__m512bh)(a.zmm), (__m512bh)(b.zmm));
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_genoa(                                               //
    simsimd_dot_bf16x32_state_genoa_t const *state_a, simsimd_dot_bf16x32_state_genoa_t const *state_b, //
    simsimd_dot_bf16x32_state_genoa_t const *state_c, simsimd_dot_bf16x32_state_genoa_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    simsimd_dot_f32x16_finalize_skylake(                                                                          //
        (simsimd_dot_f32x16_state_skylake_t const *)state_a, (simsimd_dot_f32x16_state_skylake_t const *)state_b, //
        (simsimd_dot_f32x16_state_skylake_t const *)state_c, (simsimd_dot_f32x16_state_skylake_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 512-bit dot accumulation over e4m3 scalars on Genoa.
 */
typedef struct simsimd_dot_e4m3x64_state_genoa_t {
    __m512 sum_f32x16;
} simsimd_dot_e4m3x64_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_genoa(simsimd_dot_e4m3x64_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_genoa(simsimd_dot_e4m3x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m256i a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 0);
    __m256i b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 0);
    __m512i a_bf16x32 = _simsimd_e4m3_to_bf16_genoa(a_e4m3x32);
    __m512i b_bf16x32 = _simsimd_e4m3_to_bf16_genoa(b_e4m3x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 32);
    b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 32);
    a_bf16x32 = _simsimd_e4m3_to_bf16_genoa(a_e4m3x32);
    b_bf16x32 = _simsimd_e4m3_to_bf16_genoa(b_e4m3x32);
    state->sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_genoa(                                               //
    simsimd_dot_e4m3x64_state_genoa_t const *state_a, simsimd_dot_e4m3x64_state_genoa_t const *state_b, //
    simsimd_dot_e4m3x64_state_genoa_t const *state_c, simsimd_dot_e4m3x64_state_genoa_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    simsimd_dot_f32x16_finalize_skylake(                                                                          //
        (simsimd_dot_f32x16_state_skylake_t const *)state_a, (simsimd_dot_f32x16_state_skylake_t const *)state_b, //
        (simsimd_dot_f32x16_state_skylake_t const *)state_c, (simsimd_dot_f32x16_state_skylake_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 512-bit dot accumulation over e5m2 scalars on Genoa.
 */
typedef struct simsimd_dot_e5m2x64_state_genoa_t {
    __m512 sum_f32x16;
} simsimd_dot_e5m2x64_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_genoa(simsimd_dot_e5m2x64_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_genoa(simsimd_dot_e5m2x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m256i a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 0);
    __m256i b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 0);
    __m512i a_bf16x32 = _simsimd_e5m2_to_bf16_genoa(a_e5m2x32);
    __m512i b_bf16x32 = _simsimd_e5m2_to_bf16_genoa(b_e5m2x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 32);
    b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 32);
    a_bf16x32 = _simsimd_e5m2_to_bf16_genoa(a_e5m2x32);
    b_bf16x32 = _simsimd_e5m2_to_bf16_genoa(b_e5m2x32);
    state->sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_genoa(                                               //
    simsimd_dot_e5m2x64_state_genoa_t const *state_a, simsimd_dot_e5m2x64_state_genoa_t const *state_b, //
    simsimd_dot_e5m2x64_state_genoa_t const *state_c, simsimd_dot_e5m2x64_state_genoa_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    simsimd_dot_f32x16_finalize_skylake(                                                                          //
        (simsimd_dot_f32x16_state_skylake_t const *)state_a, (simsimd_dot_f32x16_state_skylake_t const *)state_b, //
        (simsimd_dot_f32x16_state_skylake_t const *)state_c, (simsimd_dot_f32x16_state_skylake_t const *)state_d,
        results);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_f16_sapphire(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

simsimd_dot_f16_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_scalars);
        b_f16x32 = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto simsimd_dot_f16_sapphire_cycle;

    *result = (simsimd_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_sapphire(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_real_f16x32 = _mm512_setzero_ph();
    __m512h sum_imag_f16x32 = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_f16x32 = _mm512_set_epi8(               //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_pairs);
        b_f16x32 = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    sum_real_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_xor_si512(b_f16x32, sign_flip_f16x32)),
                                      _mm512_castsi512_ph(a_f16x32), sum_real_f16x32);
    sum_imag_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_shuffle_epi8(b_f16x32, swap_adjacent_f16x32)),
                                      _mm512_castsi512_ph(a_f16x32), sum_imag_f16x32);
    if (count_pairs) goto simsimd_dot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    result->real = (simsimd_f32_t)_mm512_reduce_add_ph(sum_real_f16x32);
    result->imag = (simsimd_f32_t)_mm512_reduce_add_ph(sum_imag_f16x32);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_sapphire(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                               simsimd_size_t count_pairs, simsimd_f32c_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_real_f16x32 = _mm512_setzero_ph();
    __m512h sum_imag_f16x32 = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_f16x32 = _mm512_set_epi8(               //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_vdot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_pairs);
        b_f16x32 = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    sum_real_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_real_f16x32);
    a_f16x32 = _mm512_xor_si512(a_f16x32, sign_flip_f16x32);
    b_f16x32 = _mm512_shuffle_epi8(b_f16x32, swap_adjacent_f16x32);
    sum_imag_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_imag_f16x32);
    if (count_pairs) goto simsimd_vdot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    result->real = (simsimd_f32_t)_mm512_reduce_add_ph(sum_real_f16x32);
    result->imag = (simsimd_f32_t)_mm512_reduce_add_ph(sum_imag_f16x32);
}

/*  Convert 32x E4M3 values to 32x F16 values.
 *  Uses optimized path similar to E5M2 but with bias adjustment.
 *  Denormals (exp=0) are flushed to zero (DAZ behavior).
 *
 *  E4M3 format: S EEEE  MMM        (bias=7)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 *
 *  The key difference from E5M2→F16 (which is trivial) is the bias adjustment:
 *  E5M2 and F16 share bias=15, so just shift. E4M3 needs +8 to exponent.
 */
SIMSIMD_INTERNAL __m512i _simsimd_e4m3_to_f16_sapphire(__m256i e4m3_i8x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3_i8x32);
    // Sign: bit 7 → bit 15
    __m512i sign_i16x32 = _mm512_and_si512(_mm512_slli_epi16(e4m3_i16x32, 8), _mm512_set1_epi16((short)0x8000));
    // Exp+mant (7 bits) shifted left 7, then add bias adjustment (8<<10 = 0x2000)
    __m512i exp_mant_7bit_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));
    __m512i exp_mant_biased_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(exp_mant_7bit_i16x32, 7),
                                                      _mm512_set1_epi16(0x2000));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero - single instruction!
    __mmask32 nonzero_exp_mask = _mm512_test_epi16_mask(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __m512i exp_mant_daz_i16x32 = _mm512_maskz_mov_epi16(nonzero_exp_mask, exp_mant_biased_i16x32);
    return _mm512_or_si512(sign_i16x32, exp_mant_daz_i16x32);
}

/*  Convert 32x E5M2 values to 32x F16 values.
 *  This is extremely fast because E5M2 and F16 have the same exponent bias (15).
 *  Simply zero-extend to 16-bit and shift left by 8.
 *
 *  E5M2 format: S EEEEE MM         (bias=15)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 */
SIMSIMD_INTERNAL __m512i _simsimd_e5m2_to_f16_sapphire(__m256i e5m2_i8x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2_i8x32);
    return _mm512_slli_epi16(e5m2_i16x32, 8);
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_sapphire(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                              simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

simsimd_dot_e4m3_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a_scalars);
        b_e4m3x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E4M3 to F16 and compute dot product
    __m512i a_f16x32 = _simsimd_e4m3_to_f16_sapphire(a_e4m3x32);
    __m512i b_f16x32 = _simsimd_e4m3_to_f16_sapphire(b_e4m3x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto simsimd_dot_e4m3_sapphire_cycle;

    *result = (simsimd_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                              simsimd_size_t count_scalars, simsimd_f32_t *result) {
    __m256i a_e5m2x32, b_e5m2x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

simsimd_dot_e5m2_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a_scalars);
        b_e5m2x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E5M2 to F16 and compute dot product
    // Note: E5M2 to F16 is extremely fast due to same exponent bias
    __m512i a_f16x32 = _simsimd_e5m2_to_f16_sapphire(a_e5m2x32);
    __m512i b_f16x32 = _simsimd_e5m2_to_f16_sapphire(b_e5m2x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto simsimd_dot_e5m2_sapphire_cycle;

    *result = (simsimd_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

/**
 *  @brief Running state for 32-element dot accumulation over f16 scalars on Sapphire.
 */
typedef struct simsimd_dot_f16x32_state_sapphire_t {
    __m512h sum_f16x32;
} simsimd_dot_f16x32_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_sapphire(simsimd_dot_f16x32_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_sapphire(simsimd_dot_f16x32_state_sapphire_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m512i a_f16x32 = _mm512_loadu_epi16(a.f16s);
    __m512i b_f16x32 = _mm512_loadu_epi16(b.f16s);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_sapphire(                                                 //
    simsimd_dot_f16x32_state_sapphire_t const *state_a, simsimd_dot_f16x32_state_sapphire_t const *state_b, //
    simsimd_dot_f16x32_state_sapphire_t const *state_c, simsimd_dot_f16x32_state_sapphire_t const *state_d, //
    simsimd_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction for f16 (32 elements → 1 scalar each)
    // Step 1: 32→16 for all 4 states (extract high 256-bit half and add to low half)
    // Use integer extract and cast since there's no direct _mm512_extractf16x16_ph
    __m256h sum_f16x16_a = _mm256_add_ph(
        _mm512_castph512_ph256(state_a->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_a->sum_f16x32), 1)));
    __m256h sum_f16x16_b = _mm256_add_ph(
        _mm512_castph512_ph256(state_b->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_b->sum_f16x32), 1)));
    __m256h sum_f16x16_c = _mm256_add_ph(
        _mm512_castph512_ph256(state_c->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_c->sum_f16x32), 1)));
    __m256h sum_f16x16_d = _mm256_add_ph(
        _mm512_castph512_ph256(state_d->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_d->sum_f16x32), 1)));
    // Step 2: 16→8 for all 4 states (extract high 128-bit half and add to low half)
    __m128h sum_f16x8_a = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_a),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_a), 1)));
    __m128h sum_f16x8_b = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_b),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_b), 1)));
    __m128h sum_f16x8_c = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_c),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_c), 1)));
    __m128h sum_f16x8_d = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_d),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_d), 1)));
    // Step 3: 8→4 for all 4 states (shift right by 8 bytes = 4 f16 elements, then add)
    __m128h sum_f16x4_a = _mm_add_ph(sum_f16x8_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_a), 8)));
    __m128h sum_f16x4_b = _mm_add_ph(sum_f16x8_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_b), 8)));
    __m128h sum_f16x4_c = _mm_add_ph(sum_f16x8_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_c), 8)));
    __m128h sum_f16x4_d = _mm_add_ph(sum_f16x8_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_d), 8)));
    // Step 4: 4→2 for all 4 states (shift right by 4 bytes = 2 f16 elements, then add)
    __m128h sum_f16x2_a = _mm_add_ph(sum_f16x4_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_a), 4)));
    __m128h sum_f16x2_b = _mm_add_ph(sum_f16x4_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_b), 4)));
    __m128h sum_f16x2_c = _mm_add_ph(sum_f16x4_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_c), 4)));
    __m128h sum_f16x2_d = _mm_add_ph(sum_f16x4_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_d), 4)));
    // Step 5: 2→1 for all 4 states (shift right by 2 bytes = 1 f16 element, then add)
    __m128h sum_f16x1_a = _mm_add_ph(sum_f16x2_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_a), 2)));
    __m128h sum_f16x1_b = _mm_add_ph(sum_f16x2_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_b), 2)));
    __m128h sum_f16x1_c = _mm_add_ph(sum_f16x2_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_c), 2)));
    __m128h sum_f16x1_d = _mm_add_ph(sum_f16x2_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_d), 2)));
    // Extract first f16 element and convert to f32
    results[0] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_a));
    results[1] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_b));
    results[2] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_c));
    results[3] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_d));
}

/**
 *  @brief Running state for 64-element dot accumulation over e4m3 scalars on Sapphire.
 */
typedef struct simsimd_dot_e4m3x64_state_sapphire_t {
    __m512h sum_f16x32;
} simsimd_dot_e4m3x64_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m256i a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 0);
    __m256i b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 0);
    __m512i a_f16x32 = _simsimd_e4m3_to_f16_sapphire(a_e4m3x32);
    __m512i b_f16x32 = _simsimd_e4m3_to_f16_sapphire(b_e4m3x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 32);
    b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 32);
    a_f16x32 = _simsimd_e4m3_to_f16_sapphire(a_e4m3x32);
    b_f16x32 = _simsimd_e4m3_to_f16_sapphire(b_e4m3x32);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_sapphire(                                                  //
    simsimd_dot_e4m3x64_state_sapphire_t const *state_a, simsimd_dot_e4m3x64_state_sapphire_t const *state_b, //
    simsimd_dot_e4m3x64_state_sapphire_t const *state_c, simsimd_dot_e4m3x64_state_sapphire_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f16x32 (both contain just __m512h sum)
    simsimd_dot_f16x32_finalize_sapphire(                                                                           //
        (simsimd_dot_f16x32_state_sapphire_t const *)state_a, (simsimd_dot_f16x32_state_sapphire_t const *)state_b, //
        (simsimd_dot_f16x32_state_sapphire_t const *)state_c, (simsimd_dot_f16x32_state_sapphire_t const *)state_d,
        results);
}

/**
 *  @brief Running state for 64-element dot accumulation over e5m2 scalars on Sapphire.
 */
typedef struct simsimd_dot_e5m2x64_state_sapphire_t {
    __m512h sum_f16x32;
} simsimd_dot_e5m2x64_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m256i a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 0);
    __m256i b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 0);
    __m512i a_f16x32 = _simsimd_e5m2_to_f16_sapphire(a_e5m2x32);
    __m512i b_f16x32 = _simsimd_e5m2_to_f16_sapphire(b_e5m2x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 32);
    b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 32);
    a_f16x32 = _simsimd_e5m2_to_f16_sapphire(a_e5m2x32);
    b_f16x32 = _simsimd_e5m2_to_f16_sapphire(b_e5m2x32);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_sapphire(                                                  //
    simsimd_dot_e5m2x64_state_sapphire_t const *state_a, simsimd_dot_e5m2x64_state_sapphire_t const *state_b, //
    simsimd_dot_e5m2x64_state_sapphire_t const *state_c, simsimd_dot_e5m2x64_state_sapphire_t const *state_d, //
    simsimd_f32_t *results) {
    // State is layout-compatible with f16x32 (both contain just __m512h sum)
    simsimd_dot_f16x32_finalize_sapphire(                                                                           //
        (simsimd_dot_f16x32_state_sapphire_t const *)state_a, (simsimd_dot_f16x32_state_sapphire_t const *)state_b, //
        (simsimd_dot_f16x32_state_sapphire_t const *)state_c, (simsimd_dot_f16x32_state_sapphire_t const *)state_d,
        results);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_i8_ice(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                       simsimd_size_t count_scalars, simsimd_i32_t *result) {
    __m512i a_i16x32, b_i16x32;
    __m512i sum_i32x16 = _mm512_setzero_si512();

simsimd_dot_i8_ice_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b_scalars));
        count_scalars = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b_scalars));
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a_scalars.byte[4*j]) * SignExtend16(b_scalars.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_i16x32, b_i16x32);
    if (count_scalars) goto simsimd_dot_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(sum_i32x16);
}

SIMSIMD_PUBLIC void simsimd_dot_u8_ice(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                       simsimd_size_t count_scalars, simsimd_u32_t *result) {
    __m512i a_u8x64, b_u8x64;
    __m512i a_i16x32_low, a_i16x32_high, b_i16x32_low, b_i16x32_high;
    __m512i sum_i32x16_low = _mm512_setzero_si512();
    __m512i sum_i32x16_high = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

simsimd_dot_u8_ice_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a_scalars);
        b_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_i16x32_low = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    a_i16x32_high = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    b_i16x32_low = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    b_i16x32_high = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16_low = _mm512_dpwssd_epi32(sum_i32x16_low, a_i16x32_low, b_i16x32_low);
    sum_i32x16_high = _mm512_dpwssd_epi32(sum_i32x16_high, a_i16x32_high, b_i16x32_high);
    if (count_scalars) goto simsimd_dot_u8_ice_cycle;

    *result = (simsimd_u32_t)_mm512_reduce_add_epi32(_mm512_add_epi32(sum_i32x16_low, sum_i32x16_high));
}

/**
 *  @brief Running state for 64-element dot accumulation over i8 scalars on Ice Lake.
 */
typedef struct simsimd_dot_i8x64_state_ice_t {
    __m512i sum_i32x16;
} simsimd_dot_i8x64_state_ice_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_ice(simsimd_dot_i8x64_state_ice_t *state) {
    state->sum_i32x16 = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_ice(simsimd_dot_i8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b) {
    __m512i sum_i32x16 = state->sum_i32x16;
    __m512i a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(a.i8s + 0)));
    __m512i b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(b.i8s + 0)));
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_i16x32, b_i16x32);
    a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(a.i8s + 32)));
    b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(b.i8s + 32)));
    state->sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_i16x32, b_i16x32);
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_ice(                                           //
    simsimd_dot_i8x64_state_ice_t const *state_a, simsimd_dot_i8x64_state_ice_t const *state_b, //
    simsimd_dot_i8x64_state_ice_t const *state_c, simsimd_dot_i8x64_state_ice_t const *state_d, //
    simsimd_i32_t *results) {
    // ILP-optimized 4-way horizontal reduction for i32
    // Step 1: 16->8 for all 4 states (extract high 256-bit half and add to low half)
    __m256i sum_i32x8_a = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_i32x8_b = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_i32x8_c = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_i32x8_d = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));
    // Step 2: 8->4 for all 4 states (extract high 128-bit half and add to low half)
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_a), _mm256_extracti128_si256(sum_i32x8_a, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_b), _mm256_extracti128_si256(sum_i32x8_b, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_c), _mm256_extracti128_si256(sum_i32x8_c, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_d), _mm256_extracti128_si256(sum_i32x8_d, 1));
    // Step 3: Transpose 4x4 matrix of partial sums using integer shuffles
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 4: Vertical sum - each lane becomes the final i32 result for one state
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    // Store as i32
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

/**
 *  @brief Running state for 64-element dot accumulation over u8 scalars on Ice Lake.
 */
typedef struct simsimd_dot_u8x64_state_ice_t {
    __m512i sum_i32x16_low;
    __m512i sum_i32x16_high;
} simsimd_dot_u8x64_state_ice_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_ice(simsimd_dot_u8x64_state_ice_t *state) {
    state->sum_i32x16_low = _mm512_setzero_si512();
    state->sum_i32x16_high = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_ice(simsimd_dot_u8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b) {
    __m512i sum_i32x16_low = state->sum_i32x16_low;
    __m512i sum_i32x16_high = state->sum_i32x16_high;
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

    __m512i a_u8x64 = _mm512_loadu_si512(a.u8s);
    __m512i b_u8x64 = _mm512_loadu_si512(b.u8s);
    __m512i a_i16x32_low = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    __m512i a_i16x32_high = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    __m512i b_i16x32_low = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    __m512i b_i16x32_high = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    sum_i32x16_low = _mm512_dpwssd_epi32(sum_i32x16_low, a_i16x32_low, b_i16x32_low);
    sum_i32x16_high = _mm512_dpwssd_epi32(sum_i32x16_high, a_i16x32_high, b_i16x32_high);

    state->sum_i32x16_low = sum_i32x16_low;
    state->sum_i32x16_high = sum_i32x16_high;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_ice(                                           //
    simsimd_dot_u8x64_state_ice_t const *state_a, simsimd_dot_u8x64_state_ice_t const *state_b, //
    simsimd_dot_u8x64_state_ice_t const *state_c, simsimd_dot_u8x64_state_ice_t const *state_d, //
    simsimd_u32_t *results) {
    // First, combine the low and high accumulators for each state
    __m512i sum_i32x16_a = _mm512_add_epi32(state_a->sum_i32x16_low, state_a->sum_i32x16_high);
    __m512i sum_i32x16_b = _mm512_add_epi32(state_b->sum_i32x16_low, state_b->sum_i32x16_high);
    __m512i sum_i32x16_c = _mm512_add_epi32(state_c->sum_i32x16_low, state_c->sum_i32x16_high);
    __m512i sum_i32x16_d = _mm512_add_epi32(state_d->sum_i32x16_low, state_d->sum_i32x16_high);
    // ILP-optimized 4-way horizontal reduction for u32
    // Step 1: 16->8 for all 4 states
    __m256i sum_i32x8_a = _mm256_add_epi32(_mm512_castsi512_si256(sum_i32x16_a),
                                           _mm512_extracti32x8_epi32(sum_i32x16_a, 1));
    __m256i sum_i32x8_b = _mm256_add_epi32(_mm512_castsi512_si256(sum_i32x16_b),
                                           _mm512_extracti32x8_epi32(sum_i32x16_b, 1));
    __m256i sum_i32x8_c = _mm256_add_epi32(_mm512_castsi512_si256(sum_i32x16_c),
                                           _mm512_extracti32x8_epi32(sum_i32x16_c, 1));
    __m256i sum_i32x8_d = _mm256_add_epi32(_mm512_castsi512_si256(sum_i32x16_d),
                                           _mm512_extracti32x8_epi32(sum_i32x16_d, 1));
    // Step 2: 8->4 for all 4 states
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_a), _mm256_extracti128_si256(sum_i32x8_a, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_b), _mm256_extracti128_si256(sum_i32x8_b, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_c), _mm256_extracti128_si256(sum_i32x8_c, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_d), _mm256_extracti128_si256(sum_i32x8_d, 1));
    // Step 3: Transpose 4x4 matrix
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 4: Vertical sum and store as u32
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_i8_sierra(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                          simsimd_size_t count_scalars, simsimd_i32_t *result) {

    __m256i sum_i32x8 = _mm256_setzero_si256();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        sum_i32x8 = _mm256_dpbssds_epi32(sum_i32x8, a_i8x32, b_i8x32);
    }

    // Further reduce to a single sum for each vector
    int sum_i32 = _simsimd_reduce_add_i32x8_haswell(sum_i32x8);

    // Take care of the tail:
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum_i32 += (int)(a_scalars[idx_scalars]) * b_scalars[idx_scalars];
    *result = sum_i32;
}

/**
 *  @brief Running state for 32-element dot accumulation over i8 scalars on Sierra.
 */
typedef struct simsimd_dot_i8x32_state_sierra_t {
    __m256i sum_i32x8;
} simsimd_dot_i8x32_state_sierra_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x32_init_sierra(simsimd_dot_i8x32_state_sierra_t *state) {
    state->sum_i32x8 = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x32_update_sierra(simsimd_dot_i8x32_state_sierra_t *state, simsimd_b256_vec_t a,
                                                      simsimd_b256_vec_t b) {
    __m256i sum_i32x8 = state->sum_i32x8;
    __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a.i8s));
    __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b.i8s));
    state->sum_i32x8 = _mm256_dpbssds_epi32(sum_i32x8, a_i8x32, b_i8x32);
}

SIMSIMD_INTERNAL void simsimd_dot_i8x32_finalize_sierra(                                              //
    simsimd_dot_i8x32_state_sierra_t const *state_a, simsimd_dot_i8x32_state_sierra_t const *state_b, //
    simsimd_dot_i8x32_state_sierra_t const *state_c, simsimd_dot_i8x32_state_sierra_t const *state_d, //
    simsimd_i32_t *results) {
    // ILP-optimized 4-way horizontal reduction for i32 in AVX2 (8 elements -> 1 scalar each)
    // Step 1: 8->4 for all 4 states (extract high 128-bit half and add to low half)
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(state_a->sum_i32x8),
                                        _mm256_extracti128_si256(state_a->sum_i32x8, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(state_b->sum_i32x8),
                                        _mm256_extracti128_si256(state_b->sum_i32x8, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(state_c->sum_i32x8),
                                        _mm256_extracti128_si256(state_c->sum_i32x8, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(state_d->sum_i32x8),
                                        _mm256_extracti128_si256(state_d->sum_i32x8, 1));
    // Step 2: Transpose 4x4 matrix of partial sums using integer shuffles
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 3: Vertical sum - each lane becomes the final i32 result for one state
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    // Store as i32
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SIERRA
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_dot_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                   simsimd_i32_t *result) {
#if SIMSIMD_TARGET_NEON_I8
    simsimd_dot_i8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_dot_i8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_i8_haswell(a, b, n, result);
#else
    simsimd_dot_i8_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                   simsimd_u32_t *result) {
#if SIMSIMD_TARGET_NEON_I8
    simsimd_dot_u8_neon(a, b, n, result);
#elif SIMSIMD_TARGET_ICE
    simsimd_dot_u8_ice(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_u8_haswell(a, b, n, result);
#else
    simsimd_dot_u8_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                    simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE_F16
    simsimd_dot_f16_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_dot_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_f16_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f16_haswell(a, b, n, result);
#else
    simsimd_dot_f16_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result) {
#if SIMSIMD_TARGET_GENOA
    simsimd_dot_bf16_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_bf16_haswell(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_dot_bf16_neon(a, b, n, result);
#else
    simsimd_dot_bf16_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_e4m3(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_e4m3_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dot_e4m3_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_e4m3_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_e4m3_haswell(a, b, n, result);
#else
    simsimd_dot_e4m3_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_e5m2(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                     simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_e5m2_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dot_e5m2_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_e5m2_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_e5m2_haswell(a, b, n, result);
#else
    simsimd_dot_e5m2_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                    simsimd_f32_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f32_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_dot_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f32_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f32_haswell(a, b, n, result);
#else
    simsimd_dot_f32_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                    simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f64_sve(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f64_skylake(a, b, n, result);
#else
    simsimd_dot_f64_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f16c(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                     simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_SVE_F16
    simsimd_dot_f16c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_dot_f16c_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_f16c_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f16c_haswell(a, b, n, result);
#else
    simsimd_dot_f16c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_bf16c(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                      simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_GENOA
    simsimd_dot_bf16c_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_dot_bf16c_neon(a, b, n, result);
#else
    simsimd_dot_bf16c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f32c(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                     simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f32c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_dot_f32c_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f32c_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f32c_haswell(a, b, n, result);
#else
    simsimd_dot_f32c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f64c(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                     simsimd_f64c_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f64c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f64c_skylake(a, b, n, result);
#else
    simsimd_dot_f64c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f16c(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                      simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f16c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_vdot_f16c_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_vdot_f16c_sapphire(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vdot_f16c_haswell(a, b, n, result);
#else
    simsimd_vdot_f16c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_bf16c(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                       simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_GENOA
    simsimd_vdot_bf16c_genoa(a, b, n, result);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_vdot_bf16c_neon(a, b, n, result);
#else
    simsimd_vdot_bf16c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f32c(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                      simsimd_f32c_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f32c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_vdot_f32c_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_vdot_f32c_skylake(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vdot_f32c_haswell(a, b, n, result);
#else
    simsimd_vdot_f32c_serial(a, b, n, result);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f64c(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                      simsimd_f64c_t *result) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f64c_sve(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_vdot_f64c_skylake(a, b, n, result);
#else
    simsimd_vdot_f64c_serial(a, b, n, result);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
