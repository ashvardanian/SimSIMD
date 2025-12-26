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
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integers
 *  - 8-bit signed integers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire, Sierra
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
 *  Integer i8 dot products use VPMADDUBSW (u8*i8->i16) + VPMADDWD (i16*1->i32) on Haswell,
 *  or the newer VNNI instructions VPDPBUSD/VPDPWSSD on Ice Lake+ for direct u8*i8->i32.
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
 *  NEON integer dot products use SDOT/UDOT (ARMv8.2 dotprod) for direct i8*i8->i32 or u8*u8->u32
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
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef SIMSIMD_DOT_H
#define SIMSIMD_DOT_H

#include "types.h"

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
                                     simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_e4m3(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result);
/** @copydoc simsimd_dot_f32 */
SIMSIMD_DYNAMIC void simsimd_dot_e5m2(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result);

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
                                      simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_f64c(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                      simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_f16c(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                      simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_DYNAMIC void simsimd_dot_bf16c(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_distance_t *results);

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
                                       simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_f64c(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_f16c(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                       simsimd_size_t count_pairs, simsimd_distance_t *results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_DYNAMIC void simsimd_vdot_bf16c(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                        simsimd_size_t count_pairs, simsimd_distance_t *results);

// clang-format off

/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_serial(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_serial(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_serial(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_serial(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_serial(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_serial(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_serial(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_serial(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_serial(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_serial(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_serial(simsimd_e4m3_t const* a, simsimd_e4m3_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_serial(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_accurate(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_accurate(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_accurate(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_accurate(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_accurate(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_accurate(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_neon(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_neon(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results)

typedef struct simsimd_dot_f32x16_state_neon_t simsimd_dot_f32x16_state_neon_t;
/** @copydoc simsimd_dot_f32x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_neon(simsimd_dot_f32x16_state_neon_t *state);
/** @copydoc simsimd_dot_f32x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_neon(simsimd_dot_f32x16_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f32x16_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_neon(simsimd_dot_f32x16_state_neon_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_neon(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_neon(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results)

typedef struct simsimd_dot_f16x32_state_neon_t simsimd_dot_f16x32_state_neon_t;
/** @copydoc simsimd_dot_f16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_neon(simsimd_dot_f16x32_state_neon_t *state);
/** @copydoc simsimd_dot_f16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_neon(simsimd_dot_f16x32_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_neon(simsimd_dot_f16x32_state_neon_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_I8
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_neon(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_neon(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_i8x64_state_neon_t simsimd_dot_i8x64_state_neon_t;
/** @copydoc simsimd_dot_i8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_neon(simsimd_dot_i8x64_state_neon_t *state);
/** @copydoc simsimd_dot_i8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_neon(simsimd_dot_i8x64_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_neon(simsimd_dot_i8x64_state_neon_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_u8x64_state_neon_t simsimd_dot_u8x64_state_neon_t;
/** @copydoc simsimd_dot_u8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_neon(simsimd_dot_u8x64_state_neon_t *state);
/** @copydoc simsimd_dot_u8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_neon(simsimd_dot_u8x64_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_u8x64_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_neon(simsimd_dot_u8x64_state_neon_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_NEON_I8

#if SIMSIMD_TARGET_NEON_BF16
/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_neon(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_neon(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_neon(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results)

typedef struct simsimd_dot_bf16x32_state_neon_t simsimd_dot_bf16x32_state_neon_t;
/** @copydoc simsimd_dot_bf16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_neon(simsimd_dot_bf16x32_state_neon_t *state);
/** @copydoc simsimd_dot_bf16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_neon(simsimd_dot_bf16x32_state_neon_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_bf16x32_state_neon_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_neon(simsimd_dot_bf16x32_state_neon_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_SVE
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_sve(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_sve(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_sve(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_sve(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_sve(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_sve(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_sve(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_sve(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_sve(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
#endif // SIMSIMD_TARGET_SVE

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_haswell(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_haswell(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_haswell(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_haswell(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_haswell(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_haswell(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_haswell(simsimd_e4m3_t const* a, simsimd_e4m3_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_haswell(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_haswell(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_haswell(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_f32x16_state_haswell_t simsimd_dot_f32x16_state_haswell_t;
/** @copydoc simsimd_dot_f32x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_haswell(simsimd_dot_f32x16_state_haswell_t *state);
/** @copydoc simsimd_dot_f32x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_haswell(simsimd_dot_f32x16_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f32x16_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_haswell(simsimd_dot_f32x16_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_f16x32_state_haswell_t simsimd_dot_f16x32_state_haswell_t;
/** @copydoc simsimd_dot_f16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_haswell(simsimd_dot_f16x32_state_haswell_t *state);
/** @copydoc simsimd_dot_f16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_haswell(simsimd_dot_f16x32_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_haswell(simsimd_dot_f16x32_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_bf16x32_state_haswell_t simsimd_dot_bf16x32_state_haswell_t;
/** @copydoc simsimd_dot_bf16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_haswell(simsimd_dot_bf16x32_state_haswell_t *state);
/** @copydoc simsimd_dot_bf16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_haswell(simsimd_dot_bf16x32_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_bf16x32_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_haswell(simsimd_dot_bf16x32_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e4m3x64_state_haswell_t simsimd_dot_e4m3x64_state_haswell_t;
/** @copydoc simsimd_dot_e4m3x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_haswell(simsimd_dot_e4m3x64_state_haswell_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_haswell(simsimd_dot_e4m3x64_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_haswell(simsimd_dot_e4m3x64_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e5m2x64_state_haswell_t simsimd_dot_e5m2x64_state_haswell_t;
/** @copydoc simsimd_dot_e5m2x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_haswell(simsimd_dot_e5m2x64_state_haswell_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_haswell(simsimd_dot_e5m2x64_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_haswell(simsimd_dot_e5m2x64_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_i8x64_state_haswell_t simsimd_dot_i8x64_state_haswell_t;
/** @copydoc simsimd_dot_i8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_haswell(simsimd_dot_i8x64_state_haswell_t *state);
/** @copydoc simsimd_dot_i8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_haswell(simsimd_dot_i8x64_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_haswell(simsimd_dot_i8x64_state_haswell_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_u8x64_state_haswell_t simsimd_dot_u8x64_state_haswell_t;
/** @copydoc simsimd_dot_u8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_haswell(simsimd_dot_u8x64_state_haswell_t *state);
/** @copydoc simsimd_dot_u8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_haswell(simsimd_dot_u8x64_state_haswell_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_u8x64_state_haswell_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_haswell(simsimd_dot_u8x64_state_haswell_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_dot_f64 */
SIMSIMD_PUBLIC void simsimd_dot_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f64c */
SIMSIMD_PUBLIC void simsimd_dot_f64c_skylake(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f64c */
SIMSIMD_PUBLIC void simsimd_vdot_f64c_skylake(simsimd_f64c_t const* a, simsimd_f64c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_f32 */
SIMSIMD_PUBLIC void simsimd_dot_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f32c */
SIMSIMD_PUBLIC void simsimd_dot_f32c_skylake(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f32c */
SIMSIMD_PUBLIC void simsimd_vdot_f32c_skylake(simsimd_f32c_t const* a, simsimd_f32c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_skylake(simsimd_e4m3_t const* a, simsimd_e4m3_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_skylake(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_f64x8_state_skylake_t simsimd_dot_f64x8_state_skylake_t;
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_skylake(simsimd_dot_f64x8_state_skylake_t *state);
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_skylake(simsimd_dot_f64x8_state_skylake_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f64x8_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_skylake(simsimd_dot_f64x8_state_skylake_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_f32x16_state_skylake_t simsimd_dot_f32x16_state_skylake_t;
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_skylake(simsimd_dot_f32x16_state_skylake_t *state);
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_skylake(simsimd_dot_f32x16_state_skylake_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f32x16_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_skylake(simsimd_dot_f32x16_state_skylake_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e4m3x64_state_skylake_t simsimd_dot_e4m3x64_state_skylake_t;
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_skylake(simsimd_dot_e4m3x64_state_skylake_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_skylake(simsimd_dot_e4m3x64_state_skylake_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_skylake(simsimd_dot_e4m3x64_state_skylake_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e5m2x64_state_skylake_t simsimd_dot_e5m2x64_state_skylake_t;
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_skylake(simsimd_dot_e5m2x64_state_skylake_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_skylake(simsimd_dot_e5m2x64_state_skylake_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_skylake_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_skylake(simsimd_dot_e5m2x64_state_skylake_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_ICE
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_ice(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_u8 */
SIMSIMD_PUBLIC void simsimd_dot_u8_ice(simsimd_u8_t const* a, simsimd_u8_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_i8x64_state_ice_t simsimd_dot_i8x64_state_ice_t;
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_ice(simsimd_dot_i8x64_state_ice_t *state);
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_ice(simsimd_dot_i8x64_state_ice_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_ice(simsimd_dot_i8x64_state_ice_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_u8x64_state_ice_t simsimd_dot_u8x64_state_ice_t;
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_ice(simsimd_dot_u8x64_state_ice_t *state);
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_ice(simsimd_dot_u8x64_state_ice_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_u8x64_state_ice_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_ice(simsimd_dot_u8x64_state_ice_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_GENOA
/** @copydoc simsimd_dot_bf16 */
SIMSIMD_PUBLIC void simsimd_dot_bf16_genoa(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_bf16c */
SIMSIMD_PUBLIC void simsimd_dot_bf16c_genoa(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_bf16c */
SIMSIMD_PUBLIC void simsimd_vdot_bf16c_genoa(simsimd_bf16c_t const* a, simsimd_bf16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_genoa(simsimd_e4m3_t const* a, simsimd_e4m3_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_genoa(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_bf16x32_state_genoa_t simsimd_dot_bf16x32_state_genoa_t;
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_genoa(simsimd_dot_bf16x32_state_genoa_t *state);
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_genoa(simsimd_dot_bf16x32_state_genoa_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_bf16x32_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_genoa(simsimd_dot_bf16x32_state_genoa_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e4m3x64_state_genoa_t simsimd_dot_e4m3x64_state_genoa_t;
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_genoa(simsimd_dot_e4m3x64_state_genoa_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_genoa(simsimd_dot_e4m3x64_state_genoa_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_genoa(simsimd_dot_e4m3x64_state_genoa_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e5m2x64_state_genoa_t simsimd_dot_e5m2x64_state_genoa_t;
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_genoa(simsimd_dot_e5m2x64_state_genoa_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_genoa(simsimd_dot_e5m2x64_state_genoa_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_genoa_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_genoa(simsimd_dot_e5m2x64_state_genoa_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
/** @copydoc simsimd_dot_f16 */
SIMSIMD_PUBLIC void simsimd_dot_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_f16c */
SIMSIMD_PUBLIC void simsimd_dot_f16c_sapphire(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);
/** @copydoc simsimd_vdot_f16c */
SIMSIMD_PUBLIC void simsimd_vdot_f16c_sapphire(simsimd_f16c_t const* a, simsimd_f16c_t const* b, simsimd_size_t n, simsimd_distance_t* results);

/** @copydoc simsimd_dot_e4m3 */
SIMSIMD_PUBLIC void simsimd_dot_e4m3_sapphire(simsimd_e4m3_t const* a, simsimd_e4m3_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_dot_e5m2 */
SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire_lut(simsimd_e5m2_t const* a, simsimd_e5m2_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_f16x32_state_sapphire_t simsimd_dot_f16x32_state_sapphire_t;
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_sapphire(simsimd_dot_f16x32_state_sapphire_t *state);
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_sapphire(simsimd_dot_f16x32_state_sapphire_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f16x32_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_sapphire(simsimd_dot_f16x32_state_sapphire_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e4m3x64_state_sapphire_t simsimd_dot_e4m3x64_state_sapphire_t;
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_sapphire(simsimd_dot_e4m3x64_state_sapphire_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e5m2x64_state_sapphire_t simsimd_dot_e5m2x64_state_sapphire_t;
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_sapphire_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_sapphire(simsimd_dot_e5m2x64_state_sapphire_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_SIERRA
/** @copydoc simsimd_dot_i8 */
SIMSIMD_PUBLIC void simsimd_dot_i8_sierra(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n, simsimd_distance_t* result);

typedef struct simsimd_dot_i8x64_state_sierra_t simsimd_dot_i8x64_state_sierra_t;
/** @copydoc simsimd_dot_i8x64_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_sierra(simsimd_dot_i8x64_state_sierra_t *state);
/** @copydoc simsimd_dot_i8x64_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_sierra(simsimd_dot_i8x64_state_sierra_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_sierra_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_sierra(simsimd_dot_i8x64_state_sierra_t const *state, simsimd_distance_t *result);
#endif // SIMSIMD_TARGET_SIERRA

typedef struct simsimd_dot_f64x8_state_serial_t simsimd_dot_f64x8_state_serial_t;
/** @copydoc simsimd_dot_f64x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_serial(simsimd_dot_f64x8_state_serial_t *state);
/** @copydoc simsimd_dot_f64x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_serial(simsimd_dot_f64x8_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f64x8_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_serial(simsimd_dot_f64x8_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_f32x16_state_serial_t simsimd_dot_f32x16_state_serial_t;
/** @copydoc simsimd_dot_f32x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_serial(simsimd_dot_f32x16_state_serial_t *state);
/** @copydoc simsimd_dot_f32x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_serial(simsimd_dot_f32x16_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f32x16_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_serial(simsimd_dot_f32x16_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_f16x32_state_serial_t simsimd_dot_f16x32_state_serial_t;
/** @copydoc simsimd_dot_f16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_serial(simsimd_dot_f16x32_state_serial_t *state);
/** @copydoc simsimd_dot_f16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_serial(simsimd_dot_f16x32_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_f16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_serial(simsimd_dot_f16x32_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_bf16x32_state_serial_t simsimd_dot_bf16x32_state_serial_t;
/** @copydoc simsimd_dot_bf16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_serial(simsimd_dot_bf16x32_state_serial_t *state);
/** @copydoc simsimd_dot_bf16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_serial(simsimd_dot_bf16x32_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_bf16x32_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_serial(simsimd_dot_bf16x32_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_i8x64_state_serial_t simsimd_dot_i8x64_state_serial_t;
/** @copydoc simsimd_dot_i8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_serial(simsimd_dot_i8x64_state_serial_t *state);
/** @copydoc simsimd_dot_i8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_serial(simsimd_dot_i8x64_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_i8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_serial(simsimd_dot_i8x64_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_u8x64_state_serial_t simsimd_dot_u8x64_state_serial_t;
/** @copydoc simsimd_dot_u8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_serial(simsimd_dot_u8x64_state_serial_t *state);
/** @copydoc simsimd_dot_u8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_serial(simsimd_dot_u8x64_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_u8x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_serial(simsimd_dot_u8x64_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e4m3x64_state_serial_t simsimd_dot_e4m3x64_state_serial_t;
/** @copydoc simsimd_dot_e4m3x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_serial(simsimd_dot_e4m3x64_state_serial_t *state);
/** @copydoc simsimd_dot_e4m3x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_serial(simsimd_dot_e4m3x64_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e4m3x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_serial(simsimd_dot_e4m3x64_state_serial_t const *state, simsimd_distance_t *result);

typedef struct simsimd_dot_e5m2x64_state_serial_t simsimd_dot_e5m2x64_state_serial_t;
/** @copydoc simsimd_dot_e5m2x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_serial(simsimd_dot_e5m2x64_state_serial_t *state);
/** @copydoc simsimd_dot_e5m2x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_serial(simsimd_dot_e5m2x64_state_serial_t *state, simsimd_b512_vec_t a, simsimd_b512_vec_t b);
/** @copydoc simsimd_dot_e5m2x64_state_serial_t */
SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_serial(simsimd_dot_e5m2x64_state_serial_t const *state, simsimd_distance_t *result);
// clang-format on

#define SIMSIMD_MAKE_DOT(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_dot_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_distance_t *result) {                        \
        simsimd_##accumulator_type##_t ab = 0, ai, bi;                                                         \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            load_and_convert(a + i, &ai);                                                                      \
            load_and_convert(b + i, &bi);                                                                      \
            ab += ai * bi;                                                                                     \
        }                                                                                                      \
        *result = ab;                                                                                          \
    }

#define SIMSIMD_MAKE_COMPLEX_DOT(name, input_type, accumulator_type, load_and_convert)                               \
    SIMSIMD_PUBLIC void simsimd_dot_##input_type##_##name(simsimd_##input_type##_t const *a_pairs,                   \
                                                          simsimd_##input_type##_t const *b_pairs,                   \
                                                          simsimd_size_t count_pairs, simsimd_distance_t *results) { \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                     \
        for (simsimd_size_t i = 0; i != count_pairs; ++i) {                                                          \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                             \
            load_and_convert(&(b_pairs + i)->real, &br);                                                             \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                             \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                             \
            ab_real += ar * br - ai * bi;                                                                            \
            ab_imag += ar * bi + ai * br;                                                                            \
        }                                                                                                            \
        results[0] = ab_real;                                                                                        \
        results[1] = ab_imag;                                                                                        \
    }

#define SIMSIMD_MAKE_COMPLEX_VDOT(name, input_type, accumulator_type, load_and_convert)                               \
    SIMSIMD_PUBLIC void simsimd_vdot_##input_type##_##name(simsimd_##input_type##_t const *a_pairs,                   \
                                                           simsimd_##input_type##_t const *b_pairs,                   \
                                                           simsimd_size_t count_pairs, simsimd_distance_t *results) { \
        simsimd_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                      \
        for (simsimd_size_t i = 0; i != count_pairs; ++i) {                                                           \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                              \
            load_and_convert(&(b_pairs + i)->real, &br);                                                              \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                              \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                              \
            ab_real += ar * br + ai * bi;                                                                             \
            ab_imag += ar * bi - ai * br;                                                                             \
        }                                                                                                             \
        results[0] = ab_real;                                                                                         \
        results[1] = ab_imag;                                                                                         \
    }

SIMSIMD_MAKE_DOT(serial, f64, f64, SIMSIMD_ASSIGN_FROM_TO)           // simsimd_dot_f64_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f64c, f64, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f64c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f64c, f64, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f64c_serial

SIMSIMD_MAKE_DOT(serial, f32, f32, SIMSIMD_ASSIGN_FROM_TO)           // simsimd_dot_f32_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f32c, f32, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f32c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f32c, f32, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f32c_serial

SIMSIMD_MAKE_DOT(serial, f16, f32, simsimd_f16_to_f32)           // simsimd_dot_f16_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, f16c, f32, simsimd_f16_to_f32)  // simsimd_dot_f16c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, f16c, f32, simsimd_f16_to_f32) // simsimd_vdot_f16c_serial

SIMSIMD_MAKE_DOT(serial, bf16, f32, simsimd_bf16_to_f32)           // simsimd_dot_bf16_serial
SIMSIMD_MAKE_COMPLEX_DOT(serial, bf16c, f32, simsimd_bf16_to_f32)  // simsimd_dot_bf16c_serial
SIMSIMD_MAKE_COMPLEX_VDOT(serial, bf16c, f32, simsimd_bf16_to_f32) // simsimd_vdot_bf16c_serial

SIMSIMD_MAKE_DOT(serial, i8, i64, SIMSIMD_ASSIGN_FROM_TO) // simsimd_dot_i8_serial
SIMSIMD_MAKE_DOT(serial, u8, i64, SIMSIMD_ASSIGN_FROM_TO) // simsimd_dot_u8_serial

SIMSIMD_PUBLIC void simsimd_dot_e4m3_serial(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_f32_t ab = 0, ai, bi;
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_e4m3_to_f32(a + i, &ai);
        simsimd_e4m3_to_f32(b + i, &bi);
        ab += ai * bi;
    }
    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_serial(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_f32_t ab = 0, ai, bi;
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_e5m2_to_f32(a + i, &ai);
        simsimd_e5m2_to_f32(b + i, &bi);
        ab += ai * bi;
    }
    *result = ab;
}

SIMSIMD_MAKE_DOT(accurate, f32, f64, SIMSIMD_ASSIGN_FROM_TO)           // simsimd_dot_f32_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f32c, f64, SIMSIMD_ASSIGN_FROM_TO)  // simsimd_dot_f32c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f32c, f64, SIMSIMD_ASSIGN_FROM_TO) // simsimd_vdot_f32c_accurate

SIMSIMD_MAKE_DOT(accurate, f16, f64, simsimd_f16_to_f64)           // simsimd_dot_f16_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, f16c, f64, simsimd_f16_to_f64)  // simsimd_dot_f16c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, f16c, f64, simsimd_f16_to_f64) // simsimd_vdot_f16c_accurate

SIMSIMD_MAKE_DOT(accurate, bf16, f64, simsimd_bf16_to_f64)           // simsimd_dot_bf16_accurate
SIMSIMD_MAKE_COMPLEX_DOT(accurate, bf16c, f64, simsimd_bf16_to_f64)  // simsimd_dot_bf16c_accurate
SIMSIMD_MAKE_COMPLEX_VDOT(accurate, bf16c, f64, simsimd_bf16_to_f64) // simsimd_vdot_bf16c_accurate

/**
 *  @brief Running state for 512-bit dot accumulation over f64 scalars.
 */
typedef struct simsimd_dot_f64x8_state_serial_t {
    simsimd_f64_t sums[2];
} simsimd_dot_f64x8_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_serial(simsimd_dot_f64x8_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_serial(simsimd_dot_f64x8_state_serial_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    simsimd_f64_t sum0 = state->sums[0];
    simsimd_f64_t sum1 = state->sums[1];

    sum0 += a.f64s[0] * b.f64s[0], sum1 += a.f64s[1] * b.f64s[1];
    sum0 += a.f64s[2] * b.f64s[2], sum1 += a.f64s[3] * b.f64s[3];
    sum0 += a.f64s[4] * b.f64s[4], sum1 += a.f64s[5] * b.f64s[5];
    sum0 += a.f64s[6] * b.f64s[6], sum1 += a.f64s[7] * b.f64s[7];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_serial(simsimd_dot_f64x8_state_serial_t const *state,
                                                        simsimd_distance_t *result) {
    simsimd_f64_t sum = 0;
    for (simsimd_size_t i = 0; i != 2; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over f32 scalars.
 */
typedef struct simsimd_dot_f32x16_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_f32x16_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_serial(simsimd_dot_f32x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_serial(simsimd_dot_f32x16_state_serial_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];

    sum0 += a.f32s[0] * b.f32s[0], sum1 += a.f32s[1] * b.f32s[1], sum2 += a.f32s[2] * b.f32s[2],
        sum3 += a.f32s[3] * b.f32s[3];
    sum0 += a.f32s[4] * b.f32s[4], sum1 += a.f32s[5] * b.f32s[5], sum2 += a.f32s[6] * b.f32s[6],
        sum3 += a.f32s[7] * b.f32s[7];
    sum0 += a.f32s[8] * b.f32s[8], sum1 += a.f32s[9] * b.f32s[9], sum2 += a.f32s[10] * b.f32s[10],
        sum3 += a.f32s[11] * b.f32s[11];
    sum0 += a.f32s[12] * b.f32s[12], sum1 += a.f32s[13] * b.f32s[13], sum2 += a.f32s[14] * b.f32s[14],
        sum3 += a.f32s[15] * b.f32s[15];

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_serial(simsimd_dot_f32x16_state_serial_t const *state,
                                                         simsimd_distance_t *result) {
    simsimd_f32_t sum = 0;
    for (simsimd_size_t i = 0; i != 4; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over f16 scalars.
 */
typedef struct simsimd_dot_f16x32_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_f16x32_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_serial(simsimd_dot_f16x32_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_serial(simsimd_dot_f16x32_state_serial_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (simsimd_size_t i = 0; i < 32; i += 4) {
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

SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_serial(simsimd_dot_f16x32_state_serial_t const *state,
                                                         simsimd_distance_t *result) {
    simsimd_f32_t sum = 0;
    for (simsimd_size_t i = 0; i != 4; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over bf16 scalars.
 */
typedef struct simsimd_dot_bf16x32_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_bf16x32_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_serial(simsimd_dot_bf16x32_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_serial(simsimd_dot_bf16x32_state_serial_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (simsimd_size_t i = 0; i < 32; i += 4) {
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

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_serial(simsimd_dot_bf16x32_state_serial_t const *state,
                                                          simsimd_distance_t *result) {
    simsimd_f32_t sum = 0;
    for (simsimd_size_t i = 0; i != 4; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over i8 scalars.
 */
typedef struct simsimd_dot_i8x64_state_serial_t {
    simsimd_i64_t sums[2];
} simsimd_dot_i8x64_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_serial(simsimd_dot_i8x64_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_serial(simsimd_dot_i8x64_state_serial_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    simsimd_i64_t sum0 = state->sums[0];
    simsimd_i64_t sum1 = state->sums[1];

    sum0 += (simsimd_i16_t)a.i8s[0] * (simsimd_i16_t)b.i8s[0],
        sum1 += (simsimd_i16_t)a.i8s[1] * (simsimd_i16_t)b.i8s[1];
    sum0 += (simsimd_i16_t)a.i8s[2] * (simsimd_i16_t)b.i8s[2],
        sum1 += (simsimd_i16_t)a.i8s[3] * (simsimd_i16_t)b.i8s[3];
    sum0 += (simsimd_i16_t)a.i8s[4] * (simsimd_i16_t)b.i8s[4],
        sum1 += (simsimd_i16_t)a.i8s[5] * (simsimd_i16_t)b.i8s[5];
    sum0 += (simsimd_i16_t)a.i8s[6] * (simsimd_i16_t)b.i8s[6],
        sum1 += (simsimd_i16_t)a.i8s[7] * (simsimd_i16_t)b.i8s[7];
    sum0 += (simsimd_i16_t)a.i8s[8] * (simsimd_i16_t)b.i8s[8],
        sum1 += (simsimd_i16_t)a.i8s[9] * (simsimd_i16_t)b.i8s[9];
    sum0 += (simsimd_i16_t)a.i8s[10] * (simsimd_i16_t)b.i8s[10],
        sum1 += (simsimd_i16_t)a.i8s[11] * (simsimd_i16_t)b.i8s[11];
    sum0 += (simsimd_i16_t)a.i8s[12] * (simsimd_i16_t)b.i8s[12],
        sum1 += (simsimd_i16_t)a.i8s[13] * (simsimd_i16_t)b.i8s[13];
    sum0 += (simsimd_i16_t)a.i8s[14] * (simsimd_i16_t)b.i8s[14],
        sum1 += (simsimd_i16_t)a.i8s[15] * (simsimd_i16_t)b.i8s[15];
    sum0 += (simsimd_i16_t)a.i8s[16] * (simsimd_i16_t)b.i8s[16],
        sum1 += (simsimd_i16_t)a.i8s[17] * (simsimd_i16_t)b.i8s[17];
    sum0 += (simsimd_i16_t)a.i8s[18] * (simsimd_i16_t)b.i8s[18],
        sum1 += (simsimd_i16_t)a.i8s[19] * (simsimd_i16_t)b.i8s[19];
    sum0 += (simsimd_i16_t)a.i8s[20] * (simsimd_i16_t)b.i8s[20],
        sum1 += (simsimd_i16_t)a.i8s[21] * (simsimd_i16_t)b.i8s[21];
    sum0 += (simsimd_i16_t)a.i8s[22] * (simsimd_i16_t)b.i8s[22],
        sum1 += (simsimd_i16_t)a.i8s[23] * (simsimd_i16_t)b.i8s[23];
    sum0 += (simsimd_i16_t)a.i8s[24] * (simsimd_i16_t)b.i8s[24],
        sum1 += (simsimd_i16_t)a.i8s[25] * (simsimd_i16_t)b.i8s[25];
    sum0 += (simsimd_i16_t)a.i8s[26] * (simsimd_i16_t)b.i8s[26],
        sum1 += (simsimd_i16_t)a.i8s[27] * (simsimd_i16_t)b.i8s[27];
    sum0 += (simsimd_i16_t)a.i8s[28] * (simsimd_i16_t)b.i8s[28],
        sum1 += (simsimd_i16_t)a.i8s[29] * (simsimd_i16_t)b.i8s[29];
    sum0 += (simsimd_i16_t)a.i8s[30] * (simsimd_i16_t)b.i8s[30],
        sum1 += (simsimd_i16_t)a.i8s[31] * (simsimd_i16_t)b.i8s[31];
    sum0 += (simsimd_i16_t)a.i8s[32] * (simsimd_i16_t)b.i8s[32],
        sum1 += (simsimd_i16_t)a.i8s[33] * (simsimd_i16_t)b.i8s[33];
    sum0 += (simsimd_i16_t)a.i8s[34] * (simsimd_i16_t)b.i8s[34],
        sum1 += (simsimd_i16_t)a.i8s[35] * (simsimd_i16_t)b.i8s[35];
    sum0 += (simsimd_i16_t)a.i8s[36] * (simsimd_i16_t)b.i8s[36],
        sum1 += (simsimd_i16_t)a.i8s[37] * (simsimd_i16_t)b.i8s[37];
    sum0 += (simsimd_i16_t)a.i8s[38] * (simsimd_i16_t)b.i8s[38],
        sum1 += (simsimd_i16_t)a.i8s[39] * (simsimd_i16_t)b.i8s[39];
    sum0 += (simsimd_i16_t)a.i8s[40] * (simsimd_i16_t)b.i8s[40],
        sum1 += (simsimd_i16_t)a.i8s[41] * (simsimd_i16_t)b.i8s[41];
    sum0 += (simsimd_i16_t)a.i8s[42] * (simsimd_i16_t)b.i8s[42],
        sum1 += (simsimd_i16_t)a.i8s[43] * (simsimd_i16_t)b.i8s[43];
    sum0 += (simsimd_i16_t)a.i8s[44] * (simsimd_i16_t)b.i8s[44],
        sum1 += (simsimd_i16_t)a.i8s[45] * (simsimd_i16_t)b.i8s[45];
    sum0 += (simsimd_i16_t)a.i8s[46] * (simsimd_i16_t)b.i8s[46],
        sum1 += (simsimd_i16_t)a.i8s[47] * (simsimd_i16_t)b.i8s[47];
    sum0 += (simsimd_i16_t)a.i8s[48] * (simsimd_i16_t)b.i8s[48],
        sum1 += (simsimd_i16_t)a.i8s[49] * (simsimd_i16_t)b.i8s[49];
    sum0 += (simsimd_i16_t)a.i8s[50] * (simsimd_i16_t)b.i8s[50],
        sum1 += (simsimd_i16_t)a.i8s[51] * (simsimd_i16_t)b.i8s[51];
    sum0 += (simsimd_i16_t)a.i8s[52] * (simsimd_i16_t)b.i8s[52],
        sum1 += (simsimd_i16_t)a.i8s[53] * (simsimd_i16_t)b.i8s[53];
    sum0 += (simsimd_i16_t)a.i8s[54] * (simsimd_i16_t)b.i8s[54],
        sum1 += (simsimd_i16_t)a.i8s[55] * (simsimd_i16_t)b.i8s[55];
    sum0 += (simsimd_i16_t)a.i8s[56] * (simsimd_i16_t)b.i8s[56],
        sum1 += (simsimd_i16_t)a.i8s[57] * (simsimd_i16_t)b.i8s[57];
    sum0 += (simsimd_i16_t)a.i8s[58] * (simsimd_i16_t)b.i8s[58],
        sum1 += (simsimd_i16_t)a.i8s[59] * (simsimd_i16_t)b.i8s[59];
    sum0 += (simsimd_i16_t)a.i8s[60] * (simsimd_i16_t)b.i8s[60],
        sum1 += (simsimd_i16_t)a.i8s[61] * (simsimd_i16_t)b.i8s[61];
    sum0 += (simsimd_i16_t)a.i8s[62] * (simsimd_i16_t)b.i8s[62],
        sum1 += (simsimd_i16_t)a.i8s[63] * (simsimd_i16_t)b.i8s[63];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_serial(simsimd_dot_i8x64_state_serial_t const *state,
                                                        simsimd_distance_t *result) {
    simsimd_i64_t sum = 0;
    for (simsimd_size_t i = 0; i != 2; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over u8 scalars.
 */
typedef struct simsimd_dot_u8x64_state_serial_t {
    simsimd_u64_t sums[2];
} simsimd_dot_u8x64_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_serial(simsimd_dot_u8x64_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_serial(simsimd_dot_u8x64_state_serial_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    simsimd_u64_t sum0 = state->sums[0];
    simsimd_u64_t sum1 = state->sums[1];

    sum0 += (simsimd_u16_t)a.u8s[0] * (simsimd_u16_t)b.u8s[0],
        sum1 += (simsimd_u16_t)a.u8s[1] * (simsimd_u16_t)b.u8s[1];
    sum0 += (simsimd_u16_t)a.u8s[2] * (simsimd_u16_t)b.u8s[2],
        sum1 += (simsimd_u16_t)a.u8s[3] * (simsimd_u16_t)b.u8s[3];
    sum0 += (simsimd_u16_t)a.u8s[4] * (simsimd_u16_t)b.u8s[4],
        sum1 += (simsimd_u16_t)a.u8s[5] * (simsimd_u16_t)b.u8s[5];
    sum0 += (simsimd_u16_t)a.u8s[6] * (simsimd_u16_t)b.u8s[6],
        sum1 += (simsimd_u16_t)a.u8s[7] * (simsimd_u16_t)b.u8s[7];
    sum0 += (simsimd_u16_t)a.u8s[8] * (simsimd_u16_t)b.u8s[8],
        sum1 += (simsimd_u16_t)a.u8s[9] * (simsimd_u16_t)b.u8s[9];
    sum0 += (simsimd_u16_t)a.u8s[10] * (simsimd_u16_t)b.u8s[10],
        sum1 += (simsimd_u16_t)a.u8s[11] * (simsimd_u16_t)b.u8s[11];
    sum0 += (simsimd_u16_t)a.u8s[12] * (simsimd_u16_t)b.u8s[12],
        sum1 += (simsimd_u16_t)a.u8s[13] * (simsimd_u16_t)b.u8s[13];
    sum0 += (simsimd_u16_t)a.u8s[14] * (simsimd_u16_t)b.u8s[14],
        sum1 += (simsimd_u16_t)a.u8s[15] * (simsimd_u16_t)b.u8s[15];
    sum0 += (simsimd_u16_t)a.u8s[16] * (simsimd_u16_t)b.u8s[16],
        sum1 += (simsimd_u16_t)a.u8s[17] * (simsimd_u16_t)b.u8s[17];
    sum0 += (simsimd_u16_t)a.u8s[18] * (simsimd_u16_t)b.u8s[18],
        sum1 += (simsimd_u16_t)a.u8s[19] * (simsimd_u16_t)b.u8s[19];
    sum0 += (simsimd_u16_t)a.u8s[20] * (simsimd_u16_t)b.u8s[20],
        sum1 += (simsimd_u16_t)a.u8s[21] * (simsimd_u16_t)b.u8s[21];
    sum0 += (simsimd_u16_t)a.u8s[22] * (simsimd_u16_t)b.u8s[22],
        sum1 += (simsimd_u16_t)a.u8s[23] * (simsimd_u16_t)b.u8s[23];
    sum0 += (simsimd_u16_t)a.u8s[24] * (simsimd_u16_t)b.u8s[24],
        sum1 += (simsimd_u16_t)a.u8s[25] * (simsimd_u16_t)b.u8s[25];
    sum0 += (simsimd_u16_t)a.u8s[26] * (simsimd_u16_t)b.u8s[26],
        sum1 += (simsimd_u16_t)a.u8s[27] * (simsimd_u16_t)b.u8s[27];
    sum0 += (simsimd_u16_t)a.u8s[28] * (simsimd_u16_t)b.u8s[28],
        sum1 += (simsimd_u16_t)a.u8s[29] * (simsimd_u16_t)b.u8s[29];
    sum0 += (simsimd_u16_t)a.u8s[30] * (simsimd_u16_t)b.u8s[30],
        sum1 += (simsimd_u16_t)a.u8s[31] * (simsimd_u16_t)b.u8s[31];
    sum0 += (simsimd_u16_t)a.u8s[32] * (simsimd_u16_t)b.u8s[32],
        sum1 += (simsimd_u16_t)a.u8s[33] * (simsimd_u16_t)b.u8s[33];
    sum0 += (simsimd_u16_t)a.u8s[34] * (simsimd_u16_t)b.u8s[34],
        sum1 += (simsimd_u16_t)a.u8s[35] * (simsimd_u16_t)b.u8s[35];
    sum0 += (simsimd_u16_t)a.u8s[36] * (simsimd_u16_t)b.u8s[36],
        sum1 += (simsimd_u16_t)a.u8s[37] * (simsimd_u16_t)b.u8s[37];
    sum0 += (simsimd_u16_t)a.u8s[38] * (simsimd_u16_t)b.u8s[38],
        sum1 += (simsimd_u16_t)a.u8s[39] * (simsimd_u16_t)b.u8s[39];
    sum0 += (simsimd_u16_t)a.u8s[40] * (simsimd_u16_t)b.u8s[40],
        sum1 += (simsimd_u16_t)a.u8s[41] * (simsimd_u16_t)b.u8s[41];
    sum0 += (simsimd_u16_t)a.u8s[42] * (simsimd_u16_t)b.u8s[42],
        sum1 += (simsimd_u16_t)a.u8s[43] * (simsimd_u16_t)b.u8s[43];
    sum0 += (simsimd_u16_t)a.u8s[44] * (simsimd_u16_t)b.u8s[44],
        sum1 += (simsimd_u16_t)a.u8s[45] * (simsimd_u16_t)b.u8s[45];
    sum0 += (simsimd_u16_t)a.u8s[46] * (simsimd_u16_t)b.u8s[46],
        sum1 += (simsimd_u16_t)a.u8s[47] * (simsimd_u16_t)b.u8s[47];
    sum0 += (simsimd_u16_t)a.u8s[48] * (simsimd_u16_t)b.u8s[48],
        sum1 += (simsimd_u16_t)a.u8s[49] * (simsimd_u16_t)b.u8s[49];
    sum0 += (simsimd_u16_t)a.u8s[50] * (simsimd_u16_t)b.u8s[50],
        sum1 += (simsimd_u16_t)a.u8s[51] * (simsimd_u16_t)b.u8s[51];
    sum0 += (simsimd_u16_t)a.u8s[52] * (simsimd_u16_t)b.u8s[52],
        sum1 += (simsimd_u16_t)a.u8s[53] * (simsimd_u16_t)b.u8s[53];
    sum0 += (simsimd_u16_t)a.u8s[54] * (simsimd_u16_t)b.u8s[54],
        sum1 += (simsimd_u16_t)a.u8s[55] * (simsimd_u16_t)b.u8s[55];
    sum0 += (simsimd_u16_t)a.u8s[56] * (simsimd_u16_t)b.u8s[56],
        sum1 += (simsimd_u16_t)a.u8s[57] * (simsimd_u16_t)b.u8s[57];
    sum0 += (simsimd_u16_t)a.u8s[58] * (simsimd_u16_t)b.u8s[58],
        sum1 += (simsimd_u16_t)a.u8s[59] * (simsimd_u16_t)b.u8s[59];
    sum0 += (simsimd_u16_t)a.u8s[60] * (simsimd_u16_t)b.u8s[60],
        sum1 += (simsimd_u16_t)a.u8s[61] * (simsimd_u16_t)b.u8s[61];
    sum0 += (simsimd_u16_t)a.u8s[62] * (simsimd_u16_t)b.u8s[62],
        sum1 += (simsimd_u16_t)a.u8s[63] * (simsimd_u16_t)b.u8s[63];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_serial(simsimd_dot_u8x64_state_serial_t const *state,
                                                        simsimd_distance_t *result) {
    simsimd_u64_t sum = 0;
    for (simsimd_size_t i = 0; i != 2; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over e4m3 scalars.
 */
typedef struct simsimd_dot_e4m3x64_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_e4m3x64_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_serial(simsimd_dot_e4m3x64_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_serial(simsimd_dot_e4m3x64_state_serial_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];
    simsimd_f32_t ai0, ai1, ai2, ai3;
    simsimd_f32_t bi0, bi1, bi2, bi3;
    for (simsimd_size_t i = 0; i != 64; i += 4) {
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

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_serial(simsimd_dot_e4m3x64_state_serial_t const *state,
                                                          simsimd_distance_t *result) {
    simsimd_f32_t sum = 0;
    for (simsimd_size_t i = 0; i != 4; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

/**
 *  @brief Running state for 512-bit dot accumulation over e5m2 scalars.
 */
typedef struct simsimd_dot_e5m2x64_state_serial_t {
    simsimd_f32_t sums[4];
} simsimd_dot_e5m2x64_state_serial_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_serial(simsimd_dot_e5m2x64_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_serial(simsimd_dot_e5m2x64_state_serial_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    simsimd_f32_t sum0 = state->sums[0];
    simsimd_f32_t sum1 = state->sums[1];
    simsimd_f32_t sum2 = state->sums[2];
    simsimd_f32_t sum3 = state->sums[3];
    simsimd_f32_t ai0, ai1, ai2, ai3;
    simsimd_f32_t bi0, bi1, bi2, bi3;
    for (simsimd_size_t i = 0; i != 64; i += 4) {
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

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_serial(simsimd_dot_e5m2x64_state_serial_t const *state,
                                                          simsimd_distance_t *result) {
    simsimd_f32_t sum = 0;
    for (simsimd_size_t i = 0; i != 4; ++i) sum += state->sums[i];
    *result = (simsimd_distance_t)sum;
}

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

SIMSIMD_INTERNAL float32x4_t _simsimd_partial_load_f32x4_neon(simsimd_f32_t const *x, simsimd_size_t n) {
    union {
        float32x4_t vec;
        simsimd_f32_t scalars[4];
    } result;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = x[i];
    for (; i < 4; ++i) result.scalars[i] = 0;
    return result.vec;
}

SIMSIMD_PUBLIC void simsimd_dot_f32_neon(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                         simsimd_size_t count_scalars, simsimd_distance_t *result) {
    float32x4_t ab_vec = vdupq_n_f32(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        float32x4_t a_vec = vld1q_f32(a_scalars + idx_scalars);
        float32x4_t b_vec = vld1q_f32(b_scalars + idx_scalars);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    }
    simsimd_f32_t ab = vaddvq_f32(ab_vec);
    for (; idx_scalars < count_scalars; ++idx_scalars) ab += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_neon(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_distance_t *results) {
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_vec = vld2q_f32((simsimd_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_vec = vld2q_f32((simsimd_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_vec = a_vec.val[0];
        float32x4_t a_imag_vec = a_vec.val[1];
        float32x4_t b_real_vec = b_vec.val[0];
        float32x4_t b_imag_vec = b_vec.val[1];

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmsq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_imag_vec, b_real_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = vaddvq_f32(ab_real_vec);
    simsimd_f32_t ab_imag = vaddvq_f32(ab_imag_vec);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        ab_real += ar * br - ai * bi;
        ab_imag += ar * bi + ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_neon(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_distance_t *results) {
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);
    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_vec = vld2q_f32((simsimd_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_vec = vld2q_f32((simsimd_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_vec = a_vec.val[0];
        float32x4_t a_imag_vec = a_vec.val[1];
        float32x4_t b_real_vec = b_vec.val[0];
        float32x4_t b_imag_vec = b_vec.val[1];

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmaq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmsq_f32(ab_imag_vec, a_imag_vec, b_real_vec);
    }

    // Reduce horizontal sums:
    simsimd_f32_t ab_real = vaddvq_f32(ab_real_vec);
    simsimd_f32_t ab_imag = vaddvq_f32(ab_imag_vec);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        ab_real += ar * br + ai * bi;
        ab_imag += ar * bi - ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

/**
 *  @brief Running state for 512-bit dot accumulation over f32 scalars on NEON.
 */
typedef struct simsimd_dot_f32x16_state_neon_t {
    float32x4_t sum;
} simsimd_dot_f32x16_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_neon(simsimd_dot_f32x16_state_neon_t *state) {
    state->sum = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_neon(simsimd_dot_f32x16_state_neon_t *state, simsimd_b512_vec_t a,
                                                     simsimd_b512_vec_t b) {
    float32x4_t sum = state->sum;
    sum = vfmaq_f32(sum, vreinterpretq_f32_u32(a.u32x4s[0]), vreinterpretq_f32_u32(b.u32x4s[0]));
    sum = vfmaq_f32(sum, vreinterpretq_f32_u32(a.u32x4s[1]), vreinterpretq_f32_u32(b.u32x4s[1]));
    sum = vfmaq_f32(sum, vreinterpretq_f32_u32(a.u32x4s[2]), vreinterpretq_f32_u32(b.u32x4s[2]));
    sum = vfmaq_f32(sum, vreinterpretq_f32_u32(a.u32x4s[3]), vreinterpretq_f32_u32(b.u32x4s[3]));
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_neon(simsimd_dot_f32x16_state_neon_t const *state,
                                                       simsimd_distance_t *result) {
    *result = vaddvq_f32(state->sum);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_i8_neon(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_distance_t *result) {

    int32x4_t ab_vec = vdupq_n_s32(0);
    simsimd_size_t idx_scalars = 0;

    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    // for (simsimd_size_t idx_scalars = 0; idx_scalars != n; idx_scalars += 8) {
    //     int16x8_t a_vec = vmovl_s8(vld1_s8(a_scalars + idx_scalars));
    //     int16x8_t b_vec = vmovl_s8(vld1_s8(b_scalars + idx_scalars));
    //     int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //     ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                          vmovl_s16(vget_low_s16(ab_part_vec))));
    // }
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        int8x16_t a_vec = vld1q_s8(a_scalars + idx_scalars);
        int8x16_t b_vec = vld1q_s8(b_scalars + idx_scalars);
        ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    }

    // Take care of the tail:
    simsimd_i32_t ab = vaddvq_s32(ab_vec);
    for (; idx_scalars < count_scalars; ++idx_scalars) {
        simsimd_i32_t ai = a_scalars[idx_scalars], bi = b_scalars[idx_scalars];
        ab += ai * bi;
    }

    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_u8_neon(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                        simsimd_size_t count_scalars, simsimd_distance_t *result) {

    uint32x4_t ab_vec = vdupq_n_u32(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        uint8x16_t a_vec = vld1q_u8(a_scalars + idx_scalars);
        uint8x16_t b_vec = vld1q_u8(b_scalars + idx_scalars);
        ab_vec = vdotq_u32(ab_vec, a_vec, b_vec);
    }

    // Take care of the tail:
    simsimd_u32_t ab = vaddvq_u32(ab_vec);
    for (; idx_scalars < count_scalars; ++idx_scalars) {
        simsimd_u32_t ai = a_scalars[idx_scalars], bi = b_scalars[idx_scalars];
        ab += ai * bi;
    }

    *result = ab;
}

/**
 *  @brief Running state for 512-bit dot accumulation over i8 scalars on NEON.
 */
typedef struct simsimd_dot_i8x64_state_neon_t {
    int32x4_t sum;
} simsimd_dot_i8x64_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_neon(simsimd_dot_i8x64_state_neon_t *state) {
    state->sum = vdupq_n_s32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_neon(simsimd_dot_i8x64_state_neon_t *state, simsimd_b512_vec_t a,
                                                    simsimd_b512_vec_t b) {
    int32x4_t sum = state->sum;
    sum = vdotq_s32(sum, vld1q_s8(a.i8s + 0), vld1q_s8(b.i8s + 0));
    sum = vdotq_s32(sum, vld1q_s8(a.i8s + 16), vld1q_s8(b.i8s + 16));
    sum = vdotq_s32(sum, vld1q_s8(a.i8s + 32), vld1q_s8(b.i8s + 32));
    sum = vdotq_s32(sum, vld1q_s8(a.i8s + 48), vld1q_s8(b.i8s + 48));
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_neon(simsimd_dot_i8x64_state_neon_t const *state,
                                                      simsimd_distance_t *result) {
    *result = vaddvq_s32(state->sum);
}

/**
 *  @brief Running state for 512-bit dot accumulation over u8 scalars on NEON.
 */
typedef struct simsimd_dot_u8x64_state_neon_t {
    uint32x4_t sum;
} simsimd_dot_u8x64_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_neon(simsimd_dot_u8x64_state_neon_t *state) {
    state->sum = vdupq_n_u32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_neon(simsimd_dot_u8x64_state_neon_t *state, simsimd_b512_vec_t a,
                                                    simsimd_b512_vec_t b) {
    uint32x4_t sum = state->sum;
    sum = vdotq_u32(sum, vld1q_u8(a.u8s + 0), vld1q_u8(b.u8s + 0));
    sum = vdotq_u32(sum, vld1q_u8(a.u8s + 16), vld1q_u8(b.u8s + 16));
    sum = vdotq_u32(sum, vld1q_u8(a.u8s + 32), vld1q_u8(b.u8s + 32));
    sum = vdotq_u32(sum, vld1q_u8(a.u8s + 48), vld1q_u8(b.u8s + 48));
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_neon(simsimd_dot_u8x64_state_neon_t const *state,
                                                      simsimd_distance_t *result) {
    *result = vaddvq_u32(state->sum);
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
    union {
        float16x4_t vec;
        simsimd_f16_t scalars[4];
    } result;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = x[i];
    for (; i < 4; ++i) result.scalars[i] = 0;
    return result.vec;
}

SIMSIMD_PUBLIC void simsimd_dot_f16_neon(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                         simsimd_size_t count_scalars, simsimd_distance_t *result) {
    float32x4_t a_vec, b_vec;
    float32x4_t ab_vec = vdupq_n_f32(0);
    simsimd_size_t i = 0;

simsimd_dot_f16_neon_cycle:
    if (count_scalars < 4) {
        a_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a_scalars, count_scalars));
        b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b_scalars, count_scalars));
        count_scalars = 0;
    }
    else {
        a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a_scalars));
        b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b_scalars));
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
    if (count_scalars) goto simsimd_dot_f16_neon_cycle;
    *result = vaddvq_f32(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_neon(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                          simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // A nicer approach is to use `f16` arithmetic for the dot product, but that requires
    // FMLA extensions available on Arm v8.3 and later. That we can also process 16 entries
    // at once. That's how the original implementation worked, but compiling it was a nightmare :)
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);

    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the  data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_vec = vld2_s16((short *)a_pairs);
        int16x4x2_t b_vec = vld2_s16((short *)b_pairs);
        float32x4_t a_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[0]));
        float32x4_t a_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[1]));
        float32x4_t b_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[0]));
        float32x4_t b_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[1]));

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmsq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_imag_vec, b_real_vec);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_dot_f16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += vaddvq_f32(ab_real_vec);
    results[1] += vaddvq_f32(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_neon(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // A nicer approach is to use `f16` arithmetic for the dot product, but that requires
    // FMLA extensions available on Arm v8.3 and later. That we can also process 16 entries
    // at once. That's how the original implementation worked, but compiling it was a nightmare :)
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);

    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the  data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_vec = vld2_s16((short *)a_pairs);
        int16x4x2_t b_vec = vld2_s16((short *)b_pairs);
        float32x4_t a_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[0]));
        float32x4_t a_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(a_vec.val[1]));
        float32x4_t b_real_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[0]));
        float32x4_t b_imag_vec = vcvt_f32_f16(vreinterpret_f16_s16(b_vec.val[1]));

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmaq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmsq_f32(ab_imag_vec, a_imag_vec, b_real_vec);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += vaddvq_f32(ab_real_vec);
    results[1] += vaddvq_f32(ab_imag_vec);
}

/**
 *  @brief Running state for 512-bit dot accumulation over f16 scalars on NEON.
 */
typedef struct simsimd_dot_f16x32_state_neon_t {
    float32x4_t sum;
} simsimd_dot_f16x32_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_neon(simsimd_dot_f16x32_state_neon_t *state) {
    state->sum = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_neon(simsimd_dot_f16x32_state_neon_t *state, simsimd_b512_vec_t a,
                                                     simsimd_b512_vec_t b) {
    float32x4_t sum = state->sum;
    simsimd_f16_t const *a_scalars = a.f16s;
    simsimd_f16_t const *b_scalars = b.f16s;
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 0))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 0))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 4))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 4))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 8))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 8))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 12))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 12))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 16))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 16))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 20))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 20))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 24))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 24))));
    sum = vfmaq_f32(sum, vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(a_scalars + 28))),
                    vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)(b_scalars + 28))));
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_neon(simsimd_dot_f16x32_state_neon_t const *state,
                                                       simsimd_distance_t *result) {
    *result = vaddvq_f32(state->sum);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_INTERNAL bfloat16x8_t _simsimd_partial_load_bf16x8_neon(simsimd_bf16_t const *x, simsimd_size_t n) {
    union {
        bfloat16x8_t vec;
        simsimd_bf16_t scalars[8];
    } result;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = x[i];
    for (; i < 8; ++i) result.scalars[i] = 0;
    return result.vec;
}

SIMSIMD_PUBLIC void simsimd_dot_bf16_neon(simsimd_bf16_t const *a_scalars, simsimd_bf16_t const *b_scalars,
                                          simsimd_size_t count_scalars, simsimd_distance_t *result) {
    bfloat16x8_t a_vec, b_vec;
    float32x4_t ab_vec = vdupq_n_f32(0);

simsimd_dot_bf16_neon_cycle:
    if (count_scalars < 8) {
        a_vec = _simsimd_partial_load_bf16x8_neon(a_scalars, count_scalars);
        b_vec = _simsimd_partial_load_bf16x8_neon(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)a_scalars);
        b_vec = vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    ab_vec = vbfdotq_f32(ab_vec, a_vec, b_vec);
    if (count_scalars) goto simsimd_dot_bf16_neon_cycle;

    *result = vaddvq_f32(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_bf16c_neon(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                           simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // A nicer approach is to use `bf16` arithmetic for the dot product, but that requires
    // FMLA extensions available on Arm v8.3 and later. That we can also process 16 entries
    // at once. That's how the original implementation worked, but compiling it was a nightmare :)
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);

    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the  data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_vec = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_vec = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(a_vec.val[0]));
        float32x4_t a_imag_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(a_vec.val[1]));
        float32x4_t b_real_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(b_vec.val[0]));
        float32x4_t b_imag_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(b_vec.val[1]));

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmsq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_imag_vec, b_real_vec);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_dot_bf16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += vaddvq_f32(ab_real_vec);
    results[1] += vaddvq_f32(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_bf16c_neon(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                            simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // A nicer approach is to use `bf16` arithmetic for the dot product, but that requires
    // FMLA extensions available on Arm v8.3 and later. That we can also process 16 entries
    // at once. That's how the original implementation worked, but compiling it was a nightmare :)
    float32x4_t ab_real_vec = vdupq_n_f32(0);
    float32x4_t ab_imag_vec = vdupq_n_f32(0);

    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the  data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_vec = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_vec = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(a_vec.val[0]));
        float32x4_t a_imag_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(a_vec.val[1]));
        float32x4_t b_real_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(b_vec.val[0]));
        float32x4_t b_imag_vec = vcvt_f32_bf16(vreinterpret_bf16_s16(b_vec.val[1]));

        // Compute the dot product:
        ab_real_vec = vfmaq_f32(ab_real_vec, a_real_vec, b_real_vec);
        ab_real_vec = vfmaq_f32(ab_real_vec, a_imag_vec, b_imag_vec);
        ab_imag_vec = vfmaq_f32(ab_imag_vec, a_real_vec, b_imag_vec);
        ab_imag_vec = vfmsq_f32(ab_imag_vec, a_imag_vec, b_real_vec);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_vdot_bf16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += vaddvq_f32(ab_real_vec);
    results[1] += vaddvq_f32(ab_imag_vec);
}

/**
 *  @brief Running state for 512-bit dot accumulation over bf16 scalars on NEON.
 */
typedef struct simsimd_dot_bf16x32_state_neon_t {
    float32x4_t sum;
} simsimd_dot_bf16x32_state_neon_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_neon(simsimd_dot_bf16x32_state_neon_t *state) {
    state->sum = vdupq_n_f32(0);
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_neon(simsimd_dot_bf16x32_state_neon_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    float32x4_t sum = state->sum;
    simsimd_bf16_t const *a_scalars = a.bf16s;
    simsimd_bf16_t const *b_scalars = b.bf16s;
    sum = vbfdotq_f32(sum, vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(a_scalars + 0)),
                      vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(b_scalars + 0)));
    sum = vbfdotq_f32(sum, vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(a_scalars + 8)),
                      vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(b_scalars + 8)));
    sum = vbfdotq_f32(sum, vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(a_scalars + 16)),
                      vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(b_scalars + 16)));
    sum = vbfdotq_f32(sum, vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(a_scalars + 24)),
                      vld1q_bf16((simsimd_bf16_for_arm_simd_t const *)(b_scalars + 24)));
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_neon(simsimd_dot_bf16x32_state_neon_t const *state,
                                                        simsimd_distance_t *result) {
    *result = vaddvq_f32(state->sum);
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

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_f64x4_haswell(__m256d vec) {
    // Reduce the double-precision vector to a scalar
    // Horizontal add the first and second double-precision values, and third and fourth
    __m128d vec_low = _mm256_castpd256_pd128(vec);
    __m128d vec_high = _mm256_extractf128_pd(vec, 1);
    __m128d vec128 = _mm_add_pd(vec_low, vec_high);

    // Horizontal add again to accumulate all four values into one
    vec128 = _mm_hadd_pd(vec128, vec128);

    // Convert the final sum to a scalar double-precision value and return
    return _mm_cvtsd_f64(vec128);
}

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_f32x8_haswell(__m256 vec) {
    // Convert the lower and higher 128-bit lanes of the input vector to double precision
    __m128 low_f32 = _mm256_castps256_ps128(vec);
    __m128 high_f32 = _mm256_extractf128_ps(vec, 1);

    // Convert single-precision (float) vectors to double-precision (double) vectors
    __m256d low_f64 = _mm256_cvtps_pd(low_f32);
    __m256d high_f64 = _mm256_cvtps_pd(high_f32);

    // Perform the addition in double-precision
    __m256d sum = _mm256_add_pd(low_f64, high_f64);
    return _simsimd_reduce_f64x4_haswell(sum);
}

SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_i32x8_haswell(__m256i vec) {
    __m128i low = _mm256_castsi256_si128(vec);
    __m128i high = _mm256_extracti128_si256(vec, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

SIMSIMD_PUBLIC void simsimd_dot_f32_haswell(simsimd_f32_t const *a_scalars, simsimd_f32_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_distance_t *results) {

    __m256 ab_vec = _mm256_setzero_ps();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count_scalars; idx_scalars += 8) {
        __m256 a_vec = _mm256_loadu_ps(a_scalars + idx_scalars);
        __m256 b_vec = _mm256_loadu_ps(b_scalars + idx_scalars);
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    }
    simsimd_f64_t ab = _simsimd_reduce_f32x8_haswell(ab_vec);
    for (; idx_scalars < count_scalars; ++idx_scalars) ab += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *results = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_haswell(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // The naive approach would be to use FMA and FMS instructions on different parts of the vectors.
    // Prior to that we would need to shuffle the input vectors to separate real and imaginary parts.
    // Both operations are quite expensive, and the resulting kernel would run at 2.5 GB/s.
    // __m128 ab_real_vec = _mm_setzero_ps();
    // __m128 ab_imag_vec = _mm_setzero_ps();
    // __m256i permute_vec = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    // simsimd_size_t idx_pairs = 0;
    // for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
    //     __m256 a_vec = _mm256_loadu_ps((simsimd_f32_t const *)(a_pairs + idx_pairs));
    //     __m256 b_vec = _mm256_loadu_ps((simsimd_f32_t const *)(b_pairs + idx_pairs));
    //     __m256 a_shuffled = _mm256_permutevar8x32_ps(a_vec, permute_vec);
    //     __m256 b_shuffled = _mm256_permutevar8x32_ps(b_vec, permute_vec);
    //     __m128 a_real_vec = _mm256_extractf128_ps(a_shuffled, 0);
    //     __m128 a_imag_vec = _mm256_extractf128_ps(a_shuffled, 1);
    //     __m128 b_real_vec = _mm256_extractf128_ps(b_shuffled, 0);
    //     __m128 b_imag_vec = _mm256_extractf128_ps(b_shuffled, 1);
    //     ab_real_vec = _mm_fmadd_ps(a_real_vec, b_real_vec, ab_real_vec);
    //     ab_real_vec = _mm_fnmadd_ps(a_imag_vec, b_imag_vec, ab_real_vec);
    //     ab_imag_vec = _mm_fmadd_ps(a_real_vec, b_imag_vec, ab_imag_vec);
    //     ab_imag_vec = _mm_fmadd_ps(a_imag_vec, b_real_vec, ab_imag_vec);
    // }
    //
    // Instead, we take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors. Moreover, `XOR` can be placed after the primary loop.
    // Both operations are quite cheap, and the throughput doubles from 2.5 GB/s to 5 GB/s.
    __m256 ab_real_vec = _mm256_setzero_ps();
    __m256 ab_imag_vec = _mm256_setzero_ps();
    __m256i sign_flip_vec = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_vec = _mm256_set_epi8( //
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4,                              // Points to the second f32 in 128-bit lane
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4                               // Points to the second f32 in 128-bit lane
    );

    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_vec = _mm256_loadu_ps((simsimd_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_vec = _mm256_loadu_ps((simsimd_f32_t const *)(b_pairs + idx_pairs));
        __m256 b_swapped_vec = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_vec), swap_adjacent_vec));
        ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);
        ab_imag_vec = _mm256_fmadd_ps(a_vec, b_swapped_vec, ab_imag_vec);
    }

    // Flip the sign bit in every second scalar before accumulation:
    ab_real_vec = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(ab_real_vec), sign_flip_vec));

    // Reduce horizontal sums:
    simsimd_distance_t ab_real = _simsimd_reduce_f32x8_haswell(ab_real_vec);
    simsimd_distance_t ab_imag = _simsimd_reduce_f32x8_haswell(ab_imag_vec);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        ab_real += ar * br - ai * bi;
        ab_imag += ar * bi + ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_haswell(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_distance_t *results) {

    __m256 ab_real_vec = _mm256_setzero_ps();
    __m256 ab_imag_vec = _mm256_setzero_ps();
    __m256i sign_flip_vec = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_vec = _mm256_set_epi8( //
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4,                              // Points to the second f32 in 128-bit lane
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4                               // Points to the second f32 in 128-bit lane
    );

    simsimd_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_vec = _mm256_loadu_ps((simsimd_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_vec = _mm256_loadu_ps((simsimd_f32_t const *)(b_pairs + idx_pairs));
        ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);
        b_vec = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_vec), swap_adjacent_vec));
        ab_imag_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_imag_vec);
    }

    // Flip the sign bit in every second scalar before accumulation:
    ab_imag_vec = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(ab_imag_vec), sign_flip_vec));

    // Reduce horizontal sums:
    simsimd_distance_t ab_real = _simsimd_reduce_f32x8_haswell(ab_real_vec);
    simsimd_distance_t ab_imag = _simsimd_reduce_f32x8_haswell(ab_imag_vec);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        simsimd_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        simsimd_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        ab_real += ar * br + ai * bi;
        ab_imag += ar * bi - ai * br;
    }
    results[0] = ab_real;
    results[1] = ab_imag;
}

SIMSIMD_INTERNAL __m256 _simsimd_partial_load_f16x8_haswell(simsimd_f16_t const *a, simsimd_size_t n) {
    // In case the software emulation for `f16` scalars is enabled, the `simsimd_f16_to_f32`
    // function will run. It is extremely slow, so even for the tail, let's combine serial
    // loads and stores with vectorized math.
    union {
        __m128i vec;
        simsimd_f16_t scalars[8];
    } result;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = a[i];
    for (; i < 8; ++i) result.scalars[i] = 0;
    return _mm256_cvtph_ps(result.vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f16_haswell(simsimd_f16_t const *a_scalars, simsimd_f16_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m256 a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps();

simsimd_dot_f16_haswell_cycle:
    if (count_scalars < 8) {
        a_vec = _simsimd_partial_load_f16x8_haswell(a_scalars, count_scalars);
        b_vec = _simsimd_partial_load_f16x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_scalars));
        b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_scalars));
        count_scalars -= 8, a_scalars += 8, b_scalars += 8;
    }
    // We can silence the NaNs using blends:
    //
    //     __m256 a_is_nan = _mm256_cmp_ps(a_vec, a_vec, _CMP_UNORD_Q);
    //     __m256 b_is_nan = _mm256_cmp_ps(b_vec, b_vec, _CMP_UNORD_Q);
    //     ab_vec = _mm256_blendv_ps(_mm256_fmadd_ps(a_vec, b_vec, ab_vec), ab_vec, _mm256_or_ps(a_is_nan, b_is_nan));
    //
    ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    if (count_scalars) goto simsimd_dot_f16_haswell_cycle;

    *result = _simsimd_reduce_f32x8_haswell(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_haswell(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // Ideally the implementation would load 256 bits worth of vector data at a time,
    // shuffle those within a register, split in halfs, and only then upcast.
    // That way, we are stepping through 32x 16-bit vector components at a time, or 16 dimensions.
    // Sadly, shuffling 16-bit entries in a YMM register is hard to implement efficiently.
    //
    // Simpler approach is to load 128 bits at a time, upcast, and then shuffle.
    // This mostly replicates the `simsimd_dot_f32c_haswell`.
    __m256 ab_real_vec = _mm256_setzero_ps();
    __m256 ab_imag_vec = _mm256_setzero_ps();
    __m256i sign_flip_vec = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_vec = _mm256_set_epi8( //
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4,                              // Points to the second f32 in 128-bit lane
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4                               // Points to the second f32 in 128-bit lane
    );

    while (count_pairs >= 4) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        __m256 b_swapped_vec = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_vec), swap_adjacent_vec));
        ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);
        ab_imag_vec = _mm256_fmadd_ps(a_vec, b_swapped_vec, ab_imag_vec);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Flip the sign bit in every second scalar before accumulation:
    ab_real_vec = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(ab_real_vec), sign_flip_vec));

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_dot_f16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += _simsimd_reduce_f32x8_haswell(ab_real_vec);
    results[1] += _simsimd_reduce_f32x8_haswell(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_haswell(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_distance_t *results) {

    // Ideally the implementation would load 256 bits worth of vector data at a time,
    // shuffle those within a register, split in halfs, and only then upcast.
    // That way, we are stepping through 32x 16-bit vector components at a time, or 16 dimensions.
    // Sadly, shuffling 16-bit entries in a YMM register is hard to implement efficiently.
    //
    // Simpler approach is to load 128 bits at a time, upcast, and then shuffle.
    // This mostly replicates the `simsimd_vdot_f32c_haswell`.
    __m256 ab_real_vec = _mm256_setzero_ps();
    __m256 ab_imag_vec = _mm256_setzero_ps();
    __m256i sign_flip_vec = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_vec = _mm256_set_epi8( //
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4,                              // Points to the second f32 in 128-bit lane
        11, 10, 9, 8,                            // Points to the third f32 in 128-bit lane
        15, 14, 13, 12,                          // Points to the fourth f32 in 128-bit lane
        3, 2, 1, 0,                              // Points to the first f32 in 128-bit lane
        7, 6, 5, 4                               // Points to the second f32 in 128-bit lane
    );

    while (count_pairs >= 4) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        ab_real_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_real_vec);
        b_vec = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_vec), swap_adjacent_vec));
        ab_imag_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_imag_vec);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Flip the sign bit in every second scalar before accumulation:
    ab_imag_vec = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(ab_imag_vec), sign_flip_vec));

    // Reduce horizontal sums and aggregate with the tail:
    simsimd_dot_f16c_serial(a_pairs, b_pairs, count_pairs, results);
    results[0] += _simsimd_reduce_f32x8_haswell(ab_real_vec);
    results[1] += _simsimd_reduce_f32x8_haswell(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_i8_haswell(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_distance_t *result) {

    __m256i ab_i32_low_vec = _mm256_setzero_si256();
    __m256i ab_i32_high_vec = _mm256_setzero_si256();

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
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));

        // Upcast `int8` to `int16`
        __m256i a_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 0));
        __m256i a_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 1));
        __m256i b_i16_low_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 0));
        __m256i b_i16_high_vec = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 1));

        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        ab_i32_low_vec = _mm256_add_epi32(ab_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, b_i16_low_vec));
        ab_i32_high_vec = _mm256_add_epi32(ab_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, b_i16_high_vec));
    }

    // Horizontal sum across the 256-bit register
    int ab = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(ab_i32_low_vec, ab_i32_high_vec));

    // Take care of the tail:
    for (; idx_scalars < count_scalars; ++idx_scalars) ab += (int)(a_scalars[idx_scalars]) * b_scalars[idx_scalars];
    *result = ab;
}

SIMSIMD_PUBLIC void simsimd_dot_u8_haswell(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_distance_t *result) {

    __m256i ab_i32_low_vec = _mm256_setzero_si256();
    __m256i ab_i32_high_vec = _mm256_setzero_si256();
    __m256i const zeros_vec = _mm256_setzero_si256();

    // AVX2 has no instructions for unsigned 8-bit integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_u8_vec = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8_vec = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i a_i16_low_vec = _mm256_unpacklo_epi8(a_u8_vec, zeros_vec);
        __m256i a_i16_high_vec = _mm256_unpackhi_epi8(a_u8_vec, zeros_vec);
        __m256i b_i16_low_vec = _mm256_unpacklo_epi8(b_u8_vec, zeros_vec);
        __m256i b_i16_high_vec = _mm256_unpackhi_epi8(b_u8_vec, zeros_vec);

        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        ab_i32_low_vec = _mm256_add_epi32(ab_i32_low_vec, _mm256_madd_epi16(a_i16_low_vec, b_i16_low_vec));
        ab_i32_high_vec = _mm256_add_epi32(ab_i32_high_vec, _mm256_madd_epi16(a_i16_high_vec, b_i16_high_vec));
    }

    // Horizontal sum across the 256-bit register
    int ab = _simsimd_reduce_i32x8_haswell(_mm256_add_epi32(ab_i32_low_vec, ab_i32_high_vec));

    // Take care of the tail:
    for (; idx_scalars < count_scalars; ++idx_scalars) ab += (int)(a_scalars[idx_scalars]) * b_scalars[idx_scalars];
    *result = ab;
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
    union {
        __m128i vec;
        simsimd_bf16_t scalars[8];
    } result;
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = a[i];
    for (; i < 8; ++i) result.scalars[i] = 0;
    return result.vec;
}

SIMSIMD_PUBLIC void simsimd_dot_bf16_haswell(simsimd_bf16_t const *a_scalars, simsimd_bf16_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m128i a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps();

simsimd_dot_bf16_haswell_cycle:
    if (count_scalars < 8) {
        a_vec = _simsimd_partial_load_bf16x8_haswell(a_scalars, count_scalars);
        b_vec = _simsimd_partial_load_bf16x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm_lddqu_si128((__m128i const *)a_scalars);
        b_vec = _mm_lddqu_si128((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    ab_vec = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(a_vec), _simsimd_bf16x8_to_f32x8_haswell(b_vec), ab_vec);
    if (count_scalars) goto simsimd_dot_bf16_haswell_cycle;

    *result = _simsimd_reduce_f32x8_haswell(ab_vec);
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
    union {
        __m128i vec;
        simsimd_u8_t scalars[16];
    } result;
    result.vec = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = a[i];
    return result.vec;
}

SIMSIMD_INTERNAL __m128i _simsimd_partial_load_e5m2x8_haswell(simsimd_e5m2_t const *a, simsimd_size_t n) {
    union {
        __m128i vec;
        simsimd_u8_t scalars[16];
    } result;
    result.vec = _mm_setzero_si128();
    simsimd_size_t i = 0;
    for (; i < n; ++i) result.scalars[i] = a[i];
    return result.vec;
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_haswell(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m128i a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps();

simsimd_dot_e4m3_haswell_cycle:
    if (count_scalars < 8) {
        a_vec = _simsimd_partial_load_e4m3x8_haswell(a_scalars, count_scalars);
        b_vec = _simsimd_partial_load_e4m3x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm_loadl_epi64((__m128i const *)a_scalars);
        b_vec = _mm_loadl_epi64((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    ab_vec = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(a_vec), _simsimd_e4m3x8_to_f32x8_haswell(b_vec), ab_vec);
    if (count_scalars) goto simsimd_dot_e4m3_haswell_cycle;

    *result = _simsimd_reduce_f32x8_haswell(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_haswell(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m128i a_vec, b_vec;
    __m256 ab_vec = _mm256_setzero_ps();

simsimd_dot_e5m2_haswell_cycle:
    if (count_scalars < 8) {
        a_vec = _simsimd_partial_load_e5m2x8_haswell(a_scalars, count_scalars);
        b_vec = _simsimd_partial_load_e5m2x8_haswell(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm_loadl_epi64((__m128i const *)a_scalars);
        b_vec = _mm_loadl_epi64((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    ab_vec = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(a_vec), _simsimd_e5m2x8_to_f32x8_haswell(b_vec), ab_vec);
    if (count_scalars) goto simsimd_dot_e5m2_haswell_cycle;

    *result = _simsimd_reduce_f32x8_haswell(ab_vec);
}

/**
 *  @brief Running state for 16-element dot accumulation over f32 scalars on Haswell.
 */
typedef struct simsimd_dot_f32x16_state_haswell_t {
    __m256 sum;
} simsimd_dot_f32x16_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_haswell(simsimd_dot_f32x16_state_haswell_t *state) {
    state->sum = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_haswell(simsimd_dot_f32x16_state_haswell_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    __m256 sum = state->sum;
    __m256 a0 = _mm256_loadu_ps(a.f32s + 0);
    __m256 b0 = _mm256_loadu_ps(b.f32s + 0);
    __m256 a1 = _mm256_loadu_ps(a.f32s + 8);
    __m256 b1 = _mm256_loadu_ps(b.f32s + 8);
    sum = _mm256_fmadd_ps(a0, b0, sum);
    sum = _mm256_fmadd_ps(a1, b1, sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_haswell(simsimd_dot_f32x16_state_haswell_t const *state,
                                                          simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x8_haswell(state->sum);
}

/**
 *  @brief Running state for 32-element dot accumulation over f16 scalars on Haswell.
 */
typedef struct simsimd_dot_f16x32_state_haswell_t {
    __m256 sum;
} simsimd_dot_f16x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_haswell(simsimd_dot_f16x32_state_haswell_t *state) {
    state->sum = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_haswell(simsimd_dot_f16x32_state_haswell_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    __m256 sum = state->sum;
    sum = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 0))),
                          _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 0))), sum);
    sum = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 8))),
                          _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 8))), sum);
    sum = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 16))),
                          _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 16))), sum);
    sum = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 24))),
                          _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 24))), sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_haswell(simsimd_dot_f16x32_state_haswell_t const *state,
                                                          simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x8_haswell(state->sum);
}

/**
 *  @brief Running state for 32-element dot accumulation over bf16 scalars on Haswell.
 */
typedef struct simsimd_dot_bf16x32_state_haswell_t {
    __m256 sum;
} simsimd_dot_bf16x32_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_haswell(simsimd_dot_bf16x32_state_haswell_t *state) {
    state->sum = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_haswell(simsimd_dot_bf16x32_state_haswell_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m256 sum = state->sum;
    sum = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 0))),
                          _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 0))), sum);
    sum = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 8))),
                          _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 8))), sum);
    sum = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 16))),
                          _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 16))), sum);
    sum = _mm256_fmadd_ps(_simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(a.bf16s + 24))),
                          _simsimd_bf16x8_to_f32x8_haswell(_mm_lddqu_si128((__m128i const *)(b.bf16s + 24))), sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_haswell(simsimd_dot_bf16x32_state_haswell_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x8_haswell(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e4m3 scalars on Haswell.
 */
typedef struct simsimd_dot_e4m3x64_state_haswell_t {
    __m256 sum;
} simsimd_dot_e4m3x64_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_haswell(simsimd_dot_e4m3x64_state_haswell_t *state) {
    state->sum = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_haswell(simsimd_dot_e4m3x64_state_haswell_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m256 sum = state->sum;
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 0))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 0))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 8))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 8))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 16))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 16))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 24))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 24))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 32))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 32))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 40))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 40))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 48))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 48))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 56))),
                          _simsimd_e4m3x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 56))), sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_haswell(simsimd_dot_e4m3x64_state_haswell_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x8_haswell(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e5m2 scalars on Haswell.
 */
typedef struct simsimd_dot_e5m2x64_state_haswell_t {
    __m256 sum;
} simsimd_dot_e5m2x64_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_haswell(simsimd_dot_e5m2x64_state_haswell_t *state) {
    state->sum = _mm256_setzero_ps();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_haswell(simsimd_dot_e5m2x64_state_haswell_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m256 sum = state->sum;
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 0))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 0))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 8))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 8))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 16))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 16))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 24))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 24))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 32))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 32))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 40))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 40))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 48))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 48))), sum);
    sum = _mm256_fmadd_ps(_simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 56))),
                          _simsimd_e5m2x8_to_f32x8_haswell(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 56))), sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_haswell(simsimd_dot_e5m2x64_state_haswell_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x8_haswell(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over i8 scalars on Haswell.
 */
typedef struct simsimd_dot_i8x64_state_haswell_t {
    __m256i sum_low;
    __m256i sum_high;
} simsimd_dot_i8x64_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_haswell(simsimd_dot_i8x64_state_haswell_t *state) {
    state->sum_low = _mm256_setzero_si256();
    state->sum_high = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_haswell(simsimd_dot_i8x64_state_haswell_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m256i sum_low = state->sum_low;
    __m256i sum_high = state->sum_high;

    __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a.i8s + 0));
    __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b.i8s + 0));
    __m256i a_i16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 0));
    __m256i a_i16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 1));
    __m256i b_i16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 0));
    __m256i b_i16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 1));
    sum_low = _mm256_add_epi32(sum_low, _mm256_madd_epi16(a_i16_low, b_i16_low));
    sum_high = _mm256_add_epi32(sum_high, _mm256_madd_epi16(a_i16_high, b_i16_high));

    a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a.i8s + 32));
    b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b.i8s + 32));
    a_i16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 0));
    a_i16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8_vec, 1));
    b_i16_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 0));
    b_i16_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8_vec, 1));
    sum_low = _mm256_add_epi32(sum_low, _mm256_madd_epi16(a_i16_low, b_i16_low));
    sum_high = _mm256_add_epi32(sum_high, _mm256_madd_epi16(a_i16_high, b_i16_high));

    state->sum_low = sum_low;
    state->sum_high = sum_high;
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_haswell(simsimd_dot_i8x64_state_haswell_t const *state,
                                                         simsimd_distance_t *result) {
    __m256i sum = _mm256_add_epi32(state->sum_low, state->sum_high);
    *result = _simsimd_reduce_i32x8_haswell(sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over u8 scalars on Haswell.
 */
typedef struct simsimd_dot_u8x64_state_haswell_t {
    __m256i sum_low;
    __m256i sum_high;
} simsimd_dot_u8x64_state_haswell_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_haswell(simsimd_dot_u8x64_state_haswell_t *state) {
    state->sum_low = _mm256_setzero_si256();
    state->sum_high = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_haswell(simsimd_dot_u8x64_state_haswell_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m256i sum_low = state->sum_low;
    __m256i sum_high = state->sum_high;
    __m256i const zeros_vec = _mm256_setzero_si256();

    __m256i a_u8_vec = _mm256_lddqu_si256((__m256i const *)(a.u8s + 0));
    __m256i b_u8_vec = _mm256_lddqu_si256((__m256i const *)(b.u8s + 0));
    __m256i a_i16_low = _mm256_unpacklo_epi8(a_u8_vec, zeros_vec);
    __m256i a_i16_high = _mm256_unpackhi_epi8(a_u8_vec, zeros_vec);
    __m256i b_i16_low = _mm256_unpacklo_epi8(b_u8_vec, zeros_vec);
    __m256i b_i16_high = _mm256_unpackhi_epi8(b_u8_vec, zeros_vec);
    sum_low = _mm256_add_epi32(sum_low, _mm256_madd_epi16(a_i16_low, b_i16_low));
    sum_high = _mm256_add_epi32(sum_high, _mm256_madd_epi16(a_i16_high, b_i16_high));

    a_u8_vec = _mm256_lddqu_si256((__m256i const *)(a.u8s + 32));
    b_u8_vec = _mm256_lddqu_si256((__m256i const *)(b.u8s + 32));
    a_i16_low = _mm256_unpacklo_epi8(a_u8_vec, zeros_vec);
    a_i16_high = _mm256_unpackhi_epi8(a_u8_vec, zeros_vec);
    b_i16_low = _mm256_unpacklo_epi8(b_u8_vec, zeros_vec);
    b_i16_high = _mm256_unpackhi_epi8(b_u8_vec, zeros_vec);
    sum_low = _mm256_add_epi32(sum_low, _mm256_madd_epi16(a_i16_low, b_i16_low));
    sum_high = _mm256_add_epi32(sum_high, _mm256_madd_epi16(a_i16_high, b_i16_high));

    state->sum_low = sum_low;
    state->sum_high = sum_high;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_haswell(simsimd_dot_u8x64_state_haswell_t const *state,
                                                         simsimd_distance_t *result) {
    __m256i sum = _mm256_add_epi32(state->sum_low, state->sum_high);
    *result = _simsimd_reduce_i32x8_haswell(sum);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_f32x16_skylake(__m512 a) {
    __m512 x = _mm512_add_ps(a, _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(0, 0, 3, 2)));
    __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(_mm_hadd_ps(r, r));
}

/** @brief Native F32 horizontal reduction without F64 conversion, for high-performance matmul. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_f32x16_to_f32_skylake(__m512 a) {
    __m512 x = _mm512_add_ps(a, _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(0, 0, 3, 2)));
    __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(_mm_hadd_ps(r, r));
}

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
                                            simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512 a_vec, b_vec;
    __m512 ab_vec = _mm512_setzero();

simsimd_dot_f32_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_vec = _mm512_maskz_loadu_ps(mask, a_scalars);
        b_vec = _mm512_maskz_loadu_ps(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a_scalars);
        b_vec = _mm512_loadu_ps(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
    if (count_scalars) goto simsimd_dot_f32_skylake_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f64_skylake(simsimd_f64_t const *a_scalars, simsimd_f64_t const *b_scalars,
                                            simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512d a_vec, b_vec;
    __m512d ab_vec = _mm512_setzero_pd();

simsimd_dot_f64_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_vec = _mm512_maskz_loadu_pd(mask, a_scalars);
        b_vec = _mm512_maskz_loadu_pd(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a_scalars);
        b_vec = _mm512_loadu_pd(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    ab_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_vec);
    if (count_scalars) goto simsimd_dot_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f32c_skylake(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512 a_vec, b_vec;
    __m512 ab_real_vec = _mm512_setzero();
    __m512 ab_imag_vec = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi64(0x8000000000000000);
simsimd_dot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_ps(mask, a_pairs);
        b_vec = _mm512_maskz_loadu_ps(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a_pairs);
        b_vec = _mm512_loadu_ps(b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    ab_real_vec = _mm512_fmadd_ps(b_vec, a_vec, ab_real_vec);
    b_vec = _mm512_permute_ps(b_vec, 0xB1); //? Swap adjacent entries within each pair
    ab_imag_vec = _mm512_fmadd_ps(b_vec, a_vec, ab_imag_vec);
    if (count_pairs) goto simsimd_dot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    ab_real_vec = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(ab_real_vec), sign_flip_vec));

    // Reduce horizontal sums:
    results[0] = _simsimd_reduce_f32x16_skylake(ab_real_vec);
    results[1] = _simsimd_reduce_f32x16_skylake(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f32c_skylake(simsimd_f32c_t const *a_pairs, simsimd_f32c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512 a_vec, b_vec;
    __m512 ab_real_vec = _mm512_setzero();
    __m512 ab_imag_vec = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi64(0x8000000000000000);
    __m512i const swap_adjacent_vec = _mm512_set_epi8(                  //
        59, 58, 57, 56, 63, 62, 61, 60, 51, 50, 49, 48, 55, 54, 53, 52, // 4th 128-bit lane
        43, 42, 41, 40, 47, 46, 45, 44, 35, 34, 33, 32, 39, 38, 37, 36, // 3rd 128-bit lane
        27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, // 2nd 128-bit lane
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4            // 1st 128-bit lane
    );
simsimd_vdot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_ps(mask, (simsimd_f32_t const *)a_pairs);
        b_vec = _mm512_maskz_loadu_ps(mask, (simsimd_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_ps((simsimd_f32_t const *)a_pairs);
        b_vec = _mm512_loadu_ps((simsimd_f32_t const *)b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    ab_real_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_real_vec);
    b_vec = _mm512_permute_ps(b_vec, 0xB1); //? Swap adjacent entries within each pair
    ab_imag_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_imag_vec);
    if (count_pairs) goto simsimd_vdot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    ab_imag_vec = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(ab_imag_vec), sign_flip_vec));

    // Reduce horizontal sums:
    results[0] = _simsimd_reduce_f32x16_skylake(ab_real_vec);
    results[1] = _simsimd_reduce_f32x16_skylake(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f64c_skylake(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512d a_vec, b_vec;
    __m512d ab_real_vec = _mm512_setzero_pd();
    __m512d ab_imag_vec = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set_epi64(                                     //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
simsimd_dot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_pd(mask, a_pairs);
        b_vec = _mm512_maskz_loadu_pd(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a_pairs);
        b_vec = _mm512_loadu_pd(b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    ab_real_vec = _mm512_fmadd_pd(b_vec, a_vec, ab_real_vec);
    b_vec = _mm512_permute_pd(b_vec, 0x55); //? Same as 0b01010101.
    ab_imag_vec = _mm512_fmadd_pd(b_vec, a_vec, ab_imag_vec);
    if (count_pairs) goto simsimd_dot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    ab_real_vec = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(ab_real_vec), sign_flip_vec));

    // Reduce horizontal sums:
    results[0] = _mm512_reduce_add_pd(ab_real_vec);
    results[1] = _mm512_reduce_add_pd(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f64c_skylake(simsimd_f64c_t const *a_pairs, simsimd_f64c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512d a_vec, b_vec;
    __m512d ab_real_vec = _mm512_setzero_pd();
    __m512d ab_imag_vec = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set_epi64(                                     //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
simsimd_vdot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_pd(mask, (simsimd_f32_t const *)a_pairs);
        b_vec = _mm512_maskz_loadu_pd(mask, (simsimd_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_pd((simsimd_f32_t const *)a_pairs);
        b_vec = _mm512_loadu_pd((simsimd_f32_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    ab_real_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_real_vec);
    b_vec = _mm512_permute_pd(b_vec, 0x55); //? Same as 0b01010101.
    ab_imag_vec = _mm512_fmadd_pd(a_vec, b_vec, ab_imag_vec);
    if (count_pairs) goto simsimd_vdot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    ab_imag_vec = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(ab_imag_vec), sign_flip_vec));

    // Reduce horizontal sums:
    results[0] = _mm512_reduce_add_pd(ab_real_vec);
    results[1] = _mm512_reduce_add_pd(ab_imag_vec);
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
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m128i a_i8_vec, b_i8_vec;
    __m512 ab_vec = _mm512_setzero_ps();

simsimd_dot_e4m3_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_i8_vec = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm_loadu_si128((__m128i const *)a_scalars);
        b_i8_vec = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32 = _simsimd_e4m3x16_to_f32x16_skylake(a_i8_vec);
    __m512 b_f32 = _simsimd_e4m3x16_to_f32x16_skylake(b_i8_vec);
    ab_vec = _mm512_fmadd_ps(a_f32, b_f32, ab_vec);
    if (count_scalars) goto simsimd_dot_e4m3_skylake_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_skylake(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m128i a_i8_vec, b_i8_vec;
    __m512 ab_vec = _mm512_setzero_ps();

simsimd_dot_e5m2_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_i8_vec = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm_loadu_si128((__m128i const *)a_scalars);
        b_i8_vec = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32 = _simsimd_e5m2x16_to_f32x16_skylake(a_i8_vec);
    __m512 b_f32 = _simsimd_e5m2x16_to_f32x16_skylake(b_i8_vec);
    ab_vec = _mm512_fmadd_ps(a_f32, b_f32, ab_vec);
    if (count_scalars) goto simsimd_dot_e5m2_skylake_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

/**
 *  @brief Running state for 8-element dot accumulation over f64 scalars on Skylake.
 */
typedef struct simsimd_dot_f64x8_state_skylake_t {
    __m512d sum;
} simsimd_dot_f64x8_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_f64x8_init_skylake(simsimd_dot_f64x8_state_skylake_t *state) {
    state->sum = _mm512_setzero_pd();
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_update_skylake(simsimd_dot_f64x8_state_skylake_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512d sum = state->sum;
    __m512d a_vec = _mm512_loadu_pd(a.f64s);
    __m512d b_vec = _mm512_loadu_pd(b.f64s);
    state->sum = _mm512_fmadd_pd(a_vec, b_vec, sum);
}

SIMSIMD_INTERNAL void simsimd_dot_f64x8_finalize_skylake(simsimd_dot_f64x8_state_skylake_t const *state,
                                                         simsimd_distance_t *result) {
    *result = _mm512_reduce_add_pd(state->sum);
}

/**
 *  @brief Running state for 16-element dot accumulation over f32 scalars on Skylake.
 */
typedef struct simsimd_dot_f32x16_state_skylake_t {
    __m512 sum;
} simsimd_dot_f32x16_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_f32x16_init_skylake(simsimd_dot_f32x16_state_skylake_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_update_skylake(simsimd_dot_f32x16_state_skylake_t *state, simsimd_b512_vec_t a,
                                                        simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m512 a_vec = _mm512_loadu_ps(a.f32s);
    __m512 b_vec = _mm512_loadu_ps(b.f32s);
    state->sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
}

SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_skylake(simsimd_dot_f32x16_state_skylake_t const *state,
                                                          simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/** @brief Native F32 finalize for high-performance matmul without F64 conversion overhead. */
SIMSIMD_INTERNAL void simsimd_dot_f32x16_finalize_f32_skylake(simsimd_dot_f32x16_state_skylake_t const *state,
                                                              simsimd_f32_t *result) {
    *result = _simsimd_reduce_f32x16_to_f32_skylake(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e4m3 scalars on Skylake.
 */
typedef struct simsimd_dot_e4m3x64_state_skylake_t {
    __m512 sum;
} simsimd_dot_e4m3x64_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_skylake(simsimd_dot_e4m3x64_state_skylake_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_skylake(simsimd_dot_e4m3x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m128i a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e4m3s + 0));
    __m128i b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e4m3s + 0));
    sum = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_i8_vec), _simsimd_e4m3x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e4m3s + 16));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e4m3s + 16));
    sum = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_i8_vec), _simsimd_e4m3x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e4m3s + 32));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e4m3s + 32));
    sum = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_i8_vec), _simsimd_e4m3x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e4m3s + 48));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e4m3s + 48));
    sum = _mm512_fmadd_ps(_simsimd_e4m3x16_to_f32x16_skylake(a_i8_vec), _simsimd_e4m3x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_skylake(simsimd_dot_e4m3x64_state_skylake_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e5m2 scalars on Skylake.
 */
typedef struct simsimd_dot_e5m2x64_state_skylake_t {
    __m512 sum;
} simsimd_dot_e5m2x64_state_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_skylake(simsimd_dot_e5m2x64_state_skylake_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_skylake(simsimd_dot_e5m2x64_state_skylake_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m128i a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e5m2s + 0));
    __m128i b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e5m2s + 0));
    sum = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_i8_vec), _simsimd_e5m2x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e5m2s + 16));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e5m2s + 16));
    sum = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_i8_vec), _simsimd_e5m2x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e5m2s + 32));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e5m2s + 32));
    sum = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_i8_vec), _simsimd_e5m2x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    a_i8_vec = _mm_loadu_si128((__m128i const *)(a.e5m2s + 48));
    b_i8_vec = _mm_loadu_si128((__m128i const *)(b.e5m2s + 48));
    sum = _mm512_fmadd_ps(_simsimd_e5m2x16_to_f32x16_skylake(a_i8_vec), _simsimd_e5m2x16_to_f32x16_skylake(b_i8_vec),
                          sum);
    state->sum = sum;
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_skylake(simsimd_dot_e5m2x64_state_skylake_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/*  Unified Outer-Product API: f32x2
 *  ================================
 *  Processes 2 k-elements per update, outputs 1×16 partial products.
 *  This is the minimum efficient granularity for F32 (2 FMA ops per call).
 *
 *  B packing layout (transposed, k-major):
 *    b_packed[k_offset * 16 + col] = B[col][k_base + k_offset]
 *    where k_offset ∈ {0, 1}, col ∈ {0..15}
 *
 *  Usage in cache-blocked GEMM:
 *    for k_block in 0..K by K_TILE:
 *        simsimd_dot_outer_f32x2_update_1x16_skylake(&state, &A[k_block], &B_packed[k_block * 16])
 */

/** @brief State for 1×16 f32 outer-product (2 k-elements per update). */
typedef struct simsimd_dot_outer_f32x2_state_1x16_skylake_t {
    __m512 accumulator;
} simsimd_dot_outer_f32x2_state_1x16_skylake_t;

SIMSIMD_INTERNAL void simsimd_dot_outer_f32x2_init_1x16_skylake(simsimd_dot_outer_f32x2_state_1x16_skylake_t *state) {
    state->accumulator = _mm512_setzero_ps();
}

/**
 *  @brief Update 1×16 f32 state with 2 k-elements (2 FMA ops).
 *
 *  @param state         Pointer to accumulator state.
 *  @param a_slice       Pointer to 2 f32 values from A: a[k], a[k+1]
 *  @param b_transposed  Pointer to 32 f32 values from B (transposed):
 *                       b[0..15] = B[0..15][k], b[16..31] = B[0..15][k+1]
 */
SIMSIMD_INTERNAL void simsimd_dot_outer_f32x2_update_1x16_skylake(simsimd_dot_outer_f32x2_state_1x16_skylake_t *state,
                                                                  simsimd_f32_t const *a_slice,
                                                                  simsimd_f32_t const *b_transposed) {

    __m512 a_broadcast_0 = _mm512_set1_ps(a_slice[0]);
    __m512 b_column_0 = _mm512_loadu_ps(b_transposed);
    state->accumulator = _mm512_fmadd_ps(a_broadcast_0, b_column_0, state->accumulator);

    __m512 a_broadcast_1 = _mm512_set1_ps(a_slice[1]);
    __m512 b_column_1 = _mm512_loadu_ps(b_transposed + 16);
    state->accumulator = _mm512_fmadd_ps(a_broadcast_1, b_column_1, state->accumulator);
}

SIMSIMD_INTERNAL void simsimd_dot_outer_f32x2_finalize_1x16_skylake(
    simsimd_dot_outer_f32x2_state_1x16_skylake_t const *state, simsimd_f32_t *result_row) {
    _mm512_storeu_ps(result_row, state->accumulator);
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
                                           simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512i a_i16_vec, b_i16_vec;
    __m512 ab_vec = _mm512_setzero_ps();

simsimd_dot_bf16_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a_scalars);
        b_i16_vec = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    ab_vec = _mm512_dpbf16_ps(ab_vec, (__m512bh)(a_i16_vec), (__m512bh)(b_i16_vec));
    if (count_scalars) goto simsimd_dot_bf16_genoa_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_bf16c_genoa(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                            simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512i a_vec, b_vec;
    __m512 ab_real_vec = _mm512_setzero_ps();
    __m512 ab_imag_vec = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_vec = _mm512_set_epi8(                  //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)a_pairs);
        b_vec = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_epi16((simsimd_i16_t const *)a_pairs);
        b_vec = _mm512_loadu_epi16((simsimd_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    ab_real_vec = _mm512_dpbf16_ps(ab_real_vec, (__m512bh)(_mm512_xor_si512(b_vec, sign_flip_vec)), (__m512bh)(a_vec));
    ab_imag_vec = _mm512_dpbf16_ps(ab_imag_vec, (__m512bh)(_mm512_shuffle_epi8(b_vec, swap_adjacent_vec)),
                                   (__m512bh)(a_vec));
    if (count_pairs) goto simsimd_dot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    results[0] = _simsimd_reduce_f32x16_skylake(ab_real_vec);
    results[1] = _simsimd_reduce_f32x16_skylake(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_bf16c_genoa(simsimd_bf16c_t const *a_pairs, simsimd_bf16c_t const *b_pairs,
                                             simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512i a_vec, b_vec;
    __m512 ab_real_vec = _mm512_setzero_ps();
    __m512 ab_imag_vec = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_vec = _mm512_set_epi8(                  //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)a_pairs);
        b_vec = _mm512_maskz_loadu_epi16(mask, (simsimd_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_epi16((simsimd_i16_t const *)a_pairs);
        b_vec = _mm512_loadu_epi16((simsimd_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    ab_real_vec = _mm512_dpbf16_ps(ab_real_vec, (__m512bh)(a_vec), (__m512bh)(b_vec));
    a_vec = _mm512_xor_si512(a_vec, sign_flip_vec);
    b_vec = _mm512_shuffle_epi8(b_vec, swap_adjacent_vec);
    ab_imag_vec = _mm512_dpbf16_ps(ab_imag_vec, (__m512bh)(a_vec), (__m512bh)(b_vec));
    if (count_pairs) goto simsimd_dot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    results[0] = _simsimd_reduce_f32x16_skylake(ab_real_vec);
    results[1] = _simsimd_reduce_f32x16_skylake(ab_imag_vec);
}

/*  Convert 32x E4M3 values to 32x BF16 values.
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

/*  Convert 32x E5M2 values to 32x BF16 values.
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
                                           simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m256i a_i8_vec, b_i8_vec;
    __m512 ab_vec = _mm512_setzero_ps();

simsimd_dot_e4m3_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i8_vec = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm256_loadu_epi8(a_scalars);
        b_i8_vec = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E4M3 to BF16 and compute dot product
    __m512i a_bf16 = _simsimd_e4m3_to_bf16_genoa(a_i8_vec);
    __m512i b_bf16 = _simsimd_e4m3_to_bf16_genoa(b_i8_vec);
    ab_vec = _mm512_dpbf16_ps(ab_vec, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
    if (count_scalars) goto simsimd_dot_e4m3_genoa_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_genoa(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                           simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m256i a_i8_vec, b_i8_vec;
    __m512 ab_vec = _mm512_setzero_ps();

simsimd_dot_e5m2_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i8_vec = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm256_loadu_epi8(a_scalars);
        b_i8_vec = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E5M2 to BF16 and compute dot product
    __m512i a_bf16 = _simsimd_e5m2_to_bf16_genoa(a_i8_vec);
    __m512i b_bf16 = _simsimd_e5m2_to_bf16_genoa(b_i8_vec);
    ab_vec = _mm512_dpbf16_ps(ab_vec, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
    if (count_scalars) goto simsimd_dot_e5m2_genoa_cycle;

    *result = _simsimd_reduce_f32x16_skylake(ab_vec);
}

/**
 *  @brief Running state for 32-element dot accumulation over bf16 scalars on Genoa.
 */
typedef struct simsimd_dot_bf16x32_state_genoa_t {
    __m512 sum;
} simsimd_dot_bf16x32_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_init_genoa(simsimd_dot_bf16x32_state_genoa_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_update_genoa(simsimd_dot_bf16x32_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m512i a_i16_vec = _mm512_loadu_epi16(a.bf16s);
    __m512i b_i16_vec = _mm512_loadu_epi16(b.bf16s);
    state->sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_i16_vec), (__m512bh)(b_i16_vec));
}

SIMSIMD_INTERNAL void simsimd_dot_bf16x32_finalize_genoa(simsimd_dot_bf16x32_state_genoa_t const *state,
                                                         simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e4m3 scalars on Genoa.
 */
typedef struct simsimd_dot_e4m3x64_state_genoa_t {
    __m512 sum;
} simsimd_dot_e4m3x64_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_genoa(simsimd_dot_e4m3x64_state_genoa_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_genoa(simsimd_dot_e4m3x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m256i a_i8_vec = _mm256_loadu_epi8(a.e4m3s + 0);
    __m256i b_i8_vec = _mm256_loadu_epi8(b.e4m3s + 0);
    __m512i a_bf16 = _simsimd_e4m3_to_bf16_genoa(a_i8_vec);
    __m512i b_bf16 = _simsimd_e4m3_to_bf16_genoa(b_i8_vec);
    sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
    a_i8_vec = _mm256_loadu_epi8(a.e4m3s + 32);
    b_i8_vec = _mm256_loadu_epi8(b.e4m3s + 32);
    a_bf16 = _simsimd_e4m3_to_bf16_genoa(a_i8_vec);
    b_bf16 = _simsimd_e4m3_to_bf16_genoa(b_i8_vec);
    state->sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_genoa(simsimd_dot_e4m3x64_state_genoa_t const *state,
                                                         simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e5m2 scalars on Genoa.
 */
typedef struct simsimd_dot_e5m2x64_state_genoa_t {
    __m512 sum;
} simsimd_dot_e5m2x64_state_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_genoa(simsimd_dot_e5m2x64_state_genoa_t *state) {
    state->sum = _mm512_setzero();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_genoa(simsimd_dot_e5m2x64_state_genoa_t *state, simsimd_b512_vec_t a,
                                                       simsimd_b512_vec_t b) {
    __m512 sum = state->sum;
    __m256i a_i8_vec = _mm256_loadu_epi8(a.e5m2s + 0);
    __m256i b_i8_vec = _mm256_loadu_epi8(b.e5m2s + 0);
    __m512i a_bf16 = _simsimd_e5m2_to_bf16_genoa(a_i8_vec);
    __m512i b_bf16 = _simsimd_e5m2_to_bf16_genoa(b_i8_vec);
    sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
    a_i8_vec = _mm256_loadu_epi8(a.e5m2s + 32);
    b_i8_vec = _mm256_loadu_epi8(b.e5m2s + 32);
    a_bf16 = _simsimd_e5m2_to_bf16_genoa(a_i8_vec);
    b_bf16 = _simsimd_e5m2_to_bf16_genoa(b_i8_vec);
    state->sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_bf16), (__m512bh)(b_bf16));
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_genoa(simsimd_dot_e5m2x64_state_genoa_t const *state,
                                                         simsimd_distance_t *result) {
    *result = _simsimd_reduce_f32x16_skylake(state->sum);
}

/*  BF16x2 Outer-Product API (1×16, using VDPBF16PS with pair broadcast)
 *  =====================================================================
 *
 *  For computing 16 dot products simultaneously using VDPBF16PS instruction.
 *  Each update processes 2 k-elements (1 bf16 pair) and outputs to 16 columns.
 *
 *  VDPBF16PS: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
 *
 *  By broadcasting the same pair {a0, a1} to all 16 lanes and loading
 *  one pair per column from B, we compute 16 partial dot products per call.
 *
 *  B packing layout (interleaved pairs):
 *    b_packed[pair_idx * 32 + col * 2 + 0] = B[col][2*pair_idx]
 *    b_packed[pair_idx * 32 + col * 2 + 1] = B[col][2*pair_idx + 1]
 *
 *  A single _mm512_loadu_si512(b_packed + pair_idx * 32) loads:
 *    {b[0][2p], b[0][2p+1], b[1][2p], b[1][2p+1], ..., b[15][2p], b[15][2p+1]}
 */

/** @brief State for 1×16 bf16 outer-product using VDPBF16PS. */
typedef struct simsimd_dot_outer_bf16x4_state_1x16_genoa_t {
    __m512 acc; // 16 f32 accumulators (one per output column)
} simsimd_dot_outer_bf16x4_state_1x16_genoa_t;

SIMSIMD_INTERNAL void simsimd_dot_outer_bf16x4_init_1x16_genoa(simsimd_dot_outer_bf16x4_state_1x16_genoa_t *state) {
    state->acc = _mm512_setzero_ps();
}

/**
 *  @brief Update 1×16 bf16 state with 4 k-elements (2 pairs) using VDPBF16PS.
 *
 *  @param state    Pointer to 1×16 accumulator state.
 *  @param a        Pointer to 4 bf16 values from query row (a[k:k+4]).
 *  @param b_packed Pointer to 2 blocks of 16 interleaved bf16 pairs (2 × 64 bytes).
 *                  Layout: For pair p in 0..1: b_packed[p*32..p*32+31] contains
 *                  {b[0][2p], b[0][2p+1], b[1][2p], b[1][2p+1], ..., b[15][2p], b[15][2p+1]}
 */
SIMSIMD_INTERNAL void simsimd_dot_outer_bf16x4_update_1x16_genoa(simsimd_dot_outer_bf16x4_state_1x16_genoa_t *state,
                                                                 simsimd_bf16_t const *a,
                                                                 simsimd_bf16_t const *b_packed) {
    __m512 acc = state->acc;

    // Process 2 pairs (4 k-elements)
    __m512i a_bc_0 = _mm512_set1_epi32(*(simsimd_i32_t const *)(a + 0));
    __m512i b_vec_0 = _mm512_loadu_si512(b_packed + 0);
    acc = _mm512_dpbf16_ps(acc, (__m512bh)a_bc_0, (__m512bh)b_vec_0);

    __m512i a_bc_1 = _mm512_set1_epi32(*(simsimd_i32_t const *)(a + 2));
    __m512i b_vec_1 = _mm512_loadu_si512(b_packed + 32);
    acc = _mm512_dpbf16_ps(acc, (__m512bh)a_bc_1, (__m512bh)b_vec_1);

    state->acc = acc;
}

/**
 *  @brief Finalize: Store 16 f32 results directly (NO horizontal reduction).
 */
SIMSIMD_INTERNAL void simsimd_dot_outer_bf16x4_finalize_1x16_genoa(
    simsimd_dot_outer_bf16x4_state_1x16_genoa_t const *state, simsimd_f32_t *c) {
    _mm512_storeu_ps(c, state->acc);
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
                                             simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512i a_i16_vec, b_i16_vec;
    __m512h ab_vec = _mm512_setzero_ph();

simsimd_dot_f16_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a_scalars);
        b_i16_vec = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec), ab_vec);
    if (count_scalars) goto simsimd_dot_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_f16c_sapphire(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                              simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512i a_vec, b_vec;
    __m512h ab_real_vec = _mm512_setzero_ph();
    __m512h ab_imag_vec = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_vec = _mm512_set_epi8(                  //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_vec = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_epi16(a_pairs);
        b_vec = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    ab_real_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_xor_si512(b_vec, sign_flip_vec)),
                                  _mm512_castsi512_ph(a_vec), ab_real_vec);
    ab_imag_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_shuffle_epi8(b_vec, swap_adjacent_vec)),
                                  _mm512_castsi512_ph(a_vec), ab_imag_vec);
    if (count_pairs) goto simsimd_dot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    // TODO: Optimize this with tree-like reductions
    results[0] = _mm512_reduce_add_ph(ab_real_vec);
    results[1] = _mm512_reduce_add_ph(ab_imag_vec);
}

SIMSIMD_PUBLIC void simsimd_vdot_f16c_sapphire(simsimd_f16c_t const *a_pairs, simsimd_f16c_t const *b_pairs,
                                               simsimd_size_t count_pairs, simsimd_distance_t *results) {
    __m512i a_vec, b_vec;
    __m512h ab_real_vec = _mm512_setzero_ph();
    __m512h ab_imag_vec = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_vec = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_vec = _mm512_set_epi8(                  //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

simsimd_dot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_vec = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_vec = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_vec = _mm512_loadu_epi16(a_pairs);
        b_vec = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    ab_real_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_real_vec);
    a_vec = _mm512_xor_si512(a_vec, sign_flip_vec);
    b_vec = _mm512_shuffle_epi8(b_vec, swap_adjacent_vec);
    ab_imag_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_imag_vec);
    if (count_pairs) goto simsimd_dot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    results[0] = _mm512_reduce_add_ph(ab_real_vec);
    results[1] = _mm512_reduce_add_ph(ab_imag_vec);
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
SIMSIMD_INTERNAL __m512i _simsimd_e4m3_to_f16_sapphire(__m256i fp8) {
    __m512i v = _mm512_cvtepu8_epi16(fp8);
    // Sign: bit 7 → bit 15
    __m512i sign = _mm512_and_si512(_mm512_slli_epi16(v, 8), _mm512_set1_epi16((short)0x8000));
    // Exp+mant (7 bits) shifted left 7, then add bias adjustment (8<<10 = 0x2000)
    __m512i low7 = _mm512_and_si512(v, _mm512_set1_epi16(0x7F));
    __m512i exp_mant = _mm512_add_epi16(_mm512_slli_epi16(low7, 7), _mm512_set1_epi16(0x2000));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero - single instruction!
    __mmask32 has_exp = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x78));
    __m512i masked_exp_mant = _mm512_maskz_mov_epi16(has_exp, exp_mant);
    return _mm512_or_si512(sign, masked_exp_mant);
}

/*  Convert 32x E5M2 values to 32x F16 values.
 *  This is extremely fast because E5M2 and F16 have the same exponent bias (15).
 *  Simply zero-extend to 16-bit and shift left by 8.
 *
 *  E5M2 format: S EEEEE MM         (bias=15)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 */
SIMSIMD_INTERNAL __m512i _simsimd_e5m2_to_f16_sapphire(__m256i fp8) {
    __m512i v = _mm512_cvtepu8_epi16(fp8);
    return _mm512_slli_epi16(v, 8);
}

SIMSIMD_PUBLIC void simsimd_dot_e4m3_sapphire(simsimd_e4m3_t const *a_scalars, simsimd_e4m3_t const *b_scalars,
                                              simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m256i a_i8_vec, b_i8_vec;
    __m512h ab_vec = _mm512_setzero_ph();

simsimd_dot_e4m3_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i8_vec = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm256_loadu_epi8(a_scalars);
        b_i8_vec = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E4M3 to F16 and compute dot product
    __m512i a_f16 = _simsimd_e4m3_to_f16_sapphire(a_i8_vec);
    __m512i b_f16 = _simsimd_e4m3_to_f16_sapphire(b_i8_vec);
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), ab_vec);
    if (count_scalars) goto simsimd_dot_e4m3_sapphire_cycle;

    *result = _mm512_reduce_add_ph(ab_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_e5m2_sapphire(simsimd_e5m2_t const *a_scalars, simsimd_e5m2_t const *b_scalars,
                                              simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m256i a_i8_vec, b_i8_vec;
    __m512h ab_vec = _mm512_setzero_ph();

simsimd_dot_e5m2_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i8_vec = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_i8_vec = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_i8_vec = _mm256_loadu_epi8(a_scalars);
        b_i8_vec = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E5M2 to F16 and compute dot product
    // Note: E5M2 to F16 is extremely fast due to same exponent bias
    __m512i a_f16 = _simsimd_e5m2_to_f16_sapphire(a_i8_vec);
    __m512i b_f16 = _simsimd_e5m2_to_f16_sapphire(b_i8_vec);
    ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), ab_vec);
    if (count_scalars) goto simsimd_dot_e5m2_sapphire_cycle;

    *result = _mm512_reduce_add_ph(ab_vec);
}

/**
 *  @brief Running state for 32-element dot accumulation over f16 scalars on Sapphire.
 */
typedef struct simsimd_dot_f16x32_state_sapphire_t {
    __m512h sum;
} simsimd_dot_f16x32_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_f16x32_init_sapphire(simsimd_dot_f16x32_state_sapphire_t *state) {
    state->sum = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_update_sapphire(simsimd_dot_f16x32_state_sapphire_t *state,
                                                         simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum = state->sum;
    __m512i a_i16_vec = _mm512_loadu_epi16(a.f16s);
    __m512i b_i16_vec = _mm512_loadu_epi16(b.f16s);
    state->sum = _mm512_fmadd_ph(_mm512_castsi512_ph(a_i16_vec), _mm512_castsi512_ph(b_i16_vec), sum);
}

SIMSIMD_INTERNAL void simsimd_dot_f16x32_finalize_sapphire(simsimd_dot_f16x32_state_sapphire_t const *state,
                                                           simsimd_distance_t *result) {
    *result = _mm512_reduce_add_ph(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e4m3 scalars on Sapphire.
 */
typedef struct simsimd_dot_e4m3x64_state_sapphire_t {
    __m512h sum;
} simsimd_dot_e4m3x64_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_init_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state) {
    state->sum = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_update_sapphire(simsimd_dot_e4m3x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum = state->sum;
    __m256i a_i8_vec = _mm256_loadu_epi8(a.e4m3s + 0);
    __m256i b_i8_vec = _mm256_loadu_epi8(b.e4m3s + 0);
    __m512i a_f16 = _simsimd_e4m3_to_f16_sapphire(a_i8_vec);
    __m512i b_f16 = _simsimd_e4m3_to_f16_sapphire(b_i8_vec);
    sum = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), sum);
    a_i8_vec = _mm256_loadu_epi8(a.e4m3s + 32);
    b_i8_vec = _mm256_loadu_epi8(b.e4m3s + 32);
    a_f16 = _simsimd_e4m3_to_f16_sapphire(a_i8_vec);
    b_f16 = _simsimd_e4m3_to_f16_sapphire(b_i8_vec);
    state->sum = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), sum);
}

SIMSIMD_INTERNAL void simsimd_dot_e4m3x64_finalize_sapphire(simsimd_dot_e4m3x64_state_sapphire_t const *state,
                                                            simsimd_distance_t *result) {
    *result = _mm512_reduce_add_ph(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over e5m2 scalars on Sapphire.
 */
typedef struct simsimd_dot_e5m2x64_state_sapphire_t {
    __m512h sum;
} simsimd_dot_e5m2x64_state_sapphire_t;

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_init_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state) {
    state->sum = _mm512_setzero_ph();
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_update_sapphire(simsimd_dot_e5m2x64_state_sapphire_t *state,
                                                          simsimd_b512_vec_t a, simsimd_b512_vec_t b) {
    __m512h sum = state->sum;
    __m256i a_i8_vec = _mm256_loadu_epi8(a.e5m2s + 0);
    __m256i b_i8_vec = _mm256_loadu_epi8(b.e5m2s + 0);
    __m512i a_f16 = _simsimd_e5m2_to_f16_sapphire(a_i8_vec);
    __m512i b_f16 = _simsimd_e5m2_to_f16_sapphire(b_i8_vec);
    sum = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), sum);
    a_i8_vec = _mm256_loadu_epi8(a.e5m2s + 32);
    b_i8_vec = _mm256_loadu_epi8(b.e5m2s + 32);
    a_f16 = _simsimd_e5m2_to_f16_sapphire(a_i8_vec);
    b_f16 = _simsimd_e5m2_to_f16_sapphire(b_i8_vec);
    state->sum = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16), _mm512_castsi512_ph(b_f16), sum);
}

SIMSIMD_INTERNAL void simsimd_dot_e5m2x64_finalize_sapphire(simsimd_dot_e5m2x64_state_sapphire_t const *state,
                                                            simsimd_distance_t *result) {
    *result = _mm512_reduce_add_ph(state->sum);
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
                                       simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512i a_i16_vec, b_i16_vec;
    __m512i ab_i32_vec = _mm512_setzero_si512();

simsimd_dot_i8_ice_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a_scalars));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b_scalars));
        count_scalars = 0;
    }
    else {
        a_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a_scalars));
        b_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b_scalars));
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a_scalars.byte[4*j]) * SignExtend16(b_scalars.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    ab_i32_vec = _mm512_dpwssd_epi32(ab_i32_vec, a_i16_vec, b_i16_vec);
    if (count_scalars) goto simsimd_dot_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(ab_i32_vec);
}

SIMSIMD_PUBLIC void simsimd_dot_u8_ice(simsimd_u8_t const *a_scalars, simsimd_u8_t const *b_scalars,
                                       simsimd_size_t count_scalars, simsimd_distance_t *result) {
    __m512i a_u8_vec, b_u8_vec;
    __m512i a_i16_low_vec, a_i16_high_vec, b_i16_low_vec, b_i16_high_vec;
    __m512i ab_i32_low_vec = _mm512_setzero_si512();
    __m512i ab_i32_high_vec = _mm512_setzero_si512();
    __m512i const zeros_vec = _mm512_setzero_si512();

simsimd_dot_u8_ice_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_si512(a_scalars);
        b_u8_vec = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_i16_low_vec = _mm512_unpacklo_epi8(a_u8_vec, zeros_vec);
    a_i16_high_vec = _mm512_unpackhi_epi8(a_u8_vec, zeros_vec);
    b_i16_low_vec = _mm512_unpacklo_epi8(b_u8_vec, zeros_vec);
    b_i16_high_vec = _mm512_unpackhi_epi8(b_u8_vec, zeros_vec);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    ab_i32_low_vec = _mm512_dpwssd_epi32(ab_i32_low_vec, a_i16_low_vec, b_i16_low_vec);
    ab_i32_high_vec = _mm512_dpwssd_epi32(ab_i32_high_vec, a_i16_high_vec, b_i16_high_vec);
    if (count_scalars) goto simsimd_dot_u8_ice_cycle;

    *result = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
}

/**
 *  @brief Running state for 64-element dot accumulation over i8 scalars on Ice Lake.
 */
typedef struct simsimd_dot_i8x64_state_ice_t {
    __m512i sum;
} simsimd_dot_i8x64_state_ice_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_ice(simsimd_dot_i8x64_state_ice_t *state) {
    state->sum = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_ice(simsimd_dot_i8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b) {
    __m512i sum = state->sum;
    __m512i a_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(a.i8s + 0)));
    __m512i b_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(b.i8s + 0)));
    sum = _mm512_dpwssd_epi32(sum, a_i16_vec, b_i16_vec);
    a_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(a.i8s + 32)));
    b_i16_vec = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)(b.i8s + 32)));
    state->sum = _mm512_dpwssd_epi32(sum, a_i16_vec, b_i16_vec);
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_ice(simsimd_dot_i8x64_state_ice_t const *state,
                                                     simsimd_distance_t *result) {
    *result = _mm512_reduce_add_epi32(state->sum);
}

/**
 *  @brief Running state for 64-element dot accumulation over u8 scalars on Ice Lake.
 */
typedef struct simsimd_dot_u8x64_state_ice_t {
    __m512i sum_low;
    __m512i sum_high;
} simsimd_dot_u8x64_state_ice_t;

SIMSIMD_INTERNAL void simsimd_dot_u8x64_init_ice(simsimd_dot_u8x64_state_ice_t *state) {
    state->sum_low = _mm512_setzero_si512();
    state->sum_high = _mm512_setzero_si512();
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_update_ice(simsimd_dot_u8x64_state_ice_t *state, simsimd_b512_vec_t a,
                                                   simsimd_b512_vec_t b) {
    __m512i sum_low = state->sum_low;
    __m512i sum_high = state->sum_high;
    __m512i const zeros_vec = _mm512_setzero_si512();

    __m512i a_u8_vec = _mm512_loadu_si512(a.u8s);
    __m512i b_u8_vec = _mm512_loadu_si512(b.u8s);
    __m512i a_i16_low = _mm512_unpacklo_epi8(a_u8_vec, zeros_vec);
    __m512i a_i16_high = _mm512_unpackhi_epi8(a_u8_vec, zeros_vec);
    __m512i b_i16_low = _mm512_unpacklo_epi8(b_u8_vec, zeros_vec);
    __m512i b_i16_high = _mm512_unpackhi_epi8(b_u8_vec, zeros_vec);
    sum_low = _mm512_dpwssd_epi32(sum_low, a_i16_low, b_i16_low);
    sum_high = _mm512_dpwssd_epi32(sum_high, a_i16_high, b_i16_high);

    state->sum_low = sum_low;
    state->sum_high = sum_high;
}

SIMSIMD_INTERNAL void simsimd_dot_u8x64_finalize_ice(simsimd_dot_u8x64_state_ice_t const *state,
                                                     simsimd_distance_t *result) {
    __m512i sum = _mm512_add_epi32(state->sum_low, state->sum_high);
    *result = _mm512_reduce_add_epi32(sum);
}

/*  Unified Outer-Product API: i8x8
 *  ================================
 *  Processes 8 k-elements per update, outputs 1×16 partial products.
 *  Uses 4 VPDPWSSD instructions (each processes 2 i16 = 2 sign-extended i8).
 *
 *  B packing layout (pair-interleaved, i8 sign-extended to i16 on load):
 *    b_packed[pair_index * 32 + col * 2 + offset] = B[col][pair_index * 2 + offset]
 *    where pair_index ∈ {0..3}, col ∈ {0..15}, offset ∈ {0, 1}
 *
 *  Note: VPDPWSSD operates on i16 pairs. We sign-extend i8→i16 before use.
 *        This handles signed i8 × signed i8 correctly (unlike VPDPBUSD which is u8×i8).
 */

/** @brief State for 1×16 i8 outer-product (8 k-elements per update). */
typedef struct simsimd_dot_outer_i8x8_state_1x16_ice_t {
    __m512i accumulator;
} simsimd_dot_outer_i8x8_state_1x16_ice_t;

SIMSIMD_INTERNAL void simsimd_dot_outer_i8x8_init_1x16_ice(simsimd_dot_outer_i8x8_state_1x16_ice_t *state) {
    state->accumulator = _mm512_setzero_si512();
}

/**
 *  @brief Update 1×16 i8 state with 8 k-elements (4 VPDPWSSD ops).
 *
 *  @param state               Pointer to accumulator state.
 *  @param a_slice             Pointer to 8 i8 values from A: a[k..k+7]
 *  @param b_pairs_interleaved Pointer to 128 i8 values (4 pair-blocks of 32 each):
 *                             Block p: {B[0][2p], B[0][2p+1], B[1][2p], B[1][2p+1], ...}
 */
SIMSIMD_INTERNAL void simsimd_dot_outer_i8x8_update_1x16_ice(simsimd_dot_outer_i8x8_state_1x16_ice_t *state,
                                                             simsimd_i8_t const *a_slice,
                                                             simsimd_i8_t const *b_pairs_interleaved) {

    __m512i accumulator = state->accumulator;

// Process 4 pairs of i8 values (8 total k-elements)
#define SIMSIMD_I8X8_PROCESS_PAIR(pair_index)                                                              \
    do {                                                                                                   \
        /* Sign-extend a[2p] and a[2p+1] to i16, pack into i32, broadcast */                               \
        simsimd_i16_t a_lo = (simsimd_i16_t)a_slice[pair_index * 2];                                       \
        simsimd_i16_t a_hi = (simsimd_i16_t)a_slice[pair_index * 2 + 1];                                   \
        simsimd_i32_t a_pair_packed = ((simsimd_i32_t)(a_hi & 0xFFFF) << 16) | (a_lo & 0xFFFF);            \
        __m512i a_pair_broadcast = _mm512_set1_epi32(a_pair_packed);                                       \
                                                                                                           \
        /* Load 32 i8 from B, sign-extend to 32 i16 */                                                     \
        __m256i b_i8_chunk = _mm256_loadu_si256((__m256i const *)(b_pairs_interleaved + pair_index * 32)); \
        __m512i b_i16_extended = _mm512_cvtepi8_epi16(b_i8_chunk);                                         \
                                                                                                           \
        /* VPDPWSSD: acc[i] += a_lo * b[2i] + a_hi * b[2i+1] */                                            \
        accumulator = _mm512_dpwssd_epi32(accumulator, a_pair_broadcast, b_i16_extended);                  \
    } while (0)

    SIMSIMD_I8X8_PROCESS_PAIR(0);
    SIMSIMD_I8X8_PROCESS_PAIR(1);
    SIMSIMD_I8X8_PROCESS_PAIR(2);
    SIMSIMD_I8X8_PROCESS_PAIR(3);

#undef SIMSIMD_I8X8_PROCESS_PAIR

    state->accumulator = accumulator;
}

SIMSIMD_INTERNAL void simsimd_dot_outer_i8x8_finalize_1x16_ice(simsimd_dot_outer_i8x8_state_1x16_ice_t const *state,
                                                               simsimd_i32_t *result_row) {
    _mm512_storeu_si512(result_row, state->accumulator);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE

#if SIMSIMD_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_dot_i8_sierra(simsimd_i8_t const *a_scalars, simsimd_i8_t const *b_scalars,
                                          simsimd_size_t count_scalars, simsimd_distance_t *result) {

    __m256i ab_i32_vec = _mm256_setzero_si256();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_i8_vec = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8_vec = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        ab_i32_vec = _mm256_dpbssds_epi32(ab_i32_vec, a_i8_vec, b_i8_vec);
    }

    // Further reduce to a single sum for each vector
    int ab = _simsimd_reduce_i32x8_haswell(ab_i32_vec);

    // Take care of the tail:
    for (; idx_scalars < count_scalars; ++idx_scalars) ab += (int)(a_scalars[idx_scalars]) * b_scalars[idx_scalars];
    *result = ab;
}

/**
 *  @brief Running state for 64-element dot accumulation over i8 scalars on Sierra.
 */
typedef struct simsimd_dot_i8x64_state_sierra_t {
    __m256i sum;
} simsimd_dot_i8x64_state_sierra_t;

SIMSIMD_INTERNAL void simsimd_dot_i8x64_init_sierra(simsimd_dot_i8x64_state_sierra_t *state) {
    state->sum = _mm256_setzero_si256();
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_update_sierra(simsimd_dot_i8x64_state_sierra_t *state, simsimd_b512_vec_t a,
                                                      simsimd_b512_vec_t b) {
    __m256i sum = state->sum;
    sum = _mm256_dpbssds_epi32(sum, _mm256_lddqu_si256((__m256i const *)(a.i8s + 0)),
                               _mm256_lddqu_si256((__m256i const *)(b.i8s + 0)));
    state->sum = _mm256_dpbssds_epi32(sum, _mm256_lddqu_si256((__m256i const *)(a.i8s + 32)),
                                      _mm256_lddqu_si256((__m256i const *)(b.i8s + 32)));
}

SIMSIMD_INTERNAL void simsimd_dot_i8x64_finalize_sierra(simsimd_dot_i8x64_state_sierra_t const *state,
                                                        simsimd_distance_t *result) {
    *result = _simsimd_reduce_i32x8_haswell(state->sum);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SIERRA
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_dot_i8(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                   simsimd_distance_t *d) {
#if SIMSIMD_TARGET_NEON_I8
    simsimd_dot_i8_neon(a, b, n, d);
#elif SIMSIMD_TARGET_ICE
    simsimd_dot_i8_ice(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_i8_haswell(a, b, n, d);
#else
    simsimd_dot_i8_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_u8(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                   simsimd_distance_t *d) {
#if SIMSIMD_TARGET_NEON_I8
    simsimd_dot_u8_neon(a, b, n, d);
#elif SIMSIMD_TARGET_ICE
    simsimd_dot_u8_ice(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_u8_haswell(a, b, n, d);
#else
    simsimd_dot_u8_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE_F16
    simsimd_dot_f16_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_dot_f16_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_f16_sapphire(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f16_haswell(a, b, n, d);
#else
    simsimd_dot_f16_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_GENOA
    simsimd_dot_bf16_genoa(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_bf16_haswell(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_dot_bf16_neon(a, b, n, d);
#else
    simsimd_dot_bf16_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_e4m3(simsimd_e4m3_t const *a, simsimd_e4m3_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_e4m3_sapphire(a, b, n, d);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dot_e4m3_genoa(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_e4m3_skylake(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_e4m3_haswell(a, b, n, d);
#else
    simsimd_dot_e4m3_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_e5m2(simsimd_e5m2_t const *a, simsimd_e5m2_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_e5m2_sapphire(a, b, n, d);
#elif SIMSIMD_TARGET_GENOA
    simsimd_dot_e5m2_genoa(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_e5m2_skylake(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_e5m2_haswell(a, b, n, d);
#else
    simsimd_dot_e5m2_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f32_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_dot_f32_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f32_skylake(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f32_haswell(a, b, n, d);
#else
    simsimd_dot_f32_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f64_sve(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f64_skylake(a, b, n, d);
#else
    simsimd_dot_f64_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f16c(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE_F16
    simsimd_dot_f16c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_dot_f16c_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_dot_f16c_sapphire(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f16c_haswell(a, b, n, d);
#else
    simsimd_dot_f16c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_bf16c(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *d) {
#if SIMSIMD_TARGET_GENOA
    simsimd_dot_bf16c_genoa(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_dot_bf16c_neon(a, b, n, d);
#else
    simsimd_dot_bf16c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f32c(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f32c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_dot_f32c_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f32c_skylake(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_dot_f32c_haswell(a, b, n, d);
#else
    simsimd_dot_f32c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_dot_f64c(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_dot_f64c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_dot_f64c_skylake(a, b, n, d);
#else
    simsimd_dot_f64c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f16c(simsimd_f16c_t const *a, simsimd_f16c_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f16c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_F16
    simsimd_vdot_f16c_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SAPPHIRE
    simsimd_vdot_f16c_sapphire(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vdot_f16c_haswell(a, b, n, d);
#else
    simsimd_vdot_f16c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_bf16c(simsimd_bf16c_t const *a, simsimd_bf16c_t const *b, simsimd_size_t n,
                                       simsimd_distance_t *d) {
#if SIMSIMD_TARGET_GENOA
    simsimd_vdot_bf16c_genoa(a, b, n, d);
#elif SIMSIMD_TARGET_NEON_BF16
    simsimd_vdot_bf16c_neon(a, b, n, d);
#else
    simsimd_vdot_bf16c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f32c(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f32c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_NEON
    simsimd_vdot_f32c_neon(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_vdot_f32c_skylake(a, b, n, d);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_vdot_f32c_haswell(a, b, n, d);
#else
    simsimd_vdot_f32c_serial(a, b, n, d);
#endif
}
SIMSIMD_PUBLIC void simsimd_vdot_f64c(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *d) {
#if SIMSIMD_TARGET_SVE
    simsimd_vdot_f64c_sve(a, b, n, d);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_vdot_f64c_skylake(a, b, n, d);
#else
    simsimd_vdot_f64c_serial(a, b, n, d);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
