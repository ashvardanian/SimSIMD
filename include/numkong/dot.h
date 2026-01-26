/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers.
 *  @file include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date February 24, 2024
 *
 *  Contains:
 *
 *  - Dot Product for Real and Complex vectors
 *  - Conjugate Dot Product for Complex vectors
 *
 *  For dtypes:
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
 *  - Arm: NEON, NEON+I8, NEON+F16, NEON+BF16, SVE, SVE+F16
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire Rapids, Sierra Forest
 *  - RISC-V: SpaceMIT, SiFive, XuanTie
 *
 *  @section streaming_api Streaming API
 *
 *  For compile-time dispatch and vector-at-a-time accumulation, we provide streaming helpers
 *  that accept two `nk_b512_vec_t` blocks and update a running sum for non-complex dot
 *  products. The `<count>` suffix reflects how many scalars of that type fit in a 512-bit block.
 *  The helpers are exposed per scalar type as:
 *
 *  - nk_dot_<type>x<count>_state_<isa>_t
 *  - nk_dot_<type>x<count>_init_<isa>
 *  - nk_dot_<type>x<count>_update_<isa>
 *  - nk_dot_<type>x<count>_finalize_<isa>
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
#ifndef NK_DOT_H
#define NK_DOT_H

#include "numkong/types.h"

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
NK_DYNAMIC void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e2m3(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32 */
NK_DYNAMIC void nk_dot_e3m2(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

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
NK_DYNAMIC void nk_dot_f32c(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f32c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_f64c(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f64c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_f16c(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                            nk_f32c_t *result);
/** @copydoc nk_dot_f32c */
NK_DYNAMIC void nk_dot_bf16c(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f32c_t *result);

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
NK_DYNAMIC void nk_vdot_f32c(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_f64c(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f64c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_f16c(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                             nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_DYNAMIC void nk_vdot_bf16c(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                              nk_f32c_t *result);

/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i4 */
NK_PUBLIC void nk_dot_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u4 */
NK_PUBLIC void nk_dot_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_serial(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_serial(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars.
 */
typedef struct nk_dot_f64x2_state_serial_t nk_dot_f64x2_state_serial_t;
/** @copydoc nk_dot_f64x2_state_serial_t */
NK_INTERNAL void nk_dot_f64x2_init_serial(nk_dot_f64x2_state_serial_t *state);
/** @copydoc nk_dot_f64x2_state_serial_t */
NK_INTERNAL void nk_dot_f64x2_update_serial(nk_dot_f64x2_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f64x2_state_serial_t */
NK_INTERNAL void nk_dot_f64x2_finalize_serial( //
    nk_dot_f64x2_state_serial_t const *state_a, nk_dot_f64x2_state_serial_t const *state_b,
    nk_dot_f64x2_state_serial_t const *state_c, nk_dot_f64x2_state_serial_t const *state_d, nk_size_t total_dimensions,
    nk_b256_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over f32 scalars.
 */
typedef struct nk_dot_f32x4_state_serial_t nk_dot_f32x4_state_serial_t;
/** @copydoc nk_dot_f32x4_state_serial_t */
NK_INTERNAL void nk_dot_f32x4_init_serial(nk_dot_f32x4_state_serial_t *state);
/** @copydoc nk_dot_f32x4_state_serial_t */
NK_INTERNAL void nk_dot_f32x4_update_serial(nk_dot_f32x4_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f32x4_state_serial_t */
NK_INTERNAL void nk_dot_f32x4_finalize_serial( //
    nk_dot_f32x4_state_serial_t const *state_a, nk_dot_f32x4_state_serial_t const *state_b,
    nk_dot_f32x4_state_serial_t const *state_c, nk_dot_f32x4_state_serial_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars.
 */
typedef struct nk_dot_f16x8_state_serial_t nk_dot_f16x8_state_serial_t;
/** @copydoc nk_dot_f16x8_state_serial_t */
NK_INTERNAL void nk_dot_f16x8_init_serial(nk_dot_f16x8_state_serial_t *state);
/** @copydoc nk_dot_f16x8_state_serial_t */
NK_INTERNAL void nk_dot_f16x8_update_serial(nk_dot_f16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f16x8_state_serial_t */
NK_INTERNAL void nk_dot_f16x8_finalize_serial( //
    nk_dot_f16x8_state_serial_t const *state_a, nk_dot_f16x8_state_serial_t const *state_b,
    nk_dot_f16x8_state_serial_t const *state_c, nk_dot_f16x8_state_serial_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars.
 */
typedef struct nk_dot_bf16x8_state_serial_t nk_dot_bf16x8_state_serial_t;
/** @copydoc nk_dot_bf16x8_state_serial_t */
NK_INTERNAL void nk_dot_bf16x8_init_serial(nk_dot_bf16x8_state_serial_t *state);
/** @copydoc nk_dot_bf16x8_state_serial_t */
NK_INTERNAL void nk_dot_bf16x8_update_serial(nk_dot_bf16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_bf16x8_state_serial_t */
NK_INTERNAL void nk_dot_bf16x8_finalize_serial( //
    nk_dot_bf16x8_state_serial_t const *state_a, nk_dot_bf16x8_state_serial_t const *state_b,
    nk_dot_bf16x8_state_serial_t const *state_c, nk_dot_bf16x8_state_serial_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars.
 */
typedef struct nk_dot_i8x16_state_serial_t nk_dot_i8x16_state_serial_t;
/** @copydoc nk_dot_i8x16_state_serial_t */
NK_INTERNAL void nk_dot_i8x16_init_serial(nk_dot_i8x16_state_serial_t *state);
/** @copydoc nk_dot_i8x16_state_serial_t */
NK_INTERNAL void nk_dot_i8x16_update_serial(nk_dot_i8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i8x16_state_serial_t */
NK_INTERNAL void nk_dot_i8x16_finalize_serial( //
    nk_dot_i8x16_state_serial_t const *state_a, nk_dot_i8x16_state_serial_t const *state_b,
    nk_dot_i8x16_state_serial_t const *state_c, nk_dot_i8x16_state_serial_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars.
 */
typedef struct nk_dot_u8x16_state_serial_t nk_dot_u8x16_state_serial_t;
/** @copydoc nk_dot_u8x16_state_serial_t */
NK_INTERNAL void nk_dot_u8x16_init_serial(nk_dot_u8x16_state_serial_t *state);
/** @copydoc nk_dot_u8x16_state_serial_t */
NK_INTERNAL void nk_dot_u8x16_update_serial(nk_dot_u8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_u8x16_state_serial_t */
NK_INTERNAL void nk_dot_u8x16_finalize_serial( //
    nk_dot_u8x16_state_serial_t const *state_a, nk_dot_u8x16_state_serial_t const *state_b,
    nk_dot_u8x16_state_serial_t const *state_c, nk_dot_u8x16_state_serial_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over e4m3 scalars.
 */
typedef struct nk_dot_e4m3x16_state_serial_t nk_dot_e4m3x16_state_serial_t;
/** @copydoc nk_dot_e4m3x16_state_serial_t */
NK_INTERNAL void nk_dot_e4m3x16_init_serial(nk_dot_e4m3x16_state_serial_t *state);
/** @copydoc nk_dot_e4m3x16_state_serial_t */
NK_INTERNAL void nk_dot_e4m3x16_update_serial(nk_dot_e4m3x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_e4m3x16_state_serial_t */
NK_INTERNAL void nk_dot_e4m3x16_finalize_serial( //
    nk_dot_e4m3x16_state_serial_t const *state_a, nk_dot_e4m3x16_state_serial_t const *state_b,
    nk_dot_e4m3x16_state_serial_t const *state_c, nk_dot_e4m3x16_state_serial_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over e5m2 scalars.
 */
typedef struct nk_dot_e5m2x16_state_serial_t nk_dot_e5m2x16_state_serial_t;
/** @copydoc nk_dot_e5m2x16_state_serial_t */
NK_INTERNAL void nk_dot_e5m2x16_init_serial(nk_dot_e5m2x16_state_serial_t *state);
/** @copydoc nk_dot_e5m2x16_state_serial_t */
NK_INTERNAL void nk_dot_e5m2x16_update_serial(nk_dot_e5m2x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_e5m2x16_state_serial_t */
NK_INTERNAL void nk_dot_e5m2x16_finalize_serial( //
    nk_dot_e5m2x16_state_serial_t const *state_a, nk_dot_e5m2x16_state_serial_t const *state_b,
    nk_dot_e5m2x16_state_serial_t const *state_c, nk_dot_e5m2x16_state_serial_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);

typedef struct nk_dot_f32x4_state_neon_t nk_dot_f32x4_state_neon_t;
/** @copydoc nk_dot_f32x4_state_neon_t */
NK_INTERNAL void nk_dot_f32x4_init_neon(nk_dot_f32x4_state_neon_t *state);
/** @copydoc nk_dot_f32x4_state_neon_t */
NK_INTERNAL void nk_dot_f32x4_update_neon(nk_dot_f32x4_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f32x4_state_neon_t */
NK_INTERNAL void nk_dot_f32x4_finalize_neon( //
    nk_dot_f32x4_state_neon_t const *state_a, nk_dot_f32x4_state_neon_t const *state_b,
    nk_dot_f32x4_state_neon_t const *state_c, nk_dot_f32x4_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);

#if NK_TARGET_NEONFHM
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_neonfhm(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_neonfhm(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_neonfhm(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on NEON with FMLAL.
 */
typedef struct nk_dot_f16x8_state_neonfhm_t nk_dot_f16x8_state_neonfhm_t;
/** @copydoc nk_dot_f16x8_state_neonfhm_t */
NK_INTERNAL void nk_dot_f16x8_init_neonfhm(nk_dot_f16x8_state_neonfhm_t *state);
/** @copydoc nk_dot_f16x8_state_neonfhm_t */
NK_INTERNAL void nk_dot_f16x8_update_neonfhm(nk_dot_f16x8_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f16x8_state_neonfhm_t */
NK_INTERNAL void nk_dot_f16x8_finalize_neonfhm( //
    nk_dot_f16x8_state_neonfhm_t const *state_a, nk_dot_f16x8_state_neonfhm_t const *state_b,
    nk_dot_f16x8_state_neonfhm_t const *state_c, nk_dot_f16x8_state_neonfhm_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over e2m3 scalars on NEON with FMLAL.
 */
typedef struct nk_dot_e2m3x16_state_neonfhm_t nk_dot_e2m3x16_state_neonfhm_t;
/** @copydoc nk_dot_e2m3x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e2m3x16_init_neonfhm(nk_dot_e2m3x16_state_neonfhm_t *state);
/** @copydoc nk_dot_e2m3x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e2m3x16_update_neonfhm(nk_dot_e2m3x16_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_e2m3x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e2m3x16_finalize_neonfhm( //
    nk_dot_e2m3x16_state_neonfhm_t const *state_a, nk_dot_e2m3x16_state_neonfhm_t const *state_b,
    nk_dot_e2m3x16_state_neonfhm_t const *state_c, nk_dot_e2m3x16_state_neonfhm_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over e3m2 scalars on NEON with FMLAL.
 */
typedef struct nk_dot_e3m2x16_state_neonfhm_t nk_dot_e3m2x16_state_neonfhm_t;
/** @copydoc nk_dot_e3m2x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e3m2x16_init_neonfhm(nk_dot_e3m2x16_state_neonfhm_t *state);
/** @copydoc nk_dot_e3m2x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e3m2x16_update_neonfhm(nk_dot_e3m2x16_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_e3m2x16_state_neonfhm_t */
NK_INTERNAL void nk_dot_e3m2x16_finalize_neonfhm( //
    nk_dot_e3m2x16_state_neonfhm_t const *state_a, nk_dot_e3m2x16_state_neonfhm_t const *state_b,
    nk_dot_e3m2x16_state_neonfhm_t const *state_c, nk_dot_e3m2x16_state_neonfhm_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

#endif // NK_TARGET_NEONFHM
/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on NEON.
 */
typedef struct nk_dot_f16x8_state_neon_t nk_dot_f16x8_state_neon_t;
/** @copydoc nk_dot_f16x8_state_neon_t */
NK_INTERNAL void nk_dot_f16x8_init_neon(nk_dot_f16x8_state_neon_t *state);
/** @copydoc nk_dot_f16x8_state_neon_t */
NK_INTERNAL void nk_dot_f16x8_update_neon(nk_dot_f16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f16x8_state_neon_t */
NK_INTERNAL void nk_dot_f16x8_finalize_neon( //
    nk_dot_f16x8_state_neon_t const *state_a, nk_dot_f16x8_state_neon_t const *state_b,
    nk_dot_f16x8_state_neon_t const *state_c, nk_dot_f16x8_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONSDOT
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);

typedef struct nk_dot_i8x16_state_neon_t nk_dot_i8x16_state_neon_t;
/** @copydoc nk_dot_i8x16_state_neon_t */
NK_INTERNAL void nk_dot_i8x16_init_neon(nk_dot_i8x16_state_neon_t *state);
/** @copydoc nk_dot_i8x16_state_neon_t */
NK_INTERNAL void nk_dot_i8x16_update_neon(nk_dot_i8x16_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i8x16_state_neon_t */
NK_INTERNAL void nk_dot_i8x16_finalize_neon( //
    nk_dot_i8x16_state_neon_t const *state_a, nk_dot_i8x16_state_neon_t const *state_b,
    nk_dot_i8x16_state_neon_t const *state_c, nk_dot_i8x16_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

typedef struct nk_dot_u8x16_state_neon_t nk_dot_u8x16_state_neon_t;
/** @copydoc nk_dot_u8x16_state_neon_t */
NK_INTERNAL void nk_dot_u8x16_init_neon(nk_dot_u8x16_state_neon_t *state);
/** @copydoc nk_dot_u8x16_state_neon_t */
NK_INTERNAL void nk_dot_u8x16_update_neon(nk_dot_u8x16_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_u8x16_state_neon_t */
NK_INTERNAL void nk_dot_u8x16_finalize_neon( //
    nk_dot_u8x16_state_neon_t const *state_a, nk_dot_u8x16_state_neon_t const *state_b,
    nk_dot_u8x16_state_neon_t const *state_c, nk_dot_u8x16_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over i4 scalars on NEON with SDOT.
 */
typedef struct nk_dot_i4x32_state_neonsdot_t nk_dot_i4x32_state_neonsdot_t;
/** @copydoc nk_dot_i4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_i4x32_init_neonsdot(nk_dot_i4x32_state_neonsdot_t *state);
/** @copydoc nk_dot_i4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_i4x32_update_neonsdot(nk_dot_i4x32_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_i4x32_finalize_neonsdot( //
    nk_dot_i4x32_state_neonsdot_t const *state_a, nk_dot_i4x32_state_neonsdot_t const *state_b,
    nk_dot_i4x32_state_neonsdot_t const *state_c, nk_dot_i4x32_state_neonsdot_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over u4 scalars on NEON with UDOT.
 */
typedef struct nk_dot_u4x32_state_neonsdot_t nk_dot_u4x32_state_neonsdot_t;
/** @copydoc nk_dot_u4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_u4x32_init_neonsdot(nk_dot_u4x32_state_neonsdot_t *state);
/** @copydoc nk_dot_u4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_u4x32_update_neonsdot(nk_dot_u4x32_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_u4x32_state_neonsdot_t */
NK_INTERNAL void nk_dot_u4x32_finalize_neonsdot( //
    nk_dot_u4x32_state_neonsdot_t const *state_a, nk_dot_u4x32_state_neonsdot_t const *state_b,
    nk_dot_u4x32_state_neonsdot_t const *state_c, nk_dot_u4x32_state_neonsdot_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

typedef struct nk_dot_bf16x8_state_neon_t nk_dot_bf16x8_state_neon_t;
/** @copydoc nk_dot_bf16x8_state_neon_t */
NK_INTERNAL void nk_dot_bf16x8_init_neon(nk_dot_bf16x8_state_neon_t *state);
/** @copydoc nk_dot_bf16x8_state_neon_t */
NK_INTERNAL void nk_dot_bf16x8_update_neon(nk_dot_bf16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_bf16x8_state_neon_t */
NK_INTERNAL void nk_dot_bf16x8_finalize_neon( //
    nk_dot_bf16x8_state_neon_t const *state_a, nk_dot_bf16x8_state_neon_t const *state_b,
    nk_dot_bf16x8_state_neon_t const *state_c, nk_dot_bf16x8_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_SVE
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_sve(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_sve(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
#endif // NK_TARGET_SVE

#if NK_TARGET_SVEHALF
/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_svehalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
#endif // NK_TARGET_SVEHALF

#if NK_TARGET_HASWELL
/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f16c */
NK_PUBLIC void nk_dot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f16c */
NK_PUBLIC void nk_vdot_f16c_haswell(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e2m3 */
NK_PUBLIC void nk_dot_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e3m2 */
NK_PUBLIC void nk_dot_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Running state for f32 dot accumulation using f64 for high precision on Haswell.
 *  Processes 4 f32 values at a time, upcasting to f64.
 */
typedef struct nk_dot_f32x4_state_haswell_t nk_dot_f32x4_state_haswell_t;
/** @copydoc nk_dot_f32x4_state_haswell_t */
NK_INTERNAL void nk_dot_f32x4_init_haswell(nk_dot_f32x4_state_haswell_t *state);
/** @copydoc nk_dot_f32x4_state_haswell_t */
NK_INTERNAL void nk_dot_f32x4_update_haswell(nk_dot_f32x4_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f32x4_state_haswell_t */
NK_INTERNAL void nk_dot_f32x4_finalize_haswell( //
    nk_dot_f32x4_state_haswell_t const *state_a, nk_dot_f32x4_state_haswell_t const *state_b,
    nk_dot_f32x4_state_haswell_t const *state_c, nk_dot_f32x4_state_haswell_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_f16x8_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_bf16x8_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e4m3 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e4m3x16_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e5m2 scalars on Haswell.
 *  @note Alias of nk_dot_through_f32_state_haswell_t_
 */
typedef struct nk_dot_through_f32_state_haswell_t_ nk_dot_e5m2x16_state_haswell_t;

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on Haswell.
 */
typedef struct nk_dot_i8x16_state_haswell_t nk_dot_i8x16_state_haswell_t;
/** @copydoc nk_dot_i8x16_state_haswell_t */
NK_INTERNAL void nk_dot_i8x16_init_haswell(nk_dot_i8x16_state_haswell_t *state);
/** @copydoc nk_dot_i8x16_state_haswell_t */
NK_INTERNAL void nk_dot_i8x16_update_haswell(nk_dot_i8x16_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i8x16_state_haswell_t */
NK_INTERNAL void nk_dot_i8x16_finalize_haswell( //
    nk_dot_i8x16_state_haswell_t const *state_a, nk_dot_i8x16_state_haswell_t const *state_b,
    nk_dot_i8x16_state_haswell_t const *state_c, nk_dot_i8x16_state_haswell_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on Haswell.
 */
typedef struct nk_dot_u8x16_state_haswell_t nk_dot_u8x16_state_haswell_t;
/** @copydoc nk_dot_u8x16_state_haswell_t */
NK_INTERNAL void nk_dot_u8x16_init_haswell(nk_dot_u8x16_state_haswell_t *state);
/** @copydoc nk_dot_u8x16_state_haswell_t */
NK_INTERNAL void nk_dot_u8x16_update_haswell(nk_dot_u8x16_state_haswell_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_u8x16_state_haswell_t */
NK_INTERNAL void nk_dot_u8x16_finalize_haswell( //
    nk_dot_u8x16_state_haswell_t const *state_a, nk_dot_u8x16_state_haswell_t const *state_b,
    nk_dot_u8x16_state_haswell_t const *state_c, nk_dot_u8x16_state_haswell_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_dot_f64 */
NK_PUBLIC void nk_dot_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_dot_f64c */
NK_PUBLIC void nk_dot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);
/** @copydoc nk_vdot_f64c */
NK_PUBLIC void nk_vdot_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result);

/** @copydoc nk_dot_f32 */
NK_PUBLIC void nk_dot_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_f32c */
NK_PUBLIC void nk_dot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_f32c */
NK_PUBLIC void nk_vdot_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_f16 */
NK_PUBLIC void nk_dot_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Running state for 512-bit dot accumulation over f64 scalars on Skylake.
 */
typedef struct nk_dot_f64x8_state_skylake_t nk_dot_f64x8_state_skylake_t;
/** @copydoc nk_dot_f64x8_state_skylake_t */
NK_INTERNAL void nk_dot_f64x8_init_skylake(nk_dot_f64x8_state_skylake_t *state);
/** @copydoc nk_dot_f64x8_state_skylake_t */
NK_INTERNAL void nk_dot_f64x8_update_skylake(nk_dot_f64x8_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f64x8_state_skylake_t */
NK_INTERNAL void nk_dot_f64x8_finalize_skylake( //
    nk_dot_f64x8_state_skylake_t const *state_a, nk_dot_f64x8_state_skylake_t const *state_b,
    nk_dot_f64x8_state_skylake_t const *state_c, nk_dot_f64x8_state_skylake_t const *state_d,
    nk_size_t total_dimensions, nk_b256_vec_t *result);

/**
 *  @brief Running state for f32 dot accumulation using f64 for high precision on Skylake.
 *  Processes 8 f32 values at a time, upcasting to f64.
 */
typedef struct nk_dot_f32x8_state_skylake_t nk_dot_f32x8_state_skylake_t;
/** @copydoc nk_dot_f32x8_state_skylake_t */
NK_INTERNAL void nk_dot_f32x8_init_skylake(nk_dot_f32x8_state_skylake_t *state);
/** @copydoc nk_dot_f32x8_state_skylake_t */
NK_INTERNAL void nk_dot_f32x8_update_skylake(nk_dot_f32x8_state_skylake_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_f32x8_state_skylake_t */
NK_INTERNAL void nk_dot_f32x8_finalize_skylake( //
    nk_dot_f32x8_state_skylake_t const *state_a, nk_dot_f32x8_state_skylake_t const *state_b,
    nk_dot_f32x8_state_skylake_t const *state_c, nk_dot_f32x8_state_skylake_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 128-bit dot accumulation over e4m3 scalars on Skylake.
 *  @note Alias of nk_dot_through_f32_state_skylake_t_
 */
typedef struct nk_dot_through_f32_state_skylake_t_ nk_dot_e4m3x16_state_skylake_t;

/**
 *  @brief Running state for 128-bit dot accumulation over e5m2 scalars on Skylake.
 *  @note Alias of nk_dot_through_f32_state_skylake_t_
 */
typedef struct nk_dot_through_f32_state_skylake_t_ nk_dot_e5m2x16_state_skylake_t;
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICE
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_dot_u8 */
NK_PUBLIC void nk_dot_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Running state for 64-element dot accumulation over i8 scalars on Ice Lake.
 */
typedef struct nk_dot_i8x64_state_ice_t nk_dot_i8x64_state_ice_t;
/** @copydoc nk_dot_i8x64_state_ice_t */
NK_INTERNAL void nk_dot_i8x64_init_ice(nk_dot_i8x64_state_ice_t *state);
/** @copydoc nk_dot_i8x64_state_ice_t */
NK_INTERNAL void nk_dot_i8x64_update_ice(nk_dot_i8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                         nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i8x64_state_ice_t */
NK_INTERNAL void nk_dot_i8x64_finalize_ice( //
    nk_dot_i8x64_state_ice_t const *state_a, nk_dot_i8x64_state_ice_t const *state_b,
    nk_dot_i8x64_state_ice_t const *state_c, nk_dot_i8x64_state_ice_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *results);

/**
 *  @brief Running state for 64-element dot accumulation over u8 scalars on Ice Lake.
 */
typedef struct nk_dot_u8x64_state_ice_t nk_dot_u8x64_state_ice_t;
/** @copydoc nk_dot_u8x64_state_ice_t */
NK_INTERNAL void nk_dot_u8x64_init_ice(nk_dot_u8x64_state_ice_t *state);
/** @copydoc nk_dot_u8x64_state_ice_t */
NK_INTERNAL void nk_dot_u8x64_update_ice(nk_dot_u8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                         nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_u8x64_state_ice_t */
NK_INTERNAL void nk_dot_u8x64_finalize_ice( //
    nk_dot_u8x64_state_ice_t const *state_a, nk_dot_u8x64_state_ice_t const *state_b,
    nk_dot_u8x64_state_ice_t const *state_c, nk_dot_u8x64_state_ice_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result);

#endif // NK_TARGET_ICE

#if NK_TARGET_GENOA
/** @copydoc nk_dot_bf16 */
NK_PUBLIC void nk_dot_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_bf16c */
NK_PUBLIC void nk_dot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);
/** @copydoc nk_vdot_bf16c */
NK_PUBLIC void nk_vdot_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result);

/** @copydoc nk_dot_e4m3 */
NK_PUBLIC void nk_dot_e4m3_genoa(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_dot_e5m2 */
NK_PUBLIC void nk_dot_e5m2_genoa(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Running state for 512-bit dot accumulation over bf16 scalars on Genoa.
 */
typedef struct nk_dot_bf16x32_state_genoa_t nk_dot_bf16x32_state_genoa_t;
/** @copydoc nk_dot_bf16x32_state_genoa_t */
NK_INTERNAL void nk_dot_bf16x32_init_genoa(nk_dot_bf16x32_state_genoa_t *state);
/** @copydoc nk_dot_bf16x32_state_genoa_t */
NK_INTERNAL void nk_dot_bf16x32_update_genoa(nk_dot_bf16x32_state_genoa_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                             nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_bf16x32_state_genoa_t */
NK_INTERNAL void nk_dot_bf16x32_finalize_genoa( //
    nk_dot_bf16x32_state_genoa_t const *state_a, nk_dot_bf16x32_state_genoa_t const *state_b,
    nk_dot_bf16x32_state_genoa_t const *state_c, nk_dot_bf16x32_state_genoa_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

/**
 *  @brief Running state for 512-bit dot accumulation over e4m3 scalars on Genoa.
 *  @note Alias of nk_dot_through_bf16_state_genoa_t_
 */
typedef struct nk_dot_through_bf16_state_genoa_t_ nk_dot_e4m3x32_state_genoa_t;

/**
 *  @brief Running state for 512-bit dot accumulation over e5m2 scalars on Genoa.
 *  @note Alias of nk_dot_through_bf16_state_genoa_t_
 */
typedef struct nk_dot_through_bf16_state_genoa_t_ nk_dot_e5m2x32_state_genoa_t;

/**
 *  @brief Running state for 512-bit dot accumulation over e3m2 scalars on Genoa.
 *  @note Alias of nk_dot_through_bf16_state_genoa_t_
 */
typedef struct nk_dot_through_bf16_state_genoa_t_ nk_dot_e3m2x32_state_genoa_t;

/**
 *  @brief Running state for 512-bit dot accumulation over e2m3 scalars on Genoa.
 *  @note Alias of nk_dot_through_bf16_state_genoa_t_
 */
typedef struct nk_dot_through_bf16_state_genoa_t_ nk_dot_e2m3x32_state_genoa_t;

#endif // NK_TARGET_GENOA

#if NK_TARGET_SIERRA

/** @copydoc nk_dot_i8 */
NK_PUBLIC void nk_dot_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result);

/**
 *  @brief Running state for 32-element dot accumulation over i8 scalars on Sierra.
 */
typedef struct nk_dot_i8x32_state_sierra_t nk_dot_i8x32_state_sierra_t;
/** @copydoc nk_dot_i8x32_state_sierra_t */
NK_INTERNAL void nk_dot_i8x32_init_sierra(nk_dot_i8x32_state_sierra_t *state);
/** @copydoc nk_dot_i8x32_state_sierra_t */
NK_INTERNAL void nk_dot_i8x32_update_sierra(nk_dot_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions);
/** @copydoc nk_dot_i8x32_state_sierra_t */
NK_INTERNAL void nk_dot_i8x32_finalize_sierra( //
    nk_dot_i8x32_state_sierra_t const *state_a, nk_dot_i8x32_state_sierra_t const *state_b,
    nk_dot_i8x32_state_sierra_t const *state_c, nk_dot_i8x32_state_sierra_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result);

#endif // NK_TARGET_SIERRA

/**
 *  @brief  Returns the output dtype for dot products.
 */
NK_INTERNAL nk_dtype_t nk_dot_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_e5m2_k: return nk_f32_k;
    case nk_e2m3_k: return nk_f32_k;
    case nk_e3m2_k: return nk_f32_k;
    case nk_i8_k: return nk_i32_k;
    case nk_u8_k: return nk_u32_k;
    case nk_i4_k: return nk_i32_k;
    case nk_u4_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

#include "numkong/dot/serial.h"
#include "numkong/dot/neon.h"
#include "numkong/dot/neonsdot.h"
#include "numkong/dot/neonhalf.h"
#include "numkong/dot/neonfhm.h"
#include "numkong/dot/neonbfdot.h"
#include "numkong/dot/sve.h"
#include "numkong/dot/svehalf.h"
#include "numkong/dot/haswell.h"
#include "numkong/dot/skylake.h"
#include "numkong/dot/ice.h"
#include "numkong/dot/genoa.h"
#include "numkong/dot/sierra.h"
#include "numkong/dot/spacemit.h"
#include "numkong/dot/sifive.h"
#include "numkong/dot/xuantie.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_dot_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_dot_i8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_i8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_dot_i8_ice(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_i8_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_i8_haswell(a, b, n, result);
#else
    nk_dot_i8_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_dot_u8_spacemit(a, b, n, result);
#elif NK_TARGET_NEONSDOT
    nk_dot_u8_neonsdot(a, b, n, result);
#elif NK_TARGET_ICE
    nk_dot_u8_ice(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_u8_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_u8_haswell(a, b, n, result);
#else
    nk_dot_u8_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_i4(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
#if NK_TARGET_ICE
    nk_dot_i4_ice(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_i4_spacemit(a, b, n, result);
#else
    nk_dot_i4_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_u4(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_ICE
    nk_dot_u4_ice(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_u4_spacemit(a, b, n, result);
#else
    nk_dot_u4_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SIFIVE
    nk_dot_f16_sifive(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_f16_spacemit(a, b, n, result);
#elif NK_TARGET_SVEHALF
    nk_dot_f16_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_f16_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_dot_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f16_haswell(a, b, n, result);
#else
    nk_dot_f16_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_bf16_genoa(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_bf16_spacemit(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_bf16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_bf16_haswell(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_bf16_neonbfdot(a, b, n, result);
#else
    nk_dot_bf16_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_e4m3(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e4m3_genoa(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_e4m3_spacemit(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e4m3_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e4m3_haswell(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e4m3_neon(a, b, n, result);
#else
    nk_dot_e4m3_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_e5m2(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e5m2_genoa(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_e5m2_spacemit(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e5m2_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e5m2_haswell(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e5m2_neon(a, b, n, result);
#else
    nk_dot_e5m2_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_e2m3(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e2m3_genoa(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_e2m3_spacemit(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e2m3_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e2m3_haswell(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_e2m3_neonfhm(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e2m3_neon(a, b, n, result);
#else
    nk_dot_e2m3_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_e3m2(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_dot_e3m2_genoa(a, b, n, result);
#elif NK_TARGET_SPACEMIT
    nk_dot_e3m2_spacemit(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_e3m2_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_e3m2_haswell(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_e3m2_neonfhm(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_e3m2_neon(a, b, n, result);
#else
    nk_dot_e3m2_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_SPACEMIT
    nk_dot_f32_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_dot_f32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f32_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f32_haswell(a, b, n, result);
#else
    nk_dot_f32_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SPACEMIT
    nk_dot_f64_spacemit(a, b, n, result);
#elif NK_TARGET_SVE
    nk_dot_f64_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f64_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f64_haswell(a, b, n, result);
#else
    nk_dot_f64_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVEHALF
    nk_dot_f16c_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_dot_f16c_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_dot_f16c_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f16c_haswell(a, b, n, result);
#else
    nk_dot_f16c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_GENOA
    nk_dot_bf16c_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_dot_bf16c_neonbfdot(a, b, n, result);
#else
    nk_dot_bf16c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVE
    nk_dot_f32c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f32c_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f32c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_dot_f32c_haswell(a, b, n, result);
#else
    nk_dot_f32c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_dot_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_dot_f64c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_dot_f64c_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_dot_f64c_skylake(a, b, n, result);
#else
    nk_dot_f64c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_vdot_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVEHALF
    nk_vdot_f16c_svehalf(a, b, n, result);
#elif NK_TARGET_NEONFHM
    nk_vdot_f16c_neonfhm(a, b, n, result);
#elif NK_TARGET_NEONHALF
    nk_vdot_f16c_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_f16c_haswell(a, b, n, result);
#else
    nk_vdot_f16c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_vdot_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_GENOA
    nk_vdot_bf16c_genoa(a, b, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_vdot_bf16c_neonbfdot(a, b, n, result);
#else
    nk_vdot_bf16c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_vdot_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_TARGET_SVE
    nk_vdot_f32c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_vdot_f32c_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_vdot_f32c_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_vdot_f32c_haswell(a, b, n, result);
#else
    nk_vdot_f32c_serial(a, b, n, result);
#endif
}
NK_PUBLIC void nk_vdot_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_TARGET_SVE
    nk_vdot_f64c_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_vdot_f64c_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_vdot_f64c_skylake(a, b, n, result);
#else
    nk_vdot_f64c_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
