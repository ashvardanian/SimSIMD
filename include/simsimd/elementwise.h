/**
 *  @brief SIMD-accelerated mixed-precision element-wise operations.
 *  @file include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date October 16, 2024
 *
 *  Contains following element-wise operations:
 *
 *  - Scale (Multiply) with shift: result[i] = alpha * a[i] + beta
 *  - Sum (Add): result[i] = a[i] + b[i]
 *  - Weighted Sum (WSum): result[i] = alpha * a[i] + beta * b[i]
 *  - FMA (Fused Multiply-Add): result[i] = alpha * a[i] * b[i] + beta * c[i]
 *
 *  Beyond their obvious usecases, those can be reused for vector-scalar math and other operations:
 *
 *  - Scale with beta = 0 for a pure multiply.
 *  - Sum is equivalent to WSum with alpha = beta = 1.
 *  - Average is WSum with alpha = beta = 0.5.
 *  - Elementwise multiply is FMA with beta = 0.
 *
 *  For datatypes:
 *
 *  - 64-bit IEEE floating point numbers × 64-bit scales
 *  - 32-bit IEEE floating point numbers × 32-bit scales
 *  - 16-bit IEEE floating point numbers × 32-bit scales
 *  - 16-bit brain floating point numbers × 32-bit scales
 *  - 8-bit signed and unsigned integers × 32-bit scales
 *  - 16-bit signed and unsigned integers × 32-bit scales
 *  - 32-bit signed and unsigned integers × 64-bit scales
 *  - 64-bit signed and unsigned integers × 64-bit scales
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
 *
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  FP16 conversions (VCVTPH2PS/VCVTPS2PH) are used for f16 scale/sum/wsum/fma operations, converting
 *  to f32 for arithmetic then back. The 6-7 cycle latency is amortized over vector-width elements.
 *  Saturating integer adds (VPADDSW/VPADDUSW) provide overflow protection for i16/u16 sums without
 *  branching. FMA (VFMADD231PS) is the workhorse for scale (alpha*x+beta) and wsum (alpha*a+beta*b).
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm512_cvtph_ps         VCVTPH2PS (ZMM, YMM)            7c @ p0+p5  6c @ p12+p23
 *      _mm512_cvtps_ph         VCVTPS2PH (YMM, ZMM, I8)        7c @ p0+p5  7c @ p12+p23
 *      _mm256_adds_epi16       VPADDSW (YMM, YMM, YMM)         1c @ p01    N/A
 *      _mm256_adds_epu16       VPADDUSW (YMM, YMM, YMM)        1c @ p01    N/A
 *      _mm512_fpclass_ps_mask  VFPCLASSPS (K, ZMM, I8)         3c @ p5     5c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  On ARM, i8/u8 elementwise operations convert to f16 intermediates using FCVT to maintain high
 *  vector throughput (8 elements per 128-bit register vs 4 for f32). Saturating adds (SQADD/UQADD)
 *  handle integer overflow. FMLA provides fused multiply-add for floating-point scale/wsum/fma.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vqaddq_s16              SQADD (vec)     3c @ V0123      2c @ V0123      2c @ V0123
 *      vqaddq_u16              UQADD (vec)     3c @ V0123      2c @ V0123      2c @ V0123
 *      vcvtq_f32_s32           SCVTF (vec)     3c @ V0123      3c @ V01        3c @ V01
 *      vcvtaq_s32_f32          FCVTAS (vec)    3c @ V0123      3c @ V01        3c @ V01
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_ELEMENTWISE_H
#define NK_ELEMENTWISE_H

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Element-wise scale with shift: result[i] = alpha * a[i] + beta.
 *
 *  @param[in] a The input vector.
 *  @param[in] n The number of elements in the vector.
 *  @param[in] alpha Pointer to the scaling factor (type depends on input precision).
 *  @param[in] beta Pointer to the shift (bias) value (type depends on input precision).
 *  @param[out] result The output vector.
 */
NK_DYNAMIC void nk_scale_f64(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                             nk_f64_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_f32(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                             nk_f32_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_f16(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                             nk_f16_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_bf16(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                              nk_bf16_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_i8(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                            nk_i8_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_u8(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                            nk_u8_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_i16(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                             nk_i16_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_u16(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                             nk_u16_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_i32(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                             nk_i32_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_u32(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                             nk_u32_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_i64(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                             nk_i64_t *result);
/** @copydoc nk_scale_f64 */
NK_DYNAMIC void nk_scale_u64(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                             nk_u64_t *result);

/**
 *  @brief Element-wise sum: result[i] = a[i] + b[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[out] result The output vector.
 */
NK_DYNAMIC void nk_sum_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_i16(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_i32(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_i64(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result);
/** @copydoc nk_sum_f64 */
NK_DYNAMIC void nk_sum_u64(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result);

/**
 *  @brief Weighted sum: result[i] = alpha * a[i] + beta * b[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[in] alpha Pointer to the first weight (type depends on input precision).
 *  @param[in] beta Pointer to the second weight (type depends on input precision).
 *  @param[out] result The output vector.
 */
NK_DYNAMIC void nk_wsum_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                            nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_wsum_f64 */
NK_DYNAMIC void nk_wsum_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                            nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_wsum_f64 */
NK_DYNAMIC void nk_wsum_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                            nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_wsum_f64 */
NK_DYNAMIC void nk_wsum_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                             nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_wsum_f64 */
NK_DYNAMIC void nk_wsum_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                           nk_i8_t *result);
/** @copydoc nk_wsum_f64 */
NK_DYNAMIC void nk_wsum_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                           nk_u8_t *result);

/**
 *  @brief Fused multiply-add: result[i] = alpha * a[i] * b[i] + beta * c[i].
 *
 *  @param[in] a The first input vector.
 *  @param[in] b The second input vector.
 *  @param[in] c The third input vector.
 *  @param[in] n The number of elements in the vectors.
 *  @param[in] alpha Pointer to the scaling factor for a[i] * b[i] (type depends on input precision).
 *  @param[in] beta Pointer to the scaling factor for c[i] (type depends on input precision).
 *  @param[out] result The output vector.
 */
NK_DYNAMIC void nk_fma_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                            nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_i8(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_u8(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_i16(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_u16(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_i32(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_u32(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_u32_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_i64(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_i64_t *result);
/** @copydoc nk_fma_f64 */
NK_DYNAMIC void nk_fma_u64(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_u64_t *result);

/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_f64_serial(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                   nk_f64_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_f32_serial(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_f32_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_f16_serial(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_f16_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_bf16_serial(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_bf16_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_i8_serial(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                  nk_i8_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_u8_serial(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                  nk_u8_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_i16_serial(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i16_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_u16_serial(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u16_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_i32_serial(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                   nk_i32_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_u32_serial(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                   nk_u32_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_i64_serial(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                   nk_i64_t *result);
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_u64_serial(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                   nk_u64_t *result);

/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result);
/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result);

/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                  nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                 nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                 nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                  nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                  nk_f64_t const *beta, nk_u32_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                  nk_f64_t const *beta, nk_i64_t *result);
/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                  nk_f64_t const *beta, nk_u64_t *result);

/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                 nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_i8_serial(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_i16_serial(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_i32_serial(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                                 nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                                 nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n,
                                 nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result);
/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n,
                                 nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result);

/*  Double-precision serial backends for select numeric types.
 *  For single-precision computation check out the "*_serial" counterparts.
 */
/** @copydoc nk_sum_f32 */
NK_PUBLIC void nk_sum_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_f16 */
NK_PUBLIC void nk_sum_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);
/** @copydoc nk_sum_bf16 */
NK_PUBLIC void nk_sum_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);

/** @copydoc nk_scale_f32 */
NK_PUBLIC void nk_scale_f32_accurate(nk_f32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_f32_t *result);
/** @copydoc nk_scale_f16 */
NK_PUBLIC void nk_scale_f16_accurate(nk_f16_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_f16_t *result);
/** @copydoc nk_scale_bf16 */
NK_PUBLIC void nk_scale_bf16_accurate(nk_bf16_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_bf16_t *result);
/** @copydoc nk_scale_i8 */
NK_PUBLIC void nk_scale_i8_accurate(nk_i8_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i8_t *result);
/** @copydoc nk_scale_u8 */
NK_PUBLIC void nk_scale_u8_accurate(nk_u8_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u8_t *result);

/** @copydoc nk_wsum_f32 */
NK_PUBLIC void nk_wsum_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                    nk_f64_t const *beta, nk_f32_t *result);
/** @copydoc nk_wsum_f16 */
NK_PUBLIC void nk_wsum_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                    nk_f64_t const *beta, nk_f16_t *result);
/** @copydoc nk_wsum_bf16 */
NK_PUBLIC void nk_wsum_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                     nk_f64_t const *beta, nk_bf16_t *result);
/** @copydoc nk_wsum_i8 */
NK_PUBLIC void nk_wsum_i8_accurate(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                   nk_f64_t const *beta, nk_i8_t *result);
/** @copydoc nk_wsum_u8 */
NK_PUBLIC void nk_wsum_u8_accurate(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                   nk_f64_t const *beta, nk_u8_t *result);

/** @copydoc nk_fma_f32 */
NK_PUBLIC void nk_fma_f32_accurate(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_f16 */
NK_PUBLIC void nk_fma_f16_accurate(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_bf16 */
NK_PUBLIC void nk_fma_bf16_accurate(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f64_t const *alpha, nk_f64_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_i8 */
NK_PUBLIC void nk_fma_i8_accurate(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_u8 */
NK_PUBLIC void nk_fma_u8_accurate(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_u8_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_scale_f32 */
NK_PUBLIC void nk_scale_f32_neon(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_f32_t *result);
/** @copydoc nk_scale_i16 */
NK_PUBLIC void nk_scale_i16_neon(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_i16_t *result);
/** @copydoc nk_scale_u16 */
NK_PUBLIC void nk_scale_u16_neon(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_u16_t *result);
/** @copydoc nk_scale_i32 */
NK_PUBLIC void nk_scale_i32_neon(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i32_t *result);
/** @copydoc nk_scale_u32 */
NK_PUBLIC void nk_scale_u32_neon(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u32_t *result);
/** @copydoc nk_scale_i64 */
NK_PUBLIC void nk_scale_i64_neon(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i64_t *result);
/** @copydoc nk_scale_u64 */
NK_PUBLIC void nk_scale_u64_neon(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u64_t *result);

/** @copydoc nk_sum_f32 */
NK_PUBLIC void nk_sum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_i16 */
NK_PUBLIC void nk_sum_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result);
/** @copydoc nk_sum_u16 */
NK_PUBLIC void nk_sum_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result);
/** @copydoc nk_sum_i32 */
NK_PUBLIC void nk_sum_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_sum_u32 */
NK_PUBLIC void nk_sum_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_sum_i64 */
NK_PUBLIC void nk_sum_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result);
/** @copydoc nk_sum_u64 */
NK_PUBLIC void nk_sum_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result);

/** @copydoc nk_wsum_f32 */
NK_PUBLIC void nk_wsum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                nk_f32_t const *beta, nk_f32_t *result);

/** @copydoc nk_fma_f32 */
NK_PUBLIC void nk_fma_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                               nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_i16 */
NK_PUBLIC void nk_fma_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                               nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_fma_u16 */
NK_PUBLIC void nk_fma_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                               nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_fma_i32 */
NK_PUBLIC void nk_fma_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                               nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_fma_u32 */
NK_PUBLIC void nk_fma_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                               nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result);
/** @copydoc nk_fma_i64 */
NK_PUBLIC void nk_fma_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n,
                               nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result);
/** @copydoc nk_fma_u64 */
NK_PUBLIC void nk_fma_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n,
                               nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEON_BF16
/** @copydoc nk_sum_bf16 */
NK_PUBLIC void nk_sum_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);
/** @copydoc nk_scale_bf16 */
NK_PUBLIC void nk_scale_bf16_neon(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                  nk_bf16_t *result);
/** @copydoc nk_wsum_bf16 */
NK_PUBLIC void nk_wsum_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                 nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_bf16 */
NK_PUBLIC void nk_fma_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result);
#endif // NK_TARGET_NEON_BF16

#if NK_TARGET_NEON_F16
/** @copydoc nk_sum_f16 */
NK_PUBLIC void nk_sum_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);
/** @copydoc nk_scale_f16 */
NK_PUBLIC void nk_scale_f16_neon(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_f16_t *result);
/** @copydoc nk_wsum_f16 */
NK_PUBLIC void nk_wsum_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_f16 */
NK_PUBLIC void nk_fma_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                               nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result);

/** @copydoc nk_sum_i8 */
NK_PUBLIC void nk_sum_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result);
/** @copydoc nk_sum_u8 */
NK_PUBLIC void nk_sum_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result);
/** @copydoc nk_scale_i8 */
NK_PUBLIC void nk_scale_i8_neon(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                nk_i8_t *result);
/** @copydoc nk_scale_u8 */
NK_PUBLIC void nk_scale_u8_neon(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                nk_u8_t *result);
/** @copydoc nk_wsum_i8 */
NK_PUBLIC void nk_wsum_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                               nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_wsum_u8 */
NK_PUBLIC void nk_wsum_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                               nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_fma_i8 */
NK_PUBLIC void nk_fma_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                              nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_u8 */
NK_PUBLIC void nk_fma_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                              nk_f32_t const *beta, nk_u8_t *result);
#endif // NK_TARGET_NEON_F16

#if NK_TARGET_HASWELL
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_f64_haswell(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_f64_t *result);
/** @copydoc nk_scale_f32 */
NK_PUBLIC void nk_scale_f32_haswell(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f32_t *result);
/** @copydoc nk_scale_f16 */
NK_PUBLIC void nk_scale_f16_haswell(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f16_t *result);
/** @copydoc nk_scale_bf16 */
NK_PUBLIC void nk_scale_bf16_haswell(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_bf16_t *result);
/** @copydoc nk_scale_i8 */
NK_PUBLIC void nk_scale_i8_haswell(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i8_t *result);
/** @copydoc nk_scale_u8 */
NK_PUBLIC void nk_scale_u8_haswell(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u8_t *result);
/** @copydoc nk_scale_i16 */
NK_PUBLIC void nk_scale_i16_haswell(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i16_t *result);
/** @copydoc nk_scale_u16 */
NK_PUBLIC void nk_scale_u16_haswell(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u16_t *result);
/** @copydoc nk_scale_i32 */
NK_PUBLIC void nk_scale_i32_haswell(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i32_t *result);
/** @copydoc nk_scale_u32 */
NK_PUBLIC void nk_scale_u32_haswell(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u32_t *result);

/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_sum_f32 */
NK_PUBLIC void nk_sum_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_f16 */
NK_PUBLIC void nk_sum_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);
/** @copydoc nk_sum_bf16 */
NK_PUBLIC void nk_sum_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);
/** @copydoc nk_sum_i8 */
NK_PUBLIC void nk_sum_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result);
/** @copydoc nk_sum_u8 */
NK_PUBLIC void nk_sum_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result);
/** @copydoc nk_sum_i16 */
NK_PUBLIC void nk_sum_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result);
/** @copydoc nk_sum_u16 */
NK_PUBLIC void nk_sum_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result);
/** @copydoc nk_sum_i32 */
NK_PUBLIC void nk_sum_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_sum_u32 */
NK_PUBLIC void nk_sum_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result);

/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                   nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_wsum_f32 */
NK_PUBLIC void nk_wsum_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_wsum_f16 */
NK_PUBLIC void nk_wsum_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_wsum_bf16 */
NK_PUBLIC void nk_wsum_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                    nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_wsum_i8 */
NK_PUBLIC void nk_wsum_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_wsum_u8 */
NK_PUBLIC void nk_wsum_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                  nk_f32_t const *beta, nk_u8_t *result);

/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_fma_f32 */
NK_PUBLIC void nk_fma_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_f16 */
NK_PUBLIC void nk_fma_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_bf16 */
NK_PUBLIC void nk_fma_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_i8 */
NK_PUBLIC void nk_fma_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_u8 */
NK_PUBLIC void nk_fma_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_fma_i16 */
NK_PUBLIC void nk_fma_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_fma_u16 */
NK_PUBLIC void nk_fma_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_fma_i32 */
NK_PUBLIC void nk_fma_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_fma_u32 */
NK_PUBLIC void nk_fma_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_scale_f64 */
NK_PUBLIC void nk_scale_f64_skylake(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_f64_t *result);
/** @copydoc nk_scale_f32 */
NK_PUBLIC void nk_scale_f32_skylake(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f32_t *result);
/** @copydoc nk_scale_f16 */
NK_PUBLIC void nk_scale_f16_skylake(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f16_t *result);
/** @copydoc nk_scale_bf16 */
NK_PUBLIC void nk_scale_bf16_skylake(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_bf16_t *result);
/** @copydoc nk_scale_i8 */
NK_PUBLIC void nk_scale_i8_skylake(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i8_t *result);
/** @copydoc nk_scale_u8 */
NK_PUBLIC void nk_scale_u8_skylake(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u8_t *result);
/** @copydoc nk_scale_i16 */
NK_PUBLIC void nk_scale_i16_skylake(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i16_t *result);
/** @copydoc nk_scale_u16 */
NK_PUBLIC void nk_scale_u16_skylake(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u16_t *result);
/** @copydoc nk_scale_i32 */
NK_PUBLIC void nk_scale_i32_skylake(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i32_t *result);
/** @copydoc nk_scale_u32 */
NK_PUBLIC void nk_scale_u32_skylake(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u32_t *result);
/** @copydoc nk_scale_i64 */
NK_PUBLIC void nk_scale_i64_skylake(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i64_t *result);
/** @copydoc nk_scale_u64 */
NK_PUBLIC void nk_scale_u64_skylake(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u64_t *result);

/** @copydoc nk_sum_f64 */
NK_PUBLIC void nk_sum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_sum_f32 */
NK_PUBLIC void nk_sum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_sum_bf16 */
NK_PUBLIC void nk_sum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result);

/** @copydoc nk_wsum_f64 */
NK_PUBLIC void nk_wsum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                   nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_wsum_f32 */
NK_PUBLIC void nk_wsum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_wsum_bf16 */
NK_PUBLIC void nk_wsum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                    nk_f32_t const *beta, nk_bf16_t *result);

/** @copydoc nk_fma_f64 */
NK_PUBLIC void nk_fma_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result);
/** @copydoc nk_fma_f32 */
NK_PUBLIC void nk_fma_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result);
/** @copydoc nk_fma_f16 */
NK_PUBLIC void nk_fma_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_bf16 */
NK_PUBLIC void nk_fma_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result);
/** @copydoc nk_fma_i8 */
NK_PUBLIC void nk_fma_i8_skylake(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_u8 */
NK_PUBLIC void nk_fma_u8_skylake(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                 nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result);
/** @copydoc nk_fma_i16 */
NK_PUBLIC void nk_fma_i16_skylake(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result);
/** @copydoc nk_fma_u16 */
NK_PUBLIC void nk_fma_u16_skylake(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result);
/** @copydoc nk_fma_i32 */
NK_PUBLIC void nk_fma_i32_skylake(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result);
/** @copydoc nk_fma_u32 */
NK_PUBLIC void nk_fma_u32_skylake(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result);
/** @copydoc nk_fma_i64 */
NK_PUBLIC void nk_fma_i64_skylake(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result);
/** @copydoc nk_fma_u64 */
NK_PUBLIC void nk_fma_u64_skylake(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n,
                                  nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICE
/** @copydoc nk_sum_i8 */
NK_PUBLIC void nk_sum_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result);
/** @copydoc nk_sum_u8 */
NK_PUBLIC void nk_sum_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result);
/** @copydoc nk_sum_i16 */
NK_PUBLIC void nk_sum_i16_ice(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result);
/** @copydoc nk_sum_u16 */
NK_PUBLIC void nk_sum_u16_ice(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result);
/** @copydoc nk_sum_i32 */
NK_PUBLIC void nk_sum_i32_ice(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result);
/** @copydoc nk_sum_u32 */
NK_PUBLIC void nk_sum_u32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_sum_i64 */
NK_PUBLIC void nk_sum_i64_ice(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result);
/** @copydoc nk_sum_u64 */
NK_PUBLIC void nk_sum_u64_ice(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result);
#endif // NK_TARGET_ICE

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_scale_f16 */
NK_PUBLIC void nk_scale_f16_sapphire(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f16_t *result);
/** @copydoc nk_scale_i8 */
NK_PUBLIC void nk_scale_i8_sapphire(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i8_t *result);
/** @copydoc nk_scale_u8 */
NK_PUBLIC void nk_scale_u8_sapphire(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u8_t *result);

/** @copydoc nk_sum_f16 */
NK_PUBLIC void nk_sum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result);

/** @copydoc nk_wsum_f16 */
NK_PUBLIC void nk_wsum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                    nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_wsum_i8 */
NK_PUBLIC void nk_wsum_i8_sapphire(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_wsum_u8 */
NK_PUBLIC void nk_wsum_u8_sapphire(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                   nk_f32_t const *beta, nk_u8_t *result);

/** @copydoc nk_fma_f16 */
NK_PUBLIC void nk_fma_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result);
/** @copydoc nk_fma_i8 */
NK_PUBLIC void nk_fma_i8_sapphire(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result);
/** @copydoc nk_fma_u8 */
NK_PUBLIC void nk_fma_u8_sapphire(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result);
#endif // NK_TARGET_SAPPHIRE

#define NK_MAKE_SCALE(name, input_type, accumulator_type, load_and_convert, convert_and_store) \
    NK_PUBLIC void nk_scale_##input_type##_##name(                                             \
        nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,     \
        nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) {                  \
        nk_##accumulator_type##_t alpha_val = *alpha;                                          \
        nk_##accumulator_type##_t beta_val = *beta;                                            \
        nk_##accumulator_type##_t ai, sum;                                                     \
        for (nk_size_t i = 0; i != n; ++i) {                                                   \
            load_and_convert(a + i, &ai);                                                      \
            sum = (nk_##accumulator_type##_t)(alpha_val * ai + beta_val);                      \
            convert_and_store(&sum, result + i);                                               \
        }                                                                                      \
    }
#define NK_MAKE_SUM(name, input_type, accumulator_type, load_and_convert, convert_and_store)                \
    NK_PUBLIC void nk_sum_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##input_type##_t *result) {                 \
        nk_##accumulator_type##_t ai, bi, sum;                                                              \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            sum = ai + bi;                                                                                  \
            convert_and_store(&sum, result + i);                                                            \
        }                                                                                                   \
    }

#define NK_MAKE_WSUM(name, input_type, accumulator_type, load_and_convert, convert_and_store)                          \
    NK_PUBLIC void nk_wsum_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b,           \
                                                 nk_size_t n, nk_##accumulator_type##_t const *alpha,                  \
                                                 nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                  \
        nk_##accumulator_type##_t beta_val = *beta;                                                                    \
        nk_##accumulator_type##_t ai, bi, ai_scaled, bi_scaled, sum;                                                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                           \
            load_and_convert(a + i, &ai);                                                                              \
            load_and_convert(b + i, &bi);                                                                              \
            ai_scaled = ai * alpha_val;                                                                                \
            bi_scaled = bi * beta_val;                                                                                 \
            sum = ai_scaled + bi_scaled;                                                                               \
            convert_and_store(&sum, result + i);                                                                       \
        }                                                                                                              \
    }

#define NK_MAKE_FMA(name, input_type, accumulator_type, load_and_convert, convert_and_store)                          \
    NK_PUBLIC void nk_fma_##input_type##_##name(                                                                      \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,        \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                 \
        nk_##accumulator_type##_t beta_val = *beta;                                                                   \
        nk_##accumulator_type##_t ai, bi, ci, abi_scaled, ci_scaled, sum;                                             \
        for (nk_size_t i = 0; i != n; ++i) {                                                                          \
            load_and_convert(a + i, &ai);                                                                             \
            load_and_convert(b + i, &bi);                                                                             \
            load_and_convert(c + i, &ci);                                                                             \
            abi_scaled = ai * bi * alpha_val;                                                                         \
            ci_scaled = ci * beta_val;                                                                                \
            sum = abi_scaled + ci_scaled;                                                                             \
            convert_and_store(&sum, result + i);                                                                      \
        }                                                                                                             \
    }

NK_MAKE_SUM(serial, f64, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_sum_f64_serial
NK_MAKE_SUM(serial, f32, f32, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_sum_f32_serial
NK_MAKE_SUM(serial, f16, f32, nk_f16_to_f32, nk_f32_to_f16)         // nk_sum_f16_serial
NK_MAKE_SUM(serial, bf16, f32, nk_bf16_to_f32, nk_f32_to_bf16)      // nk_sum_bf16_serial
NK_MAKE_SUM(serial, i8, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_i8)      // nk_sum_i8_serial
NK_MAKE_SUM(serial, u8, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_u8)      // nk_sum_u8_serial
NK_MAKE_SUM(serial, i16, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_i16)    // nk_sum_i16_serial
NK_MAKE_SUM(serial, u16, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_u16)    // nk_sum_u16_serial
NK_MAKE_SUM(serial, i32, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_i32)    // nk_sum_i32_serial
NK_MAKE_SUM(serial, u32, i64, NK_ASSIGN_FROM_TO, _nk_i64_to_u32)    // nk_sum_u32_serial
NK_MAKE_SUM(serial, i64, i64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_sum_i64_serial
NK_MAKE_SUM(serial, u64, u64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_sum_u64_serial

NK_MAKE_SUM(accurate, f32, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_sum_f32_accurate
NK_MAKE_SUM(accurate, f16, f64, _nk_f16_to_f64, _nk_f64_to_f16)       // nk_sum_f16_accurate
NK_MAKE_SUM(accurate, bf16, f64, _nk_bf16_to_f64, _nk_f64_to_bf16)    // nk_sum_bf16_accurate

NK_MAKE_SCALE(serial, f64, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_scale_f64_serial
NK_MAKE_SCALE(serial, f32, f32, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_scale_f32_serial
NK_MAKE_SCALE(serial, f16, f32, nk_f16_to_f32, nk_f32_to_f16)         // nk_scale_f16_serial
NK_MAKE_SCALE(serial, bf16, f32, nk_bf16_to_f32, nk_f32_to_bf16)      // nk_scale_bf16_serial
NK_MAKE_SCALE(serial, i8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i8)      // nk_scale_i8_serial
NK_MAKE_SCALE(serial, u8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u8)      // nk_scale_u8_serial
NK_MAKE_SCALE(serial, i16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i16)    // nk_scale_i16_serial
NK_MAKE_SCALE(serial, u16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u16)    // nk_scale_u16_serial
NK_MAKE_SCALE(serial, i32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i32)    // nk_scale_i32_serial
NK_MAKE_SCALE(serial, u32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u32)    // nk_scale_u32_serial
NK_MAKE_SCALE(serial, i64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i64)    // nk_scale_i64_serial
NK_MAKE_SCALE(serial, u64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u64)    // nk_scale_u64_serial

NK_MAKE_SCALE(accurate, f32, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_scale_f32_accurate
NK_MAKE_SCALE(accurate, f16, f64, _nk_f16_to_f64, _nk_f64_to_f16)       // nk_scale_f16_accurate
NK_MAKE_SCALE(accurate, bf16, f64, _nk_bf16_to_f64, _nk_f64_to_bf16)    // nk_scale_bf16_accurate
NK_MAKE_SCALE(accurate, i8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i8)      // nk_scale_i8_accurate
NK_MAKE_SCALE(accurate, u8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u8)      // nk_scale_u8_accurate

NK_MAKE_WSUM(serial, f64, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_wsum_f64_serial
NK_MAKE_WSUM(serial, f32, f32, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_wsum_f32_serial
NK_MAKE_WSUM(serial, f16, f32, nk_f16_to_f32, nk_f32_to_f16)         // nk_wsum_f16_serial
NK_MAKE_WSUM(serial, bf16, f32, nk_bf16_to_f32, nk_f32_to_bf16)      // nk_wsum_bf16_serial
NK_MAKE_WSUM(serial, i8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i8)      // nk_wsum_i8_serial
NK_MAKE_WSUM(serial, u8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u8)      // nk_wsum_u8_serial
NK_MAKE_WSUM(serial, i16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i16)    // nk_wsum_i16_serial
NK_MAKE_WSUM(serial, u16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u16)    // nk_wsum_u16_serial
NK_MAKE_WSUM(serial, i32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i32)    // nk_wsum_i32_serial
NK_MAKE_WSUM(serial, u32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u32)    // nk_wsum_u32_serial
NK_MAKE_WSUM(serial, i64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i64)    // nk_wsum_i64_serial
NK_MAKE_WSUM(serial, u64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u64)    // nk_wsum_u64_serial

NK_MAKE_WSUM(accurate, f32, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_wsum_f32_accurate
NK_MAKE_WSUM(accurate, f16, f64, _nk_f16_to_f64, _nk_f64_to_f16)       // nk_wsum_f16_accurate
NK_MAKE_WSUM(accurate, bf16, f64, _nk_bf16_to_f64, _nk_f64_to_bf16)    // nk_wsum_bf16_accurate
NK_MAKE_WSUM(accurate, i8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i8)      // nk_wsum_i8_accurate
NK_MAKE_WSUM(accurate, u8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u8)      // nk_wsum_u8_accurate

NK_MAKE_FMA(serial, f64, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_fma_f64_serial
NK_MAKE_FMA(serial, f32, f32, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_fma_f32_serial
NK_MAKE_FMA(serial, f16, f32, nk_f16_to_f32, nk_f32_to_f16)         // nk_fma_f16_serial
NK_MAKE_FMA(serial, bf16, f32, nk_bf16_to_f32, nk_f32_to_bf16)      // nk_fma_bf16_serial
NK_MAKE_FMA(serial, i8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i8)      // nk_fma_i8_serial
NK_MAKE_FMA(serial, u8, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u8)      // nk_fma_u8_serial
NK_MAKE_FMA(serial, i16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_i16)    // nk_fma_i16_serial
NK_MAKE_FMA(serial, u16, f32, NK_ASSIGN_FROM_TO, _nk_f32_to_u16)    // nk_fma_u16_serial
NK_MAKE_FMA(serial, i32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i32)    // nk_fma_i32_serial
NK_MAKE_FMA(serial, u32, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u32)    // nk_fma_u32_serial
NK_MAKE_FMA(serial, i64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i64)    // nk_fma_i64_serial
NK_MAKE_FMA(serial, u64, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u64)    // nk_fma_u64_serial

NK_MAKE_FMA(accurate, f32, f64, NK_ASSIGN_FROM_TO, NK_ASSIGN_FROM_TO) // nk_fma_f32_accurate
NK_MAKE_FMA(accurate, f16, f64, _nk_f16_to_f64, _nk_f64_to_f16)       // nk_fma_f16_accurate
NK_MAKE_FMA(accurate, bf16, f64, _nk_bf16_to_f64, _nk_f64_to_bf16)    // nk_fma_bf16_accurate
NK_MAKE_FMA(accurate, i8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_i8)      // nk_fma_i8_accurate
NK_MAKE_FMA(accurate, u8, f64, NK_ASSIGN_FROM_TO, _nk_f64_to_u8)      // nk_fma_u8_accurate

#if _NK_TARGET_X86
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

NK_PUBLIC void nk_sum_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_scale_f32_haswell(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_wsum_f32_haswell(                    //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f32_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f32_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_f32_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_sum_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d result_f64x4 = _mm256_add_pd(a_f64x4, b_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_scale_f64_haswell(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_wsum_f64_haswell(                    //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f64_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_scale_f64_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_f64_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(b_f64x4, beta_f64x4, ab_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_sum_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32(a + i, &ai);
        nk_f16_to_f32(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_f16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_f16_haswell(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_f16_to_f32(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_f16(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_f16_haswell(                    //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f16_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_f16_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32(a + i, &ai);
        nk_f16_to_f32(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_f16(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _nk_bf16x8_to_f32x8_haswell(a_bf16x8);
        __m256 b_f32x8 = _nk_bf16x8_to_f32x8_haswell(b_bf16x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_bf16x8 = _nk_f32x8_to_bf16x8_haswell(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_bf16_haswell(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m256 a_f32x8 = _nk_bf16x8_to_f32x8_haswell(a_bf16x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_bf16x8 = _nk_f32x8_to_bf16x8_haswell(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_bf16_to_f32(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_bf16_haswell(                     //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_bf16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_bf16_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_bf16_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _nk_bf16x8_to_f32x8_haswell(a_bf16x8);
        __m256 b_f32x8 = _nk_bf16x8_to_f32x8_haswell(b_bf16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        __m128i result_bf16x8 = _nk_f32x8_to_bf16x8_haswell(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_f32_haswell(                           //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 c_f32x8 = _mm256_loadu_ps(c + i);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_fma_f64_haswell(                           //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d c_f64x4 = _mm256_loadu_pd(c + i);
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d abc_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, abc_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_fma_f16_haswell(                           //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m128i c_f16x8 = _mm_lddqu_si128((__m128i const *)(c + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 c_f32x8 = _mm256_cvtph_ps(c_f16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_f16_to_f32(a + i, &ai);
        nk_f16_to_f32(b + i, &bi);
        nk_f16_to_f32(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_f16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_bf16_haswell(                             //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m128i c_bf16x8 = _mm_lddqu_si128((__m128i const *)(c + i));
        __m256 a_f32x8 = _nk_bf16x8_to_f32x8_haswell(a_bf16x8);
        __m256 b_f32x8 = _nk_bf16x8_to_f32x8_haswell(b_bf16x8);
        __m256 c_f32x8 = _nk_bf16x8_to_f32x8_haswell(c_bf16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        __m128i result_bf16x8 = _nk_f32x8_to_bf16x8_haswell(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_bf16_to_f32(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i result_i8x32 = _mm256_adds_epi8(a_i8x32, b_i8x32);
        _mm256_storeu_si256((__m256i *)(result + i), result_i8x32);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = ai + bi;
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_i8_haswell(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_val * ai + beta_val;
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_i8_haswell(                   //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_i8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_i8_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_i8_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i result_u8x32 = _mm256_adds_epu8(a_u8x32, b_u8x32);
        _mm256_storeu_si256((__m256i *)(result + i), result_u8x32);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = ai + bi;
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_u8_haswell(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_val * ai + beta_val;
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_u8_haswell(                   //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_u8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_u8_haswell(a, n, alpha, &zero, result); }
        else { nk_scale_u8_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i8_haswell(                                      //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u8_haswell(                                      //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        // Instead of serial calls to expensive `_nk_f32_to_u8`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        _nk_i64_to_i16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_i16_haswell(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_set1_ps(-32768.0f);
    __m256 max_f32x8 = _mm256_set1_ps(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(result_i32x8),
                                               _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_i16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_f32 * ai + beta_f32;
        _nk_f32_to_i16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i16_haswell(                                        //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_set1_ps(-32768.0f);
    __m256 max_f32x8 = _mm256_set1_ps(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(b + i))));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(c + i))));
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(result_i32x8),
                                               _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_i16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _nk_f32_to_i16(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_u64_t ai = a[i], bi = b[i];
        nk_u64_t sum = ai + bi;
        _nk_u64_to_u16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_u16_haswell(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_setzero_ps();
    __m256 max_f32x8 = _mm256_set1_ps(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(result_i32x8),
                                                _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_u16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_f32 * ai + beta_f32;
        _nk_f32_to_u16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u16_haswell(                                        //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_setzero_ps();
    __m256 max_f32x8 = _mm256_set1_ps(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(b + i))));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(c + i))));
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(result_i32x8),
                                                _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_u16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _nk_f32_to_u16(&sum, result + i);
    }
}

NK_INTERNAL __m256i _mm256_adds_epi32_haswell(__m256i a, __m256i b) {
    // ! There are many flavors of addition with saturation in AVX2: i8, u8, i16, and u16.
    // ! But not for larger numeric types. We have to do it manually.
    // ! https://stackoverflow.com/a/56531252/2766161
    __m256i result = _mm256_add_epi32(a, b);
    // Detect positive overflow: (a > 0) && (b > 0) && (result < a)
    __m256i positive_mask = _mm256_and_si256(_mm256_cmpgt_epi32(a, _mm256_setzero_si256()),
                                             _mm256_cmpgt_epi32(b, _mm256_setzero_si256()));
    __m256i overflow_mask = _mm256_and_si256(positive_mask, _mm256_cmpgt_epi32(a, result));

    // Detect negative overflow: (a < 0) && (b < 0) && (result > a)
    __m256i negative_mask = _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), a),
                                             _mm256_cmpgt_epi32(_mm256_setzero_si256(), b));
    __m256i underflow_mask = _mm256_and_si256(negative_mask, _mm256_cmpgt_epi32(result, a));

    // Apply saturation: Set 2147483647 for positive overflow, -2147483648 for negative overflow
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(2147483647), overflow_mask);
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(-2147483648), underflow_mask);
    return result;
}

NK_PUBLIC void nk_sum_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        _nk_i64_to_i32(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_i32_haswell(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(-2147483648.0);
    __m256d max_f64x4 = _mm256_set1_pd(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_i32x4 = _mm256_cvtpd_epi32(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_i32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i];
        nk_f64_t sum = alpha_val * ai + beta_val;
        _nk_f64_to_i32(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i32_haswell(                                        //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(-2147483648.0);
    __m256d max_f64x4 = _mm256_set1_pd(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d b_f64x4 = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(b + i)));
        __m256d c_f64x4 = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(c + i)));
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d ab_scaled_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, ab_scaled_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_i32x4 = _mm256_cvtpd_epi32(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_i32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i], ci = c[i];
        nk_f64_t sum = alpha_val * ai * bi + beta_val * ci;
        _nk_f64_to_i32(&sum, result + i);
    }
}

NK_INTERNAL __m256i _mm256_adds_epu32_haswell(__m256i a, __m256i b) {
    // TODO: Saturated addition of unsigned 32-bit integers in AVX2 isn't trivial.
    // We don't have a `_mm256_adds_epu32` or `_mm256_add_epu32` instruction.
    // We don't have a `_mm256_packus_epi64` to implement addition in 64-bit integers,
    // which we need for subsequent downcasting opertaion.
    nk_u32_t a_vals[8], b_vals[8], result_vals[8];
    _mm256_storeu_si256((__m256i *)a_vals, a);
    _mm256_storeu_si256((__m256i *)b_vals, b);

    // Perform saturating addition for each element with separate sum variables
    nk_u64_t sum0 = (nk_u64_t)a_vals[0] + (nk_u64_t)b_vals[0];
    nk_u64_t sum1 = (nk_u64_t)a_vals[1] + (nk_u64_t)b_vals[1];
    nk_u64_t sum2 = (nk_u64_t)a_vals[2] + (nk_u64_t)b_vals[2];
    nk_u64_t sum3 = (nk_u64_t)a_vals[3] + (nk_u64_t)b_vals[3];
    nk_u64_t sum4 = (nk_u64_t)a_vals[4] + (nk_u64_t)b_vals[4];
    nk_u64_t sum5 = (nk_u64_t)a_vals[5] + (nk_u64_t)b_vals[5];
    nk_u64_t sum6 = (nk_u64_t)a_vals[6] + (nk_u64_t)b_vals[6];
    nk_u64_t sum7 = (nk_u64_t)a_vals[7] + (nk_u64_t)b_vals[7];

    // Apply saturation
    result_vals[0] = (sum0 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum0;
    result_vals[1] = (sum1 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum1;
    result_vals[2] = (sum2 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum2;
    result_vals[3] = (sum3 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum3;
    result_vals[4] = (sum4 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum4;
    result_vals[5] = (sum5 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum5;
    result_vals[6] = (sum6 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum6;
    result_vals[7] = (sum7 > 0xFFFFFFFF) ? 0xFFFFFFFF : (nk_u32_t)sum7;

    // Load results back into an AVX2 vector
    return _mm256_loadu_si256((__m256i *)result_vals);
}

NK_INTERNAL __m256d _mm256_cvtepu32_pd_haswell(__m128i a) {
    // TODO: Converting unsigned 32-bit integers to double-precision floats isn't trivial in AVX2.
    // Let's convert the lower 31 bits to a double-precision float.
    // And then conditionally add 2^31 to the result if the MSB is set.
    //
    //  __m256d result = _mm256_cvtepi32_pd(_mm_and_si128(a, _mm_set1_epi32(0x7FFFFFFF)));
    //  int should_increment = (_mm_movemask_epi8(a) & 0x8888);
    //  should_increment = should_increment / 0x8888; // Transform something like 0b1000100010001000 to 0b1111
    //  __m256d incremented = _mm256_add_pd(result, _mm256_set1_pd(2147483648.0));
    //  result = _mm256_blend_pd(result, incremented, should_increment);
    nk_u32_t from[4];
    nk_f64_t to[4];
    _mm_storeu_si128((__m128i *)from, a);
    to[0] = (nk_f64_t)from[0];
    to[1] = (nk_f64_t)from[1];
    to[2] = (nk_f64_t)from[2];
    to[3] = (nk_f64_t)from[3];
    return _mm256_loadu_pd(to);
}

NK_INTERNAL __m128i _mm256_cvtpd_epu32_haswell(__m256d a) {
    //? For now let's avoid SIMD and just use serial conversion.
    nk_f64_t from[4];
    nk_u32_t to[4];
    _mm256_storeu_pd(from, a);
    to[0] = (nk_u32_t)from[0];
    to[1] = (nk_u32_t)from[1];
    to[2] = (nk_u32_t)from[2];
    to[3] = (nk_u32_t)from[3];
    return _mm_lddqu_si128((__m128i *)to);
}

NK_PUBLIC void nk_sum_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        _nk_i64_to_u32(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_u32_haswell(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(0);
    __m256d max_f64x4 = _mm256_set1_pd(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_u32x4 = _mm256_cvtpd_epu32_haswell(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_u32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i];
        nk_f64_t sum = alpha_val * ai + beta_val;
        _nk_f64_to_u32(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u32_haswell(                                        //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(0);
    __m256d max_f64x4 = _mm256_set1_pd(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d b_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(b + i)));
        __m256d c_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(c + i)));
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d ab_scaled_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, ab_scaled_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_u32x4 = _mm256_cvtpd_epu32_haswell(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_u32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i], ci = c[i];
        nk_f64_t sum = alpha_val * ai * bi + beta_val * ci;
        _nk_f64_to_u32(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,bmi2"))), \
                             apply_to = function)

NK_PUBLIC void nk_sum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d a_vec, b_vec, sum_vec;
    __mmask8 mask = 0xFF;
nk_sum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    sum_vec = _mm512_add_pd(a_vec, b_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto nk_sum_f64_skylake_cycle;
}

NK_PUBLIC void nk_scale_f64_skylake(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_scale_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        a += 8, n -= 8;
    }
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_scale_f64_skylake_cycle;
}

NK_PUBLIC void nk_wsum_f64_skylake(                    //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f64_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_scale_f64_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_f64_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, a_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_wsum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    a_scaled_f64x8 = _mm512_mul_pd(a_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(b_f64x8, beta_f64x8, a_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_wsum_f64_skylake_cycle;
}

NK_PUBLIC void nk_sum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

nk_sum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto nk_sum_f32_skylake_cycle;
}

NK_PUBLIC void nk_scale_f32_skylake(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;

nk_scale_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        a += 16, n -= 16;
    }
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_scale_f32_skylake_cycle;
}

NK_PUBLIC void nk_wsum_f32_skylake(                    //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f32_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f32_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_f32_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_wsum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_wsum_f32_skylake_cycle;
}

NK_PUBLIC void nk_sum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
nk_sum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_vec = _nk_bf16x16_to_f32x16_skylake(a_bf16_vec);
    b_vec = _nk_bf16x16_to_f32x16_skylake(b_bf16_vec);
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    sum_bf16_vec = _nk_f32x16_to_bf16x16_skylake(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto nk_sum_bf16_skylake_cycle;
}

NK_PUBLIC void nk_scale_bf16_skylake(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, result_bf16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_scale_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        a += 16, n -= 16;
    }
    a_f32x16 = _nk_bf16x16_to_f32x16_skylake(a_bf16x16);
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_bf16x16 = _nk_f32x16_to_bf16x16_skylake(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_scale_bf16_skylake_cycle;
}

NK_PUBLIC void nk_wsum_bf16_skylake(                     //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_bf16_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_bf16_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_bf16_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_wsum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = _nk_bf16x16_to_f32x16_skylake(a_bf16x16);
    b_f32x16 = _nk_bf16x16_to_f32x16_skylake(b_bf16x16);
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    result_bf16x16 = _nk_f32x16_to_bf16x16_skylake(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_wsum_bf16_skylake_cycle;
}

NK_PUBLIC void nk_fma_f64_skylake(                                        //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        c_f64x8 = _mm512_maskz_loadu_pd(mask, c);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        c_f64x8 = _mm512_loadu_pd(c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_fma_f64_skylake_cycle;
}

NK_PUBLIC void nk_fma_f32_skylake(                                        //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_fma_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        c_f32x16 = _mm512_maskz_loadu_ps(mask, c);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        c_f32x16 = _mm512_loadu_ps(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_fma_f32_skylake_cycle;
}

NK_PUBLIC void nk_fma_bf16_skylake(                                          //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, c_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_fma_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_bf16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        c_bf16x16 = _mm256_loadu_epi16(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _nk_bf16x16_to_f32x16_skylake(a_bf16x16);
    b_f32x16 = _nk_bf16x16_to_f32x16_skylake(b_bf16x16);
    c_f32x16 = _nk_bf16x16_to_f32x16_skylake(c_bf16x16);
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_bf16x16 = _nk_f32x16_to_bf16x16_skylake(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_fma_bf16_skylake_cycle;
}

NK_PUBLIC void nk_scale_i8_skylake(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, result_i8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_scale_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8x16 = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_scale_i8_skylake_cycle;
}

NK_PUBLIC void nk_fma_i8_skylake(                                      //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, b_i8x16, c_i8x16, result_i8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_fma_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_i8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_i8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8x16 = _mm_lddqu_si128((__m128i *)a);
        b_i8x16 = _mm_lddqu_si128((__m128i *)b);
        c_i8x16 = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_i8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c_i8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_fma_i8_skylake_cycle;
}

NK_PUBLIC void nk_scale_u8_skylake(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, result_u8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_scale_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8x16 = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_scale_u8_skylake_cycle;
}

NK_PUBLIC void nk_fma_u8_skylake(                                      //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, b_u8x16, c_u8x16, result_u8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_fma_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_u8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_u8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8x16 = _mm_lddqu_si128((__m128i *)a);
        b_u8x16 = _mm_lddqu_si128((__m128i *)b);
        c_u8x16 = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_u8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c_u8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_fma_u8_skylake_cycle;
}

NK_PUBLIC void nk_scale_i16_skylake(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, result_i16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-32768);
    __m512i max_i32x16 = _mm512_set1_epi32(32767);

nk_scale_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_scale_i16_skylake_cycle;
}

NK_PUBLIC void nk_fma_i16_skylake(                                        //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, b_i16x16, c_i16x16, result_i16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-32768);
    __m512i max_i32x16 = _mm512_set1_epi32(32767);

nk_fma_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_i16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_i16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_lddqu_si256((__m256i *)a);
        b_i16x16 = _mm256_lddqu_si256((__m256i *)b);
        c_i16x16 = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(b_i16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c_i16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_fma_i16_skylake_cycle;
}

NK_PUBLIC void nk_scale_u16_skylake(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, result_u16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(65535);

nk_scale_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_scale_u16_skylake_cycle;
}

NK_PUBLIC void nk_fma_u16_skylake(                                        //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, b_u16x16, c_u16x16, result_u16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(65535);

nk_fma_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_u16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_u16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_lddqu_si256((__m256i *)a);
        b_u16x16 = _mm256_lddqu_si256((__m256i *)b);
        c_u16x16 = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b_u16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(c_u16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_fma_u16_skylake_cycle;
}

NK_PUBLIC void nk_scale_i32_skylake(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, result_i32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_scale_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvttpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_scale_i32_skylake_cycle;
}

NK_PUBLIC void nk_fma_i32_skylake(                                        //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, b_i32x8, c_i32x8, result_i32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_fma_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_i32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_i32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_lddqu_si256((__m256i *)a);
        b_i32x8 = _mm256_lddqu_si256((__m256i *)b);
        c_i32x8 = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    b_f64x8 = _mm512_cvtepi32_pd(b_i32x8);
    c_f64x8 = _mm512_cvtepi32_pd(c_i32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvttpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_fma_i32_skylake_cycle;
}

NK_PUBLIC void nk_scale_u32_skylake(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, result_u32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_scale_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvttpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_scale_u32_skylake_cycle;
}

NK_PUBLIC void nk_fma_u32_skylake(                                        //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, b_u32x8, c_u32x8, result_u32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_fma_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_u32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_u32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_lddqu_si256((__m256i *)a);
        b_u32x8 = _mm256_lddqu_si256((__m256i *)b);
        c_u32x8 = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    b_f64x8 = _mm512_cvtepu32_pd(b_u32x8);
    c_f64x8 = _mm512_cvtepu32_pd(c_u32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvttpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_fma_u32_skylake_cycle;
}

NK_PUBLIC void nk_scale_i64_skylake(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_i64x8;
    __mmask8 mask = 0xFF;

nk_scale_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_scale_i64_skylake_cycle;
}

NK_PUBLIC void nk_fma_i64_skylake(                                        //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8, b_i64x8, c_i64x8, result_i64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_i64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_i64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        b_i64x8 = _mm512_loadu_si512((__m512i *)b);
        c_i64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    b_f64x8 = _mm512_cvtepi64_pd(b_i64x8);
    c_f64x8 = _mm512_cvtepi64_pd(c_i64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_fma_i64_skylake_cycle;
}

NK_PUBLIC void nk_scale_u64_skylake(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_u64x8;
    __mmask8 mask = 0xFF;

nk_scale_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_scale_u64_skylake_cycle;
}

NK_PUBLIC void nk_fma_u64_skylake(                                        //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8, b_u64x8, c_u64x8, result_u64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_u64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_u64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        b_u64x8 = _mm512_loadu_si512((__m512i *)b);
        c_u64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    b_f64x8 = _mm512_cvtepu64_pd(b_u64x8);
    c_f64x8 = _mm512_cvtepu64_pd(c_u64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_fma_u64_skylake_cycle;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

NK_PUBLIC void nk_sum_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_i8_vec, b_i8_vec;
    __m512i sum_i8_vec;
nk_sum_i8_ice_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_i8_vec = _mm512_adds_epi8(a_i8_vec, b_i8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto nk_sum_i8_ice_cycle;
}

NK_PUBLIC void nk_sum_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_u8_vec, b_u8_vec;
    __m512i sum_u8_vec;
nk_sum_u8_ice_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_u8_vec = _mm512_adds_epu8(a_u8_vec, b_u8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto nk_sum_u8_ice_cycle;
}

NK_PUBLIC void nk_sum_i16_ice(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_i16_vec, b_i16_vec;
    __m512i sum_i16_vec;
nk_sum_i16_ice_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_i16_vec = _mm512_adds_epi16(a_i16_vec, b_i16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_i16_vec);
    result += 32;
    if (n) goto nk_sum_i16_ice_cycle;
}

NK_PUBLIC void nk_sum_u16_ice(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_u16_vec, b_u16_vec;
    __m512i sum_u16_vec;
nk_sum_u16_ice_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_u16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_u16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_u16_vec = _mm512_loadu_epi16(a);
        b_u16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_u16_vec = _mm512_adds_epu16(a_u16_vec, b_u16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_u16_vec);
    result += 32;
    if (n) goto nk_sum_u16_ice_cycle;
}

NK_INTERNAL __m512i _mm512_adds_epi32_ice(__m512i a, __m512i b) {
    // ! There are many flavors of addition with saturation in AVX-512: i8, u8, i16, and u16.
    // ! But not for larger numeric types. We have to do it manually.
    // ! https://stackoverflow.com/a/56531252/2766161
    __m512i sum = _mm512_add_epi32(a, b);

    // Set constants for overflow and underflow limits
    __m512i max_val = _mm512_set1_epi32(2147483647);
    __m512i min_val = _mm512_set1_epi32(-2147483648);

    // TODO: Consider using ternary operator for performance.
    // Detect positive overflow: (a > 0) && (b > 0) && (sum < 0)
    __mmask16 a_is_positive = _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512());
    __mmask16 b_is_positive = _mm512_cmpgt_epi32_mask(b, _mm512_setzero_si512());
    __mmask16 sum_is_negative = _mm512_cmplt_epi32_mask(sum, _mm512_setzero_si512());
    __mmask16 pos_overflow_mask = _kand_mask16(_kand_mask16(a_is_positive, b_is_positive), sum_is_negative);

    // TODO: Consider using ternary operator for performance.
    // Detect negative overflow: (a < 0) && (b < 0) && (sum >= 0)
    __mmask16 a_is_negative = _mm512_cmplt_epi32_mask(a, _mm512_setzero_si512());
    __mmask16 b_is_negative = _mm512_cmplt_epi32_mask(b, _mm512_setzero_si512());
    __mmask16 sum_is_non_negative = _mm512_cmpge_epi32_mask(sum, _mm512_setzero_si512());
    __mmask16 neg_overflow_mask = _kand_mask16(_kand_mask16(a_is_negative, b_is_negative), sum_is_non_negative);

    // Apply saturation for positive overflow
    sum = _mm512_mask_blend_epi32(pos_overflow_mask, sum, max_val);
    // Apply saturation for negative overflow
    sum = _mm512_mask_blend_epi32(neg_overflow_mask, sum, min_val);
    return sum;
}

NK_INTERNAL __m512i _mm512_adds_epu32_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi32(a, b);
    __mmask16 overflow_mask = _mm512_cmp_epu32_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi32(4294967295u);
    return _mm512_mask_blend_epi32(overflow_mask, sum, max_val);
}

NK_INTERNAL __m512i _mm512_adds_epi64_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000);

    __m512i overflow = _mm512_and_si512(_mm512_xor_si512(a, b), sign_mask);  // Same sign inputs
    __m512i overflows = _mm512_or_si512(overflow, _mm512_xor_si512(sum, a)); // Overflow condition

    __m512i max_val = _mm512_set1_epi64(9223372036854775807ll);
    __m512i min_val = _mm512_set1_epi64(-9223372036854775807ll - 1);
    __m512i overflow_result = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(sum, min_val, _MM_CMPINT_LT), max_val,
                                                      min_val);

    return _mm512_mask_blend_epi64(_mm512_test_epi64_mask(overflows, overflows), sum, overflow_result);
}

NK_INTERNAL __m512i _mm512_adds_epu64_ice(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __mmask8 overflow_mask = _mm512_cmp_epu64_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi64(18446744073709551615ull);
    return _mm512_mask_blend_epi64(overflow_mask, sum, max_val);
}

NK_PUBLIC void nk_sum_i32_ice(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_i32_vec, b_i32_vec;
    __m512i sum_i32_vec;
nk_sum_i32_ice_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_i32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_i32_vec = _mm512_loadu_epi32(a);
        b_i32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_i32_vec = _mm512_adds_epi32_ice(a_i32_vec, b_i32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_i32_vec);
    result += 16;
    if (n) goto nk_sum_i32_ice_cycle;
}

NK_PUBLIC void nk_sum_u32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_u32_vec, b_u32_vec;
    __m512i sum_u32_vec;
nk_sum_u32_ice_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_u32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_u32_vec = _mm512_loadu_epi32(a);
        b_u32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_u32_vec = _mm512_adds_epu32_ice(a_u32_vec, b_u32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_u32_vec);
    result += 16;
    if (n) goto nk_sum_u32_ice_cycle;
}

NK_PUBLIC void nk_sum_i64_ice(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_i64_vec, b_i64_vec;
    __m512i sum_i64_vec;
nk_sum_i64_ice_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_i64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_i64_vec = _mm512_loadu_epi64(a);
        b_i64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_i64_vec = _mm512_adds_epi64_ice(a_i64_vec, b_i64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_i64_vec);
    result += 8;
    if (n) goto nk_sum_i64_ice_cycle;
}

NK_PUBLIC void nk_sum_u64_ice(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_u64_vec, b_u64_vec;
    __m512i sum_u64_vec;
nk_sum_u64_ice_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_u64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_u64_vec = _mm512_loadu_epi64(a);
        b_u64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_u64_vec = _mm512_adds_epu64_ice(a_u64_vec, b_u64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_u64_vec);
    result += 8;
    if (n) goto nk_sum_u64_ice_cycle;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE

#if NK_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

NK_PUBLIC void nk_sum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512h a_f16_vec, b_f16_vec;
    __m512h sum_f16_vec;
nk_sum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        b_f16_vec = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    sum_f16_vec = _mm512_add_ph(a_f16_vec, b_f16_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto nk_sum_f16_sapphire_cycle;
}

NK_PUBLIC void nk_scale_f16_sapphire(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512h a_f16x32;
    __m512h result_f16x32;
nk_scale_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        a += 32, n -= 32;
    }
    result_f16x32 = _mm512_fmadd_ph(a_f16x32, alpha_f16x32, beta_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_scale_f16_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_f16_sapphire(                   //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f16_sapphire(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f16_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_f16_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512h a_f16x32, b_f16x32;
    __m512h a_scaled_f16x32, result_f16x32;
nk_wsum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        b_f16x32 = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    a_scaled_f16x32 = _mm512_mul_ph(a_f16x32, alpha_f16x32);
    result_f16x32 = _mm512_fmadd_ph(b_f16x32, beta_f16x32, a_scaled_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_wsum_f16_sapphire_cycle;
}

NK_PUBLIC void nk_fma_f16_sapphire(                                       //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512h a_f16x32, b_f16x32, c_f16x32;
    __m512h ab_f16x32, ab_scaled_f16x32, result_f16x32;
nk_fma_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        c_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        b_f16x32 = _mm512_loadu_ph(b);
        c_f16x32 = _mm512_loadu_ph(c);
        a += 32, b += 32, c += 32, n -= 32;
    }
    ab_f16x32 = _mm512_mul_ph(a_f16x32, b_f16x32);
    ab_scaled_f16x32 = _mm512_mul_ph(ab_f16x32, alpha_f16x32);
    result_f16x32 = _mm512_fmadd_ph(c_f16x32, beta_f16x32, ab_scaled_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_fma_f16_sapphire_cycle;
}

NK_PUBLIC void nk_scale_u8_sapphire(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_u8x64, result_u8x64;
    __m512h a_f16x32_lo, a_f16x32_hi;
    __m512h result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
nk_scale_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        a += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    // Scale:
    result_f16x32_lo = _mm512_fmadd_ph(a_f16x32_lo, alpha_f16x32, beta_f16x32);
    result_f16x32_hi = _mm512_fmadd_ph(a_f16x32_hi, alpha_f16x32, beta_f16x32);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    result_u8x64 = _mm512_packus_epi16(result_i16x32_lo, result_i16x32_hi);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_scale_u8_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_u8_sapphire(                  //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_u8_ice(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_u8_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_u8_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_u8x64, b_u8x64, result_u8x64;
    __m512h a_f16x32_lo, a_f16x32_hi, b_f16x32_lo, b_f16x32_hi;
    __m512h a_scaled_f16x32_lo, a_scaled_f16x32_hi, result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
nk_wsum_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        b_u8x64 = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    b_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8x64, _mm512_setzero_si512()));
    b_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8x64, _mm512_setzero_si512()));
    // Scale:
    a_scaled_f16x32_lo = _mm512_mul_ph(a_f16x32_lo, alpha_f16x32);
    a_scaled_f16x32_hi = _mm512_mul_ph(a_f16x32_hi, alpha_f16x32);
    // Add:
    result_f16x32_lo = _mm512_fmadd_ph(b_f16x32_lo, beta_f16x32, a_scaled_f16x32_lo);
    result_f16x32_hi = _mm512_fmadd_ph(b_f16x32_hi, beta_f16x32, a_scaled_f16x32_hi);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    result_u8x64 = _mm512_packus_epi16(result_i16x32_lo, result_i16x32_hi);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_wsum_u8_sapphire_cycle;
}

NK_PUBLIC void nk_scale_i8_sapphire(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_i8x64, result_i8x64;
    __m512h a_f16x32_lo, a_f16x32_hi;
    __m512h result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
nk_scale_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8x64 = _mm512_loadu_epi8(a);
        a += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8x64)));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8x64, 1)));
    // Scale:
    result_f16x32_lo = _mm512_fmadd_ph(a_f16x32_lo, alpha_f16x32, beta_f16x32);
    result_f16x32_hi = _mm512_fmadd_ph(a_f16x32_hi, alpha_f16x32, beta_f16x32);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_i16x32_lo)),
                                      _mm512_cvtsepi16_epi8(result_i16x32_hi), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_scale_i8_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_i8_sapphire(                  //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_i8_ice(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_i8_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_i8_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_i8x64, b_i8x64, result_i8x64;
    __m512h a_f16x32_lo, a_f16x32_hi, b_f16x32_lo, b_f16x32_hi;
    __m512h a_scaled_f16x32_lo, a_scaled_f16x32_hi, result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
nk_wsum_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_i8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8x64 = _mm512_loadu_epi8(a);
        b_i8x64 = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8x64)));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8x64, 1)));
    b_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8x64)));
    b_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8x64, 1)));
    // Scale:
    a_scaled_f16x32_lo = _mm512_mul_ph(a_f16x32_lo, alpha_f16x32);
    a_scaled_f16x32_hi = _mm512_mul_ph(a_f16x32_hi, alpha_f16x32);
    // Add:
    result_f16x32_lo = _mm512_fmadd_ph(b_f16x32_lo, beta_f16x32, a_scaled_f16x32_lo);
    result_f16x32_hi = _mm512_fmadd_ph(b_f16x32_hi, beta_f16x32, a_scaled_f16x32_hi);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_i16x32_lo)),
                                      _mm512_cvtsepi16_epi8(result_i16x32_hi), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_wsum_i8_sapphire_cycle;
}

NK_PUBLIC void nk_fma_i8_sapphire(                                     //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_i8x64, b_i8x64, c_i8x64, result_i8x64;
    __m512h a_f16x32_lo, a_f16x32_hi, b_f16x32_lo, b_f16x32_hi;
    __m512h c_f16x32_lo, c_f16x32_hi, ab_f16x32_lo, ab_f16x32_hi;
    __m512h ab_scaled_f16x32_lo, ab_scaled_f16x32_hi, result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
    __m512h min_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(-128));
    __m512h max_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(127));

nk_fma_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_i8x64 = _mm512_maskz_loadu_epi8(mask, b);
        c_i8x64 = _mm512_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8x64 = _mm512_loadu_epi8(a);
        b_i8x64 = _mm512_loadu_epi8(b);
        c_i8x64 = _mm512_loadu_epi8(c);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8x64)));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8x64, 1)));
    b_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8x64)));
    b_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8x64, 1)));
    c_f16x32_lo = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(c_i8x64)));
    c_f16x32_hi = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(c_i8x64, 1)));
    // Multiply:
    ab_f16x32_lo = _mm512_mul_ph(a_f16x32_lo, b_f16x32_lo);
    ab_f16x32_hi = _mm512_mul_ph(a_f16x32_hi, b_f16x32_hi);
    // Scale:
    ab_scaled_f16x32_lo = _mm512_mul_ph(ab_f16x32_lo, alpha_f16x32);
    ab_scaled_f16x32_hi = _mm512_mul_ph(ab_f16x32_hi, alpha_f16x32);
    // Add:
    result_f16x32_lo = _mm512_fmadd_ph(c_f16x32_lo, beta_f16x32, ab_scaled_f16x32_lo);
    result_f16x32_hi = _mm512_fmadd_ph(c_f16x32_hi, beta_f16x32, ab_scaled_f16x32_hi);
    // Clip the 16-bit result to 8-bit:
    result_f16x32_lo = _mm512_max_ph(_mm512_min_ph(result_f16x32_lo, max_f16x32), min_f16x32);
    result_f16x32_hi = _mm512_max_ph(_mm512_min_ph(result_f16x32_hi, max_f16x32), min_f16x32);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    // Merge back:
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_i16x32_lo)),
                                      _mm512_cvtsepi16_epi8(result_i16x32_hi), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_fma_i8_sapphire_cycle;
}

NK_PUBLIC void nk_fma_u8_sapphire(                                     //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_set1_ph((_Float16)alpha_val);
    __m512h beta_f16x32 = _mm512_set1_ph((_Float16)beta_val);
    __m512i a_u8x64, b_u8x64, c_u8x64, result_u8x64;
    __m512h a_f16x32_lo, a_f16x32_hi, b_f16x32_lo, b_f16x32_hi;
    __m512h c_f16x32_lo, c_f16x32_hi, ab_f16x32_lo, ab_f16x32_hi;
    __m512h ab_scaled_f16x32_lo, ab_scaled_f16x32_hi, result_f16x32_lo, result_f16x32_hi;
    __m512i result_i16x32_lo, result_i16x32_hi;
    __m512h min_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(0));
    __m512h max_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(255));

nk_fma_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        c_u8x64 = _mm512_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        b_u8x64 = _mm512_loadu_epi8(b);
        c_u8x64 = _mm512_loadu_epi8(c);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast:
    a_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    b_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8x64, _mm512_setzero_si512()));
    b_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8x64, _mm512_setzero_si512()));
    c_f16x32_lo = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(c_u8x64, _mm512_setzero_si512()));
    c_f16x32_hi = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(c_u8x64, _mm512_setzero_si512()));
    // Multiply:
    ab_f16x32_lo = _mm512_mul_ph(a_f16x32_lo, b_f16x32_lo);
    ab_f16x32_hi = _mm512_mul_ph(a_f16x32_hi, b_f16x32_hi);
    // Scale:
    ab_scaled_f16x32_lo = _mm512_mul_ph(ab_f16x32_lo, alpha_f16x32);
    ab_scaled_f16x32_hi = _mm512_mul_ph(ab_f16x32_hi, alpha_f16x32);
    // Add:
    result_f16x32_lo = _mm512_fmadd_ph(c_f16x32_lo, beta_f16x32, ab_scaled_f16x32_lo);
    result_f16x32_hi = _mm512_fmadd_ph(c_f16x32_hi, beta_f16x32, ab_scaled_f16x32_hi);
    // Clip the 16-bit result to 8-bit:
    result_f16x32_lo = _mm512_max_ph(_mm512_min_ph(result_f16x32_lo, max_f16x32), min_f16x32);
    result_f16x32_hi = _mm512_max_ph(_mm512_min_ph(result_f16x32_hi, max_f16x32), min_f16x32);
    // Downcast:
    result_i16x32_lo = _mm512_cvtph_epi16(result_f16x32_lo);
    result_i16x32_hi = _mm512_cvtph_epi16(result_f16x32_hi);
    // Merge back:
    result_u8x64 = _mm512_packus_epi16(result_i16x32_lo, result_i16x32_hi);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_fma_u8_sapphire_cycle;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // _NK_TARGET_X86

#if _NK_TARGET_ARM
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

NK_PUBLIC void nk_sum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_scale_f32_neon(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_val);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t result_f32x4 = vfmaq_n_f32(beta_f32x4, a_f32x4, alpha_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_wsum_f32_neon(                       //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f32_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f32_neon(a, n, alpha, &zero, result); }
        else { nk_scale_f32_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_fma_f32_neon(                              //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t c_f32x4 = vld1q_f32(c + i);
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_sum_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t a_vec = vld1q_s16(a + i);
        int16x8_t b_vec = vld1q_s16(b + i);
        int16x8_t sum_vec = vqaddq_s16(a_vec, b_vec);
        vst1q_s16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_i16_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i16_neon(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtaq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        _nk_f32_to_i16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i16_neon(                              //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        int16x4_t b_i16x4 = vld1_s16(b + i);
        int16x4_t c_i16x4 = vld1_s16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t b_f32x4 = vcvtq_f32_s32(vmovl_s16(b_i16x4));
        float32x4_t c_f32x4 = vcvtq_f32_s32(vmovl_s16(c_i16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtaq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        _nk_f32_to_i16(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_vec = vld1q_u16(a + i);
        uint16x8_t b_vec = vld1q_u16(b + i);
        uint16x8_t sum_vec = vqaddq_u16(a_vec, b_vec);
        vst1q_u16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_u16_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u16_neon(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtaq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        _nk_f32_to_u16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u16_neon(                              //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        uint16x4_t b_u16x4 = vld1_u16(b + i);
        uint16x4_t c_u16x4 = vld1_u16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t b_f32x4 = vcvtq_f32_u32(vmovl_u16(b_u16x4));
        float32x4_t c_f32x4 = vcvtq_f32_u32(vmovl_u16(c_u16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtaq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        _nk_f32_to_u16(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t a_vec = vld1q_s32(a + i);
        int32x4_t b_vec = vld1q_s32(b + i);
        int32x4_t sum_vec = vqaddq_s32(a_vec, b_vec);
        vst1q_s32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_i32_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i32_neon(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtaq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        _nk_f64_to_i32(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i32_neon(                              //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        int32x2_t b_i32x2 = vld1_s32(b + i);
        int32x2_t c_i32x2 = vld1_s32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t b_f64x2 = vcvtq_f64_s64(vmovl_s32(b_i32x2));
        float64x2_t c_f64x2 = vcvtq_f64_s64(vmovl_s32(c_i32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtaq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        _nk_f64_to_i32(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_vec = vld1q_u32(a + i);
        uint32x4_t b_vec = vld1q_u32(b + i);
        uint32x4_t sum_vec = vqaddq_u32(a_vec, b_vec);
        vst1q_u32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_u32_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u32_neon(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtaq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        _nk_f64_to_u32(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u32_neon(                              //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        uint32x2_t b_u32x2 = vld1_u32(b + i);
        uint32x2_t c_u32x2 = vld1_u32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t b_f64x2 = vcvtq_f64_u64(vmovl_u32(b_u32x2));
        float64x2_t c_f64x2 = vcvtq_f64_u64(vmovl_u32(c_u32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtaq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        _nk_f64_to_u32(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_vec = vld1q_s64(a + i);
        int64x2_t b_vec = vld1q_s64(b + i);
        int64x2_t sum_vec = vqaddq_s64(a_vec, b_vec);
        vst1q_s64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_i64_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i64_neon(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        int64x2_t result_i64x2 = vcvtaq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        _nk_f64_to_i64(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i64_neon(                              //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        int64x2_t b_i64x2 = vld1q_s64(b + i);
        int64x2_t c_i64x2 = vld1q_s64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t b_f64x2 = vcvtq_f64_s64(b_i64x2);
        float64x2_t c_f64x2 = vcvtq_f64_s64(c_i64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        int64x2_t result_i64x2 = vcvtaq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        _nk_f64_to_i64(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);
        uint64x2_t sum_vec = vqaddq_u64(a_vec, b_vec);
        vst1q_u64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _nk_u64_sadd(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u64_neon(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        uint64x2_t result_u64x2 = vcvtaq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        _nk_f64_to_u64(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u64_neon(                              //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        uint64x2_t b_u64x2 = vld1q_u64(b + i);
        uint64x2_t c_u64x2 = vld1q_u64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t b_f64x2 = vcvtq_f64_u64(b_u64x2);
        float64x2_t c_f64x2 = vcvtq_f64_u64(c_u64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        uint64x2_t result_u64x2 = vcvtaq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        _nk_f64_to_u64(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON

#if NK_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

NK_PUBLIC void nk_sum_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_bf16_neon(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                  nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_val);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_bf16_to_f32(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_bf16_neon(                        //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_bf16_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_bf16_neon(a, n, alpha, &zero, result); }
        else { nk_scale_bf16_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_bf16_neon(                                //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t c_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)c + i));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_bf16_to_f32(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_bf16(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_BF16

#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

NK_PUBLIC void nk_sum_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_vec = vld1q_f16((float16_t const *)b + i);
        float16x8_t sum_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = ((float16_t const *)a)[i] + ((float16_t const *)b)[i];
}

NK_PUBLIC void nk_scale_f16_neon(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16;
}

NK_PUBLIC void nk_wsum_f16_neon(                       //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f16_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f16_neon(a, n, alpha, &zero, result); }
        else { nk_scale_f16_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_f16x8 = vld1q_f16((float16_t const *)b + i);
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16 * ((float16_t const *)b)[i];
}

NK_PUBLIC void nk_fma_f16_neon(                              //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_f16x8 = vld1q_f16((float16_t const *)b + i);
        float16x8_t c_f16x8 = vld1q_f16((float16_t const *)c + i);
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] * ((float16_t const *)b)[i] +
                                   beta_f16 * ((float16_t const *)c)[i];
}

NK_PUBLIC void nk_sum_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        uint8x16_t sum_vec = vqaddq_u8(a_vec, b_vec);
        vst1q_u8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_u8_neon(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                nk_u8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16;
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_u8_neon(                      //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_u8_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_u8_neon(a, n, alpha, &zero, result); }
        else { nk_scale_u8_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        uint8x8_t b_u8x8 = vld1_u8(b + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t b_f16x8 = vcvtq_f16_u16(vmovl_u8(b_u8x8));
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u8_neon(                            //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        uint8x8_t b_u8x8 = vld1_u8(b + i);
        uint8x8_t c_u8x8 = vld1_u8(c + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t b_f16x8 = vcvtq_f16_u16(vmovl_u8(b_u8x8));
        float16x8_t c_f16x8 = vcvtq_f16_u16(vmovl_u8(c_u8x8));
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        _nk_f32_to_u8(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i8_neon(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        int8x16_t sum_vec = vqaddq_s8(a_vec, b_vec);
        vst1q_s8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_i8_neon(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                nk_i8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16;
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_i8_neon(                      //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_i8_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_i8_neon(a, n, alpha, &zero, result); }
        else { nk_scale_i8_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        int8x8_t b_i8x8 = vld1_s8(b + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t b_f16x8 = vcvtq_f16_s16(vmovl_s8(b_i8x8));
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        _nk_f32_to_i8(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i8_neon(                            //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        int8x8_t b_i8x8 = vld1_s8(b + i);
        int8x8_t c_i8x8 = vld1_s8(c + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t b_f16x8 = vcvtq_f16_s16(vmovl_s8(b_i8x8));
        float16x8_t c_f16x8 = vcvtq_f16_s16(vmovl_s8(c_i8x8));
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        _nk_f32_to_i8(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // _NK_TARGET_ARM

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_sum_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_sum_f64_skylake(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_f64_haswell(a, b, n, r);
#else
    nk_sum_f64_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_sum_f32_skylake(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_f32_haswell(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_f32_neon(a, b, n, r);
#else
    nk_sum_f32_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_sum_bf16_skylake(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_bf16_haswell(a, b, n, r);
#elif NK_TARGET_NEON_BF16
    nk_sum_bf16_neon(a, b, n, r);
#else
    nk_sum_bf16_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_sum_f16_sapphire(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_f16_haswell(a, b, n, r);
#elif NK_TARGET_NEON_F16
    nk_sum_f16_neon(a, b, n, r);
#else
    nk_sum_f16_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *r) {
#if NK_TARGET_ICE
    nk_sum_i8_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_i8_haswell(a, b, n, r);
#elif NK_TARGET_NEON_F16
    nk_sum_i8_neon(a, b, n, r);
#else
    nk_sum_i8_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *r) {
#if NK_TARGET_ICE
    nk_sum_u8_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_u8_haswell(a, b, n, r);
#elif NK_TARGET_NEON_F16
    nk_sum_u8_neon(a, b, n, r);
#else
    nk_sum_u8_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_i16(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *r) {
#if NK_TARGET_ICE
    nk_sum_i16_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_i16_haswell(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_i16_neon(a, b, n, r);
#else
    nk_sum_i16_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *r) {
#if NK_TARGET_ICE
    nk_sum_u16_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_u16_haswell(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_u16_neon(a, b, n, r);
#else
    nk_sum_u16_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_i32(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *r) {
#if NK_TARGET_ICE
    nk_sum_i32_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_i32_haswell(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_i32_neon(a, b, n, r);
#else
    nk_sum_i32_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *r) {
#if NK_TARGET_ICE
    nk_sum_u32_ice(a, b, n, r);
#elif NK_TARGET_HASWELL
    nk_sum_u32_haswell(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_u32_neon(a, b, n, r);
#else
    nk_sum_u32_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_i64(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *r) {
#if NK_TARGET_ICE
    nk_sum_i64_ice(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_i64_neon(a, b, n, r);
#else
    nk_sum_i64_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_sum_u64(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *r) {
#if NK_TARGET_ICE
    nk_sum_u64_ice(a, b, n, r);
#elif NK_TARGET_NEON
    nk_sum_u64_neon(a, b, n, r);
#else
    nk_sum_u64_serial(a, b, n, r);
#endif
}

NK_PUBLIC void nk_scale_f64(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_f64_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_f64_haswell(a, n, alpha, beta, r);
#else
    nk_scale_f64_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_f32(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_f32_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_f32_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_f32_neon(a, n, alpha, beta, r);
#else
    nk_scale_f32_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_bf16(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                             nk_bf16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_bf16_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_bf16_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON_BF16
    nk_scale_bf16_neon(a, n, alpha, beta, r);
#else
    nk_scale_bf16_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_f16(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_scale_f16_sapphire(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_f16_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_scale_f16_neon(a, n, alpha, beta, r);
#else
    nk_scale_f16_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_i8(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_scale_i8_sapphire(a, n, alpha, beta, r);
#elif NK_TARGET_SKYLAKE
    nk_scale_i8_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_i8_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_scale_i8_neon(a, n, alpha, beta, r);
#else
    nk_scale_i8_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_u8(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_scale_u8_sapphire(a, n, alpha, beta, r);
#elif NK_TARGET_SKYLAKE
    nk_scale_u8_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_u8_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_scale_u8_neon(a, n, alpha, beta, r);
#else
    nk_scale_u8_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_i16(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_i16_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_i16_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_i16_neon(a, n, alpha, beta, r);
#else
    nk_scale_i16_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_u16(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_u16_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_u16_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_u16_neon(a, n, alpha, beta, r);
#else
    nk_scale_u16_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_i32(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_i32_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_i32_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_i32_neon(a, n, alpha, beta, r);
#else
    nk_scale_i32_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_u32(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_u32_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_scale_u32_haswell(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_u32_neon(a, n, alpha, beta, r);
#else
    nk_scale_u32_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_i64(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_i64_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_i64_neon(a, n, alpha, beta, r);
#else
    nk_scale_i64_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_scale_u64(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_scale_u64_skylake(a, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_scale_u64_neon(a, n, alpha, beta, r);
#else
    nk_scale_u64_serial(a, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                           nk_f64_t const *beta, nk_f64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_wsum_f64_skylake(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_f64_haswell(a, b, n, alpha, beta, r);
#else
    nk_wsum_f64_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_f32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_wsum_f32_skylake(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_f32_haswell(a, b, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_wsum_f32_neon(a, b, n, alpha, beta, r);
#else
    nk_wsum_f32_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                            nk_f32_t const *beta, nk_bf16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_wsum_bf16_skylake(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_bf16_haswell(a, b, n, alpha, beta, r);
#elif NK_TARGET_NEON_BF16
    nk_wsum_bf16_neon(a, b, n, alpha, beta, r);
#else
    nk_wsum_bf16_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                           nk_f32_t const *beta, nk_f16_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_wsum_f16_sapphire(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_f16_haswell(a, b, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_wsum_f16_neon(a, b, n, alpha, beta, r);
#else
    nk_wsum_f16_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_i8(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                          nk_i8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_wsum_i8_sapphire(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_i8_haswell(a, b, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_wsum_i8_neon(a, b, n, alpha, beta, r);
#else
    nk_wsum_i8_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_wsum_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                          nk_u8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_wsum_u8_sapphire(a, b, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_wsum_u8_haswell(a, b, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_wsum_u8_neon(a, b, n, alpha, beta, r);
#else
    nk_wsum_u8_serial(a, b, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                          nk_f64_t const *beta, nk_f64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_f64_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_f64_haswell(a, b, c, n, alpha, beta, r);
#else
    nk_fma_f64_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_f32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_f32_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_f32_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_f32_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_f32_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                           nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_bf16_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_bf16_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON_BF16
    nk_fma_bf16_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_bf16_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_f16_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_fma_f16_sapphire(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_f16_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_fma_f16_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_f16_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_i8(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                         nk_f32_t const *beta, nk_i8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_fma_i8_sapphire(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_SKYLAKE
    nk_fma_i8_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_i8_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_fma_i8_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_i8_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_u8(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, nk_f32_t const *alpha,
                         nk_f32_t const *beta, nk_u8_t *r) {
#if NK_TARGET_SAPPHIRE
    nk_fma_u8_sapphire(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_SKYLAKE
    nk_fma_i8_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_u8_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON_F16
    nk_fma_u8_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_u8_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_i16(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_i16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_i16_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_i16_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_i16_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_i16_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_u16(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, nk_f32_t const *alpha,
                          nk_f32_t const *beta, nk_u16_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_u16_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_u16_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_u16_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_u16_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_i32(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, nk_f64_t const *alpha,
                          nk_f64_t const *beta, nk_i32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_i32_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_i32_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_i32_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_i32_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_u32(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, nk_f64_t const *alpha,
                          nk_f64_t const *beta, nk_u32_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_u32_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_HASWELL
    nk_fma_u32_haswell(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_u32_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_u32_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_i64(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                          nk_f64_t const *beta, nk_i64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_i64_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_i64_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_i64_serial(a, b, c, n, alpha, beta, r);
#endif
}

NK_PUBLIC void nk_fma_u64(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n, nk_f64_t const *alpha,
                          nk_f64_t const *beta, nk_u64_t *r) {
#if NK_TARGET_SKYLAKE
    nk_fma_u64_skylake(a, b, c, n, alpha, beta, r);
#elif NK_TARGET_NEON
    nk_fma_u64_neon(a, b, c, n, alpha, beta, r);
#else
    nk_fma_u64_serial(a, b, c, n, alpha, beta, r);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif // NK_ELEMENTWISE_H
