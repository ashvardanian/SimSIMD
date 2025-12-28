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
 *  - Arm: NEON, NEON+F16
 *  - x86: Haswell, Skylake, Ice Lake, Sapphire Rapids
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

#include "numkong/types.h"

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

#if NK_TARGET_NEONBFDOT
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
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONHALF
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
#endif // NK_TARGET_NEONHALF

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

#include "numkong/elementwise/serial.h"
#include "numkong/elementwise/neon.h"
#include "numkong/elementwise/neonhalf.h"
#include "numkong/elementwise/neonbfdot.h"
#include "numkong/elementwise/haswell.h"
#include "numkong/elementwise/skylake.h"
#include "numkong/elementwise/ice.h"
#include "numkong/elementwise/sapphire.h"

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
#elif NK_TARGET_NEONBFDOT
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONBFDOT
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONBFDOT
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONBFDOT
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
#elif NK_TARGET_NEONHALF
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
