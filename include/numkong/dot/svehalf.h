/**
 *  @brief SIMD-accelerated Dot Products for SVE FP16.
 *  @file include/numkong/dot/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f16                   LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svld2_f16                   LD2H (Z.H, P/Z, [Xn])           6-8cy       1/cy
 *      svmla_f16_x                 FMLA (Z.H, P/M, Z.H, Z.H)       4cy         2/cy
 *      svmls_f16_x                 FMLS (Z.H, P/M, Z.H, Z.H)       4cy         2/cy
 *      svaddv_f16                  FADDV (H, P, Z.H)               6cy         1/cy
 *      svdup_f16                   DUP (Z.H, #imm)                 1cy         2/cy
 *      svwhilelt_b16               WHILELT (P.H, Xn, Xm)           2cy         1/cy
 *      svptrue_b16                 PTRUE (P.H, pattern)            1cy         2/cy
 *      svcnth                      CNTH (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  FP16 operations double the element count per vector compared to FP32, providing higher
 *  throughput at the cost of reduced precision. The FADDV reduction remains the bottleneck.
 */
#ifndef NK_DOT_SVEHALF_H
#define NK_DOT_SVEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVEHALF

#include "numkong/types.h"
#include "numkong/dot/serial.h"  // `nk_u1x8_popcount_`
#include "numkong/reduce/neon.h" // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

NK_PUBLIC void nk_dot_f16_svehalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_f32x = svdup_f32(0);
    do {
        svbool_t predicate_f32x = svwhilelt_b32((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat16_t a_f16x = svld1_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(a_scalars) + idx_scalars);
        svfloat16_t b_f16x = svld1_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(b_scalars) + idx_scalars);
        svfloat32_t a_f32x = svcvt_f32_f16_x(predicate_f32x, a_f16x);
        svfloat32_t b_f32x = svcvt_f32_f16_x(predicate_f32x, b_f16x);
        ab_f32x = svmla_f32_x(predicate_f32x, ab_f32x, a_f32x, b_f32x);
        idx_scalars += svcntw();
    } while (idx_scalars < count_scalars);
    *result = svaddv_f32(svptrue_b32(), ab_f32x);
}

NK_PUBLIC void nk_dot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *results) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_real_f32x = svdup_f32(0);
    svfloat32_t ab_imag_f32x = svdup_f32(0);
    do {
        svbool_t predicate_f32x = svwhilelt_b32((unsigned int)idx_scalars, (unsigned int)count_pairs);
        svfloat16x2_t a_f16x2 = svld2_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(a_pairs) + idx_scalars * 2);
        svfloat16x2_t b_f16x2 = svld2_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(b_pairs) + idx_scalars * 2);
        svfloat32_t a_real_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(a_f16x2, 0));
        svfloat32_t a_imag_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(a_f16x2, 1));
        svfloat32_t b_real_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(b_f16x2, 0));
        svfloat32_t b_imag_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(b_f16x2, 1));
        ab_real_f32x = svmla_f32_x(predicate_f32x, ab_real_f32x, a_real_f32x, b_real_f32x);
        ab_real_f32x = svmls_f32_x(predicate_f32x, ab_real_f32x, a_imag_f32x, b_imag_f32x);
        ab_imag_f32x = svmla_f32_x(predicate_f32x, ab_imag_f32x, a_real_f32x, b_imag_f32x);
        ab_imag_f32x = svmla_f32_x(predicate_f32x, ab_imag_f32x, a_imag_f32x, b_real_f32x);
        idx_scalars += svcntw();
    } while (idx_scalars < count_pairs);
    results->real = svaddv_f32(svptrue_b32(), ab_real_f32x);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_f32x);
}

NK_PUBLIC void nk_vdot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *results) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_real_f32x = svdup_f32(0);
    svfloat32_t ab_imag_f32x = svdup_f32(0);
    do {
        svbool_t predicate_f32x = svwhilelt_b32((unsigned int)idx_scalars, (unsigned int)count_pairs);
        svfloat16x2_t a_f16x2 = svld2_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(a_pairs) + idx_scalars * 2);
        svfloat16x2_t b_f16x2 = svld2_f16(predicate_f32x, (nk_f16_for_arm_simd_t const *)(b_pairs) + idx_scalars * 2);
        svfloat32_t a_real_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(a_f16x2, 0));
        svfloat32_t a_imag_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(a_f16x2, 1));
        svfloat32_t b_real_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(b_f16x2, 0));
        svfloat32_t b_imag_f32x = svcvt_f32_f16_x(predicate_f32x, svget2_f16(b_f16x2, 1));
        ab_real_f32x = svmla_f32_x(predicate_f32x, ab_real_f32x, a_real_f32x, b_real_f32x);
        ab_real_f32x = svmla_f32_x(predicate_f32x, ab_real_f32x, a_imag_f32x, b_imag_f32x);
        ab_imag_f32x = svmla_f32_x(predicate_f32x, ab_imag_f32x, a_real_f32x, b_imag_f32x);
        ab_imag_f32x = svmls_f32_x(predicate_f32x, ab_imag_f32x, a_imag_f32x, b_real_f32x);
        idx_scalars += svcntw();
    } while (idx_scalars < count_pairs);
    results->real = svaddv_f32(svptrue_b32(), ab_real_f32x);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_f32x);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM_
#endif // NK_DOT_SVEHALF_H
