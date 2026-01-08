/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/dot/sve.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

#include "numkong/types.h"
#include "numkong/dot/serial.h"  // `nk_popcount_u1`
#include "numkong/reduce/neon.h" // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_svehalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_size_t idx_scalars = 0;
    svfloat16_t ab_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat16_t a_vec = svld1_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(a_scalars + idx_scalars));
        svfloat16_t b_vec = svld1_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(b_scalars + idx_scalars));
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        idx_scalars += svcnth();
    } while (idx_scalars < count_scalars);
    nk_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    *result = ab;
}

NK_PUBLIC void nk_dot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat16_t ab_real_vec = svdup_f16(0);
    svfloat16_t ab_imag_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat16x2_t a_vec = svld2_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(a_pairs + idx_pairs));
        svfloat16x2_t b_vec = svld2_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f16(svptrue_b16(), ab_real_vec);
    results->imag = svaddv_f16(svptrue_b16(), ab_imag_vec);
}

NK_PUBLIC void nk_vdot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat16_t ab_real_vec = svdup_f16(0);
    svfloat16_t ab_imag_vec = svdup_f16(0);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat16x2_t a_vec = svld2_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(a_pairs + idx_pairs));
        svfloat16x2_t b_vec = svld2_f16(pg_vec, (nk_f16_for_arm_simd_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f16(svptrue_b16(), ab_real_vec);
    results->imag = svaddv_f16(svptrue_b16(), ab_imag_vec);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM_

#endif // NK_DOT_SVEHALF_H
