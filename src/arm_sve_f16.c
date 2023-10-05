/**
 *  @file   arm_sve_f16.h
 *  @brief  Arm SVE implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `f16` for both storage and `f32` for accumulation.
 *  - Requires compiler capabilities: +sve+fp16.
 */
#include <arm_neon.h>
#include <arm_sve.h>

#include "types.h"

simsimd_f32_t simsimd_sve_f16_l2sq(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat16_t d2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svbool_t pg_vec = svwhilelt_b16(i, d);
    simsimd_f16_t const* a = (simsimd_f16_t const*)(a_enum);
    simsimd_f16_t const* b = (simsimd_f16_t const*)(b_enum);
    do {
        svfloat16_t a_vec = svld1_f16(pg_vec, (float16_t const*)a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, (float16_t const*)b + i);
        svfloat16_t a_minus_b_vec = svsub_f16_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f16_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcnth();
        pg_vec = svwhilelt_b16(i, d);
    } while (svptest_any(svptrue_b16(), pg_vec));
    float16_t d2_f16 = svaddv_f16(svptrue_b16(), d2_vec);
    return 1 - d2_f16;
}

simsimd_f32_t simsimd_sve_f16_ip(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svbool_t pg_vec = svwhilelt_b16(i, d);
    simsimd_f16_t const* a = (simsimd_f16_t const*)(a_enum);
    simsimd_f16_t const* b = (simsimd_f16_t const*)(b_enum);
    do {
        svfloat16_t a_vec = svld1_f16(pg_vec, (float16_t const*)a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, (float16_t const*)b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        i += svcnth();
        pg_vec = svwhilelt_b16(i, d);
    } while (svptest_any(svptrue_b16(), pg_vec));
    simsimd_f16_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    return 1 - ab;
}

simsimd_f32_t simsimd_sve_f16_cos(simsimd_f16_t const* a_enum, simsimd_f16_t const* b_enum, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t a2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t b2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svbool_t pg_vec = svwhilelt_b16(i, d);
    simsimd_f16_t const* a = (simsimd_f16_t const*)(a_enum);
    simsimd_f16_t const* b = (simsimd_f16_t const*)(b_enum);
    do {
        svfloat16_t a_vec = svld1_f16(pg_vec, (float16_t const*)a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, (float16_t const*)b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f16_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f16_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcnth();
        pg_vec = svwhilelt_b16(i, d);
    } while (svptest_any(svptrue_b16(), pg_vec));

    simsimd_f16_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    simsimd_f16_t a2 = svaddv_f16(svptrue_b16(), a2_vec);
    simsimd_f16_t b2 = svaddv_f16(svptrue_b16(), b2_vec);

    // Avoid `simsimd_approximate_inverse_square_root` on Arm NEON
    simsimd_f32_t a2_b2_arr[2] = {a2, b2};
    float32x2_t a2_b2 = vld1_f32(a2_b2_arr);
    a2_b2 = vrsqrte_f32(a2_b2);
    vst1_f32(a2_b2_arr, a2_b2);
    return 1 - ab * a2_b2_arr[0] * a2_b2_arr[1];
}
