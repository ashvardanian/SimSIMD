/**
 *  @file   arm_sve_f32.h
 *  @brief  Arm SVE implementation of the most common similarity metrics for 32-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `f16` for both storage and `f32` for accumulation.
 *  - Requires compiler capabilities: +sve.
 */
#include <arm_sve.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_sve_f32_l2sq(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svbool_t pg_vec = svwhilelt_b32(i, d);
    do {
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        svfloat32_t a_minus_b_vec = svsub_f32_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntw();
        pg_vec = svwhilelt_b32(i, d);
    } while (svptest_any(svptrue_b32(), pg_vec));
    simsimd_f32_t d2 = svaddv_f32(svptrue_b32(), d2_vec);
    return d2;
}

inline static simsimd_f32_t simsimd_sve_f32_ip(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svbool_t pg_vec = svwhilelt_b32(i, d);
    do {
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        i += svcntw();
        pg_vec = svwhilelt_b32(i, d);
    } while (svptest_any(svptrue_b32(), pg_vec));
    return 1 - svaddv_f32(svptrue_b32(), ab_vec);
}

inline static simsimd_f32_t simsimd_sve_f32_cos(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    simsimd_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svbool_t pg_vec = svwhilelt_b32(i, d);
    do {
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntw();
        pg_vec = svwhilelt_b32(i, d);
    } while (svptest_any(svptrue_b32(), pg_vec));
    simsimd_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    simsimd_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    simsimd_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    return 1 - ab / (sqrtf(a2) * sqrtf(b2));
}

#ifdef __cplusplus
} // extern "C"
#endif