/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/dot/sve.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SVE_H
#define NK_DOT_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/dot/serial.h"  // `nk_popcount_b8`
#include "numkong/reduce/neon.h" // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f32_sve(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                              nk_f32_t *result) {
    nk_size_t idx_scalars = 0;
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

NK_PUBLIC void nk_dot_f32c_sve(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f32c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat32_t ab_real_vec = svdup_f32(0.f);
    svfloat32_t ab_imag_vec = svdup_f32(0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat32x2_t a_vec = svld2_f32(pg_vec, (nk_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_vec = svld2_f32(pg_vec, (nk_f32_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f32(svptrue_b32(), ab_real_vec);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_vec);
}

NK_PUBLIC void nk_vdot_f32c_sve(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f32c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat32_t ab_real_vec = svdup_f32(0.f);
    svfloat32_t ab_imag_vec = svdup_f32(0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat32x2_t a_vec = svld2_f32(pg_vec, (nk_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_vec = svld2_f32(pg_vec, (nk_f32_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f32(svptrue_b32(), ab_real_vec);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_vec);
}

NK_PUBLIC void nk_dot_f64_sve(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    nk_size_t idx_scalars = 0;
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

NK_PUBLIC void nk_dot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat64_t ab_real_vec = svdup_f64(0.);
    svfloat64_t ab_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (nk_f64_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f64(svptrue_b64(), ab_real_vec);
    results->imag = svaddv_f64(svptrue_b64(), ab_imag_vec);
}

NK_PUBLIC void nk_vdot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat64_t ab_real_vec = svdup_f64(0.);
    svfloat64_t ab_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (nk_f64_t const *)(b_pairs + idx_pairs));
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
    results->real = svaddv_f64(svptrue_b64(), ab_real_vec);
    results->imag = svaddv_f64(svptrue_b64(), ab_imag_vec);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_DOT_SVE_H
