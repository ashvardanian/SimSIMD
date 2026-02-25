/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE.
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f32                   LD1W (Z.S, P/Z, [Xn])           4-6cy       2/cy
 *      svsub_f32_x                 FSUB (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svmla_f32_x                 FMLA (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svaddv_f32                  FADDV (S, P, Z.S)               6cy         1/cy
 *      svdupq_n_f32                DUP (Z.S, #imm)                 1cy         2/cy
 *      svwhilelt_b32               WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svptrue_b32                 PTRUE (P.S, pattern)            1cy         2/cy
 *      svcntw                      CNTW (Xd)                       1cy         2/cy
 *      svld1_f64                   LD1D (Z.D, P/Z, [Xn])           4-6cy       2/cy
 *      svsub_f64_x                 FSUB (Z.D, P/M, Z.D, Z.D)       3cy         2/cy
 *      svmla_f64_x                 FMLA (Z.D, P/M, Z.D, Z.D)       4cy         2/cy
 *      svaddv_f64                  FADDV (D, P, Z.D)               6cy         1/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  Spatial operations like L2 distance and angular similarity benefit from SVE's fused
 *  multiply-add instructions. The FADDV reduction dominates the critical path for short vectors.
 */
#ifndef NK_SPATIAL_SVE_H
#define NK_SPATIAL_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`
#include "numkong/dot/sve.h"      // `nk_dot_stable_sum_f64_sve_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

NK_PUBLIC void nk_sqeuclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat64_t dist_sq_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_f64x = svptrue_b64();
    do {
        svbool_t predicate_f32x = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_f32x = svld1_f32(predicate_f32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_f32x, b + i);
        svfloat32_t diff_f32x = svsub_f32_x(predicate_f32x, a_f32x, b_f32x);
        // Widen lower half of f32 vector to f64 and accumulate
        svfloat64_t diff_f64x = svcvt_f64_f32_x(predicate_all_f64x, diff_f32x);
        dist_sq_f64x = svmla_f64_x(predicate_all_f64x, dist_sq_f64x, diff_f64x, diff_f64x);
        i += svcntd();
    } while (i < n);
    nk_f64_t dist_sq_f64 = svaddv_f64(svptrue_b64(), dist_sq_f64x);
    *result = (nk_f32_t)dist_sq_f64;
}

NK_PUBLIC void nk_euclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f32_sve(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat64_t ab_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_f64x = svptrue_b64();
    do {
        svbool_t predicate_f32x = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_f32x = svld1_f32(predicate_f32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_f32x, b + i);
        // Widen lower half of f32 vectors to f64 and accumulate
        svfloat64_t a_f64x = svcvt_f64_f32_x(predicate_all_f64x, a_f32x);
        svfloat64_t b_f64x = svcvt_f64_f32_x(predicate_all_f64x, b_f32x);
        ab_f64x = svmla_f64_x(predicate_all_f64x, ab_f64x, a_f64x, b_f64x);
        a2_f64x = svmla_f64_x(predicate_all_f64x, a2_f64x, a_f64x, a_f64x);
        b2_f64x = svmla_f64_x(predicate_all_f64x, b2_f64x, b_f64x, b_f64x);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab_f64 = svaddv_f64(svptrue_b64(), ab_f64x);
    nk_f64_t a2_f64 = svaddv_f64(svptrue_b64(), a2_f64x);
    nk_f64_t b2_f64 = svaddv_f64(svptrue_b64(), b2_f64x);
    *result = (nk_f32_t)nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Neumaier compensated summation for numerical stability
    nk_size_t i = 0;
    svfloat64_t sum_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t compensation_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_f64x = svptrue_b64();
    do {
        svbool_t predicate_f64x = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_f64x = svld1_f64(predicate_f64x, a + i);
        svfloat64_t b_f64x = svld1_f64(predicate_f64x, b + i);
        svfloat64_t diff_f64x = svsub_f64_x(predicate_f64x, a_f64x, b_f64x);
        svfloat64_t diff_sq_f64x = svmul_f64_x(predicate_f64x, diff_f64x, diff_f64x);
        // Neumaier: t = sum + x
        svfloat64_t t_f64x = svadd_f64_x(predicate_f64x, sum_f64x, diff_sq_f64x);
        svfloat64_t abs_sum_f64x = svabs_f64_x(predicate_f64x, sum_f64x);
        // diff_sq is already non-negative (it's a square), so svabs is unnecessary
        svbool_t sum_ge_x_f64x = svcmpge_f64(predicate_f64x, abs_sum_f64x, diff_sq_f64x);
        // When |sum| >= |x|: comp += (sum - t) + x; when |x| > |sum|: comp += (x - t) + sum
        svfloat64_t comp_sum_large_f64x = svadd_f64_x(predicate_f64x, svsub_f64_x(predicate_f64x, sum_f64x, t_f64x),
                                                      diff_sq_f64x);
        svfloat64_t comp_x_large_f64x = svadd_f64_x(predicate_f64x, svsub_f64_x(predicate_f64x, diff_sq_f64x, t_f64x),
                                                    sum_f64x);
        svfloat64_t comp_update_f64x = svsel_f64(sum_ge_x_f64x, comp_sum_large_f64x, comp_x_large_f64x);
        compensation_f64x = svadd_f64_x(predicate_f64x, compensation_f64x, comp_update_f64x);
        sum_f64x = t_f64x;
        i += svcntd();
    } while (i < n);
    *result = nk_dot_stable_sum_f64_sve_(predicate_all_f64x, sum_f64x, compensation_f64x);
}

NK_PUBLIC void nk_euclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_sve(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    nk_size_t i = 0;
    svfloat64_t ab_sum_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t ab_compensation_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_f64x = svptrue_b64();
    do {
        svbool_t predicate_f64x = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_f64x = svld1_f64(predicate_f64x, a + i);
        svfloat64_t b_f64x = svld1_f64(predicate_f64x, b + i);
        // TwoProd for ab: product = a*b, error = fma(a,b,-product) = -(product - a*b)
        svfloat64_t product_f64x = svmul_f64_x(predicate_f64x, a_f64x, b_f64x);
        svfloat64_t product_error_f64x = svneg_f64_x(predicate_f64x,
                                                     svnmls_f64_x(predicate_f64x, product_f64x, a_f64x, b_f64x));
        // TwoSum: (tentative_sum, sum_error) = TwoSum(sum, product)
        svfloat64_t tentative_sum_f64x = svadd_f64_x(predicate_f64x, ab_sum_f64x, product_f64x);
        svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_f64x, tentative_sum_f64x, ab_sum_f64x);
        svfloat64_t sum_error_f64x = svadd_f64_x(
            predicate_f64x,
            svsub_f64_x(predicate_f64x, ab_sum_f64x,
                        svsub_f64_x(predicate_f64x, tentative_sum_f64x, virtual_addend_f64x)),
            svsub_f64_x(predicate_f64x, product_f64x, virtual_addend_f64x));
        ab_sum_f64x = tentative_sum_f64x;
        ab_compensation_f64x = svadd_f64_x(predicate_f64x, ab_compensation_f64x,
                                           svadd_f64_x(predicate_f64x, sum_error_f64x, product_error_f64x));
        // Simple FMA for self-products (no cancellation)
        a2_f64x = svmla_f64_x(predicate_f64x, a2_f64x, a_f64x, a_f64x);
        b2_f64x = svmla_f64_x(predicate_f64x, b2_f64x, b_f64x, b_f64x);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab_f64 = nk_dot_stable_sum_f64_sve_(predicate_all_f64x, ab_sum_f64x, ab_compensation_f64x);
    nk_f64_t a2_f64 = svaddv_f64(predicate_all_f64x, a2_f64x);
    nk_f64_t b2_f64 = svaddv_f64(predicate_all_f64x, b2_f64x);
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_
#endif // NK_SPATIAL_SVE_H
