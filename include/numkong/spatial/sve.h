/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/spatial/sve.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat64_t d2_vec = svdupq_n_f64(0.0, 0.0);
    svbool_t pg_f64 = svptrue_b64();
    do {
        svbool_t pg_f32 = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_f32, a + i);
        svfloat32_t b_vec = svld1_f32(pg_f32, b + i);
        svfloat32_t diff_f32 = svsub_f32_x(pg_f32, a_vec, b_vec);
        // Widen lower half of f32 vector to f64 and accumulate
        svfloat64_t diff_f64 = svcvt_f64_f32_x(pg_f64, diff_f32);
        d2_vec = svmla_f64_x(pg_f64, d2_vec, diff_f64, diff_f64);
        i += svcntd();
    } while (i < n);
    nk_f64_t d2 = svaddv_f64(svptrue_b64(), d2_vec);
    *result = (nk_f32_t)d2;
}

NK_PUBLIC void nk_l2_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_sve(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    svbool_t pg_f64 = svptrue_b64();
    do {
        svbool_t pg_f32 = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_f32 = svld1_f32(pg_f32, a + i);
        svfloat32_t b_f32 = svld1_f32(pg_f32, b + i);
        // Widen lower half of f32 vectors to f64 and accumulate
        svfloat64_t a_f64 = svcvt_f64_f32_x(pg_f64, a_f32);
        svfloat64_t b_f64 = svcvt_f64_f32_x(pg_f64, b_f32);
        ab_vec = svmla_f64_x(pg_f64, ab_vec, a_f64, b_f64);
        a2_vec = svmla_f64_x(pg_f64, a2_vec, a_f64, a_f64);
        b2_vec = svmla_f64_x(pg_f64, b2_vec, b_f64, b_f64);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab = svaddv_f64(svptrue_b64(), ab_vec);
    nk_f64_t a2 = svaddv_f64(svptrue_b64(), a2_vec);
    nk_f64_t b2 = svaddv_f64(svptrue_b64(), b2_vec);
    *result = (nk_f32_t)nk_angular_normalize_f64_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_l2sq_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Neumaier compensated summation for numerical stability
    nk_size_t i = 0;
    svfloat64_t sum_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t compensation_vec = svdupq_n_f64(0.0, 0.0);
    svbool_t pg_true = svptrue_b64();
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        svfloat64_t diff_vec = svsub_f64_x(pg_vec, a_vec, b_vec);
        svfloat64_t diff_sq_vec = svmul_f64_x(pg_vec, diff_vec, diff_vec);
        // Neumaier: t = sum + x
        svfloat64_t t_vec = svadd_f64_x(pg_vec, sum_vec, diff_sq_vec);
        svfloat64_t abs_sum_vec = svabs_f64_x(pg_vec, sum_vec);
        // diff_sq is already non-negative (it's a square), so svabs is unnecessary
        svbool_t sum_ge_x = svcmpge_f64(pg_vec, abs_sum_vec, diff_sq_vec);
        // When |sum| >= |x|: comp += (sum - t) + x; when |x| > |sum|: comp += (x - t) + sum
        svfloat64_t comp_sum_large = svadd_f64_x(pg_vec, svsub_f64_x(pg_vec, sum_vec, t_vec), diff_sq_vec);
        svfloat64_t comp_x_large = svadd_f64_x(pg_vec, svsub_f64_x(pg_vec, diff_sq_vec, t_vec), sum_vec);
        svfloat64_t comp_update = svsel_f64(sum_ge_x, comp_sum_large, comp_x_large);
        compensation_vec = svadd_f64_x(pg_vec, compensation_vec, comp_update);
        sum_vec = t_vec;
        i += svcntd();
    } while (i < n);
    nk_f64_t d2 = svaddv_f64(pg_true, svadd_f64_x(pg_true, sum_vec, compensation_vec));
    *result = d2;
}

NK_PUBLIC void nk_l2_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_sve(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    nk_size_t i = 0;
    svfloat64_t ab_sum_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t ab_compensation_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    svbool_t pg_true = svptrue_b64();
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        // TwoProd for ab: product = a*b, error = fma(a,b,-product) = -(product - a*b)
        svfloat64_t product_vec = svmul_f64_x(pg_vec, a_vec, b_vec);
        svfloat64_t product_error_vec = svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_vec, b_vec));
        // TwoSum: (t, q) = TwoSum(sum, product)
        svfloat64_t t_vec = svadd_f64_x(pg_vec, ab_sum_vec, product_vec);
        svfloat64_t z_vec = svsub_f64_x(pg_vec, t_vec, ab_sum_vec);
        svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                                                svsub_f64_x(pg_vec, ab_sum_vec, svsub_f64_x(pg_vec, t_vec, z_vec)),
                                                svsub_f64_x(pg_vec, product_vec, z_vec));
        ab_sum_vec = t_vec;
        ab_compensation_vec = svadd_f64_x(pg_vec, ab_compensation_vec,
                                          svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        // Simple FMA for self-products (no cancellation)
        a2_vec = svmla_f64_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f64_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab = svaddv_f64(pg_true, svadd_f64_x(pg_true, ab_sum_vec, ab_compensation_vec));
    nk_f64_t a2 = svaddv_f64(pg_true, a2_vec);
    nk_f64_t b2 = svaddv_f64(pg_true, b2_vec);
    *result = nk_angular_normalize_f64_neon_(ab, a2, b2);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_SVE_H
