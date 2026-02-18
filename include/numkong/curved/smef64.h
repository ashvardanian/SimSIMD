/**
 *  @brief SIMD-accelerated Curved Space Similarity for SME F64.
 *  @file include/numkong/curved/smef64.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance using ARM SME (streaming SVE):
 *  - f32 inputs with f64 accumulation for higher precision
 *  - f64 inputs with Dot2 (Ogita-Rump-Oishi) compensated summation for ~2x precision
 *
 *  @section precision Precision Strategy
 *
 *  For f32 inputs: Upcast to f64 before accumulation provides ~2x precision headroom,
 *  avoiding catastrophic cancellation in the inner products.
 *
 *  For f64 inputs: Dot2 algorithm uses TwoProd and TwoSum error-free transformations
 *  to capture rounding errors in compensation terms, achieving near double-double precision.
 *
 *  @section sme_notes SME Streaming SVE Notes
 *
 *  On Apple M4, SVE instructions are only available inside SME streaming mode.
 *  Functions using SVE intrinsics must be marked `__arm_locally_streaming` to
 *  generate SMSTART/SMSTOP at entry/exit. NEON intrinsics cannot be called from
 *  streaming mode, so Mahalanobis functions split into a streaming kernel (SVE)
 *  and a non-streaming wrapper (NEON sqrt).
 *
 *  SVE is vector-length agnostic (VLA). Key patterns:
 *  - svcntd() returns number of f64 elements per vector (varies by hardware)
 *  - svwhilelt_b64(i, n) creates predicate for partial vectors at loop tail
 *  - All operations use predicates for clean handling of arbitrary lengths
 */
#ifndef NK_CURVED_SMEF64_H
#define NK_CURVED_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEF64

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f64_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

__arm_locally_streaming NK_PUBLIC void nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                                              nk_size_t n, nk_f32_t *result) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_f64_t outer_sum_f64 = 0.0;
    nk_size_t row = 0;

    // 4-row fast path: share b[j] load across 4 rows for bandwidth savings
    for (; row + 4 <= n; row += 4) {
        nk_f64_t a0_f64 = (nk_f64_t)a[row + 0], a1_f64 = (nk_f64_t)a[row + 1];
        nk_f64_t a2_f64 = (nk_f64_t)a[row + 2], a3_f64 = (nk_f64_t)a[row + 3];
        svfloat64_t cb_j0_f64 = svdup_f64(0), cb_j1_f64 = svdup_f64(0);
        svfloat64_t cb_j2_f64 = svdup_f64(0), cb_j3_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);
        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t b_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(b + j))));
            cb_j0_f64 = svmla_f64_x(
                predicate_tail_b64, cb_j0_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 0) * n + j)))),
                b_f64);
            cb_j1_f64 = svmla_f64_x(
                predicate_tail_b64, cb_j1_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 1) * n + j)))),
                b_f64);
            cb_j2_f64 = svmla_f64_x(
                predicate_tail_b64, cb_j2_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 2) * n + j)))),
                b_f64);
            cb_j3_f64 = svmla_f64_x(
                predicate_tail_b64, cb_j3_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 3) * n + j)))),
                b_f64);
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }
        outer_sum_f64 += a0_f64 * svaddv_f64(predicate_body_b64, cb_j0_f64) +
                         a1_f64 * svaddv_f64(predicate_body_b64, cb_j1_f64) +
                         a2_f64 * svaddv_f64(predicate_body_b64, cb_j2_f64) +
                         a3_f64 * svaddv_f64(predicate_body_b64, cb_j3_f64);
    }

    // 1-row tail
    for (; row < n; ++row) {
        nk_f64_t a_row_f64 = (nk_f64_t)a[row];
        svfloat64_t inner_sum_f64 = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);
        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t b_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(b + j))));
            svfloat64_t c_f64 = svcvt_f64_f32_x(
                predicate_tail_b64,
                svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + row * n + j))));
            inner_sum_f64 = svmla_f64_x(predicate_tail_b64, inner_sum_f64, c_f64, b_f64);
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }
        outer_sum_f64 += a_row_f64 * svaddv_f64(predicate_body_b64, inner_sum_f64);
    }

    *result = (nk_f32_t)outer_sum_f64;
}

__arm_locally_streaming static inline nk_f64_t nk_mahalanobis_f32_smef64_kernel_(nk_f32_t const *a, nk_f32_t const *b,
                                                                                 nk_f32_t const *c, nk_size_t n) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_f64_t outer_sum_f64 = 0.0;
    nk_size_t row = 0;

    // 4-row fast path: share diff_col_f64 across 4 rows
    for (; row + 4 <= n; row += 4) {
        nk_f64_t diff0_f64 = (nk_f64_t)a[row + 0] - (nk_f64_t)b[row + 0];
        nk_f64_t diff1_f64 = (nk_f64_t)a[row + 1] - (nk_f64_t)b[row + 1];
        nk_f64_t diff2_f64 = (nk_f64_t)a[row + 2] - (nk_f64_t)b[row + 2];
        nk_f64_t diff3_f64 = (nk_f64_t)a[row + 3] - (nk_f64_t)b[row + 3];
        svfloat64_t cdiff_j0_f64 = svdup_f64(0), cdiff_j1_f64 = svdup_f64(0);
        svfloat64_t cdiff_j2_f64 = svdup_f64(0), cdiff_j3_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);
        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t a_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(a + j))));
            svfloat64_t b_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(b + j))));
            svfloat64_t diff_col_f64 = svsub_f64_x(predicate_tail_b64, a_f64, b_f64);
            cdiff_j0_f64 = svmla_f64_x(
                predicate_tail_b64, cdiff_j0_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 0) * n + j)))),
                diff_col_f64);
            cdiff_j1_f64 = svmla_f64_x(
                predicate_tail_b64, cdiff_j1_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 1) * n + j)))),
                diff_col_f64);
            cdiff_j2_f64 = svmla_f64_x(
                predicate_tail_b64, cdiff_j2_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 2) * n + j)))),
                diff_col_f64);
            cdiff_j3_f64 = svmla_f64_x(
                predicate_tail_b64, cdiff_j3_f64,
                svcvt_f64_f32_x(
                    predicate_tail_b64,
                    svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + (row + 3) * n + j)))),
                diff_col_f64);
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }
        outer_sum_f64 += diff0_f64 * svaddv_f64(predicate_body_b64, cdiff_j0_f64) +
                         diff1_f64 * svaddv_f64(predicate_body_b64, cdiff_j1_f64) +
                         diff2_f64 * svaddv_f64(predicate_body_b64, cdiff_j2_f64) +
                         diff3_f64 * svaddv_f64(predicate_body_b64, cdiff_j3_f64);
    }

    // 1-row tail
    for (; row < n; ++row) {
        nk_f64_t diff_row_f64 = (nk_f64_t)a[row] - (nk_f64_t)b[row];
        svfloat64_t inner_sum_f64 = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);
        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t a_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(a + j))));
            svfloat64_t b_f64 = svcvt_f64_f32_x(
                predicate_tail_b64, svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(b + j))));
            svfloat64_t diff_col_f64 = svsub_f64_x(predicate_tail_b64, a_f64, b_f64);
            svfloat64_t c_f64 = svcvt_f64_f32_x(
                predicate_tail_b64,
                svreinterpret_f32_u64(svld1uw_u64(predicate_tail_b64, (nk_u32_t const *)(c + row * n + j))));
            inner_sum_f64 = svmla_f64_x(predicate_tail_b64, inner_sum_f64, c_f64, diff_col_f64);
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }
        outer_sum_f64 += diff_row_f64 * svaddv_f64(predicate_body_b64, inner_sum_f64);
    }

    return outer_sum_f64;
}

NK_PUBLIC void nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f32_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f32_smef64_kernel_(a, b, c, n);
    *result = nk_f32_sqrt_neon((nk_f32_t)(quadratic > 0 ? quadratic : 0));
}

__arm_locally_streaming NK_PUBLIC void nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c,
                                                              nk_size_t n, nk_f64_t *result) {
    svbool_t predicate_body_b64 = svptrue_b64();

    // Outer loop accumulators with Dot2 compensation
    nk_f64_t outer_sum_f64 = 0.0;
    nk_f64_t outer_comp_f64 = 0.0;
    nk_size_t row = 0;

    // 2-row fast path: share b_f64 load across both rows
    for (; row + 2 <= n; row += 2) {
        nk_f64_t a0_f64 = a[row + 0], a1_f64 = a[row + 1];

        svfloat64_t cb_j0_f64 = svdup_f64(0), cb_j0_comp_f64 = svdup_f64(0);
        svfloat64_t cb_j1_f64 = svdup_f64(0), cb_j1_comp_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t b_f64 = svld1_f64(predicate_tail_b64, b + j);
            svfloat64_t c0_f64 = svld1_f64(predicate_tail_b64, c + (row + 0) * n + j);
            svfloat64_t c1_f64 = svld1_f64(predicate_tail_b64, c + (row + 1) * n + j);

            // Row 0: TwoProd + TwoSum
            svfloat64_t product0_f64 = svmul_f64_x(predicate_tail_b64, c0_f64, b_f64);
            svfloat64_t prod_error0_f64 = svneg_f64_x(predicate_tail_b64,
                                                      svnmls_f64_x(predicate_tail_b64, product0_f64, c0_f64, b_f64));
            svfloat64_t t0_f64 = svadd_f64_x(predicate_tail_b64, cb_j0_f64, product0_f64);
            svfloat64_t z0_f64 = svsub_f64_x(predicate_tail_b64, t0_f64, cb_j0_f64);
            svfloat64_t sum_error0_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, cb_j0_f64, svsub_f64_x(predicate_tail_b64, t0_f64, z0_f64)),
                svsub_f64_x(predicate_tail_b64, product0_f64, z0_f64));
            cb_j0_f64 = t0_f64;
            cb_j0_comp_f64 = svadd_f64_x(predicate_tail_b64, cb_j0_comp_f64,
                                         svadd_f64_x(predicate_tail_b64, sum_error0_f64, prod_error0_f64));

            // Row 1: TwoProd + TwoSum
            svfloat64_t product1_f64 = svmul_f64_x(predicate_tail_b64, c1_f64, b_f64);
            svfloat64_t prod_error1_f64 = svneg_f64_x(predicate_tail_b64,
                                                      svnmls_f64_x(predicate_tail_b64, product1_f64, c1_f64, b_f64));
            svfloat64_t t1_f64 = svadd_f64_x(predicate_tail_b64, cb_j1_f64, product1_f64);
            svfloat64_t z1_f64 = svsub_f64_x(predicate_tail_b64, t1_f64, cb_j1_f64);
            svfloat64_t sum_error1_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, cb_j1_f64, svsub_f64_x(predicate_tail_b64, t1_f64, z1_f64)),
                svsub_f64_x(predicate_tail_b64, product1_f64, z1_f64));
            cb_j1_f64 = t1_f64;
            cb_j1_comp_f64 = svadd_f64_x(predicate_tail_b64, cb_j1_comp_f64,
                                         svadd_f64_x(predicate_tail_b64, sum_error1_f64, prod_error1_f64));

            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        // Reduce and accumulate both rows
        for (int r = 0; r < 2; ++r) {
            nk_f64_t a_row_f64 = r == 0 ? a0_f64 : a1_f64;
            nk_f64_t inner_result_f64 = svaddv_f64(predicate_body_b64, r == 0 ? cb_j0_f64 : cb_j1_f64) +
                                        svaddv_f64(predicate_body_b64, r == 0 ? cb_j0_comp_f64 : cb_j1_comp_f64);
            nk_f64_t outer_product_f64 = a_row_f64 * inner_result_f64;
            nk_f64_t outer_product_error_f64 = a_row_f64 * inner_result_f64 - outer_product_f64;
            nk_f64_t t_f64 = outer_sum_f64 + outer_product_f64;
            nk_f64_t z_f64 = t_f64 - outer_sum_f64;
            nk_f64_t sum_error_f64 = (outer_sum_f64 - (t_f64 - z_f64)) + (outer_product_f64 - z_f64);
            outer_sum_f64 = t_f64;
            outer_comp_f64 += sum_error_f64 + outer_product_error_f64;
        }
    }

    // 1-row tail
    for (; row < n; ++row) {
        nk_f64_t a_row_f64 = a[row];
        svfloat64_t inner_sum_f64 = svdup_f64(0.0);
        svfloat64_t inner_comp_f64 = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t b_f64 = svld1_f64(predicate_tail_b64, b + j);
            svfloat64_t c_f64 = svld1_f64(predicate_tail_b64, c + row * n + j);
            svfloat64_t product_f64 = svmul_f64_x(predicate_tail_b64, c_f64, b_f64);
            svfloat64_t product_error_f64 = svneg_f64_x(predicate_tail_b64,
                                                        svnmls_f64_x(predicate_tail_b64, product_f64, c_f64, b_f64));
            svfloat64_t t_f64 = svadd_f64_x(predicate_tail_b64, inner_sum_f64, product_f64);
            svfloat64_t z_f64 = svsub_f64_x(predicate_tail_b64, t_f64, inner_sum_f64);
            svfloat64_t sum_error_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, inner_sum_f64, svsub_f64_x(predicate_tail_b64, t_f64, z_f64)),
                svsub_f64_x(predicate_tail_b64, product_f64, z_f64));
            inner_sum_f64 = t_f64;
            inner_comp_f64 = svadd_f64_x(predicate_tail_b64, inner_comp_f64,
                                         svadd_f64_x(predicate_tail_b64, sum_error_f64, product_error_f64));
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        nk_f64_t inner_result_f64 = svaddv_f64(predicate_body_b64, inner_sum_f64) +
                                    svaddv_f64(predicate_body_b64, inner_comp_f64);
        nk_f64_t outer_product_f64 = a_row_f64 * inner_result_f64;
        nk_f64_t outer_product_error_f64 = a_row_f64 * inner_result_f64 - outer_product_f64;
        nk_f64_t t_f64 = outer_sum_f64 + outer_product_f64;
        nk_f64_t z_f64 = t_f64 - outer_sum_f64;
        nk_f64_t sum_error_f64 = (outer_sum_f64 - (t_f64 - z_f64)) + (outer_product_f64 - z_f64);
        outer_sum_f64 = t_f64;
        outer_comp_f64 += sum_error_f64 + outer_product_error_f64;
    }

    *result = outer_sum_f64 + outer_comp_f64;
}

__arm_locally_streaming static inline nk_f64_t nk_mahalanobis_f64_smef64_kernel_(nk_f64_t const *a, nk_f64_t const *b,
                                                                                 nk_f64_t const *c, nk_size_t n) {
    svbool_t predicate_body_b64 = svptrue_b64();

    // Outer loop accumulators with Dot2 compensation
    nk_f64_t outer_sum_f64 = 0.0;
    nk_f64_t outer_comp_f64 = 0.0;
    nk_size_t row = 0;

    // 2-row fast path: share diff_col_f64 across both rows
    for (; row + 2 <= n; row += 2) {
        nk_f64_t diff0_f64 = a[row + 0] - b[row + 0];
        nk_f64_t diff1_f64 = a[row + 1] - b[row + 1];

        svfloat64_t cdiff_j0_f64 = svdup_f64(0), cdiff_j0_comp_f64 = svdup_f64(0);
        svfloat64_t cdiff_j1_f64 = svdup_f64(0), cdiff_j1_comp_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t a_f64 = svld1_f64(predicate_tail_b64, a + j);
            svfloat64_t b_f64 = svld1_f64(predicate_tail_b64, b + j);
            svfloat64_t diff_col_f64 = svsub_f64_x(predicate_tail_b64, a_f64, b_f64);
            svfloat64_t c0_f64 = svld1_f64(predicate_tail_b64, c + (row + 0) * n + j);
            svfloat64_t c1_f64 = svld1_f64(predicate_tail_b64, c + (row + 1) * n + j);

            // Row 0: TwoProd + TwoSum
            svfloat64_t product0_f64 = svmul_f64_x(predicate_tail_b64, c0_f64, diff_col_f64);
            svfloat64_t prod_error0_f64 = svneg_f64_x(
                predicate_tail_b64, svnmls_f64_x(predicate_tail_b64, product0_f64, c0_f64, diff_col_f64));
            svfloat64_t t0_f64 = svadd_f64_x(predicate_tail_b64, cdiff_j0_f64, product0_f64);
            svfloat64_t z0_f64 = svsub_f64_x(predicate_tail_b64, t0_f64, cdiff_j0_f64);
            svfloat64_t sum_error0_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, cdiff_j0_f64, svsub_f64_x(predicate_tail_b64, t0_f64, z0_f64)),
                svsub_f64_x(predicate_tail_b64, product0_f64, z0_f64));
            cdiff_j0_f64 = t0_f64;
            cdiff_j0_comp_f64 = svadd_f64_x(predicate_tail_b64, cdiff_j0_comp_f64,
                                            svadd_f64_x(predicate_tail_b64, sum_error0_f64, prod_error0_f64));

            // Row 1: TwoProd + TwoSum
            svfloat64_t product1_f64 = svmul_f64_x(predicate_tail_b64, c1_f64, diff_col_f64);
            svfloat64_t prod_error1_f64 = svneg_f64_x(
                predicate_tail_b64, svnmls_f64_x(predicate_tail_b64, product1_f64, c1_f64, diff_col_f64));
            svfloat64_t t1_f64 = svadd_f64_x(predicate_tail_b64, cdiff_j1_f64, product1_f64);
            svfloat64_t z1_f64 = svsub_f64_x(predicate_tail_b64, t1_f64, cdiff_j1_f64);
            svfloat64_t sum_error1_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, cdiff_j1_f64, svsub_f64_x(predicate_tail_b64, t1_f64, z1_f64)),
                svsub_f64_x(predicate_tail_b64, product1_f64, z1_f64));
            cdiff_j1_f64 = t1_f64;
            cdiff_j1_comp_f64 = svadd_f64_x(predicate_tail_b64, cdiff_j1_comp_f64,
                                            svadd_f64_x(predicate_tail_b64, sum_error1_f64, prod_error1_f64));

            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        // Reduce and accumulate both rows
        for (int r = 0; r < 2; ++r) {
            nk_f64_t diff_f64 = r == 0 ? diff0_f64 : diff1_f64;
            nk_f64_t inner_result_f64 = svaddv_f64(predicate_body_b64, r == 0 ? cdiff_j0_f64 : cdiff_j1_f64) +
                                        svaddv_f64(predicate_body_b64, r == 0 ? cdiff_j0_comp_f64 : cdiff_j1_comp_f64);
            nk_f64_t outer_product_f64 = diff_f64 * inner_result_f64;
            nk_f64_t outer_product_error_f64 = diff_f64 * inner_result_f64 - outer_product_f64;
            nk_f64_t t_f64 = outer_sum_f64 + outer_product_f64;
            nk_f64_t z_f64 = t_f64 - outer_sum_f64;
            nk_f64_t sum_error_f64 = (outer_sum_f64 - (t_f64 - z_f64)) + (outer_product_f64 - z_f64);
            outer_sum_f64 = t_f64;
            outer_comp_f64 += sum_error_f64 + outer_product_error_f64;
        }
    }

    // 1-row tail
    for (; row < n; ++row) {
        nk_f64_t diff_row_f64 = a[row] - b[row];
        svfloat64_t inner_sum_f64 = svdup_f64(0.0);
        svfloat64_t inner_comp_f64 = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            svfloat64_t a_f64 = svld1_f64(predicate_tail_b64, a + j);
            svfloat64_t b_f64 = svld1_f64(predicate_tail_b64, b + j);
            svfloat64_t diff_col_f64 = svsub_f64_x(predicate_tail_b64, a_f64, b_f64);
            svfloat64_t c_f64 = svld1_f64(predicate_tail_b64, c + row * n + j);
            svfloat64_t product_f64 = svmul_f64_x(predicate_tail_b64, c_f64, diff_col_f64);
            svfloat64_t product_error_f64 = svneg_f64_x(
                predicate_tail_b64, svnmls_f64_x(predicate_tail_b64, product_f64, c_f64, diff_col_f64));
            svfloat64_t t_f64 = svadd_f64_x(predicate_tail_b64, inner_sum_f64, product_f64);
            svfloat64_t z_f64 = svsub_f64_x(predicate_tail_b64, t_f64, inner_sum_f64);
            svfloat64_t sum_error_f64 = svadd_f64_x(
                predicate_tail_b64,
                svsub_f64_x(predicate_tail_b64, inner_sum_f64, svsub_f64_x(predicate_tail_b64, t_f64, z_f64)),
                svsub_f64_x(predicate_tail_b64, product_f64, z_f64));
            inner_sum_f64 = t_f64;
            inner_comp_f64 = svadd_f64_x(predicate_tail_b64, inner_comp_f64,
                                         svadd_f64_x(predicate_tail_b64, sum_error_f64, product_error_f64));
            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        nk_f64_t inner_result_f64 = svaddv_f64(predicate_body_b64, inner_sum_f64) +
                                    svaddv_f64(predicate_body_b64, inner_comp_f64);
        nk_f64_t outer_product_f64 = diff_row_f64 * inner_result_f64;
        nk_f64_t outer_product_error_f64 = diff_row_f64 * inner_result_f64 - outer_product_f64;
        nk_f64_t t_f64 = outer_sum_f64 + outer_product_f64;
        nk_f64_t z_f64 = t_f64 - outer_sum_f64;
        nk_f64_t sum_error_f64 = (outer_sum_f64 - (t_f64 - z_f64)) + (outer_product_f64 - z_f64);
        outer_sum_f64 = t_f64;
        outer_comp_f64 += sum_error_f64 + outer_product_error_f64;
    }

    return outer_sum_f64 + outer_comp_f64;
}

NK_PUBLIC void nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f64_smef64_kernel_(a, b, c, n);
    *result = nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

__arm_locally_streaming NK_PUBLIC void nk_bilinear_f32c_smef64(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                                               nk_f32c_t const *c_pairs, nk_size_t n,
                                                               nk_f32c_t *results) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_f64_t outer_sum_real_f64 = 0.0, outer_sum_imag_f64 = 0.0;
    nk_size_t const n2 = n + n; // number of f32 elements (2 per complex pair)

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t a_real_f64 = (nk_f64_t)a_pairs[row].real;
        nk_f64_t a_imag_f64 = (nk_f64_t)a_pairs[row].imag;

        svfloat64_t inner_sum_real_f64 = svdup_f64(0), inner_sum_imag_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            // Load interleaved complex f32 pairs and deinterleave
            svbool_t predicate_tail_b32 = svwhilelt_b32(j + j, n2);
            svfloat32_t b_f32x2 = svld1_f32(predicate_tail_b32, (nk_f32_t const *)b_pairs + j + j);
            svfloat64_t b_real_f64 = svcvt_f64_f32_x(predicate_tail_b64, svtrn1_f32(b_f32x2, b_f32x2));
            svfloat64_t b_imag_f64 = svcvt_f64_f32_x(predicate_tail_b64, svtrn2_f32(b_f32x2, b_f32x2));

            svfloat32_t c_f32x2 = svld1_f32(predicate_tail_b32, (nk_f32_t const *)c_pairs + (row * n + j) * 2);
            svfloat64_t c_real_f64 = svcvt_f64_f32_x(predicate_tail_b64, svtrn1_f32(c_f32x2, c_f32x2));
            svfloat64_t c_imag_f64 = svcvt_f64_f32_x(predicate_tail_b64, svtrn2_f32(c_f32x2, c_f32x2));

            // Complex multiply-accumulate: sum += c * b
            inner_sum_real_f64 = svmla_f64_x(predicate_tail_b64, inner_sum_real_f64, c_real_f64,
                                             b_real_f64); // += c_re * b_re
            inner_sum_real_f64 = svmls_f64_x(predicate_tail_b64, inner_sum_real_f64, c_imag_f64,
                                             b_imag_f64); // -= c_im * b_im
            inner_sum_imag_f64 = svmla_f64_x(predicate_tail_b64, inner_sum_imag_f64, c_real_f64,
                                             b_imag_f64); // += c_re * b_im
            inner_sum_imag_f64 = svmla_f64_x(predicate_tail_b64, inner_sum_imag_f64, c_imag_f64,
                                             b_real_f64); // += c_im * b_re

            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        // Reduce and complex outer multiply with a[row]
        nk_f64_t inner_real_f64 = svaddv_f64(predicate_body_b64, inner_sum_real_f64);
        nk_f64_t inner_imag_f64 = svaddv_f64(predicate_body_b64, inner_sum_imag_f64);
        outer_sum_real_f64 += a_real_f64 * inner_real_f64 - a_imag_f64 * inner_imag_f64;
        outer_sum_imag_f64 += a_real_f64 * inner_imag_f64 + a_imag_f64 * inner_real_f64;
    }

    results->real = (nk_f32_t)outer_sum_real_f64;
    results->imag = (nk_f32_t)outer_sum_imag_f64;
}

__arm_locally_streaming NK_PUBLIC void nk_bilinear_f64c_smef64(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs,
                                                               nk_f64c_t const *c_pairs, nk_size_t n,
                                                               nk_f64c_t *results) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_f64_t outer_sum_real_f64 = 0.0, outer_sum_imag_f64 = 0.0;
    nk_f64_t outer_comp_real_f64 = 0.0, outer_comp_imag_f64 = 0.0;
    nk_size_t const n2 = n + n; // number of f64 elements (2 per complex pair)

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t a_real_f64 = a_pairs[row].real;
        nk_f64_t a_imag_f64 = a_pairs[row].imag;

        // Inner loop: SVE with Kahan compensation for complex c*b accumulation
        svfloat64_t inner_sum_real_f64 = svdup_f64(0), inner_sum_imag_f64 = svdup_f64(0);
        svfloat64_t inner_comp_real_f64 = svdup_f64(0), inner_comp_imag_f64 = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_tail_b64 = svwhilelt_b64(j, n);

        while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
            // Load 2 vectors of f64, deinterleave to get real/imag
            nk_size_t j2 = j + j;
            svbool_t predicate_low_b64 = svwhilelt_b64(j2, n2);
            svbool_t predicate_high_b64 = svwhilelt_b64(j2 + svcntd(), n2);
            svfloat64_t b_low_f64 = svld1_f64(predicate_low_b64, (nk_f64_t const *)b_pairs + j2);
            svfloat64_t b_high_f64 = svld1_f64(predicate_high_b64, (nk_f64_t const *)b_pairs + j2 + svcntd());
            svfloat64_t b_real_f64 = svuzp1_f64(b_low_f64, b_high_f64);
            svfloat64_t b_imag_f64 = svuzp2_f64(b_low_f64, b_high_f64);

            svfloat64_t c_low_f64 = svld1_f64(predicate_low_b64, (nk_f64_t const *)c_pairs + (row * n + j) * 2);
            svfloat64_t c_high_f64 = svld1_f64(predicate_high_b64,
                                               (nk_f64_t const *)c_pairs + (row * n + j) * 2 + svcntd());
            svfloat64_t c_real_f64 = svuzp1_f64(c_low_f64, c_high_f64);
            svfloat64_t c_imag_f64 = svuzp2_f64(c_low_f64, c_high_f64);

            // Complex multiply: prod = c * b
            // prod_real = c_re*b_re - c_im*b_im, prod_imag = c_re*b_im + c_im*b_re
            svfloat64_t prod_real_f64 = svmul_f64_x(predicate_tail_b64, c_real_f64, b_real_f64);
            prod_real_f64 = svmls_f64_x(predicate_tail_b64, prod_real_f64, c_imag_f64, b_imag_f64);
            svfloat64_t prod_imag_f64 = svmul_f64_x(predicate_tail_b64, c_real_f64, b_imag_f64);
            prod_imag_f64 = svmla_f64_x(predicate_tail_b64, prod_imag_f64, c_imag_f64, b_real_f64);

            // Kahan compensation for real part
            svfloat64_t y_real_f64 = svsub_f64_x(predicate_tail_b64, prod_real_f64, inner_comp_real_f64);
            svfloat64_t tentative_sum_real_f64 = svadd_f64_x(predicate_tail_b64, inner_sum_real_f64, y_real_f64);
            inner_comp_real_f64 = svsub_f64_x(
                predicate_tail_b64, svsub_f64_x(predicate_tail_b64, tentative_sum_real_f64, inner_sum_real_f64),
                y_real_f64);
            inner_sum_real_f64 = tentative_sum_real_f64;

            // Kahan compensation for imaginary part
            svfloat64_t y_imag_f64 = svsub_f64_x(predicate_tail_b64, prod_imag_f64, inner_comp_imag_f64);
            svfloat64_t tentative_sum_imag_f64 = svadd_f64_x(predicate_tail_b64, inner_sum_imag_f64, y_imag_f64);
            inner_comp_imag_f64 = svsub_f64_x(
                predicate_tail_b64, svsub_f64_x(predicate_tail_b64, tentative_sum_imag_f64, inner_sum_imag_f64),
                y_imag_f64);
            inner_sum_imag_f64 = tentative_sum_imag_f64;

            j += svcntd();
            predicate_tail_b64 = svwhilelt_b64(j, n);
        }

        // Reduce inner accumulators
        nk_f64_t inner_real_f64 = svaddv_f64(predicate_body_b64, inner_sum_real_f64) -
                                  svaddv_f64(predicate_body_b64, inner_comp_real_f64);
        nk_f64_t inner_imag_f64 = svaddv_f64(predicate_body_b64, inner_sum_imag_f64) -
                                  svaddv_f64(predicate_body_b64, inner_comp_imag_f64);

        // Complex multiply: a * inner_result
        nk_f64_t outer_prod_real_f64 = a_real_f64 * inner_real_f64 - a_imag_f64 * inner_imag_f64;
        nk_f64_t outer_prod_imag_f64 = a_real_f64 * inner_imag_f64 + a_imag_f64 * inner_real_f64;

        // Kahan compensation for outer loop
        nk_f64_t y_real_f64 = outer_prod_real_f64 - outer_comp_real_f64;
        nk_f64_t tentative_sum_real_f64 = outer_sum_real_f64 + y_real_f64;
        outer_comp_real_f64 = (tentative_sum_real_f64 - outer_sum_real_f64) - y_real_f64;
        outer_sum_real_f64 = tentative_sum_real_f64;

        nk_f64_t y_imag_f64 = outer_prod_imag_f64 - outer_comp_imag_f64;
        nk_f64_t tentative_sum_imag_f64 = outer_sum_imag_f64 + y_imag_f64;
        outer_comp_imag_f64 = (tentative_sum_imag_f64 - outer_sum_imag_f64) - y_imag_f64;
        outer_sum_imag_f64 = tentative_sum_imag_f64;
    }

    results->real = outer_sum_real_f64 - outer_comp_real_f64;
    results->imag = outer_sum_imag_f64 - outer_comp_imag_f64;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SMEF64
#endif // NK_TARGET_ARM_
#endif // NK_CURVED_SMEF64_H
