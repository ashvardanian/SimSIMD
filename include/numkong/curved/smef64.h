/**
 *  @brief SIMD-accelerated Curved Space Similarity for SME F64.
 *  @file include/numkong/curved/smef64.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance using ARM SVE:
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
 *  @section sve_notes SVE Implementation Notes
 *
 *  SVE is vector-length agnostic (VLA). Key patterns:
 *  - svcntd() returns number of f64 elements per vector (varies by hardware)
 *  - svwhilelt_b64(i, n) creates predicate for partial vectors at loop tail
 *  - All operations use predicates for clean handling of arbitrary lengths
 */
#ifndef NK_CURVED_SMEF64_H
#define NK_CURVED_SMEF64_H

#include "numkong/types.h"

#if NK_TARGET_ARM_
#if NK_TARGET_SMEF64

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("sve")
#endif

#include <arm_sve.h>

#include "numkong/spatial/neon.h" // nk_f64_sqrt_neon, nk_f32_sqrt_neon

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f32_t *result) {
    svbool_t pg_true = svptrue_b64();
    svfloat64_t outer_sum = svdup_f64(0.0);

    for (nk_size_t row = 0; row < n; ++row) {
        // Load a[row] and upcast to f64
        nk_f64_t a_row_f64 = (nk_f64_t)a[row];

        // Inner loop: accumulate c[row, j] * b[j] using SVE
        svfloat64_t inner_sum = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t pred = svwhilelt_b64(j, n);

        while (svptest_first(pg_true, pred)) {
            // Load b[j] as f32, upcast to f64
            svfloat32_t b_f32 = svld1_f32(svwhilelt_b32(j, n), b + j);
            svfloat64_t b_f64 = svcvt_f64_f32_x(pred, b_f32);

            // Load c[row*n + j] as f32, upcast to f64
            svfloat32_t c_f32 = svld1_f32(svwhilelt_b32(j, n), c + row * n + j);
            svfloat64_t c_f64 = svcvt_f64_f32_x(pred, c_f32);

            // Accumulate c * b into inner_sum
            inner_sum = svmla_f64_x(pred, inner_sum, c_f64, b_f64);

            j += svcntd();
            pred = svwhilelt_b64(j, n);
        }

        // Reduce inner_sum to scalar
        nk_f64_t inner_result = svaddv_f64(pg_true, inner_sum);

        // Outer accumulation: a[row] * inner_result
        svfloat64_t a_vec = svdup_f64(a_row_f64);
        svfloat64_t inner_vec = svdup_f64(inner_result);
        outer_sum = svmla_f64_x(pg_true, outer_sum, a_vec, inner_vec);
    }

    // Reduce outer_sum - it has replicated values, just take lane 0
    nk_f64_t outer_result = svaddv_f64(pg_true, outer_sum) / (nk_f64_t)svcntd();
    *result = (nk_f32_t)outer_result;
}

NK_PUBLIC void nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f32_t *result) {
    svbool_t pg_true = svptrue_b64();
    svfloat64_t outer_sum = svdup_f64(0.0);

    for (nk_size_t row = 0; row < n; ++row) {
        // Compute difference for row in f64
        nk_f64_t diff_row = (nk_f64_t)a[row] - (nk_f64_t)b[row];

        // Inner loop: accumulate c[row, j] * (a[j] - b[j])
        svfloat64_t inner_sum = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t pred = svwhilelt_b64(j, n);

        while (svptest_first(pg_true, pred)) {
            // Load a[j] and b[j], compute difference in f64
            svfloat32_t a_f32 = svld1_f32(svwhilelt_b32(j, n), a + j);
            svfloat32_t b_f32 = svld1_f32(svwhilelt_b32(j, n), b + j);
            svfloat64_t a_f64 = svcvt_f64_f32_x(pred, a_f32);
            svfloat64_t b_f64 = svcvt_f64_f32_x(pred, b_f32);
            svfloat64_t diff_col = svsub_f64_x(pred, a_f64, b_f64);

            // Load c[row*n + j], upcast to f64
            svfloat32_t c_f32 = svld1_f32(svwhilelt_b32(j, n), c + row * n + j);
            svfloat64_t c_f64 = svcvt_f64_f32_x(pred, c_f32);

            // Accumulate c * diff_col
            inner_sum = svmla_f64_x(pred, inner_sum, c_f64, diff_col);

            j += svcntd();
            pred = svwhilelt_b64(j, n);
        }

        // Reduce inner_sum to scalar
        nk_f64_t inner_result = svaddv_f64(pg_true, inner_sum);

        // Outer accumulation: diff_row * inner_result
        svfloat64_t diff_vec = svdup_f64(diff_row);
        svfloat64_t inner_vec = svdup_f64(inner_result);
        outer_sum = svmla_f64_x(pg_true, outer_sum, diff_vec, inner_vec);
    }

    // Reduce and take sqrt
    nk_f64_t outer_result = svaddv_f64(pg_true, outer_sum) / (nk_f64_t)svcntd();
    *result = nk_f32_sqrt_neon((nk_f32_t)(outer_result > 0 ? outer_result : 0));
}

NK_PUBLIC void nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result) {
    svbool_t pg_true = svptrue_b64();

    // Outer loop accumulators with Dot2 compensation
    nk_f64_t outer_sum = 0.0;
    nk_f64_t outer_compensation = 0.0;

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t a_row = a[row];

        // Inner loop with Dot2 compensation
        svfloat64_t inner_sum = svdup_f64(0.0);
        svfloat64_t inner_compensation = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t pred = svwhilelt_b64(j, n);

        while (svptest_first(pg_true, pred)) {
            svfloat64_t b_vec = svld1_f64(pred, b + j);
            svfloat64_t c_vec = svld1_f64(pred, c + row * n + j);

            // TwoProd: product = c * b, product_error = fma(c, b, -product)
            svfloat64_t product = svmul_f64_x(pred, c_vec, b_vec);
            // svnmls computes: result = acc - (op1 * op2), so -svnmls gives us fma(c, b, -product)
            svfloat64_t product_error = svneg_f64_x(pred, svnmls_f64_x(pred, product, c_vec, b_vec));

            // TwoSum: (t, sum_error) = TwoSum(inner_sum, product)
            svfloat64_t t = svadd_f64_x(pred, inner_sum, product);
            svfloat64_t z = svsub_f64_x(pred, t, inner_sum);
            svfloat64_t sum_error = svadd_f64_x(pred, svsub_f64_x(pred, inner_sum, svsub_f64_x(pred, t, z)),
                                                svsub_f64_x(pred, product, z));
            inner_sum = t;
            inner_compensation = svadd_f64_x(pred, inner_compensation, svadd_f64_x(pred, sum_error, product_error));

            j += svcntd();
            pred = svwhilelt_b64(j, n);
        }

        // Reduce inner loop results
        nk_f64_t inner_result = svaddv_f64(pg_true, inner_sum) + svaddv_f64(pg_true, inner_compensation);

        // TwoProd for outer: a_row * inner_result
        nk_f64_t outer_product = a_row * inner_result;
        nk_f64_t outer_product_error = a_row * inner_result - outer_product; // Simplified FMA substitute

        // TwoSum for outer accumulation
        nk_f64_t t = outer_sum + outer_product;
        nk_f64_t z = t - outer_sum;
        nk_f64_t sum_error = (outer_sum - (t - z)) + (outer_product - z);
        outer_sum = t;
        outer_compensation += sum_error + outer_product_error;
    }

    *result = outer_sum + outer_compensation;
}

NK_PUBLIC void nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result) {
    svbool_t pg_true = svptrue_b64();

    // Outer loop accumulators with Dot2 compensation
    nk_f64_t outer_sum = 0.0;
    nk_f64_t outer_compensation = 0.0;

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t diff_row = a[row] - b[row];

        // Inner loop with Dot2 compensation
        svfloat64_t inner_sum = svdup_f64(0.0);
        svfloat64_t inner_compensation = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t pred = svwhilelt_b64(j, n);

        while (svptest_first(pg_true, pred)) {
            svfloat64_t a_vec = svld1_f64(pred, a + j);
            svfloat64_t b_vec = svld1_f64(pred, b + j);
            svfloat64_t diff_col = svsub_f64_x(pred, a_vec, b_vec);
            svfloat64_t c_vec = svld1_f64(pred, c + row * n + j);

            // TwoProd: product = c * diff_col
            svfloat64_t product = svmul_f64_x(pred, c_vec, diff_col);
            svfloat64_t product_error = svneg_f64_x(pred, svnmls_f64_x(pred, product, c_vec, diff_col));

            // TwoSum: (t, sum_error) = TwoSum(inner_sum, product)
            svfloat64_t t = svadd_f64_x(pred, inner_sum, product);
            svfloat64_t z = svsub_f64_x(pred, t, inner_sum);
            svfloat64_t sum_error = svadd_f64_x(pred, svsub_f64_x(pred, inner_sum, svsub_f64_x(pred, t, z)),
                                                svsub_f64_x(pred, product, z));
            inner_sum = t;
            inner_compensation = svadd_f64_x(pred, inner_compensation, svadd_f64_x(pred, sum_error, product_error));

            j += svcntd();
            pred = svwhilelt_b64(j, n);
        }

        // Reduce inner loop results
        nk_f64_t inner_result = svaddv_f64(pg_true, inner_sum) + svaddv_f64(pg_true, inner_compensation);

        // TwoProd for outer: diff_row * inner_result
        nk_f64_t outer_product = diff_row * inner_result;
        nk_f64_t outer_product_error = diff_row * inner_result - outer_product;

        // TwoSum for outer accumulation
        nk_f64_t t = outer_sum + outer_product;
        nk_f64_t z = t - outer_sum;
        nk_f64_t sum_error = (outer_sum - (t - z)) + (outer_product - z);
        outer_sum = t;
        outer_compensation += sum_error + outer_product_error;
    }

    nk_f64_t quadratic = outer_sum + outer_compensation;
    *result = nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_f32c_smef64(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_f32c_t const *c_pairs,
                                       nk_size_t n, nk_f32c_t *results) {
    nk_f64_t outer_sum_real = 0.0, outer_sum_imag = 0.0;

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t a_real = (nk_f64_t)a_pairs[row].real;
        nk_f64_t a_imag = (nk_f64_t)a_pairs[row].imag;

        nk_f64_t inner_sum_real = 0.0, inner_sum_imag = 0.0;

        // Inner loop: compute c[row, j] * b[j] for complex numbers
        for (nk_size_t j = 0; j < n; ++j) {
            nk_f64_t b_real = (nk_f64_t)b_pairs[j].real;
            nk_f64_t b_imag = (nk_f64_t)b_pairs[j].imag;
            nk_f64_t c_real = (nk_f64_t)c_pairs[row * n + j].real;
            nk_f64_t c_imag = (nk_f64_t)c_pairs[row * n + j].imag;

            // Complex multiply: c * b = (c_real*b_real - c_imag*b_imag) + (c_real*b_imag + c_imag*b_real)i
            nk_f64_t prod_real = c_real * b_real - c_imag * b_imag;
            nk_f64_t prod_imag = c_real * b_imag + c_imag * b_real;

            inner_sum_real += prod_real;
            inner_sum_imag += prod_imag;
        }

        // Complex multiply: a * inner_result
        nk_f64_t outer_prod_real = a_real * inner_sum_real - a_imag * inner_sum_imag;
        nk_f64_t outer_prod_imag = a_real * inner_sum_imag + a_imag * inner_sum_real;

        outer_sum_real += outer_prod_real;
        outer_sum_imag += outer_prod_imag;
    }

    results->real = (nk_f32_t)outer_sum_real;
    results->imag = (nk_f32_t)outer_sum_imag;
}

NK_PUBLIC void nk_bilinear_f64c_smef64(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_f64c_t const *c_pairs,
                                       nk_size_t n, nk_f64c_t *results) {
    nk_f64_t outer_sum_real = 0.0, outer_sum_imag = 0.0;
    nk_f64_t outer_comp_real = 0.0, outer_comp_imag = 0.0;

    for (nk_size_t row = 0; row < n; ++row) {
        nk_f64_t a_real = a_pairs[row].real;
        nk_f64_t a_imag = a_pairs[row].imag;

        nk_f64_t inner_sum_real = 0.0, inner_sum_imag = 0.0;
        nk_f64_t inner_comp_real = 0.0, inner_comp_imag = 0.0;

        // Inner loop with Neumaier compensation
        for (nk_size_t j = 0; j < n; ++j) {
            nk_f64_t b_real = b_pairs[j].real;
            nk_f64_t b_imag = b_pairs[j].imag;
            nk_f64_t c_real = c_pairs[row * n + j].real;
            nk_f64_t c_imag = c_pairs[row * n + j].imag;

            // Complex multiply: c * b
            nk_f64_t prod_real = c_real * b_real - c_imag * b_imag;
            nk_f64_t prod_imag = c_real * b_imag + c_imag * b_real;

            // Neumaier summation for real part
            nk_f64_t t_real = inner_sum_real + prod_real;
            if (nk_f64_abs_(inner_sum_real) >= nk_f64_abs_(prod_real))
                inner_comp_real += (inner_sum_real - t_real) + prod_real;
            else inner_comp_real += (prod_real - t_real) + inner_sum_real;
            inner_sum_real = t_real;

            // Neumaier summation for imaginary part
            nk_f64_t t_imag = inner_sum_imag + prod_imag;
            if (nk_f64_abs_(inner_sum_imag) >= nk_f64_abs_(prod_imag))
                inner_comp_imag += (inner_sum_imag - t_imag) + prod_imag;
            else inner_comp_imag += (prod_imag - t_imag) + inner_sum_imag;
            inner_sum_imag = t_imag;
        }

        inner_sum_real += inner_comp_real;
        inner_sum_imag += inner_comp_imag;

        // Complex multiply: a * inner_result
        nk_f64_t outer_prod_real = a_real * inner_sum_real - a_imag * inner_sum_imag;
        nk_f64_t outer_prod_imag = a_real * inner_sum_imag + a_imag * inner_sum_real;

        // Neumaier for outer loop
        nk_f64_t t_real = outer_sum_real + outer_prod_real;
        if (nk_f64_abs_(outer_sum_real) >= nk_f64_abs_(outer_prod_real))
            outer_comp_real += (outer_sum_real - t_real) + outer_prod_real;
        else outer_comp_real += (outer_prod_real - t_real) + outer_sum_real;
        outer_sum_real = t_real;

        nk_f64_t t_imag = outer_sum_imag + outer_prod_imag;
        if (nk_f64_abs_(outer_sum_imag) >= nk_f64_abs_(outer_prod_imag))
            outer_comp_imag += (outer_sum_imag - t_imag) + outer_prod_imag;
        else outer_comp_imag += (outer_prod_imag - t_imag) + outer_sum_imag;
        outer_sum_imag = t_imag;
    }

    results->real = outer_sum_real + outer_comp_real;
    results->imag = outer_sum_imag + outer_comp_imag;
}

#if defined(__cplusplus)
}
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_SMEF64
#endif // NK_TARGET_ARM_
#endif // NK_CURVED_SMEF64_H
