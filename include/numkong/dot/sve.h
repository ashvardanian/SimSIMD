/**
 *  @brief SIMD-accelerated Dot Products for SVE.
 *  @file include/numkong/dot/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f32                   LD1W (Z.S, P/Z, [Xn])           4-6cy       2/cy
 *      svld2_f32                   LD2W (Z.S, P/Z, [Xn])           6-8cy       1/cy
 *      svmla_f32_x                 FMLA (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svmls_f32_x                 FMLS (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svaddv_f32                  FADDV (S, P, Z.S)               6cy         1/cy
 *      svdup_f32                   DUP (Z.S, #imm)                 1cy         2/cy
 *      svwhilelt_b32               WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svptrue_b32                 PTRUE (P.S, pattern)            1cy         2/cy
 *      svcntw                      CNTW (Xd)                       1cy         2/cy
 *      svcntd                      CNTD (Xd)                       1cy         2/cy
 *      svld1_f64                   LD1D (Z.D, P/Z, [Xn])           4-6cy       2/cy
 *      svld2_f64                   LD2D (Z.D, P/Z, [Xn])           6-8cy       1/cy
 *      svmla_f64_x                 FMLA (Z.D, P/M, Z.D, Z.D)       4cy         2/cy
 *      svmls_f64_x                 FMLS (Z.D, P/M, Z.D, Z.D)       4cy         2/cy
 *      svaddv_f64                  FADDV (D, P, Z.D)               6cy         1/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The FADDV horizontal reduction has higher latency (6cy) compared to vertical operations,
 *  making it beneficial to accumulate in vector registers and reduce only at the end.
 */
#ifndef NK_DOT_SVE_H
#define NK_DOT_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE

#include "numkong/types.h"
#include "numkong/dot/serial.h"  // `nk_u1x8_popcount_`
#include "numkong/reduce/neon.h" // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** @brief Compensated horizontal sum of SVE f64 lanes via TwoSum tree reduction.
 *
 *  Uses svtbl to extract the upper half at each tree level. Out-of-range indices
 *  return 0 (SVE spec), which is harmless since only the lower half is meaningful
 *  after each halving stage.
 */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64_sve_(svbool_t predicate, svfloat64_t sum, svfloat64_t compensation) {
    // Stage 0: TwoSum merge of sum + compensation (parallel across all active lanes)
    svfloat64_t tentative_sum = svadd_f64_x(predicate, sum, compensation);
    svfloat64_t virtual_addend = svsub_f64_x(predicate, tentative_sum, sum);
    svfloat64_t accumulated_error = svadd_f64_x(
        predicate, svsub_f64_x(predicate, sum, svsub_f64_x(predicate, tentative_sum, virtual_addend)),
        svsub_f64_x(predicate, compensation, virtual_addend));

    // Tree reduction: TwoSum halving at each level, log2(VL) iterations
    for (unsigned int half = (unsigned int)svcntd() / 2; half > 0; half >>= 1) {
        svuint64_t upper_indices = svadd_n_u64_x(predicate, svindex_u64(0, 1), half);
        svfloat64_t upper_sum = svtbl_f64(tentative_sum, upper_indices);
        svfloat64_t upper_error = svtbl_f64(accumulated_error, upper_indices);
        // TwoSum: lower_half + upper_half
        svfloat64_t halved_tentative_sum = svadd_f64_x(predicate, tentative_sum, upper_sum);
        svfloat64_t halved_virtual_addend = svsub_f64_x(predicate, halved_tentative_sum, tentative_sum);
        svfloat64_t rounding_error = svadd_f64_x(
            predicate,
            svsub_f64_x(predicate, tentative_sum, svsub_f64_x(predicate, halved_tentative_sum, halved_virtual_addend)),
            svsub_f64_x(predicate, upper_sum, halved_virtual_addend));
        tentative_sum = halved_tentative_sum;
        accumulated_error = svadd_f64_x(predicate, svadd_f64_x(predicate, accumulated_error, upper_error),
                                        rounding_error);
    }
    // Result is in lane 0
    svbool_t pg_first = svwhilelt_b64((uint32_t)0, (uint32_t)1);
    return svlastb_f64(pg_first, tentative_sum) + svlastb_f64(pg_first, accumulated_error);
}

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
    // Dot2 (Ogita-Rump-Oishi) compensated accumulation via TwoProd + TwoSum
    nk_size_t idx_scalars = 0;
    svfloat64_t sum_vec = svdup_f64(0.);
    svfloat64_t compensation_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_scalars, (unsigned int)count_scalars);
        svfloat64_t a_vec = svld1_f64(pg_vec, a_scalars + idx_scalars);
        svfloat64_t b_vec = svld1_f64(pg_vec, b_scalars + idx_scalars);
        // TwoProd: product = a*b, error = -(product - a*b) negated
        svfloat64_t product_vec = svmul_f64_x(pg_vec, a_vec, b_vec);
        svfloat64_t product_error_vec = svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_vec, b_vec));
        // TwoSum: tentative_sum = sum + product
        svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_vec, product_vec);
        svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_vec);
        svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
            svsub_f64_x(pg_vec, sum_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
            svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
        sum_vec = tentative_sum_vec;
        compensation_vec = svadd_f64_x(pg_vec, compensation_vec,
                                       svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        idx_scalars += svcntd();
    } while (idx_scalars < count_scalars);
    *result = nk_dot_stable_sum_f64_sve_(svptrue_b64(), sum_vec, compensation_vec);
}

NK_PUBLIC void nk_dot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    // Dot2 compensated accumulation for complex dot product: (a_re + i*a_im)(b_re + i*b_im)
    // real = a_re*b_re - a_im*b_im, imag = a_re*b_im + a_im*b_re
    nk_size_t idx_pairs = 0;
    svfloat64_t sum_real_vec = svdup_f64(0.);
    svfloat64_t comp_real_vec = svdup_f64(0.);
    svfloat64_t sum_imag_vec = svdup_f64(0.);
    svfloat64_t comp_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (nk_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_vec = svget2_f64(a_vec, 0);
        svfloat64_t a_imag_vec = svget2_f64(a_vec, 1);
        svfloat64_t b_real_vec = svget2_f64(b_vec, 0);
        svfloat64_t b_imag_vec = svget2_f64(b_vec, 1);

        // TwoProd + TwoSum for real part: sum_real += a_re*b_re
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_real_vec, b_real_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_real_vec, b_real_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_real_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_real_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_real_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_real_vec = tentative_sum_vec;
            comp_real_vec = svadd_f64_x(pg_vec, comp_real_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        // TwoProd + TwoSum for real part: sum_real -= a_im*b_im
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_imag_vec, b_imag_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_imag_vec, b_imag_vec));
            svfloat64_t neg_product_vec = svneg_f64_x(pg_vec, product_vec);
            svfloat64_t neg_product_error_vec = svneg_f64_x(pg_vec, product_error_vec);
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_real_vec, neg_product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_real_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_real_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, neg_product_vec, virtual_addend_vec));
            sum_real_vec = tentative_sum_vec;
            comp_real_vec = svadd_f64_x(pg_vec, comp_real_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, neg_product_error_vec));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_re*b_im
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_real_vec, b_imag_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_real_vec, b_imag_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_imag_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_imag_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_imag_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_imag_vec = tentative_sum_vec;
            comp_imag_vec = svadd_f64_x(pg_vec, comp_imag_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_im*b_re
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_imag_vec, b_real_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_imag_vec, b_real_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_imag_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_imag_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_imag_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_imag_vec = tentative_sum_vec;
            comp_imag_vec = svadd_f64_x(pg_vec, comp_imag_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    svbool_t pg_true = svptrue_b64();
    results->real = nk_dot_stable_sum_f64_sve_(pg_true, sum_real_vec, comp_real_vec);
    results->imag = nk_dot_stable_sum_f64_sve_(pg_true, sum_imag_vec, comp_imag_vec);
}

NK_PUBLIC void nk_vdot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    // Dot2 compensated conjugate dot product: conj(a) · b = (a_re - i*a_im)(b_re + i*b_im)
    // real = a_re*b_re + a_im*b_im, imag = a_re*b_im - a_im*b_re
    nk_size_t idx_pairs = 0;
    svfloat64_t sum_real_vec = svdup_f64(0.);
    svfloat64_t comp_real_vec = svdup_f64(0.);
    svfloat64_t sum_imag_vec = svdup_f64(0.);
    svfloat64_t comp_imag_vec = svdup_f64(0.);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)idx_pairs, (unsigned int)count_pairs);
        svfloat64x2_t a_vec = svld2_f64(pg_vec, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_vec = svld2_f64(pg_vec, (nk_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_vec = svget2_f64(a_vec, 0);
        svfloat64_t a_imag_vec = svget2_f64(a_vec, 1);
        svfloat64_t b_real_vec = svget2_f64(b_vec, 0);
        svfloat64_t b_imag_vec = svget2_f64(b_vec, 1);

        // TwoProd + TwoSum for real part: sum_real += a_re*b_re
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_real_vec, b_real_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_real_vec, b_real_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_real_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_real_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_real_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_real_vec = tentative_sum_vec;
            comp_real_vec = svadd_f64_x(pg_vec, comp_real_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        // TwoProd + TwoSum for real part: sum_real += a_im*b_im (conjugate: + not -)
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_imag_vec, b_imag_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_imag_vec, b_imag_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_real_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_real_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_real_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_real_vec = tentative_sum_vec;
            comp_real_vec = svadd_f64_x(pg_vec, comp_real_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_re*b_im
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_real_vec, b_imag_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_real_vec, b_imag_vec));
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_imag_vec, product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_imag_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_imag_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, product_vec, virtual_addend_vec));
            sum_imag_vec = tentative_sum_vec;
            comp_imag_vec = svadd_f64_x(pg_vec, comp_imag_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, product_error_vec));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag -= a_im*b_re (conjugate: - not +)
        {
            svfloat64_t product_vec = svmul_f64_x(pg_vec, a_imag_vec, b_real_vec);
            svfloat64_t product_error_vec =
                svneg_f64_x(pg_vec, svnmls_f64_x(pg_vec, product_vec, a_imag_vec, b_real_vec));
            svfloat64_t neg_product_vec = svneg_f64_x(pg_vec, product_vec);
            svfloat64_t neg_product_error_vec = svneg_f64_x(pg_vec, product_error_vec);
            svfloat64_t tentative_sum_vec = svadd_f64_x(pg_vec, sum_imag_vec, neg_product_vec);
            svfloat64_t virtual_addend_vec = svsub_f64_x(pg_vec, tentative_sum_vec, sum_imag_vec);
            svfloat64_t sum_error_vec = svadd_f64_x(pg_vec,
                svsub_f64_x(pg_vec, sum_imag_vec, svsub_f64_x(pg_vec, tentative_sum_vec, virtual_addend_vec)),
                svsub_f64_x(pg_vec, neg_product_vec, virtual_addend_vec));
            sum_imag_vec = tentative_sum_vec;
            comp_imag_vec = svadd_f64_x(pg_vec, comp_imag_vec,
                                        svadd_f64_x(pg_vec, sum_error_vec, neg_product_error_vec));
        }
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    svbool_t pg_true = svptrue_b64();
    results->real = nk_dot_stable_sum_f64_sve_(pg_true, sum_real_vec, comp_real_vec);
    results->imag = nk_dot_stable_sum_f64_sve_(pg_true, sum_imag_vec, comp_imag_vec);
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
#endif // NK_DOT_SVE_H
