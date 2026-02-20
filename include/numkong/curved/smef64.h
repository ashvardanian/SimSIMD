/**
 *  @brief SIMD-accelerated Curved Space Similarity for SME F64.
 *  @file include/numkong/curved/smef64.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance using ARM SME FMOPA:
 *  - f32 inputs: GEMV via f64 FMOPA (widening load f32→f64, no Ozaki needed)
 *  - f64 inputs: GEMV via f64 FMOPA with Ozaki 3-way splitting for near-2x precision
 *  - f32c/f64c complex: 4-FMOPA complex GEMV with FMOPS for the c_im*b_im subtraction
 *  - f64c uses Ozaki splitting (24 FMOPA/FMOPS per step, ZA0-6)
 *  - Mahalanobis: d=a-b computed first, then GEMV v=C*d via FMOPA, then d^T*v
 *
 *  @section architecture GEMV-via-FMOPA Architecture
 *
 *  For a^T C b, decompose as v = C * b (GEMV via FMOPA), then result = a^T * v (dot).
 *  Stage C rows into ZA0 horizontally, read columns vertically (hardware transpose),
 *  broadcast b[k] and accumulate via FMOPA into ZA1+. Every column of the accumulator
 *  is identical (broadcast Zm), so read any single column for v.
 *
 *  For Mahalanobis d^T C d where d=a-b, compute d[k]=a[k]-b[k] first (exact for
 *  f32→f64 widening), then GEMV v=C*d via the same FMOPA pipeline. This avoids
 *  catastrophic cancellation from separate FMOPA(a)+FMOPS(b) when a≈b.
 *
 *  @section precision Precision Strategy
 *
 *  For f32 inputs: Widening to f64 via FMOPA provides exact f64 accumulation.
 *  For f64 inputs: Ozaki 3-way splitting (19+17+17 mantissa bits) into ZA1-3
 *  accumulators, matching the GEMM kernel in dots/smef64.h.
 *  For f64c inputs: Ozaki splitting with 7 ZA tiles (ZA1-3 v_re, ZA4-6 v_im).
 */
#ifndef NK_CURVED_SMEF64_H
#define NK_CURVED_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEF64

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f64_sqrt_neon`
#include "numkong/dots/sme.h"     // nk_sme_zero_za64_tile_0_, etc.
#include "numkong/dots/smef64.h"  // nk_f64_smef64_ozaki_{mask_19_bits_,split_f64_}

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat64_t c_row_f64 = svcvt_f64_f32_x(
                    batch_predicate_b64,
                    svreinterpret_f32_u64(svld1uw_u64(batch_predicate_b64, (nk_u32_t const *)(c + (row + r) * n + j))));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, c_row_f64);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_col_f64, svdup_f64((nk_f64_t)b[j + k]));
            }
        }

        svfloat64_t v_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0);
        svfloat64_t a_f64 = svcvt_f64_f32_x(
            row_predicate_b64, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64, (nk_u32_t const *)(a + row))));
        outer_sum_f64 += svaddv_f64(predicate_body_b64, svmul_f64_x(row_predicate_b64, a_f64, v_f64));
    }

    *result = (nk_f32_t)outer_sum_f64;
}

// ─────────────────────────────────────────────────────────────────────────────
// f32 Mahalanobis: GEMV v = C*d via FMOPA, where d = a - b (exact in f64)
// ZA0.D = C staging, ZA1.D = GEMV accumulator
// ─────────────────────────────────────────────────────────────────────────────

__arm_locally_streaming __arm_new("za") static inline nk_f64_t
    nk_mahalanobis_f32_smef64_kernel_(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n) {

    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat64_t c_row_f64 = svcvt_f64_f32_x(
                    batch_predicate_b64,
                    svreinterpret_f32_u64(svld1uw_u64(batch_predicate_b64, (nk_u32_t const *)(c + (row + r) * n + j))));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, c_row_f64);
            }

            // GEMV: v += C_col * d[j+k], where d[k] = a[k] - b[k] (exact in f64 for f32 inputs)
            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                nk_f64_t d_k = (nk_f64_t)a[j + k] - (nk_f64_t)b[j + k];
                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_col_f64, svdup_f64(d_k));
            }
        }

        // v = C*d, dot with d[row:row+tile]
        svfloat64_t v_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0);
        svfloat64_t a_f64 = svcvt_f64_f32_x(
            row_predicate_b64, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64, (nk_u32_t const *)(a + row))));
        svfloat64_t b_f64 = svcvt_f64_f32_x(
            row_predicate_b64, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64, (nk_u32_t const *)(b + row))));
        svfloat64_t d_f64 = svsub_f64_x(row_predicate_b64, a_f64, b_f64);
        outer_sum_f64 += svaddv_f64(predicate_body_b64, svmul_f64_x(row_predicate_b64, d_f64, v_f64));
    }

    return outer_sum_f64;
}

NK_PUBLIC void nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f32_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f32_smef64_kernel_(a, b, c, n);
    *result = (nk_f32_t)nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                          nk_f64_t *result) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    svuint64_t ozaki_mask_19 = svdup_u64(nk_f64_smef64_ozaki_mask_19_bits_());
    svuint64_t ozaki_mask_17 = svdup_u64(nk_f64_smef64_ozaki_mask_17_bits_());

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tiles_1_3_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++)
                svld1_hor_za64(0, r, batch_predicate_b64, c + (row + r) * n + j);

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svuint64_t c_bits_u64 = svreinterpret_u64_f64(c_col_f64);
                svfloat64_t c_s0_f64 = svreinterpret_f64_u64(
                    svand_u64_x(predicate_body_b64, c_bits_u64, ozaki_mask_19));
                svfloat64_t c_res_f64 = svsub_f64_x(predicate_body_b64, c_col_f64, c_s0_f64);
                svuint64_t c_res_bits_u64 = svreinterpret_u64_f64(c_res_f64);
                svfloat64_t c_s1_f64 = svreinterpret_f64_u64(
                    svand_u64_x(predicate_body_b64, c_res_bits_u64, ozaki_mask_17));
                svfloat64_t c_s2_f64 = svsub_f64_x(predicate_body_b64, c_res_f64, c_s1_f64);

                nk_f64_t b_s0, b_s1, b_s2;
                nk_f64_smef64_ozaki_split_f64_(b[j + k], &b_s0, &b_s1, &b_s2);

                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(b_s0)); // i+j=0
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(b_s1)); // i+j=1
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s1_f64, svdup_f64(b_s0)); // i+j=1
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(b_s2)); // i+j=2
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s1_f64, svdup_f64(b_s1)); // i+j=2
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s2_f64, svdup_f64(b_s0)); // i+j=2
            }
        }

        // Sum ZA3 + ZA2 + ZA1 (smallest magnitude first)
        svfloat64_t v_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 3, 0);
        v_f64 = svadd_f64_x(predicate_body_b64, v_f64, svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 2, 0));
        v_f64 = svadd_f64_x(predicate_body_b64, v_f64, svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0));

        svfloat64_t a_f64 = svld1_f64(row_predicate_b64, a + row);
        outer_sum_f64 += svaddv_f64(predicate_body_b64, svmul_f64_x(row_predicate_b64, a_f64, v_f64));
    }

    *result = outer_sum_f64;
}

__arm_locally_streaming __arm_new("za") static inline nk_f64_t
    nk_mahalanobis_f64_smef64_kernel_(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n) {

    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    svuint64_t ozaki_mask_19 = svdup_u64(nk_f64_smef64_ozaki_mask_19_bits_());
    svuint64_t ozaki_mask_17 = svdup_u64(nk_f64_smef64_ozaki_mask_17_bits_());

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tiles_1_3_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++)
                svld1_hor_za64(0, r, batch_predicate_b64, c + (row + r) * n + j);

            // Ozaki FMOPA with d[j+k] = a[j+k] - b[j+k]
            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svuint64_t c_bits_u64 = svreinterpret_u64_f64(c_col_f64);
                svfloat64_t c_s0_f64 = svreinterpret_f64_u64(
                    svand_u64_x(predicate_body_b64, c_bits_u64, ozaki_mask_19));
                svfloat64_t c_res_f64 = svsub_f64_x(predicate_body_b64, c_col_f64, c_s0_f64);
                svuint64_t c_res_bits_u64 = svreinterpret_u64_f64(c_res_f64);
                svfloat64_t c_s1_f64 = svreinterpret_f64_u64(
                    svand_u64_x(predicate_body_b64, c_res_bits_u64, ozaki_mask_17));
                svfloat64_t c_s2_f64 = svsub_f64_x(predicate_body_b64, c_res_f64, c_s1_f64);

                nk_f64_t d_k = a[j + k] - b[j + k];
                nk_f64_t d_s0, d_s1, d_s2;
                nk_f64_smef64_ozaki_split_f64_(d_k, &d_s0, &d_s1, &d_s2);

                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(d_s0)); // i+j=0
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(d_s1)); // i+j=1
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s1_f64, svdup_f64(d_s0)); // i+j=1
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s0_f64, svdup_f64(d_s2)); // i+j=2
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s1_f64, svdup_f64(d_s1)); // i+j=2
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s2_f64, svdup_f64(d_s0)); // i+j=2
            }
        }

        // Sum ZA3 + ZA2 + ZA1 (smallest magnitude first)
        svfloat64_t v_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 3, 0);
        v_f64 = svadd_f64_x(predicate_body_b64, v_f64, svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 2, 0));
        v_f64 = svadd_f64_x(predicate_body_b64, v_f64, svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0));

        // Dot v with d[row:row+tile]
        svfloat64_t a_f64 = svld1_f64(row_predicate_b64, a + row);
        svfloat64_t b_f64 = svld1_f64(row_predicate_b64, b + row);
        svfloat64_t d_f64 = svsub_f64_x(row_predicate_b64, a_f64, b_f64);
        outer_sum_f64 += svaddv_f64(predicate_body_b64, svmul_f64_x(row_predicate_b64, d_f64, v_f64));
    }

    return outer_sum_f64;
}

NK_PUBLIC void nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f64_smef64_kernel_(a, b, c, n);
    *result = nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_bilinear_f32c_smef64(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_f32c_t const *c_pairs,
                                           nk_size_t n, nk_f32c_t *results) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_real_f64 = 0.0, outer_sum_imag_f64 = 0.0;

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);
        svzero_mask_za(nk_sme_zero_za64_tile_2_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
            svbool_t batch_predicate_b32 = svwhilelt_b32((uint64_t)0, (uint64_t)(batch_size + batch_size));

            // Pass 1: Stage C_re into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat32_t c_f32x2 = svld1_f32(batch_predicate_b32,
                                                (nk_f32_t const *)c_pairs + ((row + r) * n + j) * 2);
                svfloat64_t c_real_f64 = svcvt_f64_f32_x(batch_predicate_b64, svtrn1_f32(c_f32x2, c_f32x2));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, c_real_f64);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_re_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_re_col_f64,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].real)); // v_re += c_re * b_re
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_re_col_f64,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].imag)); // v_im += c_re * b_im
            }

            // Pass 2: Stage C_im into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat32_t c_f32x2 = svld1_f32(batch_predicate_b32,
                                                (nk_f32_t const *)c_pairs + ((row + r) * n + j) * 2);
                svfloat64_t c_imag_f64 = svcvt_f64_f32_x(batch_predicate_b64, svtrn2_f32(c_f32x2, c_f32x2));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, c_imag_f64);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_im_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_im_col_f64,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].real)); // v_im += c_im * b_re
                svmops_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_im_col_f64,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].imag)); // v_re -= c_im * b_im
            }
        }

        svfloat64_t v_re_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0);
        svfloat64_t v_im_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 2, 0);

        // Deinterleave a[row:row+tile]
        svbool_t row_predicate_b32 = svwhilelt_b32((uint64_t)0, (uint64_t)(rows_remaining + rows_remaining));
        svfloat32_t a_f32x2 = svld1_f32(row_predicate_b32, (nk_f32_t const *)a_pairs + row * 2);
        svfloat64_t a_re_f64 = svcvt_f64_f32_x(row_predicate_b64, svtrn1_f32(a_f32x2, a_f32x2));
        svfloat64_t a_im_f64 = svcvt_f64_f32_x(row_predicate_b64, svtrn2_f32(a_f32x2, a_f32x2));

        // Complex dot: a * v
        outer_sum_real_f64 += svaddv_f64(
            predicate_body_b64, svsub_f64_x(row_predicate_b64, svmul_f64_x(row_predicate_b64, a_re_f64, v_re_f64),
                                            svmul_f64_x(row_predicate_b64, a_im_f64, v_im_f64)));
        outer_sum_imag_f64 += svaddv_f64(
            predicate_body_b64, svadd_f64_x(row_predicate_b64, svmul_f64_x(row_predicate_b64, a_re_f64, v_im_f64),
                                            svmul_f64_x(row_predicate_b64, a_im_f64, v_re_f64)));
    }

    results->real = (nk_f32_t)outer_sum_real_f64;
    results->imag = (nk_f32_t)outer_sum_imag_f64;
}

__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_bilinear_f64c_smef64(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_f64c_t const *c_pairs,
                                           nk_size_t n, nk_f64c_t *results) {
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_real_f64 = 0.0, outer_sum_imag_f64 = 0.0;

    svuint64_t ozaki_mask_19 = svdup_u64(nk_f64_smef64_ozaki_mask_19_bits_());
    svuint64_t ozaki_mask_17 = svdup_u64(nk_f64_smef64_ozaki_mask_17_bits_());

    for (nk_size_t row = 0; row < n; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= n) ? tile_dimension : (n - row);
        svbool_t row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_remaining);

        // Zero ZA1-6 (v_re Ozaki in ZA1-3, v_im Ozaki in ZA4-6)
        svzero_mask_za(nk_sme_zero_za64_tiles_1_6_);

        for (nk_size_t j = 0; j < n; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= n) ? tile_dimension : (n - j);
            svbool_t batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

            // Pass 1: Stage C_re into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                nk_size_t base = ((row + r) * n + j) * 2;
                svfloat64_t c_low_f64 = svld1_f64(batch_predicate_b64, (nk_f64_t const *)c_pairs + base);
                svfloat64_t c_high_f64 = svld1_f64(batch_predicate_b64, (nk_f64_t const *)c_pairs + base + svcntd());
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, svuzp1_f64(c_low_f64, c_high_f64));
            }

            // 12 MOPAs: Ozaki c_re × b_re → ZA1-3 (v_re), c_re × b_im → ZA4-6 (v_im)
            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svuint64_t c_bits = svreinterpret_u64_f64(c_col_f64);
                svfloat64_t c_s0 = svreinterpret_f64_u64(svand_u64_x(predicate_body_b64, c_bits, ozaki_mask_19));
                svfloat64_t c_res = svsub_f64_x(predicate_body_b64, c_col_f64, c_s0);
                svuint64_t c_res_bits = svreinterpret_u64_f64(c_res);
                svfloat64_t c_s1 = svreinterpret_f64_u64(svand_u64_x(predicate_body_b64, c_res_bits, ozaki_mask_17));
                svfloat64_t c_s2 = svsub_f64_x(predicate_body_b64, c_res, c_s1);

                nk_f64_t br0, br1, br2;
                nk_f64_smef64_ozaki_split_f64_(b_pairs[j + k].real, &br0, &br1, &br2);
                nk_f64_t bi0, bi1, bi2;
                nk_f64_smef64_ozaki_split_f64_(b_pairs[j + k].imag, &bi0, &bi1, &bi2);

                // v_re += c_re * b_re (6 FMOPAs into ZA1-3)
                svmopa_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br0));
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br1));
                svmopa_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(br0));
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br2));
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(br1));
                svmopa_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s2, svdup_f64(br0));

                // v_im += c_re * b_im (6 FMOPAs into ZA4-6)
                svmopa_za64_f64_m(4, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi0));
                svmopa_za64_f64_m(5, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi1));
                svmopa_za64_f64_m(5, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(bi0));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi2));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(bi1));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s2, svdup_f64(bi0));
            }

            // Pass 2: Stage C_im into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                nk_size_t base = ((row + r) * n + j) * 2;
                svfloat64_t c_low_f64 = svld1_f64(batch_predicate_b64, (nk_f64_t const *)c_pairs + base);
                svfloat64_t c_high_f64 = svld1_f64(batch_predicate_b64, (nk_f64_t const *)c_pairs + base + svcntd());
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64, svuzp2_f64(c_low_f64, c_high_f64));
            }

            // 12 MOPAs: Ozaki c_im × b_re → ZA4-6 (v_im), c_im × b_im → ZA1-3 (FMOPS for v_re)
            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, k);
                svuint64_t c_bits = svreinterpret_u64_f64(c_col_f64);
                svfloat64_t c_s0 = svreinterpret_f64_u64(svand_u64_x(predicate_body_b64, c_bits, ozaki_mask_19));
                svfloat64_t c_res = svsub_f64_x(predicate_body_b64, c_col_f64, c_s0);
                svuint64_t c_res_bits = svreinterpret_u64_f64(c_res);
                svfloat64_t c_s1 = svreinterpret_f64_u64(svand_u64_x(predicate_body_b64, c_res_bits, ozaki_mask_17));
                svfloat64_t c_s2 = svsub_f64_x(predicate_body_b64, c_res, c_s1);

                nk_f64_t br0, br1, br2;
                nk_f64_smef64_ozaki_split_f64_(b_pairs[j + k].real, &br0, &br1, &br2);
                nk_f64_t bi0, bi1, bi2;
                nk_f64_smef64_ozaki_split_f64_(b_pairs[j + k].imag, &bi0, &bi1, &bi2);

                // v_im += c_im * b_re (6 FMOPAs into ZA4-6)
                svmopa_za64_f64_m(4, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br0));
                svmopa_za64_f64_m(5, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br1));
                svmopa_za64_f64_m(5, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(br0));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(br2));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(br1));
                svmopa_za64_f64_m(6, row_predicate_b64, row_predicate_b64, c_s2, svdup_f64(br0));

                // v_re -= c_im * b_im (6 FMOPS into ZA1-3)
                svmops_za64_f64_m(1, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi0));
                svmops_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi1));
                svmops_za64_f64_m(2, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(bi0));
                svmops_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s0, svdup_f64(bi2));
                svmops_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s1, svdup_f64(bi1));
                svmops_za64_f64_m(3, row_predicate_b64, row_predicate_b64, c_s2, svdup_f64(bi0));
            }
        }

        // Sum v_re: ZA3 + ZA2 + ZA1 (smallest magnitude first)
        svfloat64_t v_re_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 3, 0);
        v_re_f64 = svadd_f64_x(predicate_body_b64, v_re_f64,
                               svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 2, 0));
        v_re_f64 = svadd_f64_x(predicate_body_b64, v_re_f64,
                               svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 1, 0));

        // Sum v_im: ZA6 + ZA5 + ZA4
        svfloat64_t v_im_f64 = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 6, 0);
        v_im_f64 = svadd_f64_x(predicate_body_b64, v_im_f64,
                               svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 5, 0));
        v_im_f64 = svadd_f64_x(predicate_body_b64, v_im_f64,
                               svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 4, 0));

        // Load and deinterleave a[row:row+tile]
        svfloat64_t a_low_f64 = svld1_f64(row_predicate_b64, (nk_f64_t const *)a_pairs + row * 2);
        svfloat64_t a_high_f64 = svld1_f64(row_predicate_b64, (nk_f64_t const *)a_pairs + row * 2 + svcntd());
        svfloat64_t a_re_f64 = svuzp1_f64(a_low_f64, a_high_f64);
        svfloat64_t a_im_f64 = svuzp2_f64(a_low_f64, a_high_f64);

        // Complex dot: a * v
        outer_sum_real_f64 += svaddv_f64(
            predicate_body_b64, svsub_f64_x(row_predicate_b64, svmul_f64_x(row_predicate_b64, a_re_f64, v_re_f64),
                                            svmul_f64_x(row_predicate_b64, a_im_f64, v_im_f64)));
        outer_sum_imag_f64 += svaddv_f64(
            predicate_body_b64, svadd_f64_x(row_predicate_b64, svmul_f64_x(row_predicate_b64, a_re_f64, v_im_f64),
                                            svmul_f64_x(row_predicate_b64, a_im_f64, v_re_f64)));
    }

    results->real = outer_sum_real_f64;
    results->imag = outer_sum_imag_f64;
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
