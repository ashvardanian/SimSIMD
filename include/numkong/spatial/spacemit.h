/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for SpacemiT (RVV 1.0) CPUs.
 *  @file include/numkong/spatial/spacemit.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  RVV uses vector length agnostic programming where:
 *  - `vsetvl_e*m*(n)` sets VL = min(n, VLMAX) and returns actual VL
 *  - Loads/stores with VL automatically handle partial vectors (tail elements)
 *  - No explicit masking needed for simple reductions
 *
 *  This file contains base RVV 1.0 operations (i8, u8, f32, f64).
 *  For f16 (Zvfh) see sifive.h, for bf16 (Zvfbfwma) see xuantie.h.
 *
 *  Precision strategies matching Skylake:
 *  - i8 L2: diff (i8-i8 → i16), square (i16 × i16 → i32), reduce to i32
 *  - u8 L2: |diff| via widening, square → u32, reduce to u32
 *  - f32: Widen to f64 for accumulation, downcast result to f32
 *  - f64: Direct f64 accumulation
 */
#ifndef NK_SPATIAL_SPACEMIT_H
#define NK_SPATIAL_SPACEMIT_H

#if NK_TARGET_RISCV_
#if NK_TARGET_SPACEMIT

#include "numkong/types.h"
#include "numkong/cast/spacemit.h" // `nk_e4m3m1_to_f32m4_spacemit_`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    vint32m1_t sum_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);
        // Widening subtract: i8 - i8 → i16
        vint16m2_t diff_i16m2 = __riscv_vwsub_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        // Widening square: i16 × i16 → i32
        vint32m4_t sq_i32m4 = __riscv_vwmul_vv_i32m4(diff_i16m2, diff_i16m2, vector_length);
        // Reduce to scalar
        sum_i32m1 = __riscv_vredsum_vs_i32m4_i32m1(sq_i32m4, sum_i32m1, vector_length);
    }
    *result = (nk_u32_t)__riscv_vmv_x_s_i32m1_i32(sum_i32m1);
}

NK_PUBLIC void nk_l2_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i8_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

NK_PUBLIC void nk_l2sq_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);
        // Compute |a - b| using saturating subtraction: max(a-b, b-a) = (a -sat b) | (b -sat a)
        vuint8m1_t diff_ab_u8m1 = __riscv_vssubu_vv_u8m1(a_u8m1, b_u8m1, vector_length);
        vuint8m1_t diff_ba_u8m1 = __riscv_vssubu_vv_u8m1(b_u8m1, a_u8m1, vector_length);
        vuint8m1_t abs_diff_u8m1 = __riscv_vor_vv_u8m1(diff_ab_u8m1, diff_ba_u8m1, vector_length);
        // Widening multiply: u8 × u8 → u16
        vuint16m2_t sq_u16m2 = __riscv_vwmulu_vv_u16m2(abs_diff_u8m1, abs_diff_u8m1, vector_length);
        // Widening reduce: u16 → u32
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(sq_u16m2, sum_u32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_l2_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u8_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

NK_PUBLIC void nk_l2sq_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);
        // Compute difference in f32
        vfloat32m1_t diff_f32m1 = __riscv_vfsub_vv_f32m1(a_f32m1, b_f32m1, vector_length);
        // Widening square: f32 × f32 → f64
        vfloat64m2_t sq_f64m2 = __riscv_vfwmul_vv_f64m2(diff_f32m1, diff_f32m1, vector_length);
        // Ordered reduction sum
        sum_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(sq_f64m2, sum_f64m1, vector_length);
    }
    // Downcast f64 result to f32
    *result = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(sum_f64m1);
}

NK_PUBLIC void nk_l2_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_l2sq_f32_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

NK_PUBLIC void nk_l2sq_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f64_t *result) {
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);
        // Compute difference and accumulate diff² into vector lanes
        vfloat64m1_t diff_f64m1 = __riscv_vfsub_vv_f64m1(a_f64m1, b_f64m1, vector_length);
        sum_f64m1 = __riscv_vfmacc_vv_f64m1(sum_f64m1, diff_f64m1, diff_f64m1, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_f64m1, zero_f64m1, vector_length_max));
}

NK_PUBLIC void nk_l2_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    nk_l2sq_f64_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f64_sqrt_serial(*result);
}

NK_PUBLIC void nk_angular_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    vint32m1_t dot_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t a_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t b_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);

        // dot += a × b (widened to i32)
        vint16m2_t ab_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        dot_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_i16m2, dot_i32m1, vector_length);

        // a_norm_sq += a × a
        vint16m2_t aa_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, a_i8m1, vector_length);
        a_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(aa_i16m2, a_norm_sq_i32m1, vector_length);

        // b_norm_sq += b × b
        vint16m2_t bb_i16m2 = __riscv_vwmul_vv_i16m2(b_i8m1, b_i8m1, vector_length);
        b_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(bb_i16m2, b_norm_sq_i32m1, vector_length);
    }

    nk_i32_t dot_i32 = __riscv_vmv_x_s_i32m1_i32(dot_i32m1);
    nk_i32_t a_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(a_norm_sq_i32m1);
    nk_i32_t b_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(b_norm_sq_i32m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_i32 == 0 && b_norm_sq_i32 == 0) { *result = 0.0f; }
    else if (dot_i32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_i32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_i32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_i32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_angular_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    vuint32m1_t dot_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t a_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t b_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);

        // dot += a × b (widened to u32)
        vuint16m2_t ab_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, b_u8m1, vector_length);
        dot_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_u16m2, dot_u32m1, vector_length);

        // a_norm_sq += a × a
        vuint16m2_t aa_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, a_u8m1, vector_length);
        a_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(aa_u16m2, a_norm_sq_u32m1, vector_length);

        // b_norm_sq += b × b
        vuint16m2_t bb_u16m2 = __riscv_vwmulu_vv_u16m2(b_u8m1, b_u8m1, vector_length);
        b_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(bb_u16m2, b_norm_sq_u32m1, vector_length);
    }

    nk_u32_t dot_u32 = __riscv_vmv_x_s_u32m1_u32(dot_u32m1);
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(a_norm_sq_u32m1);
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(b_norm_sq_u32m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_u32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_u32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_angular_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    vfloat64m1_t dot_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t a_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t b_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);

        // dot += a × b (widened to f64)
        vfloat64m2_t ab_f64m2 = __riscv_vfwmul_vv_f64m2(a_f32m1, b_f32m1, vector_length);
        dot_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(ab_f64m2, dot_f64m1, vector_length);

        // a_norm_sq += a × a
        vfloat64m2_t aa_f64m2 = __riscv_vfwmul_vv_f64m2(a_f32m1, a_f32m1, vector_length);
        a_norm_sq_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(aa_f64m2, a_norm_sq_f64m1, vector_length);

        // b_norm_sq += b × b
        vfloat64m2_t bb_f64m2 = __riscv_vfwmul_vv_f64m2(b_f32m1, b_f32m1, vector_length);
        b_norm_sq_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(bb_f64m2, b_norm_sq_f64m1, vector_length);
    }

    nk_f64_t dot_f64 = __riscv_vfmv_f_s_f64m1_f64(dot_f64m1);
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(a_norm_sq_f64m1);
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(b_norm_sq_f64m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0f; }
    else if (dot_f64 == 0.0) { *result = 1.0f; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_serial(a_norm_sq_f64) * nk_f64_rsqrt_serial(b_norm_sq_f64);
        *result = (nk_f32_t)(unclipped > 0 ? unclipped : 0);
    }
}

NK_PUBLIC void nk_angular_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f64_t *result) {
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t dot_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    vfloat64m1_t a_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    vfloat64m1_t b_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);

        // Accumulate into vector lanes using FMA
        dot_f64m1 = __riscv_vfmacc_vv_f64m1(dot_f64m1, a_f64m1, b_f64m1, vector_length);
        a_norm_sq_f64m1 = __riscv_vfmacc_vv_f64m1(a_norm_sq_f64m1, a_f64m1, a_f64m1, vector_length);
        b_norm_sq_f64m1 = __riscv_vfmacc_vv_f64m1(b_norm_sq_f64m1, b_f64m1, b_f64m1, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    nk_f64_t dot_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(dot_f64m1, zero_f64m1, vector_length_max));
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(a_norm_sq_f64m1, zero_f64m1, vector_length_max));
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(b_norm_sq_f64m1, zero_f64m1, vector_length_max));

    // Normalize: 1 - dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0; }
    else if (dot_f64 == 0.0) { *result = 1.0; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_serial(a_norm_sq_f64) * nk_f64_rsqrt_serial(b_norm_sq_f64);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_l2sq_f16_spacemit(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_spacemit_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_spacemit_(b_u16m1, vector_length);

        // Compute difference and square in f32 (matching Skylake precision)
        vfloat32m2_t diff_f32m2 = __riscv_vfsub_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t sq_f32m2 = __riscv_vfmul_vv_f32m2(diff_f32m2, diff_f32m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(sq_f32m2, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_l2_f16_spacemit(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_l2sq_f16_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

NK_PUBLIC void nk_angular_f16_spacemit(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    vfloat32m1_t dot_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_spacemit_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_spacemit_(b_u16m1, vector_length);

        // dot += a * b in f32 (matching Skylake precision)
        vfloat32m2_t ab_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        dot_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, dot_f32m1, vector_length);

        // a_norm_sq += a * a
        vfloat32m2_t aa_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, a_f32m2, vector_length);
        a_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32m2, a_norm_sq_f32m1, vector_length);

        // b_norm_sq += b * b
        vfloat32m2_t bb_f32m2 = __riscv_vfmul_vv_f32m2(b_f32m2, b_f32m2, vector_length);
        b_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32m2, b_norm_sq_f32m1, vector_length);
    }

    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(dot_f32m1);
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(a_norm_sq_f32m1);
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(b_norm_sq_f32m1);

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_serial(a_norm_sq_f32) * nk_f32_rsqrt_serial(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_l2sq_bf16_spacemit(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_spacemit_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_spacemit_(b_u16m1, vector_length);

        // Compute difference and square in f32 (matching Skylake precision)
        vfloat32m2_t diff_f32m2 = __riscv_vfsub_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t sq_f32m2 = __riscv_vfmul_vv_f32m2(diff_f32m2, diff_f32m2, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(sq_f32m2, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_l2_bf16_spacemit(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_l2sq_bf16_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

NK_PUBLIC void nk_angular_bf16_spacemit(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    vfloat32m1_t dot_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_spacemit_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_spacemit_(b_u16m1, vector_length);

        // dot += a * b in f32 (matching Skylake precision)
        vfloat32m2_t ab_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        dot_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(ab_f32m2, dot_f32m1, vector_length);

        // a_norm_sq += a * a
        vfloat32m2_t aa_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, a_f32m2, vector_length);
        a_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(aa_f32m2, a_norm_sq_f32m1, vector_length);

        // b_norm_sq += b * b
        vfloat32m2_t bb_f32m2 = __riscv_vfmul_vv_f32m2(b_f32m2, b_f32m2, vector_length);
        b_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m2_f32m1(bb_f32m2, b_norm_sq_f32m1, vector_length);
    }

    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(dot_f32m1);
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(a_norm_sq_f32m1);
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(b_norm_sq_f32m1);

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_serial(a_norm_sq_f32) * nk_f32_rsqrt_serial(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_l2sq_e4m3_spacemit(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_spacemit_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_spacemit_(b_u8m1, vector_length);

        // Compute difference and square in f32 (matching Skylake precision)
        vfloat32m4_t diff_f32m4 = __riscv_vfsub_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t sq_f32m4 = __riscv_vfmul_vv_f32m4(diff_f32m4, diff_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(sq_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_l2_e4m3_spacemit(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_l2sq_e4m3_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

NK_PUBLIC void nk_angular_e4m3_spacemit(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    vfloat32m1_t dot_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_spacemit_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_spacemit_(b_u8m1, vector_length);

        // dot += a * b in f32 (matching Skylake precision)
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        dot_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, dot_f32m1, vector_length);

        // a_norm_sq += a * a
        vfloat32m4_t aa_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, a_f32m4, vector_length);
        a_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(aa_f32m4, a_norm_sq_f32m1, vector_length);

        // b_norm_sq += b * b
        vfloat32m4_t bb_f32m4 = __riscv_vfmul_vv_f32m4(b_f32m4, b_f32m4, vector_length);
        b_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(bb_f32m4, b_norm_sq_f32m1, vector_length);
    }

    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(dot_f32m1);
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(a_norm_sq_f32m1);
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(b_norm_sq_f32m1);

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_serial(a_norm_sq_f32) * nk_f32_rsqrt_serial(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_l2sq_e5m2_spacemit(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_spacemit_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_spacemit_(b_u8m1, vector_length);

        // Compute difference and square in f32 (matching Skylake precision)
        vfloat32m4_t diff_f32m4 = __riscv_vfsub_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t sq_f32m4 = __riscv_vfmul_vv_f32m4(diff_f32m4, diff_f32m4, vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(sq_f32m4, sum_f32m1, vector_length);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_l2_e5m2_spacemit(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_l2sq_e5m2_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

NK_PUBLIC void nk_angular_e5m2_spacemit(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    vfloat32m1_t dot_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t a_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t b_norm_sq_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_spacemit_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_spacemit_(b_u8m1, vector_length);

        // dot += a * b in f32 (matching Skylake precision)
        vfloat32m4_t ab_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        dot_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(ab_f32m4, dot_f32m1, vector_length);

        // a_norm_sq += a * a
        vfloat32m4_t aa_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, a_f32m4, vector_length);
        a_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(aa_f32m4, a_norm_sq_f32m1, vector_length);

        // b_norm_sq += b * b
        vfloat32m4_t bb_f32m4 = __riscv_vfmul_vv_f32m4(b_f32m4, b_f32m4, vector_length);
        b_norm_sq_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(bb_f32m4, b_norm_sq_f32m1, vector_length);
    }

    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(dot_f32m1);
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(a_norm_sq_f32m1);
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(b_norm_sq_f32m1);

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_serial(a_norm_sq_f32) * nk_f32_rsqrt_serial(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_l2sq_i4_spacemit(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    nk_size_t n_bytes = count_scalars / 2;
    vint32m1_t sum_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        // Load packed bytes
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        // Extract high nibble (even indices)
        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        // Sign extend high nibble: (x ^ 8) - 8
        vint8m1_t a_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_high_u8m1), 8, vector_length), 8, vector_length);
        // Extract low nibble (odd indices)
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);
        // Sign extend low nibble
        vint8m1_t a_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_low_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_low_u8m1), 8, vector_length), 8, vector_length);
        // Compute differences (widening: i8 - i8 → i16)
        vint16m2_t diff_high_i16m2 = __riscv_vwsub_vv_i16m2(a_high_i8m1, b_high_i8m1, vector_length);
        vint16m2_t diff_low_i16m2 = __riscv_vwsub_vv_i16m2(a_low_i8m1, b_low_i8m1, vector_length);
        // Square (widening: i16 x i16 → i32)
        vint32m4_t sq_high_i32m4 = __riscv_vwmul_vv_i32m4(diff_high_i16m2, diff_high_i16m2, vector_length);
        vint32m4_t sq_low_i32m4 = __riscv_vwmul_vv_i32m4(diff_low_i16m2, diff_low_i16m2, vector_length);
        // Reduce
        sum_i32m1 = __riscv_vredsum_vs_i32m4_i32m1(sq_high_i32m4, sum_i32m1, vector_length);
        sum_i32m1 = __riscv_vredsum_vs_i32m4_i32m1(sq_low_i32m4, sum_i32m1, vector_length);
    }
    *result = (nk_u32_t)__riscv_vmv_x_s_i32m1_i32(sum_i32m1);
}

NK_PUBLIC void nk_l2_i4_spacemit(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i4_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_i4_spacemit(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    nk_size_t n_bytes = count_scalars / 2;
    vint32m1_t dot_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t a_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t b_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);

    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vint8m1_t a_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_high_u8m1), 8, vector_length), 8, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);
        vint8m1_t a_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_low_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_low_u8m1), 8, vector_length), 8, vector_length);

        // dot += a * b for high nibbles
        vint16m2_t ab_high_i16m2 = __riscv_vwmul_vv_i16m2(a_high_i8m1, b_high_i8m1, vector_length);
        dot_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_high_i16m2, dot_i32m1, vector_length);
        // dot += a * b for low nibbles
        vint16m2_t ab_low_i16m2 = __riscv_vwmul_vv_i16m2(a_low_i8m1, b_low_i8m1, vector_length);
        dot_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_low_i16m2, dot_i32m1, vector_length);
        // a_norm_sq for high and low nibbles
        vint16m2_t aa_high_i16m2 = __riscv_vwmul_vv_i16m2(a_high_i8m1, a_high_i8m1, vector_length);
        a_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(aa_high_i16m2, a_norm_sq_i32m1, vector_length);
        vint16m2_t aa_low_i16m2 = __riscv_vwmul_vv_i16m2(a_low_i8m1, a_low_i8m1, vector_length);
        a_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(aa_low_i16m2, a_norm_sq_i32m1, vector_length);
        // b_norm_sq for high and low nibbles
        vint16m2_t bb_high_i16m2 = __riscv_vwmul_vv_i16m2(b_high_i8m1, b_high_i8m1, vector_length);
        b_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(bb_high_i16m2, b_norm_sq_i32m1, vector_length);
        vint16m2_t bb_low_i16m2 = __riscv_vwmul_vv_i16m2(b_low_i8m1, b_low_i8m1, vector_length);
        b_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(bb_low_i16m2, b_norm_sq_i32m1, vector_length);
    }

    nk_i32_t dot_i32 = __riscv_vmv_x_s_i32m1_i32(dot_i32m1);
    nk_i32_t a_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(a_norm_sq_i32m1);
    nk_i32_t b_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(b_norm_sq_i32m1);

    if (a_norm_sq_i32 == 0 && b_norm_sq_i32 == 0) { *result = 0.0f; }
    else if (dot_i32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_i32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_i32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_i32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_l2sq_u4_spacemit(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    nk_size_t n_bytes = count_scalars / 2;
    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        // Extract high nibble
        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        // Extract low nibble
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);
        // Compute |a - b| using saturating subtraction: max(a-b, b-a)
        vuint8m1_t diff_high_ab_u8m1 = __riscv_vssubu_vv_u8m1(a_high_u8m1, b_high_u8m1, vector_length);
        vuint8m1_t diff_high_ba_u8m1 = __riscv_vssubu_vv_u8m1(b_high_u8m1, a_high_u8m1, vector_length);
        vuint8m1_t abs_diff_high_u8m1 = __riscv_vor_vv_u8m1(diff_high_ab_u8m1, diff_high_ba_u8m1, vector_length);
        vuint8m1_t diff_low_ab_u8m1 = __riscv_vssubu_vv_u8m1(a_low_u8m1, b_low_u8m1, vector_length);
        vuint8m1_t diff_low_ba_u8m1 = __riscv_vssubu_vv_u8m1(b_low_u8m1, a_low_u8m1, vector_length);
        vuint8m1_t abs_diff_low_u8m1 = __riscv_vor_vv_u8m1(diff_low_ab_u8m1, diff_low_ba_u8m1, vector_length);
        // Square (widening: u8 x u8 → u16)
        vuint16m2_t sq_high_u16m2 = __riscv_vwmulu_vv_u16m2(abs_diff_high_u8m1, abs_diff_high_u8m1, vector_length);
        vuint16m2_t sq_low_u16m2 = __riscv_vwmulu_vv_u16m2(abs_diff_low_u8m1, abs_diff_low_u8m1, vector_length);
        // Reduce (widening: u16 → u32)
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(sq_high_u16m2, sum_u32m1, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(sq_low_u16m2, sum_u32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_l2_u4_spacemit(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u4_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_u4_spacemit(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    nk_size_t n_bytes = count_scalars / 2;
    vuint32m1_t dot_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t a_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t b_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // dot += a * b for high nibbles
        vuint16m2_t ab_high_u16m2 = __riscv_vwmulu_vv_u16m2(a_high_u8m1, b_high_u8m1, vector_length);
        dot_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_high_u16m2, dot_u32m1, vector_length);
        // dot += a * b for low nibbles
        vuint16m2_t ab_low_u16m2 = __riscv_vwmulu_vv_u16m2(a_low_u8m1, b_low_u8m1, vector_length);
        dot_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_low_u16m2, dot_u32m1, vector_length);
        // a_norm_sq for high and low nibbles
        vuint16m2_t aa_high_u16m2 = __riscv_vwmulu_vv_u16m2(a_high_u8m1, a_high_u8m1, vector_length);
        a_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(aa_high_u16m2, a_norm_sq_u32m1, vector_length);
        vuint16m2_t aa_low_u16m2 = __riscv_vwmulu_vv_u16m2(a_low_u8m1, a_low_u8m1, vector_length);
        a_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(aa_low_u16m2, a_norm_sq_u32m1, vector_length);
        // b_norm_sq for high and low nibbles
        vuint16m2_t bb_high_u16m2 = __riscv_vwmulu_vv_u16m2(b_high_u8m1, b_high_u8m1, vector_length);
        b_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(bb_high_u16m2, b_norm_sq_u32m1, vector_length);
        vuint16m2_t bb_low_u16m2 = __riscv_vwmulu_vv_u16m2(b_low_u8m1, b_low_u8m1, vector_length);
        b_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(bb_low_u16m2, b_norm_sq_u32m1, vector_length);
    }

    nk_u32_t dot_u32 = __riscv_vmv_x_s_u32m1_u32(dot_u32m1);
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(a_norm_sq_u32m1);
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(b_norm_sq_u32m1);

    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_u32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_u32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SPACEMIT
#endif // NK_TARGET_RISCV_

#endif // NK_SPATIAL_SPACEMIT_H
