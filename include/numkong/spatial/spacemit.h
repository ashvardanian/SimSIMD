/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for SpacemiT (RVV 1.0).
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
 *  - i8 L2: diff (i8-i8 -> i16), square (i16*i16 -> i32), reduce to i32
 *  - u8 L2: |diff| via widening, square -> u32, reduce to u32
 *  - f32: Widen to f64 for accumulation, downcast result to f32
 *  - f64: Direct f64 accumulation
 */
#ifndef NK_SPATIAL_SPACEMIT_H
#define NK_SPATIAL_SPACEMIT_H

#if NK_TARGET_RISCV_
#if NK_TARGET_SPACEMIT

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  L2 squared distance of two i8 vectors with i32 accumulation on SpacemiT.
 *
 *  Uses widening subtract (i8-i8 -> i16) followed by widening multiply (i16*i16 -> i32).
 *  VL-based loop handles all tail elements automatically.
 */
NK_PUBLIC void nk_l2sq_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    vint32m1_t sum_i32x1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8x1 = __riscv_vle8_v_i8m1(a_scalars, vl);
        vint8m1_t b_i8x1 = __riscv_vle8_v_i8m1(b_scalars, vl);
        // Widening subtract: i8 - i8 -> i16
        vint16m2_t diff_i16x2 = __riscv_vwsub_vv_i16m2(a_i8x1, b_i8x1, vl);
        // Widening square: i16 * i16 -> i32
        vint32m4_t sq_i32x4 = __riscv_vwmul_vv_i32m4(diff_i16x2, diff_i16x2, vl);
        // Reduce to scalar
        sum_i32x1 = __riscv_vredsum_vs_i32m4_i32m1(sq_i32x4, sum_i32x1, vl);
    }
    *result = (nk_u32_t)__riscv_vmv_x_s_i32m1_i32(sum_i32x1);
}

NK_PUBLIC void nk_l2_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i8_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

/**
 *  @brief  L2 squared distance of two u8 vectors with u32 accumulation on SpacemiT.
 *
 *  Computes |a-b| by taking the max of (a-b) and (b-a) using saturating subtraction.
 *  The result is always non-negative and fits in u8.
 */
NK_PUBLIC void nk_l2sq_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_u32_t *result) {
    vuint32m1_t sum_u32x1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8x1 = __riscv_vle8_v_u8m1(a_scalars, vl);
        vuint8m1_t b_u8x1 = __riscv_vle8_v_u8m1(b_scalars, vl);
        // Compute |a - b| using saturating subtraction: max(a-b, b-a) = (a -sat b) | (b -sat a)
        vuint8m1_t diff_ab_u8x1 = __riscv_vssubu_vv_u8m1(a_u8x1, b_u8x1, vl);
        vuint8m1_t diff_ba_u8x1 = __riscv_vssubu_vv_u8m1(b_u8x1, a_u8x1, vl);
        vuint8m1_t abs_diff_u8x1 = __riscv_vor_vv_u8m1(diff_ab_u8x1, diff_ba_u8x1, vl);
        // Widening multiply: u8 * u8 -> u16
        vuint16m2_t sq_u16x2 = __riscv_vwmulu_vv_u16m2(abs_diff_u8x1, abs_diff_u8x1, vl);
        // Widening reduce: u16 -> u32
        sum_u32x1 = __riscv_vwredsumu_vs_u16m2_u32m1(sq_u16x2, sum_u32x1, vl);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32x1);
}

NK_PUBLIC void nk_l2_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u8_spacemit(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_serial((nk_f32_t)d2);
}

/**
 *  @brief  L2 squared distance of two f32 vectors with f64 accumulation on SpacemiT.
 *
 *  Uses widening operations (f32 -> f64) for higher precision accumulation,
 *  matching the Skylake strategy for avoiding catastrophic cancellation.
 */
NK_PUBLIC void nk_l2sq_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    vfloat64m1_t sum_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32x1 = __riscv_vle32_v_f32m1(a_scalars, vl);
        vfloat32m1_t b_f32x1 = __riscv_vle32_v_f32m1(b_scalars, vl);
        // Compute difference in f32
        vfloat32m1_t diff_f32x1 = __riscv_vfsub_vv_f32m1(a_f32x1, b_f32x1, vl);
        // Widening square: f32 × f32 → f64
        vfloat64m2_t sq_f64x2 = __riscv_vfwmul_vv_f64m2(diff_f32x1, diff_f32x1, vl);
        // Ordered reduction sum
        sum_f64x1 = __riscv_vfredusum_vs_f64m2_f64m1(sq_f64x2, sum_f64x1, vl);
    }
    // Downcast f64 result to f32
    *result = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(sum_f64x1);
}

NK_PUBLIC void nk_l2_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_l2sq_f32_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_serial(*result);
}

/**
 *  @brief  L2 squared distance of two f64 vectors on SpacemiT.
 */
NK_PUBLIC void nk_l2sq_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f64_t *result) {
    size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64x1 = __riscv_vle64_v_f64m1(a_scalars, vl);
        vfloat64m1_t b_f64x1 = __riscv_vle64_v_f64m1(b_scalars, vl);
        // Compute difference and accumulate diff² into vector lanes
        vfloat64m1_t diff_f64x1 = __riscv_vfsub_vv_f64m1(a_f64x1, b_f64x1, vl);
        sum_f64x1 = __riscv_vfmacc_vv_f64m1(sum_f64x1, diff_f64x1, diff_f64x1, vl);
    }
    // Single horizontal reduction at the end
    vfloat64m1_t zero_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_f64x1, zero_f64x1, vlmax));
}

NK_PUBLIC void nk_l2_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    nk_l2sq_f64_spacemit(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f64_sqrt_serial(*result);
}

/**
 *  @brief  Angular (cosine) distance of two i8 vectors on SpacemiT.
 *
 *  Computes: 1 - dot(a,b) / (‖a‖ × ‖b‖)
 *  Uses i32 accumulation for dot product and norms, then normalizes in f32.
 */
NK_PUBLIC void nk_angular_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    vint32m1_t dot_i32x1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t a_norm_sq_i32x1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t b_norm_sq_i32x1 = __riscv_vmv_v_x_i32m1(0, 1);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8x1 = __riscv_vle8_v_i8m1(a_scalars, vl);
        vint8m1_t b_i8x1 = __riscv_vle8_v_i8m1(b_scalars, vl);

        // dot += a * b (widened to i32)
        vint16m2_t ab_i16x2 = __riscv_vwmul_vv_i16m2(a_i8x1, b_i8x1, vl);
        dot_i32x1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_i16x2, dot_i32x1, vl);

        // a_norm_sq += a * a
        vint16m2_t aa_i16x2 = __riscv_vwmul_vv_i16m2(a_i8x1, a_i8x1, vl);
        a_norm_sq_i32x1 = __riscv_vwredsum_vs_i16m2_i32m1(aa_i16x2, a_norm_sq_i32x1, vl);

        // b_norm_sq += b * b
        vint16m2_t bb_i16x2 = __riscv_vwmul_vv_i16m2(b_i8x1, b_i8x1, vl);
        b_norm_sq_i32x1 = __riscv_vwredsum_vs_i16m2_i32m1(bb_i16x2, b_norm_sq_i32x1, vl);
    }

    nk_i32_t dot_i32 = __riscv_vmv_x_s_i32m1_i32(dot_i32x1);
    nk_i32_t a_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(a_norm_sq_i32x1);
    nk_i32_t b_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(b_norm_sq_i32x1);

    // Normalize: 1 - dot / sqrt(a_norm_sq * b_norm_sq)
    if (a_norm_sq_i32 == 0 && b_norm_sq_i32 == 0) { *result = 0.0f; }
    else if (dot_i32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_i32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_i32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_i32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

/**
 *  @brief  Angular (cosine) distance of two u8 vectors on SpacemiT.
 *
 *  Computes: 1 - dot(a,b) / (‖a‖ × ‖b‖)
 *  Uses u32 accumulation for dot product and norms, then normalizes in f32.
 */
NK_PUBLIC void nk_angular_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    vuint32m1_t dot_u32x1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t a_norm_sq_u32x1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t b_norm_sq_u32x1 = __riscv_vmv_v_x_u32m1(0, 1);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8x1 = __riscv_vle8_v_u8m1(a_scalars, vl);
        vuint8m1_t b_u8x1 = __riscv_vle8_v_u8m1(b_scalars, vl);

        // dot += a * b (widened to u32)
        vuint16m2_t ab_u16x2 = __riscv_vwmulu_vv_u16m2(a_u8x1, b_u8x1, vl);
        dot_u32x1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_u16x2, dot_u32x1, vl);

        // a_norm_sq += a * a
        vuint16m2_t aa_u16x2 = __riscv_vwmulu_vv_u16m2(a_u8x1, a_u8x1, vl);
        a_norm_sq_u32x1 = __riscv_vwredsumu_vs_u16m2_u32m1(aa_u16x2, a_norm_sq_u32x1, vl);

        // b_norm_sq += b * b
        vuint16m2_t bb_u16x2 = __riscv_vwmulu_vv_u16m2(b_u8x1, b_u8x1, vl);
        b_norm_sq_u32x1 = __riscv_vwredsumu_vs_u16m2_u32m1(bb_u16x2, b_norm_sq_u32x1, vl);
    }

    nk_u32_t dot_u32 = __riscv_vmv_x_s_u32m1_u32(dot_u32x1);
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(a_norm_sq_u32x1);
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(b_norm_sq_u32x1);

    // Normalize: 1 - dot / sqrt(a_norm_sq * b_norm_sq)
    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_u32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_u32 * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

/**
 *  @brief  Angular (cosine) distance of two f32 vectors on SpacemiT.
 *
 *  Uses f64 accumulation for all three accumulators (dot, norm_a_sq, norm_b_sq),
 *  matching the Skylake strategy for avoiding catastrophic cancellation.
 */
NK_PUBLIC void nk_angular_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    vfloat64m1_t dot_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t a_norm_sq_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t b_norm_sq_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, 1);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32x1 = __riscv_vle32_v_f32m1(a_scalars, vl);
        vfloat32m1_t b_f32x1 = __riscv_vle32_v_f32m1(b_scalars, vl);

        // dot += a * b (widened to f64)
        vfloat64m2_t ab_f64x2 = __riscv_vfwmul_vv_f64m2(a_f32x1, b_f32x1, vl);
        dot_f64x1 = __riscv_vfredusum_vs_f64m2_f64m1(ab_f64x2, dot_f64x1, vl);

        // a_norm_sq += a * a
        vfloat64m2_t aa_f64x2 = __riscv_vfwmul_vv_f64m2(a_f32x1, a_f32x1, vl);
        a_norm_sq_f64x1 = __riscv_vfredusum_vs_f64m2_f64m1(aa_f64x2, a_norm_sq_f64x1, vl);

        // b_norm_sq += b * b
        vfloat64m2_t bb_f64x2 = __riscv_vfwmul_vv_f64m2(b_f32x1, b_f32x1, vl);
        b_norm_sq_f64x1 = __riscv_vfredusum_vs_f64m2_f64m1(bb_f64x2, b_norm_sq_f64x1, vl);
    }

    nk_f64_t dot_f64 = __riscv_vfmv_f_s_f64m1_f64(dot_f64x1);
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(a_norm_sq_f64x1);
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(b_norm_sq_f64x1);

    // Normalize: 1 - dot / sqrt(a_norm_sq * b_norm_sq)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0f; }
    else if (dot_f64 == 0.0) { *result = 1.0f; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_serial(a_norm_sq_f64) * nk_f64_rsqrt_serial(b_norm_sq_f64);
        *result = (nk_f32_t)(unclipped > 0 ? unclipped : 0);
    }
}

/**
 *  @brief  Angular (cosine) distance of two f64 vectors on SpacemiT.
 *
 *  Uses direct f64 accumulation for all three accumulators.
 */
NK_PUBLIC void nk_angular_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f64_t *result) {
    size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t dot_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t a_norm_sq_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t b_norm_sq_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);

    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64x1 = __riscv_vle64_v_f64m1(a_scalars, vl);
        vfloat64m1_t b_f64x1 = __riscv_vle64_v_f64m1(b_scalars, vl);

        // Accumulate into vector lanes using FMA
        dot_f64x1 = __riscv_vfmacc_vv_f64m1(dot_f64x1, a_f64x1, b_f64x1, vl);
        a_norm_sq_f64x1 = __riscv_vfmacc_vv_f64m1(a_norm_sq_f64x1, a_f64x1, a_f64x1, vl);
        b_norm_sq_f64x1 = __riscv_vfmacc_vv_f64m1(b_norm_sq_f64x1, b_f64x1, b_f64x1, vl);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat64m1_t zero_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t dot_f64 = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(dot_f64x1, zero_f64x1, vlmax));
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(a_norm_sq_f64x1, zero_f64x1, vlmax));
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(b_norm_sq_f64x1, zero_f64x1, vlmax));

    // Normalize: 1 - dot / sqrt(a_norm_sq × b_norm_sq)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0; }
    else if (dot_f64 == 0.0) { *result = 1.0; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_serial(a_norm_sq_f64) * nk_f64_rsqrt_serial(b_norm_sq_f64);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SPACEMIT
#endif // NK_TARGET_RISCV_

#endif // NK_SPATIAL_SPACEMIT_H
