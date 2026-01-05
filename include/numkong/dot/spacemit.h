/**
 *  @brief SIMD-accelerated Dot Products for Real Numbers optimized for SpacemiT (RVV 1.0).
 *  @file include/numkong/dot/spacemit.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  SpacemiT K1 and similar chips implement base RVV 1.0 without half-precision extensions.
 *  RVV uses vector length agnostic programming where:
 *  - `vsetvl_e*m*(n)` sets VL = min(n, VLMAX) and returns actual VL
 *  - Loads/stores with VL automatically handle partial vectors (tail elements)
 *  - No explicit masking needed for simple reductions
 *
 *  This file contains base RVV 1.0 operations (i8, u8, f32, f64).
 *  For f16 (Zvfh) see sifive.h, for bf16 (Zvfbfwma) see xuantie.h.
 *
 *  Widening operations:
 *  - i8 × i8 → i16 via vwmul, then i16 reduction → i32 via vwredsum
 *  - f32 × f32 → f64 via vfwmul (for precision, like Skylake)
 */
#ifndef NK_DOT_SPACEMIT_H
#define NK_DOT_SPACEMIT_H

#if NK_TARGET_RISCV_
#if NK_TARGET_SPACEMIT

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Dot product of two i8 vectors with i32 accumulation on SpacemiT.
 *
 *  Uses widening multiply (i8 × i8 → i16) followed by widening reduction sum (i16 -> i32).
 *  VL-based loop handles all tail elements automatically.
 */
NK_PUBLIC void nk_dot_i8_spacemit(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_i32_t *result) {
    vint32m1_t sum_i32x1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8x1 = __riscv_vle8_v_i8m1(a_scalars, vl);
        vint8m1_t b_i8x1 = __riscv_vle8_v_i8m1(b_scalars, vl);
        // Widening multiply: i8 × i8 → i16
        vint16m2_t ab_i16x2 = __riscv_vwmul_vv_i16m2(a_i8x1, b_i8x1, vl);
        // Widening reduction sum: i16 vector -> i32 scalar
        sum_i32x1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_i16x2, sum_i32x1, vl);
    }
    *result = __riscv_vmv_x_s_i32m1_i32(sum_i32x1);
}

/**
 *  @brief  Dot product of two u8 vectors with u32 accumulation on SpacemiT.
 */
NK_PUBLIC void nk_dot_u8_spacemit(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_u32_t *result) {
    vuint32m1_t sum_u32x1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8x1 = __riscv_vle8_v_u8m1(a_scalars, vl);
        vuint8m1_t b_u8x1 = __riscv_vle8_v_u8m1(b_scalars, vl);
        // Widening multiply: u8 * u8 -> u16
        vuint16m2_t ab_u16x2 = __riscv_vwmulu_vv_u16m2(a_u8x1, b_u8x1, vl);
        // Widening reduction sum: u16 vector -> u32 scalar
        sum_u32x1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_u16x2, sum_u32x1, vl);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32x1);
}

/**
 *  @brief  Dot product of two f32 vectors with f64 accumulation on SpacemiT.
 *
 *  Uses widening multiply (f32 × f32 → f64) for precision, then downcasts result to f32.
 *  This matches the Skylake strategy for avoiding catastrophic cancellation.
 */
NK_PUBLIC void nk_dot_f32_spacemit(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    vfloat64m1_t sum_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32x1 = __riscv_vle32_v_f32m1(a_scalars, vl);
        vfloat32m1_t b_f32x1 = __riscv_vle32_v_f32m1(b_scalars, vl);
        // Widening multiply: f32 × f32 → f64
        vfloat64m2_t ab_f64x2 = __riscv_vfwmul_vv_f64m2(a_f32x1, b_f32x1, vl);
        // Ordered reduction sum
        sum_f64x1 = __riscv_vfredusum_vs_f64m2_f64m1(ab_f64x2, sum_f64x1, vl);
    }
    // Downcast f64 result to f32
    *result = (nk_f32_t)__riscv_vfmv_f_s_f64m1_f64(sum_f64x1);
}

/**
 *  @brief  Dot product of two f64 vectors on SpacemiT.
 *
 *  Uses fused multiply-accumulate for efficiency, followed by horizontal reduction.
 */
NK_PUBLIC void nk_dot_f64_spacemit(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f64_t *result) {
    // Accumulate partial sums into vector lanes, then one final horizontal reduction
    size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (size_t vl; count_scalars > 0; count_scalars -= vl, a_scalars += vl, b_scalars += vl) {
        vl = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64x1 = __riscv_vle64_v_f64m1(a_scalars, vl);
        vfloat64m1_t b_f64x1 = __riscv_vle64_v_f64m1(b_scalars, vl);
        // Accumulate a × b into vector lanes
        sum_f64x1 = __riscv_vfmacc_vv_f64m1(sum_f64x1, a_f64x1, b_f64x1, vl);
    }
    // Single horizontal reduction at the end with VLMAX
    vfloat64m1_t zero_f64x1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_f64x1, zero_f64x1, vlmax));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SPACEMIT
#endif // NK_TARGET_RISCV_

#endif // NK_DOT_SPACEMIT_H
