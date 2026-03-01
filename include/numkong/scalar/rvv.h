/**
 *  @brief SIMD-accelerated Scalar Math Helpers for RISC-V.
 *  @file include/numkong/scalar/rvv.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  RVV scalar helpers use vector instructions with VL=1 for hardware sqrt/rsqrt.
 *  `vfrsqrt7` provides 7-bit mantissa precision; Newton-Raphson refines to full precision.
 */
#ifndef NK_SCALAR_RVV_H
#define NK_SCALAR_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC nk_f32_t nk_f32_rsqrt_rvv(nk_f32_t number) {
    vfloat32m1_t x = __riscv_vfmv_s_f_f32m1(number, 1);
    vfloat32m1_t y = __riscv_vfrsqrt7_v_f32m1(x, 1);
    vfloat32m1_t half = __riscv_vfmv_s_f_f32m1(0.5f, 1);
    vfloat32m1_t three_half = __riscv_vfmv_s_f_f32m1(1.5f, 1);
    vfloat32m1_t half_x = __riscv_vfmul_vv_f32m1(half, x, 1);
    // Iteration 1
    vfloat32m1_t y_sq = __riscv_vfmul_vv_f32m1(y, y, 1);
    vfloat32m1_t half_x_y_sq = __riscv_vfmul_vv_f32m1(half_x, y_sq, 1);
    vfloat32m1_t factor = __riscv_vfsub_vv_f32m1(three_half, half_x_y_sq, 1);
    y = __riscv_vfmul_vv_f32m1(y, factor, 1);
    // Iteration 2
    y_sq = __riscv_vfmul_vv_f32m1(y, y, 1);
    half_x_y_sq = __riscv_vfmul_vv_f32m1(half_x, y_sq, 1);
    factor = __riscv_vfsub_vv_f32m1(three_half, half_x_y_sq, 1);
    y = __riscv_vfmul_vv_f32m1(y, factor, 1);
    return __riscv_vfmv_f_s_f32m1_f32(y);
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_rvv(nk_f32_t number) {
    vfloat32m1_t x = __riscv_vfmv_s_f_f32m1(number, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfsqrt_v_f32m1(x, 1));
}

NK_PUBLIC nk_f64_t nk_f64_rsqrt_rvv(nk_f64_t number) {
    vfloat64m1_t x = __riscv_vfmv_s_f_f64m1(number, 1);
    vfloat64m1_t y = __riscv_vfrsqrt7_v_f64m1(x, 1);
    vfloat64m1_t half = __riscv_vfmv_s_f_f64m1(0.5, 1);
    vfloat64m1_t three_half = __riscv_vfmv_s_f_f64m1(1.5, 1);
    vfloat64m1_t half_x = __riscv_vfmul_vv_f64m1(half, x, 1);
    // Iteration 1
    vfloat64m1_t y_sq = __riscv_vfmul_vv_f64m1(y, y, 1);
    vfloat64m1_t half_x_y_sq = __riscv_vfmul_vv_f64m1(half_x, y_sq, 1);
    vfloat64m1_t factor = __riscv_vfsub_vv_f64m1(three_half, half_x_y_sq, 1);
    y = __riscv_vfmul_vv_f64m1(y, factor, 1);
    // Iteration 2
    y_sq = __riscv_vfmul_vv_f64m1(y, y, 1);
    half_x_y_sq = __riscv_vfmul_vv_f64m1(half_x, y_sq, 1);
    factor = __riscv_vfsub_vv_f64m1(three_half, half_x_y_sq, 1);
    y = __riscv_vfmul_vv_f64m1(y, factor, 1);
    // Iteration 3
    y_sq = __riscv_vfmul_vv_f64m1(y, y, 1);
    half_x_y_sq = __riscv_vfmul_vv_f64m1(half_x, y_sq, 1);
    factor = __riscv_vfsub_vv_f64m1(three_half, half_x_y_sq, 1);
    y = __riscv_vfmul_vv_f64m1(y, factor, 1);
    return __riscv_vfmv_f_s_f64m1_f64(y);
}

NK_PUBLIC nk_f64_t nk_f64_sqrt_rvv(nk_f64_t number) {
    vfloat64m1_t x = __riscv_vfmv_s_f_f64m1(number, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfsqrt_v_f64m1(x, 1));
}

NK_PUBLIC nk_f32_t nk_f32_fma_rvv(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    vfloat32m1_t va = __riscv_vfmv_s_f_f32m1(a, 1);
    vfloat32m1_t vc = __riscv_vfmv_s_f_f32m1(c, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfmacc_vf_f32m1(vc, b, va, 1));
}

NK_PUBLIC nk_f64_t nk_f64_fma_rvv(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    vfloat64m1_t va = __riscv_vfmv_s_f_f64m1(a, 1);
    vfloat64m1_t vc = __riscv_vfmv_s_f_f64m1(c, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfmacc_vf_f64m1(vc, b, va, 1));
}

NK_PUBLIC nk_u8_t nk_u8_saturating_add_rvv(nk_u8_t a, nk_u8_t b) {
    vuint8m1_t va = __riscv_vmv_v_x_u8m1(a, 1);
    vuint8m1_t vb = __riscv_vmv_v_x_u8m1(b, 1);
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vsaddu_vv_u8m1(va, vb, 1));
}

NK_PUBLIC nk_i8_t nk_i8_saturating_add_rvv(nk_i8_t a, nk_i8_t b) {
    vint8m1_t va = __riscv_vmv_v_x_i8m1(a, 1);
    vint8m1_t vb = __riscv_vmv_v_x_i8m1(b, 1);
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vsadd_vv_i8m1(va, vb, 1));
}

NK_PUBLIC nk_u16_t nk_u16_saturating_add_rvv(nk_u16_t a, nk_u16_t b) {
    vuint16m1_t va = __riscv_vmv_v_x_u16m1(a, 1);
    vuint16m1_t vb = __riscv_vmv_v_x_u16m1(b, 1);
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vsaddu_vv_u16m1(va, vb, 1));
}

NK_PUBLIC nk_i16_t nk_i16_saturating_add_rvv(nk_i16_t a, nk_i16_t b) {
    vint16m1_t va = __riscv_vmv_v_x_i16m1(a, 1);
    vint16m1_t vb = __riscv_vmv_v_x_i16m1(b, 1);
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vsadd_vv_i16m1(va, vb, 1));
}

NK_PUBLIC nk_u32_t nk_u32_saturating_add_rvv(nk_u32_t a, nk_u32_t b) {
    vuint32m1_t va = __riscv_vmv_v_x_u32m1(a, 1);
    vuint32m1_t vb = __riscv_vmv_v_x_u32m1(b, 1);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vsaddu_vv_u32m1(va, vb, 1));
}

NK_PUBLIC nk_i32_t nk_i32_saturating_add_rvv(nk_i32_t a, nk_i32_t b) {
    vint32m1_t va = __riscv_vmv_v_x_i32m1(a, 1);
    vint32m1_t vb = __riscv_vmv_v_x_i32m1(b, 1);
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vsadd_vv_i32m1(va, vb, 1));
}

NK_PUBLIC nk_u64_t nk_u64_saturating_add_rvv(nk_u64_t a, nk_u64_t b) {
    vuint64m1_t va = __riscv_vmv_v_x_u64m1(a, 1);
    vuint64m1_t vb = __riscv_vmv_v_x_u64m1(b, 1);
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vsaddu_vv_u64m1(va, vb, 1));
}

NK_PUBLIC nk_i64_t nk_i64_saturating_add_rvv(nk_i64_t a, nk_i64_t b) {
    vint64m1_t va = __riscv_vmv_v_x_i64m1(a, 1);
    vint64m1_t vb = __riscv_vmv_v_x_i64m1(b, 1);
    return __riscv_vmv_x_s_i64m1_i64(__riscv_vsadd_vv_i64m1(va, vb, 1));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_SCALAR_RVV_H
