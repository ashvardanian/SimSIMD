/**
 *  @brief SIMD-accelerated Scalar Math Helpers for LoongArch LASX.
 *  @file include/numkong/scalar/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  LASX provides `xvfrsqrte` (reciprocal sqrt estimate, ~14 bits precision) and
 *  `xvfsqrt` (full-precision sqrt). The rsqrt estimate with one Newton-Raphson
 *  iteration gives ~28 bits for f32 — sufficient for angular normalization.
 *  Full-precision sqrt uses the hardware `xvfsqrt` instruction.
 *  Broadcast via `xvreplgr2vr`, extract via `xvpickve2gr` — no memory round-trips.
 */
#ifndef NK_SCALAR_LOONGSONASX_H
#define NK_SCALAR_LOONGSONASX_H

#if NK_TARGET_LOONGARCH_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC nk_f32_t nk_f32_rsqrt_loongsonasx(nk_f32_t x) {
    nk_fui32_t conv;
    conv.f = x;
    __m256i x_f32x8 = __lasx_xvreplgr2vr_w((int)conv.u);
    __m256i estimate_f32x8 = __lasx_xvfrsqrte_s(x_f32x8);
    // One Newton-Raphson refinement: y' = 0.5 × y × (3 − x × y²)
    __m256i three_f32x8 = __lasx_xvfreplgr2vr_s(3.0f);
    __m256i half_f32x8 = __lasx_xvfreplgr2vr_s(0.5f);
    __m256i y_sq_f32x8 = __lasx_xvfmul_s(estimate_f32x8, estimate_f32x8);
    __m256i refinement_f32x8 = __lasx_xvfsub_s(three_f32x8, __lasx_xvfmul_s(x_f32x8, y_sq_f32x8));
    __m256i result_f32x8 = __lasx_xvfmul_s(__lasx_xvfmul_s(half_f32x8, estimate_f32x8), refinement_f32x8);
    conv.u = (nk_u32_t)__lasx_xvpickve2gr_w(result_f32x8, 0);
    return conv.f;
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_loongsonasx(nk_f32_t x) { return x > 0 ? x * nk_f32_rsqrt_loongsonasx(x) : 0; }

NK_PUBLIC nk_f64_t nk_f64_sqrt_loongsonasx(nk_f64_t x) {
    nk_fui64_t conv;
    conv.f = x;
    __m256i v_f64x4 = __lasx_xvreplgr2vr_d((long long)conv.u);
    __m256i result_f64x4 = __lasx_xvfsqrt_d(v_f64x4);
    conv.u = (nk_u64_t)__lasx_xvpickve2gr_d(result_f64x4, 0);
    return conv.f;
}

NK_PUBLIC nk_f64_t nk_f64_rsqrt_loongsonasx(nk_f64_t x) { return 1.0 / nk_f64_sqrt_loongsonasx(x); }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH_
#endif // NK_SCALAR_LOONGSONASX_H
