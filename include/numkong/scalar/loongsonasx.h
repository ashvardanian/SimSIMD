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

/** @brief Broadcast f32 scalar into all 4 lanes of a 128-bit register (GCC/Clang portable). */
NK_INTERNAL __m128 nk_xvreplgr2vr_s_128_(float x) {
    nk_fui32_t c;
    c.f = x;
    return (__m128)__lsx_vreplgr2vr_w((int)c.u);
}

/** @brief Broadcast f32 scalar into all 8 lanes of a 256-bit register (GCC/Clang portable). */
NK_INTERNAL __m256 nk_xvfreplgr2vr_s_(float x) {
    nk_fui32_t c;
    c.f = x;
    return (__m256)__lasx_xvreplgr2vr_w((int)c.u);
}

/** @brief Broadcast f64 scalar into all 4 lanes of a 256-bit register (GCC/Clang portable). */
NK_INTERNAL __m256d nk_xvfreplgr2vr_d_(double x) {
    nk_fui64_t c;
    c.f = x;
    return (__m256d)__lasx_xvreplgr2vr_d((long long)c.u);
}

NK_PUBLIC nk_f32_t nk_f32_rsqrt_loongsonasx(nk_f32_t x) {
    __m256 x_f32x8 = nk_xvfreplgr2vr_s_(x);
    __m256 estimate_f32x8 = __lasx_xvfrsqrte_s(x_f32x8);
    // One Newton-Raphson refinement: y' = 0.5 × y × (3 − x × y²)
    __m256 three_f32x8 = nk_xvfreplgr2vr_s_(3.0f);
    __m256 half_f32x8 = nk_xvfreplgr2vr_s_(0.5f);
    __m256 y_sq_f32x8 = __lasx_xvfmul_s(estimate_f32x8, estimate_f32x8);
    __m256 refinement_f32x8 = __lasx_xvfsub_s(three_f32x8, __lasx_xvfmul_s(x_f32x8, y_sq_f32x8));
    __m256 result_f32x8 = __lasx_xvfmul_s(__lasx_xvfmul_s(half_f32x8, estimate_f32x8), refinement_f32x8);
    nk_b256_vec_t vec;
    vec.ymm_ps = result_f32x8;
    return vec.f32s[0];
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_loongsonasx(nk_f32_t x) { return x > 0 ? x * nk_f32_rsqrt_loongsonasx(x) : 0; }

NK_PUBLIC nk_f64_t nk_f64_sqrt_loongsonasx(nk_f64_t x) {
    __m256d x_f64x4 = nk_xvfreplgr2vr_d_(x);
    __m256d result_f64x4 = __lasx_xvfsqrt_d(x_f64x4);
    nk_b256_vec_t vec;
    vec.ymm_pd = result_f64x4;
    return vec.f64s[0];
}

NK_PUBLIC nk_f64_t nk_f64_rsqrt_loongsonasx(nk_f64_t x) { return 1.0 / nk_f64_sqrt_loongsonasx(x); }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH_
#endif // NK_SCALAR_LOONGSONASX_H
