/**
 *  @brief SIMD-accelerated Type Conversions and Load/Store Helpers for LoongArch LASX (256-bit).
 *  @file include/numkong/cast/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  @section loongsonasx_cast_instructions Key LASX Load/Store Instructions
 *
 *      Intrinsic                      Instruction       Description
 *      __lasx_xvld(ptr, 0)            XVLD              256-bit aligned/unaligned load
 *      __lasx_xvst(v, ptr, 0)         XVST              256-bit aligned/unaligned store
 *      __lasx_xvreplgr2vr_w(bits)     XVREPLGR2VR.W     Broadcast i32 to 8 lanes
 *      __lasx_xvreplgr2vr_d(bits)     XVREPLGR2VR.D     Broadcast i64 to 4 lanes
 *      __lasx_xvffint_s_w(v)          XVFFINT.S.W       4x i32 -> f32 (per 128-bit lane)
 *      __lasx_xvfrsqrt_s(v)           XVFRSQRT.S        f32 full-precision reciprocal sqrt
 *      __lasx_xvfsqrt_s(v)            XVFSQRT.S         f32 full-precision sqrt
 *      __lasx_xvfsqrt_d(v)            XVFSQRT.D         f64 full-precision sqrt
 *
 *  LASX is a 256-bit extension; all vector registers are 256-bit `__m256i`. For 128-bit
 *  `nk_b128_vec_t` operations, `__lasx_xvld` safely loads into the low 128 bits (the high
 *  128 bits are zeroed or undefined depending on context). For 128-bit stores we use `memcpy`
 *  to avoid writing beyond the intended 16 bytes. Partial loads/stores delegate to serial
 *  helpers since LASX lacks masked load/store instructions.
 */
#ifndef NK_CAST_LOONGSONASX_H
#define NK_CAST_LOONGSONASX_H

#if NK_TARGET_LOONGARCH_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/cast/serial.h"        // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x4_serial_`
#include "numkong/scalar/loongsonasx.h" // `nk_xvreplgr2vr_s_128_`, `nk_xvfreplgr2vr_s_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 256-bit full load (LASX). */
NK_INTERNAL void nk_load_b256_loongsonasx_(void const *src, nk_b256_vec_t *dst) { dst->ymm = __lasx_xvld(src, 0); }

/** @brief Type-agnostic 256-bit full store (LASX). */
NK_INTERNAL void nk_store_b256_loongsonasx_(nk_b256_vec_t const *src, void *dst) { __lasx_xvst(src->ymm, dst, 0); }

/** @brief Type-agnostic 128-bit full load (LSX subset of LASX). */
NK_INTERNAL void nk_load_b128_loongsonasx_(void const *src, nk_b128_vec_t *dst) { dst->xmm = __lsx_vld(src, 0); }

/** @brief Type-agnostic 128-bit full store (LSX subset of LASX). */
NK_INTERNAL void nk_store_b128_loongsonasx_(nk_b128_vec_t const *src, void *dst) { __lsx_vst(src->xmm, dst, 0); }

/** @brief Convert 8 × bf16 → 8 × f32 by zero-extending u16 → u32 and shifting left 16 (LASX). */
NK_INTERNAL __m256i nk_bf16x8_to_f32x8_loongsonasx_(__m128i bf16_i16x8) {
    // Zero-extend low 8 × u16 → 8 × u32, then shift left 16 to place bf16 mantissa in f32 position
    __m256i extended_u32x8 = __lasx_xvsllwil_wu_hu(__lasx_cast_128(bf16_i16x8), 0);
    return __lasx_xvslli_w(extended_u32x8, 16);
}

/** @brief Load 8 × bf16 from memory, convert to 8 × f32, store in 256-bit vector (LASX). */
NK_INTERNAL void nk_load_bf16x8_to_f32x8_loongsonasx_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_bf16x8_to_f32x8_loongsonasx_(__lsx_vld(src, 0));
}

/** @brief Partial load for bf16 elements (up to 8) with conversion to f32 (LASX). */
NK_INTERNAL void nk_partial_load_bf16x8_to_f32x8_loongsonasx_(nk_bf16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm = nk_bf16x8_to_f32x8_loongsonasx_(vec.xmm);
}

/**
 *  @brief Convert 8 × f16 → 8 × f32 via Giesen's magic-number multiply trick (LASX, branchless).
 *  @see https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 *
 *  Shifts the 15-bit magnitude into F32 position, multiplies by 2^112 (0x77800000)
 *  to rebias the exponent and normalize subnormals in one step. Inf/NaN (exp=31)
 *  overflows the multiply and is fixed with a comparison + blend.
 */
NK_INTERNAL __m256i nk_f16x8_to_f32x8_loongsonasx_(__m128i f16_i16x8) {
    __m256i raw_u32x8 = __lasx_xvsllwil_wu_hu(__lasx_cast_128(f16_i16x8), 0);

    // Extract sign and unsigned magnitude
    __m256i sign_u32x8 = __lasx_xvand_v(raw_u32x8, __lasx_xvreplgr2vr_w(0x8000));
    __m256i sign_f32_u32x8 = __lasx_xvslli_w(sign_u32x8, 16);
    __m256i magnitude_u32x8 = __lasx_xvand_v(raw_u32x8, __lasx_xvreplgr2vr_w(0x7FFF));

    // Shift mantissa+exponent into F32 position and multiply by magic 2^112
    __m256i shifted_u32x8 = __lasx_xvslli_w(magnitude_u32x8, 13);
    __m256i magic_u32x8 = __lasx_xvreplgr2vr_w(0x77800000);
    __m256i rebiased_u32x8 = (__m256i)__lasx_xvfmul_s((__m256)shifted_u32x8, (__m256)magic_u32x8);

    // Fix inf/NaN: exp=31 overflows the multiply, detect and apply direct rebias
    __m256i infnan_threshold_u32x8 = __lasx_xvreplgr2vr_w(0x38800000);
    __m256i infnan_mask_u32x8 = __lasx_xvsle_wu(infnan_threshold_u32x8, shifted_u32x8);
    __m256i direct_u32x8 = __lasx_xvor_v(shifted_u32x8, __lasx_xvreplgr2vr_w(0x70000000));
    __m256i result_u32x8 = __lasx_xvbitsel_v(rebiased_u32x8, direct_u32x8, infnan_mask_u32x8);

    // Apply sign
    return __lasx_xvor_v(result_u32x8, sign_f32_u32x8);
}

/** @brief Load 8 × f16 from memory, convert to 8 × f32 via Giesen's trick (LASX). */
NK_INTERNAL void nk_load_f16x8_to_f32x8_loongsonasx_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(__lsx_vld(src, 0));
}

/** @brief Partial load for f16 elements (up to 8) with conversion to f32 (LASX). */
NK_INTERNAL void nk_partial_load_f16x8_to_f32x8_loongsonasx_(nk_f16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(vec.xmm);
}

#pragma endregion - Type Punned Loads and Stores

#pragma region - Vectorized From-Dot Helpers

/** @brief Safe square root of 8 floats with zero-clamping for numerical stability (LASX 256-bit). */
NK_INTERNAL __m256 nk_sqrt_f32x8_loongsonasx_(__m256 x_f32x8) {
    __m256 zero_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    return __lasx_xvfsqrt_s(__lasx_xvfmax_s(x_f32x8, zero_f32x8));
}

/** @brief Safe square root of 4 floats with zero-clamping for numerical stability (LSX 128-bit). */
NK_INTERNAL __m128 nk_sqrt_f32x4_loongsonasx_(__m128 x_f32x4) {
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    return __lsx_vfsqrt_s(__lsx_vfmax_s(x_f32x4, zero_f32x4));
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs (LSX 128-bit f32). */
NK_INTERNAL void nk_angular_through_f32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                              nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = dots.xmm_ps;
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_(query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, target_sumsqs.xmm_ps);
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    results->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs (LSX 128-bit f32). */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                                nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = dots.xmm_ps;
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_(query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, target_sumsqs.xmm_ps);
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    // dist_sq = sum_sq − 2 × dots = -(2 × dots − sum_sq)
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    results->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

/** @brief Angular from_dot for native f64: 1 − dot / √(query_sumsq × target_sumsq) for 4 pairs (LASX 256-bit). */
NK_INTERNAL void nk_angular_through_f64_from_dot_loongsonasx_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                              nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    __m256d dots_f64x4 = dots.ymm_pd;
    __m256d query_sumsq_f64x4 = nk_xvfreplgr2vr_d_(query_sumsq);
    __m256d products_f64x4 = __lasx_xvfmul_d(query_sumsq_f64x4, target_sumsqs.ymm_pd);
    __m256d sqrt_products_f64x4 = __lasx_xvfsqrt_d(products_f64x4);
    __m256d normalized_f64x4 = __lasx_xvfdiv_d(dots_f64x4, sqrt_products_f64x4);
    __m256d one_f64x4 = nk_xvfreplgr2vr_d_(1.0);
    __m256d angular_f64x4 = __lasx_xvfsub_d(one_f64x4, normalized_f64x4);
    __m256d zero_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    results->ymm_pd = __lasx_xvfmax_d(angular_f64x4, zero_f64x4);
}

/** @brief Euclidean from_dot for native f64: √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs (LASX 256-bit). */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_loongsonasx_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                                nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    __m256d dots_f64x4 = dots.ymm_pd;
    __m256d query_sumsq_f64x4 = nk_xvfreplgr2vr_d_(query_sumsq);
    __m256d sum_sq_f64x4 = __lasx_xvfadd_d(query_sumsq_f64x4, target_sumsqs.ymm_pd);
    __m256d two_f64x4 = nk_xvfreplgr2vr_d_(2.0);
    // dist_sq = sum_sq − 2 × dots = -(2 × dots − sum_sq)
    __m256d dist_sq_f64x4 = __lasx_xvfnmsub_d(two_f64x4, dots_f64x4, sum_sq_f64x4);
    __m256d zero_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    results->ymm_pd = __lasx_xvfsqrt_d(__lasx_xvfmax_d(dist_sq_f64x4, zero_f64x4));
}

/** @brief Angular from_dot for i32 accumulators: cast i32 → f32, rsqrt+NR, clamp. 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_angular_through_i32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                              nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots.xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs.xmm));
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    results->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot for i32 accumulators: cast i32 → f32, then √(a² + b² − 2ab). 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                                nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots.xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs.xmm));
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    results->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast u32 → f32, rsqrt+NR, clamp. 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_angular_through_u32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots.xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs.xmm));
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    results->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot for u32 accumulators: cast u32 → f32, then √(a² + b² − 2ab). 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_loongsonasx_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                                nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots.xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs.xmm));
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    results->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

#pragma endregion - Vectorized From - Dot Helpers

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH_
#endif // NK_CAST_LOONGSONASX_H
