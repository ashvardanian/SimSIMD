/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_H
#define NK_REDUCE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // Serial fallbacks

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 128-bit full load (NEON). */
NK_INTERNAL void nk_load_b128_neon_(void const *src, nk_b128_vec_t *dst) {
    dst->u8x16 = vld1q_u8((nk_u8_t const *)src);
}

/** @brief Type-agnostic 64-bit full load (NEON). */
NK_INTERNAL void nk_load_b64_neon_(void const *src, nk_b64_vec_t *dst) { dst->u8x8 = vld1_u8((nk_u8_t const *)src); }

#pragma endregion - Type Punned Loads and Stores

#pragma region - Vectorized Conversions

/** @brief Convert 4× e4m3 → f32x4 via bit manipulation (NEON).
 *  E4M3 format: S EEEE MMM (bias=7). F32: sign<<31, (exp+120)<<23, mant<<20. */
NK_INTERNAL float32x4_t nk_e4m3x4_to_f32x4_neon_(nk_e4m3_t const *src) {
    uint8x8_t e4m3_u8x8 = vcreate_u8(*(uint32_t const *)src);
    uint16x8_t e4m3_u16x8 = vmovl_u8(e4m3_u8x8);
    uint32x4_t v_u32x4 = vmovl_u16(vget_low_u16(e4m3_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vshrq_n_u32(vandq_u32(v_u32x4, vdupq_n_u32(0x80)), 7), 31);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(v_u32x4, 3), vdupq_n_u32(0x0F));
    uint32x4_t mant_u32x4 = vandq_u32(v_u32x4, vdupq_n_u32(0x07));
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(120)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 20);
    uint32x4_t f32_bits_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));
    uint32x4_t zero_mask_u32x4 = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    f32_bits_u32x4 = vbicq_u32(f32_bits_u32x4, zero_mask_u32x4);
    return vreinterpretq_f32_u32(f32_bits_u32x4);
}

/** @brief Convert 4× e5m2 → f32x4 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). F32: sign<<31, (exp+112)<<23, mant<<21. */
NK_INTERNAL float32x4_t nk_e5m2x4_to_f32x4_neon_(nk_e5m2_t const *src) {
    uint8x8_t e5m2_u8x8 = vcreate_u8(*(uint32_t const *)src);
    uint16x8_t e5m2_u16x8 = vmovl_u8(e5m2_u8x8);
    uint32x4_t v_u32x4 = vmovl_u16(vget_low_u16(e5m2_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vshrq_n_u32(vandq_u32(v_u32x4, vdupq_n_u32(0x80)), 7), 31);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(v_u32x4, 2), vdupq_n_u32(0x1F));
    uint32x4_t mant_u32x4 = vandq_u32(v_u32x4, vdupq_n_u32(0x03));
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(112)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 21);
    uint32x4_t f32_bits_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));
    uint32x4_t zero_mask_u32x4 = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    f32_bits_u32x4 = vbicq_u32(f32_bits_u32x4, zero_mask_u32x4);
    return vreinterpretq_f32_u32(f32_bits_u32x4);
}

/** @brief Convert 8× e4m3 → f16x8 via bit manipulation (NEON).
 *  E4M3 format: S EEEE MMM (bias=7). F16: sign<<15, (exp+8)<<10, mant<<7. */
NK_INTERNAL float16x8_t nk_e4m3x8_to_f16x8_neon_(uint8x8_t e4m3_u8x8) {
    uint16x8_t v_u16x8 = vmovl_u8(e4m3_u8x8);
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 3), vdupq_n_u16(0x0F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x07));
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(8)), 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 7);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);
    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

/** @brief Convert 8× e5m2 → f16x8 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). F16: sign<<15, exp<<10, mant<<8. */
NK_INTERNAL float16x8_t nk_e5m2x8_to_f16x8_neon_(uint8x8_t e5m2_u8x8) {
    uint16x8_t v_u16x8 = vmovl_u8(e5m2_u8x8);
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 2), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x03));
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(exp_u16x8, 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 8);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);
    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

/** @brief Convert f16x8 → 8× e4m3 (NEON). */
NK_INTERNAL uint8x8_t nk_f16x8_to_e4m3x8_neon_(float16x8_t f16x8) {
    uint16x8_t bits_u16x8 = vreinterpretq_u16_f16(f16x8);
    uint16x8_t sign_u16x8 = vshrq_n_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x8000)), 8);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(bits_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vshrq_n_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x03FF)), 7);
    int16x8_t exp_rebias_i16x8 = vsubq_s16(vreinterpretq_s16_u16(exp_u16x8), vdupq_n_s16(8));
    exp_rebias_i16x8 = vmaxq_s16(exp_rebias_i16x8, vdupq_n_s16(0));
    exp_rebias_i16x8 = vminq_s16(exp_rebias_i16x8, vdupq_n_s16(15));
    uint16x8_t e4m3_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(vshlq_n_u16(vreinterpretq_u16_s16(exp_rebias_i16x8), 3),
                                                            vandq_u16(mant_u16x8, vdupq_n_u16(0x07))));
    return vmovn_u16(e4m3_u16x8);
}

/** @brief Convert f16x8 → 8× e5m2 (NEON). */
NK_INTERNAL uint8x8_t nk_f16x8_to_e5m2x8_neon_(float16x8_t f16x8) {
    uint16x8_t bits_u16x8 = vreinterpretq_u16_f16(f16x8);
    uint16x8_t sign_u16x8 = vshrq_n_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x8000)), 8);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(bits_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vshrq_n_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x03FF)), 8);
    uint16x8_t e5m2_u16x8 = vorrq_u16(sign_u16x8,
                                      vorrq_u16(vshlq_n_u16(exp_u16x8, 2), vandq_u16(mant_u16x8, vdupq_n_u16(0x03))));
    return vmovn_u16(e5m2_u16x8);
}

#pragma endregion - Vectorized Conversions

#pragma region - Converting Loads and Stores

/** @brief Partial load for E4M3 elements (up to 4) with expansion to f32x4 (NEON). */
NK_INTERNAL float32x4_t nk_partial_load_e4m3x4_to_f32x4_neon_(nk_e4m3_t const *src, nk_size_t n) {
    (void)src;
    (void)n;
    return vdupq_n_f32(0); // TODO: implement e4m3 partial load for NEON
}

/** @brief Partial load for E5M2 elements (up to 4) with expansion to f32x4 (NEON). */
NK_INTERNAL float32x4_t nk_partial_load_e5m2x4_to_f32x4_neon_(nk_e5m2_t const *src, nk_size_t n) {
    (void)src;
    (void)n;
    return vdupq_n_f32(0); // TODO: implement e5m2 partial load for NEON
}

#pragma endregion - Converting Loads and Stores

#pragma region - Scalar Conversions

/** @brief Convert f16 to f32 scalar using NEON vector conversion. */
NK_PUBLIC void nk_f16_to_f32_neon(nk_f16_t const *src, nk_f32_t *dest) {
    float16x4_t f16vec = vld1_dup_f16((nk_f16_for_arm_simd_t const *)src);
    float32x4_t f32vec = vcvt_f32_f16(f16vec);
    *dest = vgetq_lane_f32(f32vec, 0);
}

/** @brief Convert f32 to f16 scalar using NEON vector conversion. */
NK_PUBLIC void nk_f32_to_f16_neon(nk_f32_t const *src, nk_f16_t *dest) {
    float32x4_t f32vec = vdupq_n_f32(*src);
    float16x4_t f16vec = vcvt_f16_f32(f32vec);
    vst1_lane_f16((nk_f16_for_arm_simd_t *)dest, f16vec, 0);
}

#pragma endregion - Scalar Conversions

#pragma region - Public API

NK_PUBLIC void nk_cast_neon(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    return nk_cast_serial(from, from_type, n, to, to_type);
}

#pragma endregion - Public API

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEON_H