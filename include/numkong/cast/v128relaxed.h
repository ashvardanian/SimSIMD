/**
 *  @brief SIMD-accelerated Type Conversions for WASM.
 *  @file include/numkong/cast/v128relaxed.h
 */

#ifndef NK_CAST_V128RELAXED_H
#define NK_CAST_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** @brief Native WASM SIMD 128-bit load. */
NK_INTERNAL void nk_load_b128_v128relaxed_(void const *src, nk_b128_vec_t *dst) { dst->v128 = wasm_v128_load(src); }
/** @brief Native WASM SIMD 256-bit load using two v128 loads. */
NK_INTERNAL void nk_load_b256_v128relaxed_(void const *src, nk_b256_vec_t *dst) {
    dst->v128s[0] = wasm_v128_load(src);
    dst->v128s[1] = wasm_v128_load((char const *)src + 16);
}
/** @brief Native WASM SIMD 128-bit store. */
NK_INTERNAL void nk_store_b128_v128relaxed_(nk_b128_vec_t const *src, void *dst) { wasm_v128_store(dst, src->v128); }
/** @brief Native WASM SIMD 256-bit store using two v128 stores. */
NK_INTERNAL void nk_store_b256_v128relaxed_(nk_b256_vec_t const *src, void *dst) {
    wasm_v128_store(dst, src->v128s[0]);
    wasm_v128_store((char *)dst + 16, src->v128s[1]);
}

/** @brief BF16 is the upper 16 bits of F32, so zero-extend to u32 and shift left by 16. */
NK_INTERNAL nk_b128_vec_t nk_bf16x4_to_f32x4_v128relaxed_(nk_b64_vec_t bf16_vec) {
    v128_t bf16_u16x4_in_u64 = wasm_i64x2_splat(bf16_vec.u64);
    v128_t bf16_u32x4_low = wasm_u32x4_extend_low_u16x8(bf16_u16x4_in_u64);
    nk_b128_vec_t result;
    result.v128 = wasm_i32x4_shl(bf16_u32x4_low, 16);
    return result;
}

/**
 *  @brief F16→F32 via Giesen's magic-number multiply trick.
 *  @see https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 *
 *  Shifts the 15-bit magnitude into F32 exponent+mantissa position, then multiplies
 *  by 2^112 (magic = 0x77800000) to rebias the exponent. This single multiply also
 *  correctly normalizes F16 subnormals into F32 normals — no branching or FPU
 *  integer-to-float conversion needed. Inf/NaN (exp=31) overflows the multiply and
 *  is fixed with a comparison + blend.
 */
NK_INTERNAL nk_b128_vec_t nk_f16x4_to_f32x4_v128relaxed_(nk_b64_vec_t f16_vec) {
    v128_t raw_u16x4_in_u64 = wasm_i64x2_splat(f16_vec.u64);
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(raw_u16x4_in_u64);

    // Extract sign and unsigned magnitude
    v128_t sign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x8000));
    v128_t sign_f32_u32x4 = wasm_i32x4_shl(sign_u32x4, 16);
    v128_t magnitude_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x7FFF));

    // Shift mantissa+exponent into F32 position and multiply by magic 2^112
    v128_t shifted_u32x4 = wasm_i32x4_shl(magnitude_u32x4, 13);
    v128_t magic_f32x4 = wasm_i32x4_splat(0x77800000);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)magic_f32x4);

    // Fix inf/NaN: exp=31 after shift becomes 0x1F<<13 = 0x000F8000, ×2^112 overflows.
    // Detect via threshold on shifted magnitude and apply direct rebias instead.
    v128_t infnan_threshold_u32x4 = wasm_i32x4_splat(0x38800000);
    v128_t infnan_mask_u32x4 = wasm_u32x4_ge(shifted_u32x4, infnan_threshold_u32x4);
    v128_t direct_u32x4 = wasm_v128_or(shifted_u32x4, wasm_i32x4_splat(0x70000000));
    v128_t result_u32x4 = wasm_i32x4_relaxed_laneselect(direct_u32x4, rebiased_f32x4, infnan_mask_u32x4);

    // Apply sign
    result_u32x4 = wasm_v128_or(result_u32x4, sign_f32_u32x4);

    nk_b128_vec_t result;
    result.v128 = result_u32x4;
    return result;
}

/**
 *  @brief E4M3→F32: 4-bit exponent (bias=7→127, delta=120), 3-bit mantissa (shift by 20).
 *  Subnormal via FPU: mant * (1/512) = mant * 2^-9.  NaN only at exp=15,mant=7.
 */
NK_INTERNAL nk_b128_vec_t nk_e4m3x4_to_f32x4_v128relaxed_(nk_b32_vec_t e4m3_vec) {
    v128_t e4m3_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e4m3_vec.u32)));
    v128_t exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(e4m3_u32x4, 3), wasm_i32x4_splat(0x0F));
    v128_t mant_u32x4 = wasm_v128_and(e4m3_u32x4, wasm_i32x4_splat(0x07));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_u32x4_shr(e4m3_u32x4, 7), 31);
    v128_t f32_exp_u32x4 = wasm_i32x4_shl(wasm_i32x4_add(exp_u32x4, wasm_i32x4_splat(120)), 23);
    v128_t f32_mant_u32x4 = wasm_i32x4_shl(mant_u32x4, 20);
    v128_t normal_bits_u32x4 = wasm_v128_or(sign_u32x4, wasm_v128_or(f32_exp_u32x4, f32_mant_u32x4));
    v128_t subnorm_abs_f32x4 = wasm_f32x4_mul(wasm_f32x4_convert_u32x4(mant_u32x4), wasm_f32x4_splat(1.0f / 512.0f));
    v128_t subnorm_f32x4 = wasm_v128_or(subnorm_abs_f32x4, sign_u32x4);
    v128_t exp_zero_mask = wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(0));
    v128_t is_nan_mask = wasm_v128_and(wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(15)),
                                       wasm_i32x4_eq(mant_u32x4, wasm_i32x4_splat(7)));
    v128_t exceptional_mask = wasm_v128_or(exp_zero_mask, is_nan_mask);
    if (!wasm_v128_any_true(exceptional_mask)) {
        nk_b128_vec_t result;
        result.v128 = normal_bits_u32x4;
        return result;
    }
    v128_t result_u32x4 = wasm_i32x4_relaxed_laneselect(subnorm_f32x4, normal_bits_u32x4, exp_zero_mask);
    if (wasm_v128_any_true(is_nan_mask)) {
        v128_t nan_bits = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7FC00000));
        result_u32x4 = wasm_i32x4_relaxed_laneselect(nan_bits, result_u32x4, is_nan_mask);
    }
    nk_b128_vec_t result;
    result.v128 = result_u32x4;
    return result;
}

/**
 *  @brief E5M2→F32: same exponent encoding as F16 (5-bit, bias=15, delta=112), 2-bit mantissa (shift by 21).
 *  Subnormal via FPU: mant * (1/65536) = mant * 2^-16.  Inf at exp=31,mant=0; NaN otherwise.
 */
NK_INTERNAL nk_b128_vec_t nk_e5m2x4_to_f32x4_v128relaxed_(nk_b32_vec_t e5m2_vec) {
    v128_t e5m2_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e5m2_vec.u32)));
    v128_t exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(e5m2_u32x4, 2), wasm_i32x4_splat(0x1F));
    v128_t mant_u32x4 = wasm_v128_and(e5m2_u32x4, wasm_i32x4_splat(0x03));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_u32x4_shr(e5m2_u32x4, 7), 31);
    v128_t f32_exp_u32x4 = wasm_i32x4_shl(wasm_i32x4_add(exp_u32x4, wasm_i32x4_splat(112)), 23);
    v128_t f32_mant_u32x4 = wasm_i32x4_shl(mant_u32x4, 21);
    v128_t normal_bits_u32x4 = wasm_v128_or(sign_u32x4, wasm_v128_or(f32_exp_u32x4, f32_mant_u32x4));
    v128_t subnorm_abs_f32x4 = wasm_f32x4_mul(wasm_f32x4_convert_u32x4(mant_u32x4), wasm_f32x4_splat(1.0f / 65536.0f));
    v128_t subnorm_f32x4 = wasm_v128_or(subnorm_abs_f32x4, sign_u32x4);
    v128_t exp_zero_mask = wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(0));
    v128_t exp_max_mask = wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(31));
    v128_t exceptional_mask = wasm_v128_or(exp_zero_mask, exp_max_mask);
    if (!wasm_v128_any_true(exceptional_mask)) {
        nk_b128_vec_t result;
        result.v128 = normal_bits_u32x4;
        return result;
    }
    v128_t result_u32x4 = wasm_i32x4_relaxed_laneselect(subnorm_f32x4, normal_bits_u32x4, exp_zero_mask);
    v128_t mant_zero_mask = wasm_i32x4_eq(mant_u32x4, wasm_i32x4_splat(0));
    v128_t inf_bits_u32x4 = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7F800000));
    v128_t nan_bits_u32x4 = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7FC00000));
    v128_t special_bits_u32x4 = wasm_i32x4_relaxed_laneselect(inf_bits_u32x4, nan_bits_u32x4, mant_zero_mask);
    result_u32x4 = wasm_i32x4_relaxed_laneselect(special_bits_u32x4, result_u32x4, exp_max_mask);
    nk_b128_vec_t result;
    result.v128 = result_u32x4;
    return result;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_CAST_V128RELAXED_H
