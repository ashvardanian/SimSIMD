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
    v128_t bf16_u16x4_in_u64 = wasm_v128_load64_zero(&bf16_vec.u64);
    v128_t bf16_u32x4_low = wasm_u32x4_extend_low_u16x8(bf16_u16x4_in_u64);
    nk_b128_vec_t result;
    result.v128 = wasm_i32x4_shl(bf16_u32x4_low, 16);
    return result;
}

/**
 *  @brief F16→F32: extract sign/exp/mantissa, rebias exponent (F16 bias=15, F32 bias=127, delta=112),
 *  Giesen magic-multiply: shift 15-bit magnitude left by 13, multiply by 2^112 to rebias.
 *  Handles zero, denormals, and normals in a single f32x4.mul. Inf/NaN fixup for exp=31.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL nk_b128_vec_t nk_f16x4_to_f32x4_v128relaxed_(nk_b64_vec_t f16_vec) {
    v128_t f16_u16x4_in_u64 = wasm_v128_load64_zero(&f16_vec.u64);
    v128_t f16_u32x4 = wasm_u32x4_extend_low_u16x8(f16_u16x4_in_u64);

    // Extract sign: (raw & 0x8000) << 16 → f32 sign bit
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_v128_and(f16_u32x4, wasm_i32x4_splat(0x8000)), 16);
    // Strip sign to get 15-bit magnitude, shift left by 13 so F16 exponent overlaps f32 exponent
    v128_t nonsign_u32x4 = wasm_v128_and(f16_u32x4, wasm_i32x4_splat(0x7FFF));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 13);

    // Magic multiply: reinterpret as f32 × 2^112 rebiases from F16 (bias=15) to f32 (bias=127).
    v128_t result_u32x4 = wasm_f32x4_mul(shifted_u32x4, wasm_i32x4_splat(0x77800000)); // 2^112 = (254-15)<<23

    // Inf/NaN fixup (branchless): result >= 2^16 means f16 exp=31 → OR in f32 exponent=255.
    // When no inf/NaN lanes exist, the OR bits are masked to zero — no effect.
    v128_t is_infnan = wasm_u32x4_ge(result_u32x4, wasm_i32x4_splat(0x47800000));
    result_u32x4 = wasm_v128_or(result_u32x4, wasm_v128_and(is_infnan, wasm_i32x4_splat(0x7F800000)));

    nk_b128_vec_t result;
    result.v128 = wasm_v128_or(result_u32x4, sign_u32x4);
    return result;
}

/**
 *  @brief E4M3→F32 via Giesen magic-multiply (WASM Relaxed SIMD).
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single f32x4.mul. NaN fixup for magnitude 0x7F.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL nk_b128_vec_t nk_e4m3x4_to_f32x4_v128relaxed_(nk_b32_vec_t e4m3_vec) {
    v128_t e4m3_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_v128_load32_zero(&e4m3_vec.u32)));

    // Extract sign: (raw >> 7) << 31
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_u32x4_shr(e4m3_u32x4, 7), 31);
    // Strip sign to get 7-bit magnitude, shift left by 20 so E4M3 exponent overlaps f32 exponent
    v128_t nonsign_u32x4 = wasm_v128_and(e4m3_u32x4, wasm_i32x4_splat(0x7F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 20);

    // Magic multiply: reinterpret as f32 × 2^120
    v128_t result_u32x4 = wasm_f32x4_mul(shifted_u32x4, wasm_i32x4_splat(0x7B800000)); // 2^120 = (254-7)<<23

    // NaN fixup (branchless): E4M3FN NaN only at magnitude 0x7F → blend in quiet NaN bits.
    v128_t is_nan_mask = wasm_i32x4_eq(nonsign_u32x4, wasm_i32x4_splat(0x7F));
    v128_t nan_bits = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7FC00000));
    result_u32x4 = wasm_i32x4_relaxed_laneselect(nan_bits, result_u32x4, is_nan_mask);

    nk_b128_vec_t result;
    result.v128 = wasm_v128_or(result_u32x4, sign_u32x4);
    return result;
}

/**
 *  @brief E5M2→F32 via Giesen magic-multiply (WASM Relaxed SIMD).
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single f32x4.mul. Inf/NaN fixup for exp=31.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL nk_b128_vec_t nk_e5m2x4_to_f32x4_v128relaxed_(nk_b32_vec_t e5m2_vec) {
    v128_t e5m2_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_v128_load32_zero(&e5m2_vec.u32)));

    // Extract sign: (raw >> 7) << 31
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_u32x4_shr(e5m2_u32x4, 7), 31);
    // Strip sign to get 7-bit magnitude, shift left by 21 so E5M2 exponent overlaps f32 exponent
    v128_t nonsign_u32x4 = wasm_v128_and(e5m2_u32x4, wasm_i32x4_splat(0x7F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 21);

    // Magic multiply: reinterpret as f32 × 2^112
    v128_t result_u32x4 = wasm_f32x4_mul(shifted_u32x4, wasm_i32x4_splat(0x77800000)); // 2^112 = (254-15)<<23

    // Inf/NaN fixup (branchless): nonsign > 123 means exp=31 → OR in f32 exponent=255.
    v128_t is_infnan = wasm_u32x4_gt(nonsign_u32x4, wasm_i32x4_splat(123));
    result_u32x4 = wasm_v128_or(result_u32x4, wasm_v128_and(is_infnan, wasm_i32x4_splat(0x7F800000)));

    nk_b128_vec_t result;
    result.v128 = wasm_v128_or(result_u32x4, sign_u32x4);
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
