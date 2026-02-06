/**
 *  @file v128relaxed.h
 *  @brief      WASM SIMD (v128) type conversion helpers for BF16/F16 to F32.
 *  @author     Ash Vardanian
 *  @date       January 31, 2026
 */

#ifndef NK_CAST_V128RELAXED_H
#define NK_CAST_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h" // For scalar fallback

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL nk_b128_vec_t nk_bf16x4_to_f32x4_v128relaxed_(nk_b64_vec_t bf16_vec) {
    // Load 4x u16 (64 bits) into lower half of v128, zero upper half
    v128_t bf16_u16x4_in_u64 = wasm_v128_load64_zero(&bf16_vec.u64);

    // Widen u16 → u32: [u16, u16, u16, u16, 0, 0, 0, 0] → [u32, u32, u32, u32]
    // Uses zero-extension (upper 16 bits of each u32 become 0)
    v128_t bf16_u32x4_low = wasm_u32x4_extend_low_u16x8(bf16_u16x4_in_u64);

    // Shift left by 16 bits: moves BF16 into F32 position
    // BF16: [S|EEEEEEEE|MMMMMMM|0000000000000000]
    // F32:  [S|EEEEEEEE|MMMMMMM00000000000000000]
    nk_b128_vec_t result;
    result.v128 = wasm_i32x4_shl(bf16_u32x4_low, 16);
    return result;
}

NK_INTERNAL nk_b128_vec_t nk_f16x4_to_f32x4_v128relaxed_(nk_b64_vec_t f16_vec) {
    // Load 4x u16 into v128, zero-extend to u32x4
    v128_t f16_u16x4_in_u64 = wasm_v128_load64_zero(&f16_vec.u64);
    v128_t f16_u32x4 = wasm_u32x4_extend_low_u16x8(f16_u16x4_in_u64);

    // Extract bit fields
    v128_t sign_u32x4 = wasm_v128_and(f16_u32x4, wasm_i32x4_splat(0x8000));                  // Bit 15
    v128_t exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(f16_u32x4, 10), wasm_i32x4_splat(0x1F)); // Bits 14-10
    v128_t mant_u32x4 = wasm_v128_and(f16_u32x4, wasm_i32x4_splat(0x03FF));                  // Bits 9-0

    // Shift sign to F32 position (bit 31)
    v128_t sign_f32_u32x4 = wasm_i32x4_shl(sign_u32x4, 16);

    // Normal (exp ∈ [1, 30])
    // Rebias exponent: F16 bias=15, F32 bias=127 → add 112
    // Shift mantissa: 10 bits → 23 bits (shift left by 13)
    v128_t exp_rebiased_u32x4 = wasm_i32x4_add(exp_u32x4, wasm_i32x4_splat(112));
    v128_t normal_exp_u32x4 = wasm_i32x4_shl(exp_rebiased_u32x4, 23);
    v128_t normal_mant_u32x4 = wasm_i32x4_shl(mant_u32x4, 13);
    v128_t normal_bits_u32x4 = wasm_v128_or(sign_f32_u32x4, wasm_v128_or(normal_exp_u32x4, normal_mant_u32x4));

    // Zero (exp=0, mant=0)
    v128_t zero_bits_u32x4 = sign_f32_u32x4; // Just sign bit

    // Infinity/NaN (exp=31)
    // Infinity: 0x7F800000 | sign
    // NaN: 0x7F800000 | sign | (mant << 13) [preserves NaN payload]
    v128_t inf_nan_bits_u32x4 = wasm_v128_or(
        sign_f32_u32x4, wasm_v128_or(wasm_i32x4_splat(0x7F800000), wasm_i32x4_shl(mant_u32x4, 13)));

    // Denormal (exp=0, mant≠0) - FPU-based normalization
    // F16 denormal value = 2^-14 × (0.mantissa_bits) = mantissa_bits × 2^-24
    //
    // Strategy: Use FPU to normalize by converting to float and multiplying by magic constant
    // 1. Convert mantissa (integer) to F32: cvt_u32_to_f32(mant)
    // 2. Multiply by 2^-24 (magic constant 0x33800000 in F32)
    // 3. FPU normalizes automatically, giving correct F32 representation
    // 4. Reinterpret as bits and apply sign

    // Convert mantissa u32 → f32 (each lane independently)
    v128_t mant_f32x4 = wasm_f32x4_convert_u32x4(mant_u32x4);

    // Multiply by 2^-24 (F32 hex: 0x33800000)
    v128_t magic_f32x4 = wasm_f32x4_splat(0x1p-24f); // 2^-24 in hex float notation
    v128_t denorm_normalized_f32x4 = wasm_f32x4_mul(mant_f32x4, magic_f32x4);

    // Reinterpret f32x4 as u32x4 bits (v128_t is polymorphic - just assign)
    v128_t denorm_bits_u32x4 = denorm_normalized_f32x4;

    // Apply sign (OR with sign bit, since denorm result is always positive)
    denorm_bits_u32x4 = wasm_v128_or(denorm_bits_u32x4, sign_f32_u32x4);

    // Build Masks
    v128_t exp_zero_mask = wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(0));
    v128_t mant_zero_mask = wasm_i32x4_eq(mant_u32x4, wasm_i32x4_splat(0));
    v128_t exp_max_mask = wasm_i32x4_eq(exp_u32x4, wasm_i32x4_splat(31));

    v128_t is_zero_mask = wasm_v128_and(exp_zero_mask, mant_zero_mask);        // exp=0 AND mant=0
    v128_t is_denormal_mask = wasm_v128_andnot(exp_zero_mask, mant_zero_mask); // exp=0 AND mant≠0

    // Blend the results
    v128_t result_u32x4 = normal_bits_u32x4;

    // Apply zero where exp=0 && mant=0
    result_u32x4 = wasm_v128_bitselect(zero_bits_u32x4, result_u32x4, is_zero_mask);

    // Apply denormal where exp=0 && mant≠0
    result_u32x4 = wasm_v128_bitselect(denorm_bits_u32x4, result_u32x4, is_denormal_mask);

    // Apply inf/NaN where exp=31
    result_u32x4 = wasm_v128_bitselect(inf_nan_bits_u32x4, result_u32x4, exp_max_mask);

    nk_b128_vec_t result;
    result.v128 = result_u32x4;
    return result;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_CAST_V128RELAXED_H
