/**
 *  @file v128relaxed.h
 *  @brief      WASM SIMD (Relaxed SIMD) horizontal reduction operations.
 *  @author     Ash Vardanian
 *  @date       January 31, 2026
 */

#ifndef NK_REDUCE_V128RELAXED_H
#define NK_REDUCE_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** @brief Horizontal sum of 4 floats using shuffle tree. */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x4_v128relaxed_(v128_t vec_f32x4) {
    v128_t high_f32x4 = wasm_i32x4_shuffle(vec_f32x4, vec_f32x4, 2, 3, 0, 0);
    v128_t sum1_f32x4 = wasm_f32x4_add(vec_f32x4, high_f32x4);
    v128_t high2_f32x4 = wasm_i32x4_shuffle(sum1_f32x4, sum1_f32x4, 1, 0, 0, 0);
    v128_t sum2_f32x4 = wasm_f32x4_add(sum1_f32x4, high2_f32x4);
    return wasm_f32x4_extract_lane(sum2_f32x4, 0);
}

/** @brief Horizontal sum of 2 doubles using single shuffle. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x2_v128relaxed_(v128_t vec_f64x2) {
    v128_t high_f64x2 = wasm_i64x2_shuffle(vec_f64x2, vec_f64x2, 1, 0);
    v128_t sum_f64x2 = wasm_f64x2_add(vec_f64x2, high_f64x2);
    return wasm_f64x2_extract_lane(sum_f64x2, 0);
}

/** @brief Horizontal sum of 4 signed 32-bit integers using shuffle tree. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x4_v128relaxed_(v128_t vec_i32x4) {
    v128_t high_i32x4 = wasm_i32x4_shuffle(vec_i32x4, vec_i32x4, 2, 3, 0, 0);
    v128_t sum1_i32x4 = wasm_i32x4_add(vec_i32x4, high_i32x4);
    v128_t high2_i32x4 = wasm_i32x4_shuffle(sum1_i32x4, sum1_i32x4, 1, 0, 0, 0);
    v128_t sum2_i32x4 = wasm_i32x4_add(sum1_i32x4, high2_i32x4);
    return wasm_i32x4_extract_lane(sum2_i32x4, 0);
}

/** @brief Horizontal sum of 4 unsigned 32-bit integers using shuffle tree. */
NK_INTERNAL nk_u32_t nk_reduce_add_u32x4_v128relaxed_(v128_t vec_u32x4) {
    v128_t high_u32x4 = wasm_i32x4_shuffle(vec_u32x4, vec_u32x4, 2, 3, 0, 0);
    v128_t sum1_u32x4 = wasm_i32x4_add(vec_u32x4, high_u32x4);
    v128_t high2_u32x4 = wasm_i32x4_shuffle(sum1_u32x4, sum1_u32x4, 1, 0, 0, 0);
    v128_t sum2_u32x4 = wasm_i32x4_add(sum1_u32x4, high2_u32x4);
    return (nk_u32_t)wasm_i32x4_extract_lane(sum2_u32x4, 0);
}

/** @brief  Horizontal sum of 16 unsigned 8-bit integers using pairwise widening. */
NK_INTERNAL nk_u32_t nk_reduce_add_u8x16_v128relaxed_(v128_t vec_u8x16) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(vec_u8x16);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    return nk_reduce_add_u32x4_v128relaxed_(sum_u32x4);
}

NK_INTERNAL nk_i64_t nk_reduce_add_i64x2_v128relaxed_(v128_t vec_i64x2) {
    v128_t high_i64x2 = wasm_i64x2_shuffle(vec_i64x2, vec_i64x2, 1, 0);
    v128_t sum_i64x2 = wasm_i64x2_add(vec_i64x2, high_i64x2);
    return (nk_i64_t)wasm_i64x2_extract_lane(sum_i64x2, 0);
}

NK_INTERNAL nk_u64_t nk_reduce_add_u64x2_v128relaxed_(v128_t vec_u64x2) {
    v128_t high_u64x2 = wasm_i64x2_shuffle(vec_u64x2, vec_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(vec_u64x2, high_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

NK_INTERNAL nk_i32_t nk_reduce_add_i16x8_v128relaxed_(v128_t vec_i16x8) {
    v128_t pairwise_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(vec_i16x8);
    return nk_reduce_add_i32x4_v128relaxed_(pairwise_i32x4);
}

NK_INTERNAL nk_i64_t nk_reduce_add_i32x4_to_i64_v128relaxed_(v128_t vec_i32x4) {
    v128_t low_i64x2 = wasm_i64x2_extend_low_i32x4(vec_i32x4);
    v128_t high_i64x2 = wasm_i64x2_extend_high_i32x4(vec_i32x4);
    v128_t sum_i64x2 = wasm_i64x2_add(low_i64x2, high_i64x2);
    return nk_reduce_add_i64x2_v128relaxed_(sum_i64x2);
}

NK_INTERNAL nk_u64_t nk_reduce_add_u32x4_to_u64_v128relaxed_(v128_t vec_u32x4) {
    v128_t low_u64x2 = wasm_u64x2_extend_low_u32x4(vec_u32x4);
    v128_t high_u64x2 = wasm_u64x2_extend_high_u32x4(vec_u32x4);
    v128_t sum_u64x2 = wasm_i64x2_add(low_u64x2, high_u64x2);
    return nk_reduce_add_u64x2_v128relaxed_(sum_u64x2);
}

NK_INTERNAL v128_t nk_u64_sadd_epi64_v128relaxed_(v128_t a_u64x2, v128_t b_u64x2) {
    v128_t result_u64x2 = wasm_i64x2_add(a_u64x2, b_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t a_biased_i64x2 = wasm_v128_xor(a_u64x2, sign_bit_i64x2);
    v128_t result_biased_i64x2 = wasm_v128_xor(result_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(a_biased_i64x2, result_biased_i64x2);
    return wasm_v128_or(result_u64x2, overflow_u64x2);
}

NK_INTERNAL v128_t nk_i64_smul_sq_epi64_v128relaxed_(v128_t val_i64x2) {
    v128_t sign_i64x2 = wasm_i64x2_gt(wasm_i64x2_splat(0), val_i64x2);
    v128_t abs_val_u64x2 = wasm_i64x2_sub(wasm_v128_xor(val_i64x2, sign_i64x2), sign_i64x2);
    v128_t low_halves_i32x4 = wasm_i32x4_shuffle(abs_val_u64x2, abs_val_u64x2, 0, 2, 0, 0);
    v128_t low_squared_u64x2 = wasm_u64x2_extmul_low_u32x4(low_halves_i32x4, low_halves_i32x4);
    v128_t high_bits_u64x2 = wasm_u64x2_shr(abs_val_u64x2, 32);
    v128_t is_small_u64x2 = wasm_i64x2_eq(high_bits_u64x2, wasm_i64x2_splat(0));
    v128_t saturated_u64x2 = wasm_i64x2_splat(NK_I64_MAX);
    return wasm_i32x4_relaxed_laneselect(low_squared_u64x2, saturated_u64x2, is_small_u64x2);
}

NK_INTERNAL v128_t nk_u64_smul_sq_epi64_v128relaxed_(v128_t val_u64x2) {
    v128_t low_halves_i32x4 = wasm_i32x4_shuffle(val_u64x2, val_u64x2, 0, 2, 0, 0);
    v128_t low_squared_u64x2 = wasm_u64x2_extmul_low_u32x4(low_halves_i32x4, low_halves_i32x4);
    v128_t high_bits_u64x2 = wasm_u64x2_shr(val_u64x2, 32);
    v128_t is_small_u64x2 = wasm_i64x2_eq(high_bits_u64x2, wasm_i64x2_splat(0));
    v128_t saturated_u64x2 = wasm_i64x2_splat((nk_i64_t)-1);
    return wasm_i32x4_relaxed_laneselect(low_squared_u64x2, saturated_u64x2, is_small_u64x2);
}

NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x2_v128relaxed_(v128_t v_u64x2) {
    v128_t swapped_u64x2 = wasm_i64x2_shuffle(v_u64x2, v_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(v_u64x2, swapped_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t v_biased_i64x2 = wasm_v128_xor(v_u64x2, sign_bit_i64x2);
    v128_t sum_biased_i64x2 = wasm_v128_xor(sum_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(v_biased_i64x2, sum_biased_i64x2);
    sum_u64x2 = wasm_v128_or(sum_u64x2, overflow_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_REDUCE_V128RELAXED_H
