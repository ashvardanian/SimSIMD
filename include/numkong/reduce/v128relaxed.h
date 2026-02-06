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

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_REDUCE_V128RELAXED_H
