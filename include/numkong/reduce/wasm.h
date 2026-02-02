/**
 *  @file       wasm.h
 *  @brief      WASM SIMD (Relaxed SIMD) horizontal reduction operations.
 *  @author     Ash Vardanian
 *  @date       January 31, 2026
 */

#ifndef NK_REDUCE_WASM_H
#define NK_REDUCE_WASM_H

#if NK_TARGET_V128RELAXED
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Horizontal sum of 4 floats using shuffle tree. */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x4_wasm_(v128_t vec) {
    // First reduction: add lanes [0,2] and [1,3]
    v128_t hi = wasm_i32x4_shuffle(vec, vec, 2, 3, 0, 0); // [c,d,a,a]
    v128_t sum1 = wasm_f32x4_add(vec, hi);                // [a+c, b+d, *, *]

    // Second reduction: add lanes [0] and [1]
    v128_t hi2 = wasm_i32x4_shuffle(sum1, sum1, 1, 0, 0, 0); // [b+d, *, *, *]
    v128_t sum2 = wasm_f32x4_add(sum1, hi2);                 // [a+b+c+d, *, *, *]

    return wasm_f32x4_extract_lane(sum2, 0);
}

/** @brief Horizontal sum of 2 doubles using single shuffle. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x2_wasm_(v128_t vec) {
    v128_t hi = wasm_i64x2_shuffle(vec, vec, 1, 0); // [b,a]
    v128_t sum = wasm_f64x2_add(vec, hi);           // [a+b, *]
    return wasm_f64x2_extract_lane(sum, 0);
}

/** @brief Horizontal sum of 4 signed 32-bit integers using shuffle tree. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x4_wasm_(v128_t vec) {
    v128_t hi = wasm_i32x4_shuffle(vec, vec, 2, 3, 0, 0);
    v128_t sum1 = wasm_i32x4_add(vec, hi);
    v128_t hi2 = wasm_i32x4_shuffle(sum1, sum1, 1, 0, 0, 0);
    v128_t sum2 = wasm_i32x4_add(sum1, hi2);
    return wasm_i32x4_extract_lane(sum2, 0);
}

/** @brief Horizontal sum of 4 unsigned 32-bit integers using shuffle tree. */
NK_INTERNAL nk_u32_t nk_reduce_add_u32x4_wasm_(v128_t vec) {
    v128_t hi = wasm_i32x4_shuffle(vec, vec, 2, 3, 0, 0);
    v128_t sum1 = wasm_i32x4_add(vec, hi);
    v128_t hi2 = wasm_i32x4_shuffle(sum1, sum1, 1, 0, 0, 0);
    v128_t sum2 = wasm_i32x4_add(sum1, hi2);
    return (nk_u32_t)wasm_i32x4_extract_lane(sum2, 0);
}

/** @brief  Horizontal sum of 16 unsigned 8-bit integers using pairwise widening. */
NK_INTERNAL nk_u32_t nk_reduce_add_u8x16_wasm_(v128_t vec_u8x16) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(vec_u8x16);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    return nk_reduce_add_u32x4_wasm_(sum_u32x4);
}

#if defined(__cplusplus)
}
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_REDUCE_WASM_H
