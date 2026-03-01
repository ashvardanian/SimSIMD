/**
 *  @brief SIMD-accelerated Scalar Math Helpers for WASM.
 *  @file include/numkong/scalars/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalars.h
 */
#ifndef NK_SCALARS_V128RELAXED_H
#define NK_SCALARS_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_v128relaxed(nk_f32_t x) {
    return wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(x)), 0);
}
NK_PUBLIC nk_f64_t nk_f64_sqrt_v128relaxed(nk_f64_t x) {
    return wasm_f64x2_extract_lane(wasm_f64x2_sqrt(wasm_f64x2_splat(x)), 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SCALARS_V128RELAXED_H
