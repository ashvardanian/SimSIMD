/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for WASM.
 *  @file include/numkong/spatial/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 2, 2026
 *
 *  Contains:
 *  - Euclidean (L2) distance
 *  - Squared Euclidean (L2SQ) distance
 *  - Angular distance (1 - cosine similarity)
 *
 *  For dtypes:
 *  - 64-bit IEEE floating point (f64)
 *  - 32-bit IEEE floating point (f32)
 *  - 16-bit IEEE floating point (f16)
 *  - 16-bit brain floating point (bf16)
 *
 *  Key improvements:
 *  - F32→F64 upcast for angular_f32 (matches Haswell/NEON precision strategy)
 *  - Parallel SIMD sqrt for normalization (computes both sqrts simultaneously)
 *  - Edge case handling (zero vectors, numerical stability)
 *  - Uses relaxed FMA for optimal throughput
 *
 *  @see For pattern references:
 *  - Haswell: include/numkong/spatial/haswell.h
 *  - NEON: include/numkong/spatial/neon.h
 */

#ifndef NK_SPATIAL_V128RELAXED_H
#define NK_SPATIAL_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_INTERNAL nk_f32_t nk_f32_sqrt_v128relaxed_(nk_f32_t x) {
    return wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(x)), 0);
}

NK_INTERNAL nk_f64_t nk_f64_sqrt_v128relaxed_(nk_f64_t x) {
    return wasm_f64x2_extract_lane(wasm_f64x2_sqrt(wasm_f64x2_splat(x)), 0);
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_v128relaxed_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    // Edge case: both vectors have zero magnitude
    if (a2 == 0.0 && b2 == 0.0) return 0.0;
    // Edge case: dot product is zero (perpendicular or one vector is zero)
    if (ab == 0.0) return 1.0;

    // Compute both square roots in parallel using SIMD (more efficient than 2 scalar sqrts)
    v128_t squares_f64x2 = wasm_f64x2_make(a2, b2);
    v128_t sqrts_f64x2 = wasm_f64x2_sqrt(squares_f64x2);
    nk_f64_t a_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 0);
    nk_f64_t b_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 1);

    // Compute angular distance: 1 - cosine_similarity
    nk_f64_t result = 1.0 - ab / (a_sqrt * b_sqrt);

    // Clamp negative results to 0 (can occur due to floating-point rounding)
    return result > 0.0 ? result : 0.0;
}

#pragma region - Traditional Floats

NK_PUBLIC void nk_sqeuclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_sqeuclidean_f32_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b32x4_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b32x4_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_serial_(a_scalars, &a_vec);
        nk_load_b128_serial_(b_scalars, &b_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    v128_t diff_f32x4 = wasm_f32x4_sub(a_vec.v128, b_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_f32_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_sqeuclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_sqeuclidean_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_serial_(a_scalars, &a_vec);
        nk_load_b128_serial_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t diff_f64x2 = wasm_f64x2_sub(a_vec.v128, b_vec.v128);
    sum_f64x2 = wasm_f64x2_relaxed_madd(diff_f64x2, diff_f64x2, sum_f64x2);
    if (count_scalars) goto nk_sqeuclidean_f64_v128relaxed_cycle;

    *result = nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
}

NK_PUBLIC void nk_euclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_f32_v128relaxed(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128relaxed_(l2sq);
}

NK_PUBLIC void nk_euclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t l2sq;
    nk_sqeuclidean_f64_v128relaxed(a, b, n, &l2sq);
    *result = nk_f64_sqrt_v128relaxed_(l2sq);
}

NK_PUBLIC void nk_angular_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // F32 → F64 upcast for numerical stability
    v128_t ab_f64x2 = wasm_f64x2_splat(0.0);
    v128_t a2_f64x2 = wasm_f64x2_splat(0.0);
    v128_t b2_f64x2 = wasm_f64x2_splat(0.0);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f32_vec, b_f32_vec;

nk_angular_f32_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b32x2_serial_(a_scalars, &a_f32_vec, count_scalars);
        nk_partial_load_b32x2_serial_(b_scalars, &b_f32_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f32_vec);
        nk_load_b64_serial_(b_scalars, &b_f32_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // Upcast F32x2 → F64x2 for high-precision accumulation
    v128_t a_f32x2 = wasm_v128_load64_zero(&a_f32_vec.u64);
    v128_t b_f32x2 = wasm_v128_load64_zero(&b_f32_vec.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);

    // Accumulate: ab += a·b, a2 += a·a, b2 += b·b
    ab_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, b_f64x2, ab_f64x2);
    a2_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, a_f64x2, a2_f64x2);
    b2_f64x2 = wasm_f64x2_relaxed_madd(b_f64x2, b_f64x2, b2_f64x2);
    if (count_scalars) goto nk_angular_f32_v128relaxed_cycle;

    // Reduce and normalize using F64 arithmetic
    nk_f64_t ab_f64 = nk_reduce_add_f64x2_v128relaxed_(ab_f64x2);
    nk_f64_t a2_f64 = nk_reduce_add_f64x2_v128relaxed_(a2_f64x2);
    nk_f64_t b2_f64 = nk_reduce_add_f64x2_v128relaxed_(b2_f64x2);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_angular_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t ab_f64x2 = wasm_f64x2_splat(0.0);
    v128_t a2_f64x2 = wasm_f64x2_splat(0.0);
    v128_t b2_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_angular_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_serial_(a_scalars, &a_vec);
        nk_load_b128_serial_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // Accumulate: ab += a·b, a2 += a·a, b2 += b·b
    ab_f64x2 = wasm_f64x2_relaxed_madd(a_vec.v128, b_vec.v128, ab_f64x2);
    a2_f64x2 = wasm_f64x2_relaxed_madd(a_vec.v128, a_vec.v128, a2_f64x2);
    b2_f64x2 = wasm_f64x2_relaxed_madd(b_vec.v128, b_vec.v128, b2_f64x2);
    if (count_scalars) goto nk_angular_f64_v128relaxed_cycle;

    // Reduce and normalize
    nk_f64_t ab = nk_reduce_add_f64x2_v128relaxed_(ab_f64x2);
    nk_f64_t a2 = nk_reduce_add_f64x2_v128relaxed_(a2_f64x2);
    nk_f64_t b2 = nk_reduce_add_f64x2_v128relaxed_(b2_f64x2);
    *result = nk_angular_normalize_f64_v128relaxed_(ab, a2, b2);
}

#pragma endregion - Traditional Floats
#pragma region - Smaller Floats

NK_PUBLIC void nk_sqeuclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_sqeuclidean_f16_v128relaxed_cycle:
    // Tail or full load
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_f16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_f16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f16_vec);
        nk_load_b64_serial_(b_scalars, &b_f16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert f16 → f32 (4 elements)
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);

    // Accumulate (a - b)²
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);

    if (count_scalars) goto nk_sqeuclidean_f16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_f16_v128relaxed(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128relaxed_(l2sq);
}

NK_PUBLIC void nk_angular_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_angular_f16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_f16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_f16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f16_vec);
        nk_load_b64_serial_(b_scalars, &b_f16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert f16 → f32
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);

    // Triple accumulation: ab, a², b²
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);

    if (count_scalars) goto nk_angular_f16_v128relaxed_cycle;

    // Reduce accumulators
    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);

    // Normalize using f64 helper (handles edge cases: zero vectors, perpendicular, clamping)
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_bf16_vec, b_bf16_vec;

nk_sqeuclidean_bf16_v128relaxed_cycle:
    // Tail or full load
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_bf16_vec);
        nk_load_b64_serial_(b_scalars, &b_bf16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert bf16 → f32 (4 elements)
    nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
    nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);

    // Accumulate (a - b)²
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);

    if (count_scalars) goto nk_sqeuclidean_bf16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_bf16_v128relaxed(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128relaxed_(l2sq);
}

NK_PUBLIC void nk_angular_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_bf16_vec, b_bf16_vec;

nk_angular_bf16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_bf16_vec);
        nk_load_b64_serial_(b_scalars, &b_bf16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert bf16 → f32
    nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
    nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);

    // Triple accumulation: ab, a², b²
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);

    if (count_scalars) goto nk_angular_bf16_v128relaxed_cycle;

    // Reduce accumulators
    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);

    // Normalize using f64 helper (handles edge cases: zero vectors, perpendicular, clamping)
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

#pragma endregion - Smaller Floats

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SPATIAL_V128RELAXED_H
