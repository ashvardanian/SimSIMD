/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/spatial/neon.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_NEON_H
#define NK_SPATIAL_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL nk_f32_t nk_sqrt_f32_neon_(nk_f32_t x) { return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0); }
NK_INTERNAL nk_f64_t nk_sqrt_f64_neon_(nk_f64_t x) { return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0); }
NK_INTERNAL nk_f32_t nk_angular_normalize_f32_neon_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts = vrsqrte_f32(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    vst1_f32(squares_arr, rsqrts);
    nk_f32_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_neon_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5*2^-12.
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts = vrsqrteq_f64(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    rsqrts = vmulq_f64(rsqrts, vrsqrtsq_f64(vmulq_f64(squares, rsqrts), rsqrts));
    vst1q_f64(squares_arr, rsqrts);
    nk_f64_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NK_PUBLIC void nk_l2_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_neon(a, b, n, result);
    *result = nk_sqrt_f32_neon_(*result);
}
NK_PUBLIC void nk_l2sq_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    nk_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        nk_f32_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

NK_PUBLIC void nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t ab_vec = vdupq_n_f32(0), a2_vec = vdupq_n_f32(0), b2_vec = vdupq_n_f32(0);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        ab_vec = vfmaq_f32(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f32(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f32(b2_vec, b_vec, b_vec);
    }
    nk_f32_t ab = vaddvq_f32(ab_vec), a2 = vaddvq_f32(a2_vec), b2 = vaddvq_f32(b2_vec);
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = (nk_f32_t)nk_angular_normalize_f64_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_l2_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_neon(a, b, n, result);
    *result = nk_sqrt_f64_neon_(*result);
}
NK_PUBLIC void nk_l2sq_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t sum_vec = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        float64x2_t diff_vec = vsubq_f64(a_vec, b_vec);
        sum_vec = vfmaq_f64(sum_vec, diff_vec, diff_vec);
    }
    nk_f64_t sum = vaddvq_f64(sum_vec);
    for (; i < n; ++i) {
        nk_f64_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

NK_PUBLIC void nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t ab_vec = vdupq_n_f64(0), a2_vec = vdupq_n_f64(0), b2_vec = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        ab_vec = vfmaq_f64(ab_vec, a_vec, b_vec);
        a2_vec = vfmaq_f64(a2_vec, a_vec, a_vec);
        b2_vec = vfmaq_f64(b2_vec, b_vec, b_vec);
    }
    nk_f64_t ab = vaddvq_f64(ab_vec), a2 = vaddvq_f64(a2_vec), b2 = vaddvq_f64(b2_vec);
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = nk_angular_normalize_f64_neon_(ab, a2, b2);
}

typedef nk_dot_f32x4_state_neon_t nk_angular_f32x4_state_neon_t;
NK_INTERNAL void nk_angular_f32x4_init_neon(nk_angular_f32x4_state_neon_t *state) { nk_dot_f32x4_init_neon(state); }
NK_INTERNAL void nk_angular_f32x4_update_neon(nk_angular_f32x4_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f32x4_update_neon(state, a, b);
}
NK_INTERNAL void nk_angular_f32x4_finalize_neon(nk_angular_f32x4_state_neon_t const *state_a,
                                                nk_angular_f32x4_state_neon_t const *state_b,
                                                nk_angular_f32x4_state_neon_t const *state_c,
                                                nk_angular_f32x4_state_neon_t const *state_d, nk_f32_t query_norm,
                                                nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single vectorized call
    nk_f32_t dots[4];
    nk_dot_f32x4_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F64 vectors for parallel processing (2x float64x2_t for precision)
    float64x2_t dots_ab = {(nk_f64_t)dots[0], (nk_f64_t)dots[1]};
    float64x2_t dots_cd = {(nk_f64_t)dots[2], (nk_f64_t)dots[3]};

    nk_f64_t query_norm_sq = (nk_f64_t)query_norm * (nk_f64_t)query_norm;
    float64x2_t query_sq = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab = {(nk_f64_t)target_norm_a, (nk_f64_t)target_norm_b};
    float64x2_t target_norms_cd = {(nk_f64_t)target_norm_c, (nk_f64_t)target_norm_d};
    float64x2_t target_sq_ab = vmulq_f64(target_norms_ab, target_norms_ab);
    float64x2_t target_sq_cd = vmulq_f64(target_norms_cd, target_norms_cd);

    // Compute products for normalization: query_sq * target_sq
    float64x2_t products_ab = vmulq_f64(query_sq, target_sq_ab);
    float64x2_t products_cd = vmulq_f64(query_sq, target_sq_cd);

    // Vectorized rsqrt with Newton-Raphson (2 iterations for ~48-bit precision)
    float64x2_t rsqrt_ab = vrsqrteq_f64(products_ab);
    float64x2_t rsqrt_cd = vrsqrteq_f64(products_cd);
    rsqrt_ab = vmulq_f64(rsqrt_ab, vrsqrtsq_f64(vmulq_f64(products_ab, rsqrt_ab), rsqrt_ab));
    rsqrt_cd = vmulq_f64(rsqrt_cd, vrsqrtsq_f64(vmulq_f64(products_cd, rsqrt_cd), rsqrt_cd));
    rsqrt_ab = vmulq_f64(rsqrt_ab, vrsqrtsq_f64(vmulq_f64(products_ab, rsqrt_ab), rsqrt_ab));
    rsqrt_cd = vmulq_f64(rsqrt_cd, vrsqrtsq_f64(vmulq_f64(products_cd, rsqrt_cd), rsqrt_cd));

    // Compute angular distance = 1 - dot * rsqrt(product)
    float64x2_t ones = vdupq_n_f64(1.0);
    float64x2_t zeros = vdupq_n_f64(0.0);
    float64x2_t result_ab = vsubq_f64(ones, vmulq_f64(dots_ab, rsqrt_ab));
    float64x2_t result_cd = vsubq_f64(ones, vmulq_f64(dots_cd, rsqrt_cd));

    // Clamp to [0, inf)
    result_ab = vmaxq_f64(result_ab, zeros);
    result_cd = vmaxq_f64(result_cd, zeros);

    // Handle edge cases with vectorized selects
    uint64x2_t products_zero_ab = vceqq_f64(products_ab, zeros);
    uint64x2_t products_zero_cd = vceqq_f64(products_cd, zeros);
    uint64x2_t dots_zero_ab = vceqq_f64(dots_ab, zeros);
    uint64x2_t dots_zero_cd = vceqq_f64(dots_cd, zeros);

    // Both zero -> result = 0; products zero but dots nonzero -> result = 1
    uint64x2_t both_zero_ab = vandq_u64(products_zero_ab, dots_zero_ab);
    uint64x2_t both_zero_cd = vandq_u64(products_zero_cd, dots_zero_cd);
    result_ab = vbslq_f64(both_zero_ab, zeros, result_ab);
    result_cd = vbslq_f64(both_zero_cd, zeros, result_cd);

    uint64x2_t prod_zero_dot_nonzero_ab = vandq_u64(products_zero_ab, vmvnq_u64(dots_zero_ab));
    uint64x2_t prod_zero_dot_nonzero_cd = vandq_u64(products_zero_cd, vmvnq_u64(dots_zero_cd));
    result_ab = vbslq_f64(prod_zero_dot_nonzero_ab, ones, result_ab);
    result_cd = vbslq_f64(prod_zero_dot_nonzero_cd, ones, result_cd);

    // Convert to F32 and store
    float32x2_t result_ab_f32 = vcvt_f32_f64(result_ab);
    float32x2_t result_cd_f32 = vcvt_f32_f64(result_cd);
    float32x4_t result_vec = vcombine_f32(result_ab_f32, result_cd_f32);
    vst1q_f32(results, result_vec);
}

typedef nk_dot_f32x4_state_neon_t nk_l2_f32x4_state_neon_t;
NK_INTERNAL void nk_l2_f32x4_init_neon(nk_l2_f32x4_state_neon_t *state) { nk_dot_f32x4_init_neon(state); }
NK_INTERNAL void nk_l2_f32x4_update_neon(nk_l2_f32x4_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f32x4_update_neon(state, a, b);
}
NK_INTERNAL void nk_l2_f32x4_finalize_neon(nk_l2_f32x4_state_neon_t const *state_a,
                                           nk_l2_f32x4_state_neon_t const *state_b,
                                           nk_l2_f32x4_state_neon_t const *state_c,
                                           nk_l2_f32x4_state_neon_t const *state_d, nk_f32_t query_norm,
                                           nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                           nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products
    nk_f32_t dots[4];
    nk_dot_f32x4_finalize_neon(state_a, state_b, state_c, state_d, dots);

    // Build F64 vectors (for precision as in original)
    float64x2_t dots_ab = {(nk_f64_t)dots[0], (nk_f64_t)dots[1]};
    float64x2_t dots_cd = {(nk_f64_t)dots[2], (nk_f64_t)dots[3]};

    nk_f64_t query_norm_sq = (nk_f64_t)query_norm * (nk_f64_t)query_norm;
    float64x2_t query_sq = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab = {(nk_f64_t)target_norm_a, (nk_f64_t)target_norm_b};
    float64x2_t target_norms_cd = {(nk_f64_t)target_norm_c, (nk_f64_t)target_norm_d};
    float64x2_t target_sq_ab = vmulq_f64(target_norms_ab, target_norms_ab);
    float64x2_t target_sq_cd = vmulq_f64(target_norms_cd, target_norms_cd);

    // Compute dist_sq = query_sq + target_sq - 2*dot using FMA
    float64x2_t neg_two = vdupq_n_f64(-2.0);
    float64x2_t sum_sq_ab = vaddq_f64(query_sq, target_sq_ab);
    float64x2_t sum_sq_cd = vaddq_f64(query_sq, target_sq_cd);
    float64x2_t dist_sq_ab = vfmaq_f64(sum_sq_ab, neg_two, dots_ab);
    float64x2_t dist_sq_cd = vfmaq_f64(sum_sq_cd, neg_two, dots_cd);

    // Clamp negative values to zero (numerical stability)
    float64x2_t zeros = vdupq_n_f64(0.0);
    dist_sq_ab = vmaxq_f64(dist_sq_ab, zeros);
    dist_sq_cd = vmaxq_f64(dist_sq_cd, zeros);

    // Compute sqrt using hardware vsqrtq_f64
    float64x2_t dist_ab = vsqrtq_f64(dist_sq_ab);
    float64x2_t dist_cd = vsqrtq_f64(dist_sq_cd);

    // Convert to F32 and store
    float32x2_t dist_ab_f32 = vcvt_f32_f64(dist_ab);
    float32x2_t dist_cd_f32 = vcvt_f32_f64(dist_cd);
    float32x4_t dist_vec = vcombine_f32(dist_ab_f32, dist_cd_f32);
    vst1q_f32(results, dist_vec);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEON_H