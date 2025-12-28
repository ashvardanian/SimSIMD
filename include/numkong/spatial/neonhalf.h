/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/spatial/neonhalf.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_NEONHALF_H
#define NK_SPATIAL_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/neonhalf.h" // nk_partial_load_f16x4_to_f32x4_neonhalf_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t distance_sq_f32x4 = vdupq_n_f32(0);

nk_l2sq_f16_neonhalf_cycle:
    if (n < 4) {
        nk_partial_load_f16x4_to_f32x4_neonhalf_(a, n, &a_f32x4);
        nk_partial_load_f16x4_to_f32x4_neonhalf_(b, n, &b_f32x4);
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    distance_sq_f32x4 = vfmaq_f32(distance_sq_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_l2sq_f16_neonhalf_cycle;

    *result = vaddvq_f32(distance_sq_f32x4);
}
NK_PUBLIC void nk_l2_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f16_neonhalf(a, b, n, result);
    *result = nk_sqrt_f32_neon_(*result);
}

NK_PUBLIC void nk_angular_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t dot_product_f32x4 = vdupq_n_f32(0), a_norm_sq_f32x4 = vdupq_n_f32(0), b_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t a_f32x4, b_f32x4;

nk_angular_f16_neonhalf_cycle:
    if (n < 4) {
        nk_partial_load_f16x4_to_f32x4_neonhalf_(a, n, &a_f32x4);
        nk_partial_load_f16x4_to_f32x4_neonhalf_(b, n, &b_f32x4);
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    dot_product_f32x4 = vfmaq_f32(dot_product_f32x4, a_f32x4, b_f32x4);
    a_norm_sq_f32x4 = vfmaq_f32(a_norm_sq_f32x4, a_f32x4, a_f32x4);
    b_norm_sq_f32x4 = vfmaq_f32(b_norm_sq_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_f16_neonhalf_cycle;

    nk_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    nk_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    nk_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = nk_angular_normalize_f32_neon_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef nk_dot_f16x8_state_neonhalf_t nk_angular_f16x8_state_neonhalf_t;
NK_INTERNAL void nk_angular_f16x8_init_neonhalf(nk_angular_f16x8_state_neonhalf_t *state) {
    nk_dot_f16x8_init_neonhalf(state);
}
NK_INTERNAL void nk_angular_f16x8_update_neonhalf(nk_angular_f16x8_state_neonhalf_t *state, nk_b128_vec_t a,
                                                  nk_b128_vec_t b) {
    nk_dot_f16x8_update_neonhalf(state, a, b);
}
NK_INTERNAL void nk_angular_f16x8_finalize_neonhalf(nk_angular_f16x8_state_neonhalf_t const *state_a,
                                                    nk_angular_f16x8_state_neonhalf_t const *state_b,
                                                    nk_angular_f16x8_state_neonhalf_t const *state_c,
                                                    nk_angular_f16x8_state_neonhalf_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single call
    nk_f32_t dots[4];
    nk_dot_f16x8_finalize_neonhalf(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors for parallel processing
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);

    // Compute products for normalization: query_sq * target_sq
    float32x4_t products = vmulq_f32(query_sq, target_sq);

    // Vectorized rsqrt with Newton-Raphson (2 iterations)
    float32x4_t rsqrt_vec = vrsqrteq_f32(products);
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));

    // Compute angular distance = 1 - dot * rsqrt(product)
    float32x4_t ones = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t result_vec = vsubq_f32(ones, vmulq_f32(dots_vec, rsqrt_vec));

    // Clamp to [0, inf) and handle edge cases
    result_vec = vmaxq_f32(result_vec, zeros);
    uint32x4_t products_zero = vceqq_f32(products, zeros);
    uint32x4_t dots_zero = vceqq_f32(dots_vec, zeros);
    uint32x4_t both_zero = vandq_u32(products_zero, dots_zero);
    result_vec = vbslq_f32(both_zero, zeros, result_vec);
    uint32x4_t prod_zero_dot_nonzero = vandq_u32(products_zero, vmvnq_u32(dots_zero));
    result_vec = vbslq_f32(prod_zero_dot_nonzero, ones, result_vec);

    vst1q_f32(results, result_vec);
}

typedef nk_dot_f16x8_state_neonhalf_t nk_l2_f16x8_state_neonhalf_t;
NK_INTERNAL void nk_l2_f16x8_init_neonhalf(nk_l2_f16x8_state_neonhalf_t *state) { nk_dot_f16x8_init_neonhalf(state); }
NK_INTERNAL void nk_l2_f16x8_update_neonhalf(nk_l2_f16x8_state_neonhalf_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f16x8_update_neonhalf(state, a, b);
}
NK_INTERNAL void nk_l2_f16x8_finalize_neonhalf(nk_l2_f16x8_state_neonhalf_t const *state_a,
                                               nk_l2_f16x8_state_neonhalf_t const *state_b,
                                               nk_l2_f16x8_state_neonhalf_t const *state_c,
                                               nk_l2_f16x8_state_neonhalf_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products
    nk_f32_t dots[4];
    nk_dot_f16x8_finalize_neonhalf(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);

    // Compute dist_sq = query_sq + target_sq - 2*dot using FMA
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    float32x4_t sum_sq = vaddq_f32(query_sq, target_sq);
    float32x4_t dist_sq = vfmaq_f32(sum_sq, neg_two, dots_vec);

    // Clamp and sqrt
    float32x4_t zeros = vdupq_n_f32(0.0f);
    dist_sq = vmaxq_f32(dist_sq, zeros);
    float32x4_t dist_vec = vsqrtq_f32(dist_sq);

    vst1q_f32(results, dist_vec);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEONHALF_H