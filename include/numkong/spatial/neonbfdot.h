/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/spatial/neonbfdot.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_NEONBFDOT_H
#define NK_SPATIAL_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/neon.h" // nk_partial_load_b16x8_neon_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_angular_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {

    // Similar to `nk_angular_i8_neon`, we can use the `BFMMLA` instruction through
    // the `vbfmmlaq_f32` intrinsic to compute matrix products and later drop 1/4 of values.
    // The only difference is that `zip` isn't provided for `bf16` and we need to reinterpret back
    // and forth before zipping. Same as with integers, on modern Arm CPUs, this "smart"
    // approach is actually slower by around 25%.
    //
    //   float32x4_t products_low_vec = vdupq_n_f32(0.0f);
    //   float32x4_t products_high_vec = vdupq_n_f32(0.0f);
    //   for (; i + 8 <= n; i += 8) {
    //       bfloat16x8_t a_vec = vld1q_bf16((nk_bf16_for_arm_simd_t const*)a + i);
    //       bfloat16x8_t b_vec = vld1q_bf16((nk_bf16_for_arm_simd_t const*)b + i);
    //       int16x8_t a_vec_s16 = vreinterpretq_s16_bf16(a_vec);
    //       int16x8_t b_vec_s16 = vreinterpretq_s16_bf16(b_vec);
    //       int16x8x2_t y_w_vecs_s16 = vzipq_s16(a_vec_s16, b_vec_s16);
    //       bfloat16x8_t y_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[0]);
    //       bfloat16x8_t w_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[1]);
    //       bfloat16x4_t a_low = vget_low_bf16(a_vec);
    //       bfloat16x4_t b_low = vget_low_bf16(b_vec);
    //       bfloat16x4_t a_high = vget_high_bf16(a_vec);
    //       bfloat16x4_t b_high = vget_high_bf16(b_vec);
    //       bfloat16x8_t x_vec = vcombine_bf16(a_low, b_low);
    //       bfloat16x8_t v_vec = vcombine_bf16(a_high, b_high);
    //       products_low_vec = vbfmmlaq_f32(products_low_vec, x_vec, y_vec);
    //       products_high_vec = vbfmmlaq_f32(products_high_vec, v_vec, w_vec);
    //   }
    //   float32x4_t products_vec = vaddq_f32(products_high_vec, products_low_vec);
    //   nk_f32_t a2 = products_vec[0], ab = products_vec[1], b2 = products_vec[3];
    //
    // Another way of accomplishing the same thing is to process the odd and even elements separately,
    // using special `vbfmlaltq_f32` and `vbfmlalbq_f32` intrinsics:
    //
    //      ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_vec, b_vec);
    //      ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_vec, b_vec);
    //      a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_vec, a_vec);
    //      a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_vec, a_vec);
    //      b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_vec, b_vec);
    //      b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_vec, b_vec);
    //

    float32x4_t dot_product_f32x4 = vdupq_n_f32(0);
    float32x4_t a_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t b_norm_sq_f32x4 = vdupq_n_f32(0);
    bfloat16x8_t a_bf16x8, b_bf16x8;

nk_angular_bf16_neonbfdot_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_neon_(a, n, &a_vec);
        nk_partial_load_b16x8_neon_(b, n, &b_vec);
        a_bf16x8 = vreinterpretq_bf16_u16(a_vec.u16x8);
        b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
        n = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)a);
        b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)b);
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x4 = vbfdotq_f32(dot_product_f32x4, a_bf16x8, b_bf16x8);
    a_norm_sq_f32x4 = vbfdotq_f32(a_norm_sq_f32x4, a_bf16x8, a_bf16x8);
    b_norm_sq_f32x4 = vbfdotq_f32(b_norm_sq_f32x4, b_bf16x8, b_bf16x8);
    if (n) goto nk_angular_bf16_neonbfdot_cycle;

    // Avoid `nk_f32_approximate_inverse_square_root` on Arm NEON
    nk_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    nk_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    nk_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = nk_angular_normalize_f32_neon_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2sq_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t diff_high_f32x4, diff_low_f32x4;
    float32x4_t distance_sq_high_f32x4 = vdupq_n_f32(0), distance_sq_low_f32x4 = vdupq_n_f32(0);

nk_l2sq_bf16_neonbfdot_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_neon_(a, n, &a_vec);
        nk_partial_load_b16x8_neon_(b, n, &b_vec);
        bfloat16x8_t a_bf16x8 = vreinterpretq_bf16_u16(a_vec.u16x8);
        bfloat16x8_t b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
        diff_high_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_bf16x8)), vcvt_f32_bf16(vget_high_bf16(b_bf16x8)));
        diff_low_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_bf16x8)), vcvt_f32_bf16(vget_low_bf16(b_bf16x8)));
        n = 0;
    }
    else {
        bfloat16x8_t a_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)a);
        bfloat16x8_t b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)b);
        diff_high_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_high_bf16(a_bf16x8)), vcvt_f32_bf16(vget_high_bf16(b_bf16x8)));
        diff_low_f32x4 = vsubq_f32(vcvt_f32_bf16(vget_low_bf16(a_bf16x8)), vcvt_f32_bf16(vget_low_bf16(b_bf16x8)));
        n -= 8, a += 8, b += 8;
    }
    distance_sq_high_f32x4 = vfmaq_f32(distance_sq_high_f32x4, diff_high_f32x4, diff_high_f32x4);
    distance_sq_low_f32x4 = vfmaq_f32(distance_sq_low_f32x4, diff_low_f32x4, diff_low_f32x4);
    if (n) goto nk_l2sq_bf16_neonbfdot_cycle;

    *result = vaddvq_f32(vaddq_f32(distance_sq_high_f32x4, distance_sq_low_f32x4));
}
NK_PUBLIC void nk_l2_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_bf16_neonbfdot(a, b, n, result);
    *result = nk_sqrt_f32_neon_(*result);
}

typedef nk_dot_bf16x8_state_neonbfdot_t nk_angular_bf16x8_state_neonbfdot_t;
NK_INTERNAL void nk_angular_bf16x8_init_neonbfdot(nk_angular_bf16x8_state_neonbfdot_t *state) {
    nk_dot_bf16x8_init_neonbfdot(state);
}
NK_INTERNAL void nk_angular_bf16x8_update_neonbfdot(nk_angular_bf16x8_state_neonbfdot_t *state, nk_b128_vec_t a,
                                                    nk_b128_vec_t b) {
    nk_dot_bf16x8_update_neonbfdot(state, a, b);
}
NK_INTERNAL void nk_angular_bf16x8_finalize_neonbfdot(nk_angular_bf16x8_state_neonbfdot_t const *state_a,
                                                      nk_angular_bf16x8_state_neonbfdot_t const *state_b,
                                                      nk_angular_bf16x8_state_neonbfdot_t const *state_c,
                                                      nk_angular_bf16x8_state_neonbfdot_t const *state_d,
                                                      nk_f32_t query_norm, nk_f32_t target_norm_a,
                                                      nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                      nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single call
    nk_f32_t dots[4];
    nk_dot_bf16x8_finalize_neonbfdot(state_a, state_b, state_c, state_d, dots);

    // Build F32 vectors for parallel processing
    float32x4_t dots_vec = vld1q_f32(dots);
    float32x4_t query_sq = vdupq_n_f32(query_norm * query_norm);
    float32x4_t target_norms = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t target_sq = vmulq_f32(target_norms, target_norms);
    float32x4_t products = vmulq_f32(query_sq, target_sq);

    // Vectorized rsqrt with Newton-Raphson
    float32x4_t rsqrt_vec = vrsqrteq_f32(products);
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));
    rsqrt_vec = vmulq_f32(rsqrt_vec, vrsqrtsq_f32(vmulq_f32(products, rsqrt_vec), rsqrt_vec));

    // Compute angular distance and handle edge cases
    float32x4_t ones = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t result_vec = vsubq_f32(ones, vmulq_f32(dots_vec, rsqrt_vec));
    result_vec = vmaxq_f32(result_vec, zeros);

    uint32x4_t products_zero = vceqq_f32(products, zeros);
    uint32x4_t dots_zero = vceqq_f32(dots_vec, zeros);
    uint32x4_t both_zero = vandq_u32(products_zero, dots_zero);
    result_vec = vbslq_f32(both_zero, zeros, result_vec);
    uint32x4_t prod_zero_dot_nonzero = vandq_u32(products_zero, vmvnq_u32(dots_zero));
    result_vec = vbslq_f32(prod_zero_dot_nonzero, ones, result_vec);

    vst1q_f32(results, result_vec);
}

typedef nk_dot_bf16x8_state_neonbfdot_t nk_l2_bf16x8_state_neonbfdot_t;
NK_INTERNAL void nk_l2_bf16x8_init_neonbfdot(nk_l2_bf16x8_state_neonbfdot_t *state) {
    nk_dot_bf16x8_init_neonbfdot(state);
}
NK_INTERNAL void nk_l2_bf16x8_update_neonbfdot(nk_l2_bf16x8_state_neonbfdot_t *state, nk_b128_vec_t a,
                                               nk_b128_vec_t b) {
    nk_dot_bf16x8_update_neonbfdot(state, a, b);
}
NK_INTERNAL void nk_l2_bf16x8_finalize_neonbfdot(nk_l2_bf16x8_state_neonbfdot_t const *state_a,
                                                 nk_l2_bf16x8_state_neonbfdot_t const *state_b,
                                                 nk_l2_bf16x8_state_neonbfdot_t const *state_c,
                                                 nk_l2_bf16x8_state_neonbfdot_t const *state_d, nk_f32_t query_norm,
                                                 nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                 nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products
    nk_f32_t dots[4];
    nk_dot_bf16x8_finalize_neonbfdot(state_a, state_b, state_c, state_d, dots);

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
#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEONBFDOT_H