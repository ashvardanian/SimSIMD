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
#include "numkong/spatial/neon.h"    // nk_angular_f32x4_finalize_neon_, nk_l2_f32x4_finalize_neon_
#include "numkong/dot/neonhalf.h"    // nk_dot_f16x4_state_neonhalf_t

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
    *result = nk_f32_sqrt_neon(*result);
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

typedef nk_dot_f16x4_state_neonhalf_t nk_angular_f16x4_state_neonhalf_t;
NK_INTERNAL void nk_angular_f16x4_init_neonhalf(nk_angular_f16x4_state_neonhalf_t *state) {
    nk_dot_f16x4_init_neonhalf(state);
}
NK_INTERNAL void nk_angular_f16x4_update_neonhalf(nk_angular_f16x4_state_neonhalf_t *state, nk_b64_vec_t a,
                                                  nk_b64_vec_t b) {
    nk_dot_f16x4_update_neonhalf(state, a, b);
}
NK_INTERNAL void nk_angular_f16x4_finalize_neonhalf(nk_angular_f16x4_state_neonhalf_t const *state_a,
                                                    nk_angular_f16x4_state_neonhalf_t const *state_b,
                                                    nk_angular_f16x4_state_neonhalf_t const *state_c,
                                                    nk_angular_f16x4_state_neonhalf_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_f16x4_finalize_neonhalf(state_a, state_b, state_c, state_d, &dots_vec);
    nk_angular_f32x4_finalize_neon_f32_(dots_vec.f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                        target_norm_d, results);
}

typedef nk_dot_f16x4_state_neonhalf_t nk_l2_f16x4_state_neonhalf_t;
NK_INTERNAL void nk_l2_f16x4_init_neonhalf(nk_l2_f16x4_state_neonhalf_t *state) { nk_dot_f16x4_init_neonhalf(state); }
NK_INTERNAL void nk_l2_f16x4_update_neonhalf(nk_l2_f16x4_state_neonhalf_t *state, nk_b64_vec_t a, nk_b64_vec_t b) {
    nk_dot_f16x4_update_neonhalf(state, a, b);
}
NK_INTERNAL void nk_l2_f16x4_finalize_neonhalf(nk_l2_f16x4_state_neonhalf_t const *state_a,
                                               nk_l2_f16x4_state_neonhalf_t const *state_b,
                                               nk_l2_f16x4_state_neonhalf_t const *state_c,
                                               nk_l2_f16x4_state_neonhalf_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_f16x4_finalize_neonhalf(state_a, state_b, state_c, state_d, &dots_vec);
    nk_l2_f32x4_finalize_neon_f32_(dots_vec.f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                   target_norm_d, results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEONHALF_H
