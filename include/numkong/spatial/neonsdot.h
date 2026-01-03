/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/spatial/neonsdot.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_NEONSDOT_H
#define NK_SPATIAL_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // nk_angular_f32x4_finalize_neon_f32_, nk_l2_f32x4_finalize_neon_f32_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {

    // The naive approach is to upcast 8-bit signed integers into 16-bit signed integers
    // for subtraction, then multiply within 16-bit integers and accumulate the results
    // into 32-bit integers. This approach is slow on modern Arm CPUs. On Graviton 4,
    // that approach results in 17 GB/s of throughput, compared to 39 GB/s for `i8`
    // dot-products.
    //
    // Luckily we can use the `vabdq_s8` which technically returns `i8` values, but it's a
    // matter of reinterpret-casting! That approach boosts us to 33 GB/s of throughput.
    uint32x4_t distance_sq_u32x4 = vdupq_n_u32(0);
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a + i);
        int8x16_t b_i8x16 = vld1q_s8(b + i);
        uint8x16_t diff_u8x16 = vreinterpretq_u8_s8(vabdq_s8(a_i8x16, b_i8x16));
        distance_sq_u32x4 = vdotq_u32(distance_sq_u32x4, diff_u8x16, diff_u8x16);
    }
    nk_u32_t distance_sq_u32 = vaddvq_u32(distance_sq_u32x4);
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)a[i] - b[i];
        distance_sq_u32 += (nk_u32_t)(diff_i32 * diff_i32);
    }
    *result = distance_sq_u32;
}
NK_PUBLIC void nk_l2_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_l2sq_i8_neonsdot(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_neon((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_i8_neonsdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    nk_size_t i = 0;

    // Variant 1.
    // If the 128-bit `vdot_s32` intrinsic is unavailable, we can use the 64-bit `vdot_s32`.
    //
    //  int32x4_t ab_vec = vdupq_n_s32(0);
    //  int32x4_t a2_vec = vdupq_n_s32(0);
    //  int32x4_t b2_vec = vdupq_n_s32(0);
    //  for (nk_size_t i = 0; i != n; i += 8) {
    //      int16x8_t a_vec = vmovl_s8(vld1_s8(a + i));
    //      int16x8_t b_vec = vmovl_s8(vld1_s8(b + i));
    //      int16x8_t ab_part_vec = vmulq_s16(a_vec, b_vec);
    //      int16x8_t a2_part_vec = vmulq_s16(a_vec, a_vec);
    //      int16x8_t b2_part_vec = vmulq_s16(b_vec, b_vec);
    //      ab_vec = vaddq_s32(ab_vec, vaddq_s32(vmovl_s16(vget_high_s16(ab_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(ab_part_vec))));
    //      a2_vec = vaddq_s32(a2_vec, vaddq_s32(vmovl_s16(vget_high_s16(a2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(a2_part_vec))));
    //      b2_vec = vaddq_s32(b2_vec, vaddq_s32(vmovl_s16(vget_high_s16(b2_part_vec)), //
    //                                           vmovl_s16(vget_low_s16(b2_part_vec))));
    //  }
    //
    // Variant 2.
    // With the 128-bit `vdotq_s32` intrinsic, we can use the following code:
    //
    //  for (; i + 16 <= n; i += 16) {
    //      int8x16_t a_vec = vld1q_s8(a + i);
    //      int8x16_t b_vec = vld1q_s8(b + i);
    //      ab_vec = vdotq_s32(ab_vec, a_vec, b_vec);
    //      a2_vec = vdotq_s32(a2_vec, a_vec, a_vec);
    //      b2_vec = vdotq_s32(b2_vec, b_vec, b_vec);
    //  }
    //
    // Variant 3.
    // To use MMLA instructions, we need to reorganize the contents of the vectors.
    // On input we have `a_vec` and `b_vec`:
    //
    //   a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]
    //   b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //
    // We will be multiplying matrices of size 2x8 and 8x2. So we need to perform a few shuffles:
    //
    //   X =
    //      a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    //      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    //   Y =
    //      a[0], b[0],
    //      a[1], b[1],
    //      a[2], b[2],
    //      a[3], b[3],
    //      a[4], b[4],
    //      a[5], b[5],
    //      a[6], b[6],
    //      a[7], b[7]
    //
    //   V =
    //      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
    //      b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    //   W =
    //      a[8],   b[8],
    //      a[9],   b[9],
    //      a[10],  b[10],
    //      a[11],  b[11],
    //      a[12],  b[12],
    //      a[13],  b[13],
    //      a[14],  b[14],
    //      a[15],  b[15]
    //
    // Performing matrix multiplications we can aggregate into a matrix `products_low_vec` and `products_high_vec`:
    //
    //      X * X, X * Y                V * W, V * V
    //      Y * X, Y * Y                W * W, W * V
    //
    // Of those values we need only 3/4, as the (X * Y) and (Y * X) are the same.
    //
    //      int32x4_t products_low_vec = vdupq_n_s32(0), products_high_vec = vdupq_n_s32(0);
    //      int8x16_t a_low_b_low_vec, a_high_b_high_vec;
    //      for (; i + 16 <= n; i += 16) {
    //          int8x16_t a_vec = vld1q_s8(a + i);
    //          int8x16_t b_vec = vld1q_s8(b + i);
    //          int8x16x2_t y_w_vecs = vzipq_s8(a_vec, b_vec);
    //          int8x16_t x_vec = vcombine_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
    //          int8x16_t v_vec = vcombine_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
    //          products_low_vec = vmmlaq_s32(products_low_vec, x_vec, y_w_vecs.val[0]);
    //          products_high_vec = vmmlaq_s32(products_high_vec, v_vec, y_w_vecs.val[1]);
    //      }
    //      int32x4_t products_vec = vaddq_s32(products_high_vec, products_low_vec);
    //      nk_i32_t a2 = products_vec[0];
    //      nk_i32_t ab = products_vec[1];
    //      nk_i32_t b2 = products_vec[3];
    //
    // That solution is elegant, but it requires the additional `+i8mm` extension and is currently slower,
    // at least on AWS Graviton 3.
    int32x4_t dot_product_i32x4 = vdupq_n_s32(0);
    int32x4_t a_norm_sq_i32x4 = vdupq_n_s32(0);
    int32x4_t b_norm_sq_i32x4 = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a + i);
        int8x16_t b_i8x16 = vld1q_s8(b + i);
        dot_product_i32x4 = vdotq_s32(dot_product_i32x4, a_i8x16, b_i8x16);
        a_norm_sq_i32x4 = vdotq_s32(a_norm_sq_i32x4, a_i8x16, a_i8x16);
        b_norm_sq_i32x4 = vdotq_s32(b_norm_sq_i32x4, b_i8x16, b_i8x16);
    }
    nk_i32_t dot_product_i32 = vaddvq_s32(dot_product_i32x4);
    nk_i32_t a_norm_sq_i32 = vaddvq_s32(a_norm_sq_i32x4);
    nk_i32_t b_norm_sq_i32 = vaddvq_s32(b_norm_sq_i32x4);

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_neon_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_l2sq_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    uint32x4_t distance_sq_u32x4 = vdupq_n_u32(0);
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a + i);
        uint8x16_t b_u8x16 = vld1q_u8(b + i);
        uint8x16_t diff_u8x16 = vabdq_u8(a_u8x16, b_u8x16);
        distance_sq_u32x4 = vdotq_u32(distance_sq_u32x4, diff_u8x16, diff_u8x16);
    }
    nk_u32_t distance_sq_u32 = vaddvq_u32(distance_sq_u32x4);
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)a[i] - b[i];
        distance_sq_u32 += (nk_u32_t)(diff_i32 * diff_i32);
    }
    *result = distance_sq_u32;
}
NK_PUBLIC void nk_l2_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u8_neonsdot(a, b, n, &d2);
    *result = nk_f32_sqrt_neon((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_u8_neonsdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {

    nk_size_t i = 0;
    uint32x4_t ab_vec = vdupq_n_u32(0);
    uint32x4_t a2_vec = vdupq_n_u32(0);
    uint32x4_t b2_vec = vdupq_n_u32(0);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        ab_vec = vdotq_u32(ab_vec, a_vec, b_vec);
        a2_vec = vdotq_u32(a2_vec, a_vec, a_vec);
        b2_vec = vdotq_u32(b2_vec, b_vec, b_vec);
    }
    nk_u32_t ab = vaddvq_u32(ab_vec);
    nk_u32_t a2 = vaddvq_u32(a2_vec);
    nk_u32_t b2 = vaddvq_u32(b2_vec);

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_u32_t ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

typedef nk_dot_i8x16_state_neonsdot_t nk_angular_i8x16_state_neonsdot_t;
NK_INTERNAL void nk_angular_i8x16_init_neonsdot(nk_angular_i8x16_state_neonsdot_t *state) {
    nk_dot_i8x16_init_neonsdot(state);
}
NK_INTERNAL void nk_angular_i8x16_update_neonsdot(nk_angular_i8x16_state_neonsdot_t *state, nk_b128_vec_t a,
                                                  nk_b128_vec_t b) {
    nk_dot_i8x16_update_neonsdot(state, a, b);
}
NK_INTERNAL void nk_angular_i8x16_finalize_neonsdot(nk_angular_i8x16_state_neonsdot_t const *state_a,
                                                    nk_angular_i8x16_state_neonsdot_t const *state_b,
                                                    nk_angular_i8x16_state_neonsdot_t const *state_c,
                                                    nk_angular_i8x16_state_neonsdot_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x16_finalize_neonsdot(state_a, state_b, state_c, state_d, dots_vec.i32s);
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots_vec.i32x4);
    nk_angular_f32x4_finalize_neon_f32_(dots_f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                        target_norm_d, results);
}

typedef nk_dot_i8x16_state_neonsdot_t nk_l2_i8x16_state_neonsdot_t;
NK_INTERNAL void nk_l2_i8x16_init_neonsdot(nk_l2_i8x16_state_neonsdot_t *state) { nk_dot_i8x16_init_neonsdot(state); }
NK_INTERNAL void nk_l2_i8x16_update_neonsdot(nk_l2_i8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_i8x16_update_neonsdot(state, a, b);
}
NK_INTERNAL void nk_l2_i8x16_finalize_neonsdot(nk_l2_i8x16_state_neonsdot_t const *state_a,
                                               nk_l2_i8x16_state_neonsdot_t const *state_b,
                                               nk_l2_i8x16_state_neonsdot_t const *state_c,
                                               nk_l2_i8x16_state_neonsdot_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x16_finalize_neonsdot(state_a, state_b, state_c, state_d, dots_vec.i32s);
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots_vec.i32x4);
    nk_l2_f32x4_finalize_neon_f32_(dots_f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c, target_norm_d,
                                   results);
}

typedef nk_dot_u8x16_state_neonsdot_t nk_angular_u8x16_state_neonsdot_t;
NK_INTERNAL void nk_angular_u8x16_init_neonsdot(nk_angular_u8x16_state_neonsdot_t *state) {
    nk_dot_u8x16_init_neonsdot(state);
}
NK_INTERNAL void nk_angular_u8x16_update_neonsdot(nk_angular_u8x16_state_neonsdot_t *state, nk_b128_vec_t a,
                                                  nk_b128_vec_t b) {
    nk_dot_u8x16_update_neonsdot(state, a, b);
}
NK_INTERNAL void nk_angular_u8x16_finalize_neonsdot(nk_angular_u8x16_state_neonsdot_t const *state_a,
                                                    nk_angular_u8x16_state_neonsdot_t const *state_b,
                                                    nk_angular_u8x16_state_neonsdot_t const *state_c,
                                                    nk_angular_u8x16_state_neonsdot_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_u8x16_finalize_neonsdot(state_a, state_b, state_c, state_d, dots_vec.u32s);
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots_vec.u32x4);
    nk_angular_f32x4_finalize_neon_f32_(dots_f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                        target_norm_d, results);
}

typedef nk_dot_u8x16_state_neonsdot_t nk_l2_u8x16_state_neonsdot_t;
NK_INTERNAL void nk_l2_u8x16_init_neonsdot(nk_l2_u8x16_state_neonsdot_t *state) { nk_dot_u8x16_init_neonsdot(state); }
NK_INTERNAL void nk_l2_u8x16_update_neonsdot(nk_l2_u8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_u8x16_update_neonsdot(state, a, b);
}
NK_INTERNAL void nk_l2_u8x16_finalize_neonsdot(nk_l2_u8x16_state_neonsdot_t const *state_a,
                                               nk_l2_u8x16_state_neonsdot_t const *state_b,
                                               nk_l2_u8x16_state_neonsdot_t const *state_c,
                                               nk_l2_u8x16_state_neonsdot_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_u8x16_finalize_neonsdot(state_a, state_b, state_c, state_d, dots_vec.u32s);
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots_vec.u32x4);
    nk_l2_f32x4_finalize_neon_f32_(dots_f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c, target_norm_d,
                                   results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEONSDOT_H