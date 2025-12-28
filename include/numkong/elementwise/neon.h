/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/elementwise/neon.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_NEON_H
#define NK_ELEMENTWISE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_scale_f32_neon(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_val);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t result_f32x4 = vfmaq_n_f32(beta_f32x4, a_f32x4, alpha_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_wsum_f32_neon(                       //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f32_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f32_neon(a, n, alpha, &zero, result); }
        else { nk_scale_f32_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_fma_f32_neon(                              //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t c_f32x4 = vld1q_f32(c + i);
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_sum_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t a_vec = vld1q_s16(a + i);
        int16x8_t b_vec = vld1q_s16(b + i);
        int16x8_t sum_vec = vqaddq_s16(a_vec, b_vec);
        vst1q_s16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_i16_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i16_neon(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtaq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        nk_f32_to_i16_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i16_neon(                              //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        int16x4_t b_i16x4 = vld1_s16(b + i);
        int16x4_t c_i16x4 = vld1_s16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t b_f32x4 = vcvtq_f32_s32(vmovl_s16(b_i16x4));
        float32x4_t c_f32x4 = vcvtq_f32_s32(vmovl_s16(c_i16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtaq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        nk_f32_to_i16_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_vec = vld1q_u16(a + i);
        uint16x8_t b_vec = vld1q_u16(b + i);
        uint16x8_t sum_vec = vqaddq_u16(a_vec, b_vec);
        vst1q_u16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_u16_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u16_neon(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                 nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtaq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        nk_f32_to_u16_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u16_neon(                              //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        uint16x4_t b_u16x4 = vld1_u16(b + i);
        uint16x4_t c_u16x4 = vld1_u16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t b_f32x4 = vcvtq_f32_u32(vmovl_u16(b_u16x4));
        float32x4_t c_f32x4 = vcvtq_f32_u32(vmovl_u16(c_u16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtaq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        nk_f32_to_u16_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t a_vec = vld1q_s32(a + i);
        int32x4_t b_vec = vld1q_s32(b + i);
        int32x4_t sum_vec = vqaddq_s32(a_vec, b_vec);
        vst1q_s32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_i32_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i32_neon(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtaq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_i32_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i32_neon(                              //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        int32x2_t b_i32x2 = vld1_s32(b + i);
        int32x2_t c_i32x2 = vld1_s32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t b_f64x2 = vcvtq_f64_s64(vmovl_s32(b_i32x2));
        float64x2_t c_f64x2 = vcvtq_f64_s64(vmovl_s32(c_i32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtaq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_i32_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_vec = vld1q_u32(a + i);
        uint32x4_t b_vec = vld1q_u32(b + i);
        uint32x4_t sum_vec = vqaddq_u32(a_vec, b_vec);
        vst1q_u32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_u32_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u32_neon(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtaq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_u32_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u32_neon(                              //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        uint32x2_t b_u32x2 = vld1_u32(b + i);
        uint32x2_t c_u32x2 = vld1_u32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t b_f64x2 = vcvtq_f64_u64(vmovl_u32(b_u32x2));
        float64x2_t c_f64x2 = vcvtq_f64_u64(vmovl_u32(c_u32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtaq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_u32_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_vec = vld1q_s64(a + i);
        int64x2_t b_vec = vld1q_s64(b + i);
        int64x2_t sum_vec = vqaddq_s64(a_vec, b_vec);
        vst1q_s64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_i64_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_i64_neon(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        int64x2_t result_i64x2 = vcvtaq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_i64_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i64_neon(                              //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        int64x2_t b_i64x2 = vld1q_s64(b + i);
        int64x2_t c_i64x2 = vld1q_s64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t b_f64x2 = vcvtq_f64_s64(b_i64x2);
        float64x2_t c_f64x2 = vcvtq_f64_s64(c_i64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        int64x2_t result_i64x2 = vcvtaq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_i64_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);
        uint64x2_t sum_vec = vqaddq_u64(a_vec, b_vec);
        vst1q_u64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) nk_u64_sadd_(a + i, b + i, result + i);
}

NK_PUBLIC void nk_scale_u64_neon(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        uint64x2_t result_u64x2 = vcvtaq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_u64_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u64_neon(                              //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        uint64x2_t b_u64x2 = vld1q_u64(b + i);
        uint64x2_t c_u64x2 = vld1q_u64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t b_f64x2 = vcvtq_f64_u64(b_u64x2);
        float64x2_t c_f64x2 = vcvtq_f64_u64(c_u64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        uint64x2_t result_u64x2 = vcvtaq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_u64_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_vec = vld1q_f64(a + i);
        float64x2_t b_vec = vld1q_f64(b + i);
        float64x2_t sum_vec = vaddq_f64(a_vec, b_vec);
        vst1q_f64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_scale_f64_neon(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                 nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_wsum_f64_neon(                       //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {

    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f64_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_scale_f64_neon(a, n, alpha, &zero, result); }
        else { nk_scale_f64_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t a_scaled_f64x2 = vmulq_n_f64(a_f64x2, alpha_val);
        float64x2_t b_scaled_f64x2 = vmulq_n_f64(b_f64x2, beta_val);
        float64x2_t result_f64x2 = vaddq_f64(a_scaled_f64x2, b_scaled_f64x2);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_fma_f64_neon(                              //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t c_f64x2 = vld1q_f64(c + i);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_ELEMENTWISE_NEON_H