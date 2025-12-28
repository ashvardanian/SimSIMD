/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/elementwise/neonhalf.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_NEONHALF_H
#define NK_ELEMENTWISE_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_vec = vld1q_f16((float16_t const *)b + i);
        float16x8_t sum_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = ((float16_t const *)a)[i] + ((float16_t const *)b)[i];
}

NK_PUBLIC void nk_scale_f16_neonhalf(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16;
}

NK_PUBLIC void nk_wsum_f16_neonhalf(                   //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f16_neonhalf(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f16_neonhalf(a, n, alpha, &zero, result); }
        else { nk_scale_f16_neonhalf(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_f16x8 = vld1q_f16((float16_t const *)b + i);
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16 * ((float16_t const *)b)[i];
}

NK_PUBLIC void nk_fma_f16_neonhalf(                          //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_f16x8 = vld1q_f16((float16_t const *)b + i);
        float16x8_t c_f16x8 = vld1q_f16((float16_t const *)c + i);
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        vst1q_f16((float16_t *)result + i, result_f16x8);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] * ((float16_t const *)b)[i] +
                                   beta_f16 * ((float16_t const *)c)[i];
}

NK_PUBLIC void nk_sum_u8_neonhalf(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        uint8x16_t sum_vec = vqaddq_u8(a_vec, b_vec);
        vst1q_u8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        nk_f32_to_u8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_u8_neonhalf(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16;
        nk_f32_to_u8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_u8_neonhalf(                  //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_u8_neonhalf(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_u8_neonhalf(a, n, alpha, &zero, result); }
        else { nk_scale_u8_neonhalf(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        uint8x8_t b_u8x8 = vld1_u8(b + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t b_f16x8 = vcvtq_f16_u16(vmovl_u8(b_u8x8));
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        nk_f32_to_u8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_u8_neonhalf(                        //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8x8 = vld1_u8(a + i);
        uint8x8_t b_u8x8 = vld1_u8(b + i);
        uint8x8_t c_u8x8 = vld1_u8(c + i);
        float16x8_t a_f16x8 = vcvtq_f16_u16(vmovl_u8(a_u8x8));
        float16x8_t b_f16x8 = vcvtq_f16_u16(vmovl_u8(b_u8x8));
        float16x8_t c_f16x8 = vcvtq_f16_u16(vmovl_u8(c_u8x8));
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        uint8x8_t result_u8x8 = vqmovn_u16(vcvtaq_u16_f16(result_f16x8));
        vst1_u8(result + i, result_u8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        nk_f32_to_u8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_sum_i8_neonhalf(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        int8x16_t sum_vec = vqaddq_s8(a_vec, b_vec);
        vst1q_s8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        nk_f32_to_i8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_i8_neonhalf(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16;
        nk_f32_to_i8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_i8_neonhalf(                  //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_i8_neonhalf(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_i8_neonhalf(a, n, alpha, &zero, result); }
        else { nk_scale_i8_neonhalf(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha_val;
    float16_t beta_f16 = (float16_t)beta_val;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        int8x8_t b_i8x8 = vld1_s8(b + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t b_f16x8 = vcvtq_f16_s16(vmovl_s8(b_i8x8));
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        nk_f32_to_i8_(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_i8_neonhalf(                        //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8x8 = vld1_s8(a + i);
        int8x8_t b_i8x8 = vld1_s8(b + i);
        int8x8_t c_i8x8 = vld1_s8(c + i);
        float16x8_t a_f16x8 = vcvtq_f16_s16(vmovl_s8(a_i8x8));
        float16x8_t b_f16x8 = vcvtq_f16_s16(vmovl_s8(b_i8x8));
        float16x8_t c_f16x8 = vcvtq_f16_s16(vmovl_s8(c_i8x8));
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t result_f16x8 = vfmaq_n_f16(ab_scaled_f16x8, c_f16x8, beta_f16);
        int8x8_t result_i8x8 = vqmovn_s16(vcvtaq_s16_f16(result_f16x8));
        vst1_s8(result + i, result_i8x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        nk_f32_to_i8_(&sum, result + i);
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_ELEMENTWISE_NEONHALF_H