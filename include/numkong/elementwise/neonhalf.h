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
#include "numkong/cast/serial.h" // nk_f32_to_i8_serial

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

    // There are several special cases we may want to implement:
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
        nk_f32_to_u8_serial(&sum, result + i);
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
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_u8_neonhalf(                  //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
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
        nk_f32_to_u8_serial(&sum, result + i);
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
        nk_f32_to_u8_serial(&sum, result + i);
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
        nk_f32_to_i8_serial(&sum, result + i);
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
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_i8_neonhalf(                  //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
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
        nk_f32_to_i8_serial(&sum, result + i);
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
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

/** @brief Convert 8 E4M3 values to f16x8 via bit manipulation (NEON).
 *  E4M3 format: S EEEE MMM (bias=7). F16: sign<<15, (exp+8)<<10, mant<<7. */
NK_INTERNAL float16x8_t nk_e4m3x8_to_f16x8_neonhalf_(uint8x8_t e4m3_u8x8) {
    // Widen to 16-bit lanes
    uint16x8_t v_u16x8 = vmovl_u8(e4m3_u8x8);

    // Extract sign, exponent, and mantissa
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 3), vdupq_n_u16(0x0F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x07));

    // Build f16 representation: sign | ((exp + 8) << 10) | (mant << 7)
    // F16 has bias=15, E4M3 has bias=7, so we add (15-7)=8 to convert exponent
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(8)), 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 7);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));

    // Zero out denormals (when exp == 0)
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);

    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

/** @brief Convert f16x8 to 8 E4M3 values via bit manipulation (NEON).
 *  F16: S EEEEE MMMMMMMMMM (bias=15). E4M3: sign<<7, (exp-8)<<3, mant>>7. */
NK_INTERNAL uint8x8_t nk_f16x8_to_e4m3x8_neonhalf_(float16x8_t f16_f16x8) {
    uint16x8_t v_u16x8 = vreinterpretq_u16_f16(f16_f16x8);

    // Extract sign, exponent, and mantissa from f16
    uint16x8_t sign_u16x8 = vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x8000)), 8);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x03FF));

    // Convert exponent: E4M3 has bias=7, F16 has bias=15, so subtract 8
    // Clamp exponent to [0, 15] for E4M3 range
    uint16x8_t exp_adj_u16x8 = vqsubq_u16(exp_u16x8, vdupq_n_u16(8));
    exp_adj_u16x8 = vminq_u16(exp_adj_u16x8, vdupq_n_u16(15));

    // Build E4M3: sign | (exp << 3) | (mant >> 7)
    uint16x8_t e4m3_exp_u16x8 = vshlq_n_u16(exp_adj_u16x8, 3);
    uint16x8_t e4m3_mant_u16x8 = vshrq_n_u16(mant_u16x8, 7);
    uint16x8_t e4m3_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(e4m3_exp_u16x8, e4m3_mant_u16x8));

    // Zero out when original exponent is too small (underflow to zero)
    uint16x8_t underflow_mask_u16x8 = vcltq_u16(exp_u16x8, vdupq_n_u16(8));
    e4m3_u16x8 = vbicq_u16(e4m3_u16x8, underflow_mask_u16x8);

    // Narrow to 8-bit
    return vmovn_u16(e4m3_u16x8);
}

/** @brief Convert 8 E5M2 values to f16x8 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). F16: sign<<15, exp<<10, mant<<8. */
NK_INTERNAL float16x8_t nk_e5m2x8_to_f16x8_neonhalf_(uint8x8_t e5m2_u8x8) {
    // Widen to 16-bit lanes
    uint16x8_t v_u16x8 = vmovl_u8(e5m2_u8x8);

    // Extract sign, exponent, and mantissa
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 2), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x03));

    // Build f16 representation: sign | (exp << 10) | (mant << 8)
    // F16 has bias=15, E5M2 has bias=15, so exponent stays the same
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(exp_u16x8, 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 8);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));

    // Zero out denormals (when exp == 0)
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);

    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

/** @brief Convert f16x8 to 8 E5M2 values via bit manipulation (NEON).
 *  F16: S EEEEE MMMMMMMMMM (bias=15). E5M2: sign<<7, exp<<2, mant>>8. */
NK_INTERNAL uint8x8_t nk_f16x8_to_e5m2x8_neonhalf_(float16x8_t f16_f16x8) {
    uint16x8_t v_u16x8 = vreinterpretq_u16_f16(f16_f16x8);

    // Extract sign, exponent, and mantissa from f16
    uint16x8_t sign_u16x8 = vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x8000)), 8);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x03FF));

    // Build E5M2: sign | (exp << 2) | (mant >> 8)
    // E5M2 has same bias as F16 (15), so exponent stays the same
    uint16x8_t e5m2_exp_u16x8 = vshlq_n_u16(exp_u16x8, 2);
    uint16x8_t e5m2_mant_u16x8 = vshrq_n_u16(mant_u16x8, 8);
    uint16x8_t e5m2_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(e5m2_exp_u16x8, e5m2_mant_u16x8));

    // Zero out denormals (when exp == 0)
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    e5m2_u16x8 = vbicq_u16(e5m2_u16x8, zero_mask_u16x8);

    // Narrow to 8-bit
    return vmovn_u16(e5m2_u16x8);
}

/*  E4M3 elementwise operations  */

NK_PUBLIC void nk_sum_e4m3_neonhalf(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e4m3x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e4m3x8 = vld1_u8((uint8_t const *)b + i);
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(a_e4m3x8);
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(b_e4m3x8);
        float16x8_t result_f16x8 = vaddq_f16(a_f16x8, b_f16x8);
        uint8x8_t result_e4m3x8 = nk_f16x8_to_e4m3x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, sum_f32;
        nk_e4m3_to_f32_serial(a + i, &a_f32);
        nk_e4m3_to_f32_serial(b + i, &b_f32);
        sum_f32 = a_f32 + b_f32;
        nk_f32_to_e4m3_serial(&sum_f32, result + i);
    }
}

NK_PUBLIC void nk_sum_e5m2_neonhalf(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e5m2x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e5m2x8 = vld1_u8((uint8_t const *)b + i);
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(a_e5m2x8);
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(b_e5m2x8);
        float16x8_t result_f16x8 = vaddq_f16(a_f16x8, b_f16x8);
        uint8x8_t result_e5m2x8 = nk_f16x8_to_e5m2x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, sum_f32;
        nk_e5m2_to_f32_serial(a + i, &a_f32);
        nk_e5m2_to_f32_serial(b + i, &b_f32);
        sum_f32 = a_f32 + b_f32;
        nk_f32_to_e5m2_serial(&sum_f32, result + i);
    }
}

NK_PUBLIC void nk_scale_e4m3_neonhalf(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_e4m3_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e4m3x8 = vld1_u8((uint8_t const *)a + i);
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(a_e4m3x8);
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        uint8x8_t result_e4m3x8 = nk_f16x8_to_e4m3x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, scaled_f32;
        nk_e4m3_to_f32_serial(a + i, &a_f32);
        scaled_f32 = *alpha * a_f32 + *beta;
        nk_f32_to_e4m3_serial(&scaled_f32, result + i);
    }
}

NK_PUBLIC void nk_scale_e5m2_neonhalf(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_e5m2_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    float16x8_t alpha_f16x8 = vdupq_n_f16(alpha_f16);
    float16x8_t beta_f16x8 = vdupq_n_f16(beta_f16);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e5m2x8 = vld1_u8((uint8_t const *)a + i);
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(a_e5m2x8);
        float16x8_t result_f16x8 = vfmaq_f16(beta_f16x8, a_f16x8, alpha_f16x8);
        uint8x8_t result_e5m2x8 = nk_f16x8_to_e5m2x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, scaled_f32;
        nk_e5m2_to_f32_serial(a + i, &a_f32);
        scaled_f32 = *alpha * a_f32 + *beta;
        nk_f32_to_e5m2_serial(&scaled_f32, result + i);
    }
}

NK_PUBLIC void nk_wsum_e4m3_neonhalf(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                     nk_f32_t const *beta, nk_e4m3_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e4m3x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e4m3x8 = vld1_u8((uint8_t const *)b + i);
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(a_e4m3x8);
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(b_e4m3x8);
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        uint8x8_t result_e4m3x8 = nk_f16x8_to_e4m3x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, wsum_f32;
        nk_e4m3_to_f32_serial(a + i, &a_f32);
        nk_e4m3_to_f32_serial(b + i, &b_f32);
        wsum_f32 = *alpha * a_f32 + *beta * b_f32;
        nk_f32_to_e4m3_serial(&wsum_f32, result + i);
    }
}

NK_PUBLIC void nk_wsum_e5m2_neonhalf(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                     nk_f32_t const *beta, nk_e5m2_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e5m2x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e5m2x8 = vld1_u8((uint8_t const *)b + i);
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(a_e5m2x8);
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(b_e5m2x8);
        float16x8_t a_scaled_f16x8 = vmulq_n_f16(a_f16x8, alpha_f16);
        float16x8_t b_scaled_f16x8 = vmulq_n_f16(b_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(a_scaled_f16x8, b_scaled_f16x8);
        uint8x8_t result_e5m2x8 = nk_f16x8_to_e5m2x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, wsum_f32;
        nk_e5m2_to_f32_serial(a + i, &a_f32);
        nk_e5m2_to_f32_serial(b + i, &b_f32);
        wsum_f32 = *alpha * a_f32 + *beta * b_f32;
        nk_f32_to_e5m2_serial(&wsum_f32, result + i);
    }
}

NK_PUBLIC void nk_fma_e4m3_neonhalf(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e4m3x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e4m3x8 = vld1_u8((uint8_t const *)b + i);
        uint8x8_t c_e4m3x8 = vld1_u8((uint8_t const *)c + i);
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(a_e4m3x8);
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(b_e4m3x8);
        float16x8_t c_f16x8 = nk_e4m3x8_to_f16x8_neonhalf_(c_e4m3x8);
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t c_scaled_f16x8 = vmulq_n_f16(c_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(ab_scaled_f16x8, c_scaled_f16x8);
        uint8x8_t result_e4m3x8 = nk_f16x8_to_e4m3x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, c_f32, fma_f32;
        nk_e4m3_to_f32_serial(a + i, &a_f32);
        nk_e4m3_to_f32_serial(b + i, &b_f32);
        nk_e4m3_to_f32_serial(c + i, &c_f32);
        fma_f32 = *alpha * a_f32 * b_f32 + *beta * c_f32;
        nk_f32_to_e4m3_serial(&fma_f32, result + i);
    }
}

NK_PUBLIC void nk_fma_e5m2_neonhalf(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result) {
    float16_t alpha_f16 = (float16_t)*alpha;
    float16_t beta_f16 = (float16_t)*beta;
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_e5m2x8 = vld1_u8((uint8_t const *)a + i);
        uint8x8_t b_e5m2x8 = vld1_u8((uint8_t const *)b + i);
        uint8x8_t c_e5m2x8 = vld1_u8((uint8_t const *)c + i);
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(a_e5m2x8);
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(b_e5m2x8);
        float16x8_t c_f16x8 = nk_e5m2x8_to_f16x8_neonhalf_(c_e5m2x8);
        float16x8_t ab_f16x8 = vmulq_f16(a_f16x8, b_f16x8);
        float16x8_t ab_scaled_f16x8 = vmulq_n_f16(ab_f16x8, alpha_f16);
        float16x8_t c_scaled_f16x8 = vmulq_n_f16(c_f16x8, beta_f16);
        float16x8_t result_f16x8 = vaddq_f16(ab_scaled_f16x8, c_scaled_f16x8);
        uint8x8_t result_e5m2x8 = nk_f16x8_to_e5m2x8_neonhalf_(result_f16x8);
        vst1_u8((uint8_t *)result + i, result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t a_f32, b_f32, c_f32, fma_f32;
        nk_e5m2_to_f32_serial(a + i, &a_f32);
        nk_e5m2_to_f32_serial(b + i, &b_f32);
        nk_e5m2_to_f32_serial(c + i, &c_f32);
        fma_f32 = *alpha * a_f32 * b_f32 + *beta * c_f32;
        nk_f32_to_e5m2_serial(&fma_f32, result + i);
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
