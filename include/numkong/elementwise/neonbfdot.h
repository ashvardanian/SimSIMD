/**
 *  @brief SIMD-accelerated Elementwise Operations using BF16 for Arm NEON-capable CPUs.
 *  @file include/numkong/elementwise/neonbfdot.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_NEONBFDOT_H
#define NK_ELEMENTWISE_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_scale_bf16_neonbfdot(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                       nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_val);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_bf16_to_f32(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_wsum_bf16_neonbfdot(                   //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_bf16_neonbfdot(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_bf16_neonbfdot(a, n, alpha, &zero, result); }
        else { nk_scale_bf16_neonbfdot(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_bf16(&sum, result + i);
    }
}

NK_PUBLIC void nk_fma_bf16_neonbfdot(                           //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t c_f32x4 = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)c + i));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(result_f32x4));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_bf16_to_f32(a + i, &ai);
        nk_bf16_to_f32(b + i, &bi);
        nk_bf16_to_f32(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_bf16(&sum, result + i);
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_

#endif // NK_ELEMENTWISE_NEONBFDOT_H
