/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neon.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_NEON_H
#define NK_DOT_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f32_neon(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    // Upcast f32 to f64 for accumulation (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count_scalars; idx_scalars += 2) {
        float32x2_t a_f32x2 = vld1_f32(a_scalars + idx_scalars);
        float32x2_t b_f32x2 = vld1_f32(b_scalars + idx_scalars);
        float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
        float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
        sum_f64x2 = vfmaq_f64(sum_f64x2, a_f64x2, b_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(sum_f64x2);
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum_f64 += (nk_f64_t)a_scalars[idx_scalars] * (nk_f64_t)b_scalars[idx_scalars];
    *result = (nk_f32_t)sum_f64;
}

NK_PUBLIC void nk_dot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f32c_t *result) {
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute the dot product: real = a_r*b_r - a_i*b_i, imag = a_r*b_i + a_i*b_r
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmsq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br - ai * bi;
        sum_imag_f64 += ar * bi + ai * br;
    }
    result->real = (nk_f32_t)sum_real_f64;
    result->imag = (nk_f32_t)sum_imag_f64;
}

NK_PUBLIC void nk_vdot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f32c_t *result) {
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute conjugate dot product: real = a_r*b_r + a_i*b_i, imag = a_r*b_i - a_i*b_r
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmsq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br + ai * bi;
        sum_imag_f64 += ar * bi - ai * br;
    }
    result->real = (nk_f32_t)sum_real_f64;
    result->imag = (nk_f32_t)sum_imag_f64;
}

/**
 *  @brief Running state for 64-bit dot accumulation over f32 scalars on NEON.
 *
 *  Processes 2 f32 values at a time, upcasting to f64 for accumulation to avoid
 *  catastrophic cancellation in long reductions.
 */
typedef struct nk_dot_f32x2_state_neon_t {
    float64x2_t sum_f64x2;
} nk_dot_f32x2_state_neon_t;

NK_INTERNAL void nk_dot_f32x2_init_neon(nk_dot_f32x2_state_neon_t *state) { state->sum_f64x2 = vdupq_n_f64(0); }

NK_INTERNAL void nk_dot_f32x2_update_neon(nk_dot_f32x2_state_neon_t *state, nk_b64_vec_t a, nk_b64_vec_t b) {
    // Upcast 2 f32s to f64s for high-precision accumulation
    float32x2_t a_f32x2 = vreinterpret_f32_u32(a.u32x2);
    float32x2_t b_f32x2 = vreinterpret_f32_u32(b.u32x2);
    float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
    float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
    state->sum_f64x2 = vfmaq_f64(state->sum_f64x2, a_f64x2, b_f64x2);
}

NK_INTERNAL void nk_dot_f32x2_finalize_neon(                                            //
    nk_dot_f32x2_state_neon_t const *state_a, nk_dot_f32x2_state_neon_t const *state_b, //
    nk_dot_f32x2_state_neon_t const *state_c, nk_dot_f32x2_state_neon_t const *state_d, //
    nk_b128_vec_t *result) {
    // Reduce each f64x2 → f64, downcast to f32, pack into f32x4
    float32x4_t sums_f32x4 = {(nk_f32_t)vaddvq_f64(state_a->sum_f64x2), (nk_f32_t)vaddvq_f64(state_b->sum_f64x2),
                              (nk_f32_t)vaddvq_f64(state_c->sum_f64x2), (nk_f32_t)vaddvq_f64(state_d->sum_f64x2)};
    result->u32x4 = vreinterpretq_u32_f32(sums_f32x4);
}

NK_PUBLIC void nk_dot_f64_neon(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                               nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count_scalars; idx_scalars += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a_scalars + idx_scalars);
        float64x2_t b_f64x2 = vld1q_f64(b_scalars + idx_scalars);
        // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
        float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t product_error_f64x2 = vfmsq_f64(product_f64x2, a_f64x2, b_f64x2);
        product_error_f64x2 = vnegq_f64(product_error_f64x2);
        // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
        float64x2_t t_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
        float64x2_t z_f64x2 = vsubq_f64(t_f64x2, sum_f64x2);
        float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(t_f64x2, z_f64x2)),
                                                vsubq_f64(product_f64x2, z_f64x2));
        // Update: sum = t, compensation += q + r
        sum_f64x2 = t_f64x2;
        compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
    }
    // Reduce and combine sum + compensation
    nk_f64_t sum = vaddvq_f64(vaddq_f64(sum_f64x2, compensation_f64x2));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack the input arrays into real and imaginary parts:
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)(a_pairs + idx_pairs));
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)(b_pairs + idx_pairs));
        float64x2_t a_real_f64x2 = a_f64x2x2.val[0];
        float64x2_t a_imag_f64x2 = a_f64x2x2.val[1];
        float64x2_t b_real_f64x2 = b_f64x2x2.val[0];
        float64x2_t b_imag_f64x2 = b_f64x2x2.val[1];

        // Real part: a_r*b_r - a_i*b_i (using TwoProd and TwoSum)
        // First term: a_r * b_r
        float64x2_t prod_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
        float64x2_t err_rr_f64x2 = vnegq_f64(vfmsq_f64(prod_rr_f64x2, a_real_f64x2, b_real_f64x2));
        float64x2_t t_real_f64x2 = vaddq_f64(sum_real_f64x2, prod_rr_f64x2);
        float64x2_t z_real_f64x2 = vsubq_f64(t_real_f64x2, sum_real_f64x2);
        float64x2_t err_sum_real_f64x2 = vaddq_f64(vsubq_f64(sum_real_f64x2, vsubq_f64(t_real_f64x2, z_real_f64x2)),
                                                   vsubq_f64(prod_rr_f64x2, z_real_f64x2));
        sum_real_f64x2 = t_real_f64x2;
        compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(err_sum_real_f64x2, err_rr_f64x2));
        // Second term: -a_i * b_i
        float64x2_t prod_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
        float64x2_t err_ii_f64x2 = vnegq_f64(vfmsq_f64(prod_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
        t_real_f64x2 = vsubq_f64(sum_real_f64x2, prod_ii_f64x2);
        z_real_f64x2 = vsubq_f64(sum_real_f64x2, t_real_f64x2);
        err_sum_real_f64x2 = vaddq_f64(vsubq_f64(z_real_f64x2, prod_ii_f64x2),
                                       vsubq_f64(sum_real_f64x2, vaddq_f64(t_real_f64x2, z_real_f64x2)));
        sum_real_f64x2 = t_real_f64x2;
        compensation_real_f64x2 = vsubq_f64(compensation_real_f64x2, vaddq_f64(err_sum_real_f64x2, err_ii_f64x2));

        // Imag part: a_r*b_i + a_i*b_r (using TwoProd and TwoSum)
        // First term: a_r * b_i
        float64x2_t prod_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
        float64x2_t err_ri_f64x2 = vnegq_f64(vfmsq_f64(prod_ri_f64x2, a_real_f64x2, b_imag_f64x2));
        float64x2_t t_imag_f64x2 = vaddq_f64(sum_imag_f64x2, prod_ri_f64x2);
        float64x2_t z_imag_f64x2 = vsubq_f64(t_imag_f64x2, sum_imag_f64x2);
        float64x2_t err_sum_imag_f64x2 = vaddq_f64(vsubq_f64(sum_imag_f64x2, vsubq_f64(t_imag_f64x2, z_imag_f64x2)),
                                                   vsubq_f64(prod_ri_f64x2, z_imag_f64x2));
        sum_imag_f64x2 = t_imag_f64x2;
        compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(err_sum_imag_f64x2, err_ri_f64x2));
        // Second term: a_i * b_r
        float64x2_t prod_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
        float64x2_t err_ir_f64x2 = vnegq_f64(vfmsq_f64(prod_ir_f64x2, a_imag_f64x2, b_real_f64x2));
        t_imag_f64x2 = vaddq_f64(sum_imag_f64x2, prod_ir_f64x2);
        z_imag_f64x2 = vsubq_f64(t_imag_f64x2, sum_imag_f64x2);
        err_sum_imag_f64x2 = vaddq_f64(vsubq_f64(sum_imag_f64x2, vsubq_f64(t_imag_f64x2, z_imag_f64x2)),
                                       vsubq_f64(prod_ir_f64x2, z_imag_f64x2));
        sum_imag_f64x2 = t_imag_f64x2;
        compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(err_sum_imag_f64x2, err_ir_f64x2));
    }

    // Reduce and combine sum + compensation
    nk_f64_t sum_real = vaddvq_f64(vaddq_f64(sum_real_f64x2, compensation_real_f64x2));
    nk_f64_t sum_imag = vaddvq_f64(vaddq_f64(sum_imag_f64x2, compensation_imag_f64x2));

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f64c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br - ai * bi;
        sum_imag += ar * bi + ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_PUBLIC void nk_vdot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack the input arrays into real and imaginary parts:
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)(a_pairs + idx_pairs));
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)(b_pairs + idx_pairs));
        float64x2_t a_real_f64x2 = a_f64x2x2.val[0];
        float64x2_t a_imag_f64x2 = a_f64x2x2.val[1];
        float64x2_t b_real_f64x2 = b_f64x2x2.val[0];
        float64x2_t b_imag_f64x2 = b_f64x2x2.val[1];

        // Real part: a_r*b_r + a_i*b_i (using TwoProd and TwoSum)
        // First term: a_r * b_r
        float64x2_t prod_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
        float64x2_t err_rr_f64x2 = vnegq_f64(vfmsq_f64(prod_rr_f64x2, a_real_f64x2, b_real_f64x2));
        float64x2_t t_real_f64x2 = vaddq_f64(sum_real_f64x2, prod_rr_f64x2);
        float64x2_t z_real_f64x2 = vsubq_f64(t_real_f64x2, sum_real_f64x2);
        float64x2_t err_sum_real_f64x2 = vaddq_f64(vsubq_f64(sum_real_f64x2, vsubq_f64(t_real_f64x2, z_real_f64x2)),
                                                   vsubq_f64(prod_rr_f64x2, z_real_f64x2));
        sum_real_f64x2 = t_real_f64x2;
        compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(err_sum_real_f64x2, err_rr_f64x2));
        // Second term: +a_i * b_i (note: + instead of - for conjugate)
        float64x2_t prod_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
        float64x2_t err_ii_f64x2 = vnegq_f64(vfmsq_f64(prod_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
        t_real_f64x2 = vaddq_f64(sum_real_f64x2, prod_ii_f64x2);
        z_real_f64x2 = vsubq_f64(t_real_f64x2, sum_real_f64x2);
        err_sum_real_f64x2 = vaddq_f64(vsubq_f64(sum_real_f64x2, vsubq_f64(t_real_f64x2, z_real_f64x2)),
                                       vsubq_f64(prod_ii_f64x2, z_real_f64x2));
        sum_real_f64x2 = t_real_f64x2;
        compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(err_sum_real_f64x2, err_ii_f64x2));

        // Imag part: a_r*b_i - a_i*b_r (using TwoProd and TwoSum)
        // First term: a_r * b_i
        float64x2_t prod_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
        float64x2_t err_ri_f64x2 = vnegq_f64(vfmsq_f64(prod_ri_f64x2, a_real_f64x2, b_imag_f64x2));
        float64x2_t t_imag_f64x2 = vaddq_f64(sum_imag_f64x2, prod_ri_f64x2);
        float64x2_t z_imag_f64x2 = vsubq_f64(t_imag_f64x2, sum_imag_f64x2);
        float64x2_t err_sum_imag_f64x2 = vaddq_f64(vsubq_f64(sum_imag_f64x2, vsubq_f64(t_imag_f64x2, z_imag_f64x2)),
                                                   vsubq_f64(prod_ri_f64x2, z_imag_f64x2));
        sum_imag_f64x2 = t_imag_f64x2;
        compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(err_sum_imag_f64x2, err_ri_f64x2));
        // Second term: -a_i * b_r (note: - instead of + for conjugate)
        float64x2_t prod_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
        float64x2_t err_ir_f64x2 = vnegq_f64(vfmsq_f64(prod_ir_f64x2, a_imag_f64x2, b_real_f64x2));
        t_imag_f64x2 = vsubq_f64(sum_imag_f64x2, prod_ir_f64x2);
        z_imag_f64x2 = vsubq_f64(sum_imag_f64x2, t_imag_f64x2);
        err_sum_imag_f64x2 = vaddq_f64(vsubq_f64(z_imag_f64x2, prod_ir_f64x2),
                                       vsubq_f64(sum_imag_f64x2, vaddq_f64(t_imag_f64x2, z_imag_f64x2)));
        sum_imag_f64x2 = t_imag_f64x2;
        compensation_imag_f64x2 = vsubq_f64(compensation_imag_f64x2, vaddq_f64(err_sum_imag_f64x2, err_ir_f64x2));
    }

    // Reduce and combine sum + compensation
    nk_f64_t sum_real = vaddvq_f64(vaddq_f64(sum_real_f64x2, compensation_real_f64x2));
    nk_f64_t sum_imag = vaddvq_f64(vaddq_f64(sum_imag_f64x2, compensation_imag_f64x2));

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f64c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br + ai * bi;
        sum_imag += ar * bi - ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars on NEON.
 *
 *  Uses the Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product.
 */
typedef struct nk_dot_f64x2_state_neon_t {
    float64x2_t sum_f64x2;
    float64x2_t compensation_f64x2;
} nk_dot_f64x2_state_neon_t;

NK_INTERNAL void nk_dot_f64x2_init_neon(nk_dot_f64x2_state_neon_t *state) {
    state->sum_f64x2 = vdupq_n_f64(0);
    state->compensation_f64x2 = vdupq_n_f64(0);
}

NK_INTERNAL void nk_dot_f64x2_update_neon(nk_dot_f64x2_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    float64x2_t sum_f64x2 = state->sum_f64x2;
    float64x2_t compensation_f64x2 = state->compensation_f64x2;
    float64x2_t a_f64x2 = vreinterpretq_f64_u64(a.u64x2);
    float64x2_t b_f64x2 = vreinterpretq_f64_u64(b.u64x2);

    // TwoProd: h = a * b, r = fma(a, b, -h) captures the rounding error
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    float64x2_t t_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
    float64x2_t z_f64x2 = vsubq_f64(t_f64x2, sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(t_f64x2, z_f64x2)),
                                            vsubq_f64(product_f64x2, z_f64x2));

    // Update: sum = t, compensation += q + r
    state->sum_f64x2 = t_f64x2;
    state->compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL void nk_dot_f64x2_finalize_neon(                                            //
    nk_dot_f64x2_state_neon_t const *state_a, nk_dot_f64x2_state_neon_t const *state_b, //
    nk_dot_f64x2_state_neon_t const *state_c, nk_dot_f64x2_state_neon_t const *state_d, //
    nk_b256_vec_t *result) {
    // Combine sum + compensation before horizontal reduction, then pack and store
    float64x2_t sums_ab_f64x2 = {vaddvq_f64(vaddq_f64(state_a->sum_f64x2, state_a->compensation_f64x2)),
                                 vaddvq_f64(vaddq_f64(state_b->sum_f64x2, state_b->compensation_f64x2))};
    float64x2_t sums_cd_f64x2 = {vaddvq_f64(vaddq_f64(state_c->sum_f64x2, state_c->compensation_f64x2)),
                                 vaddvq_f64(vaddq_f64(state_d->sum_f64x2, state_d->compensation_f64x2))};
    vst1q_f64(result->f64s + 0, sums_ab_f64x2);
    vst1q_f64(result->f64s + 2, sums_cd_f64x2);
}

NK_PUBLIC void nk_dot_e4m3_neon(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neon_cycle:
    if (count_scalars < 4) {
        a_f32x4 = nk_partial_load_e4m3x4_to_f32x4_neon_(a_scalars, count_scalars);
        b_f32x4 = nk_partial_load_e4m3x4_to_f32x4_neon_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x4 = nk_e4m3x4_to_f32x4_neon_(a_scalars);
        b_f32x4 = nk_e4m3x4_to_f32x4_neon_(b_scalars);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    if (count_scalars) goto nk_dot_e4m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_neon(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neon_cycle:
    if (count_scalars < 4) {
        a_f32x4 = nk_partial_load_e5m2x4_to_f32x4_neon_(a_scalars, count_scalars);
        b_f32x4 = nk_partial_load_e5m2x4_to_f32x4_neon_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x4 = nk_e5m2x4_to_f32x4_neon_(a_scalars);
        b_f32x4 = nk_e5m2x4_to_f32x4_neon_(b_scalars);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    if (count_scalars) goto nk_dot_e5m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEON_H
