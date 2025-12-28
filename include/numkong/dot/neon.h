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
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL float32x4_t nk_partial_load_f32x4_neon_(nk_f32_t const *x, nk_size_t n) {
    nk_b128_vec_t result;
    result.u32x4 = vdupq_n_u32(0);
    nk_size_t i = 0;
    for (; i < n; ++i) result.f32s[i] = x[i];
    return vreinterpretq_f32_u32(result.u32x4);
}

NK_INTERNAL void nk_partial_store_f32x4_neon_(float32x4_t vec, nk_f32_t *x, nk_size_t n) {
    nk_b128_vec_t u;
    u.u32x4 = vreinterpretq_u32_f32(vec);
    if (n > 0) x[0] = u.f32s[0];
    if (n > 1) x[1] = u.f32s[1];
    if (n > 2) x[2] = u.f32s[2];
    if (n > 3) x[3] = u.f32s[3];
}

NK_INTERNAL void nk_partial_store_i32x4_neon_(int32x4_t vec, nk_i32_t *x, nk_size_t n) {
    nk_b128_vec_t u;
    u.u32x4 = vreinterpretq_u32_s32(vec);
    if (n > 0) x[0] = u.i32s[0];
    if (n > 1) x[1] = u.i32s[1];
    if (n > 2) x[2] = u.i32s[2];
    if (n > 3) x[3] = u.i32s[3];
}

NK_PUBLIC void nk_dot_f32_neon(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a_scalars + idx_scalars);
        float32x4_t b_f32x4 = vld1q_f32(b_scalars + idx_scalars);
        sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    }
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_f32x4x2 = vld2q_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_f32x4x2 = vld2q_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_f32x4 = a_f32x4x2.val[0];
        float32x4_t a_imag_f32x4 = a_f32x4x2.val[1];
        float32x4_t b_real_f32x4 = b_f32x4x2.val[0];
        float32x4_t b_imag_f32x4 = b_f32x4x2.val[1];

        // Compute the dot product:
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
    }

    // Reduce horizontal sums:
    nk_f32_t sum_real = vaddvq_f32(sum_real_f32x4);
    nk_f32_t sum_imag = vaddvq_f32(sum_imag_f32x4);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br - ai * bi;
        sum_imag += ar * bi + ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_PUBLIC void nk_vdot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        // Unpack the input arrays into real and imaginary parts:
        float32x4x2_t a_f32x4x2 = vld2q_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x4x2_t b_f32x4x2 = vld2q_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        float32x4_t a_real_f32x4 = a_f32x4x2.val[0];
        float32x4_t a_imag_f32x4 = a_f32x4x2.val[1];
        float32x4_t b_real_f32x4 = b_f32x4x2.val[0];
        float32x4_t b_imag_f32x4 = b_f32x4x2.val[1];

        // Compute the dot product:
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
    }

    // Reduce horizontal sums:
    nk_f32_t sum_real = vaddvq_f32(sum_real_f32x4);
    nk_f32_t sum_imag = vaddvq_f32(sum_imag_f32x4);

    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f32_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real += ar * br + ai * bi;
        sum_imag += ar * bi - ai * br;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

/**
 *  @brief Running state for 128-bit dot accumulation over f32 scalars on NEON.
 */
typedef struct nk_dot_f32x4_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_f32x4_state_neon_t;

NK_INTERNAL void nk_dot_f32x4_init_neon(nk_dot_f32x4_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f32x4_update_neon(nk_dot_f32x4_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    float32x4_t sum_f32x4 = state->sum_f32x4;
    sum_f32x4 = vfmaq_f32(sum_f32x4, vreinterpretq_f32_u32(a.u32x4), vreinterpretq_f32_u32(b.u32x4));
    state->sum_f32x4 = sum_f32x4;
}

NK_INTERNAL void nk_dot_f32x4_finalize_neon(                                            //
    nk_dot_f32x4_state_neon_t const *state_a, nk_dot_f32x4_state_neon_t const *state_b, //
    nk_dot_f32x4_state_neon_t const *state_c, nk_dot_f32x4_state_neon_t const *state_d, //
    nk_f32_t *results) {
    results[0] = vaddvq_f32(state_a->sum_f32x4);
    results[1] = vaddvq_f32(state_b->sum_f32x4);
    results[2] = vaddvq_f32(state_c->sum_f32x4);
    results[3] = vaddvq_f32(state_d->sum_f32x4);
}

/** @brief Type-agnostic 128-bit full load (NEON). */
NK_INTERNAL void nk_load_b128_neon_(void const *src, nk_b128_vec_t *dst) {
    dst->u8x16 = vld1q_u8((nk_u8_t const *)src);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEON_H