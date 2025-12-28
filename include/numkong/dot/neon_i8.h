/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neon_i8.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_NEON_I8_H
#define NK_DOT_NEON_I8_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_neon(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                              nk_i32_t *result) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        int8x16_t a_i8x16 = vld1q_s8(a_scalars + idx_scalars);
        int8x16_t b_i8x16 = vld1q_s8(b_scalars + idx_scalars);
        sum_i32x4 = vdotq_s32(sum_i32x4, a_i8x16, b_i8x16);
    }
    nk_i32_t sum = vaddvq_s32(sum_i32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_neon(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                              nk_u32_t *result) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count_scalars; idx_scalars += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a_scalars + idx_scalars);
        uint8x16_t b_u8x16 = vld1q_u8(b_scalars + idx_scalars);
        sum_u32x4 = vdotq_u32(sum_u32x4, a_u8x16, b_u8x16);
    }
    nk_u32_t sum = vaddvq_u32(sum_u32x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on NEON.
 */
typedef struct nk_dot_i8x16_state_neon_t {
    int32x4_t sum_i32x4;
} nk_dot_i8x16_state_neon_t;

NK_INTERNAL void nk_dot_i8x16_init_neon(nk_dot_i8x16_state_neon_t *state) { state->sum_i32x4 = vdupq_n_s32(0); }

NK_INTERNAL void nk_dot_i8x16_update_neon(nk_dot_i8x16_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    int32x4_t sum_i32x4 = state->sum_i32x4;
    sum_i32x4 = vdotq_s32(sum_i32x4, vreinterpretq_s8_u32(a.u32x4s[0]), vreinterpretq_s8_u32(b.u32x4s[0]));
    state->sum_i32x4 = sum_i32x4;
}

NK_INTERNAL void nk_dot_i8x16_finalize_neon(                                            //
    nk_dot_i8x16_state_neon_t const *state_a, nk_dot_i8x16_state_neon_t const *state_b, //
    nk_dot_i8x16_state_neon_t const *state_c, nk_dot_i8x16_state_neon_t const *state_d, //
    nk_i32_t *results) {
    results[0] = (nk_i32_t)vaddvq_s32(state_a->sum_i32x4);
    results[1] = (nk_i32_t)vaddvq_s32(state_b->sum_i32x4);
    results[2] = (nk_i32_t)vaddvq_s32(state_c->sum_i32x4);
    results[3] = (nk_i32_t)vaddvq_s32(state_d->sum_i32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on NEON.
 */
typedef struct nk_dot_u8x16_state_neon_t {
    uint32x4_t sum_u32x4;
} nk_dot_u8x16_state_neon_t;

NK_INTERNAL void nk_dot_u8x16_init_neon(nk_dot_u8x16_state_neon_t *state) { state->sum_u32x4 = vdupq_n_u32(0); }

NK_INTERNAL void nk_dot_u8x16_update_neon(nk_dot_u8x16_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    uint32x4_t sum_u32x4 = state->sum_u32x4;
    sum_u32x4 = vdotq_u32(sum_u32x4, vreinterpretq_u8_u32(a.u32x4s[0]), vreinterpretq_u8_u32(b.u32x4s[0]));
    state->sum_u32x4 = sum_u32x4;
}

NK_INTERNAL void nk_dot_u8x16_finalize_neon(                                            //
    nk_dot_u8x16_state_neon_t const *state_a, nk_dot_u8x16_state_neon_t const *state_b, //
    nk_dot_u8x16_state_neon_t const *state_c, nk_dot_u8x16_state_neon_t const *state_d, //
    nk_u32_t *results) {
    results[0] = (nk_u32_t)vaddvq_u32(state_a->sum_u32x4);
    results[1] = (nk_u32_t)vaddvq_u32(state_b->sum_u32x4);
    results[2] = (nk_u32_t)vaddvq_u32(state_c->sum_u32x4);
    results[3] = (nk_u32_t)vaddvq_u32(state_d->sum_u32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_I8
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEON_I8_H