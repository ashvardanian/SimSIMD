/**
 *  @brief SIMD-accelerated Dot Products for NEON SDOT.
 *  @file include/numkong/dot/neonsdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonsdot_instructions ARM NEON SDOT/UDOT Instructions (ARMv8.4-DotProd)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vdotq_s32                   SDOT (V.4S, V.16B, V.16B)       3cy         2/cy        4/cy
 *      vdotq_u32                   UDOT (V.4S, V.16B, V.16B)       3cy         2/cy        4/cy
 *      vld1q_s8                    LD1 (V.16B)                     4cy         2/cy        3/cy
 *      vld1q_u8                    LD1 (V.16B)                     4cy         2/cy        3/cy
 *      vaddvq_s32                  ADDV (V.4S)                     4cy         1/cy        2/cy
 *      vaddvq_u32                  ADDV (V.4S)                     4cy         1/cy        2/cy
 *
 *  The ARMv8.4-DotProd extension provides SDOT/UDOT instructions critical for int8 quantized ML
 *  inference. Each instruction computes four dot products of 4-element int8 vectors, accumulating
 *  into int32 lanes, processing 16 multiply-accumulates per instruction.
 *
 *  SDOT handles signed int8 operands while UDOT handles unsigned. The 3-cycle latency with 2/cy
 *  throughput on A76 (4/cy on newer cores) enables int8 matrix multiplication for
 *  quantized neural network inference, where 8-bit weights reduce memory bandwidth by 4x vs FP32.
 */
#ifndef NK_DOT_NEONSDOT_H
#define NK_DOT_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_neonsdot(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
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

NK_PUBLIC void nk_dot_u8_neonsdot(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
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
typedef struct nk_dot_i8x16_state_neonsdot_t {
    int32x4_t sum_i32x4;
} nk_dot_i8x16_state_neonsdot_t;

NK_INTERNAL void nk_dot_i8x16_init_neonsdot(nk_dot_i8x16_state_neonsdot_t *state) { state->sum_i32x4 = vdupq_n_s32(0); }

NK_INTERNAL void nk_dot_i8x16_update_neonsdot(nk_dot_i8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    int32x4_t sum_i32x4 = state->sum_i32x4;
    sum_i32x4 = vdotq_s32(sum_i32x4, vreinterpretq_s8_u32(a.u32x4), vreinterpretq_s8_u32(b.u32x4));
    state->sum_i32x4 = sum_i32x4;
}

NK_INTERNAL void nk_dot_i8x16_finalize_neonsdot(                                                //
    nk_dot_i8x16_state_neonsdot_t const *state_a, nk_dot_i8x16_state_neonsdot_t const *state_b, //
    nk_dot_i8x16_state_neonsdot_t const *state_c, nk_dot_i8x16_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->i32s[0] = vaddvq_s32(state_a->sum_i32x4);
    result->i32s[1] = vaddvq_s32(state_b->sum_i32x4);
    result->i32s[2] = vaddvq_s32(state_c->sum_i32x4);
    result->i32s[3] = vaddvq_s32(state_d->sum_i32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on NEON.
 */
typedef struct nk_dot_u8x16_state_neonsdot_t {
    uint32x4_t sum_u32x4;
} nk_dot_u8x16_state_neonsdot_t;

NK_INTERNAL void nk_dot_u8x16_init_neonsdot(nk_dot_u8x16_state_neonsdot_t *state) { state->sum_u32x4 = vdupq_n_u32(0); }

NK_INTERNAL void nk_dot_u8x16_update_neonsdot(nk_dot_u8x16_state_neonsdot_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint32x4_t sum_u32x4 = state->sum_u32x4;
    sum_u32x4 = vdotq_u32(sum_u32x4, vreinterpretq_u8_u32(a.u32x4), vreinterpretq_u8_u32(b.u32x4));
    state->sum_u32x4 = sum_u32x4;
}

NK_INTERNAL void nk_dot_u8x16_finalize_neonsdot(                                                //
    nk_dot_u8x16_state_neonsdot_t const *state_a, nk_dot_u8x16_state_neonsdot_t const *state_b, //
    nk_dot_u8x16_state_neonsdot_t const *state_c, nk_dot_u8x16_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = vaddvq_u32(state_a->sum_u32x4);
    result->u32s[1] = vaddvq_u32(state_b->sum_u32x4);
    result->u32s[2] = vaddvq_u32(state_c->sum_u32x4);
    result->u32s[3] = vaddvq_u32(state_d->sum_u32x4);
}

NK_PUBLIC void nk_dot_i4_neonsdot(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    //
    // ARM NEON SDOT handles signed×signed directly, so we use direct sign-extension:
    // Extract nibbles [0,15], sign-extend to i8 [-8,7] via shift trick, then SDOT.
    // No algebraic correction needed unlike x86 DPBUSD.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint8x16_t a_i4x32, b_i4x32;

nk_dot_i4_neonsdot_cycle:
    if (n_bytes < 16) {
        // Partial load for tail handling
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_u8_t const *a_ptr = (nk_u8_t const *)a;
        nk_u8_t const *b_ptr = (nk_u8_t const *)b;
        for (nk_size_t i = 0; i < n_bytes; i++) {
            a_vec.u8s[i] = a_ptr[i];
            b_vec.u8s[i] = b_ptr[i];
        }
        a_i4x32 = a_vec.u8x16;
        b_i4x32 = b_vec.u8x16;
        n_bytes = 0;
    }
    else {
        a_i4x32 = vld1q_u8((nk_u8_t const *)a);
        b_i4x32 = vld1q_u8((nk_u8_t const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles as unsigned [0,15]
    uint8x16_t a_lo_u8x16 = vandq_u8(a_i4x32, nibble_mask_u8x16);
    uint8x16_t a_hi_u8x16 = vshrq_n_u8(a_i4x32, 4);
    uint8x16_t b_lo_u8x16 = vandq_u8(b_i4x32, nibble_mask_u8x16);
    uint8x16_t b_hi_u8x16 = vshrq_n_u8(b_i4x32, 4);

    // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
    // This converts unsigned [0,15] to signed [-8,7]
    int8x16_t a_lo_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_lo_u8x16), 4), 4);
    int8x16_t a_hi_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_hi_u8x16), 4), 4);
    int8x16_t b_lo_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_lo_u8x16), 4), 4);
    int8x16_t b_hi_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_hi_u8x16), 4), 4);

    // SDOT for signed dot product - no correction needed!
    sum_i32x4 = vdotq_s32(sum_i32x4, a_lo_i8x16, b_lo_i8x16);
    sum_i32x4 = vdotq_s32(sum_i32x4, a_hi_i8x16, b_hi_i8x16);

    if (n_bytes) goto nk_dot_i4_neonsdot_cycle;

    *result = vaddvq_s32(sum_i32x4);
}

NK_PUBLIC void nk_dot_u4_neonsdot(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Values are ∈ [0,15], so UDOT can be used directly.
    //
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint8x16_t a_u4x32, b_u4x32;

nk_dot_u4_neonsdot_cycle:
    if (n_bytes < 16) {
        // Partial load for tail handling
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_u8_t const *a_ptr = (nk_u8_t const *)a;
        nk_u8_t const *b_ptr = (nk_u8_t const *)b;
        for (nk_size_t i = 0; i < n_bytes; i++) {
            a_vec.u8s[i] = a_ptr[i];
            b_vec.u8s[i] = b_ptr[i];
        }
        a_u4x32 = a_vec.u8x16;
        b_u4x32 = b_vec.u8x16;
        n_bytes = 0;
    }
    else {
        a_u4x32 = vld1q_u8((nk_u8_t const *)a);
        b_u4x32 = vld1q_u8((nk_u8_t const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // Extract low and high nibbles - values in [0,15] work directly with UDOT
    uint8x16_t a_lo_u8x16 = vandq_u8(a_u4x32, nibble_mask_u8x16);
    uint8x16_t a_hi_u8x16 = vshrq_n_u8(a_u4x32, 4);
    uint8x16_t b_lo_u8x16 = vandq_u8(b_u4x32, nibble_mask_u8x16);
    uint8x16_t b_hi_u8x16 = vshrq_n_u8(b_u4x32, 4);

    // UDOT directly on unsigned nibbles
    sum_u32x4 = vdotq_u32(sum_u32x4, a_lo_u8x16, b_lo_u8x16);
    sum_u32x4 = vdotq_u32(sum_u32x4, a_hi_u8x16, b_hi_u8x16);

    if (n_bytes) goto nk_dot_u4_neonsdot_cycle;

    *result = vaddvq_u32(sum_u32x4);
}

struct nk_dot_i4x32_state_neonsdot_t {
    int32x4_t product_sum_i32x4;
};

NK_INTERNAL void nk_dot_i4x32_init_neonsdot(nk_dot_i4x32_state_neonsdot_t *state) {
    state->product_sum_i32x4 = vdupq_n_s32(0);
}

NK_INTERNAL void nk_dot_i4x32_update_neonsdot(nk_dot_i4x32_state_neonsdot_t *state, nk_b128_vec_t a_i4x32,
                                              nk_b128_vec_t b_i4x32, nk_size_t depth_offset,
                                              nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);

    // Extract nibbles as unsigned first
    uint8x16_t a_low_u8x16 = vandq_u8(a_i4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t a_high_u8x16 = vshrq_n_u8(a_i4x32.u8x16, 4);
    uint8x16_t b_low_u8x16 = vandq_u8(b_i4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t b_high_u8x16 = vshrq_n_u8(b_i4x32.u8x16, 4);

    // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
    int8x16_t a_low_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_low_u8x16), 4), 4);
    int8x16_t a_high_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(a_high_u8x16), 4), 4);
    int8x16_t b_low_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_low_u8x16), 4), 4);
    int8x16_t b_high_i8x16 = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(b_high_u8x16), 4), 4);

    // SDOT for signed dot product - no correction needed!
    int32x4_t product_sum_i32x4 = state->product_sum_i32x4;
    product_sum_i32x4 = vdotq_s32(product_sum_i32x4, a_low_i8x16, b_low_i8x16);
    product_sum_i32x4 = vdotq_s32(product_sum_i32x4, a_high_i8x16, b_high_i8x16);
    state->product_sum_i32x4 = product_sum_i32x4;
}

NK_INTERNAL void nk_dot_i4x32_finalize_neonsdot(                                                //
    nk_dot_i4x32_state_neonsdot_t const *state_a, nk_dot_i4x32_state_neonsdot_t const *state_b, //
    nk_dot_i4x32_state_neonsdot_t const *state_c, nk_dot_i4x32_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Simple reduction - no correction formula needed with sign-extension approach!
    result->i32s[0] = vaddvq_s32(state_a->product_sum_i32x4);
    result->i32s[1] = vaddvq_s32(state_b->product_sum_i32x4);
    result->i32s[2] = vaddvq_s32(state_c->product_sum_i32x4);
    result->i32s[3] = vaddvq_s32(state_d->product_sum_i32x4);
}

struct nk_dot_u4x32_state_neonsdot_t {
    uint32x4_t product_sum_u32x4;
};

NK_INTERNAL void nk_dot_u4x32_init_neonsdot(nk_dot_u4x32_state_neonsdot_t *state) {
    state->product_sum_u32x4 = vdupq_n_u32(0);
}

NK_INTERNAL void nk_dot_u4x32_update_neonsdot(nk_dot_u4x32_state_neonsdot_t *state, nk_b128_vec_t a_u4x32,
                                              nk_b128_vec_t b_u4x32, nk_size_t depth_offset,
                                              nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);

    // Extract nibbles - values in [0,15] work directly with UDOT
    uint8x16_t a_low_u8x16 = vandq_u8(a_u4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t a_high_u8x16 = vshrq_n_u8(a_u4x32.u8x16, 4);
    uint8x16_t b_low_u8x16 = vandq_u8(b_u4x32.u8x16, nibble_mask_u8x16);
    uint8x16_t b_high_u8x16 = vshrq_n_u8(b_u4x32.u8x16, 4);

    // UDOT directly on unsigned nibbles
    uint32x4_t product_sum_u32x4 = state->product_sum_u32x4;
    product_sum_u32x4 = vdotq_u32(product_sum_u32x4, a_low_u8x16, b_low_u8x16);
    product_sum_u32x4 = vdotq_u32(product_sum_u32x4, a_high_u8x16, b_high_u8x16);
    state->product_sum_u32x4 = product_sum_u32x4;
}

NK_INTERNAL void nk_dot_u4x32_finalize_neonsdot(                                                //
    nk_dot_u4x32_state_neonsdot_t const *state_a, nk_dot_u4x32_state_neonsdot_t const *state_b, //
    nk_dot_u4x32_state_neonsdot_t const *state_c, nk_dot_u4x32_state_neonsdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Simple reduction - no correction formula needed!
    result->u32s[0] = vaddvq_u32(state_a->product_sum_u32x4);
    result->u32s[1] = vaddvq_u32(state_b->product_sum_u32x4);
    result->u32s[2] = vaddvq_u32(state_c->product_sum_u32x4);
    result->u32s[3] = vaddvq_u32(state_d->product_sum_u32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEONSDOT_H
