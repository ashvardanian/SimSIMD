/**
 *  @brief SIMD-accelerated Binary Similarity Measures optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/binary/neon.h
 *  @sa include/numkong/binary.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_BINARY_NEON_H
#define NK_BINARY_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/types.h"
#include "numkong/reduce/neon.h"   // nk_reduce_add_u8x16_neon_
#include "numkong/binary/serial.h" // nk_popcount_u1

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_hamming_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, NK_BITS_PER_BYTE);
    nk_u32_t differences = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            uint8x16_t xor_popcount_u8x16 = vcntq_u8(veorq_u8(a_u8x16, b_u8x16));
            popcount_u8x16 = vaddq_u8(popcount_u8x16, xor_popcount_u8x16);
        }
        differences += nk_reduce_add_u8x16_neon_(popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i) differences += nk_popcount_u1(a[i] ^ b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, NK_BITS_PER_BYTE);
    nk_u32_t intersection_count = 0, union_count = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t intersection_popcount_u8x16 = vdupq_n_u8(0);
        uint8x16_t union_popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            intersection_popcount_u8x16 = vaddq_u8(intersection_popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
            union_popcount_u8x16 = vaddq_u8(union_popcount_u8x16, vcntq_u8(vorrq_u8(a_u8x16, b_u8x16)));
        }
        intersection_count += nk_reduce_add_u8x16_neon_(intersection_popcount_u8x16);
        union_count += nk_reduce_add_u8x16_neon_(union_popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i)
        intersection_count += nk_popcount_u1(a[i] & b[i]), union_count += nk_popcount_u1(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 1.0f;
}

NK_PUBLIC void nk_jaccard_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    nk_size_t i = 0;
    uint32x4_t intersection_count_u32x4 = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i);
        uint32x4_t b_u32x4 = vld1q_u32(b + i);
        uint32x4_t equality_mask = vceqq_u32(a_u32x4, b_u32x4);
        intersection_count_u32x4 = vaddq_u32(intersection_count_u32x4, vshrq_n_u32(equality_mask, 31));
    }
    intersection_count += vaddvq_u32(intersection_count_u32x4);
    for (; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 1.0f;
}

typedef struct nk_jaccard_b128_state_neon_t {
    uint32x4_t intersection_count_u32x4;
} nk_jaccard_b128_state_neon_t;

NK_INTERNAL void nk_jaccard_b128_init_neon(nk_jaccard_b128_state_neon_t *state) {
    state->intersection_count_u32x4 = vdupq_n_u32(0);
}

NK_INTERNAL void nk_jaccard_b128_update_neon(nk_jaccard_b128_state_neon_t *state, uint8x16_t a, uint8x16_t b) {

    // Process one 128-bit chunk (native ARM NEON register size).
    // Uses vector accumulation - horizontal sum deferred to finalize.
    //
    // ARM NEON instruction characteristics:
    //   `vandq_u8`:    Bitwise AND, 1 cycle latency
    //   `vcntq_u8`:    Byte popcount (16 bytes → 16 popcounts), 1-2 cycles
    //   `vpaddlq_u8`:  Pairwise widening add u8 → u16, 1 cycle
    //   `vpaddlq_u16`: Pairwise widening add u16 → u32, 1 cycle
    //   `vaddq_u32`:   Vector add u32x4, 1 cycle
    // Total: ~5-6 cycles per 128-bit chunk (no horizontal sum penalty per update)

    // Step 1: Compute intersection bits (A AND B)
    uint8x16_t intersection_u8x16 = vandq_u8(a, b);

    // Step 2: Byte-level popcount - each byte contains count of set bits (0-8)
    uint8x16_t popcount_u8x16 = vcntq_u8(intersection_u8x16);

    // Step 3: Pairwise widening reduction chain
    // u8x16 → u16x8: pairs of adjacent bytes summed into 16-bit
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    // u16x8 → u32x4: pairs of 16-bit values summed into 32-bit
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);

    // Step 4: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vaddq_u32(state->intersection_count_u32x4, popcount_u32x4);
}

NK_INTERNAL void nk_jaccard_b128_finalize_neon(nk_jaccard_b128_state_neon_t const *state_a,
                                               nk_jaccard_b128_state_neon_t const *state_b,
                                               nk_jaccard_b128_state_neon_t const *state_c,
                                               nk_jaccard_b128_state_neon_t const *state_d, nk_f32_t query_popcount,
                                               nk_f32_t target_popcount_a, nk_f32_t target_popcount_b,
                                               nk_f32_t target_popcount_c, nk_f32_t target_popcount_d,
                                               nk_f32_t *results) {

    // Horizontal sum each state's vector accumulator via `vaddvq_u32` (ARMv8.1+, 2-3 cycles)
    // This is done once at finalize, not per-update, for better throughput.
    uint32x4_t intersection_u32x4 = (uint32x4_t) {
        vaddvq_u32(state_a->intersection_count_u32x4), vaddvq_u32(state_b->intersection_count_u32x4),
        vaddvq_u32(state_c->intersection_count_u32x4), vaddvq_u32(state_d->intersection_count_u32x4)};
    float32x4_t intersection_f32x4 = vcvtq_f32_u32(intersection_u32x4);

    // Compute union using |A OR B| = |A| + |B| - |A AND B|
    float32x4_t query_f32x4 = vdupq_n_f32(query_popcount);
    float32x4_t targets_f32x4 = (float32x4_t) {target_popcount_a, target_popcount_b, target_popcount_c,
                                               target_popcount_d};
    float32x4_t union_f32x4 = vsubq_f32(vaddq_f32(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case (empty vectors → distance = 1.0)
    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    uint32x4_t zero_union_mask = vceqq_f32(union_f32x4, vdupq_n_f32(0.0f));
    float32x4_t safe_union_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, union_f32x4);

    // Fast reciprocal with Newton-Raphson refinement:
    //   `vrecpeq_f32`: ~12-bit estimate, 1 cycle
    //   `vrecpsq_f32`: Newton-Raphson step computes (2 - a*b), 1 cycle
    //   `vmulq_f32`: multiply, 1 cycle
    // One N-R iteration: ~24-bit accuracy, sufficient for f32 (23 mantissa bits).
    // Total: ~3-4 cycles vs ~10-14 cycles for division.
    float32x4_t union_reciprocal_f32x4 = vrecpeq_f32(safe_union_f32x4);
    union_reciprocal_f32x4 = vmulq_f32(union_reciprocal_f32x4, vrecpsq_f32(safe_union_f32x4, union_reciprocal_f32x4));

    // Compute Jaccard distance = 1 - intersection/union
    float32x4_t ratio_f32x4 = vmulq_f32(intersection_f32x4, union_reciprocal_f32x4);
    float32x4_t jaccard_f32x4 = vsubq_f32(one_f32x4, ratio_f32x4);
    float32x4_t result_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, jaccard_f32x4);

    vst1q_f32(results, result_f32x4);
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

#endif // NK_BINARY_NEON_H
