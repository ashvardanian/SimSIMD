/**
 *  @brief SIMD-accelerated Set Similarity Measures optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/set/ice.h
 *  @sa include/numkong/set.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SET_ICE_H
#define NK_SET_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_hamming_u1_ice(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    nk_u32_t xor_count;
    // It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_bytes <= 64) { // Up to 512 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        __m512i xor_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_u8x64, b_u8x64));
        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    else if (n_bytes <= 128) { // Up to 1024 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 64);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8));
    }
    else if (n_bytes <= 192) { // Up to 1536 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 128);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(
            xor_popcount_three_u64x8, _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else if (n_bytes <= 256) { // Up to 2048 bits.
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(mask, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(mask, b + 192);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        __m512i xor_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_four_u8x64, b_four_u8x64));
        xor_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(xor_popcount_four_u64x8, xor_popcount_three_u64x8),
                             _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else {
        __m512i xor_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    nk_hamming_u1_ice_cycle:
        if (n_bytes < 64) {
            __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
            a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
            n_bytes = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_bytes -= 64;
        }
        __m512i xor_u8x64 = _mm512_xor_si512(a_u8x64, b_u8x64);
        xor_popcount_u64x8 = _mm512_add_epi64(xor_popcount_u64x8, _mm512_popcnt_epi64(xor_u8x64));
        if (n_bytes) goto nk_hamming_u1_ice_cycle;

        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    *result = xor_count;
}

NK_PUBLIC void nk_jaccard_u1_ice(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    nk_u32_t intersection_count = 0, union_count = 0;
    //  It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_bytes <= 64) { // Up to 512 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b);
        __m512i intersection_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_u8x64, b_u8x64));
        __m512i union_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_u8x64, b_u8x64));
        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    else if (n_bytes <= 128) { // Up to 1024 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 64);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8));
        union_count = _mm512_reduce_add_epi64(_mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8));
    }
    else if (n_bytes <= 192) { // Up to 1536 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 128);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        intersection_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(intersection_popcount_three_u64x8,
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(union_popcount_three_u64x8,
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else if (n_bytes <= 256) { // Up to 2048 bits.
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b + 192);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        __m512i intersection_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_four_u8x64, b_four_u8x64));
        __m512i union_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_four_u8x64, b_four_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(intersection_popcount_four_u64x8, intersection_popcount_three_u64x8),
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(union_popcount_four_u64x8, union_popcount_three_u64x8),
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else {
        __m512i intersection_popcount_u64x8 = _mm512_setzero_si512();
        __m512i union_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    nk_jaccard_u1_ice_cycle:
        if (n_bytes < 64) {
            __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
            a_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b);
            n_bytes = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_bytes -= 64;
        }
        __m512i intersection_u8x64 = _mm512_and_si512(a_u8x64, b_u8x64);
        __m512i union_u8x64 = _mm512_or_si512(a_u8x64, b_u8x64);
        intersection_popcount_u64x8 = _mm512_add_epi64(intersection_popcount_u64x8,
                                                       _mm512_popcnt_epi64(intersection_u8x64));
        union_popcount_u64x8 = _mm512_add_epi64(union_popcount_u64x8, _mm512_popcnt_epi64(union_u8x64));
        if (n_bytes) goto nk_jaccard_u1_ice_cycle;

        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 1.0f;
}

NK_PUBLIC void nk_jaccard_u32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 16; n_remaining -= 16, a += 16, b += 16) {
        __m512i a_u32x16 = _mm512_loadu_epi32(a);
        __m512i b_u32x16 = _mm512_loadu_epi32(b);
        __mmask16 equality_mask = _mm512_cmpeq_epi32_mask(a_u32x16, b_u32x16);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    if (n_remaining) {
        __mmask16 load_mask = (__mmask16)_bzhi_u32(0xFFFF, n_remaining);
        __m512i a_u32x16 = _mm512_maskz_loadu_epi32(load_mask, a);
        __m512i b_u32x16 = _mm512_maskz_loadu_epi32(load_mask, b);
        __mmask16 equality_mask = _mm512_mask_cmpeq_epi32_mask(load_mask, a_u32x16, b_u32x16);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_mask);
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 1.0f;
}

NK_PUBLIC void nk_hamming_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_u32_t differences = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 64; n_remaining -= 64, a += 64, b += 64) {
        __m512i a_u8x64 = _mm512_loadu_si512((__m512i const *)a);
        __m512i b_u8x64 = _mm512_loadu_si512((__m512i const *)b);
        __mmask64 neq_mask = _mm512_cmpneq_epi8_mask(a_u8x64, b_u8x64);
        differences += _mm_popcnt_u64(neq_mask);
    }
    if (n_remaining) {
        __mmask64 load_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_remaining);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(load_mask, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(load_mask, b);
        __mmask64 neq_mask = _mm512_mask_cmpneq_epi8_mask(load_mask, a_u8x64, b_u8x64);
        differences += _mm_popcnt_u64(neq_mask);
    }
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u16_ice(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t matches = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 32; n_remaining -= 32, a += 32, b += 32) {
        __m512i a_u16x32 = _mm512_loadu_si512((__m512i const *)a);
        __m512i b_u16x32 = _mm512_loadu_si512((__m512i const *)b);
        __mmask32 equality_mask = _mm512_cmpeq_epi16_mask(a_u16x32, b_u16x32);
        matches += _mm_popcnt_u32(equality_mask);
    }
    if (n_remaining) {
        __mmask32 load_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n_remaining);
        __m512i a_u16x32 = _mm512_maskz_loadu_epi16(load_mask, a);
        __m512i b_u16x32 = _mm512_maskz_loadu_epi16(load_mask, b);
        __mmask32 equality_mask = _mm512_mask_cmpeq_epi16_mask(load_mask, a_u16x32, b_u16x32);
        matches += _mm_popcnt_u32(equality_mask);
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 1.0f;
}

struct nk_hamming_b512_state_ice_t {
    __m512i intersection_count_i64x8;
};

NK_INTERNAL void nk_hamming_b512_init_ice(nk_hamming_b512_state_ice_t *state) {
    state->intersection_count_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_hamming_b512_update_ice(nk_hamming_b512_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->intersection_count_i64x8 = _mm512_add_epi64(state->intersection_count_i64x8,
                                                       _mm512_popcnt_epi64(_mm512_xor_si512(a.zmm, b.zmm)));
}

NK_INTERNAL void nk_hamming_b512_finalize_ice( //
    nk_hamming_b512_state_ice_t const *state_a, nk_hamming_b512_state_ice_t const *state_b,
    nk_hamming_b512_state_ice_t const *state_c, nk_hamming_b512_state_ice_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = (nk_u32_t)_mm512_reduce_add_epi64(state_a->intersection_count_i64x8);
    result->u32s[1] = (nk_u32_t)_mm512_reduce_add_epi64(state_b->intersection_count_i64x8);
    result->u32s[2] = (nk_u32_t)_mm512_reduce_add_epi64(state_c->intersection_count_i64x8);
    result->u32s[3] = (nk_u32_t)_mm512_reduce_add_epi64(state_d->intersection_count_i64x8);
}

struct nk_jaccard_b512_state_ice_t {
    __m512i intersection_count_i64x8;
};

NK_INTERNAL void nk_jaccard_b512_init_ice(nk_jaccard_b512_state_ice_t *state) {
    state->intersection_count_i64x8 = _mm512_setzero_si512();
}

NK_INTERNAL void nk_jaccard_b512_update_ice(nk_jaccard_b512_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b,
                                            nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->intersection_count_i64x8 = _mm512_add_epi64(state->intersection_count_i64x8,
                                                       _mm512_popcnt_epi64(_mm512_and_si512(a.zmm, b.zmm)));
}

NK_INTERNAL void nk_jaccard_b512_finalize_ice( //
    nk_jaccard_b512_state_ice_t const *state_a, nk_jaccard_b512_state_ice_t const *state_b,
    nk_jaccard_b512_state_ice_t const *state_c, nk_jaccard_b512_state_ice_t const *state_d, nk_f32_t query_popcount,
    nk_f32_t target_popcount_a, nk_f32_t target_popcount_b, nk_f32_t target_popcount_c, nk_f32_t target_popcount_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // Port-optimized 4-way horizontal reduction using early i64 → i32 truncation.
    //
    // Key insight: `_mm_hadd_epi32` uses ports p01, not p5, avoiding the shuffle bottleneck.
    // By truncating to i32 early, we can use hadd for reduction instead of expensive shuffles.
    //
    // Ice Lake execution ports:
    // - p0: Division, reciprocal (`VRCP14PS`: 4cy latency, 1/cy throughput)
    // - p01: FP mul/add/fma, hadd (`VMULPS`/`VPHADDD`: 3cy latency, 0.5/cy throughput)
    // - p015: Integer add (`VPADDD`: 1cy latency, 0.33/cy throughput)
    // - p5: Shuffles/extracts (`VEXTRACTI128`: 3cy latency, 1/cy throughput)

    // Step 1: Truncate 8x i64 → 8x i32 per state (fits in YMM)
    // `VPMOVQD` (ZMM → YMM): 4cy latency, 0.5/cy throughput, port p01
    __m256i a_i32x8 = _mm512_cvtepi64_epi32(state_a->intersection_count_i64x8);
    __m256i b_i32x8 = _mm512_cvtepi64_epi32(state_b->intersection_count_i64x8);
    __m256i c_i32x8 = _mm512_cvtepi64_epi32(state_c->intersection_count_i64x8);
    __m256i d_i32x8 = _mm512_cvtepi64_epi32(state_d->intersection_count_i64x8);

    // Step 2: Reduce 8x i32 → 4x i32 (add high 128-bit lane to low)
    // - `VEXTRACTI128`: 3cy latency, 1/cy throughput, port p5
    // - `VPADDD` (XMM): 1cy latency, 0.33/cy throughput, ports p015
    __m128i a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(a_i32x8), _mm256_extracti128_si256(a_i32x8, 1));
    __m128i b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(b_i32x8), _mm256_extracti128_si256(b_i32x8, 1));
    __m128i c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(c_i32x8), _mm256_extracti128_si256(c_i32x8, 1));
    __m128i d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(d_i32x8), _mm256_extracti128_si256(d_i32x8, 1));

    // Step 3: Reduce 4x i32 → 2x i32 using horizontal add (uses p01, not p5!)
    // - `VPHADDD` (XMM): 3cy latency, 0.5/cy throughput, ports p01
    __m128i ab_i32x4 = _mm_hadd_epi32(a_i32x4, b_i32x4); // [a01, a23, b01, b23]
    __m128i cd_i32x4 = _mm_hadd_epi32(c_i32x4, d_i32x4); // [c01, c23, d01, d23]

    // Step 4: Reduce 2x i32 → 1x i32 per state (final horizontal add)
    __m128i intersection_i32x4 = _mm_hadd_epi32(ab_i32x4, cd_i32x4); // [a, b, c, d]

    // Step 5: Direct i32 → f32 conversion (simpler than i64 → f64 → f32 path)
    // - `VCVTDQ2PS` (XMM): 4cy latency, 0.5/cy throughput, port p01
    __m128 intersection_f32x4 = _mm_cvtepi32_ps(intersection_i32x4);

    // Compute Jaccard distance: 1 - intersection ÷ union
    // where union = query_popcount + target_popcount - intersection
    __m128 query_f32x4 = _mm_set1_ps(query_popcount);
    __m128 targets_f32x4 = _mm_setr_ps(target_popcount_a, target_popcount_b, target_popcount_c, target_popcount_d);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case: if union == 0, result = 1.0
    __m128 zero_union_mask = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_mask);

    // Fast reciprocal with Newton-Raphson refinement:
    // - `VRCP14PS`: 4cy latency, 1/cy throughput, port p0 (~14-bit precision)
    // Newton-Raphson: rcp' = rcp × (2 - x × rcp) doubles precision to ~28 bits
    // - `VFNMADD`: 4cy latency, 0.5/cy throughput, ports p01
    // - `VMULPS`: 4cy latency, 0.5/cy throughput, ports p01
    // Total: ~12cy vs `VDIVPS` 11cy latency but 3cy throughput - NR wins on throughput
    __m128 union_reciprocal_f32x4 = _mm_rcp14_ps(safe_union_f32x4);
    union_reciprocal_f32x4 = _mm_mul_ps(union_reciprocal_f32x4,
                                        _mm_fnmadd_ps(safe_union_f32x4, union_reciprocal_f32x4, _mm_set1_ps(2.0f)));

    __m128 ratio_f32x4 = _mm_mul_ps(intersection_f32x4, union_reciprocal_f32x4);
    __m128 jaccard_f32x4 = _mm_sub_ps(one_f32x4, ratio_f32x4);
    result->xmm_ps = _mm_blendv_ps(jaccard_f32x4, one_f32x4, zero_union_mask);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_SET_ICE_H
