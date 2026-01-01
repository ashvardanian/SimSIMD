/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for AMD Genoa CPUs.
 *  @file include/numkong/dot/genoa.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_GENOA_H
#define NK_DOT_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // nk_reduce_add_f32x16_skylake_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_bf16_genoa(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_bf16_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a_scalars);
        b_bf16x32 = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto nk_dot_bf16_genoa_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_bf16c_genoa(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                  nk_f32c_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_real_f32x16 = _mm512_setzero_ps();
    __m512 sum_imag_f32x16 = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_bf16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_bf16x32 = _mm512_set_epi8(              //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

nk_dot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, (nk_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, (nk_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    sum_real_f32x16 = _mm512_dpbf16_ps(sum_real_f32x16, (__m512bh)(_mm512_xor_si512(b_bf16x32, sign_flip_bf16x32)),
                                       (__m512bh)(a_bf16x32));
    sum_imag_f32x16 = _mm512_dpbf16_ps(
        sum_imag_f32x16, (__m512bh)(_mm512_shuffle_epi8(b_bf16x32, swap_adjacent_bf16x32)), (__m512bh)(a_bf16x32));
    if (count_pairs) goto nk_dot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    result->real = nk_reduce_add_f32x16_skylake_(sum_real_f32x16);
    result->imag = nk_reduce_add_f32x16_skylake_(sum_imag_f32x16);
}

NK_PUBLIC void nk_vdot_bf16c_genoa(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    __m512i a_bf16x32, b_bf16x32;
    __m512 sum_real_f32x16 = _mm512_setzero_ps();
    __m512 sum_imag_f32x16 = _mm512_setzero_ps();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_bf16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_bf16x32 = _mm512_set_epi8(              //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

nk_vdot_bf16c_genoa_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, (nk_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, (nk_i16_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)a_pairs);
        b_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    sum_real_f32x16 = _mm512_dpbf16_ps(sum_real_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_bf16x32 = _mm512_xor_si512(a_bf16x32, sign_flip_bf16x32);
    b_bf16x32 = _mm512_shuffle_epi8(b_bf16x32, swap_adjacent_bf16x32);
    sum_imag_f32x16 = _mm512_dpbf16_ps(sum_imag_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_pairs) goto nk_vdot_bf16c_genoa_cycle;

    // Reduce horizontal sums:
    result->real = nk_reduce_add_f32x16_skylake_(sum_real_f32x16);
    result->imag = nk_reduce_add_f32x16_skylake_(sum_imag_f32x16);
}

/**
 *  @brief Convert 32x E4M3 values to 32x BF16 values.
 *
 *  Uses optimized path with fused exp+mant extraction.
 *  Denormals (exp=0, mant!=0) are flushed to zero (DAZ behavior).
 *
 *  E4M3 format: S EEEE MMM (bias=7, range: 2^-6 to 448)
 *  BF16 format: S EEEEEEEE MMMMMMM (bias=127)
 *  Conversion: sign<<8, (exp+120)<<7, mant<<4
 */
NK_INTERNAL __m512i nk_e4m3x32_to_bf16x32_genoa_(__m256i e4m3x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3x32);
    // Sign: shift bit 7 to bit 15
    __m512i sign_i16x32 = _mm512_and_si512(_mm512_slli_epi16(e4m3_i16x32, 8), _mm512_set1_epi16((short)0x8000));
    // Lower 7 bits contain exp (4) and mant (3): shift left 4 and add bias
    __m512i low7_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));
    __m512i exp_mant_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(low7_i16x32, 4), _mm512_set1_epi16(0x3C00));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero
    __mmask32 has_exp_mask = _mm512_test_epi16_mask(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __m512i masked_exp_mant_i16x32 = _mm512_maskz_mov_epi16(has_exp_mask, exp_mant_i16x32);
    return _mm512_or_si512(sign_i16x32, masked_exp_mant_i16x32);
}

/**
 *  @brief Convert 32x E5M2 values to 32x BF16 values.
 *
 *  Uses optimized path with fused exp+mant extraction.
 *  Denormals (exp=0, mant!=0) are flushed to zero (DAZ behavior).
 *
 *  E5M2 format: S EEEEE MM (bias=15, range: 2^-14 to 57344)
 *  BF16 format: S EEEEEEEE MMMMMMM (bias=127)
 *  Conversion: sign<<8, (exp+112)<<7, mant<<5
 */
NK_INTERNAL __m512i nk_e5m2x32_to_bf16x32_genoa_(__m256i e5m2x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(_mm512_slli_epi16(e5m2_i16x32, 8), _mm512_set1_epi16((short)0x8000));
    // Lower 7 bits: exp(5) + mant(2), shift left 5 and add bias
    __m512i low7_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7F));
    __m512i exp_mant_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(low7_i16x32, 5), _mm512_set1_epi16(0x3800));
    // DAZ: use TEST to check if exp bits (bits 6-2) are nonzero
    __mmask32 has_exp_mask = _mm512_test_epi16_mask(e5m2_i16x32, _mm512_set1_epi16(0x7C));
    __m512i masked_exp_mant_i16x32 = _mm512_maskz_mov_epi16(has_exp_mask, exp_mant_i16x32);
    return _mm512_or_si512(sign_i16x32, masked_exp_mant_i16x32);
}

NK_PUBLIC void nk_dot_e4m3_genoa(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e4m3_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a_scalars);
        b_e4m3x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E4M3 to BF16 and compute dot product
    __m512i a_bf16x32 = nk_e4m3x32_to_bf16x32_genoa_(a_e4m3x32);
    __m512i b_bf16x32 = nk_e4m3x32_to_bf16x32_genoa_(b_e4m3x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto nk_dot_e4m3_genoa_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e5m2_genoa(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    __m256i a_e5m2x32, b_e5m2x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e5m2_genoa_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a_scalars);
        b_e5m2x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Convert E5M2 to BF16 and compute dot product
    __m512i a_bf16x32 = nk_e5m2x32_to_bf16x32_genoa_(a_e5m2x32);
    __m512i b_bf16x32 = nk_e5m2x32_to_bf16x32_genoa_(b_e5m2x32);
    sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    if (count_scalars) goto nk_dot_e5m2_genoa_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

typedef struct nk_dot_bf16x32_state_genoa_t {
    __m512 sum_f32x16;
} nk_dot_bf16x32_state_genoa_t;

NK_INTERNAL void nk_dot_bf16x32_init_genoa(nk_dot_bf16x32_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_bf16x32_update_genoa(nk_dot_bf16x32_state_genoa_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    state->sum_f32x16 = _mm512_dpbf16_ps(state->sum_f32x16, (__m512bh)(a.zmm), (__m512bh)(b.zmm));
}

NK_INTERNAL void nk_dot_bf16x32_finalize_genoa(                                               //
    nk_dot_bf16x32_state_genoa_t const *state_a, nk_dot_bf16x32_state_genoa_t const *state_b, //
    nk_dot_bf16x32_state_genoa_t const *state_c, nk_dot_bf16x32_state_genoa_t const *state_d, //
    nk_b128_vec_t *result) {
    nk_dot_f32x16_finalize_skylake_wout_compensation_(state_a->sum_f32x16, state_b->sum_f32x16, state_c->sum_f32x16,
                                                      state_d->sum_f32x16, result);
}

typedef struct nk_dot_e4m3x32_state_genoa_t {
    __m512 sum_f32x16;
} nk_dot_e4m3x32_state_genoa_t;

NK_INTERNAL void nk_dot_e4m3x32_init_genoa(nk_dot_e4m3x32_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e4m3x32_update_genoa(nk_dot_e4m3x32_state_genoa_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m512i a_bf16x32 = nk_e4m3x32_to_bf16x32_genoa_(a.ymm);
    __m512i b_bf16x32 = nk_e4m3x32_to_bf16x32_genoa_(b.ymm);
    state->sum_f32x16 = _mm512_dpbf16_ps(state->sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
}

NK_INTERNAL void nk_dot_e4m3x32_finalize_genoa(                                               //
    nk_dot_e4m3x32_state_genoa_t const *state_a, nk_dot_e4m3x32_state_genoa_t const *state_b, //
    nk_dot_e4m3x32_state_genoa_t const *state_c, nk_dot_e4m3x32_state_genoa_t const *state_d, //
    nk_b128_vec_t *result) {
    nk_dot_f32x16_finalize_skylake_wout_compensation_(state_a->sum_f32x16, state_b->sum_f32x16, state_c->sum_f32x16,
                                                      state_d->sum_f32x16, result);
}

typedef struct nk_dot_e5m2x32_state_genoa_t {
    __m512 sum_f32x16;
} nk_dot_e5m2x32_state_genoa_t;

NK_INTERNAL void nk_dot_e5m2x32_init_genoa(nk_dot_e5m2x32_state_genoa_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e5m2x32_update_genoa(nk_dot_e5m2x32_state_genoa_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m512i a_bf16x32 = nk_e5m2x32_to_bf16x32_genoa_(a.ymm);
    __m512i b_bf16x32 = nk_e5m2x32_to_bf16x32_genoa_(b.ymm);
    state->sum_f32x16 = _mm512_dpbf16_ps(state->sum_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
}

NK_INTERNAL void nk_dot_e5m2x32_finalize_genoa(                                               //
    nk_dot_e5m2x32_state_genoa_t const *state_a, nk_dot_e5m2x32_state_genoa_t const *state_b, //
    nk_dot_e5m2x32_state_genoa_t const *state_c, nk_dot_e5m2x32_state_genoa_t const *state_d, //
    nk_b128_vec_t *result) {
    nk_dot_f32x16_finalize_skylake_wout_compensation_(state_a->sum_f32x16, state_b->sum_f32x16, state_c->sum_f32x16,
                                                      state_d->sum_f32x16, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_

#endif // NK_DOT_GENOA_H