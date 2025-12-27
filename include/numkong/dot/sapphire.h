/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/dot/sapphire.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SAPPHIRE_H
#define NK_DOT_SAPPHIRE_H

#if _NK_TARGET_X86
#if NK_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_sapphire(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

nk_dot_f16_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_scalars);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_scalars);
        b_f16x32 = _mm512_loadu_epi16(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto nk_dot_f16_sapphire_cycle;

    *result = (nk_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_dot_f16c_sapphire(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_real_f16x32 = _mm512_setzero_ph();
    __m512h sum_imag_f16x32 = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_f16x32 = _mm512_set_epi8(               //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

nk_dot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_pairs);
        b_f16x32 = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    sum_real_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_xor_si512(b_f16x32, sign_flip_f16x32)),
                                      _mm512_castsi512_ph(a_f16x32), sum_real_f16x32);
    sum_imag_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(_mm512_shuffle_epi8(b_f16x32, swap_adjacent_f16x32)),
                                      _mm512_castsi512_ph(a_f16x32), sum_imag_f16x32);
    if (count_pairs) goto nk_dot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    result->real = (nk_f32_t)_mm512_reduce_add_ph(sum_real_f16x32);
    result->imag = (nk_f32_t)_mm512_reduce_add_ph(sum_imag_f16x32);
}

NK_PUBLIC void nk_vdot_f16c_sapphire(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                     nk_f32c_t *result) {
    __m512i a_f16x32, b_f16x32;
    __m512h sum_real_f16x32 = _mm512_setzero_ph();
    __m512h sum_imag_f16x32 = _mm512_setzero_ph();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f16x32 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_f16x32 = _mm512_set_epi8(               //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

nk_vdot_f16c_sapphire_cycle:
    if (count_pairs < 16) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a_pairs);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a_pairs);
        b_f16x32 = _mm512_loadu_epi16(b_pairs);
        a_pairs += 16, b_pairs += 16, count_pairs -= 16;
    }
    // TODO: Consider using `_mm512_fmaddsub` and `_mm512_fcmadd_pch`
    sum_real_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_real_f16x32);
    a_f16x32 = _mm512_xor_si512(a_f16x32, sign_flip_f16x32);
    b_f16x32 = _mm512_shuffle_epi8(b_f16x32, swap_adjacent_f16x32);
    sum_imag_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_imag_f16x32);
    if (count_pairs) goto nk_vdot_f16c_sapphire_cycle;

    // Reduce horizontal sums:
    result->real = (nk_f32_t)_mm512_reduce_add_ph(sum_real_f16x32);
    result->imag = (nk_f32_t)_mm512_reduce_add_ph(sum_imag_f16x32);
}

/*  Convert 32x E4M3 values to 32x F16 values.
 *  Uses optimized path similar to E5M2 but with bias adjustment.
 *  Denormals (exp=0) are flushed to zero (DAZ behavior).
 *
 *  E4M3 format: S EEEE  MMM        (bias=7)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 *
 *  The key difference from E5M2→F16 (which is trivial) is the bias adjustment:
 *  E5M2 and F16 share bias=15, so just shift. E4M3 needs +8 to exponent.
 */
NK_INTERNAL __m512i _nk_e4m3_to_f16_sapphire(__m256i e4m3_i8x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3_i8x32);
    // Sign: bit 7 → bit 15
    __m512i sign_i16x32 = _mm512_and_si512(_mm512_slli_epi16(e4m3_i16x32, 8), _mm512_set1_epi16((short)0x8000));
    // Exp+mant (7 bits) shifted left 7, then add bias adjustment (8<<10 = 0x2000)
    __m512i exp_mant_7bit_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));
    __m512i exp_mant_biased_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(exp_mant_7bit_i16x32, 7),
                                                      _mm512_set1_epi16(0x2000));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero - single instruction!
    __mmask32 nonzero_exp_mask = _mm512_test_epi16_mask(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __m512i exp_mant_daz_i16x32 = _mm512_maskz_mov_epi16(nonzero_exp_mask, exp_mant_biased_i16x32);
    return _mm512_or_si512(sign_i16x32, exp_mant_daz_i16x32);
}

/*  Convert 32x E5M2 values to 32x F16 values.
 *  This is extremely fast because E5M2 and F16 have the same exponent bias (15).
 *  Simply zero-extend to 16-bit and shift left by 8.
 *
 *  E5M2 format: S EEEEE MM         (bias=15)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 */
NK_INTERNAL __m512i _nk_e5m2_to_f16_sapphire(__m256i e5m2_i8x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2_i8x32);
    return _mm512_slli_epi16(e5m2_i16x32, 8);
}

NK_PUBLIC void nk_dot_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

nk_dot_e4m3_sapphire_cycle:
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
    // Convert E4M3 to F16 and compute dot product
    __m512i a_f16x32 = _nk_e4m3_to_f16_sapphire(a_e4m3x32);
    __m512i b_f16x32 = _nk_e4m3_to_f16_sapphire(b_e4m3x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto nk_dot_e4m3_sapphire_cycle;

    *result = (nk_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_dot_e5m2_sapphire(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    __m256i a_e5m2x32, b_e5m2x32;
    __m512h sum_f16x32 = _mm512_setzero_ph();

nk_dot_e5m2_sapphire_cycle:
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
    // Convert E5M2 to F16 and compute dot product
    // Note: E5M2 to F16 is extremely fast due to same exponent bias
    __m512i a_f16x32 = _nk_e5m2_to_f16_sapphire(a_e5m2x32);
    __m512i b_f16x32 = _nk_e5m2_to_f16_sapphire(b_e5m2x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    if (count_scalars) goto nk_dot_e5m2_sapphire_cycle;

    *result = (nk_f32_t)_mm512_reduce_add_ph(sum_f16x32);
}

typedef struct nk_dot_f16x32_state_sapphire_t {
    __m512h sum_f16x32;
} nk_dot_f16x32_state_sapphire_t;

NK_INTERNAL void nk_dot_f16x32_init_sapphire(nk_dot_f16x32_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

NK_INTERNAL void nk_dot_f16x32_update_sapphire(nk_dot_f16x32_state_sapphire_t *state, nk_b512_vec_t a,
                                               nk_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m512i a_f16x32 = _mm512_loadu_epi16(a.f16s);
    __m512i b_f16x32 = _mm512_loadu_epi16(b.f16s);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

NK_INTERNAL void nk_dot_f16x32_finalize_sapphire(                                                 //
    nk_dot_f16x32_state_sapphire_t const *state_a, nk_dot_f16x32_state_sapphire_t const *state_b, //
    nk_dot_f16x32_state_sapphire_t const *state_c, nk_dot_f16x32_state_sapphire_t const *state_d, //
    nk_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction for f16 (32 elements → 1 scalar each)
    // Step 1: 32→16 for all 4 states (extract high 256-bit half and add to low half)
    // Use integer extract and cast since there's no direct _mm512_extractf16x16_ph
    __m256h sum_f16x16_a = _mm256_add_ph(
        _mm512_castph512_ph256(state_a->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_a->sum_f16x32), 1)));
    __m256h sum_f16x16_b = _mm256_add_ph(
        _mm512_castph512_ph256(state_b->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_b->sum_f16x32), 1)));
    __m256h sum_f16x16_c = _mm256_add_ph(
        _mm512_castph512_ph256(state_c->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_c->sum_f16x32), 1)));
    __m256h sum_f16x16_d = _mm256_add_ph(
        _mm512_castph512_ph256(state_d->sum_f16x32),
        _mm256_castsi256_ph(_mm512_extracti32x8_epi32(_mm512_castph_si512(state_d->sum_f16x32), 1)));
    // Step 2: 16→8 for all 4 states (extract high 128-bit half and add to low half)
    __m128h sum_f16x8_a = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_a),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_a), 1)));
    __m128h sum_f16x8_b = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_b),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_b), 1)));
    __m128h sum_f16x8_c = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_c),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_c), 1)));
    __m128h sum_f16x8_d = _mm_add_ph(_mm256_castph256_ph128(sum_f16x16_d),
                                     _mm_castsi128_ph(_mm256_extracti128_si256(_mm256_castph_si256(sum_f16x16_d), 1)));
    // Step 3: 8→4 for all 4 states (shift right by 8 bytes = 4 f16 elements, then add)
    __m128h sum_f16x4_a = _mm_add_ph(sum_f16x8_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_a), 8)));
    __m128h sum_f16x4_b = _mm_add_ph(sum_f16x8_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_b), 8)));
    __m128h sum_f16x4_c = _mm_add_ph(sum_f16x8_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_c), 8)));
    __m128h sum_f16x4_d = _mm_add_ph(sum_f16x8_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x8_d), 8)));
    // Step 4: 4→2 for all 4 states (shift right by 4 bytes = 2 f16 elements, then add)
    __m128h sum_f16x2_a = _mm_add_ph(sum_f16x4_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_a), 4)));
    __m128h sum_f16x2_b = _mm_add_ph(sum_f16x4_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_b), 4)));
    __m128h sum_f16x2_c = _mm_add_ph(sum_f16x4_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_c), 4)));
    __m128h sum_f16x2_d = _mm_add_ph(sum_f16x4_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x4_d), 4)));
    // Step 5: 2→1 for all 4 states (shift right by 2 bytes = 1 f16 element, then add)
    __m128h sum_f16x1_a = _mm_add_ph(sum_f16x2_a, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_a), 2)));
    __m128h sum_f16x1_b = _mm_add_ph(sum_f16x2_b, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_b), 2)));
    __m128h sum_f16x1_c = _mm_add_ph(sum_f16x2_c, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_c), 2)));
    __m128h sum_f16x1_d = _mm_add_ph(sum_f16x2_d, _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(sum_f16x2_d), 2)));
    // Extract first f16 element and convert to f32
    results[0] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_a));
    results[1] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_b));
    results[2] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_c));
    results[3] = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), sum_f16x1_d));
}

typedef struct nk_dot_e4m3x64_state_sapphire_t {
    __m512h sum_f16x32;
} nk_dot_e4m3x64_state_sapphire_t;

NK_INTERNAL void nk_dot_e4m3x64_init_sapphire(nk_dot_e4m3x64_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

NK_INTERNAL void nk_dot_e4m3x64_update_sapphire(nk_dot_e4m3x64_state_sapphire_t *state, nk_b512_vec_t a,
                                                nk_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m256i a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 0);
    __m256i b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 0);
    __m512i a_f16x32 = _nk_e4m3_to_f16_sapphire(a_e4m3x32);
    __m512i b_f16x32 = _nk_e4m3_to_f16_sapphire(b_e4m3x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    a_e4m3x32 = _mm256_loadu_epi8(a.e4m3s + 32);
    b_e4m3x32 = _mm256_loadu_epi8(b.e4m3s + 32);
    a_f16x32 = _nk_e4m3_to_f16_sapphire(a_e4m3x32);
    b_f16x32 = _nk_e4m3_to_f16_sapphire(b_e4m3x32);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

NK_INTERNAL void nk_dot_e4m3x64_finalize_sapphire(                                                  //
    nk_dot_e4m3x64_state_sapphire_t const *state_a, nk_dot_e4m3x64_state_sapphire_t const *state_b, //
    nk_dot_e4m3x64_state_sapphire_t const *state_c, nk_dot_e4m3x64_state_sapphire_t const *state_d, //
    nk_f32_t *results) {
    // State is layout-compatible with f16x32 (both contain just __m512h sum)
    nk_dot_f16x32_finalize_sapphire(                                                                      //
        (nk_dot_f16x32_state_sapphire_t const *)state_a, (nk_dot_f16x32_state_sapphire_t const *)state_b, //
        (nk_dot_f16x32_state_sapphire_t const *)state_c, (nk_dot_f16x32_state_sapphire_t const *)state_d, results);
}

typedef struct nk_dot_e5m2x64_state_sapphire_t {
    __m512h sum_f16x32;
} nk_dot_e5m2x64_state_sapphire_t;

NK_INTERNAL void nk_dot_e5m2x64_init_sapphire(nk_dot_e5m2x64_state_sapphire_t *state) {
    state->sum_f16x32 = _mm512_setzero_ph();
}

NK_INTERNAL void nk_dot_e5m2x64_update_sapphire(nk_dot_e5m2x64_state_sapphire_t *state, nk_b512_vec_t a,
                                                nk_b512_vec_t b) {
    __m512h sum_f16x32 = state->sum_f16x32;
    __m256i a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 0);
    __m256i b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 0);
    __m512i a_f16x32 = _nk_e5m2_to_f16_sapphire(a_e5m2x32);
    __m512i b_f16x32 = _nk_e5m2_to_f16_sapphire(b_e5m2x32);
    sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
    a_e5m2x32 = _mm256_loadu_epi8(a.e5m2s + 32);
    b_e5m2x32 = _mm256_loadu_epi8(b.e5m2s + 32);
    a_f16x32 = _nk_e5m2_to_f16_sapphire(a_e5m2x32);
    b_f16x32 = _nk_e5m2_to_f16_sapphire(b_e5m2x32);
    state->sum_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32), sum_f16x32);
}

NK_INTERNAL void nk_dot_e5m2x64_finalize_sapphire(                                                  //
    nk_dot_e5m2x64_state_sapphire_t const *state_a, nk_dot_e5m2x64_state_sapphire_t const *state_b, //
    nk_dot_e5m2x64_state_sapphire_t const *state_c, nk_dot_e5m2x64_state_sapphire_t const *state_d, //
    nk_f32_t *results) {
    // State is layout-compatible with f16x32 (both contain just __m512h sum)
    nk_dot_f16x32_finalize_sapphire(                                                                      //
        (nk_dot_f16x32_state_sapphire_t const *)state_a, (nk_dot_f16x32_state_sapphire_t const *)state_b, //
        (nk_dot_f16x32_state_sapphire_t const *)state_c, (nk_dot_f16x32_state_sapphire_t const *)state_d, results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // _NK_TARGET_X86

#endif // NK_DOT_SAPPHIRE_H