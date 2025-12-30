/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/dot/skylake.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SKYLAKE_H
#define NK_DOT_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/reduce/skylake.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f32_skylake(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_f32x16 = _mm512_setzero();

nk_dot_f32_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a_scalars);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a_scalars);
        b_f32x16 = _mm512_loadu_ps(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_f32_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_f64_skylake(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_f64x8 = _mm512_setzero_pd();

nk_dot_f64_skylake_cycle:
    if (count_scalars < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_scalars);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_scalars);
        b_f64x8 = _mm512_loadu_pd(b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_f64x8);
    if (count_scalars) goto nk_dot_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_dot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_real_f32x16 = _mm512_setzero();
    __m512 sum_imag_f32x16 = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f32x16 = _mm512_set1_epi64(0x8000000000000000);
nk_dot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a_pairs);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a_pairs);
        b_f32x16 = _mm512_loadu_ps(b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    sum_real_f32x16 = _mm512_fmadd_ps(b_f32x16, a_f32x16, sum_real_f32x16);
    b_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1); //? Swap adjacent entries within each pair
    sum_imag_f32x16 = _mm512_fmadd_ps(b_f32x16, a_f32x16, sum_imag_f32x16);
    if (count_pairs) goto nk_dot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x16 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(sum_real_f32x16), sign_flip_f32x16));

    // Reduce horizontal sums:
    result->real = nk_reduce_add_f32x16_skylake_(sum_real_f32x16);
    result->imag = nk_reduce_add_f32x16_skylake_(sum_imag_f32x16);
}

NK_PUBLIC void nk_vdot_f32c_skylake(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m512 a_f32x16, b_f32x16;
    __m512 sum_real_f32x16 = _mm512_setzero();
    __m512 sum_imag_f32x16 = _mm512_setzero();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f32x16 = _mm512_set1_epi64(0x8000000000000000);
nk_vdot_f32c_skylake_cycle:
    if (count_pairs < 8) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, (nk_f32_t const *)a_pairs);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, (nk_f32_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps((nk_f32_t const *)a_pairs);
        b_f32x16 = _mm512_loadu_ps((nk_f32_t const *)b_pairs);
        a_pairs += 8, b_pairs += 8, count_pairs -= 8;
    }
    sum_real_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_real_f32x16);
    b_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1); //? Swap adjacent entries within each pair
    sum_imag_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_imag_f32x16);
    if (count_pairs) goto nk_vdot_f32c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x16 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(sum_imag_f32x16), sign_flip_f32x16));

    // Reduce horizontal sums:
    result->real = nk_reduce_add_f32x16_skylake_(sum_real_f32x16);
    result->imag = nk_reduce_add_f32x16_skylake_(sum_imag_f32x16);
}

NK_PUBLIC void nk_dot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_dot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a_pairs);
        b_f64x8 = _mm512_loadu_pd(b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    sum_real_f64x8 = _mm512_fmadd_pd(b_f64x8, a_f64x8, sum_real_f64x8);
    b_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.
    sum_imag_f64x8 = _mm512_fmadd_pd(b_f64x8, a_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_dot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_real_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_vdot_f64c_skylake(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f64c_t *result) {
    __m512d a_f64x8, b_f64x8;
    __m512d sum_real_f64x8 = _mm512_setzero_pd();
    __m512d sum_imag_f64x8 = _mm512_setzero_pd();

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_f64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );
nk_vdot_f64c_skylake_cycle:
    if (count_pairs < 4) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, count_pairs * 2);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, (nk_f64_t const *)b_pairs);
        count_pairs = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd((nk_f64_t const *)a_pairs);
        b_f64x8 = _mm512_loadu_pd((nk_f64_t const *)b_pairs);
        a_pairs += 4, b_pairs += 4, count_pairs -= 4;
    }
    sum_real_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_real_f64x8);
    b_f64x8 = _mm512_permute_pd(b_f64x8, 0x55); //? Same as 0b01010101.
    sum_imag_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, sum_imag_f64x8);
    if (count_pairs) goto nk_vdot_f64c_skylake_cycle;

    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(sum_imag_f64x8), sign_flip_f64x8));

    // Reduce horizontal sums:
    result->real = _mm512_reduce_add_pd(sum_real_f64x8);
    result->imag = _mm512_reduce_add_pd(sum_imag_f64x8);
}

NK_PUBLIC void nk_dot_f16_skylake(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256i a_f16x16, b_f16x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_f16_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a_scalars);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_f16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_bf16_skylake(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256i a_bf16x16, b_bf16x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_bf16_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a_scalars);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_bf16x16 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    __m512 b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_bf16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e4m3_skylake(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_e4m3x16, b_e4m3x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e4m3_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_e5m2_skylake(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_e5m2x16, b_e5m2x16;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_dot_e5m2_skylake_cycle:
    if (count_scalars < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, count_scalars);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a_scalars);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    sum_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, sum_f32x16);
    if (count_scalars) goto nk_dot_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_dot_i8_skylake(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes at a time and widen to i16
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_i16x32 = _mm512_cvtepi8_epi16(a_i8x32);
        __m512i b_i16x32 = _mm512_cvtepi8_epi16(b_i8x32);
        // _mm512_madd_epi16: multiply adjacent pairs of i16, add pairs to produce i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_i16x32, b_i16x32));
    }
    nk_i32_t sum = _mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_skylake(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        // Load 32 bytes and zero-extend to i16 (u8 -> u16 via zero-extension)
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b_scalars + idx_scalars));
        __m512i a_u16x32 = _mm512_cvtepu8_epi16(a_u8x32);
        __m512i b_u16x32 = _mm512_cvtepu8_epi16(b_u8x32);
        // _mm512_madd_epi16: multiply adjacent pairs, add pairs to produce i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(a_u16x32, b_u16x32));
    }
    nk_u32_t sum = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

typedef struct nk_dot_f64x8_state_skylake_t {
    __m512d sum_f64x8;
} nk_dot_f64x8_state_skylake_t;

NK_INTERNAL void nk_dot_f64x8_init_skylake(nk_dot_f64x8_state_skylake_t *state) {
    state->sum_f64x8 = _mm512_setzero_pd();
}

NK_INTERNAL void nk_dot_f64x8_update_skylake(nk_dot_f64x8_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    state->sum_f64x8 = _mm512_fmadd_pd(a.zmm_pd, b.zmm_pd, state->sum_f64x8);
}

NK_INTERNAL void nk_dot_f64x8_finalize_skylake(                                               //
    nk_dot_f64x8_state_skylake_t const *state_a, nk_dot_f64x8_state_skylake_t const *state_b, //
    nk_dot_f64x8_state_skylake_t const *state_c, nk_dot_f64x8_state_skylake_t const *state_d, //
    nk_f64_t *results) {
    // ILP-optimized 4-way horizontal reduction for f64
    // Step 1: 8->4 for all 4 states (extract high 256-bit half and add to low half)
    __m256d reduced_a = _mm256_add_pd(_mm512_castpd512_pd256(state_a->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_a->sum_f64x8, 1));
    __m256d reduced_b = _mm256_add_pd(_mm512_castpd512_pd256(state_b->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_b->sum_f64x8, 1));
    __m256d reduced_c = _mm256_add_pd(_mm512_castpd512_pd256(state_c->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_c->sum_f64x8, 1));
    __m256d reduced_d = _mm256_add_pd(_mm512_castpd512_pd256(state_d->sum_f64x8),
                                      _mm512_extractf64x4_pd(state_d->sum_f64x8, 1));
    // Step 2: 4->2 for all 4 states (extract high 128-bit half and add to low half)
    __m128d partial_a = _mm_add_pd(_mm256_castpd256_pd128(reduced_a), _mm256_extractf128_pd(reduced_a, 1));
    __m128d partial_b = _mm_add_pd(_mm256_castpd256_pd128(reduced_b), _mm256_extractf128_pd(reduced_b, 1));
    __m128d partial_c = _mm_add_pd(_mm256_castpd256_pd128(reduced_c), _mm256_extractf128_pd(reduced_c, 1));
    __m128d partial_d = _mm_add_pd(_mm256_castpd256_pd128(reduced_d), _mm256_extractf128_pd(reduced_d, 1));
    // Step 3: 2->1 for each state and combine into 4-element result
    // Each __m128d has [low, high], need to add them to get final scalar
    __m128d sum_ab = _mm_add_pd(_mm_unpacklo_pd(partial_a, partial_b), _mm_unpackhi_pd(partial_a, partial_b));
    __m128d sum_cd = _mm_add_pd(_mm_unpacklo_pd(partial_c, partial_d), _mm_unpackhi_pd(partial_c, partial_d));
    // Store as f64
    _mm_storeu_pd(results, sum_ab);
    _mm_storeu_pd(results + 2, sum_cd);
}

typedef struct nk_dot_f32x16_state_skylake_t {
    __m512 sum_f32x16;
} nk_dot_f32x16_state_skylake_t;

NK_INTERNAL void nk_dot_f32x16_init_skylake(nk_dot_f32x16_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_f32x16_update_skylake(nk_dot_f32x16_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    state->sum_f32x16 = _mm512_fmadd_ps(a.zmm_ps, b.zmm_ps, state->sum_f32x16);
}

NK_INTERNAL void nk_dot_f32x16_finalize_skylake(                                                //
    nk_dot_f32x16_state_skylake_t const *state_a, nk_dot_f32x16_state_skylake_t const *state_b, //
    nk_dot_f32x16_state_skylake_t const *state_c, nk_dot_f32x16_state_skylake_t const *state_d, //
    nk_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction
    // Step 1: 16->8 for all 4 states (extract high 256-bit half and add to low half)
    __m256 reduced_a = _mm256_add_ps(_mm512_castps512_ps256(state_a->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_a->sum_f32x16, 1));
    __m256 reduced_b = _mm256_add_ps(_mm512_castps512_ps256(state_b->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_b->sum_f32x16, 1));
    __m256 reduced_c = _mm256_add_ps(_mm512_castps512_ps256(state_c->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_c->sum_f32x16, 1));
    __m256 reduced_d = _mm256_add_ps(_mm512_castps512_ps256(state_d->sum_f32x16),
                                     _mm512_extractf32x8_ps(state_d->sum_f32x16, 1));
    // Step 2: 8->4 for all 4 states (extract high 128-bit half and add to low half)
    __m128 partial_a = _mm_add_ps(_mm256_castps256_ps128(reduced_a), _mm256_extractf128_ps(reduced_a, 1));
    __m128 partial_b = _mm_add_ps(_mm256_castps256_ps128(reduced_b), _mm256_extractf128_ps(reduced_b, 1));
    __m128 partial_c = _mm_add_ps(_mm256_castps256_ps128(reduced_c), _mm256_extractf128_ps(reduced_c, 1));
    __m128 partial_d = _mm_add_ps(_mm256_castps256_ps128(reduced_d), _mm256_extractf128_ps(reduced_d, 1));
    // Step 3: Transpose 4x4 matrix of partial sums - now each row has one element from each state
    __m128 transpose_ab_lo = _mm_unpacklo_ps(partial_a, partial_b);
    __m128 transpose_cd_lo = _mm_unpacklo_ps(partial_c, partial_d);
    __m128 transpose_ab_hi = _mm_unpackhi_ps(partial_a, partial_b);
    __m128 transpose_cd_hi = _mm_unpackhi_ps(partial_c, partial_d);
    __m128 sum_lane_0 = _mm_movelh_ps(transpose_ab_lo, transpose_cd_lo);
    __m128 sum_lane_1 = _mm_movehl_ps(transpose_cd_lo, transpose_ab_lo);
    __m128 sum_lane_2 = _mm_movelh_ps(transpose_ab_hi, transpose_cd_hi);
    __m128 sum_lane_3 = _mm_movehl_ps(transpose_cd_hi, transpose_ab_hi);
    // Step 4: Vertical sum - each lane becomes the final result for one state
    __m128 final_sum = _mm_add_ps(_mm_add_ps(sum_lane_0, sum_lane_1), _mm_add_ps(sum_lane_2, sum_lane_3));
    _mm_storeu_ps(results, final_sum);
}

typedef struct nk_dot_e4m3x64_state_skylake_t {
    __m512 sum_f32x16;
} nk_dot_e4m3x64_state_skylake_t;

NK_INTERNAL void nk_dot_e4m3x64_init_skylake(nk_dot_e4m3x64_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e4m3x64_update_skylake(nk_dot_e4m3x64_state_skylake_t *state, nk_b512_vec_t a,
                                               nk_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m128i a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 0));
    __m128i b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 0));
    sum_f32x16 = _mm512_fmadd_ps(nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16), nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16),
                                 sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 16));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 16));
    sum_f32x16 = _mm512_fmadd_ps(nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16), nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16),
                                 sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 32));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 32));
    sum_f32x16 = _mm512_fmadd_ps(nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16), nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16),
                                 sum_f32x16);
    a_e4m3x16 = _mm_loadu_si128((__m128i const *)(a.e4m3s + 48));
    b_e4m3x16 = _mm_loadu_si128((__m128i const *)(b.e4m3s + 48));
    sum_f32x16 = _mm512_fmadd_ps(nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16), nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16),
                                 sum_f32x16);
    state->sum_f32x16 = sum_f32x16;
}

NK_INTERNAL void nk_dot_e4m3x64_finalize_skylake(                                                 //
    nk_dot_e4m3x64_state_skylake_t const *state_a, nk_dot_e4m3x64_state_skylake_t const *state_b, //
    nk_dot_e4m3x64_state_skylake_t const *state_c, nk_dot_e4m3x64_state_skylake_t const *state_d, //
    nk_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    nk_dot_f32x16_finalize_skylake(                                                                     //
        (nk_dot_f32x16_state_skylake_t const *)state_a, (nk_dot_f32x16_state_skylake_t const *)state_b, //
        (nk_dot_f32x16_state_skylake_t const *)state_c, (nk_dot_f32x16_state_skylake_t const *)state_d, results);
}

typedef struct nk_dot_e5m2x64_state_skylake_t {
    __m512 sum_f32x16;
} nk_dot_e5m2x64_state_skylake_t;

NK_INTERNAL void nk_dot_e5m2x64_init_skylake(nk_dot_e5m2x64_state_skylake_t *state) {
    state->sum_f32x16 = _mm512_setzero();
}

NK_INTERNAL void nk_dot_e5m2x64_update_skylake(nk_dot_e5m2x64_state_skylake_t *state, nk_b512_vec_t a,
                                               nk_b512_vec_t b) {
    __m512 sum_f32x16 = state->sum_f32x16;
    __m128i a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 0));
    __m128i b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 0));
    sum_f32x16 = _mm512_fmadd_ps(nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16), nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16),
                                 sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 16));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 16));
    sum_f32x16 = _mm512_fmadd_ps(nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16), nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16),
                                 sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 32));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 32));
    sum_f32x16 = _mm512_fmadd_ps(nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16), nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16),
                                 sum_f32x16);
    a_e5m2x16 = _mm_loadu_si128((__m128i const *)(a.e5m2s + 48));
    b_e5m2x16 = _mm_loadu_si128((__m128i const *)(b.e5m2s + 48));
    sum_f32x16 = _mm512_fmadd_ps(nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16), nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16),
                                 sum_f32x16);
    state->sum_f32x16 = sum_f32x16;
}

NK_INTERNAL void nk_dot_e5m2x64_finalize_skylake(                                                 //
    nk_dot_e5m2x64_state_skylake_t const *state_a, nk_dot_e5m2x64_state_skylake_t const *state_b, //
    nk_dot_e5m2x64_state_skylake_t const *state_c, nk_dot_e5m2x64_state_skylake_t const *state_d, //
    nk_f32_t *results) {
    // State is layout-compatible with f32x16 (both contain just __m512 sum_f32x16)
    nk_dot_f32x16_finalize_skylake(                                                                     //
        (nk_dot_f32x16_state_skylake_t const *)state_a, (nk_dot_f32x16_state_skylake_t const *)state_b, //
        (nk_dot_f32x16_state_skylake_t const *)state_c, (nk_dot_f32x16_state_skylake_t const *)state_d, results);
}

/** @brief Type-agnostic 512-bit full load (Skylake AVX-512). */
NK_INTERNAL void nk_load_b512_skylake_(void const *src, nk_b512_vec_t *dst) { dst->zmm = _mm512_loadu_si512(src); }

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_DOT_SKYLAKE_H