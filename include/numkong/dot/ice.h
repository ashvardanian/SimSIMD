/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/dot/ice.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_ICE_H
#define NK_DOT_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_i8_ice(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                             nk_i32_t *result) {
    __m512i a_i16x32, b_i16x32;
    __m512i sum_i32x16 = _mm512_setzero_si512();

nk_dot_i8_ice_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b_scalars));
        count_scalars = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a_scalars));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b_scalars));
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a_scalars.byte[4*j]) * SignExtend16(b_scalars.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_i16x32, b_i16x32);
    if (count_scalars) goto nk_dot_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(sum_i32x16);
}

NK_PUBLIC void nk_dot_u8_ice(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                             nk_u32_t *result) {
    __m512i a_u8x64, b_u8x64;
    __m512i a_low_i16x32, a_high_i16x32, b_low_i16x32, b_high_i16x32;
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

nk_dot_u8_ice_cycle:
    if (count_scalars < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, count_scalars);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a_scalars);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a_scalars);
        b_u8x64 = _mm512_loadu_si512(b_scalars);
        a_scalars += 64, b_scalars += 64, count_scalars -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_low_i16x32, b_low_i16x32);
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_high_i16x32, b_high_i16x32);
    if (count_scalars) goto nk_dot_u8_ice_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(sum_i32x16);
}

typedef struct nk_dot_i8x32_state_ice_t {
    __m512i sum_i32x16;
} nk_dot_i8x32_state_ice_t;

NK_INTERNAL void nk_dot_i8x32_init_ice(nk_dot_i8x32_state_ice_t *state) { state->sum_i32x16 = _mm512_setzero_si512(); }

NK_INTERNAL void nk_dot_i8x32_update_ice(nk_dot_i8x32_state_ice_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m512i a_i16x32 = _mm512_cvtepi8_epi16(a.ymm);
    __m512i b_i16x32 = _mm512_cvtepi8_epi16(b.ymm);
    state->sum_i32x16 = _mm512_dpwssd_epi32(state->sum_i32x16, a_i16x32, b_i16x32);
}

NK_INTERNAL void nk_dot_i8x32_finalize_ice(                                           //
    nk_dot_i8x32_state_ice_t const *state_a, nk_dot_i8x32_state_ice_t const *state_b, //
    nk_dot_i8x32_state_ice_t const *state_c, nk_dot_i8x32_state_ice_t const *state_d, //
    nk_b128_vec_t *results) {
    // ILP-optimized 4-way horizontal reduction for i32
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    results->xmm = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                 _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
}

typedef struct nk_dot_u8x64_state_ice_t {
    __m512i sum_i32x16;
} nk_dot_u8x64_state_ice_t;

NK_INTERNAL void nk_dot_u8x64_init_ice(nk_dot_u8x64_state_ice_t *state) { state->sum_i32x16 = _mm512_setzero_si512(); }

NK_INTERNAL void nk_dot_u8x64_update_ice(nk_dot_u8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    __m512i sum_i32x16 = state->sum_i32x16;
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

    __m512i a_u8x64 = _mm512_loadu_si512(a.u8s);
    __m512i b_u8x64 = _mm512_loadu_si512(b.u8s);
    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    __m512i a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    __m512i a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    __m512i b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    __m512i b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);
    // Unfortunately we can't use the `_mm512_dpbusd_epi32` intrinsics here either,
    // as it's asymmetric with respect to the sign of the input arguments:
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    // So we have to use the `_mm512_dpwssd_epi32` intrinsics instead, upcasting
    // to 16-bit beforehand.
    sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_low_i16x32, b_low_i16x32);
    state->sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, a_high_i16x32, b_high_i16x32);
}

NK_INTERNAL void nk_dot_u8x64_finalize_ice(                                           //
    nk_dot_u8x64_state_ice_t const *state_a, nk_dot_u8x64_state_ice_t const *state_b, //
    nk_dot_u8x64_state_ice_t const *state_c, nk_dot_u8x64_state_ice_t const *state_d, //
    nk_b128_vec_t *results) {
    // ILP-optimized 4-way horizontal reduction for u32
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_a->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_a->sum_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_b->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_b->sum_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_c->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_c->sum_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(state_d->sum_i32x16),
                                           _mm512_extracti32x8_epi32(state_d->sum_i32x16, 1));
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    results->xmm = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                 _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_DOT_ICE_H
